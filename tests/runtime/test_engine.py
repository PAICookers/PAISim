import numpy as np
import pytest
from paicorelib.framelib.frame_defs import FrameHeader

from paisim import (
    Array2x2Board,
    ChipCoord,
    CoreAddr,
    CoreCoord,
    Port,
    PortDirection,
    Recorder,
    Simulator,
    SingleBoard,
    ThreadId,
    TraceFilter,
    TraceLevel,
)
from paisim.engine.core import OfflineCore, OfflineLayout, RelayCore
from paisim.engine.errors import UndefinedHardwareBehavior
from paisim.runtime.errors import (
    InvalidFrame,
    InvalidInput,
    ResourceLimit,
    SimulationFailure,
    UnsupportedFrame,
)
from tests.protocol.fixtures import control, offline_config, route, work

C0 = ChipCoord(0, 0)
CPU = Port(C0)
ROOT = CoreAddr(C0, CoreCoord(0, 2))


def execute(sim: Simulator, words: list[int], ingress: Port = CPU) -> np.ndarray:
    sim.feed(np.asarray(words, dtype=np.uint64), ingress=ingress)
    sim.run()
    return sim.drain_output()


def _feed(sim: Simulator, words: list[int], ingress: Port = CPU) -> None:
    sim.feed(np.asarray(words, dtype=np.uint64), ingress=ingress)


def test_offline_compute_and_mutable_object_views_write_back() -> None:
    sim = Simulator(SingleBoard())
    output = execute(
        sim,
        [
            *offline_config(CPU, ROOT, threshold=100),
            work(CPU, ROOT, 7),
            control(CPU, ROOT, 12, 1),
        ],
    )
    assert output.tolist() == [0xE000000000000000]
    core = sim.array.chip(C0).core(CoreCoord(0, 2))
    assert isinstance(core, OfflineCore)
    neuron = core.neurons[0]
    assert neuron.layout is OfflineLayout.FULL and neuron.voltage == 7
    revision = core.revision
    neuron.voltage = -4
    assert int(core.neuron_sram[0, 0]) & 0xFFFF_FFFF == 0xFFFF_FFFC
    neuron.set_parameter("thres_pos", 23)
    assert neuron.parameter("thres_pos") == 23
    core.lut[5] = (-7, 9)
    assert core.lut[5] == (-7, 9)
    low, high = core.weights[8]
    core.weights[8] = (low + 1, high)
    assert core.revision >= revision + 4
    assert not core.neuron_sram.flags.writeable
    assert not core.weights.array().flags.writeable
    assert not core.lut.array().flags.writeable


def test_partial_package_is_inert_then_commits_atomically() -> None:
    stream = offline_config(CPU, ROOT)
    sim = Simulator(SingleBoard())
    _feed(sim, stream[:2])
    sim.run()
    assert sim.array.chip(C0).cores == {}
    with pytest.raises(InvalidFrame, match="truncated"):
        sim.finish_input()
    _feed(sim, stream[2:])
    sim.run()
    assert isinstance(sim.array.chip(C0).core(ROOT.core), OfflineCore)


def test_pending_packet_separates_frame_header_from_raw_word() -> None:
    sim = Simulator(SingleBoard())
    header = offline_config(CPU, ROOT)[0]
    _feed(sim, [header])
    sim.run()
    packet = sim._packets[CPU]
    assert packet.header is FrameHeader.CONFIG_TYPE1
    assert packet.raw_word == header


def test_configured_core_cannot_be_replaced() -> None:
    sim = Simulator(SingleBoard())
    execute(sim, offline_config(CPU, ROOT))
    core = sim.array.chip(C0).core(ROOT.core)
    assert isinstance(core, OfflineCore)
    before = core.registers.copy()
    with pytest.raises(InvalidFrame, match="cannot be replaced"):
        execute(sim, offline_config(CPU, ROOT, threshold=99))
    assert np.array_equal(core.registers, before)


def test_initialize_requires_a_configured_thread() -> None:
    with pytest.raises(ValueError, match="no configured threads"):
        Simulator(SingleBoard()).initialize()


def test_zero_step_sync_completes_without_advancing() -> None:
    sim = Simulator(SingleBoard())
    output = execute(
        sim,
        [
            *offline_config(CPU, ROOT),
            control(CPU, ROOT, 12, 0),
        ],
    )
    core = sim.array.chip(C0).core(ROOT.core)
    assert isinstance(core, OfflineCore)
    assert output.tolist() == [0xE000000000000000]
    assert core.work_steps == 0
    assert sim.pending_steps == 0 and not sim.scheduler.runnable


def test_bad_complete_packet_leaves_no_created_core_or_half_write() -> None:
    sim = Simulator(SingleBoard())
    _feed(sim, [route(CPU, ROOT) | 3, 4097 << 14, 0, 0])
    with pytest.raises(InvalidFrame):
        sim.run()
    assert sim.array.chip(C0).cores == {}


def test_undefined_hardware_behavior_maps_to_unsupported_frame(monkeypatch) -> None:
    sim = Simulator(SingleBoard())

    def fail(_packet: object) -> None:
        raise UndefinedHardwareBehavior("hardware behavior is undefined")

    monkeypatch.setattr(sim, "_commit_packet", fail)
    _feed(sim, offline_config(CPU, ROOT))
    with pytest.raises(
        UnsupportedFrame, match="hardware behavior is undefined"
    ) as caught:
        sim.run()
    assert caught.value.diagnostic.code == "unsupported_behavior"


def test_init_test_and_snapshot_are_observable_and_stable() -> None:
    sim = Simulator(SingleBoard())
    execute(sim, offline_config(CPU, ROOT, initial_v=5))
    execute(sim, [work(CPU, ROOT, 9), control(CPU, ROOT, 13)])
    core = sim.array.chip(C0).core(ROOT.core)
    assert isinstance(core, OfflineCore)
    assert core.neurons[0].voltage == 5 and core.input_sram.sum() == 0
    request = 6 << 60 | route(CPU, ROOT) | 1 << 23 | 2
    response = execute(sim, [request])
    assert len(response) == 3
    snap = sim.snapshot([ROOT])
    voltage = snap["cores"][ROOT]["voltage"]
    assert isinstance(voltage, np.ndarray) and not voltage.flags.writeable


def test_two_threads_are_runnable_with_independent_ticks() -> None:
    other = CoreAddr(C0, CoreCoord(0, 3))
    sim = Simulator(SingleBoard())
    _feed(
        sim,
        [
            *offline_config(CPU, ROOT, threshold=100, thread=0),
            *offline_config(CPU, other, threshold=100, thread=1),
            work(CPU, ROOT, 2),
            work(CPU, other, 3),
            control(CPU, ROOT, 12, 3),
            control(CPU, other, 12, 2),
        ],
    )
    for _ in range(32):
        if len(sim.scheduler.threads) == 2:
            break
        assert sim.advance() == 1
    threads = {item.id: item for item in sim.scheduler.threads}
    assert set(threads) == {0, 1}
    sim.advance(2)
    assert threads[0].tick > 0 and threads[1].tick > 0
    sim.run()
    assert (threads[0].tick, threads[1].tick) == (3, 2)
    assert all(item.done for item in threads.values())


def test_cross_chip_data_and_global_thread_done_busy() -> None:
    sim = Simulator(Array2x2Board())
    array = sim.array
    right = ChipCoord(1, 0)
    remote = CoreAddr(right, CoreCoord(0, 2))
    execute(sim, offline_config(CPU, remote, threshold=100, thread=4))
    _feed(sim, [work(CPU, remote, 6), control(CPU, remote, 12, 2)])
    assert sim.advance(2) == 2
    thread = sim.scheduler.threads[0]
    assert thread.busy and not thread.done and thread.budget == 2
    sim.run()
    assert thread.tick == 2 and thread.done and not thread.busy
    core = array.chip(right).core(remote.core)
    assert isinstance(core, OfflineCore) and core.neurons[0].voltage == 6
    assert sim.drain_output().tolist() == [0xE000000000000004]


def test_cross_chip_global_signal_tree_advances_all_gated_members() -> None:
    events = []
    sim = Simulator(Array2x2Board(), events.append)
    array = sim.array
    left, right = ChipCoord(0, 0), ChipCoord(1, 0)
    root = CoreAddr(left, CoreCoord(8, 2))
    remote = CoreAddr(right, CoreCoord(0, 5))
    _feed(
        sim,
        [
            *offline_config(CPU, root, send=64 | (1 << 3), thread=6),
            *offline_config(CPU, remote, receive=1 << 2, thread=9),
            control(CPU, root, 12, 1),
        ],
    )
    sim.run()
    for addr in (root, remote):
        core = array.chip(addr.chip).core(addr.core)
        assert isinstance(core, OfflineCore) and core.work_steps == 1
    assert any(
        event.kind == "signal" and event.destination == remote for event in events
    )


def test_online_storage_init_and_atomic_sync_update_rejection() -> None:
    target = CoreAddr(C0, CoreCoord(0, 1))
    bits = route(CPU, target)
    sim = Simulator(SingleBoard())
    with pytest.raises(UnsupportedFrame, match="online core execution"):
        execute(sim, [bits | 5, 0, 1 << 61, 1, 64 << 52, 1 << 48])
    assert sim.array.chip(C0).cores == {}


def test_observer_is_optional_and_reports_events_only_when_enabled() -> None:
    events = []
    sim = Simulator(SingleBoard(), events.append)
    execute(sim, offline_config(CPU, ROOT))
    assert events and events[0].kind == "frame"
    quiet = Simulator(SingleBoard())
    execute(quiet, offline_config(CPU, ROOT))
    assert quiet.snapshot()["sequence"] > 0


def test_trace_filter_selects_core_commit_events() -> None:
    events = []
    sim = Simulator(
        SingleBoard(),
        events.append,
        trace_filter=TraceFilter(
            cores=frozenset({ROOT}), kinds=frozenset({"core_step"})
        ),
    )
    execute(
        sim,
        [
            *offline_config(CPU, ROOT, threshold=100),
            work(CPU, ROOT, 7),
            control(CPU, ROOT, 12, 1),
        ],
    )
    assert [(event.kind, event.core, event.tick) for event in events] == [
        ("core_step", ROOT, 1)
    ]


def test_events_classify_ports_and_complete_follows_data_output() -> None:
    events = []
    sim = Simulator(
        SingleBoard(),
        events.append,
        trace_filter=TraceFilter(level=TraceLevel.DEBUG),
    )
    execute(
        sim,
        [
            *offline_config(CPU, ROOT, threshold=1),
            work(CPU, ROOT, 7),
            control(CPU, ROOT, 12, 1),
        ],
    )

    assert any(
        event.kind == "payload"
        and event.direction is PortDirection.E2I
        and event.port == CPU
        for event in events
    )
    outputs = [event for event in events if event.direction is PortDirection.I2E]
    assert [event.kind for event in outputs] == ["frame", "complete"]
    assert outputs[-1].kind == "complete"
    assert outputs[-1].sequence > outputs[0].sequence


def test_final_step_event_order_is_core_complete_step() -> None:
    observed = []
    sim = Simulator(
        SingleBoard(),
        observed.append,
        trace_filter=TraceFilter(kinds=frozenset({"core_step", "complete", "step"})),
    )
    execute(
        sim,
        [
            *offline_config(CPU, ROOT, threshold=100),
            work(CPU, ROOT, 7),
            control(CPU, ROOT, 12, 1),
        ],
    )
    core = sim.array.chip(C0).core(ROOT.core)
    assert isinstance(core, OfflineCore)
    assert [event.kind for event in observed] == [
        "core_step",
        "complete",
        "step",
    ]
    assert core.work_steps == 1


def test_observer_failure_preserves_committed_core_state() -> None:
    def fail(event: object) -> None:
        if getattr(event, "kind", None) == "core_step":
            raise RuntimeError("stop observing")

    sim = Simulator(SingleBoard(), fail)
    with pytest.raises(SimulationFailure) as caught:
        execute(
            sim,
            [
                *offline_config(CPU, ROOT, threshold=100),
                work(CPU, ROOT, 7),
                control(CPU, ROOT, 12, 1),
            ],
        )
    assert caught.value.diagnostic.code == "observer_error"
    assert caught.value.diagnostic.effect == "step_partial"
    core = sim.array.chip(C0).core(ROOT.core)
    assert isinstance(core, OfflineCore) and core.work_steps == 1


def test_recorder_printer_failure_uses_observer_error_semantics() -> None:
    def fail(_event: object) -> None:
        raise RuntimeError("stop printing")

    recorder = Recorder(printer=fail)
    sim = Simulator(
        SingleBoard(),
        recorder,
        trace_filter=recorder.trace_filter,
    )
    with pytest.raises(SimulationFailure) as caught:
        execute(
            sim,
            [
                *offline_config(CPU, ROOT, threshold=100),
                work(CPU, ROOT, 7),
                control(CPU, ROOT, 12, 1),
            ],
        )
    assert caught.value.diagnostic.code == "observer_error"


def test_multicast_work_failure_is_atomic(monkeypatch: pytest.MonkeyPatch) -> None:
    other = CoreAddr(C0, CoreCoord(0, 3))
    sim = Simulator(SingleBoard())
    execute(sim, offline_config(CPU, ROOT))
    core = sim.array.chip(C0).core(ROOT.core)
    assert isinstance(core, OfflineCore)
    before = core.input_sram.copy()
    monkeypatch.setattr(sim.router, "route", lambda _source, _word: (ROOT, other))
    _feed(sim, [work(CPU, ROOT, 7)])
    with pytest.raises(InvalidFrame, match="unconfigured core"):
        sim.run()
    assert np.array_equal(core.input_sram, before)


def test_multicast_storage_rejects_empty_core_before_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    other = CoreAddr(C0, CoreCoord(0, 3))
    sim = Simulator(SingleBoard())
    execute(sim, offline_config(CPU, ROOT))
    core = sim.array.chip(C0).core(ROOT.core)
    assert isinstance(core, OfflineCore)
    sim.array.chip(C0).put(RelayCore(other, send=0, receive=0, thread_id=ThreadId(0)))
    before = core.lut.array().copy()
    monkeypatch.setattr(sim.router, "route", lambda _source, _word: (ROOT, other))
    _feed(sim, [1 << 60 | 1, 7])
    with pytest.raises(UnsupportedFrame, match="empty cores"):
        sim.run()
    assert np.array_equal(core.lut.array(), before)


def test_feed_chunking_does_not_change_replay_result() -> None:
    stream = [
        *offline_config(CPU, ROOT, threshold=100),
        work(CPU, ROOT, 7),
        control(CPU, ROOT, 12, 2),
    ]
    whole = Simulator(SingleBoard())
    expected = execute(whole, stream)
    split = Simulator(SingleBoard())
    for word in stream:
        _feed(split, [word])
    split.run()
    assert np.array_equal(split.drain_output(), expected)
    assert split.snapshot()["threads"] == whole.snapshot()["threads"]


@pytest.mark.parametrize(
    "words",
    [
        [1],
        iter([1]),
        np.asarray([1], dtype=np.int64),
        np.asarray([[1]], dtype=np.uint64),
        np.asarray([1.0]),
    ],
    ids=["list", "iterator", "signed-array", "rank", "float-array"],
)
def test_feed_rejects_untrusted_values_without_state_change(words) -> None:
    sim = Simulator(SingleBoard())
    with pytest.raises(InvalidInput):
        sim.feed(words, ingress=CPU)
    assert sim.pending_words == 0 and sim.array.chip(C0).cores == {}
    execute(sim, offline_config(CPU, ROOT))


def test_feed_rejects_array_over_queue_limit_before_mutation() -> None:
    sim = Simulator(SingleBoard(), max_pending_words=8)
    with pytest.raises(ResourceLimit):
        sim.feed(np.zeros(9, dtype=np.uint64), ingress=CPU)
    assert sim.pending_words == 0


def test_input_ring_wraps_after_256_slots_without_losing_thread_state() -> None:
    sim = Simulator(SingleBoard())
    execute(sim, offline_config(CPU, ROOT, threshold=10000))
    for tick in range(300):
        execute(
            sim,
            [work(CPU, ROOT, 1, slot=tick % 256), control(CPU, ROOT, 12, 1)],
        )
    core = sim.array.chip(C0).core(ROOT.core)
    assert isinstance(core, OfflineCore)
    assert core.work_steps == 300 and core.neurons[0].voltage == 300
    assert core.input_sram.sum() == 0
