import pytest

from paisim import Array2x2Board, ChipCoord, CoreAddr, CoreCoord, Simulator
from paisim.engine.core import CompletionCore, RelayCore
from paisim.frames import route_bits
from paisim.runtime.scheduler import SignalTree
from paisim.topology.routing import FrameRouter
from paisim.topology.types import ThreadId


def test_t4_has_exactly_four_xy_links() -> None:
    array = Simulator(Array2x2Board()).array
    edges = {frozenset((link.a, link.b)) for link in array.links}
    assert edges == {
        frozenset((ChipCoord(0, 0), ChipCoord(1, 0))),
        frozenset((ChipCoord(0, 1), ChipCoord(1, 1))),
        frozenset((ChipCoord(0, 0), ChipCoord(0, 1))),
        frozenset((ChipCoord(1, 0), ChipCoord(1, 1))),
    }


def test_data_crosses_x_and_y_but_diagonal_chip_needs_two_edges() -> None:
    array = Simulator(Array2x2Board()).array
    router = FrameRouter(array)
    source = CoreAddr(ChipCoord(0, 0), CoreCoord(0, 0))
    diagonal = router.route(source, route_bits((0, 9, 11, 0, 0, 0)))
    assert diagonal == (CoreAddr(ChipCoord(1, 1), CoreCoord(0, 2)),)
    assert array.link(ChipCoord(0, 0), ChipCoord(1, 0))
    assert array.link(ChipCoord(1, 0), ChipCoord(1, 1))


@pytest.mark.parametrize("copies", [False, True])
def test_every_xy_boundary_cross_is_rejected(copies: bool) -> None:
    router = FrameRouter(Simulator(Array2x2Board()).array)
    source = CoreAddr(ChipCoord(0, 0), CoreCoord(8, 8))
    route = (0, 0, 0, 1, 0, 0) if copies else (1, 0, 0, 0, 0, 0)
    with pytest.raises(NotImplementedError, match="XY route"):
        router.route(source, route_bits(route))


def test_signal_link_fans_one_boundary_source_to_nine_receivers() -> None:
    array = Simulator(Array2x2Board()).array
    link = array.link(ChipCoord(0, 0), ChipCoord(1, 0))
    source = CoreAddr(ChipCoord(0, 0), CoreCoord(8, 4))
    targets = link.fanout(source)
    assert len(targets) == 9
    assert {addr.core for addr in targets} == {CoreCoord(0, y) for y in range(9)}


def test_global_signal_uses_9_to_1_to_9_receive_gates() -> None:
    array = Simulator(Array2x2Board()).array
    left, right = ChipCoord(0, 0), ChipCoord(1, 0)
    root = CoreAddr(left, CoreCoord(8, 4))
    array.chip(left).put(RelayCore(root, send=1 << 3, receive=0, thread_id=ThreadId(3)))
    for y in range(9):
        addr = CoreAddr(right, CoreCoord(0, y))
        core = (
            CompletionCore(addr, send=64, receive=1 << 2, thread_id=ThreadId(3))
            if y == 0
            else RelayCore(addr, send=0, receive=1 << 2, thread_id=ThreadId(3))
        )
        array.chip(right).put(core)
    tree = SignalTree.build(array, root)
    assert len(tree.nodes) == 10
    assert tree.completions == (CoreAddr(right, CoreCoord(0, 0)),)


def test_untagged_signal_link_has_no_thread_owner() -> None:
    array = Simulator(Array2x2Board()).array
    left, right = ChipCoord(0, 0), ChipCoord(1, 0)
    root = CoreAddr(left, CoreCoord(8, 4))
    remote = CoreAddr(right, CoreCoord(0, 0))
    array.chip(left).put(RelayCore(root, send=1 << 3, receive=0, thread_id=ThreadId(1)))
    array.chip(right).put(
        CompletionCore(remote, send=64, receive=1 << 2, thread_id=ThreadId(2))
    )
    tree = SignalTree.build(array, root)
    assert remote in tree.nodes


def test_signal_tree_order_is_independent_of_core_insertion_order() -> None:
    def build(order: range) -> tuple[CoreAddr, ...]:
        array = Simulator(Array2x2Board()).array
        left, right = ChipCoord(0, 0), ChipCoord(1, 0)
        root = CoreAddr(left, CoreCoord(8, 4))
        array.chip(left).put(
            RelayCore(root, send=1 << 3, receive=0, thread_id=ThreadId(3))
        )
        for y in order:
            addr = CoreAddr(right, CoreCoord(0, y))
            array.chip(right).put(
                CompletionCore(addr, send=64, receive=1 << 2, thread_id=ThreadId(3))
            )
        return SignalTree.build(array, root).nodes

    assert build(range(9)) == build(range(8, -1, -1))


def test_failed_signal_tree_emits_no_edges() -> None:
    array = Simulator(Array2x2Board()).array
    left, right = ChipCoord(0, 0), ChipCoord(1, 0)
    root = CoreAddr(left, CoreCoord(8, 4))
    remote = CoreAddr(right, CoreCoord(0, 4))
    array.chip(left).put(RelayCore(root, send=1 << 3, receive=0, thread_id=ThreadId(3)))
    array.chip(right).put(
        RelayCore(remote, send=0, receive=1 << 2, thread_id=ThreadId(3))
    )
    events = []
    with pytest.raises(ValueError, match="no local participant"):
        SignalTree.build(array, root, lambda *edge: events.append(edge))
    assert events == []
