import io
import json
from pathlib import Path
from typing import Literal, cast

from paisim import (
    ChipCoord,
    Port,
    PortDirection,
    Recorder,
    SimEvent,
    TerminalPrinter,
    ThreadId,
    TraceFilter,
    TraceLevel,
)
from paisim.runtime.observability import EventKind
from paisim.topology.types import CoreAddr, CoreCoord

CPU = Port(ChipCoord(0, 0))
CORE = CoreAddr(ChipCoord(0, 0), CoreCoord(0, 2))


def event(
    sequence: int,
    kind: EventKind,
    direction: PortDirection | None,
    word: int | None = None,
) -> SimEvent:
    return SimEvent(
        sequence=sequence,
        kind=kind,
        direction=direction,
        source=CORE,
        destination=CPU.addr,
        port=CPU if direction is not None else None,
        word=word,
        raw_word=word,
        thread=ThreadId(1),
        tick=sequence,
        core=CORE,
    )


def test_trace_levels_include_outputs_and_progress() -> None:
    output = event(1, "complete", PortDirection.I2E, 1)
    step = event(2, "step", None)
    internal = event(3, "frame", PortDirection.I2I, 2)
    assert TraceLevel.OUTPUT.includes(output.kind, output.direction)
    assert not TraceLevel.OUTPUT.includes(step.kind, step.direction)
    assert TraceLevel.PROGRESS.includes(step.kind, step.direction)
    assert TraceLevel.TRAFFIC.includes(internal.kind, internal.direction)


def test_trace_filter_outputs_shortcut_and_port_filter() -> None:
    recorder = Recorder(TraceFilter.outputs(), max_records=10)
    recorder(event(1, "complete", PortDirection.I2E, 1))
    recorder(event(2, "step", None))
    assert [item.sequence for item in recorder.records] == [1]

    filtered = TraceFilter(
        level=TraceLevel.OUTPUT,
        directions=frozenset({PortDirection.I2E}),
        ports=frozenset({CPU}),
        threads=frozenset({ThreadId(1)}),
        ticks=frozenset({3}),
        kinds=frozenset({"complete"}),
    )
    assert filtered.matches(event(3, "complete", PortDirection.I2E, 3))
    assert not filtered.matches(event(4, "complete", PortDirection.I2E, 4))


def test_recorder_retains_recent_records_and_exports(tmp_path: Path) -> None:
    recorder = Recorder(TraceFilter.outputs(), max_records=2)
    for sequence in range(3):
        recorder(event(sequence, "complete", PortDirection.I2E, sequence))
    assert [item.sequence for item in recorder.records] == [1, 2]
    assert recorder.dropped_count == 1

    json_path = tmp_path / "events.jsonl"
    recorder.export(json_path, "jsonl")
    lines = [json.loads(line) for line in json_path.read_text().splitlines()]
    assert [line["sequence"] for line in lines] == [1, 2]

    raw_path = tmp_path / "output.bin"
    recorder.export(raw_path, "raw-u64-le")
    assert raw_path.read_bytes() == (1).to_bytes(8, "little") + (2).to_bytes(
        8, "little"
    )
    recorder.clear()
    assert recorder.records == () and recorder.dropped_count == 0


def test_terminal_printer_formats_words() -> None:
    for format, expected in (
        ("hex", "0x0000000000000001"),
        ("decimal", "1"),
        (
            "bin",
            "0b00000000_00000000_00000000_00000000_00000000_00000000_00000000_00000001",
        ),
    ):
        stream = io.StringIO()
        TerminalPrinter(cast("Literal['hex', 'decimal', 'bin']", format), stream)(
            event(1, "complete", PortDirection.I2E, 1)
        )
        assert f"word={expected}" in stream.getvalue()
