"""Bounded event recording and human-readable output for PAISim."""

import json
import sys
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, TextIO

from ..topology.types import CoreAddr, Port
from .observability import PortDirection, SimEvent, TraceFilter

RecordExportFormat = Literal["jsonl", "raw-u64-le"]
TerminalFormat = Literal["hex", "decimal", "bin"]


class TerminalPrinter:
    """Print selected events using a stable human-readable word format."""

    def __init__(
        self, format: TerminalFormat = "hex", stream: TextIO | None = None
    ) -> None:
        if format not in ("hex", "decimal", "bin"):
            raise ValueError("terminal format must be hex, decimal, or bin")
        self.format = format
        self.stream = stream

    def __call__(self, event: SimEvent) -> None:
        """Write one event to the configured text stream."""
        stream = self.stream if self.stream is not None else sys.stdout
        direction = "-" if event.direction is None else event.direction.value
        source = _address(event.source)
        port = _port(event.port)
        word = _word(event.word, self.format)
        fields = (
            f"seq={event.sequence}",
            f"kind={event.kind}",
            f"direction={direction}",
            f"source={source}",
            f"port={port}",
            f"thread={event.thread}",
            f"tick={event.tick}",
            f"word={word}",
        )
        stream.write(" ".join(fields) + "\n")


class Recorder:
    """Record selected simulator events with bounded in-memory retention.

    The recorder is callable and can be passed directly as ``on_event``. It
    keeps the newest ``max_records`` events, counts discarded older events,
    and can optionally print each retained event through ``TerminalPrinter``.
    """

    def __init__(
        self,
        trace_filter: TraceFilter | None = None,
        printer: Callable[[SimEvent], None] | None = None,
        max_records: int = 10_000,
    ) -> None:
        self.trace_filter = trace_filter or TraceFilter.outputs()
        self.printer = printer
        self._records: deque[SimEvent] = deque()
        self._dropped_count = 0
        self.max_records = max_records

    @property
    def max_records(self) -> int:
        """Return the current in-memory retention limit."""
        return self._max_records

    @max_records.setter
    def max_records(self, value: int) -> None:
        if value < 1:
            raise ValueError("max_records must be a positive int")
        self._max_records = value
        while len(self._records) > value:
            self._records.popleft()
            self._dropped_count += 1

    @property
    def records(self) -> tuple[SimEvent, ...]:
        """Return an immutable snapshot of retained events."""
        return tuple(self._records)

    @property
    def dropped_count(self) -> int:
        """Return how many older events were discarded due to the limit."""
        return self._dropped_count

    def __call__(self, event: SimEvent) -> None:
        """Record and optionally print one event accepted by the filter."""
        if not self.trace_filter.matches(event):
            return
        if len(self._records) >= self.max_records:
            self._records.popleft()
            self._dropped_count += 1
        self._records.append(event)
        if self.printer is not None:
            self.printer(event)

    def clear(self) -> None:
        """Discard retained events and reset the discarded-event counter."""
        self._records.clear()
        self._dropped_count = 0

    def export(self, path: str | Path, format: RecordExportFormat = "jsonl") -> None:
        """Export retained events to one JSONL or raw uint64 file."""
        if format == "jsonl":
            with Path(path).open("w", encoding="utf-8") as stream:
                for event in self._records:
                    stream.write(json.dumps(_event_dict(event), separators=(",", ":")))
                    stream.write("\n")
            return
        if format == "raw-u64-le":
            with Path(path).open("wb") as stream:
                for event in self._records:
                    if event.direction is PortDirection.I2E and event.word is not None:
                        stream.write(int(event.word).to_bytes(8, "little"))
            return
        raise ValueError("export format must be jsonl or raw-u64-le")


def _word(value: int | None, format: TerminalFormat) -> str:
    if value is None:
        return "-"
    value = int(value)
    if format == "hex":
        return f"0x{value:016x}"
    if format == "decimal":
        return str(value)
    bits = f"{value:064b}"
    return "0b" + "_".join(bits[index : index + 8] for index in range(0, 64, 8))


def _address(value: CoreAddr | None) -> str:
    if value is None:
        return "-"
    return f"({value.chip.x},{value.chip.y}):({value.core.x},{value.core.y})"


def _port(value: Port | None) -> str:
    if value is None:
        return "-"
    return f"({value.chip.x},{value.chip.y}):({value.core.x},{value.core.y})"


def _event_dict(event: SimEvent) -> dict[str, Any]:
    return {
        "sequence": event.sequence,
        "kind": event.kind,
        "direction": None if event.direction is None else event.direction.value,
        "source": _address_dict(event.source),
        "destination": _address_dict(event.destination),
        "port": _port_dict(event.port),
        "word": event.word,
        "raw_word": event.raw_word,
        "thread": event.thread,
        "tick": event.tick,
        "core": _address_dict(event.core),
        "route": None
        if event.route is None
        else [_address_dict(value) for value in event.route],
    }


def _address_dict(value: CoreAddr | None) -> dict[str, int] | None:
    if value is None:
        return None
    return {
        "chip_x": value.chip.x,
        "chip_y": value.chip.y,
        "core_x": value.core.x,
        "core_y": value.core.y,
    }


def _port_dict(value: Port | None) -> dict[str, int] | None:
    if value is None:
        return None
    return {
        "chip_x": value.chip.x,
        "chip_y": value.chip.y,
        "core_x": value.core.x,
        "core_y": value.core.y,
    }
