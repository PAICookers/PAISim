"""Typed event selection for the frame-driven simulator."""

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Literal

from ..topology.types import CoreAddr, Port, ThreadId

EventKind = Literal["frame", "payload", "signal", "step", "complete", "core_step"]


class PortDirection(Enum):
    """Direction of a frame relative to the simulator boundary."""

    E2I = "e2i"
    I2I = "i2i"
    I2E = "i2e"


class TraceLevel(Enum):
    """Cumulative observation detail from output-only to core debugging."""

    OUTPUT = "output"
    PROGRESS = "progress"
    TRAFFIC = "traffic"
    DEBUG = "debug"

    def includes(self, kind: EventKind, direction: PortDirection | None) -> bool:
        """Return whether this level includes one event."""
        if self is TraceLevel.OUTPUT:
            return direction is PortDirection.I2E and kind in {
                "frame",
                "payload",
                "complete",
            }
        if self is TraceLevel.PROGRESS:
            return TraceLevel.OUTPUT.includes(kind, direction) or kind == "step"
        if self is TraceLevel.TRAFFIC:
            return TraceLevel.PROGRESS.includes(kind, direction) or kind in {
                "frame",
                "payload",
                "signal",
            }
        return True


@dataclass(frozen=True, slots=True)
class SimEvent:
    """Immutable event delivered after a modeled operation boundary.

    ``sequence`` is monotonic within one simulator. ``kind`` identifies the
    modeled boundary and ``direction`` identifies movement across the
    simulator boundary when the event is a frame or signal. ``destination``
    remains a physical core address; ``port`` identifies a declared host
    endpoint when applicable. ``word`` is the event value and ``raw_word``
    preserves the routed protocol word when one exists.
    """

    sequence: int
    kind: EventKind
    direction: PortDirection | None = None
    source: CoreAddr | None = None
    destination: CoreAddr | None = None
    port: Port | None = None
    word: int | None = None
    raw_word: int | None = None
    thread: ThreadId | None = None
    tick: int | None = None
    core: CoreAddr | None = None
    route: tuple[CoreAddr, ...] | None = None


@dataclass(frozen=True, slots=True)
class TraceFilter:
    """Select which events reach a simulator callback.

    All supplied fields are combined with logical AND. ``None`` means
    unrestricted; an empty allow-list matches nothing.
    """

    level: TraceLevel | None = None
    directions: frozenset[PortDirection] | None = None
    ports: frozenset[Port] | None = None
    cores: frozenset[CoreAddr] | None = None
    threads: frozenset[ThreadId] | None = None
    ticks: frozenset[int] | None = None
    kinds: frozenset[EventKind] | None = None

    @classmethod
    def outputs(cls) -> "TraceFilter":
        """Return the default filter for declared board outputs."""
        return cls(level=TraceLevel.OUTPUT)

    def matches(self, event: SimEvent) -> bool:
        """Return whether an already-constructed event passes this filter."""
        if self.level is not None and not self.level.includes(
            event.kind, event.direction
        ):
            return False
        if self.directions is not None and event.direction not in self.directions:
            return False
        if self.ports is not None and event.port not in self.ports:
            return False
        if self.kinds is not None and event.kind not in self.kinds:
            return False
        if self.threads is not None and event.thread not in self.threads:
            return False
        if self.ticks is not None and event.tick not in self.ticks:
            return False
        if self.cores is not None:
            addresses = {
                value
                for value in (event.source, event.destination, event.core)
                if value is not None
            }
            if not addresses.intersection(self.cores):
                return False
        return True


SimEventHandler = Callable[[SimEvent], None]
"""Callback type used by :class:`~paisim.runtime.simulator.Simulator`."""
