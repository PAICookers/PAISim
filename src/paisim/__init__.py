"""Public PAISim facade for constructing and observing a deterministic run.

The root package intentionally exports board declarations, physical addresses,
``Simulator``, and observation types.  Raw frame helpers, artifact adapters,
and core internals live in their concrete submodules.  A run returns raw
``numpy.uint64`` response arrays through ``Simulator.drain_output``; it never
prints results implicitly.
"""

from .runtime.observability import (
    PortDirection,
    SimEvent,
    SimEventHandler,
    TraceFilter,
    TraceLevel,
)
from .runtime.recorder import Recorder, TerminalPrinter
from .runtime.simulator import Simulator
from .topology.board import (
    Array2x2Board,
    BoardDecl,
    CustomBoard,
    SingleBoard,
    TargetBoard,
)
from .topology.types import ChipCoord, CoreAddr, CoreCoord, Port, ThreadId

__all__ = [
    "Array2x2Board",
    "BoardDecl",
    "ChipCoord",
    "CoreAddr",
    "CoreCoord",
    "CustomBoard",
    "Port",
    "PortDirection",
    "Recorder",
    "SimEvent",
    "SimEventHandler",
    "Simulator",
    "SingleBoard",
    "TargetBoard",
    "TerminalPrinter",
    "ThreadId",
    "TraceFilter",
    "TraceLevel",
]
