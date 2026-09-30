"""Host diagnostics for invalid stimulus and unsupported simulation behavior.

Diagnostics are not PAICORE response frames. A failure never fabricates an ACK
or COMPLETE; the retained state and output prefix describe effects already made.
"""

from dataclasses import dataclass
from typing import Literal

ErrorCode = Literal[
    "invalid_input",
    "invalid_packet",
    "invalid_address",
    "target_mismatch",
    "unsupported_target",
    "incomplete_config",
    "invalid_state",
    "unsupported_behavior",
    "resource_limit",
    "internal_error",
    "observer_error",
    "session_failed",
]
Stage = Literal[
    "api",
    "packet",
    "route",
    "configuration",
    "work",
    "control",
    "compute",
    "observer",
    "replay",
]
Effect = Literal["unchanged", "prefix_preserved", "step_partial"]


@dataclass(frozen=True)
class Diagnostic:
    """Stable context at the first rejected operation.

    ``code`` and ``stage`` classify the failure.  Optional word, source/target,
    field, tick, and expected/actual values identify the first failing boundary.
    ``effect`` states whether the accepted prefix remains, while ``recoverable``
    is reserved for a caller that can supply the missing or corrected input.
    Diagnostics describe host/simulator state; they are not PAICORE response
    frames and never imply an ACK or COMPLETE.
    """

    code: ErrorCode
    message: str
    stage: Stage
    word_index: int | None = None
    raw_word: int | None = None
    packet_header_index: int | None = None
    packet_word_index: int | None = None
    source: tuple[int, int] | None = None
    target: tuple[int, int] | None = None
    group: tuple[int, int] | None = None
    tick: int | None = None
    field: str | None = None
    expected: str | None = None
    actual: str | None = None
    recoverable: bool = False
    effect: Effect = "prefix_preserved"


class FrameError(Exception):
    """Base for public failures carrying an inspectable :class:`Diagnostic`."""

    def __init__(self, diagnostic: Diagnostic) -> None:
        self.diagnostic = diagnostic
        super().__init__(
            f"{diagnostic.code}: {diagnostic.message} "
            f"(stage={diagnostic.stage}, word={diagnostic.word_index}, "
            f"target={diagnostic.target}, tick={diagnostic.tick})"
        )


class InvalidFrame(FrameError, ValueError):
    """A modeled protocol/state constraint is violated."""


class InvalidInput(InvalidFrame, TypeError):
    """Invalid host API input, rejected before enqueueing any words."""


class UnsupportedFrame(FrameError, NotImplementedError):
    """The behavior is not modeled; this is not proof of hardware illegality."""


class ResourceLimit(FrameError, ValueError):
    """Host resource budget exceeded, not chip completion."""


class SimulationFailure(FrameError, RuntimeError):
    """Internal execution error, observer failure, or failed session reuse."""
