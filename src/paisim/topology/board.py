"""Board declarations and their validated runtime snapshots."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import final

from .types import ChipCoord, Port

Edge = tuple[ChipCoord, ChipCoord]


@dataclass(frozen=True, slots=True)
class BoardDecl:
    """User-authored board topology consumed by :class:`Simulator`.

    ``chips`` lists non-negative chip coordinates.  ``links`` lists explicit
    adjacent X/Y chip pairs, while ``ingress`` and ``outputs`` list host CPU
    ports.  The declaration is input data only: validation and tuple copying
    happen in :func:`snapshot_board`, so mutating a custom board after a
    simulator is constructed cannot alter that simulator's topology.
    """

    chips: Sequence[ChipCoord]
    links: Sequence[Edge]
    ingress: Sequence[Port]
    outputs: Sequence[Port]


@dataclass(frozen=True, slots=True)
class BoardSnapshot:
    """Validated immutable board data owned by one simulator instance.

    All collections are tuples.  A snapshot identifies the physical routing
    graph and legal CPU endpoints; it does not contain runtime core/thread
    state and can be safely included in :meth:`Simulator.snapshot` output.
    """

    chips: tuple[ChipCoord, ...]
    links: tuple[Edge, ...]
    ingress: tuple[Port, ...]
    outputs: tuple[Port, ...]

    def __post_init__(self) -> None:
        _validate_board(self.chips, self.links, self.ingress, self.outputs)


def snapshot_board(decl: BoardDecl) -> BoardSnapshot:
    """Copy a board declaration into a validated simulator snapshot.

    Args:
        decl: Ordered chip, link, ingress, and output collections.

    Returns:
        A detached :class:`BoardSnapshot` with tuple-backed collections.

    Raises:
        TypeError: If the declaration or one of its collections has the wrong
            shape or element type.
        ValueError: If chips, ports, or links are duplicated, missing, or not
            adjacent in the X/Y mesh.
    """
    return BoardSnapshot(
        tuple(decl.chips), tuple(decl.links), tuple(decl.ingress), tuple(decl.outputs)
    )


def _validate_board(
    chips: tuple[ChipCoord, ...],
    links: tuple[Edge, ...],
    ingress: tuple[Port, ...],
    outputs: tuple[Port, ...],
) -> None:
    if not all(isinstance(items, tuple) for items in (chips, links, ingress, outputs)):
        raise TypeError("board snapshot fields must be tuples")
    if not all(isinstance(chip, ChipCoord) for chip in chips):
        raise TypeError("board chips must be ChipCoord objects")
    if not all(isinstance(port, Port) for port in (*ingress, *outputs)):
        raise TypeError("board CPU ports must be Port objects")
    if not chips or len(set(chips)) != len(chips):
        raise ValueError("board chips must be nonempty and unique")
    if not ingress or not outputs:
        raise ValueError("board CPU ingress and output ports must be nonempty")
    if len(set(ingress)) != len(ingress) or len(set(outputs)) != len(outputs):
        raise ValueError("board CPU ports must be unique")

    chip_set = set(chips)
    if any(port.chip not in chip_set for port in (*ingress, *outputs)):
        raise ValueError("CPU port references a chip outside the board")
    _validate_links(links, chip_set)


def _validate_links(links: tuple[Edge, ...], chips: set[ChipCoord]) -> None:
    edges: set[frozenset[ChipCoord]] = set()
    for pair in links:
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise TypeError("board links must be pairs of chip coordinates")
        a, b = pair
        if not isinstance(a, ChipCoord) or not isinstance(b, ChipCoord):
            raise TypeError("board link endpoints must be ChipCoord objects")
        if a not in chips or b not in chips:
            raise ValueError("board link references a missing chip")
        if abs(a.x - b.x) + abs(a.y - b.y) != 1:
            raise ValueError("board links must join adjacent X/Y chips")
        edge = frozenset((a, b))
        if edge in edges:
            raise ValueError("duplicate board link")
        edges.add(edge)


class TargetBoard(ABC):
    """Board declaration interface; runtime behavior remains in ``Simulator``.

    Implementations expose passive :class:`BoardDecl` data through
    ``declaration``.  They do not own live chips, threads, or a connection to a
    physical board.
    """

    @property
    @abstractmethod
    def declaration(self) -> BoardDecl:
        """Return this board's editable or immutable source declaration."""


@dataclass(slots=True)
class CustomBoard(TargetBoard):
    """Editable board declaration backed by ordinary Python lists.

    Use this for a custom functional topology.  Supply every chip, X/Y link,
    CPU ingress, and output explicitly; missing links are not inferred.  The
    lists may be edited until a :class:`Simulator` snapshots them.
    """

    chips: list[ChipCoord]
    links: list[Edge]
    ingress: list[Port]
    outputs: list[Port]

    @property
    def declaration(self) -> BoardDecl:
        """Return the current editable lists as a passive board declaration."""
        return BoardDecl(self.chips, self.links, self.ingress, self.outputs)


_ORIGIN = ChipCoord(0, 0)
_MESH = (_ORIGIN, ChipCoord(1, 0), ChipCoord(0, 1), ChipCoord(1, 1))
_SINGLE = BoardDecl((_ORIGIN,), (), (Port(_ORIGIN),), (Port(_ORIGIN),))
_ARRAY2X2 = BoardDecl(
    _MESH,
    (
        (_MESH[0], _MESH[1]),
        (_MESH[2], _MESH[3]),
        (_MESH[0], _MESH[2]),
        (_MESH[1], _MESH[3]),
    ),
    tuple(Port(chip) for chip in _MESH),
    tuple(Port(chip) for chip in _MESH),
)


@final
@dataclass(frozen=True, slots=True)
class SingleBoard(TargetBoard):
    """One-chip preset at ``ChipCoord(0, 0)`` with one CPU port and no links."""

    @property
    def declaration(self) -> BoardDecl:
        """Return the immutable single-chip declaration."""
        return _SINGLE


@final
@dataclass(frozen=True, slots=True)
class Array2x2Board(TargetBoard):
    """Four-chip X/Y mesh preset with one CPU port per chip."""

    @property
    def declaration(self) -> BoardDecl:
        """Return the four-chip declaration and its four adjacent links."""
        return _ARRAY2X2
