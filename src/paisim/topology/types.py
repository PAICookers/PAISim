"""Physical chip containment and explicit X/Y board signal links."""

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NewType

from ..hardware import (
    CORE_COORD_MIN,
    CORE_GRID_COORD_MAX,
    CORE_GRID_SIDE,
    ONLINE_CORE_ROW_COUNT,
)

if TYPE_CHECKING:
    from ..engine.core import Core
    from ..runtime.scheduler import CoreThread

ThreadId = NewType("ThreadId", int)


@dataclass(frozen=True, order=True, slots=True)
class ChipCoord:
    """Non-negative physical chip coordinate in the board mesh."""

    x: int
    y: int

    def __post_init__(self) -> None:
        if min(self.x, self.y) < 0:
            raise ValueError("chip coordinates must be nonnegative integers")


@dataclass(frozen=True, order=True, slots=True)
class CoreCoord:
    """Nine-by-nine core coordinate within one chip.

    ``(0, 0)`` is the CPU endpoint.  The upper two rows, except the CPU, are
    online-core positions; all other valid coordinates are offline-capable
    positions.  The simulator uses this identity when resolving route fields.
    """

    x: int
    y: int

    def __post_init__(self) -> None:
        if not (
            CORE_COORD_MIN <= self.x <= CORE_GRID_COORD_MAX
            and CORE_COORD_MIN <= self.y <= CORE_GRID_COORD_MAX
        ):
            raise ValueError(
                "core coordinate must be within the "
                f"{CORE_GRID_SIDE}x{CORE_GRID_SIDE} chip map"
            )

    @property
    def online(self) -> bool:
        """Return whether this coordinate is one of the online-core slots."""
        return self.y < ONLINE_CORE_ROW_COUNT and (self.x, self.y) != (0, 0)

    @property
    def cpu(self) -> bool:
        """Return whether this coordinate is the chip CPU endpoint ``(0, 0)``."""
        return (self.x, self.y) == (0, 0)


@dataclass(frozen=True, order=True, slots=True)
class CoreAddr:
    """Stable address combining a physical chip and an in-chip core coordinate."""

    chip: ChipCoord
    core: CoreCoord


@dataclass(frozen=True, slots=True)
class Port:
    """Host CPU endpoint declared by a board.

    ``Port(chip)`` addresses that chip's CPU core ``(0, 0)``.  ``addr`` is the
    corresponding :class:`CoreAddr`; pass the ``Port`` itself to
    :meth:`Simulator.feed`, not an arbitrary core address.
    """

    chip: ChipCoord
    core: CoreCoord = CoreCoord(0, 0)

    def __post_init__(self) -> None:
        if not isinstance(self.chip, ChipCoord) or not isinstance(self.core, CoreCoord):
            raise TypeError("port needs a chip and core coordinate")

    @property
    def addr(self) -> CoreAddr:
        """Return the core address represented by this host port."""
        return CoreAddr(self.chip, self.core)


@dataclass(frozen=True, slots=True)
class SignalLink:
    """One untagged X/Y board signal link shared by nine edge cores.

    The link models physical fanout only; it does not carry thread ownership
    metadata.  Thread filtering happens at each receiving core's local signal
    bits, matching the simulator's hardware-level boundary.
    """

    a: ChipCoord
    b: ChipCoord

    def other(self, chip: ChipCoord) -> ChipCoord:
        """Return the opposite endpoint or reject a chip outside this link."""
        if chip == self.a:
            return self.b
        if chip == self.b:
            return self.a
        raise ValueError(f"chip {chip} is not on signal link {self.a}->{self.b}")

    def fanout(self, source: CoreAddr) -> tuple[CoreAddr, ...]:
        """Return the nine physical receivers after the board-side 9-to-1 merge."""
        target = self.other(source.chip)
        dx, dy = target.x - source.chip.x, target.y - source.chip.y
        if dx:
            expected = CORE_GRID_COORD_MAX if dx > 0 else CORE_COORD_MIN
            if source.core.x != expected:
                raise ValueError("source is not on the X boundary")
            edge = CORE_COORD_MIN if dx > 0 else CORE_GRID_COORD_MAX
            return tuple(
                CoreAddr(target, CoreCoord(edge, y)) for y in range(CORE_GRID_SIDE)
            )
        expected = CORE_GRID_COORD_MAX if dy > 0 else CORE_COORD_MIN
        if source.core.y != expected:
            raise ValueError("source is not on the Y boundary")
        edge = CORE_COORD_MIN if dy > 0 else CORE_GRID_COORD_MAX
        return tuple(
            CoreAddr(target, CoreCoord(x, edge)) for x in range(CORE_GRID_SIDE)
        )


@dataclass(slots=True)
class Chip:
    """Runtime container for one chip's configured cores and threads."""

    coord: ChipCoord
    cores: dict[CoreCoord, "Core"] = field(default_factory=dict)
    threads: dict[ThreadId, "CoreThread"] = field(default_factory=dict)

    def core(self, coord: CoreCoord) -> "Core":
        """Return a configured core or raise ``KeyError`` for an empty slot."""
        try:
            return self.cores[coord]
        except KeyError as exc:
            raise KeyError(f"unconfigured core {CoreAddr(self.coord, coord)}") from exc

    def put(self, core: "Core") -> None:
        """Install a core after checking that it belongs to this chip."""
        if core.addr.chip != self.coord:
            raise ValueError("core belongs to another chip")
        self.cores[core.addr.core] = core


class ChipArray:
    """Fresh runtime chips and only the declared physical X/Y links.

    A ``ChipArray`` is created per simulator, so core mutations and thread
    state are never shared between simulator instances.  Use :meth:`chip`,
    :meth:`link`, and :meth:`iter_chips` for traversal; the backing mapping is
    intentionally private.
    """

    def __init__(
        self,
        chips: tuple[ChipCoord, ...],
        edges: tuple[tuple[ChipCoord, ChipCoord], ...],
    ) -> None:
        if not chips or len(set(chips)) != len(chips):
            raise ValueError("chip coordinates must be nonempty and unique")
        self._chips = {coord: Chip(coord) for coord in chips}
        self.links = tuple(SignalLink(a, b) for a, b in edges)
        self._link_map = {frozenset((link.a, link.b)): link for link in self.links}

    def chip(self, coord: ChipCoord) -> Chip:
        """Return the runtime chip at ``coord`` or raise ``KeyError``."""
        if not self.has_chip(coord):
            raise KeyError(f"chip {coord} is not present")
        return self._chips[coord]

    def link(self, a: ChipCoord, b: ChipCoord) -> SignalLink:
        """Return a declared undirected link or raise ``ValueError``."""
        link = frozenset((a, b))
        if link not in self._link_map:
            raise ValueError(f"chips {a} and {b} have no declared X/Y link")
        return self._link_map[frozenset((a, b))]

    def iter_chips(self) -> Iterator[Chip]:
        """Iterate runtime chips without exposing the backing map."""
        return iter(self._chips.values())

    @property
    def chip_coords(self) -> tuple[ChipCoord, ...]:
        """Return the declared chip coordinates."""
        return tuple(self._chips)

    def has_chip(self, coord: ChipCoord) -> bool:
        """Return whether ``coord`` belongs to this runtime board snapshot."""
        return coord in self._chips
