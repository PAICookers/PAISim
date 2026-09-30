"""Logical frame routing with explicit chip identity and declared board hops."""

from dataclasses import dataclass
from functools import lru_cache

from paicorelib.framelib.frame_defs import FFV2
from paicorelib.framelib.parser_v2 import decode_aer_route_fields

from ..frames import ROUTE_MASK
from ..hardware import CORE_COORD_MIN, CORE_GRID_COORD_MAX
from .types import ChipArray, ChipCoord, CoreAddr, CoreCoord, Port


@dataclass(frozen=True, slots=True)
class _Pos:
    chip: ChipCoord
    core: CoreCoord

    @property
    def addr(self) -> CoreAddr:
        return CoreAddr(self.chip, self.core)


class FrameRouter:
    """Expand Z/X/Y AER routes while validating board edges.

    Routing is pure with respect to core state: it returns destination
    :class:`CoreAddr` values and raises for undeclared board hops or unsupported
    diagonal boundary crossings.  It does not enqueue, write, or execute a
    frame.
    """

    def __init__(
        self, array: ChipArray, outputs: tuple[Port, ...] | None = None
    ) -> None:
        self.array = array
        self.outputs = frozenset(
            port.addr
            for port in (
                outputs
                if outputs is not None
                else tuple(Port(coord) for coord in array.chip_coords)
            )
        )

    def route(self, source: CoreAddr, word: int) -> tuple[CoreAddr, ...]:
        """Return deduplicated destinations encoded by ``word`` from ``source``."""
        return self._route(
            source.chip.x,
            source.chip.y,
            source.core.x,
            source.core.y,
            word & ROUTE_MASK,
        )

    @lru_cache(maxsize=8192)
    def _route(
        self, chip_x: int, chip_y: int, core_x: int, core_y: int, bits: int
    ) -> tuple[CoreAddr, ...]:
        source = CoreAddr(ChipCoord(chip_x, chip_y), CoreCoord(core_x, core_y))
        fields = tuple(
            (bits >> shift) & FFV2.GENERAL_CORE_XY_ADDR_MASK
            for shift in (
                FFV2.GENERAL_CORE_XY_ADDR_OFFSET,
                FFV2.GENERAL_CORE_X_ADDR_OFFSET,
                FFV2.GENERAL_CORE_Y_ADDR_OFFSET,
                FFV2.GENERAL_COPY_XY_ADDR_OFFSET,
                FFV2.GENERAL_COPY_X_ADDR_OFFSET,
                FFV2.GENERAL_COPY_Y_ADDR_OFFSET,
            )
        )
        route = decode_aer_route_fields(*fields)
        positions = [_Pos(source.chip, source.core)]
        positions = self._stage(
            positions, route.offset_z, route.copy_z, 1, 1, diagonal=True
        )
        positions = self._stage(
            positions, route.offset_x, route.copy_x, 1, 0, diagonal=False
        )
        positions = self._stage(
            positions, route.offset_y, route.copy_y, 0, 1, diagonal=False
        )
        seen: set[CoreAddr] = set()
        result: list[CoreAddr] = []
        for pos in positions:
            if pos.addr not in seen:
                seen.add(pos.addr)
                result.append(pos.addr)
        return tuple(result)

    def is_output(self, addr: CoreAddr) -> bool:
        """Return whether ``addr`` is one of the board's declared output ports."""
        return addr in self.outputs

    def _stage(
        self,
        positions: list[_Pos],
        offset: int,
        copies: int,
        dx: int,
        dy: int,
        diagonal: bool,
    ) -> list[_Pos]:
        moved = [self._move(pos, offset, dx, dy, diagonal) for pos in positions]
        if not copies:
            return moved
        sign = 1 if copies > 0 else -1
        expanded: list[_Pos] = []
        for pos in moved:
            expanded.append(pos)
            current = pos
            for _ in range(abs(copies)):
                current = self._move(current, sign, dx, dy, diagonal)
                expanded.append(current)
        return expanded

    def _move(self, pos: _Pos, count: int, dx: int, dy: int, diagonal: bool) -> _Pos:
        sign = 1 if count >= 0 else -1
        current = pos
        for _ in range(abs(count)):
            x = current.core.x + sign * dx
            y = current.core.y + sign * dy
            if diagonal and not (
                CORE_COORD_MIN <= x <= CORE_GRID_COORD_MAX
                and CORE_COORD_MIN <= y <= CORE_GRID_COORD_MAX
            ):
                raise NotImplementedError("XY route cannot cross a chip boundary")
            chip_x, chip_y = current.chip.x, current.chip.y
            if x < 0:
                chip_x -= 1
                x = CORE_GRID_COORD_MAX
            elif x > CORE_GRID_COORD_MAX:
                chip_x += 1
                x = 0
            if y < 0:
                chip_y -= 1
                y = CORE_GRID_COORD_MAX
            elif y > CORE_GRID_COORD_MAX:
                chip_y += 1
                y = 0
            if chip_x < CORE_COORD_MIN or chip_y < CORE_COORD_MIN:
                raise NotImplementedError("route leaves the configured chip array")
            chip = ChipCoord(chip_x, chip_y)
            if not self.array.has_chip(chip):
                raise NotImplementedError("route leaves the configured chip array")
            if chip != current.chip:
                self.array.link(current.chip, chip)
            current = _Pos(chip, CoreCoord(x, y))
        return current

    def delivered(self, word: int) -> int:
        """Strip route bits and return the protocol payload/header remainder."""
        return word & ~ROUTE_MASK
