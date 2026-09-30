"""Declared board topology, immutable presets, and editable user declarations."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from paisim import (
    Array2x2Board,
    BoardDecl,
    ChipCoord,
    CoreAddr,
    CoreCoord,
    CustomBoard,
    Port,
    Simulator,
    SingleBoard,
    TargetBoard,
)
from paisim.engine.core import CompletionCore, RelayCore
from paisim.frames import route_bits
from paisim.runtime.scheduler import SignalTree
from paisim.topology.board import BoardSnapshot, snapshot_board
from paisim.topology.routing import FrameRouter
from paisim.topology.types import ThreadId

C0 = ChipCoord(0, 0)
C1 = ChipCoord(1, 0)
C2 = ChipCoord(2, 0)
CPU = Port(C0)


@pytest.mark.parametrize(
    "board,count,edges",
    [(SingleBoard(), 1, 0), (Array2x2Board(), 4, 4)],
    ids=["single", "array2x2"],
)
def test_preset_declarations_are_complete_and_immutable(
    board: TargetBoard, count: int, edges: int
) -> None:
    spec = snapshot_board(board.declaration)
    assert len(spec.chips) == count and len(spec.links) == edges
    assert spec.ingress == spec.outputs == tuple(Port(chip) for chip in spec.chips)
    with pytest.raises(FrozenInstanceError):
        board.extra = 1  # type: ignore[attr-defined]
    with pytest.raises(FrozenInstanceError):
        spec.chips = ()  # type: ignore[misc]


def test_user_subclass_edits_only_affect_future_simulators() -> None:
    class UserBoard(TargetBoard):
        def __init__(self) -> None:
            self.chips = [C0, C1]
            self.edges = [(C0, C1)]
            self.cpu = [CPU]

        @property
        def declaration(self) -> BoardDecl:
            return BoardDecl(self.chips, self.edges, self.cpu, self.cpu)

    board = UserBoard()
    first = Simulator(board)
    board.chips.append(C2)
    board.edges.append((C1, C2))
    board.cpu.append(Port(C2, CoreCoord(1, 1)))
    second = Simulator(board)
    assert set(first.array.chip_coords) == {C0, C1}
    assert set(second.array.chip_coords) == {C0, C1, C2}
    assert len(first.array.links) == 1 and len(second.array.links) == 2
    with pytest.raises(ValueError, match="ingress"):
        first.feed(np.empty(0, dtype=np.uint64), ingress=board.cpu[-1])
    second.feed(np.empty(0, dtype=np.uint64), ingress=board.cpu[-1])
    with pytest.raises(KeyError):
        first.array.chip(C2)
    with pytest.raises(FrozenInstanceError):
        first.array.link(C0, C1).a = C2  # type: ignore[misc]


def test_board_snapshot_copies_mutable_declarations() -> None:
    class Editable(TargetBoard):
        def __init__(self) -> None:
            self.chips = [C0, C1]
            self.links = [(C0, C1)]
            self.cpu = [CPU]

        @property
        def declaration(self) -> BoardDecl:
            return BoardDecl(self.chips, self.links, self.cpu, self.cpu)

    board = Editable()
    sim = Simulator(board)
    board.chips.append(C2)
    board.links.clear()
    board.cpu.clear()
    assert sim.board.chips == (C0, C1)
    assert sim.board.links == ((C0, C1),)
    assert sim.board.ingress == (CPU,)


def test_declaration_is_passive_and_snapshot_is_validation_boundary() -> None:
    declaration = BoardDecl([C0, C0], [], [CPU], [CPU])
    with pytest.raises(ValueError, match="unique"):
        snapshot_board(declaration)


def test_snapshot_rejects_direct_invalid_construction() -> None:
    with pytest.raises(ValueError, match="unique"):
        BoardSnapshot((C0, C0), (), (CPU,), (CPU,))


@pytest.mark.parametrize(
    "chips,links,ingress,outputs,message",
    [
        ([C0, C0], [], [CPU], [CPU], "unique"),
        ([C0], [(C0, C1)], [CPU], [CPU], "missing"),
        ([C0, C1], [(C0, C1), (C1, C0)], [CPU], [CPU], "duplicate"),
        ([C0, C2], [(C0, C2)], [CPU], [CPU], "adjacent"),
        ([C0, ChipCoord(1, 1)], [(C0, ChipCoord(1, 1))], [CPU], [CPU], "adjacent"),
        ([C0], [], [Port(C1)], [CPU], "CPU port"),
        ([C0], [], [CPU], [], "nonempty"),
    ],
    ids=[
        "duplicate-chip",
        "missing-chip",
        "duplicate-edge",
        "gap",
        "diagonal",
        "foreign-port",
        "no-output",
    ],
)
def test_invalid_custom_board_is_rejected_before_runtime(
    chips: list[ChipCoord],
    links: list[tuple[ChipCoord, ChipCoord]],
    ingress: list[Port],
    outputs: list[Port],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        Simulator(CustomBoard(chips, links, ingress, outputs))


def test_custom_missing_mesh_edge_blocks_diagonal_route() -> None:
    up = ChipCoord(1, 1)
    board = CustomBoard([C0, C1, up], [(C0, C1)], [CPU], [CPU])
    sim = Simulator(board)
    router = FrameRouter(sim.array, sim.board.outputs)
    with pytest.raises(ValueError, match="no declared"):
        router.route(CPU.addr, route_bits((0, 9, 11, 0, 0, 0)))


@pytest.mark.parametrize(
    "source,field",
    [
        (CoreCoord(8, 8), (1, 0, 0, 0, 0, 0)),
        (CoreCoord(0, 0), (-1, 0, 0, 0, 0, 0)),
        (CoreCoord(8, 8), (0, 0, 0, 1, 0, 0)),
        (CoreCoord(0, 0), (0, 0, 0, -1, 0, 0)),
    ],
    ids=["xy+", "xy-", "copy-xy+", "copy-xy-"],
)
def test_xy_route_never_crosses_declared_chip_edge(
    source: CoreCoord, field: tuple[int, ...]
) -> None:
    router = Simulator(Array2x2Board()).router
    with pytest.raises(NotImplementedError, match="XY route"):
        router.route(CoreAddr(C0, source), route_bits(field))


def test_custom_signal_link_fanout_beyond_two_by_two() -> None:
    board = CustomBoard([C1, C2], [(C1, C2)], [Port(C1)], [Port(C1)])
    sim = Simulator(board)
    root = CoreAddr(C1, CoreCoord(8, 4))
    remote = CoreAddr(C2, CoreCoord(0, 5))
    sim.array.chip(C1).put(
        RelayCore(root, send=1 << 3, receive=0, thread_id=ThreadId(9))
    )
    sim.array.chip(C2).put(
        CompletionCore(remote, send=64, receive=1 << 2, thread_id=ThreadId(9))
    )
    tree = SignalTree.build(sim.array, root)
    assert tree.completions == (remote,)
    assert len(sim.array.link(C1, C2).fanout(root)) == 9
