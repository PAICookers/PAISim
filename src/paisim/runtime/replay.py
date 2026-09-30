"""Bounded replay of recorded wire words, without tensor or driving-policy inference.

Manifest events reference files beneath the manifest directory. Events feed the
recorded slices in order, drain only actual SYNC budgets, and never add frames.
NPY 1.0/2.0 integer vectors and explicitly little-endian raw uint64 are supported.
"""

import hashlib
import io
import json
from pathlib import Path
from typing import Any, Literal, NamedTuple, cast

import numpy as np
from numpy.lib import format as npy_format

from ..engine.model import FrameArray
from ..topology.board import (
    Array2x2Board,
    BoardSnapshot,
    CustomBoard,
    SingleBoard,
    TargetBoard,
    snapshot_board,
)
from ..topology.types import ChipCoord, CoreCoord, Port, ThreadId
from .simulator import Simulator

FrameFormat = Literal["npy", "raw-u64-le"]
_DEFAULT_BYTES = 64 * 1024 * 1024
_MANIFEST_BYTES = 1024 * 1024


class ReplayError(ValueError):
    """Replay failed; the accepted prefix remains observable on ``simulator``."""

    def __init__(self, message: str, simulator: Simulator) -> None:
        super().__init__(message)
        self.simulator = simulator


class _Event(NamedTuple):
    path: Path
    format: FrameFormat
    sha256: str
    ingress: Port
    thread: ThreadId
    start: int
    stop: int | None


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _read_bytes(path: Path, limit: int) -> bytes:
    """Check before allocation, then bound the read even if file size changes."""
    if not path.is_file():
        raise ValueError(f"not a regular file: {path}")
    if path.stat().st_size > limit:
        raise ValueError(f"file exceeds byte budget {limit}: {path}")
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError(f"file exceeds byte budget {limit}: {path}")
    return data


def _decode_frames(data: bytes, format: FrameFormat) -> FrameArray:
    if format == "raw-u64-le":
        if len(data) % 8:
            raise ValueError("raw-u64-le byte length must be divisible by eight")
        return np.frombuffer(data, dtype="<u8").astype(np.uint64, copy=True)
    if format != "npy":
        raise ValueError(f"unsupported frame format: {format!r}")
    stream = io.BytesIO(data)
    version = npy_format.read_magic(stream)
    if version == (1, 0):
        shape, _, dtype = npy_format.read_array_header_1_0(stream)
    elif version == (2, 0):
        shape, _, dtype = npy_format.read_array_header_2_0(stream)
    else:
        raise ValueError(f"unsupported NPY version: {version}")
    if len(shape) != 1 or dtype.kind not in "iu" or dtype.itemsize not in (1, 2, 4, 8):
        raise ValueError(
            "NPY frames must be a one-dimensional integer array <= 64 bits"
        )
    count = _integer(shape[0], "NPY frame count")
    offset = stream.tell()
    if count * dtype.itemsize != len(data) - offset:
        raise ValueError("NPY payload length does not match its declared shape")
    # Reading from an already bounded buffer never allocates from an unchecked
    # NPY shape. Object/pickle dtypes have already been rejected above.
    words = np.frombuffer(data, dtype=dtype, count=count, offset=offset)
    if dtype.kind == "i" and np.any(words < 0):
        raise ValueError("NPY frame words must be nonnegative")
    return words.astype(np.uint64, copy=True)


def load_frames(
    path: str | Path,
    format: FrameFormat,
    *,
    max_bytes: int = _DEFAULT_BYTES,
) -> FrameArray:
    """Read a bounded one-dimensional integer frame file.

    ``format`` must be explicit: ``raw-u64-le`` reads little-endian uint64
    bytes, while ``npy`` accepts NPY 1.0/2.0 integer arrays up to 64 bits.
    The result is copied into a ``np.uint64`` array; no simulator state changes.
    """
    limit = _integer(max_bytes, "max_bytes")
    return _decode_frames(_read_bytes(Path(path), limit), format)


def _object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be an Any with string keys")
    return cast(dict[str, Any], value)


def _chip(value: Any, name: str) -> ChipCoord:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{name} must be [x, y]")
    return ChipCoord(_integer(value[0], f"{name} x"), _integer(value[1], f"{name} y"))


def _port(value: Any, name: str) -> Port:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError(f"{name} must be [chip_x, chip_y, core_x, core_y]")
    return Port(
        ChipCoord(
            _integer(value[0], f"{name} chip x"),
            _integer(value[1], f"{name} chip y"),
        ),
        CoreCoord(
            _integer(value[2], f"{name} core x"),
            _integer(value[3], f"{name} core y"),
        ),
    )


def _board(value: Any) -> TargetBoard:
    if value == "single":
        return SingleBoard()
    if value == "array2x2":
        return Array2x2Board()
    document = _object(value, "board")
    if set(document) != {"chips", "links", "ingress", "outputs"}:
        raise ValueError("custom board needs chips/links/ingress/outputs")
    chips, links, ingress, outputs = (
        document[name] for name in ("chips", "links", "ingress", "outputs")
    )
    if not all(isinstance(items, list) for items in (chips, links, ingress, outputs)):
        raise ValueError("custom board collections must be arrays")
    chips, links, ingress, outputs = cast(
        tuple[list[Any], list[Any], list[Any], list[Any]],
        (chips, links, ingress, outputs),
    )
    edges: list[tuple[ChipCoord, ChipCoord]] = []
    for i, pair in enumerate(links):
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError(f"board link {i} must be two chip coordinates")
        edges.append((_chip(pair[0], f"link {i} a"), _chip(pair[1], f"link {i} b")))
    board = CustomBoard(
        [_chip(item, f"chip {i}") for i, item in enumerate(chips)],
        edges,
        [_port(item, f"ingress {i}") for i, item in enumerate(ingress)],
        [_port(item, f"output {i}") for i, item in enumerate(outputs)],
    )
    return board


def board_document(spec: BoardSnapshot) -> str | dict[str, Any]:
    """Convert a validated board snapshot to a manifest-safe description.

    Presets return ``"single"`` or ``"array2x2"``.  Custom snapshots return a
    JSON-compatible Any containing explicit chips, links, ingress ports, and
    output ports; runtime core/thread state is intentionally excluded.
    """
    if spec == snapshot_board(SingleBoard().declaration):
        return "single"
    if spec == snapshot_board(Array2x2Board().declaration):
        return "array2x2"

    def chip(coord: ChipCoord) -> list[int]:
        return [coord.x, coord.y]

    def port(item: Port) -> list[int]:
        return [item.chip.x, item.chip.y, item.core.x, item.core.y]

    return {
        "chips": [chip(coord) for coord in spec.chips],
        "links": [[chip(a), chip(b)] for a, b in spec.links],
        "ingress": [port(item) for item in spec.ingress],
        "outputs": [port(item) for item in spec.outputs],
    }


def _events(
    document: Any, directory: Path, limit: int
) -> tuple[TargetBoard, list[_Event]]:
    manifest = _object(document, "manifest")
    if set(manifest) != {"version", "target", "board", "events"}:
        raise ValueError("manifest fields must be version/target/board/events")
    if _integer(manifest["version"], "manifest version") != 3:
        raise ValueError("unsupported manifest version")
    if manifest["target"] != "paicore-2.5":
        raise ValueError("unsupported manifest target")
    board = _board(manifest["board"])
    events = manifest["events"]
    if not isinstance(events, list) or len(events) > limit:
        raise ValueError(f"events must be a list with at most {limit} entries")
    result = []
    for index, raw_event in enumerate(events):
        event = _object(raw_event, f"event {index}")
        required = {
            "path",
            "format",
            "sha256",
            "ingressChip",
            "ingressCore",
            "threadId",
        }
        if not required <= event.keys() or event.keys() - required - {"start", "stop"}:
            raise ValueError(f"event {index} has missing or unknown fields")
        relative = event["path"]
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
        ):
            raise ValueError("event path must be a nonempty relative path")
        if ".." in Path(relative).parts:
            raise ValueError("event path must not contain parent traversal")
        path = (directory / relative).resolve()
        if not path.is_relative_to(directory):
            raise ValueError("event path or symlink escapes the manifest directory")
        format = event["format"]
        if format not in ("npy", "raw-u64-le"):
            raise ValueError("event format must be npy or raw-u64-le")
        # Explicit branches narrow the external Any to the public Literal.
        selected: FrameFormat = "npy" if format == "npy" else "raw-u64-le"
        digest = event["sha256"]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
        ):
            raise ValueError("event sha256 must be 64 lowercase hexadecimal digits")
        chip_raw, core_raw = event["ingressChip"], event["ingressCore"]
        if not isinstance(chip_raw, list) or len(chip_raw) != 2:
            raise ValueError("event ingressChip must be [x, y]")
        if not isinstance(core_raw, list) or len(core_raw) != 2:
            raise ValueError("event ingressCore must be [x, y]")
        chip = ChipCoord(
            _integer(chip_raw[0], "ingress chip x"),
            _integer(chip_raw[1], "ingress chip y"),
        )
        core = CoreCoord(
            _integer(core_raw[0], "ingress core x"),
            _integer(core_raw[1], "ingress core y"),
        )
        ingress_port = Port(chip, core)
        thread = ThreadId(_integer(event["threadId"], "event threadId"))
        start = _integer(event.get("start", 0), "event start")
        stop_raw = event.get("stop")
        stop = None if stop_raw is None else _integer(stop_raw, "event stop", start)
        result.append(_Event(path, selected, digest, ingress_port, thread, start, stop))
    return board, result


def _unique_fields(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate manifest field: {key}")
        result[key] = value
    return result


def replay_manifest(
    path: str | Path,
    *,
    max_bytes: int = _DEFAULT_BYTES,
    max_events: int = 4096,
    max_run_events: int = 1_000_000,
) -> Simulator:
    """Replay a bounded manifest into a new cold simulator.

    ``max_bytes`` limits cumulative referenced bytes, counting repeated files
    again. ``max_run_events`` limits input transactions and thread steps. Exhaustion
    raises ReplayError with the paused
    simulator, not a fabricated COMPLETE. Format/manifest errors before any
    execution raise ValueError; execution/read failures preserve their cause.
    The returned simulator retains raw response words for ``drain_output`` and
    exposes the accepted state prefix through ``snapshot``.  No tensor mapping,
    compiler invocation, or application-level post-processing is inferred.
    """
    byte_budget = _integer(max_bytes, "max_bytes")
    event_limit = _integer(max_events, "max_events")
    run_budget = _integer(max_run_events, "max_run_events")
    manifest_path = Path(path).resolve()
    try:
        document = json.loads(
            _read_bytes(manifest_path, _MANIFEST_BYTES),
            object_pairs_hook=_unique_fields,
        )
    except RecursionError as exc:
        raise ValueError("manifest nesting exceeds parser limit") from exc
    board, events = _events(document, manifest_path.parent, event_limit)
    simulator = Simulator(board)
    if any(event.ingress not in simulator.board.ingress for event in events):
        raise ValueError("event ingress is not a declared board CPU port")
    try:
        for event in events:
            data = _read_bytes(event.path, byte_budget)
            byte_budget -= len(data)
            if hashlib.sha256(data).hexdigest() != event.sha256:
                raise ValueError(f"SHA-256 mismatch: {event.path.name}")
            frames = _decode_frames(data, event.format)
            stop = frames.size if event.stop is None else event.stop
            if not 0 <= event.start <= stop <= frames.size:
                raise ValueError(f"event frame range is outside {event.path.name}")
            # A bounded feed chunk also respects the engine's independent queue
            # budget; cutting a CONFIG payload is legal and does not add events.
            for start in range(event.start, stop, 65_536):
                simulator.feed(
                    frames[start : min(start + 65_536, stop)], ingress=event.ingress
                )
            used = simulator.run(run_budget)
            run_budget -= used
            if run_budget == 0 and (simulator.pending_words or simulator.pending_steps):
                raise ValueError(
                    "replay event budget exhausted; pending work is retained"
                )
        simulator.finish_input()
    except (ValueError, TypeError, RuntimeError, OSError) as exc:
        raise ReplayError(f"replay failed: {exc}", simulator) from exc
    return simulator
