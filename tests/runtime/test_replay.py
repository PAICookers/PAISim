import hashlib
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.lib import format as npy_format

from paisim import (
    ChipCoord,
    CoreAddr,
    CoreCoord,
    CustomBoard,
    Port,
    Simulator,
    SingleBoard,
)
from paisim.runtime.replay import (
    ReplayError,
    board_document,
    load_frames,
    replay_manifest,
)
from tests.protocol.fixtures import control, offline_config, work

C0 = ChipCoord(0, 0)
CPU = Port(C0)
ROOT = CoreAddr(C0, CoreCoord(0, 2))


def _frames(directory: Path, name: str, words: list[int]) -> dict[str, Any]:
    path = directory / name
    array = np.asarray(words, np.uint64)
    if name.endswith(".npy"):
        np.save(path, array, allow_pickle=False)
        kind = "npy"
    else:
        path.write_bytes(array.astype("<u8").tobytes())
        kind = "raw-u64-le"
    return {
        "path": name,
        "format": kind,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "ingressChip": [0, 0],
        "ingressCore": [0, 0],
        "threadId": 0,
    }


def _manifest(
    directory: Path, events: list[dict[str, Any]], board: object = "single"
) -> Path:
    path = directory / "replay.json"
    path.write_text(
        json.dumps(
            {
                "version": 3,
                "target": "paicore-2.5",
                "board": board,
                "events": events,
            }
        )
    )
    return path


def test_raw_and_npy_readers_return_owned_uint64(tmp_path: Path) -> None:
    words = [0, 1, (1 << 64) - 1]
    for name in ("frames.raw", "frames.npy"):
        event = _frames(tmp_path, name, words)
        data = load_frames(tmp_path / name, event["format"])
        assert data.dtype == np.uint64 and data.tolist() == words and data.flags.owndata


def test_segmented_replay_matches_direct_execution(tmp_path: Path) -> None:
    words = [
        *offline_config(CPU, ROOT, threshold=100),
        work(CPU, ROOT, 3),
        control(CPU, ROOT, 12, 1),
    ]
    direct = Simulator(SingleBoard())
    direct.feed(np.asarray(words, dtype=np.uint64), ingress=CPU)
    direct.run()
    expected = direct.drain_output()
    raw = _frames(tmp_path, "a.raw", words[:5])
    npy = _frames(tmp_path, "b.npy", words[5:])
    replayed = replay_manifest(_manifest(tmp_path, [raw, npy]))
    np.testing.assert_array_equal(replayed.drain_output(), expected)
    assert replayed.array.chip(C0).core(ROOT.core).neurons[0].voltage == 3


def test_manifest_requires_board_ingress_chip_and_thread(tmp_path: Path) -> None:
    event = _frames(tmp_path, "frames.raw", [])
    path = _manifest(tmp_path, [event], board="array2x2")
    assert len(replay_manifest(path).array.chip_coords) == 4
    for field in ("ingressChip", "ingressCore", "threadId"):
        bad = dict(event)
        bad.pop(field)
        with pytest.raises(ValueError, match="missing or unknown"):
            replay_manifest(_manifest(tmp_path, [bad]))
    document = json.loads(path.read_text())
    document["version"] = 1
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="version"):
        replay_manifest(path)


def test_replay_hash_path_and_run_budget_are_bounded(tmp_path: Path) -> None:
    words = [*offline_config(CPU, ROOT), control(CPU, ROOT, 12, 3)]
    event = _frames(tmp_path, "frames.raw", words)
    with pytest.raises(ReplayError, match="event budget") as caught:
        replay_manifest(_manifest(tmp_path, [event]), max_run_events=7)
    assert caught.value.simulator.pending_words or caught.value.simulator.pending_steps
    with pytest.raises(ReplayError, match="SHA-256"):
        replay_manifest(_manifest(tmp_path, [{**event, "sha256": "0" * 64}]))
    with pytest.raises(ValueError, match="parent traversal"):
        replay_manifest(_manifest(tmp_path, [{**event, "path": "../escape.raw"}]))


@pytest.mark.parametrize("dtype", [">u8", ">i8", "<u2", "i1"])
def test_positive_npy_signed_and_endian_values_preserved(
    tmp_path: Path, dtype: str
) -> None:
    path = tmp_path / "words.npy"
    np.save(path, np.asarray([0, 127], dtype=dtype), allow_pickle=False)
    assert load_frames(path, "npy").tolist() == [0, 127]


@pytest.mark.parametrize(
    "array",
    [
        np.asarray([-1], dtype=np.int64),
        np.asarray([1.5]),
        np.asarray([[1]], dtype=np.uint64),
        np.asarray([True]),
        np.asarray([{"untrusted": "object"}], dtype=object),
    ],
    ids=["negative", "float", "rank", "boolean", "pickle"],
)
def test_npy_rejects_untrusted_shapes_and_values(
    tmp_path: Path, array: np.ndarray
) -> None:
    path = tmp_path / "bad.npy"
    np.save(path, array)
    with pytest.raises(ValueError):
        load_frames(path, "npy")


def test_replay_file_limits_and_payload_length_checked_before_allocation(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "bad.raw"
    raw.write_bytes(b"123456789")
    with pytest.raises(ValueError, match="byte budget"):
        load_frames(raw, "raw-u64-le", max_bytes=8)
    with pytest.raises(ValueError, match="divisible"):
        load_frames(raw, "raw-u64-le")
    buffer = io.BytesIO()
    npy_format.write_array_header_1_0(
        buffer, {"shape": (1 << 50,), "fortran_order": False, "descr": "<u8"}
    )
    path = tmp_path / "huge.npy"
    path.write_bytes(buffer.getvalue())
    with pytest.raises(ValueError, match="payload length"):
        load_frames(path, "npy")
    np.save(path, np.asarray([1], np.uint64), allow_pickle=False)
    path.write_bytes(path.read_bytes() + b"garbage")
    with pytest.raises(ValueError, match="payload length"):
        load_frames(path, "npy")


@pytest.mark.parametrize(
    "changes",
    [
        {"path": "/absolute.raw"},
        {"ingressChip": [True, 0]},
        {"ingressCore": [1, 1]},
        {"format": "autodetect"},
        {"sha256": "xyz"},
        {"start": -1},
        {"start": 2, "stop": 1},
        {"extra": 0},
    ],
    ids=[
        "absolute-path",
        "bool-chip",
        "undeclared-CPU",
        "format",
        "hash",
        "negative-start",
        "reverse-range",
        "extra",
    ],
)
def test_invalid_manifest_event_is_rejected_before_replay(
    tmp_path: Path, changes: dict[str, Any]
) -> None:
    event = _frames(tmp_path, "a.raw", [0])
    with pytest.raises(ValueError):
        replay_manifest(_manifest(tmp_path, [{**event, **changes}]))


def test_manifest_duplicate_keys_and_symlink_escape_are_rejected(
    tmp_path: Path,
) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"version":3,"version":3}')
    with pytest.raises(ValueError, match="duplicate"):
        replay_manifest(path)
    outside = tmp_path.parent / f"{tmp_path.name}-outside.raw"
    outside.write_bytes(b"\0" * 8)
    (tmp_path / "escape.raw").symlink_to(outside)
    event = _frames(tmp_path, "a.raw", [0])
    with pytest.raises(ValueError, match="escapes"):
        replay_manifest(_manifest(tmp_path, [{**event, "path": "escape.raw"}]))


def test_custom_board_manifest_round_trip_is_self_contained(tmp_path: Path) -> None:
    remote = CoreAddr(ChipCoord(1, 0), CoreCoord(0, 2))
    board = CustomBoard(
        [C0, ChipCoord(1, 0), ChipCoord(2, 0)],
        [(C0, ChipCoord(1, 0))],
        [CPU],
        [CPU, Port(ChipCoord(1, 0))],
    )
    direct = Simulator(board)
    words = [
        *offline_config(CPU, remote, threshold=100),
        work(CPU, remote, 3),
        control(CPU, remote, 12, 1),
    ]
    direct.feed(np.asarray(words, dtype=np.uint64), ingress=CPU)
    direct.run()
    event = _frames(tmp_path, "a.raw", words)
    document = board_document(direct.board)
    assert isinstance(document, dict) and document["links"] == [[[0, 0], [1, 0]]]
    replayed = replay_manifest(_manifest(tmp_path, [event], board=document))
    np.testing.assert_array_equal(replayed.drain_output(), direct.drain_output())
    assert replayed.board == direct.board
    board.links.append((ChipCoord(1, 0), ChipCoord(2, 0)))
    assert len(replayed.array.links) == len(direct.array.links) == 1


@pytest.mark.parametrize(
    "invalid",
    [
        {
            "chips": [[0, 0]],
            "links": [[[0, 0], [1, 0]]],
            "ingress": [[0, 0, 0, 0]],
            "outputs": [[0, 0, 0, 0]],
        },
        {
            "chips": [[0, 0]],
            "links": [],
            "ingress": [[1, 0, 0, 0]],
            "outputs": [[0, 0, 0, 0]],
        },
        {"chips": [[0, 0]], "links": [], "ingress": [[0, 0, 0, 0]], "outputs": []},
    ],
    ids=["missing-neighbor", "foreign-CPU", "empty-output"],
)
def test_invalid_custom_board_rejected_before_event_feed(
    tmp_path: Path, invalid: dict[str, Any]
) -> None:
    event = _frames(tmp_path, "a.raw", [0])
    with pytest.raises(ValueError):
        replay_manifest(_manifest(tmp_path, [event], board=invalid))
