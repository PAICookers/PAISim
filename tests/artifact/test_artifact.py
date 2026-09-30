"""Tensor mapping expectations constructed without compiler/runtime helpers."""

import numpy as np
import pytest

from paisim.frames import work_word
from tests.artifact.fixtures import document, load


def test_word_order_signed_input_slot_carry_and_zero_suppression(tmp_path):
    artifact = load(tmp_path)
    assert artifact.config_frames.tolist() == [0x0123456789ABCDEF]
    assert artifact.encode_inputs({"x": np.array([[-3, 0]])}, 64).tolist() == [
        0x9000000000000000 | 2 << 42 | 1 << 17 | 4 << 8 | 13,
    ]
    doc = document()
    doc["configFrames"]["wordOrder"] = "LOW_FIRST"
    assert load(tmp_path, doc).config_frames.tolist() == [0x89ABCDEF01234567]


def test_data_mapping_preserves_external_unaligned_identifiers(tmp_path):
    artifact = load(tmp_path)
    words = np.array(
        [0x800000000000010F, 0x8000000000020007, 0xE000000000000000], np.uint64
    )
    decoded = artifact.decode_outputs(words, 2)["y"]
    assert decoded.tolist() == [[[0, -1]], [[7, 0]]]
    with pytest.raises(ValueError, match="unmapped"):
        artifact.decode_outputs(np.array([0x8000000000000201], np.uint64), 2)
    with pytest.raises(NotImplementedError, match="wrap"):
        artifact.decode_outputs(words, 257)


def test_potential_lane_assembly_and_incomplete_rejection(tmp_path):
    doc = document()
    item = doc["ioMapping"]["threads"][0]["outputMappings"]["items"][0]
    item.update(kind="VOLTAGE", bitWidth=32, shape={"size": [1]})
    item["entries"] = [{"elemIdx": 0, "axonBitIdx": 0}]
    artifact = load(tmp_path, doc)
    words = np.array(
        [
            work_word(0, 0, lane * 8, value, voltage=True)
            for lane, value in enumerate([0x04, 0x03, 0x02, 0xFF])
        ],
        np.uint64,
    )
    assert artifact.decode_outputs(words, 1)["y"].tolist() == [[-16645372]]
    with pytest.raises(ValueError, match="incomplete"):
        artifact.decode_outputs(words[:-1], 1)


def test_potential_mapping_accepts_producer_base_addresses(tmp_path):
    doc = document()
    item = doc["ioMapping"]["threads"][0]["outputMappings"]["items"][0]
    item.update(kind="VOLTAGE", bitWidth=32, shape={"size": [9]})
    bases = [0, 1, 2, 3, 4, 5, 6, 7, 32]
    item["entries"] = [
        {"elemIdx": index, "axonBitIdx": base} for index, base in enumerate(bases)
    ]
    artifact = load(tmp_path, doc)
    words = np.array(
        [
            work_word(0, 0, base + lane * 8, 0x11 + lane * 0x11, voltage=True)
            for base in bases
            for lane in range(4)
        ],
        np.uint64,
    )
    decoded = artifact.decode_outputs(words, 1)["y"]
    assert decoded.shape == (1, 9)
    assert np.all(decoded[0] == 0x44332211)


def test_input_validation_and_unsupported_schema(tmp_path):
    artifact = load(tmp_path)
    with pytest.raises(ValueError, match="integer dtype"):
        artifact.encode_inputs({"x": np.array([[1.1, 0.0]])}, 0)
    with pytest.raises(ValueError, match="range"):
        artifact.encode_inputs({"x": np.array([[8, 0]])}, 0)
    with pytest.raises(ValueError, match="names"):
        artifact.encode_inputs({"wrong": np.zeros((1, 2), int)}, 0)
    doc = document()
    doc["schemaVersion"] = 2
    with pytest.raises(NotImplementedError, match="schema"):
        load(tmp_path, doc)


@pytest.mark.parametrize(
    "change", ["unknown-field", "unknown-enum"], ids=["unknown-field", "unknown-enum"]
)
def test_json_uses_protobuf_descriptor_validation(tmp_path, change):
    doc = document()
    if change == "unknown-field":
        doc["unexpected"] = 1
    else:
        doc["ioMapping"]["threads"][0]["runtime"]["decodeMode"] = "UNKNOWN"
    with pytest.raises(ValueError, match="malformed json artifact"):
        load(tmp_path, doc)
