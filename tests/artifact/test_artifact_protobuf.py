"""Small Protobuf boundary checks; JSON remains the readable fixture format."""

import pytest
from google.protobuf.json_format import ParseDict

from paisim import ChipCoord, CoreAddr, CoreCoord, Port, Simulator, SingleBoard
from paisim._generated.compile_artifacts_pb2 import (
    CompileArtifacts,
    RuntimeParams,
    ThreadIOMapping,
)
from paisim.artifact.io import read_artifact
from paisim.engine.core import OfflineCore
from tests.artifact.fixtures import document
from tests.protocol.fixtures import offline_config


def test_protobuf_artifact_preserves_config_words(tmp_path) -> None:
    message = ParseDict(document(), CompileArtifacts())
    path = tmp_path / "config.pb"
    path.write_bytes(message.SerializeToString())
    frames, threads = read_artifact(path)
    assert frames.tolist() == [0x0123456789ABCDEF]
    assert isinstance(threads[0], ThreadIOMapping)
    assert threads[0].runtime.decode_mode == RuntimeParams.STREAM


def test_simulator_from_protobuf_configures_and_initializes_detached_model(
    tmp_path,
) -> None:
    cpu = Port(ChipCoord(0, 0))
    root = CoreAddr(ChipCoord(0, 0), CoreCoord(0, 2))
    words64 = offline_config(cpu, root)
    words32 = [part for word in words64 for part in (word >> 32, word & 0xFFFF_FFFF)]
    doc = {
        "schemaVersion": 1,
        "configFrames": {"wordOrder": "HIGH_FIRST", "words": words32},
        "ioMapping": {
            "threads": [
                {
                    "threadId": 0,
                    "rootCoreOffset": {"y": 2},
                    "runtime": {"decodeMode": "STREAM", "timesteps": 1, "syncSteps": 1},
                    "inputMappings": {"items": []},
                    "outputMappings": {"targetLcn": 0, "items": []},
                    "coreTicks": [],
                }
            ]
        },
    }
    message = ParseDict(doc, CompileArtifacts())
    path = tmp_path / "model.pb"
    path.write_bytes(message.SerializeToString())

    sim = Simulator.from_artifact(path, SingleBoard())

    assert sim.artifact is not None
    assert sim.artifact.config_frames.tolist() == words64
    core = sim.array.chip(ChipCoord(0, 0)).core(CoreCoord(0, 2))
    assert isinstance(core, OfflineCore)
    static = core.snapshot({"config", "neurons", "weights", "lut"})
    assert static.config == core.config
    assert (
        static.neurons is not None and len(static.neurons) == core.decoded.voltage.size
    )
    assert static.weights is not None and len(static.weights) == len(
        core.decoded.synapses
    )
    assert static.lut is not None and len(static.lut) == 256
    assert sim.scheduler.threads[0].done
    view = sim.artifact.config_frames
    with pytest.raises(ValueError):
        view[0] = 0
    copied = sim.artifact.threads[0]
    copied.thread_id = 9
    assert sim.artifact.threads[0].thread_id == 0

    cold = Simulator.from_artifact(path, SingleBoard(), auto_init=False)
    assert len(cold.scheduler.threads) == 1
    assert cold.scheduler.threads[0].tick == 0
    cold.initialize()
    assert cold.scheduler.threads[0].done

    path.write_bytes(b"changed after construction")
    assert sim.artifact.config_frames.tolist() == words64
    assert sim.scheduler.threads[0].done
    with pytest.raises(ValueError):
        Simulator.from_artifact(path, SingleBoard())
