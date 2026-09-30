"""Stage 1 storage ownership and detached observation checks."""

import numpy as np
import pytest
from paicorelib.neuron_defs_v2 import OutputType

from paisim import ChipCoord, CoreAddr, CoreCoord, Port
from paisim.engine.core import OfflineCore
from paisim.engine.model import FoldLayout, LutEntry, NeuronRecord, WeightRef
from tests.protocol.fixtures import offline_config


def _core() -> OfflineCore:
    return OfflineCore(CoreAddr(ChipCoord(0, 0), CoreCoord(1, 2)))


def _configured_core() -> OfflineCore:
    core = _core()
    core.configure(np.zeros(3, dtype=np.uint64))
    return core


def _configured_neuron_core() -> OfflineCore:
    core = _core()
    words = offline_config(Port(ChipCoord(0, 0)), core.addr)
    core.configure(np.asarray(words[1:4], dtype=np.uint64))
    core.write(2, 0, np.asarray(words[5:9], dtype=np.uint64))
    return core


@pytest.fixture
def configured_neuron_core() -> OfflineCore:
    return _configured_neuron_core()


def test_input_sram_has_hardware_shape_and_detached_read_view() -> None:
    core = _core()
    assert core.input_sram.shape == (256, 64)
    assert core.input_sram.dtype == np.uint8
    before = core.input_sram.copy()
    core.input_sram[0, 0] = 0xA5
    assert before[0, 0] == 0
    assert core.input_sram[0, 0] == 0xA5


def test_weights_and_lut_expose_read_only_raw_arrays() -> None:
    core = _core()
    weights = core.weights.array()
    lut = core.lut.array()
    assert weights.dtype == np.uint64
    assert weights.shape == (4096, 2)
    assert lut.dtype == np.uint64
    assert lut.shape == (256,)
    assert not weights.flags.writeable
    assert not lut.flags.writeable


def test_mutation_increments_revision_without_aliasing_public_arrays() -> None:
    core = _core()
    revision = core.revision
    raw_weights = core.weights.array().copy()
    core.weights[3] = (7, 11)
    assert core.revision == revision + 1
    assert core.weights[3] == (7, 11)
    assert np.array_equal(raw_weights[3], np.zeros(2, dtype=np.uint64))

    revision = core.revision
    raw_lut = core.lut.array().copy()
    core.lut[4] = (-9, 13)
    assert core.revision == revision + 1
    assert core.lut[4] == (-9, 13)
    assert raw_lut[4] == 0


def test_snapshot_is_detached_and_read_only() -> None:
    core = _configured_core()
    snapshot = core.snapshot()
    assert snapshot.registers.dtype == np.uint64
    assert snapshot.neuron_sram.dtype == np.uint64
    assert snapshot.input_sram.dtype == np.uint8
    assert snapshot.lut_sram.dtype == np.uint64
    for array in (
        snapshot.registers,
        snapshot.neuron_sram,
        snapshot.input_sram,
        snapshot.lut_sram,
    ):
        assert not array.flags.writeable
    core.input_sram[0, 0] = 0x5A
    assert snapshot.input_sram[0, 0] == 0


def test_snapshot_selection_avoids_unrequested_copies() -> None:
    snapshot = _configured_core().snapshot({"config", "lut"})
    assert snapshot.config is not None
    assert snapshot.registers is not None
    assert snapshot.lut is not None
    assert snapshot.input_sram is None
    assert snapshot.neurons is None
    assert snapshot.weights is None
    assert snapshot.selected == frozenset({"config", "lut"})


def test_snapshot_records_preserve_decoded_fields_and_revision(
    configured_neuron_core: OfflineCore,
) -> None:
    core = configured_neuron_core
    snapshot = core.snapshot()
    assert snapshot.revision == core.revision
    assert snapshot.neurons == (NeuronRecord(0, 0, 1, 0, 0, 0, 0, OutputType.VALUE),)
    assert snapshot.weights == (WeightRef(0, (0,), (), (0,)),)
    assert snapshot.lut is not None and len(snapshot.lut) == 256
    assert snapshot.lut[0] == LutEntry(0, 0, 0, 0)
    assert snapshot.config is not None
    assert snapshot.config["neuron_number"] == 2
    assert snapshot.neuron_sram is not None
    assert snapshot.neuron_sram.dtype == np.uint64


def test_fold_layout_is_selectively_observable() -> None:
    core = _core()
    neuron_sram = np.zeros((4096, 2), dtype=np.uint64)
    neuron_sram[0] = [1 << 32 | 1 << 33 | 10 << 47 | 10 << 35, 0]
    neuron_sram[1, 0] = 10 << 44
    neuron_sram[2] = [1 << 62 | 2 << 29 | 1, 1 << 53 | 1 << 42 | 1 << 31]
    neuron_sram[3, 0] = 1
    words = offline_config(Port(ChipCoord(0, 0)), core.addr)
    registers = np.asarray(words[1:4], dtype=np.uint64)
    registers[0] = np.uint64((int(registers[0]) & ~(0xFFF << 14)) | (4 << 14))
    core.configure(registers)
    core.write(2, 0, neuron_sram[:4].reshape(-1))
    snapshot = core.snapshot({"folds"})
    assert snapshot.folds == (FoldLayout(1, (1, 1, 1), (1, 0, 0), (0,)),)
    assert snapshot.neuron_sram is None


def test_static_mutations_invalidate_decoded_state(
    configured_neuron_core: OfflineCore,
) -> None:
    core = configured_neuron_core
    old_decoded = core.decoded
    core.neurons[0].set_parameter("thres_pos", 2)
    assert core.decoded is not old_decoded

    old_decoded = core.decoded
    core.lut[0] = (3, 4)
    assert core.decoded is not old_decoded


def test_static_mutation_is_atomic_after_execution_starts() -> None:
    core = _configured_core()
    core.step(0)
    before_weight = core.weights[0]
    before_lut = core.lut[0]
    with pytest.raises(RuntimeError):
        core.weights[0] = (1, 0)
    with pytest.raises(RuntimeError):
        core.lut[0] = (1, 2)
    assert core.weights[0] == before_weight
    assert core.lut[0] == before_lut
