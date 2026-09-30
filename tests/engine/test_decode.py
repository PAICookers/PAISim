"""Literal SRAM fixtures independent of the frame encoder."""

import numpy as np
import pytest
from paicorelib.core_defs_v2 import (
    AddPotentialMode,
    CSCAccelerateMode,
    DataSign,
    DataWidth,
    PoolingMode,
    SNNMode,
    ZeroOutputMode,
)

from paisim.engine.decode import decode_core
from paisim.engine.model import core_config


def storage():
    return np.zeros((4096, 2), np.uint64), np.zeros(256, np.uint64)


def test_full_half_signed_parameters_and_untruncated_dense():
    ram, lut = storage()
    ram[0] = [
        1 << 32 | 8 << 47 | 27 << 35 | 0xFFFFFFFC,
        3 << 56 | 8 << 47 | 35 << 41 | 2 << 35,
    ]
    ram[1] = [
        17 << 44 | 63 << 33 | 0xFFFFD << 13 | 0xFFF,
        1 << 62 | 0xFFFE << 46 | 1 << 45 | 0xFFFFFFF6 << 12,
    ]
    ram[2] = [8 << 47 | 27 << 35 | 10, 4 << 47]
    ram[8:28] = 0xFEFEFEFEFEFEFEFE
    lut[0] = 0xFFFFFFFD << 8 | 255
    core = decode_core(
        dict(
            neuron_number=3,
            input_width=3,
            weight_width=3,
            weight_sign=1,
            output_sign=1,
            lcn=3,
        ),
        ram,
        lut,
    )
    assert core.voltage.tolist() == [-4, 10]
    for key, expected in dict(
        reset_v=-2, thres_neg=-10, thres_pos=17, leak_tau=-1, leak_v=-3, init_v=-1
    ).items():
        assert core.parameters[key].tolist() == [expected, expected]
    assert core.routes[0].tolist() == [-3, 2, 0, 0, 0, 0]
    assert core.voltage_locations.tolist() == [[0, 0], [2, 0]]
    assert len(core.synapses) == 1
    assert core.synapses[0].posts.tolist() == [0, 1]
    assert core.synapses[0].weights.tolist() == [-2] * 320
    assert core.synapses[0].inputs.tolist() == list(range(320))
    assert core.lut_thresholds[0] == -3 and core.lut_values[0] == -1
    ram[:] = 0
    assert core.voltage[0] == -4


def test_fold_transition_strides_voltage_lanes_axon_carry():
    ram, lut = storage()
    ram[0] = [1 << 32 | 1 << 33 | 10 << 47 | 10 << 35 | 9, 510 << 47 | 2 << 56]
    ram[1, 0] = 10 << 44
    ram[2] = [
        1 << 62 | 20 << 51 | 4 << 40 | 2 << 29 | 8,
        2 << 53 | 2 << 42 | 2 << 31 | 10 << 20 | 4 << 9,
    ]
    ram[3] = [1 | 2 << 32, 3 | 4 << 32]
    ram[4] = [5 | 6 << 32, 7]
    ram[10, 0] = 1
    core = decode_core(dict(neuron_number=5), ram, lut)
    assert core.voltage.tolist() == [9, 1, 2, 3, 4, 5, 6, 7]
    assert core.synapses[0].offsets.tolist() == [0, 1, 5, 6, 16, 17, 21, 22]
    assert core.axons.tolist() == [510, 0, 4, 6, 26, 28, 32, 34]
    assert core.ticks.tolist() == [2, 3, 3, 3, 3, 3, 3, 3]
    assert core.voltage_locations.tolist() == [
        [0, 0],
        [3, 0],
        [3, 1],
        [3, 2],
        [3, 3],
        [4, 0],
        [4, 1],
        [4, 2],
    ]
    assert core.folds[0].count == 8
    assert core.folds[0].ranges == (2, 2, 2)
    assert core.folds[0].skews == (1, 4, 10)
    assert core.folds[0].offsets == (0, 1, 5, 6, 16, 17, 21, 22)


@pytest.mark.parametrize(
    "width,count,shift", [(1, 7, 16), (2, 7, 16), (4, 6, 32), (8, 5, 48)]
)
def test_csc_widths_padding_and_bit_indices(width, count, shift):
    ram, lut = storage()
    ram[0] = [1 << 32 | 10 << 47 | 10 << 35 | 8 << 59, 0]
    ram[1, 0] = 1 << 12
    values = [1] * (count - 1) + [0]
    indices = [8 * (i + 2) for i in range(count - 1)] + [65535]
    lo = sum(v << (i * width) for i, v in enumerate(values))
    lo |= sum(indices[i] << (shift + i * 16) for i in range(count - 4))
    hi = sum(v << (i * 16) for i, v in enumerate(indices[count - 4 :]))
    ram[10] = [lo, hi]
    core = decode_core(
        dict(neuron_number=2, input_width=3, weight_width=width.bit_length() - 1),
        ram,
        lut,
    )
    group = core.synapses[0]
    assert group.inputs.tolist() == list(range(2, count + 1))
    assert group.weights.tolist() == [1] * (count - 1)
    assert group.offsets.tolist() == [1]


def test_direct_add_full_masks_and_partial_mask_rejection():
    ram, lut = storage()
    ram[0] = [1 << 32 | 10 << 47 | 10 << 35, 1]
    ram[10] = [0xFFFFFFFF, 0xFFFFFFFF]
    core = decode_core(dict(neuron_number=2, add_potential=1), ram, lut)
    assert core.synapses[0].inputs.tolist() == [0, 2]
    assert core.synapses[0].offsets.tolist() == [1]
    ram[10, 0] = 1
    with pytest.raises(ValueError, match="complete 32-bit"):
        decode_core(dict(neuron_number=2, add_potential=1), ram, lut)


def test_record_boundary_errors():
    ram, lut = storage()
    with pytest.raises(ValueError, match="preceding full"):
        decode_core(dict(neuron_number=1), ram, lut)
    ram[0, 0] = 1 << 32
    with pytest.raises(ValueError, match="Part2"):
        decode_core(dict(neuron_number=1), ram, lut)
    ram[0, 0] = 1 << 32 | 1 << 33
    with pytest.raises(ValueError, match="fold attributes"):
        decode_core(dict(neuron_number=2), ram, lut)


def test_rejected_configuration_does_not_mutate_input_sram_or_decoded_storage():
    """Malformed decode input is rejected before caller-owned arrays change."""
    neuron_sram, lut_sram = storage()
    neuron_sram[0, 0] = 1 << 32 | 1 << 33
    before_neurons = neuron_sram.copy()
    before_lut = lut_sram.copy()
    with pytest.raises(ValueError, match="preceding full|Part2|fold attributes"):
        decode_core({"neuron_number": 1}, neuron_sram, lut_sram)
    np.testing.assert_array_equal(neuron_sram, before_neurons)
    np.testing.assert_array_equal(lut_sram, before_lut)


def test_empty_core():
    core = decode_core({}, *storage())
    assert core.config["neuron_number"] == 0
    assert core.config["input_width"] == 0
    assert core.routes.shape == (0, 6)
    assert core.voltage_locations.shape == (0, 2)
    assert core.synapses == ()


def test_folded_half_inherits_full_parameters_and_csc_acceleration_alias():
    ram, lut = storage()
    # A full neuron followed by a folded half neuron shares CSC parameters.
    ram[0] = [1 << 32 | 10 << 47 | 10 << 35, 0]
    ram[1] = [123 << 44 | 1 << 12 | 10, 0]
    ram[2] = [1 << 33 | 10 << 47 | 10 << 35 | 5, 0]
    ram[3] = [1 << 62 | 1 << 29 | 2, 1 << 53 | 1 << 42 | 2 << 31]
    ram[4, 0] = 0xFFFFFFFE
    ram[10, 0] = 1
    core = decode_core(dict(neuron_number=5, csc_accelerate=1), ram, lut)
    assert core.voltage.tolist() == [0, 5, -2]
    assert core.parameters["thres_pos"].tolist() == [123, 123, 123]
    assert core.parameters["init_v"].tolist() == [0, 0, 0]
    assert core.synapses[0].posts.tolist() == [0, 1, 2]
    assert core.synapses[0].offsets.tolist() == [0, 0, 1]
    assert core.synapses[0].weights.tolist() == [1]


def test_signed_csc_more_than_preview_limit():
    ram, lut = storage()
    ram[0, 0] = 1 << 32 | 10 << 47 | 69 << 35
    ram[1, 0] = 1 << 12
    # Each 8-bit CSC row has 5 weights and five absolute 16-bit indices.
    for row in range(60):
        base = row * 5
        ram[10 + row] = [
            0xFFFFFFFFFF | base << 48,
            (base + 1) | (base + 2) << 16 | (base + 3) << 32 | (base + 4) << 48,
        ]
    core = decode_core(dict(neuron_number=2, weight_width=3, weight_sign=1), ram, lut)
    assert core.synapses[0].weights.tolist() == [-1] * 300
    assert core.synapses[0].inputs.tolist() == list(range(300))


def test_nonzero_misaligned_csc_address_rejected():
    ram, lut = storage()
    ram[0, 0] = 1 << 32 | 10 << 47 | 10 << 35
    ram[1, 0] = 1 << 12
    ram[10, 0] = 1 | 3 << 48
    with pytest.raises(ValueError, match="not aligned"):
        decode_core(dict(neuron_number=2, input_width=3, weight_width=3), ram, lut)


@pytest.mark.parametrize(
    "register,value,error",
    [
        ("neuron_number", -1, ValueError),
        ("neuron_number", 4097, ValueError),
        ("weight_width", 4, ValueError),
        ("global_receive", 64, ValueError),
        ("delay_cycle", 65536, ValueError),
        ("width_cycle", 256, ValueError),
        ("typo_tick_start", 1, ValueError),
        ("tick_start", 1.0, TypeError),
        ("tick_start", True, TypeError),
    ],
)
def test_core_register_dictionary_rejects_unknown_or_out_of_range_values(
    register, value, error
):
    with pytest.raises(error):
        decode_core({register: value}, *storage())


def test_decoded_array_dtypes_and_public_config_are_owned():
    raw_config = {"neuron_number": 2, "delay_cycle": 65535, "width_cycle": 255}
    ram, lut = storage()
    ram[0, 0] = 1 << 32 | 8 << 47 | 8 << 35
    ram[8, 0] = 1
    decoded = decode_core(raw_config, ram, lut)
    raw_config["neuron_number"] = 4000
    assert decoded.config["neuron_number"] == 2
    assert decoded.voltage.dtype == np.int32
    assert decoded.lut_thresholds.dtype == np.int32
    assert decoded.lut_values.dtype == np.int16
    assert decoded.output_types.dtype == np.int8
    assert decoded.routes.dtype == np.int64
    assert decoded.voltage_locations.dtype == np.int64
    assert all(value.dtype == np.int64 for value in decoded.parameters.values())
    group = decoded.synapses[0]
    assert group.posts.dtype == group.inputs.dtype == group.offsets.dtype == np.int64
    assert group.weights.dtype == np.int16


def test_core_config_uses_paicorelib_enums_for_closed_set_registers():
    config = core_config(
        {
            "snn_ann": 1,
            "max_pooling": 1,
            "add_potential": 1,
            "zero_output": 1,
            "input_sign": 1,
            "input_width": 3,
            "output_sign": 1,
            "output_width": 2,
            "weight_sign": 1,
            "weight_width": 1,
            "csc_accelerate": 1,
            "lcn": 0,
            "target_lcn": 0,
        }
    )
    assert config["snn_ann"] is SNNMode.ANN
    assert config["max_pooling"] is PoolingMode.MAX
    assert config["add_potential"] is AddPotentialMode.DIRECT_ADD
    assert config["zero_output"] is ZeroOutputMode.ENABLE
    assert config["input_sign"] is DataSign.SIGNED
    assert config["input_width"] is DataWidth.WIDTH_8BIT
    assert config["output_sign"] is DataSign.SIGNED
    assert config["output_width"] is DataWidth.WIDTH_4BIT
    assert config["weight_sign"] is DataSign.SIGNED
    assert config["weight_width"] is DataWidth.WIDTH_2BIT
    assert config["csc_accelerate"] is CSCAccelerateMode.ENABLE
    raw = core_config({"lcn": 8, "target_lcn": 9})
    assert (raw["lcn"], raw["target_lcn"]) == (8, 9)
