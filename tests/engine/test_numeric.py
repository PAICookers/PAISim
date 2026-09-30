"""Hand-calculated numerical contracts; these are not hardware acceptance tests."""

import numpy as np
import pytest

from paisim.engine.errors import UndefinedHardwareBehavior
from paisim.engine.model import DecodedCore, SynapseGroup
from paisim.engine.numeric import step, validate_configuration


def core(size=1, **parameters):
    """A wide-threshold, no-reset core with one independent input per neuron."""
    defaults = {
        "reset_mode": 2,
        "reset_v": 0,
        "thres_neg_mode": 0,
        "thres_pos_mode": 0,
        "thres_neg": -(1 << 31),
        "thres_pos": (1 << 31) - 1,
        "lateral_inhi": 0,
        "leak_multi_sequence": 1,
        "leak_multi_input": 0,
        "leak_multi_mode": 0,
        "leak_add_mode": 0,
        "leak_tau": 0,
        "leak_v": 0,
        "init_v": 0,
    }
    defaults.update(parameters)
    zeros = np.zeros(size, dtype=np.int32)
    return DecodedCore(
        config={"snn_ann": 0, "max_pooling": 0, "add_potential": 0, "output_width": 3},
        voltage=zeros.copy(),
        parameters={
            key: np.asarray(value, dtype=np.int64) for key, value in defaults.items()
        },
        routes=np.zeros((size, 6), dtype=np.int32),
        axons=zeros.copy(),
        ticks=zeros.copy(),
        output_types=zeros.copy(),
        voltage_locations=np.zeros((size, 2), dtype=np.int32),
        parameter_locations=np.zeros(size, dtype=np.int32),
        layouts=np.zeros(size, dtype=np.uint8),
        synapses=(
            SynapseGroup(
                posts=np.arange(size, dtype=np.int32),
                inputs=np.array([0], dtype=np.int32),
                weights=np.array([1], dtype=np.int8),
                offsets=np.arange(size, dtype=np.int32),
            ),
        ),
        lut_thresholds=np.arange(256, dtype=np.int32),
        lut_values=np.arange(256, dtype=np.int16),
    )


def run(config, inputs, voltage=None, **kwargs):
    if voltage is None:
        voltage = config.voltage
    return step(config, np.asarray(inputs), np.asarray(voltage), **kwargs)


def test_shared_sparse_weights_offsets_and_unsigned_input():
    config = core(3)
    config.synapses = (
        SynapseGroup(
            np.array([0, 2]), np.array([0, 2]), np.array([-2, 3]), np.array([0, 1])
        ),
    )
    result = run(config, np.array([255, 10, 2, 5], dtype=np.uint8), [1, 9, -1])
    np.testing.assert_array_equal(result.voltage, [-503, 9, -6])
    np.testing.assert_array_equal(config.voltage, [0, 0, 0])


@pytest.mark.parametrize(
    "reset_mode, expected", [(0, [2, -2, 1]), (1, [2, -2, 1]), (2, [7, -7, 1])]
)
def test_signed_thresholds_three_resets_and_pre_reset_output(reset_mode, expected):
    config = core(3, thres_pos=5, thres_neg=-5, reset_mode=reset_mode, reset_v=2)
    result = run(config, [7, -7, 1])
    np.testing.assert_array_equal(result.output, [1, -1, 0])
    np.testing.assert_array_equal(result.potential, [7, -7, 1])
    np.testing.assert_array_equal(result.voltage, expected)
    assert result.positive


def test_threshold_equality_prefers_positive_branch():
    result = run(core(3, thres_pos=0, thres_neg=0), [-1, 0, 1])
    np.testing.assert_array_equal(result.output, [-1, 1, 1])


def test_per_neuron_modes_and_range_clamp():
    config = core(
        4,
        thres_pos=5,
        thres_neg=-5,
        thres_pos_mode=[1, 0, 0, 0],
        thres_neg_mode=[0, 1, 0, 0],
        reset_mode=[0, 0, 1, 2],
        reset_v=1,
    )
    result = run(config, [9, -9, -7, 8])
    np.testing.assert_array_equal(result.potential, [5, -5, -7, 8])
    np.testing.assert_array_equal(result.output, [0, 0, -1, 1])
    np.testing.assert_array_equal(result.voltage, [5, -5, -2, 8])


def test_additive_reverse_leak_and_zero_input_evolution():
    config = core(3, leak_add_mode=1, leak_v=-2)
    first = run(config, [0, 0, 0], [3, -3, 0])
    np.testing.assert_array_equal(first.voltage, [1, -1, 0])
    second = run(config, [0, 0, 0], first.voltage)
    # The formula crosses zero; it does not add an undocumented clamp.
    np.testing.assert_array_equal(second.voltage, [-1, 1, 0])


@pytest.mark.parametrize(
    "sequence, potential, output, voltage", [(0, 3, 0, 3), (1, 6, 1, 1)]
)
def test_multiplicative_leak_order(sequence, potential, output, voltage):
    config = core(
        thres_pos=5,
        reset_mode=0,
        reset_v=2,
        leak_tau=-1,
        leak_multi_sequence=sequence,
    )
    result = run(config, [6])
    assert result.potential.tolist() == [potential]
    assert result.output.tolist() == [output]
    assert result.voltage.tolist() == [voltage]


def test_leak_mode_and_input_scaling_are_independent_registers():
    config = core(
        4,
        reset_v=2,
        leak_tau=-1,
        leak_multi_sequence=0,
        leak_multi_mode=[0, 1, 0, 1],
        leak_multi_input=[0, 0, 1, 1],
    )
    # Charges are 10,10,7,7; enabled leak subtracts floor((V-2)/2).
    result = run(config, [6, 6, 6, 6], [4, 4, 4, 4])
    np.testing.assert_array_equal(result.voltage, [5, 6, 3, 5])


def test_signed_shifts_and_wrap_at_arithmetic_boundaries():
    config = core(4, leak_tau=[-1, -32, 1, 31], leak_multi_sequence=0)
    result = run(config, [-3, -1, (1 << 31) - 1, 1])
    np.testing.assert_array_equal(result.voltage, [-2, -1, -2, -(1 << 31)])
    result = run(core(leak_v=1), [(1 << 31) - 1])
    assert result.potential.tolist() == [-(1 << 31)]


def test_mac_widens_before_multiply_and_wraps_final_sum():
    config = core()
    config.synapses = (
        SynapseGroup(
            np.array([0]), np.array([0]), np.array([127], dtype=np.int8), np.array([0])
        ),
    )
    assert run(config, np.array([127], dtype=np.int8)).voltage.tolist() == [16129]
    config.config["add_potential"] = 1
    config.synapses = (
        SynapseGroup(np.array([0]), np.array([0, 1]), np.array([1, 1]), np.array([0])),
    )
    assert run(config, [(1 << 31) - 1, 2]).voltage.tolist() == [-(1 << 31) + 1]


def test_vmm_accumulator_stays_int32_without_input_truncation():
    config = core()
    config.synapses = (
        SynapseGroup(
            np.array([0]),
            np.array([0, 1]),
            np.array([127, 127], dtype=np.int8),
            np.array([0]),
        ),
    )
    result = run(config, np.array([127, 127], dtype=np.int32))
    assert result.voltage.dtype == np.int32
    assert result.voltage.tolist() == [32258]


def test_pool_replaces_history_and_ignores_disconnected_inputs():
    config = core(2)
    config.config["max_pooling"] = 1
    config.synapses = (
        SynapseGroup(
            np.array([0]), np.array([0, 1, 2]), np.array([1, 1, 0]), np.array([0])
        ),
    )
    result = run(config, [-5, -2, 100], [99, 99])
    np.testing.assert_array_equal(result.voltage, [-2, 0])


def test_lateral_inhibition_is_previous_step_core_state():
    config = core(2, lateral_inhi=[1, 0], reset_v=[3, 4])
    result = run(config, [1, 1], [20, 20], previous_positive=True)
    np.testing.assert_array_equal(result.voltage, [4, 21])


@pytest.mark.parametrize("width", [0, 1, 2, 3])
def test_sar_uses_physical_lut_addresses_for_output_width(width):
    config = core(3)
    config.config.update(snn_ann=1, output_width=width)
    block = 256 // (1 << (1 << width))
    result = run(config, [127, 128, 255])
    np.testing.assert_array_equal(
        result.output, [127 // block * block, 128, 255 // block * block]
    )


def test_unsorted_lut_is_not_sorted_or_bucketized_and_signed_reset():
    config = core(3, reset_mode=[0, 1, 0], reset_v=3)
    config.config.update(snn_ann=1, output_width=0)
    config.lut_thresholds[:] = 1000
    config.lut_thresholds[0] = -100
    config.lut_thresholds[128] = 5
    config.lut_values[:] = 17
    config.lut_values[0] = -2
    config.lut_values[128] = 3
    result = run(config, [4, 7, 7])
    np.testing.assert_array_equal(result.output, [-2, 3, 3])
    np.testing.assert_array_equal(result.voltage, [-3, 2, 3])


def test_ann_zero_does_not_reset_and_observed_arrays_do_not_alias():
    config = core(reset_mode=0, reset_v=3)
    config.config["snn_ann"] = 1
    config.lut_values[:] = 0
    incoming = np.array([8], dtype=np.int32)
    voltage = np.array([2], dtype=np.int32)
    result = run(config, incoming, voltage)
    assert result.voltage.tolist() == [10]
    assert incoming.tolist() == [8] and voltage.tolist() == [2]
    result.voltage[0] = 999
    assert result.potential.tolist() == [10]
    assert not result.positive


def test_invalid_register_or_input_fails_explicitly():
    with pytest.raises(ValueError, match="one-dimensional integer"):
        run(core(), [1.0])
    with pytest.raises(ValueError, match="reset_mode"):
        run(core(reset_mode=3), [1])
    with pytest.raises(ValueError, match="outside the input"):
        run(core(2), [1])


@pytest.mark.parametrize("field", ["lcn", "target_lcn"])
def test_high_lcn_reports_undefined_hardware_behavior(field):
    config = core()
    config.config[field] = 8
    with pytest.raises(
        UndefinedHardwareBehavior, match="hardware behavior is undefined"
    ):
        validate_configuration(config)


def test_pool_and_direct_add_report_undefined_hardware_behavior():
    config = core()
    config.config.update(max_pooling=1, add_potential=1)
    message = "hardware behavior is undefined"
    with pytest.raises(UndefinedHardwareBehavior, match=message):
        validate_configuration(config)
    with pytest.raises(UndefinedHardwareBehavior, match=message):
        run(config, [1])


def test_large_folded_group_crosses_bounded_gather_chunks():
    config = core(520)
    config.synapses = (
        SynapseGroup(
            np.arange(520), np.arange(512), np.ones(512, dtype=np.int8), np.arange(520)
        ),
    )
    result = run(config, np.ones(1031, dtype=np.int8))
    np.testing.assert_array_equal(result.voltage, np.full(520, 512))


def test_static_configuration_validation_catches_neuron_index_and_input_extent():
    config = core()
    config.config["input_width"] = 3
    config.synapses[0].posts[0] = 1
    with pytest.raises(ValueError, match="postsynaptic"):
        validate_configuration(config)
    config.synapses[0].posts[0] = 0
    config.synapses[0].offsets[0] = 64
    with pytest.raises(ValueError, match="configured input vector"):
        validate_configuration(config)
    config.config["lcn"] = 1
    validate_configuration(config)
