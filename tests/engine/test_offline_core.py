"""Offline-core numerical checks independent from the simulator scheduler."""

import numpy as np
import pytest
from numpy.typing import NDArray

from paisim import ChipCoord, CoreAddr, CoreCoord
from paisim.engine.core import OfflineCore
from paisim.engine.model import DecodedCore, NeuronParameters, SynapseGroup
from paisim.engine.numeric import step


def _core(size: int) -> DecodedCore:
    zeros: NDArray[np.int32] = np.zeros(size, dtype=np.int32)
    parameters: NeuronParameters = {
        "reset_mode": np.zeros(size, dtype=np.int64) + 2,
        "reset_v": np.zeros(size, dtype=np.int64),
        "thres_neg_mode": np.zeros(size, dtype=np.int64),
        "thres_pos_mode": np.zeros(size, dtype=np.int64),
        "thres_neg": np.zeros(size, dtype=np.int64) - (1 << 31),
        "thres_pos": np.zeros(size, dtype=np.int64) + (1 << 31) - 1,
        "lateral_inhi": np.zeros(size, dtype=np.int64),
        "leak_multi_sequence": np.zeros(size, dtype=np.int64) + 1,
        "leak_multi_input": np.zeros(size, dtype=np.int64),
        "leak_multi_mode": np.zeros(size, dtype=np.int64),
        "leak_add_mode": np.zeros(size, dtype=np.int64),
        "leak_tau": np.zeros(size, dtype=np.int64),
        "leak_v": np.zeros(size, dtype=np.int64),
        "init_v": np.zeros(size, dtype=np.int64),
    }
    return DecodedCore(
        config={"snn_ann": 0, "max_pooling": 0, "add_potential": 0, "output_width": 3},
        voltage=zeros.copy(),
        parameters=parameters,
        routes=np.zeros((size, 6), dtype=np.int64),
        axons=zeros.astype(np.int64),
        ticks=zeros.astype(np.int64),
        output_types=zeros.astype(np.int8),
        voltage_locations=np.zeros((size, 2), dtype=np.int64),
        parameter_locations=zeros.astype(np.int64),
        layouts=np.zeros(size, dtype=np.uint8),
        synapses=(
            SynapseGroup(
                posts=np.arange(size, dtype=np.int64),
                inputs=np.array([0], dtype=np.int64),
                weights=np.array([3], dtype=np.int16),
                offsets=np.arange(size, dtype=np.int64),
            ),
        ),
        lut_thresholds=np.arange(256, dtype=np.int32),
        lut_values=np.arange(256, dtype=np.int16),
    )


def _wrap_scalar(value: int) -> int:
    """Apply the modeled signed-int32 two's-complement narrowing."""
    return ((value + (1 << 31)) & 0xFFFF_FFFF) - (1 << 31)


def _scalar_step(
    core: DecodedCore,
    inputs: NDArray[np.int32],
    voltage: NDArray[np.int32],
) -> tuple[NDArray[np.int32], NDArray[np.int16], NDArray[np.int32]]:
    """Compute the simple no-leak SNN fixture without calling ``numeric.step``."""
    charges = [0] * voltage.size
    for group in core.synapses:
        for post, offset in zip(
            group.posts.tolist(), group.offsets.tolist(), strict=True
        ):
            for base, weight in zip(
                group.inputs.tolist(), group.weights.tolist(), strict=True
            ):
                if weight:
                    charges[post] += int(inputs[base + offset]) * int(weight)

    next_voltage: list[int] = []
    output: list[int] = []
    potential: list[int] = []
    parameters = core.parameters
    for index, old in enumerate(voltage.tolist()):
        value = _wrap_scalar(old + charges[index])
        potential.append(value)
        positive = (
            parameters["thres_pos_mode"][index] == 0
            and value >= parameters["thres_pos"][index]
        )
        negative = (
            not positive
            and parameters["thres_neg_mode"][index] == 0
            and value <= parameters["thres_neg"][index]
        )
        output.append(1 if positive else -1 if negative else 0)
        next_voltage.append(value)
    return (
        np.asarray(next_voltage, dtype=np.int32),
        np.asarray(output, dtype=np.int16),
        np.asarray(potential, dtype=np.int32),
    )


def test_vectorized_step_matches_independent_scalar_steps() -> None:
    vector = _core(4)
    inputs = np.array([2, 3, 5, 7], dtype=np.int32)
    initial = np.array([11, -2, 4, -8], dtype=np.int32)
    expected_voltage, expected_output, expected_potential = _scalar_step(
        vector, inputs, initial
    )

    result = step(vector, inputs, initial)
    np.testing.assert_array_equal(result.voltage, expected_voltage)
    np.testing.assert_array_equal(result.output, expected_output)
    np.testing.assert_array_equal(result.potential, expected_potential)


@pytest.mark.parametrize(
    "value, weight, expected",
    [
        (np.int64((1 << 31) - 1), np.int16(2), -2),
        (np.int64(-(1 << 31)), np.int16(1), -(1 << 31)),
    ],
)
def test_signed_int32_writeback_wraps_without_saturation(
    value: np.int64, weight: np.int16, expected: int
) -> None:
    core = _core(1)
    core.synapses = (
        SynapseGroup(
            posts=np.array([0], dtype=np.int64),
            inputs=np.array([0], dtype=np.int64),
            weights=np.asarray([weight], dtype=np.int16),
            offsets=np.array([0], dtype=np.int64),
        ),
    )
    result = step(
        core, np.asarray([value], dtype=np.int64), np.asarray([0], dtype=np.int64)
    )
    assert result.voltage[0] == expected


def test_offline_core_step_consumes_one_logical_tick() -> None:
    core = OfflineCore(CoreAddr(ChipCoord(0, 0), CoreCoord(1, 2)))
    core.configure(np.zeros(3, dtype=np.uint64))
    core.input_window.write_frame(
        timestep=0,
        target_lcn=0,
        tick_relative=0,
        axon=0,
        data=7,
        width=8,
    )
    result = core.step(0)
    assert result.voltage.size == 0
    assert core.work_steps == 1
    assert core.input_sram[0].sum() == 0
