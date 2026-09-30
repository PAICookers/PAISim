"""Shared semantic arrays consumed by the decoder and NumPy execution kernel.

Weights remain shared by a physical neuron group, including folded neurons.
Input addresses below are decoded *element* indices, not packed bit addresses.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, TypedDict, cast

import numpy as np
from numpy.typing import NDArray
from paicorelib.core_defs_v2 import (
    AddPotentialMode,
    CSCAccelerateMode,
    DataSign,
    DataWidth,
    OfflineCoreRegLimV2,
    PoolingMode,
    SNNMode,
    ZeroOutputMode,
)
from paicorelib.framelib.frame_defs import OfflineConfigFrame1FormatV2 as OffF1
from paicorelib.neuron_defs_v2 import OutputType

FrameArray = NDArray[np.uint64]
VoltageArray = NDArray[np.int32]
ParameterArray = NDArray[np.int64]
IndexArray = NDArray[np.int64]
WeightArray = NDArray[np.int16]
OutputArray = NDArray[np.int16]
OutputTypeArray = NDArray[np.int8]
LayoutArray = NDArray[np.uint8]
IntegerArray = NDArray[np.integer[Any]]


class CoreConfig(TypedDict):
    """Complete decoded register set used by the execution engine."""

    snn_ann: SNNMode
    max_pooling: PoolingMode
    add_potential: AddPotentialMode
    zero_output: ZeroOutputMode
    input_sign: DataSign
    input_width: DataWidth
    output_sign: DataSign
    output_width: DataWidth
    weight_sign: DataSign
    weight_width: DataWidth
    lcn: int
    target_lcn: int
    axon_skew: int
    neuron_number: int
    test_core_xy: int
    test_core_x: int
    test_core_y: int
    global_send: int
    csc_accelerate: CSCAccelerateMode
    global_receive: int
    thread_number: int
    busy_cycle: int
    delay_cycle: int
    width_cycle: int
    tick_start: int
    tick_duration: int
    tick_initial: int


class NeuronParameters(TypedDict):
    """Per-neuron int64 registers (or broadcastable scalar arrays for kernels)."""

    reset_mode: ParameterArray
    reset_v: ParameterArray
    thres_neg_mode: ParameterArray
    thres_pos_mode: ParameterArray
    thres_neg: ParameterArray
    thres_pos: ParameterArray
    lateral_inhi: ParameterArray
    leak_multi_sequence: ParameterArray
    leak_multi_input: ParameterArray
    leak_multi_mode: ParameterArray
    leak_add_mode: ParameterArray
    leak_tau: ParameterArray
    leak_v: ParameterArray
    init_v: ParameterArray


def core_config(values: Mapping[str, Any]) -> CoreConfig:
    """Validate integer register values before narrowing a public dictionary.

    This validates wire ranges, not whether a combination is supported by the
    execution policy. In particular, high-bit LCN is preserved for the engine
    to diagnose instead of being silently masked to a supported value.
    """
    limits = {
        "snn_ann": OffF1.Word1.SNN_ANN_MASK,
        "max_pooling": OffF1.Word1.MAX_POOLING_MASK,
        "add_potential": OffF1.Word1.ADD_POTENTIAL_MASK,
        "zero_output": OffF1.Word1.ZERO_OUTPUT_MASK,
        "input_sign": OffF1.Word1.INPUT_SIGN_MASK,
        "input_width": OffF1.Word1.INPUT_WIDTH_MASK,
        "output_sign": OffF1.Word1.OUTPUT_SIGN_MASK,
        "output_width": OffF1.Word1.OUTPUT_WIDTH_MASK,
        "weight_sign": OffF1.Word1.WEIGHT_SIGN_MASK,
        "weight_width": OffF1.Word1.WEIGHT_WIDTH_MASK,
        "lcn": OffF1.Word1.LCN_MASK,
        "target_lcn": OffF1.Word1.TARGET_LCN_MASK,
        "axon_skew": OffF1.Word1.AXON_SKEW_MASK,
        "neuron_number": OfflineCoreRegLimV2.NEURON_NUMBER_MAX,
        "test_core_xy": OffF1.Word1.TEST_CORE_XY_MASK,
        "test_core_x": OffF1.Word1.TEST_CORE_X_MASK,
        "test_core_y": OffF1.Word1.TEST_CORE_Y_HIGH2_MASK
        | (
            OffF1.Word2.TEST_CORE_Y_LOW4_MASK
            << OffF1.Word1.TEST_CORE_Y_HIGH2_MASK.bit_length()
        ),
        "global_send": OffF1.Word2.GLOBAL_SEND_MASK,
        "csc_accelerate": OffF1.Word2.CSC_ACCELERATE_MASK,
        "global_receive": OffF1.Word2.GLOBAL_RECEIVE_MASK,
        "thread_number": OffF1.Word2.THREAD_NUMBER_MASK,
        "busy_cycle": OffF1.Word2.BUSY_CYCLE_MASK,
        "delay_cycle": OffF1.Word2.DELAY_CYCLE_MASK,
        "width_cycle": OffF1.Word2.WIDTH_CYCLE_MASK,
        "tick_start": OffF1.Word3.TICK_START_MASK,
        "tick_duration": OffF1.Word3.TICK_DURATION_MASK,
        "tick_initial": OffF1.Word3.TICK_INITIAL_MASK,
    }
    enum_fields = {
        "snn_ann": SNNMode,
        "max_pooling": PoolingMode,
        "add_potential": AddPotentialMode,
        "zero_output": ZeroOutputMode,
        "input_sign": DataSign,
        "input_width": DataWidth,
        "output_sign": DataSign,
        "output_width": DataWidth,
        "weight_sign": DataSign,
        "weight_width": DataWidth,
        "csc_accelerate": CSCAccelerateMode,
    }
    result = {
        name: (enum_type(0) if (enum_type := enum_fields.get(name)) else 0)
        for name in limits
    }
    for name, value in values.items():
        if name not in limits:
            raise ValueError(f"unknown core register {name!r}")
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer register value")
        if not 0 <= value <= limits[name]:
            raise ValueError(f"{name} is outside its register range")
        enum_type = enum_fields.get(name)
        result[name] = value if enum_type is None else enum_type(value)
    # Every key and value was checked above; the dynamic dictionary is narrowed
    # once at this validation boundary.
    return cast(CoreConfig, result)


@dataclass
class SynapseGroup:
    """One weight vector shared by postsynaptic neurons with input offsets."""

    posts: IndexArray
    inputs: IndexArray
    weights: WeightArray
    offsets: IndexArray


@dataclass(frozen=True, slots=True)
class FoldLayout:
    """Decoded fold transition parameters and logical input offsets."""

    count: int
    ranges: tuple[int, int, int]
    skews: tuple[int, int, int]
    offsets: tuple[int, ...]


@dataclass
class DecodedCore:
    """Complete compute configuration; runtime voltage is kept by the engine."""

    config: CoreConfig
    voltage: VoltageArray
    parameters: NeuronParameters
    routes: IndexArray
    axons: IndexArray
    ticks: IndexArray
    output_types: OutputTypeArray
    voltage_locations: IndexArray
    parameter_locations: IndexArray
    layouts: LayoutArray
    synapses: tuple[SynapseGroup, ...]
    lut_thresholds: VoltageArray
    lut_values: OutputArray
    folds: tuple[FoldLayout, ...] = ()


@dataclass(frozen=True, slots=True)
class LutEntry:
    """One decoded offline LUT entry with its original packed word."""

    index: int
    threshold: int
    activation: int
    raw: int


@dataclass(frozen=True, slots=True)
class WeightRef:
    """A physical synapse group reference, without copying its weights."""

    group: int
    posts: tuple[int, ...]
    inputs: tuple[int, ...]
    offsets: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class NeuronRecord:
    """Stable static description of one logical offline neuron."""

    index: int
    layout: int
    parameter_row: int
    voltage_row: int
    voltage_lane: int
    tick: int
    axon: int
    output_type: OutputType


@dataclass(frozen=True, slots=True)
class CoreSnapshot:
    """Detached, read-only observation of one offline core."""

    revision: int
    selected: frozenset[str]
    config: CoreConfig | None
    registers: FrameArray | None
    neuron_sram: FrameArray | None
    input_sram: NDArray[np.uint8] | None
    lut_sram: FrameArray | None
    voltage: VoltageArray | None
    neurons: tuple[NeuronRecord, ...] | None
    weights: tuple[WeightRef, ...] | None
    lut: tuple[LutEntry, ...] | None
    folds: tuple[FoldLayout, ...] | None


SnapshotPart = Literal[
    "config", "input", "neurons", "weights", "lut", "folds", "voltage"
]


def snapshot_parts(selection: Iterable[str] | None) -> frozenset[SnapshotPart]:
    """Validate component names used by selective core observations."""
    parts = frozenset(
        ("config", "input", "neurons", "weights", "lut", "folds", "voltage")
        if selection is None
        else selection
    )
    allowed = {
        "config",
        "input",
        "neurons",
        "weights",
        "lut",
        "folds",
        "voltage",
    }
    if not parts <= allowed:
        unknown = sorted(set(parts) - allowed)
        raise ValueError(f"unknown snapshot selection: {unknown}")
    return cast(frozenset[SnapshotPart], parts)


@dataclass
class StepResult:
    """Stored voltage, DATA values and sampled pre-reset potential per neuron."""

    voltage: VoltageArray
    output: OutputArray
    potential: VoltageArray
    positive: bool
