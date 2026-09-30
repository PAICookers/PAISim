"""Vectorized integer dynamics for one configured offline core.

The order follows the documented stages and the existing toolchain convention:
charge, additive leak, optional pre-comparison leak, range clamp, fire/LUT,
reset, optional post-comparison leak. The stored potential output is sampled
after range clamp and before reset. Every arithmetic stage wraps to signed
32 bits; negative right shifts are arithmetic. These overflow, sampling and
stage-order choices are provisional numerical assumptions, not an
RTL- or silicon-validated claim. Threshold ties prefer the positive branch,
following the application's if/elif formula.
"""

import numpy as np
from numpy.typing import NDArray
from paicorelib.core_defs_v2 import AddPotentialMode, PoolingMode, SNNMode
from paicorelib.neuron_defs_v2 import OfflineNeuRegLimV2

from ..hardware import (
    BINARY_MODE_MAX,
    DIRECT_POTENTIAL_BITS,
    DOCUMENTED_LCN_MAX,
    I32_BITS,
    I32_MASK,
    OFFLINE_INPUT_BITS,
    OFFLINE_LUT_ENTRIES,
    RESET_MODE_MAX,
    SUPPORTED_DATA_WIDTHS,
)
from .errors import UndefinedHardwareBehavior
from .model import DecodedCore, IntegerArray, NeuronParameters, StepResult

_I64 = NDArray[np.int64]
_I32 = NDArray[np.int32]
_I32_MIN = OfflineNeuRegLimV2.THRES_NEG_MIN
_I32_MAX = OfflineNeuRegLimV2.THRES_NEG_MAX
# Bound gather and product temporaries; folded groups do not expand to dense W.
_GATHER_ELEMENTS = 262_144


def _wrap(value: _I64) -> _I64:
    """Represent two's-complement int32 arithmetic in an int64 working array."""
    sign_bit = 1 << (I32_BITS - 1)
    return ((value + sign_bit) & I32_MASK) - sign_bit


def _shift(value: _I64, exponent: _I64) -> _I64:
    # Both branches are evaluated by where: clip the unused shift counts.
    return _wrap(
        np.where(
            exponent >= 0,
            value << np.maximum(exponent, 0),
            value >> np.maximum(-exponent, 0),
        )
    )


def _leak(value: _I64, parameters: NeuronParameters) -> _I64:
    reset = parameters["reset_v"]
    return np.where(
        parameters["leak_multi_mode"] == 1,
        _wrap(value - _shift(_wrap(value - reset), parameters["leak_tau"])),
        _shift(value, parameters["leak_tau"]),
    )


def _integer_vector(value: IntegerArray, name: str) -> _I64:
    array = value
    if array.ndim != 1 or array.dtype.kind not in "iu":
        raise ValueError(f"{name} must be a one-dimensional integer array")
    if np.any(array < _I32_MIN) or np.any(array > _I32_MAX):
        raise ValueError(f"{name} must fit signed int32")
    return array.astype(np.int64, copy=False)


def _int32_vector(value: IntegerArray, name: str) -> _I32:
    array = value
    if array.ndim != 1 or array.dtype.kind not in "iu":
        raise ValueError(f"{name} must be a one-dimensional integer array")
    if np.any(array < _I32_MIN) or np.any(array > _I32_MAX):
        raise ValueError(f"{name} must fit signed int32")
    return array.astype(np.int32, copy=False)


def _parameters(core: DecodedCore, size: int) -> NeuronParameters:
    """Accept scalar or per-neuron registers, rejecting reserved encodings."""
    p = core.parameters

    def checked(name: str, raw: _I64, lower: int, upper: int) -> _I64:
        if raw.ndim > 1 or raw.dtype.kind not in "iu":
            raise ValueError(f"{name} must be an integer scalar or vector")
        if np.any(raw < lower) or np.any(raw > upper):
            raise ValueError(f"{name} is outside its register range")
        try:
            return np.broadcast_to(raw, (size,)).astype(np.int64, copy=False)
        except ValueError as exc:
            raise ValueError(f"{name} must broadcast to {size} neurons") from exc

    result = NeuronParameters(
        reset_mode=checked("reset_mode", p["reset_mode"], 0, RESET_MODE_MAX),
        reset_v=checked(
            "reset_v",
            p["reset_v"],
            OfflineNeuRegLimV2.RESET_V_MIN,
            OfflineNeuRegLimV2.RESET_V_MAX,
        ),
        thres_neg_mode=checked(
            "thres_neg_mode", p["thres_neg_mode"], 0, BINARY_MODE_MAX
        ),
        thres_pos_mode=checked(
            "thres_pos_mode", p["thres_pos_mode"], 0, BINARY_MODE_MAX
        ),
        thres_neg=checked("thres_neg", p["thres_neg"], _I32_MIN, _I32_MAX),
        thres_pos=checked("thres_pos", p["thres_pos"], _I32_MIN, _I32_MAX),
        lateral_inhi=checked("lateral_inhi", p["lateral_inhi"], 0, BINARY_MODE_MAX),
        leak_multi_sequence=checked(
            "leak_multi_sequence", p["leak_multi_sequence"], 0, BINARY_MODE_MAX
        ),
        leak_multi_input=checked(
            "leak_multi_input", p["leak_multi_input"], 0, BINARY_MODE_MAX
        ),
        leak_multi_mode=checked(
            "leak_multi_mode", p["leak_multi_mode"], 0, BINARY_MODE_MAX
        ),
        leak_add_mode=checked("leak_add_mode", p["leak_add_mode"], 0, BINARY_MODE_MAX),
        leak_tau=checked(
            "leak_tau",
            p["leak_tau"],
            OfflineNeuRegLimV2.LEAK_TAU_MIN,
            OfflineNeuRegLimV2.LEAK_TAU_MAX,
        ),
        leak_v=checked(
            "leak_v",
            p["leak_v"],
            OfflineNeuRegLimV2.LEAK_V_MIN,
            OfflineNeuRegLimV2.LEAK_V_MAX,
        ),
        init_v=checked(
            "init_v",
            p["init_v"],
            OfflineNeuRegLimV2.VJT_INITIAL_MIN,
            OfflineNeuRegLimV2.VJT_INITIAL_MAX,
        ),
    )
    if np.any(result["thres_pos"] < result["thres_neg"]):
        raise ValueError("positive threshold must not be below negative threshold")
    return result


def validate_configuration(core: DecodedCore) -> None:
    """Reject statically invalid execution state before advancing a group tick.

    The decoder calls this once after constructing owned semantic arrays. The
    public ``step`` boundary still validates mutable arrays supplied directly
    by callers. No input data are allocated merely to validate address bounds.
    """
    size = core.voltage.size
    _parameters(core, size)
    config = core.config
    lcn = config.get("lcn", 0)
    if lcn > DOCUMENTED_LCN_MAX or config.get("target_lcn", 0) > DOCUMENTED_LCN_MAX:
        raise UndefinedHardwareBehavior(
            f"hardware behavior is undefined for LCN codes above "
            f"{DOCUMENTED_LCN_MAX}; confirm with "
            "chip designers or hardware testing"
        )
    direct = config.get("add_potential", AddPotentialMode.NORMAL) == (
        AddPotentialMode.DIRECT_ADD
    )
    if direct and config.get("max_pooling", PoolingMode.AVERAGE) == PoolingMode.MAX:
        raise UndefinedHardwareBehavior(
            "hardware behavior is undefined when max_pooling and add_potential "
            "are enabled together; confirm with chip designers or hardware testing"
        )
    width = (
        DIRECT_POTENTIAL_BITS
        if direct
        else SUPPORTED_DATA_WIDTHS[config.get("input_width", 0)]
    )
    input_size = (OFFLINE_INPUT_BITS << lcn) // width
    for group in core.synapses:
        posts = _integer_vector(group.posts, "posts")
        bases = _integer_vector(group.inputs, "synapse inputs")
        offsets = _integer_vector(group.offsets, "offsets")
        weights = _integer_vector(group.weights, "weights")
        if posts.shape != offsets.shape or bases.shape != weights.shape:
            raise ValueError("synapse group array lengths disagree")
        if np.any(posts < 0) or np.any(posts >= size):
            raise ValueError("postsynaptic neuron index is out of range")
        if not posts.size or not bases.size:
            continue
        if bases.min() + offsets.min() < 0 or bases.max() + offsets.max() >= input_size:
            raise ValueError(
                "synapse input address is outside the configured input vector"
            )


def _aggregate(core: DecodedCore, inputs: _I32, size: int) -> _I32:
    """Gather fan-ins with a signed-int32 accumulator; never build dense W."""
    pool = core.config["max_pooling"] == PoolingMode.MAX
    direct = core.config["add_potential"] == AddPotentialMode.DIRECT_ADD
    if pool and direct:
        raise UndefinedHardwareBehavior(
            "hardware behavior is undefined when max_pooling and add_potential "
            "are enabled together; confirm with chip designers or hardware testing"
        )
    result = np.full(size, _I32_MIN if pool else 0, dtype=np.int32)
    seen = np.zeros(size, dtype=np.bool_)
    for group in core.synapses:
        posts = _integer_vector(group.posts, "posts")
        bases = _integer_vector(group.inputs, "synapse inputs")
        offsets = _integer_vector(group.offsets, "offsets")
        weights = _int32_vector(group.weights, "weights")
        if offsets.shape != posts.shape or bases.shape != weights.shape:
            raise ValueError("synapse group array lengths disagree")
        if np.any(posts < 0) or np.any(posts >= size):
            raise ValueError("postsynaptic neuron index is out of range")
        if not bases.size or not posts.size:
            continue
        if bases.min() + offsets.min() < 0 or (
            bases.max() + offsets.max() >= inputs.size
        ):
            raise ValueError("synapse input address is outside the input vector")
        # Keep decoded zero weights semantically absent, including pool masks.
        active = weights != 0
        bases, weights = bases[active], weights[active]
        if not bases.size:
            continue
        rows = max(1, _GATHER_ELEMENTS // bases.size)
        for start in range(0, posts.size, rows):
            post = posts[start : start + rows]
            offset = offsets[start : start + rows]
            values = inputs[bases[None, :] + offset[:, None]]
            if pool:
                np.maximum.at(result, post, values.max(axis=1))
            else:
                if not direct:
                    values = values * weights
                np.add.at(result, post, values.sum(axis=1, dtype=np.int32))
            seen[post] = True
    # Empty fan-in is defined as zero, also in max-pool mode.
    result[~seen] = 0
    return result


def _lookup(core: DecodedCore, potential: _I64) -> tuple[_I64, _I64]:
    """Use physical SAR addresses, including reduced output precision."""
    thresholds = _integer_vector(core.lut_thresholds, "LUT thresholds")
    values = _integer_vector(core.lut_values, "LUT values")
    if thresholds.shape != (OFFLINE_LUT_ENTRIES,) or values.shape != (
        OFFLINE_LUT_ENTRIES,
    ):
        raise ValueError(
            f"ANN requires {OFFLINE_LUT_ENTRIES} threshold and activation entries"
        )
    width = core.config["output_width"]
    if width not in range(len(SUPPORTED_DATA_WIDTHS)):
        raise ValueError("output_width must be encoded as 0, 1, 2, or 3")
    bits = 1 << width
    block = OFFLINE_LUT_ENTRIES >> bits
    prefix = np.zeros(potential.size, dtype=np.int64)
    for bit in range(bits - 1, -1, -1):
        candidate = prefix | (1 << bit)
        prefix = np.where(potential >= thresholds[candidate * block], candidate, prefix)
    address = prefix * block
    return values[address], thresholds[address]


def step(
    core: DecodedCore,
    inputs: IntegerArray,
    voltage: IntegerArray,
    previous_positive: bool = False,
) -> StepResult:
    """Compute one active step without modifying configuration or input arrays.

    Inputs are decoded signed/unsigned values, or int32 potentials for direct
    addition. Time/slot selection, initialization, zero-output suppression and
    frame encoding belong to the engine. Max pooling replaces the old potential
    with the maximum selected input (application figure 1.24); it does not add
    that maximum to the previous potential. ``potential`` samples the clamped
    pre-reset voltage; ``voltage`` is the final state for the next active step.
    """
    old = _integer_vector(voltage, "voltage")
    incoming = _int32_vector(inputs, "inputs")
    if old.size != core.voltage.size:
        raise ValueError("voltage size does not match the decoded neuron count")
    p = _parameters(core, old.size)
    if core.config["snn_ann"] not in (SNNMode.SNN, SNNMode.ANN):
        raise ValueError("snn_ann must be 0 or 1")
    if core.config["max_pooling"] not in (PoolingMode.AVERAGE, PoolingMode.MAX):
        raise ValueError("max_pooling must be 0 or 1")
    if core.config["add_potential"] not in (
        AddPotentialMode.NORMAL,
        AddPotentialMode.DIRECT_ADD,
    ):
        raise ValueError("add_potential must be 0 or 1")

    old = np.where(previous_positive & (p["lateral_inhi"] == 1), p["reset_v"], old)
    charge = _aggregate(core, incoming, old.size)
    charge = np.where(p["leak_multi_input"] == 1, _shift(charge, p["leak_tau"]), charge)
    value = (
        charge if core.config["max_pooling"] == PoolingMode.MAX else _wrap(old + charge)
    )
    additive = np.where(
        p["leak_add_mode"] == 0, p["leak_v"], np.sign(value) * p["leak_v"]
    )
    value = _wrap(value + additive)
    value = np.where(p["leak_multi_sequence"] == 0, _leak(value, p), value)
    value = np.where(p["thres_pos_mode"] == 1, np.minimum(value, p["thres_pos"]), value)
    value = np.where(p["thres_neg_mode"] == 1, np.maximum(value, p["thres_neg"]), value)
    potential = value.astype(np.int32)

    if core.config["snn_ann"] == SNNMode.ANN:
        output, threshold = _lookup(core, value)
        positive, negative = output > 0, output < 0
    else:
        positive = (p["thres_pos_mode"] == 0) & (value >= p["thres_pos"])
        negative = ~positive & (p["thres_neg_mode"] == 0) & (value <= p["thres_neg"])
        output = positive.astype(np.int64) - negative.astype(np.int64)
        threshold = np.where(positive, p["thres_pos"], p["thres_neg"])
    fired = positive | negative
    hard_reset = np.where(positive, p["reset_v"], -p["reset_v"])
    value = np.where(fired & (p["reset_mode"] == 0), hard_reset, value)
    value = np.where(fired & (p["reset_mode"] == 1), _wrap(value - threshold), value)
    value = np.where(p["leak_multi_sequence"] == 1, _leak(value, p), value)
    return StepResult(
        voltage=value.astype(np.int32),
        output=output.astype(np.int16),
        potential=potential,
        positive=bool(np.any(positive)),
    )
