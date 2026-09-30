"""Decode offline SRAM into complete execution arrays, without UI previews.

Field layouts follow PAICORE 2.5 OfflineConfigFrame3FormatV2. Fold strides
are transition increments (Y fastest), as emitted by backendv2/fold_neu.py;
they are not Cartesian strides. Raw SRAM remains owned by the caller.
"""

from collections.abc import Mapping
from typing import cast

import numpy as np
from numpy.typing import NDArray
from paicorelib.framelib.frame_defs import (
    OfflineConfigFrame3FormatV2 as F3,
)

from ..hardware import (
    BYTE_BITS,
    CSC_INDEX_BITS,
    CSC_INDEX_LANES_PER_U64,
    CSC_INDEX_MASK,
    CSC_WEIGHT_INDEX_SHIFTS,
    CSC_WEIGHT_LANES,
    DIRECT_POTENTIAL_BITS,
    FRAME_WORD_BITS,
    I32_BITS,
    I32_MASK,
    NEURON_SRAM_WORDS_PER_ROW,
    OFFLINE_INPUT_BITS,
    OFFLINE_LUT_ENTRIES,
    OFFLINE_NEURON_ROWS,
    OFFLINE_WORK_TIMESTEP_COUNT,
    RESET_MODE_MAX,
    ROUTE_FIELD_SIGN_BIT,
    ROUTE_FIELD_VALUE_MASK,
    SUPPORTED_DATA_WIDTHS,
    VOLTAGE_LANES_PER_SRAM_ROW,
    VOLTAGE_LANES_PER_U64,
    WORK_DATA_SIGN_BIT,
)
from .model import (
    CoreConfig,
    DecodedCore,
    FoldLayout,
    NeuronParameters,
    SynapseGroup,
    core_config,
)
from .numeric import validate_configuration


def _field(word: int, layout: type, name: str) -> int:
    offset = cast(int, getattr(layout, f"{name}_OFFSET"))
    mask = cast(int, getattr(layout, f"{name}_MASK"))
    return (word >> offset) & mask


def _signed(value: int, width: int) -> int:
    return (value ^ (1 << (width - 1))) - (1 << (width - 1))


def _field_width(layout: type, name: str) -> int:
    return cast(int, getattr(layout, f"{name}_MASK")).bit_length()


def _parameters(low: int, high: int, accelerated: bool) -> tuple[dict[str, int], int]:
    """Read full Part2; following half records inherit this entire table."""
    w3, w4 = F3.Full.Word3, F3.Full.Word4
    specs = (
        ("lateral_inhi", "LATERAL_INHIBITION", False),
        ("leak_multi_sequence", "LEAK_MULTI_SEQUENCE", False),
        ("leak_multi_input", "LEAK_MULTI_INPUT", False),
        ("leak_multi_mode", "LEAK_MULTI_MODE", False),
        ("leak_add_mode", "LEAK_ADD_MODE", False),
        ("leak_tau", "LEAK_TAU", True),
        ("leak_v", "LEAK_V", True),
        ("init_v", "VJT_INITIAL", True),
    )
    values = {}
    for key, field, signed in specs:
        value = _field(low, w3, field)
        values[key] = _signed(value, _field_width(w3, field)) if signed else value
    for key, field, width in (
        ("reset_mode", "RESET_MODE", False),
        ("reset_v", "RESET_V", True),
        ("thres_neg_mode", "THRESHOLD_NEG_MODE", False),
        ("thres_pos_mode", "THRESHOLD_POS_MODE", False),
        ("thres_neg", "THRESHOLD_NEG", True),
    ):
        value = _field(high, w4, field)
        values[key] = _signed(value, _field_width(w4, field)) if width else value
    threshold_pos_low_bits = _field_width(w3, "THRESHOLD_POS_LOW20")
    threshold_pos_bits = threshold_pos_low_bits + _field_width(
        w4, "THRESHOLD_POS_HIGH12"
    )
    values["thres_pos"] = _signed(
        _field(low, w3, "THRESHOLD_POS_LOW20")
        | (_field(high, w4, "THRESHOLD_POS_HIGH12") << threshold_pos_low_bits),
        threshold_pos_bits,
    )
    compress = _field(low, w3, "WEIGHT_COMPRESS")
    if accelerated and compress:
        # Compiler uses this field as a weight-address alias, not initial voltage.
        # Accelerated pointer progression still belongs to the execution engine.
        values["init_v"] = 0
    if values["reset_mode"] > RESET_MODE_MAX:
        raise ValueError("reset_mode is reserved")
    return values, compress


def _weights(
    sram: NDArray[np.uint64],
    start: int,
    end: int,
    width: int,
    input_width: int,
    signed: bool,
    sparse: bool,
) -> tuple[NDArray[np.int64], NDArray[np.int16]]:
    """Decode every nonzero coefficient; CSC indices are absolute bit offsets."""
    if end < start:
        return np.empty(0, np.int64), np.empty(0, np.int16)
    words = sram[start : end + 1]
    mask = (1 << width) - 1
    if sparse:
        count = CSC_WEIGHT_LANES[width]
        shift = CSC_WEIGHT_INDEX_SHIFTS[width]
        lo, hi = words[:, 0:1], words[:, 1:2]
        raw = (lo >> (np.arange(count, dtype=np.uint64) * width)) & mask
        indices = (
            np.concatenate(
                (
                    (
                        lo
                        >> (
                            shift
                            + np.arange(
                                count - CSC_INDEX_LANES_PER_U64,
                                dtype=np.uint64,
                            )
                            * CSC_INDEX_BITS
                        )
                    )
                    & CSC_INDEX_MASK,
                    (
                        hi
                        >> (
                            np.arange(
                                CSC_INDEX_LANES_PER_U64,
                                dtype=np.uint64,
                            )
                            * CSC_INDEX_BITS
                        )
                    )
                    & CSC_INDEX_MASK,
                ),
                axis=1,
            )
            .ravel()
            .astype(np.int64)
        )
        raw = raw.ravel().astype(np.int16)
    else:
        raw = (
            (
                (
                    words.reshape(-1, 1)
                    >> (np.arange(FRAME_WORD_BITS // width, dtype=np.uint64) * width)
                )
                & mask
            )
            .ravel()
            .astype(np.int16)
        )
        indices = np.arange(raw.size, dtype=np.int64) * input_width
    # Zero CSC lanes are storage padding, not input accesses. Dropping zero
    # dense lanes likewise avoids accessing padding beyond configured fan-in.
    used = raw != 0
    indices, raw = indices[used], raw[used]
    if signed:
        raw = ((raw ^ (1 << (width - 1))) - (1 << (width - 1))).astype(np.int16)
    if np.any(indices % input_width):
        raise ValueError("nonzero CSC index is not aligned to input element width")
    return indices // input_width, raw


def _fold_offsets(
    number: int, ranges: tuple[int, int, int], skews: tuple[int, int, int]
) -> NDArray[np.int64]:
    """Expand transition increments, including the base neuron at offset zero."""
    ry, rx, rxy = ranges
    if min(ranges) < 1 or number > ry * rx * rxy:
        raise ValueError("fold count exceeds its nonzero fold ranges")
    steps = np.arange(1, number, dtype=np.int64)
    increments = np.full(number - 1, skews[0], dtype=np.int64)
    increments[steps % ry == 0] = skews[1]
    increments[steps % (ry * rx) == 0] = skews[2]
    return np.concatenate((np.zeros(1, np.int64), np.cumsum(increments)))


def decode_core(
    config: CoreConfig | Mapping[str, int],
    neuron_sram: NDArray[np.uint64],
    lut_sram: NDArray[np.uint64],
) -> DecodedCore:
    """Decode full/half/folded neurons and shared dense/CSC weight vectors.

    ``neuron_number`` counts 128-bit parameter rows, not logical neurons.
    A half neuron requires a preceding full neuron. Returned arrays are owned
    snapshots; neither raw input SRAM is mutated or retained by reference.
    """
    registers = core_config(config)
    if (
        neuron_sram.shape != (OFFLINE_NEURON_ROWS, NEURON_SRAM_WORDS_PER_ROW)
        or neuron_sram.dtype != np.uint64
    ):
        raise ValueError(
            f"neuron_sram must be uint64[{OFFLINE_NEURON_ROWS}, "
            f"{NEURON_SRAM_WORDS_PER_ROW}]"
        )
    if lut_sram.shape != (OFFLINE_LUT_ENTRIES,) or lut_sram.dtype != np.uint64:
        raise ValueError(f"lut_sram must be uint64[{OFFLINE_LUT_ENTRIES}]")
    rows = registers["neuron_number"]
    if not 0 <= rows <= OFFLINE_NEURON_ROWS:
        raise ValueError("neuron_number is outside SRAM")
    iw_raw, ww_raw = registers["input_width"], registers["weight_width"]
    if iw_raw >= len(SUPPORTED_DATA_WIDTHS) or ww_raw >= len(SUPPORTED_DATA_WIDTHS):
        raise ValueError("offline input/weight width must encode 1, 2, 4 or 8 bits")
    iw, ww = SUPPORTED_DATA_WIDTHS[iw_raw], SUPPORTED_DATA_WIDTHS[ww_raw]
    direct = bool(registers["add_potential"])
    if direct:
        iw = ww = 1
    parameters, compression = None, 0
    all_parameters: list[dict[str, int]] = []
    voltages: list[int] = []
    routes: list[list[int]] = []
    axons: list[int] = []
    ticks: list[int] = []
    types: list[int] = []
    locations: list[tuple[int, int]] = []
    parameter_locations: list[int] = []
    layouts: list[int] = []
    fold_layouts: list[FoldLayout] = []
    grouped: dict[tuple[int, int, int], tuple[list[int], list[int]]] = {}
    parameter_row: int | None = None
    row = 0
    while row < rows:
        low, high = map(int, neuron_sram[row])
        base_row = row
        w1, w2 = F3.Full.Word1, F3.Full.Word2
        full = _field(low, w1, "NEURON_TYPE")
        folded = _field(low, w1, "FOLD_TYPE")
        start = _field(low, w1, "WEIGHT_ADDRESS_START")
        end = _field(low, w1, "WEIGHT_ADDRESS_END")
        skew = _field(low, w1, "WEIGHT_SKEW_LOW5") | (
            _field(high, w2, "WEIGHT_SKEW_HIGH11")
            << F3.Full.Word1.WEIGHT_SKEW_LOW5_MASK.bit_length()
        )
        row += 1
        if full:
            if row >= rows:
                raise ValueError("full neuron Part2 exceeds neuron_number")
            parameter_row = row
            parameters, compression = _parameters(
                int(neuron_sram[row, 0]),
                int(neuron_sram[row, 1]),
                bool(registers["csc_accelerate"]),
            )
            row += 1
        if parameters is None:
            raise ValueError("half neuron has no preceding full-neuron parameters")
        assert parameter_row is not None
        count, offsets, destinations = 1, np.zeros(1, np.int64), np.zeros(1, np.int64)
        v = [_signed(_field(low, w1, "VJT"), I32_BITS)]
        loc = [(base_row, 0)]
        if folded:
            if row >= rows:
                raise ValueError("fold attributes exceed neuron_number")
            fl, fh = map(int, neuron_sram[row])
            f1, f2 = F3.Fold.Word1, F3.Fold.Word2
            count = _field(fl, f1, "FOLD_NUMBER")
            storage = (
                count - 1 + VOLTAGE_LANES_PER_SRAM_ROW - 1
            ) // VOLTAGE_LANES_PER_SRAM_ROW
            if count < 1 or row + 1 + storage > rows:
                raise ValueError("fold voltage storage exceeds neuron_number")
            ranges = (
                _field(fh, f2, "FOLD_RANGE_Y"),
                _field(fh, f2, "FOLD_RANGE_X"),
                _field(fh, f2, "FOLD_RANGE_XY"),
            )
            skews = (
                _field(fl, f1, "FOLD_SKEW_Y_LOW2")
                | (
                    _field(fh, f2, "FOLD_SKEW_Y_HIGH9")
                    << F3.Fold.Word1.FOLD_SKEW_Y_LOW2_MASK.bit_length()
                ),
                _field(fh, f2, "FOLD_SKEW_X"),
                _field(fh, f2, "FOLD_SKEW_XY"),
            )
            offsets = _fold_offsets(count, ranges, skews)
            fold_layouts.append(
                FoldLayout(
                    count=count,
                    ranges=ranges,
                    skews=skews,
                    offsets=tuple(int(value) for value in offsets),
                )
            )
            destinations = _fold_offsets(
                count,
                ranges,
                (
                    _field(fl, f1, "FOLD_AXON_Y"),
                    _field(fl, f1, "FOLD_AXON_X"),
                    _field(fl, f1, "FOLD_AXON_XY"),
                ),
            )
            for index in range(count - 1):
                vr, lane = (
                    row + 1 + index // VOLTAGE_LANES_PER_SRAM_ROW,
                    index % VOLTAGE_LANES_PER_SRAM_ROW,
                )
                word = int(neuron_sram[vr, lane // VOLTAGE_LANES_PER_U64])
                v.append(
                    _signed(
                        (word >> (I32_BITS * (lane % VOLTAGE_LANES_PER_U64)))
                        & I32_MASK,
                        I32_BITS,
                    )
                )
                loc.append((vr, lane))
            row += 1 + storage
        offsets += skew
        divisor = DIRECT_POTENTIAL_BITS if direct else iw
        if np.any(offsets % divisor):
            raise ValueError("weight skew is not aligned to input element width")
        route = []
        for field in ("CORE_XY", "CORE_X", "CORE_Y", "COPY_XY", "COPY_X", "COPY_Y"):
            raw = _field(high, w2, f"ADDR_{field}")
            route.append(
                -(raw & ROUTE_FIELD_VALUE_MASK) if raw & ROUTE_FIELD_SIGN_BIT else raw
            )
        address = _field(high, w2, "ADDR_AXON") + destinations
        tick = _field(high, w2, "TICK_RELATIVE") + address // OFFLINE_INPUT_BITS
        first = len(voltages)
        all_parameters.extend([parameters] * count)
        voltages.extend(v)
        routes.extend([route] * count)
        axons.extend(address % OFFLINE_INPUT_BITS)
        ticks.extend(tick % OFFLINE_WORK_TIMESTEP_COUNT)
        types.extend([_field(low, w1, "OUTPUT_TYPE")] * count)
        locations.extend(loc)
        parameter_locations.extend([parameter_row] * count)
        layouts.extend([2 if folded else 0 if full else 1] * count)
        key = (start, end, compression)
        posts, input_offsets = grouped.setdefault(key, ([], []))
        posts.extend(range(first, first + count))
        input_offsets.extend(offsets // divisor)
    synapses = []
    for (start, end, compression), (posts, group_offsets) in grouped.items():
        indices, weights = _weights(
            neuron_sram,
            start,
            end,
            ww,
            iw,
            bool(registers["weight_sign"]) and not direct,
            bool(compression),
        )
        if direct and indices.size:
            # DIRECT_ADD selects complete 32-bit potentials through bit masks.
            # Arbitrary partial masks have no scalar weighted-sum equivalent.
            elements, counts = np.unique(
                indices // DIRECT_POTENTIAL_BITS, return_counts=True
            )
            if (
                np.any(counts != DIRECT_POTENTIAL_BITS)
                or np.unique(indices).size != indices.size
            ):
                raise ValueError("DIRECT_ADD requires complete 32-bit connection masks")
            indices, weights = elements, np.ones(elements.size, np.int16)
        synapses.append(
            SynapseGroup(
                np.asarray(posts, np.int64),
                indices,
                weights,
                np.asarray(group_offsets, np.int64),
            )
        )
    thresholds = (
        ((lut_sram >> BYTE_BITS) & I32_MASK).astype(np.uint32).view(np.int32).copy()
    )
    values = (lut_sram & ((1 << BYTE_BITS) - 1)).astype(np.int16)
    if registers["output_sign"]:
        values = (values ^ WORK_DATA_SIGN_BIT) - WORK_DATA_SIGN_BIT

    def column(name: str) -> NDArray[np.int64]:
        return np.asarray([p[name] for p in all_parameters], np.int64)

    decoded_parameters = NeuronParameters(
        reset_mode=column("reset_mode"),
        reset_v=column("reset_v"),
        thres_neg_mode=column("thres_neg_mode"),
        thres_pos_mode=column("thres_pos_mode"),
        thres_neg=column("thres_neg"),
        thres_pos=column("thres_pos"),
        lateral_inhi=column("lateral_inhi"),
        leak_multi_sequence=column("leak_multi_sequence"),
        leak_multi_input=column("leak_multi_input"),
        leak_multi_mode=column("leak_multi_mode"),
        leak_add_mode=column("leak_add_mode"),
        leak_tau=column("leak_tau"),
        leak_v=column("leak_v"),
        init_v=column("init_v"),
    )
    decoded = DecodedCore(
        config=registers,
        voltage=np.asarray(voltages, np.int32),
        parameters=decoded_parameters,
        routes=np.asarray(routes, np.int64).reshape(-1, 6),
        axons=np.asarray(axons, np.int64),
        ticks=np.asarray(ticks, np.int64),
        output_types=np.asarray(types, np.int8),
        voltage_locations=np.asarray(locations, np.int64).reshape(-1, 2),
        parameter_locations=np.asarray(parameter_locations, np.int64),
        layouts=np.asarray(layouts, np.uint8),
        synapses=tuple(synapses),
        lut_thresholds=thresholds,
        lut_values=values,
        folds=tuple(fold_layouts),
    )
    validate_configuration(decoded)
    return decoded
