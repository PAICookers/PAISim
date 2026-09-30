"""PAICORE 2.5 word helpers; geometry comes from PAIlib.

Routes use signed magnitude. Work slots are physical eight-bit timestamps;
these helpers never infer model latency or tensor shape.
"""

import numpy as np
from numpy.typing import NDArray
from paicorelib.framelib.frame_defs import (
    FFV2,
    FrameHeader,
    OfflineWorkFrame1FormatV2,
)

from .hardware import (
    DOCUMENTED_LCN_MAX,
    OFFLINE_AXON_VALUES,
    OFFLINE_INPUT_BITS,
    OFFLINE_INPUT_ROWS,
    OFFLINE_WORK_TIMESTEP_COUNT,
    OFFLINE_WORK_TIMESTEP_LOW_BITS,
    WORK_DATA_VALUES,
)

ROUTE_MASK = (FFV2.GENERAL_CORE_ADDR_MASK << FFV2.GENERAL_CORE_ADDR_OFFSET) | (
    FFV2.GENERAL_COPY_ADDR_MASK << FFV2.GENERAL_COPY_ADDR_OFFSET
)
INPUT_ROWS = OFFLINE_INPUT_ROWS
INPUT_BITS = OFFLINE_INPUT_BITS


def route_bits(route: tuple[int, ...]) -> int:
    """Encode six signed-magnitude AER route fields into the route bit range.

    ``route`` is ``(offset_z, offset_x, offset_y, copies_z, copies_x,
    copies_y)``.  Each field must be an integer in ``[-31, 31]``; negative
    values set the protocol sign bit.  The returned word contains only route
    bits and can be ORed with a frame header/payload built by the caller.
    """
    if len(route) != 6:
        raise ValueError("route requires six signed fields")
    word = 0
    fields = (
        (FFV2.GENERAL_CORE_XY_ADDR_OFFSET, FFV2.GENERAL_CORE_XY_ADDR_MASK),
        (FFV2.GENERAL_CORE_X_ADDR_OFFSET, FFV2.GENERAL_CORE_X_ADDR_MASK),
        (FFV2.GENERAL_CORE_Y_ADDR_OFFSET, FFV2.GENERAL_CORE_Y_ADDR_MASK),
        (FFV2.GENERAL_COPY_XY_ADDR_OFFSET, FFV2.GENERAL_COPY_XY_ADDR_MASK),
        (FFV2.GENERAL_COPY_X_ADDR_OFFSET, FFV2.GENERAL_COPY_X_ADDR_MASK),
        (FFV2.GENERAL_COPY_Y_ADDR_OFFSET, FFV2.GENERAL_COPY_Y_ADDR_MASK),
    )
    for value, (shift, mask) in zip(route, fields, strict=True):
        limit = mask >> 1
        if not -limit <= value <= limit:
            raise ValueError("route field outside signed-magnitude six-bit range")
        sign = mask ^ limit
        word |= (abs(value) | (sign if value < 0 else 0)) << shift
    return word


def work_word(
    bits: int, slot: int, axon: int, data: int, *, voltage: bool = False
) -> int:
    """Pack one WORK word from route bits, an 8-bit slot, and an axon byte.

    ``data`` is the raw unsigned payload byte.  Set ``voltage=True`` only for
    one lane of a 32-bit potential response; ordinary DATA words leave it
    false.  ``bits`` should normally be the value returned by :func:`route_bits`.
    """
    if (
        not 0 <= slot < OFFLINE_WORK_TIMESTEP_COUNT
        or not 0 <= axon < OFFLINE_AXON_VALUES
        or not 0 <= data < WORK_DATA_VALUES
    ):
        raise ValueError("invalid slot, axon or data byte")
    header = FrameHeader.WORK_TYPE3 if voltage else FrameHeader.WORK_TYPE1
    return (
        header.value << FFV2.GENERAL_HEADER_OFFSET
        | (bits & ROUTE_MASK)
        | (slot >> OFFLINE_WORK_TIMESTEP_LOW_BITS)
        << OfflineWorkFrame1FormatV2.TIMESTEP_HIGH7_OFFSET
        | (slot & OfflineWorkFrame1FormatV2.TIMESTEP_MASK)
        << OfflineWorkFrame1FormatV2.TIMESTEP_OFFSET
        | (axon & OfflineWorkFrame1FormatV2.AXON_ADDR_MASK)
        << OfflineWorkFrame1FormatV2.AXON_ADDR_OFFSET
        | (data & OfflineWorkFrame1FormatV2.DATA_MASK)
        << OfflineWorkFrame1FormatV2.DATA_OFFSET
    )


def work_fields(word: int) -> tuple[int, int, int, bool]:
    """Decode a WORK word as ``(slot, axon, byte, voltage)``.

    The function reads protocol fields only; it does not validate that ``word``
    is routed to a configured core or that a voltage lane has matching peers.
    """
    return (
        (
            (word >> OfflineWorkFrame1FormatV2.TIMESTEP_HIGH7_OFFSET)
            & OfflineWorkFrame1FormatV2.TIMESTEP_HIGH7_MASK
        )
        << OFFLINE_WORK_TIMESTEP_LOW_BITS
        | (word >> OfflineWorkFrame1FormatV2.TIMESTEP_OFFSET)
        & OfflineWorkFrame1FormatV2.TIMESTEP_MASK,
        (word >> OfflineWorkFrame1FormatV2.AXON_ADDR_OFFSET)
        & OfflineWorkFrame1FormatV2.AXON_ADDR_MASK,
        (word >> OfflineWorkFrame1FormatV2.DATA_OFFSET)
        & OfflineWorkFrame1FormatV2.DATA_MASK,
        FrameHeader((word >> FFV2.GENERAL_HEADER_OFFSET) & FFV2.GENERAL_HEADER_MASK)
        in (FrameHeader.WORK_TYPE3, FrameHeader.WORK_TYPE4),
    )


def lcn_factor(code: int) -> int:
    """Return physical rows represented by an exponent-coded LCN value.

    PAICORE encodes LCN as ``0..DOCUMENTED_LCN_MAX`` and the corresponding
    fan-in factor is ``1 << code``. This helper validates that range before
    calculating it.
    """
    if not 0 <= code <= DOCUMENTED_LCN_MAX:
        raise ValueError(f"LCN code must be in range [0, {DOCUMENTED_LCN_MAX}]")
    return 1 << code


def resolve_work_timestep(timestep: int, target_lcn: int, tick_relative: int) -> int:
    """Resolve one logical timestep to an 8-bit physical WORK timestamp.

    ``target_lcn`` is the exponent encoded by ``LCN_EX`` and ``tick_relative``
    selects a row inside that LCN group.  The logical timestep wraps to the
    remaining eight-bit timestamp width; this helper does not model an epoch.
    """
    if timestep < 0:
        raise ValueError("timestep must be nonnegative")
    factor = lcn_factor(target_lcn)
    if not 0 <= tick_relative < factor:
        raise ValueError("tick_relative is outside the target LCN group")
    wrapped = timestep % (INPUT_ROWS // factor)
    return wrapped * factor + tick_relative


def input_rows(timestep: int, input_lcn: int) -> NDArray[np.intp]:
    """Return the physical input rows consumed by one logical timestep.

    The result is a fixed-dtype NumPy array of row indices in ``0..255``.  Row
    selection wraps at the physical input-buffer boundary and does not mutate
    any buffer.
    """
    if timestep < 0:
        raise ValueError("timestep must be nonnegative")
    factor = lcn_factor(input_lcn)
    base = (timestep * factor) % INPUT_ROWS
    return (base + np.arange(factor, dtype=np.intp)) % INPUT_ROWS
