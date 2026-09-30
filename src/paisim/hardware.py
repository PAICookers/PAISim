"""Named PAICORE v2 hardware dimensions used by the simulator.

Values that paicorelib exposes are derived directly from its frame formats or
register limits.  The few composite SRAM dimensions are named here because
paicorelib exposes their field widths, but not the simulator's byte/word view.
"""

from math import isqrt
from typing import Final

from paicorelib.core_defs_v2 import OfflineCoreRegLimV2
from paicorelib.framelib.frame_defs import (
    FFV2,
    OfflineConfigFrame1FormatV2,
    OfflineConfigFrame3FormatV2,
    OfflineControlFrame1FormatV2,
    OfflineWorkFrame1FormatV2,
    OnlineWorkFrame1FormatV2,
)
from paicorelib.framelib.parser_v2 import CONFIG1_WORDS, LUT_ENTRIES, SRAM_WORDS
from paicorelib.hw_defs import HwParamsV2
from paicorelib.neuron_defs_v2 import OfflineNeuRegLimV2

BYTE_BITS: Final[int] = 8
BYTE_MASK: Final[int] = (1 << BYTE_BITS) - 1
FRAME_WORD_BITS: Final[int] = FFV2.FRAME_LENGTH
FRAME_TYPE_SHIFT: Final[int] = 2  # FrameHeader packs FrameType in its high two bits.

OFFLINE_INPUT_BITS: Final[int] = OfflineNeuRegLimV2.ADDR_AXON_MAX + 1
OFFLINE_INPUT_BYTES: Final[int] = OFFLINE_INPUT_BITS // BYTE_BITS
OFFLINE_WORK_TIMESTEP_COUNT: Final[int] = (
    OfflineWorkFrame1FormatV2.TIMESTEP_HIGH7_MASK + 1
) << OfflineWorkFrame1FormatV2.TIMESTEP_MASK.bit_length()
OFFLINE_WORK_TIMESTEP_LOW_BITS: Final[int] = (
    OfflineWorkFrame1FormatV2.TIMESTEP_MASK.bit_length()
)
# Current PAICORE 2.5 documented LCN domain. Encodings 8..15 remain valid
# wire values, but their execution behavior is undefined.
DOCUMENTED_LCN_MAX: Final[int] = 7
OFFLINE_INPUT_ROWS: Final[int] = OFFLINE_WORK_TIMESTEP_COUNT

ONLINE_INPUT_BITS: Final[int] = OnlineWorkFrame1FormatV2.AXON_ADDR_MASK + 1
ONLINE_INPUT_BYTES: Final[int] = ONLINE_INPUT_BITS // BYTE_BITS
ONLINE_INPUT_ROWS: Final[int] = OnlineWorkFrame1FormatV2.TIMESTEP_MASK + 1

OFFLINE_NEURON_ROWS: Final[int] = OfflineCoreRegLimV2.NEURON_NUMBER_MAX
OFFLINE_LUT_ENTRIES: Final[int] = LUT_ENTRIES
OFFLINE_LUT_WORDS: Final[int] = OFFLINE_LUT_ENTRIES
NEURON_SRAM_WORDS_PER_ROW: Final[int] = SRAM_WORDS
OFFLINE_NEURON_SRAM_WORDS: Final[int] = OFFLINE_NEURON_ROWS * NEURON_SRAM_WORDS_PER_ROW

OFFLINE_CONFIG_WORDS: Final[int] = CONFIG1_WORDS
# V2 online CONFIG spans Word1..Word5; paicorelib exposes the word formats,
# but no aggregate count.
ONLINE_CONFIG_WORDS: Final[int] = 5
SUPPORTED_DATA_WIDTHS: Final[tuple[int, ...]] = tuple(
    1 << code for code in range(OfflineConfigFrame1FormatV2.Word1.INPUT_WIDTH_MASK + 1)
)
OFFLINE_INPUT_FRAME_WORDS: Final[int] = OFFLINE_INPUT_BYTES // (
    FRAME_WORD_BITS // BYTE_BITS
)
ONLINE_INPUT_FRAME_WORDS: Final[int] = ONLINE_INPUT_BYTES // (
    FRAME_WORD_BITS // BYTE_BITS
)

WORK_DATA_MASK: Final[int] = OfflineWorkFrame1FormatV2.DATA_MASK
WORK_DATA_VALUES: Final[int] = WORK_DATA_MASK + 1
WORK_DATA_BITS: Final[int] = WORK_DATA_MASK.bit_length()
WORK_DATA_SIGN_BIT: Final[int] = 1 << (WORK_DATA_BITS - 1)
OFFLINE_AXON_VALUES: Final[int] = OfflineWorkFrame1FormatV2.AXON_ADDR_MASK + 1
OFFLINE_AXON_BITS: Final[int] = OfflineWorkFrame1FormatV2.AXON_ADDR_MASK.bit_length()
ONLINE_AXON_VALUES: Final[int] = OnlineWorkFrame1FormatV2.AXON_ADDR_MASK + 1

I32_MASK: Final[int] = (OfflineNeuRegLimV2.THRES_POS_MAX << 1) | 1
I32_BITS: Final[int] = I32_MASK.bit_length()
POTENTIAL_LANES: Final[int] = I32_BITS // BYTE_BITS
POTENTIAL_LANE_MASK: Final[int] = (1 << POTENTIAL_LANES) - 1
POTENTIAL_LANE_AXON_STRIDE: Final[int] = BYTE_BITS
POTENTIAL_LANE_SHIFT: Final[int] = POTENTIAL_LANE_AXON_STRIDE.bit_length() - 1
POTENTIAL_BASE_SPAN: Final[int] = POTENTIAL_LANE_AXON_STRIDE * (POTENTIAL_LANES - 1)
VOLTAGE_LANES_PER_U64: Final[int] = FRAME_WORD_BITS // I32_BITS
VOLTAGE_LANES_PER_SRAM_ROW: Final[int] = (
    NEURON_SRAM_WORDS_PER_ROW * VOLTAGE_LANES_PER_U64
)
DIRECT_POTENTIAL_BITS: Final[int] = I32_BITS
DIRECT_POTENTIAL_BYTES: Final[int] = DIRECT_POTENTIAL_BITS // BYTE_BITS
# RESET_MODE uses one reserved encoding; derive the usable range from its wire
# field instead of repeating the encoded limit in the execution layer.
RESET_MODE_MAX: Final[int] = OfflineConfigFrame3FormatV2.Full.Word4.RESET_MODE_MASK - 1
BINARY_MODE_MAX: Final[int] = (
    OfflineConfigFrame3FormatV2.Full.Word3.LATERAL_INHIBITION_MASK
)
CSC_INDEX_BITS: Final[int] = 16
CSC_INDEX_MASK: Final[int] = (1 << CSC_INDEX_BITS) - 1
CSC_INDEX_LANES_PER_U64: Final[int] = FRAME_WORD_BITS // CSC_INDEX_BITS
# Sparse weight words use a format-specific mix of data and index lanes; the
# frame library exposes the fields but not this aggregate packing table.
CSC_WEIGHT_LANES: Final[dict[int, int]] = {1: 7, 2: 7, 4: 6, 8: 5}
CSC_WEIGHT_INDEX_SHIFTS: Final[dict[int, int]] = {
    1: CSC_INDEX_BITS,
    2: CSC_INDEX_BITS,
    4: FRAME_WORD_BITS // 2,
    8: FRAME_WORD_BITS - CSC_INDEX_BITS,
}
ROUTE_FIELD_BITS: Final[int] = FFV2.GENERAL_CORE_XY_ADDR_MASK.bit_length()
ROUTE_FIELD_SIGN_BIT: Final[int] = 1 << (ROUTE_FIELD_BITS - 1)
ROUTE_FIELD_VALUE_MASK: Final[int] = ROUTE_FIELD_SIGN_BIT - 1
AXON_SKEW_SIGN_BIT: Final[int] = 1 << (
    OfflineConfigFrame1FormatV2.Word1.AXON_SKEW_MASK.bit_length() - 1
)
AXON_SKEW_MODULUS: Final[int] = OfflineConfigFrame1FormatV2.Word1.AXON_SKEW_MASK + 1

CORE_GRID_SIDE: Final[int] = isqrt(HwParamsV2.N_CORE_MAX_INCHIP)
CORE_COORD_MIN: Final[int] = 0
CORE_GRID_COORD_MAX: Final[int] = CORE_GRID_SIDE - 1
ONLINE_CORE_ROW_COUNT: Final[int] = (
    HwParamsV2.N_CORE_ONLINE + HwParamsV2.N_CORE_RV_CPU + CORE_GRID_SIDE - 1
) // CORE_GRID_SIDE
GLOBAL_SEND_ROOT_BIT: Final[int] = 1 << (
    OfflineCoreRegLimV2.GLOBAL_SEND_MAX.bit_length() - 1
)
GLOBAL_SEND_DIRECTION_BITS: Final[tuple[int, ...]] = tuple(
    range(GLOBAL_SEND_ROOT_BIT.bit_length() - 2, -1, -1)
)

# CONFIG package storage is expressed in payload words. paicorelib exposes the
# SRAM shapes, but not the package start-address units or alignment contract.
OFFLINE_NEURON_PACKAGE_ADDRESS_ROWS: Final[int] = 8  # CONFIG3 start unit: 8 rows.
OFFLINE_NEURON_PACKAGE_ALIGNMENT: Final[int] = NEURON_SRAM_WORDS_PER_ROW
OFFLINE_INPUT_PACKAGE_ALIGNMENT: Final[int] = OFFLINE_INPUT_FRAME_WORDS
ONLINE_NEURON_PACKAGE_ADDRESS_ROWS: Final[int] = 1  # Online CONFIG3 uses one row.
OFFLINE_PACKAGE_FACTORS: Final[tuple[int, int, int, int]] = (
    1,
    1,
    OFFLINE_NEURON_PACKAGE_ADDRESS_ROWS * NEURON_SRAM_WORDS_PER_ROW,
    OFFLINE_INPUT_FRAME_WORDS,
)
ONLINE_PACKAGE_FACTORS: Final[tuple[int, int, int, int]] = (
    1,
    1,
    ONLINE_NEURON_PACKAGE_ADDRESS_ROWS * NEURON_SRAM_WORDS_PER_ROW,
    ONLINE_INPUT_FRAME_WORDS,
)
OFFLINE_PACKAGE_CAPACITIES: Final[tuple[int, int, int, int]] = (
    OFFLINE_CONFIG_WORDS,
    OFFLINE_LUT_WORDS,
    OFFLINE_NEURON_SRAM_WORDS,
    OFFLINE_INPUT_ROWS * OFFLINE_INPUT_FRAME_WORDS,
)
OFFLINE_PACKAGE_ALIGNMENTS: Final[tuple[int, int, int, int]] = (
    1,
    1,
    OFFLINE_NEURON_PACKAGE_ALIGNMENT,
    OFFLINE_INPUT_PACKAGE_ALIGNMENT,
)
SYNC_STEP_MASK: Final[int] = OfflineControlFrame1FormatV2.N_TIMESTEP_MASK
