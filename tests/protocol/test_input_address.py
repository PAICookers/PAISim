"""Stage 1 input-address and LCN invariants.

These tests deliberately keep LCN as the exponent-coded value used by the
wire format.  A producer's target LCN is independent from a consumer's LCN;
the decoder must preserve both values instead of imposing a compiler-level
matching rule.
"""

import numpy as np
import pytest
from numpy.typing import NDArray
from paicorelib.core_defs import LCN_EX

from paisim.engine.decode import decode_core
from paisim.engine.input_buffer import InputWindow
from paisim.frames import input_rows, resolve_work_timestep


@pytest.mark.parametrize("member", tuple(LCN_EX), ids=lambda item: item.name)
def test_lcn_enum_value_is_an_exponent(member: LCN_EX) -> None:
    """LCN enum values select a power-of-two access factor."""
    assert member.value >= 0
    assert 1 << member.value in (1, 2, 4, 8, 16, 32, 64, 128)


@pytest.mark.parametrize(
    ("input_lcn", "target_lcn"),
    [
        (LCN_EX.LCN_1X, LCN_EX.LCN_2X),
        (LCN_EX.LCN_2X, LCN_EX.LCN_1X),
        (LCN_EX.LCN_1X, LCN_EX.LCN_128X),
    ],
    ids=["1x-to-2x", "2x-to-1x", "1x-to-128x"],
)
def test_target_lcn_mismatch_is_preserved_and_legal(
    input_lcn: LCN_EX, target_lcn: LCN_EX
) -> None:
    """A legal frame is not rejected by a consumer-side LCN comparison."""
    decoded = decode_core(
        {
            "lcn": input_lcn.value,
            "target_lcn": target_lcn.value,
        },
        np.zeros((4096, 2), dtype=np.uint64),
        np.zeros(256, dtype=np.uint64),
    )
    assert decoded.config["lcn"] == input_lcn.value
    assert decoded.config["target_lcn"] == target_lcn.value


@pytest.mark.parametrize(
    ("timestep", "target_code", "tick_relative"),
    [(0, 0, 0), (3, 1, 1), (17, 2, 3), (255, 7, 127)],
)
def test_work_timestep_encoding_is_unambiguous(
    timestep: int, target_code: int, tick_relative: int
) -> None:
    """The frame address has a target-LCN group and a local offset."""
    assert 0 <= tick_relative < 1 << target_code
    encoded = resolve_work_timestep(timestep, target_code, tick_relative)
    period = 256 // (1 << target_code)
    assert encoded >> target_code == timestep % period
    assert encoded & ((1 << target_code) - 1) == tick_relative


@pytest.mark.parametrize("lcn_code", range(4), ids=lambda code: f"{1 << code}x")
@pytest.mark.parametrize("timestep", [0, 1, 63, 255])
def test_input_rows_group_one_logical_tick_by_input_lcn(
    timestep: int, lcn_code: int
) -> None:
    """A logical tick reads a contiguous power-of-two row group."""
    factor = 1 << lcn_code
    expected: NDArray[np.intp] = (
        timestep * factor + np.arange(factor, dtype=np.intp)
    ) % 256
    np.testing.assert_array_equal(input_rows(timestep, lcn_code), expected)


def test_input_window_keeps_write_and_read_lcn_independent() -> None:
    """A target-LCN write remains readable at its physical work row."""
    raw: NDArray[np.uint8] = np.zeros((256, 64), dtype=np.uint8)
    window = InputWindow(raw)
    window.write_frame(
        timestep=2,
        target_lcn=LCN_EX.LCN_2X.value,
        tick_relative=1,
        axon=0,
        data=0xA5,
        width=8,
    )
    # target LCN 2X writes work row 2*2+1=5; input LCN is selected only when
    # a consumer reads a logical tick, so the write itself is never rejected.
    np.testing.assert_array_equal(window.packed(5, LCN_EX.LCN_1X.value)[0], 0xA5)
    window.clear(5, LCN_EX.LCN_1X.value)
    assert raw[5, 0] == 0
