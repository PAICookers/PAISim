"""Independently encoded CONFIG/WORK/SYNC wire-level numeric regressions."""

from typing import Literal

import numpy as np
import pytest
from paicorelib.framelib.frame_defs import FFV2, FrameHeader

from paisim import ChipCoord, CoreAddr, CoreCoord, Port, Simulator, SingleBoard
from paisim.frames import work_fields, work_word

CPU = Port(ChipCoord(0, 0))
ROOT = CoreAddr(CPU.chip, CoreCoord(0, 2))
_DEST = 2 << 42
_RETURN = 34 << 29
_WEIGHT = 8 << 47 | 8 << 35
_COMPLETE = 0xE000000000000000
_SYNC = 0xC000000000000000 | _DEST | 1


def _neuron(*, potential: bool = False, compressed: bool = False) -> list[int]:
    return [
        _WEIGHT | 1 << 32 | potential << 34,
        _RETURN,
        1000 << 44 | 1 << 42 | compressed << 12,
        2 << 62 | ((-1000) & 0xFFFFFFFF) << 12,
    ]


def _config(
    neurons: list[int],
    weight: int,
    *,
    extra: int = 0,
    weight_width: int = 3,
    output_width: int = 3,
    lut: list[int] | None = None,
) -> list[int]:
    words = [
        _DEST | 3,
        3 << 57
        | output_width << 54
        | weight_width << 51
        | (len(neurons) // 2) << 14
        | 2
        | extra,
        2 << 60 | 64 << 53,
        1 << 48,
        2 << 60 | _DEST | len(neurons),
        *neurons,
        2 << 60 | _DEST | 1 << 14 | 2,
        weight,
        0,
    ]
    if lut is not None:
        words.extend([1 << 60 | _DEST | 256, *lut])
    return words


def _work(value: int, *, axon: int = 0) -> int:
    return 8 << 60 | _DEST | axon << 8 | value


def _voltage(value: int, *, axon: int = 0) -> list[int]:
    return [
        10 << 60 | (axon + lane * 8) << 8 | ((value >> (lane * 8)) & 255)
        for lane in range(4)
    ]


def _execute(sim: Simulator, words: list[int]) -> list[int]:
    sim.feed(np.asarray(words, dtype=np.uint64), ingress=CPU)
    sim.run()
    return sim.drain_output().tolist()


def test_voltage_work_frame_keeps_voltage_type_for_high_slot() -> None:
    word = work_word(0, 255, 0, 0xA5, voltage=True)
    assert work_fields(word) == (255, 0, 0xA5, True)


@pytest.mark.parametrize("header", tuple(FrameHeader), ids=lambda item: item.name)
def test_all_frame_headers_round_trip_through_v2_header_field(
    header: FrameHeader,
) -> None:
    word = header.value << FFV2.GENERAL_HEADER_OFFSET
    assert (
        FrameHeader((word >> FFV2.GENERAL_HEADER_OFFSET) & FFV2.GENERAL_HEADER_MASK)
        is header
    )


@pytest.mark.parametrize(
    "width", [0, 1, 2, 3], ids=["1-bit", "2-bit", "4-bit", "8-bit"]
)
def test_signed_weights_are_decoded_through_frames(width: int) -> None:
    sim = Simulator(SingleBoard())
    words = _config(
        _neuron(potential=True),
        (1 << (1 << width)) - 1,
        weight_width=width,
        extra=1 << 53,
    )
    assert _execute(sim, [*words, _work(3), _SYNC]) == [*_voltage(-3), _COMPLETE]
    assert _execute(sim, [6 << 60 | _DEST | 1 << 23 | 2]) == [
        6 << 60 | 2,
        words[5] | 0xFFFFFFFD,
        words[6],
    ]


@pytest.mark.parametrize(
    "width", [0, 1, 2, 3], ids=["1-bit", "2-bit", "4-bit", "8-bit"]
)
@pytest.mark.parametrize("value", [2, 127, 128, 255])
def test_ann_lut_address_and_wire_precision(width: int, value: int) -> None:
    block = 256 >> (1 << width)
    lut = [(address << 8) | (address // block) for address in range(256)]
    sim = Simulator(SingleBoard())
    assert _execute(
        sim,
        [
            *_config(
                _neuron(), 1, extra=1 << 63 | 1 << 60, output_width=width, lut=lut
            ),
            _work(value),
            _SYNC,
        ],
    ) == [8 << 60 | (value // block), _COMPLETE]
    assert sim.snapshot()["cores"][ROOT]["voltage"].tolist() == [value]


@pytest.mark.parametrize("storage", ["full", "half", "fold"])
@pytest.mark.parametrize("compressed", [False, True], ids=["dense", "csc"])
def test_offline_layouts_compute_identical_raw_frames(
    storage: Literal["full", "half", "fold"], compressed: bool
) -> None:
    first = _neuron(potential=True, compressed=compressed)
    if storage == "fold":
        first[0] |= 1 << 33
        neurons = [*first, 32 << 29 | 2, 1 << 53 | 1 << 42 | 2 << 31 | 2, 0, 0]
    else:
        second = _neuron(potential=True, compressed=compressed)
        second[0] |= 8 << 59
        second[1] |= 32 << 47
        if storage == "half":
            second[0] &= ~(1 << 32)
            second = second[:2]
        neurons = first + second
    sim = Simulator(SingleBoard())
    assert _execute(
        sim,
        [*_config(neurons, 2), _work(3), _work(4, axon=8), _SYNC],
    ) == [*_voltage(6), *_voltage(8, axon=32), _COMPLETE]
    assert sim.snapshot()["cores"][ROOT]["voltage"].tolist() == [6, 8]
