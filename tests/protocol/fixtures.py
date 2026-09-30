"""Small hand-authored v2 frame fixtures for public-API tests."""

from paisim import CoreAddr, Port
from paisim.frames import route_bits, work_word


def route(source: Port | CoreAddr, target: CoreAddr) -> int:
    addr = source.addr if isinstance(source, Port) else source
    gx0 = addr.chip.x * 9 + addr.core.x
    gy0 = addr.chip.y * 9 + addr.core.y
    gx1 = target.chip.x * 9 + target.core.x
    gy1 = target.chip.y * 9 + target.core.y
    return route_bits((0, gx1 - gx0, gy1 - gy0, 0, 0, 0))


def offline_config(
    ingress: Port,
    target: CoreAddr,
    *,
    tick_start: int = 1,
    tick_duration: int = 0,
    tick_initial: int = 0,
    send: int = 64,
    receive: int = 0,
    threshold: int = 1,
    initial_v: int = 0,
    thread: int = 0,
) -> list[int]:
    bits = route(ingress, target)
    test_x = 0 if target.core.x == 0 else 32 + target.core.x
    test_y = 32 + target.core.y
    registers = [
        3 << 57 | 3 << 51 | 2 << 14 | test_x << 2 | test_y >> 4,
        (test_y & 15) << 60 | send << 53 | receive << 46 | thread << 36,
        tick_start << 48 | tick_duration << 16 | tick_initial,
    ]
    neuron = [
        1 << 32 | 8 << 47 | 8 << 35,
        test_x << 35 | test_y << 29,
        threshold << 44 | (initial_v & 0xFFF),
        1 << 45 | (0x80000000 << 12),
    ]
    return [
        bits | 3,
        *registers,
        2 << 60 | bits | 4,
        *neuron,
        2 << 60 | bits | 1 << 14 | 2,
        1,
        0,
    ]


def work(ingress: Port, target: CoreAddr, value: int, *, slot: int = 0) -> int:
    return work_word(route(ingress, target), slot, 0, value)


def control(ingress: Port, target: CoreAddr, kind: int, value: int = 0) -> int:
    return kind << 60 | route(ingress, target) | value
