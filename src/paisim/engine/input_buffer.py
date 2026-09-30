"""Logical access to the PAICORE offline input SRAM."""

from typing import Final

import numpy as np
from numpy.typing import NDArray

from ..frames import INPUT_BITS, INPUT_ROWS, input_rows, resolve_work_timestep
from ..hardware import (
    BYTE_BITS,
    BYTE_MASK,
    DIRECT_POTENTIAL_BYTES,
    OFFLINE_INPUT_BYTES,
    SUPPORTED_DATA_WIDTHS,
)

ByteArray = NDArray[np.uint8]
INPUT_BYTES: Final[int] = OFFLINE_INPUT_BYTES


class InputWindow:
    """View over one 256-row by 512-bit input buffer.

    The view operates on the core's live ``uint8`` backing array.  Reads return
    copies; ``write_frame`` and ``clear`` intentionally mutate that array.
    """

    def __init__(self, raw: ByteArray) -> None:
        """Wrap a ``(256, 64)`` uint8 input SRAM array without copying it."""
        if raw.shape != (INPUT_ROWS, INPUT_BYTES) or raw.dtype != np.uint8:
            raise ValueError("input SRAM must have shape (256, 64) and dtype uint8")
        self._raw = raw

    def values(
        self, timestep: int, lcn: int, width: int, signed: bool, direct: bool
    ) -> tuple[NDArray[np.int32], NDArray[np.intp]]:
        """Decode one logical tick and return ``(values, consumed_rows)``.

        ``width`` selects 1/2/4/8-bit packed fields; ``signed`` sign-extends
        those fields.  ``direct=True`` interprets the selected bytes as little-
        endian int32 values.  The returned values use signed int32 storage and
        the row array uses NumPy's platform index dtype.
        """
        rows = self.rows(timestep, lcn)
        packed = self._raw[rows].reshape(-1)
        if direct:
            if packed.size % DIRECT_POTENTIAL_BYTES:
                raise ValueError("direct-potential input is not int32 aligned")
            return packed.view(f"<i{DIRECT_POTENTIAL_BYTES}"), rows
        shifts = np.arange(0, BYTE_BITS, width, dtype=np.uint8)
        decoded = ((packed[:, None] >> shifts) & ((1 << width) - 1)).reshape(-1)
        values = decoded.astype(np.int16)
        if signed:
            values[values >= (1 << (width - 1))] -= 1 << width
        return values.astype(np.int32), rows

    def rows(self, timestep: int, lcn: int) -> NDArray[np.intp]:
        """Return physical row indices consumed by ``timestep`` and ``lcn``."""
        return input_rows(timestep, lcn)

    def packed(self, timestep: int, lcn: int) -> ByteArray:
        """Return a copy of packed bytes consumed by one logical tick."""
        return self._raw[self.rows(timestep, lcn)].reshape(-1).copy()

    def clear(self, timestep: int, lcn: int) -> None:
        """Zero all rows consumed by one logical tick in the live buffer."""
        self._raw[self.rows(timestep, lcn)] = 0

    def write_frame(
        self,
        *,
        timestep: int,
        target_lcn: int,
        tick_relative: int,
        axon: int,
        data: int,
        width: int,
    ) -> None:
        """Write one aligned work value into the target LCN timestamp group.

        ``axon`` and ``data`` are expressed in the configured field width;
        writes crossing byte boundaries or the 512-bit input row are rejected.
        """
        slot = resolve_work_timestep(timestep, target_lcn, tick_relative)
        if not 0 <= axon < INPUT_BITS or width not in SUPPORTED_DATA_WIDTHS:
            raise ValueError("invalid input-buffer work address")
        if axon % width or axon + width > INPUT_BITS or data < 0 or data >= 1 << width:
            raise ValueError("unaligned or out-of-range input-buffer work value")
        byte, shift = divmod(axon, BYTE_BITS)
        if shift + width > BYTE_BITS:
            raise ValueError("work value crosses an input byte")
        mask = ((1 << width) - 1) << shift
        keep = np.uint8((~mask) & BYTE_MASK)
        put = np.uint8((data << shift) & mask)
        self._raw[slot, byte] = (self._raw[slot, byte] & keep) | put
