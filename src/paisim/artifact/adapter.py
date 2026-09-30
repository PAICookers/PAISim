"""Optional schema-v1 artifact I/O adapter, independent of the execution engine.

Only compiled frame words configure computation. Mappings encode host tensors
and interpret returned words; they never supply a hidden reference network.
"""

import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from paicorelib.framelib.frame_defs import FFV2, FrameHeader, FrameType
from paicorelib.framelib.frame_defs import OfflineWorkFrame1FormatV2 as OffW1

from .._generated.compile_artifacts_pb2 import (
    DataType,
    InputTensorMapping,
    OutputTensorMapping,
    ThreadIOMapping,
)
from ..engine.core import OfflineCore
from ..engine.model import FrameArray, IntegerArray
from ..frames import route_bits, work_fields
from ..hardware import (
    BYTE_BITS,
    BYTE_MASK,
    DOCUMENTED_LCN_MAX,
    FRAME_TYPE_SHIFT,
    I32_BITS,
    OFFLINE_AXON_BITS,
    OFFLINE_AXON_VALUES,
    OFFLINE_WORK_TIMESTEP_COUNT,
    POTENTIAL_LANE_AXON_STRIDE,
    POTENTIAL_LANE_MASK,
    POTENTIAL_LANE_SHIFT,
    POTENTIAL_LANES,
    WORK_DATA_MASK,
)
from ..runtime.simulator import Simulator
from ..topology.types import CoreAddr, Port
from .io import MAX_TENSOR_ELEMENTS, ArtifactFormat, read_artifact

_SIGNED_DATA_TYPES = frozenset(
    (DataType.INT1, DataType.INT2, DataType.INT4, DataType.INT8)
)


@dataclass(frozen=True, slots=True)
class ArtifactView:
    """Read-only, detached metadata retained by ``Simulator.from_artifact``.

    The view owns copies of configuration words and generated Protobuf thread
    mappings.  It is metadata only: feeding or decoding still happens through
    ``Simulator`` or :class:`Artifact`.
    """

    _frames: FrameArray
    _thread_data: tuple[ThreadIOMapping, ...]

    @property
    def config_frames(self) -> FrameArray:
        """Return the loaded configuration words as a read-only ``uint64`` array."""
        frames = self._frames.view()
        frames.flags.writeable = False
        return frames

    @property
    def threads(self) -> tuple[ThreadIOMapping, ...]:
        """Return deep copies of thread mappings so callers cannot mutate the view."""
        return tuple(deepcopy(thread) for thread in self._thread_data)


def _route(offset: Any, copies: Any | None = None) -> int:
    copy = copies or type(offset)()
    return route_bits((offset.xy, offset.x, offset.y, copy.xy, copy.x, copy.y))


def _format(item: InputTensorMapping | OutputTensorMapping) -> tuple[int, bool]:
    if (
        isinstance(item, OutputTensorMapping)
        and item.kind == OutputTensorMapping.VOLTAGE
    ):
        return I32_BITS, True
    dtype = item.entries[0].dtype
    return item.bit_width, dtype in _SIGNED_DATA_TYPES


def check_metadata(
    simulator: Simulator, threads: Sequence[ThreadIOMapping], *, ingress: Port
) -> None:
    """Compare mapped register claims against already loaded raw configuration.

    This never initializes, advances, reconstructs, or repairs a core. It checks
    roots, explicit core tick records and input destinations. Schema-v1 output
    mappings carry no source-core identity, so output width/target-LCN cannot be
    inferred reliably and are not guessed from names or core ordering.
    """
    configs = {
        CoreAddr(chip.coord, coord): core.config
        for chip in simulator.array.iter_chips()
        for coord, core in chip.cores.items()
        if isinstance(core, OfflineCore) and core.config is not None
    }

    def targets(offset: Any, copies: Any | None = None) -> tuple[CoreAddr, ...]:
        try:
            result = simulator.router.route(ingress.addr, _route(offset, copies))
        except (ValueError, NotImplementedError) as exc:
            raise ValueError(f"metadata route is unsupported: {exc}") from exc
        for addr in result:
            if addr.core.cpu or addr.core.online:
                raise ValueError(f"metadata core {addr}: unsupported target")
            if addr not in configs:
                raise ValueError(
                    f"metadata core {addr}: expected configured offline core, "
                    "actual unconfigured"
                )
        return result

    def match(
        coord: CoreAddr,
        field: Literal[
            "thread_number",
            "tick_start",
            "tick_duration",
            "tick_initial",
            "input_width",
            "input_sign",
        ],
        expected: int,
    ) -> None:
        config = configs[coord]
        actual = config[field]
        if actual != expected:
            raise ValueError(
                f"metadata core {coord} field {field}: "
                f"expected {expected}, actual {actual}"
            )

    for thread in threads:
        for root in targets(thread.root_core_offset):
            match(root, "thread_number", thread.thread_id)
        for core_tick in thread.core_ticks:
            for coord in targets(core_tick.core_offset):
                match(coord, "tick_start", core_tick.tick.tick_start)
                match(coord, "tick_duration", core_tick.tick.tick_duration)
                match(coord, "tick_initial", core_tick.tick.tick_initial)
        for item in thread.input_mappings.items:
            for entry in item.entries:
                for coord in targets(entry.core_offset, entry.copy_count):
                    match(coord, "input_width", item.bit_width.bit_length() - 1)
                    match(coord, "input_sign", int(entry.dtype in _SIGNED_DATA_TYPES))


def resolve_thread_roots(
    simulator: Simulator, threads: Sequence[ThreadIOMapping], ingress: Port
) -> tuple[CoreAddr, ...]:
    """Resolve and validate one configured root for each artifact thread."""
    roots: list[CoreAddr] = []
    seen: set[CoreAddr] = set()
    for thread in threads:
        targets = simulator.router.route(ingress.addr, _route(thread.root_core_offset))
        if len(targets) != 1:
            raise ValueError("thread root mapping must select exactly one core")
        root = targets[0]
        if root in seen:
            raise ValueError(f"multiple threads use root {root}")
        seen.add(root)
        roots.append(root)
    return tuple(roots)


class Artifact:
    """A single compiled thread with bounded STREAM tensor conversion.

    Use the raw Simulator for multi-thread orchestration and timestamp epochs
    beyond this adapter's unambiguous output timestamp window.
    """

    def __init__(
        self, path: str | Path, *, format: ArtifactFormat | None = None
    ) -> None:
        """Read one schema-v1 artifact for tensor/frame conversion.

        Args:
            path: JSON or Protobuf artifact path.  The suffix selects the
                format unless ``format`` is supplied explicitly.
            format: Optional ``"json"`` or ``"protobuf"`` override.

        The adapter requires exactly one thread and STREAM output timestamps.
        It validates mapping shapes, dtypes, routes, LCN values, and bounded
        tensor sizes during construction.  It does not create a simulator or
        execute configuration frames; use ``self.config_frames`` with a
        separately constructed ``Simulator``.

        Raises:
            ValueError: For malformed schema or out-of-range mappings.
            NotImplementedError: For multi-thread, non-STREAM, or unsupported
                potential-input artifacts.
        """
        self.config_frames, threads = read_artifact(path, format=format)
        if len(threads) != 1:
            raise NotImplementedError("tensor adapter currently requires one thread")
        self.thread = threads[0]
        self.runtime = self.thread.runtime
        if self.runtime.decode_mode != 0:
            raise NotImplementedError(
                "tensor adapter requires STREAM output timestamps"
            )
        self.root_route = _route(self.thread.root_core_offset)
        self.inputs = self.thread.input_mappings.items
        self.outputs = self.thread.output_mappings.items
        self.target_lcn = self.thread.output_mappings.target_lcn
        self._input_tables: dict[
            str, tuple[NDArray[np.int64], FrameArray, FrameArray, FrameArray]
        ] = {}
        for item in self.inputs:
            entries = item.entries
            bits, _ = _format(item)
            if bits == I32_BITS:
                raise NotImplementedError(
                    "potential input tensor encoding is not yet supported"
                )
            elems = np.array([e.elem_idx for e in entries], np.int64)
            if np.any(elems < 0) or np.any(elems >= math.prod(item.shape.size)):
                raise ValueError("input mapping element out of range")
            lcn = np.array([e.target_lcn for e in entries], np.uint64)
            ticks = np.array([e.tick_relative for e in entries], np.uint64)
            axons = np.array([e.addr_axon for e in entries], np.uint64)
            if (
                np.any(lcn > DOCUMENTED_LCN_MAX)
                or np.any(ticks >= (1 << lcn))
                or np.any(axons >= OFFLINE_AXON_VALUES)
            ):
                raise ValueError("input mapping address/LCN out of range")
            routes = np.array(
                [_route(e.core_offset, e.copy_count) for e in entries], np.uint64
            )
            self._input_tables[item.name] = (
                elems,
                lcn,
                ticks,
                routes | (axons << np.uint64(OffW1.AXON_ADDR_OFFSET)),
            )

    def encode_inputs(
        self, values: Mapping[str, IntegerArray], timestep: int
    ) -> FrameArray:
        """Encode one mapped integer sample step as zero-suppressed WORK frames.

        ``values`` must contain exactly the artifact's input names, with each
        array matching its declared shape and signed integer range.  ``timestep``
        is non-negative; its physical timestamp wraps at the protocol's 8-bit
        boundary.  The result is a one-dimensional ``np.uint64`` array and the
        adapter emits no words for zero-valued mapped elements.
        """
        if timestep < 0:
            raise ValueError("timestep must be a nonnegative integer")
        if set(values) != {item.name for item in self.inputs}:
            raise ValueError("input names must exactly match artifact mappings")
        frames = []
        for item in self.inputs:
            value = np.asarray(values[item.name])
            if value.shape != tuple(item.shape.size) or value.dtype.kind not in "iu":
                raise ValueError("input tensor requires mapped shape and integer dtype")
            bits, signed = _format(item)
            lower, upper = (
                (-(1 << (bits - 1)), (1 << (bits - 1)) - 1)
                if signed
                else (0, (1 << bits) - 1)
            )
            if np.any(value < lower) or np.any(value > upper):
                raise ValueError("input tensor values exceed mapped integer range")
            elems, lcn, relative, base = self._input_tables[item.name]
            payload = (
                value.reshape(-1)[elems].astype(np.int64) & ((1 << bits) - 1)
            ).astype(np.uint64)
            slots = (
                (np.uint64(timestep % OFFLINE_WORK_TIMESTEP_COUNT) << lcn) + relative
            ) & np.uint64(OFFLINE_WORK_TIMESTEP_COUNT - 1)
            words = (
                np.uint64(FrameHeader.WORK_TYPE1.value << FFV2.GENERAL_HEADER_OFFSET)
                | base
                | (
                    (slots >> np.uint64(OffW1.TIMESTEP_MASK.bit_length()))
                    << np.uint64(OffW1.TIMESTEP_HIGH7_OFFSET)
                )
                | (
                    (slots & np.uint64(OffW1.TIMESTEP_MASK))
                    << np.uint64(OffW1.TIMESTEP_OFFSET)
                )
                | (payload & np.uint64(WORK_DATA_MASK))
            )
            frames.append(words[payload != 0])
        return np.concatenate(frames) if frames else np.empty(0, np.uint64)

    def decode_outputs(
        self, words: FrameArray, timesteps: int
    ) -> dict[str, NDArray[np.int32]]:
        """Decode DATA and complete potential lanes; retain raw frames separately.

        Missing DATA frames mean zero under zero suppression. Missing potential
        lanes and unmapped work outputs are errors, not invented zero values.
        ``words`` must be a one-dimensional ``np.uint64`` response array.  The
        returned mapping contains one ``np.int32`` array per output name with
        shape ``(timesteps, *declared_shape)``; raw response words remain the
        caller's responsibility.
        """
        if not 1 <= timesteps <= OFFLINE_WORK_TIMESTEP_COUNT >> self.target_lcn:
            raise NotImplementedError(
                "output timestamp wrap needs an explicit epoch adapter"
            )
        if (
            sum(math.prod(item.shape.size) for item in self.outputs) * timesteps
            > MAX_TENSOR_ELEMENTS
        ):
            raise ValueError("decoded output exceeds adapter element budget")
        if (
            not isinstance(words, np.ndarray)
            or words.ndim != 1
            or words.dtype != np.uint64
        ):
            raise ValueError("output frames require a one-dimensional uint64 array")
        result: dict[str, NDArray[np.int32]] = {}
        mapping: dict[tuple[bool, int], tuple[str, int, int, bool]] = {}
        masks: dict[tuple[str, int], NDArray[np.uint8]] = {}
        for item in self.outputs:
            name = item.name
            count = math.prod(item.shape.size)
            result[name] = np.zeros((timesteps, count), np.int32)
            bits, signed = _format(item)
            voltage = item.kind == OutputTensorMapping.VOLTAGE
            for entry in item.entries:
                key = voltage, entry.axon_bit_idx
                if key in mapping:
                    raise ValueError("ambiguous duplicate output mapping address")
                element = entry.elem_idx
                if not 0 <= element < count:
                    raise ValueError("output mapping element out of range")
                mapping[key] = name, element, bits, signed
                if voltage:
                    masks[name, element] = np.zeros(timesteps, np.uint8)
        for raw in words:
            word = int(raw)
            header = FrameHeader(
                (word >> FFV2.GENERAL_HEADER_OFFSET) & FFV2.GENERAL_HEADER_MASK
            )
            if header is FrameHeader.CTRL_TYPE3:
                continue
            if FrameType(header.value >> FRAME_TYPE_SHIFT) is not FrameType.WORK:
                raise ValueError("output adapter expects WORK/COMPLETE only")
            slot, axon, byte, voltage = work_fields(word)
            tick = slot >> self.target_lcn
            address = ((slot << OFFLINE_AXON_BITS) | axon) & (
                (1 << (OFFLINE_AXON_BITS + self.target_lcn)) - 1
            )
            lane = (
                (address >> POTENTIAL_LANE_SHIFT) & (POTENTIAL_LANES - 1)
                if voltage
                else 0
            )
            key = voltage, address - lane * POTENTIAL_LANE_AXON_STRIDE
            if key not in mapping or tick >= timesteps:
                raise ValueError(f"unmapped output frame or timestep: {word:#018x}")
            name, elem, bits, signed = mapping[key]
            if voltage:
                unsigned = result[name].view(np.uint32)
                mask = np.uint32(BYTE_MASK << (BYTE_BITS * lane))
                unsigned[tick, elem] = unsigned[tick, elem] & ~mask | np.uint32(
                    byte << (BYTE_BITS * lane)
                )
                masks[name, elem][tick] |= 1 << lane
            else:
                value = byte & ((1 << bits) - 1)
                if signed and value & (1 << (bits - 1)):
                    value -= 1 << bits
                result[name][tick, elem] = value
        if any(np.any(mask != POTENTIAL_LANE_MASK) for mask in masks.values()):
            raise ValueError("incomplete potential output lanes")
        return {
            item.name: result[item.name].reshape(timesteps, *item.shape.size)
            for item in self.outputs
        }
