"""Bounded readers for the generated schema-v1 artifact messages."""

import math
from pathlib import Path
from typing import Final, Literal, cast

import numpy as np
from paicorelib.core_defs_v2 import OfflineCoreRegLimV2
from paicorelib.neuron_defs_v2 import OfflineNeuRegLimV2

from .._generated.compile_artifacts_pb2 import (
    CompileArtifacts,
    CopyCount,
    CoreOffset,
    DataType,
    OutputTensorMapping,
    RuntimeParams,
    Shape,
    ThreadIOMapping,
    TickParams,
)
from ..engine.model import FrameArray
from ..hardware import (
    DOCUMENTED_LCN_MAX,
    I32_BITS,
    OFFLINE_INPUT_BITS,
    POTENTIAL_BASE_SPAN,
    SYNC_STEP_MASK,
    WORK_DATA_BITS,
)

ArtifactFormat = Literal["json", "protobuf"]
MAX_FILE_BYTES = 256 * 1024 * 1024
MAX_TENSOR_ELEMENTS = 16 * 1024 * 1024
PROTOBUF_UINT32_MAX: Final[int] = (1 << 32) - 1
_DTYPES = {
    "UINT1": (1, False),
    "INT1": (1, True),
    "UINT2": (2, False),
    "INT2": (2, True),
    "UINT4": (4, False),
    "INT4": (4, True),
    "UINT8": (8, False),
    "INT8": (8, True),
}


def _uint(value: int, name: str, maximum: int = PROTOBUF_UINT32_MAX) -> int:
    if not 0 <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [0, {maximum}]")
    return value


def _text(value: str, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _offset(value: CoreOffset | CopyCount) -> None:
    for name in ("xy", "x", "y"):
        component = getattr(value, name)
        if not isinstance(component, int) or not (
            OfflineNeuRegLimV2.ADDR_CORE_COORD_MIN
            <= component
            <= OfflineNeuRegLimV2.ADDR_CORE_COORD_MAX
        ):
            raise ValueError("route offset must be an integer in [-31, 31]")


def _tick(value: TickParams) -> None:
    _uint(value.tick_start, "tick_start", OfflineCoreRegLimV2.TICK_START_MAX)
    _uint(
        value.tick_duration,
        "tick_duration",
        OfflineCoreRegLimV2.TICK_DURATION_MAX,
    )
    _uint(value.tick_initial, "tick_initial", OfflineCoreRegLimV2.TICK_INITIAL_MAX)


def _shape(value: Shape) -> tuple[int, ...]:
    shape = tuple(
        _uint(size, "shape dimension", MAX_TENSOR_ELEMENTS) for size in value.size
    )
    if not shape or len(shape) > 32 or any(size == 0 for size in shape):
        raise ValueError("shape requires 1..32 positive dimensions")
    if math.prod(shape) > MAX_TENSOR_ELEMENTS:
        raise ValueError("shape exceeds adapter element budget")
    return shape


def _dtype_name(value: int, name: str) -> str:
    try:
        return DataType.Code.Name(value)
    except ValueError as exc:
        raise ValueError(f"unknown {name}") from exc


def _dtype(value: int, bits: int, *, voltage: bool = False) -> str:
    name = _dtype_name(value, "dtype")
    if voltage:
        if bits != I32_BITS or name != "NOT_SET":
            raise ValueError(
                f"VOLTAGE requires bit_width {I32_BITS} and an unset dtype"
            )
        return "INT32"
    if name not in _DTYPES or _DTYPES[name][0] != bits:
        raise ValueError("DATA dtype must be set and agree with bit_width")
    return name


def _require_field(message: ThreadIOMapping | CompileArtifacts, field: str) -> None:
    if not message.HasField(field):
        raise ValueError(f"{field} must be set")


def normalize_thread(thread: ThreadIOMapping) -> ThreadIOMapping:
    """Validate a generated thread message before it reaches tensor adapters."""
    _uint(thread.thread_id, "thread_id", OfflineCoreRegLimV2.THREAD_NUMBER_MAX)
    _offset(thread.root_core_offset)
    _require_field(thread, "runtime")
    runtime = thread.runtime
    _uint(runtime.timesteps, "timesteps")
    _uint(runtime.tick_depth, "tick_depth")
    _uint(runtime.sync_steps, "sync_steps", SYNC_STEP_MASK)
    if runtime.timesteps == 0 or runtime.sync_steps < runtime.timesteps:
        raise ValueError(
            "runtime requires positive timesteps and sync_steps >= timesteps"
        )
    try:
        RuntimeParams.DecodeMode.Name(runtime.decode_mode)
    except ValueError as exc:
        raise ValueError("unknown decode_mode") from exc

    _require_field(thread, "input_mappings")
    inputs = thread.input_mappings
    names: set[str] = set()
    for item in inputs.items:
        name = _text(item.name, "input name")
        if name in names:
            raise ValueError("duplicate input name")
        names.add(name)
        if not item.HasField("shape"):
            raise ValueError("shape must be set")
        shape = _shape(item.shape)
        bits = _uint(item.bit_width, "bit_width", WORK_DATA_BITS)
        input_entries = item.entries
        for input_entry in input_entries:
            lcn = _uint(input_entry.target_lcn, "target_lcn", DOCUMENTED_LCN_MAX)
            axon = _uint(
                input_entry.addr_axon, "addr_axon", OfflineNeuRegLimV2.ADDR_AXON_MAX
            )
            if (
                bits not in (1, 2, 4, 8)
                or axon % bits
                or axon + bits > OFFLINE_INPUT_BITS
            ):
                raise ValueError("input address alignment/bit_width invalid")
            _uint(input_entry.elem_idx, "elem_idx", math.prod(shape) - 1)
            _offset(input_entry.core_offset)
            _offset(input_entry.copy_count)
            _uint(input_entry.tick_relative, "tick_relative", (1 << lcn) - 1)
            _uint(input_entry.copy_id, "copy_id")
            _dtype(input_entry.dtype, bits)
        if (
            not input_entries
            or len(
                {
                    _dtype_name(input_entry.dtype, "dtype")
                    for input_entry in input_entries
                }
            )
            != 1
        ):
            raise ValueError("input entries require a consistent dtype")
        _tick(item.tick)

    _require_field(thread, "output_mappings")
    outputs = thread.output_mappings
    target_lcn = _uint(outputs.target_lcn, "output target_lcn", DOCUMENTED_LCN_MAX)
    names.clear()
    addresses: set[tuple[bool, int]] = set()
    for output in outputs.items:
        name = _text(output.name, "output name")
        if name in names:
            raise ValueError("duplicate output name")
        names.add(name)
        if not output.HasField("shape"):
            raise ValueError("shape must be set")
        shape = _shape(output.shape)
        try:
            kind_name = OutputTensorMapping.OutputKind.Name(output.kind)
        except ValueError as exc:
            raise ValueError("unknown output kind") from exc
        voltage = kind_name == "VOLTAGE"
        bits = _uint(output.bit_width, "bit_width", I32_BITS)
        output_entries = output.entries
        for output_entry in output_entries:
            address = _uint(
                output_entry.axon_bit_idx,
                "axon_bit_idx",
                (OFFLINE_INPUT_BITS << target_lcn) - 1,
            )
            if (
                voltage
                and address + POTENTIAL_BASE_SPAN >= OFFLINE_INPUT_BITS << target_lcn
            ):
                raise ValueError("VOLTAGE output requires a valid base address")
            key = voltage, address
            if key in addresses:
                raise ValueError("ambiguous duplicate output mapping address")
            addresses.add(key)
            _uint(output_entry.elem_idx, "elem_idx", math.prod(shape) - 1)
            _uint(output_entry.copy_id, "copy_id")
            _dtype(output_entry.dtype, bits, voltage=voltage)
        if (
            not output_entries
            or len(
                {
                    _dtype_name(output_entry.dtype, "dtype")
                    for output_entry in output_entries
                }
            )
            != 1
        ):
            raise ValueError("output entries require a consistent dtype")
        _tick(output.tick)

    for core_tick in thread.core_ticks:
        _offset(core_tick.core_offset)
        _tick(core_tick.tick)
        for node in core_tick.nodes:
            _text(node, "node")
    return thread


def _read_message(data: bytes, selected: ArtifactFormat) -> CompileArtifacts:
    """Parse JSON and binary input into one generated message type."""
    try:
        from google.protobuf.json_format import Parse, ParseError
        from google.protobuf.message import DecodeError
    except ImportError as exc:
        raise ImportError(
            "Protobuf runtime is required by the base paisim install"
        ) from exc
    message = CompileArtifacts()
    try:
        if selected == "protobuf":
            message.ParseFromString(data)
            known = CompileArtifacts()
            known.CopyFrom(message)
            known.DiscardUnknownFields()
            if known.SerializeToString() != message.SerializeToString():
                raise ValueError("unknown Protobuf schema-v1 fields")
        else:
            Parse(data.decode("utf-8"), message, ignore_unknown_fields=False)
    except (
        DecodeError,
        ParseError,
        UnicodeDecodeError,
        ValueError,
        RecursionError,
    ) as exc:
        raise ValueError(f"malformed {selected} artifact") from exc
    return message


def read_artifact(
    path: str | Path, *, format: ArtifactFormat | None = None
) -> tuple[FrameArray, list[ThreadIOMapping]]:
    """Read and validate a schema-v1 artifact.

    Args:
        path: JSON or Protobuf file containing paired 32-bit configuration
            words and thread I/O mappings.
        format: Optional explicit format; otherwise ``.json`` and ``.pb`` are
            recognized from the suffix.

    Returns:
        ``(config_frames, threads)`` where configuration frames are a read-only
        one-dimensional ``np.uint64`` array and ``threads`` are validated
        generated Protobuf messages.

    The reader enforces file, tensor, route, and schema budgets.  It is a pure
    file operation: it does not construct or advance a simulator.
    """
    path = Path(path)
    candidate = format or {".json": "json", ".pb": "protobuf"}.get(path.suffix)
    if candidate not in ("json", "protobuf"):
        raise ValueError("explicit artifact format required for unknown suffix")
    selected = cast(ArtifactFormat, candidate)
    with path.open("rb") as stream:
        data = stream.read(MAX_FILE_BYTES + 1)
    if len(data) > MAX_FILE_BYTES:
        raise ValueError("artifact exceeds file byte budget")
    doc = _read_message(data, selected)
    if doc.schema_version != 1:
        raise NotImplementedError("only compile-artifact schema 1 is supported")
    _require_field(doc, "config_frames")
    frames = doc.config_frames
    words = [_uint(word, "config_frames.words") for word in frames.words]
    if len(words) % 2:
        raise ValueError("config_frames.words must contain paired uint32 words")
    order = frames.word_order
    if order not in (0, 1):
        raise ValueError("unknown config_frames.word_order")
    array = np.asarray(words, dtype=np.uint64).reshape(-1, 2)
    hi, lo = (0, 1) if order == 0 else (1, 0)
    config_frames = array[:, hi] << np.uint64(32) | array[:, lo]
    config_frames.flags.writeable = False
    _require_field(doc, "io_mapping")
    io_mapping = doc.io_mapping
    threads = [normalize_thread(thread) for thread in io_mapping.threads]
    if not threads or len({thread.thread_id for thread in threads}) != len(threads):
        raise ValueError("artifact requires unique thread IDs")
    return config_frames, threads
