"""Mutable core objects backed by one authoritative SRAM representation."""

from collections.abc import Callable, Iterable, Iterator, Mapping
from enum import IntEnum
from typing import TypeAlias, cast

import numpy as np
from numpy.typing import NDArray
from paicorelib.core_defs_v2 import AddPotentialMode
from paicorelib.framelib.frame_defs import (
    OfflineConfigFrame2FormatV2 as OffF2,
)
from paicorelib.framelib.frame_defs import (
    OfflineConfigFrame3FormatV2 as OffF3,
)
from paicorelib.framelib.frame_defs import (
    OnlineConfigFrame1FormatV2 as OnF1,
)
from paicorelib.framelib.frame_defs import (
    OnlineConfigFrame2FormatV2 as OnF2,
)
from paicorelib.framelib.parser_v2 import decode_core_config
from paicorelib.neuron_defs_v2 import OfflineNeuRegLimV2, OutputType

from ..frames import route_bits
from ..hardware import (
    BYTE_BITS,
    FRAME_WORD_BITS,
    I32_BITS,
    I32_MASK,
    NEURON_SRAM_WORDS_PER_ROW,
    OFFLINE_CONFIG_WORDS,
    OFFLINE_INPUT_BYTES,
    OFFLINE_INPUT_ROWS,
    OFFLINE_LUT_ENTRIES,
    OFFLINE_NEURON_ROWS,
    ONLINE_CONFIG_WORDS,
    ONLINE_INPUT_BYTES,
    ONLINE_INPUT_ROWS,
    SUPPORTED_DATA_WIDTHS,
    VOLTAGE_LANES_PER_U64,
)
from ..topology.types import CoreAddr, ThreadId
from .decode import decode_core
from .input_buffer import InputWindow
from .model import (
    CoreConfig,
    CoreSnapshot,
    DecodedCore,
    FrameArray,
    LutEntry,
    NeuronRecord,
    StepResult,
    VoltageArray,
    WeightRef,
    core_config,
    snapshot_parts,
)
from .numeric import step as numeric_step


class OfflineLayout(IntEnum):
    """Wire layout selected for an offline neuron: full, half, or folded."""

    FULL = 0
    HALF = 1
    FOLD = 2


def _readonly(array: FrameArray) -> FrameArray:
    view = array.view()
    view.flags.writeable = False
    return view


def _field(word: int, layout: type, name: str) -> int:
    offset = cast(int, getattr(layout, f"{name}_OFFSET"))
    mask = cast(int, getattr(layout, f"{name}_MASK"))
    return (word >> offset) & mask


def _replace(word: int, layout: type, name: str, value: int) -> int:
    offset = cast(int, getattr(layout, f"{name}_OFFSET"))
    mask = cast(int, getattr(layout, f"{name}_MASK"))
    return (word & ~(mask << offset)) | ((value & mask) << offset)


class Weights:
    """Mutable view of a core's raw 128-bit weight SRAM rows.

    Indexing returns ``(low_u64, high_u64)`` Python integers.  Assignment
    validates both words, writes the backing SRAM, increments the owning core's
    revision, and invalidates decoded caches.  :meth:`array` is read-only.
    """

    def __init__(self, raw: FrameArray, touch: Callable[[], None]) -> None:
        self._raw = raw
        self._touch = touch

    def __len__(self) -> int:
        """Return the number of 128-bit rows."""
        return len(self._raw)

    def __getitem__(self, row: int) -> tuple[int, int]:
        """Return one row as two unsigned 64-bit words."""
        low, high = self._raw[row]
        return int(low), int(high)

    def __setitem__(self, row: int, value: tuple[int, int]) -> None:
        """Replace one row and invalidate the owning core's decoded state."""
        if not -len(self._raw) <= row < len(self._raw):
            raise IndexError(row)
        low, high = value
        if not 0 <= low < 1 << FRAME_WORD_BITS or not 0 <= high < 1 << FRAME_WORD_BITS:
            raise ValueError("SRAM words must be unsigned 64-bit integers")
        self._touch()
        self._raw[row] = low, high

    def array(self) -> FrameArray:
        """Return a read-only view of the packed ``(rows, 2)`` SRAM array."""
        return _readonly(self._raw)


class Lut:
    """Mutable threshold/activation lookup table owned by one compute core.

    ``lut[index]`` is ``(signed_threshold, activation)``.  Activation width is
    eight bits for offline cores and sixteen bits for online cores; assignment
    updates the wire representation and invalidates static decoded state.
    """

    def __init__(
        self, raw: FrameArray, touch: Callable[[], None], *, online: bool
    ) -> None:
        self._raw = raw
        self._touch = touch
        activation = OnF2.ACTIVATION_MASK if online else OffF2.ACTIVATION_MASK
        self._shift = activation.bit_length()
        self._value_mask = activation

    def __len__(self) -> int:
        """Return the fixed LUT entry count."""
        return OFFLINE_LUT_ENTRIES

    def __getitem__(self, item: int) -> tuple[int, int]:
        """Return one decoded ``(threshold, activation)`` pair."""
        word = int(self._raw[item])
        threshold = (word >> self._shift) & I32_MASK
        if threshold & (1 << (I32_BITS - 1)):
            threshold -= 1 << I32_BITS
        return threshold, word & self._value_mask

    def __setitem__(self, item: int, value: tuple[int, int]) -> None:
        """Write one decoded pair after validating signed threshold and width."""
        if not -len(self._raw) <= item < len(self._raw):
            raise IndexError(item)
        threshold, activation = value
        if not -(1 << (I32_BITS - 1)) <= threshold < 1 << (I32_BITS - 1):
            raise ValueError("LUT threshold must fit signed int32")
        if not 0 <= activation <= self._value_mask:
            raise ValueError("LUT activation is outside its wire width")
        self._touch()
        self._raw[item] = np.uint64(
            ((threshold & I32_MASK) << self._shift) | activation
        )

    def array(self) -> FrameArray:
        """Return a read-only view of the packed LUT words."""
        return _readonly(self._raw)


class OfflineCore:
    """Offline compute core with lazy decoding and controlled mutable views.

    The raw register/SRAM arrays are authoritative.  ``decoded`` and
    ``neurons`` are views derived from them; writing through ``Weights``,
    ``Lut``, or an ``OfflineNeuron`` updates the same backing storage and bumps
    ``revision``.  Use :meth:`step` for a direct single-core tick or let
    ``Simulator`` schedule the core through frame-driven CONFIG/WORK/SYNC
    transactions.
    """

    def __init__(self, addr: CoreAddr) -> None:
        if addr.core.cpu or addr.core.online:
            raise ValueError("offline core requires an offline coordinate")
        self.addr = addr
        self.registers = np.zeros(OFFLINE_CONFIG_WORDS, np.uint64)
        self._neuron_sram = np.zeros(
            (OFFLINE_NEURON_ROWS, NEURON_SRAM_WORDS_PER_ROW), np.uint64
        )
        self.input_sram = np.zeros((OFFLINE_INPUT_ROWS, OFFLINE_INPUT_BYTES), np.uint8)
        self.config: CoreConfig | None = None
        self.revision = 0
        self.work_steps = 0
        self._started = False
        self.previous_positive = False
        self.voltage: VoltageArray | None = None
        self._decoded: DecodedCore | None = None
        self._routes: tuple[int, ...] = ()
        self.lut = Lut(
            np.zeros(OFFLINE_LUT_ENTRIES, np.uint64), self._touch_static, online=False
        )
        self.weights = Weights(self._neuron_sram, self._touch_static)
        self.neurons = OfflineNeuronBank(self)

    @property
    def input_window(self) -> InputWindow:
        """Return a view that reads/writes logical rows in packed input SRAM."""
        return InputWindow(self.input_sram)

    @property
    def neuron_sram(self) -> FrameArray:
        """Return the packed neuron/weight SRAM as a read-only uint64 array."""
        return _readonly(self._neuron_sram)

    @property
    def send(self) -> int:
        """Return global-send bits from the last complete configuration."""
        return 0 if self.config is None else self.config["global_send"]

    @property
    def receive(self) -> int:
        """Return global-receive bits from the last complete configuration."""
        return 0 if self.config is None else self.config["global_receive"]

    @property
    def thread_id(self) -> ThreadId:
        """Return the configured thread identifier, or zero before CONFIG."""
        return ThreadId(0 if self.config is None else self.config["thread_number"])

    @property
    def decoded(self) -> DecodedCore:
        """Return lazily decoded static state, rebuilding after a static write."""
        if self.config is None:
            raise ValueError(f"core {self.addr} has no complete configuration")
        if self._decoded is None:
            self._decoded = decode_core(self.config, self._neuron_sram, self.lut._raw)
            if self.voltage is None:
                self.voltage = self._decoded.voltage.copy()
            self._routes = tuple(
                route_bits(tuple(int(value) for value in row))
                for row in self._decoded.routes
            )
        return self._decoded

    @property
    def output_routes(self) -> tuple[int, ...]:
        """Return cached route words for decoded output neurons."""
        self.decoded
        return self._routes

    def _touch(self) -> None:
        self.revision += 1
        self._decoded = None
        self._routes = ()
        self.voltage = None

    def _touch_static(self) -> None:
        if self._started:
            raise RuntimeError("static core state can only change while idle")
        self._touch()

    def configure(self, words: FrameArray) -> None:
        """Commit three CONFIG register words while the core is idle."""
        self._touch_static_guard()
        if words.shape != (OFFLINE_CONFIG_WORDS,):
            raise ValueError("offline core configuration requires three words")
        config = core_config(decode_core_config(words))
        if config["neuron_number"] > OFFLINE_NEURON_ROWS:
            raise ValueError("neuron_number exceeds offline SRAM")
        self.registers[:] = words
        self.config = config
        self.voltage = None
        self._touch()

    def _touch_static_guard(self) -> None:
        if self._started:
            raise RuntimeError("static core state can only change while idle")

    def write(self, kind: int, offset: int, words: FrameArray) -> None:
        """Write one storage region using the protocol's CONFIG kind index.

        ``kind`` 0 selects registers, 1 the LUT, 2 neuron/weight SRAM, and 3
        input SRAM.  Static regions reject writes after execution starts;
        malformed ranges raise without fabricating a partial semantic update.
        """
        if kind == 0:
            if offset != 0:
                raise ValueError("core registers start at zero")
            self.configure(words)
            return
        if kind == 1:
            self._touch_static_guard()
            self.lut._raw[offset : offset + len(words)] = words
        elif kind == 2:
            self._touch_static_guard()
            self._neuron_sram.reshape(-1)[offset : offset + len(words)] = words
        elif kind == 3:
            raw = words.astype("<u8").view(np.uint8)
            word_bytes = FRAME_WORD_BITS // BYTE_BITS
            self.input_sram.reshape(-1)[
                offset * word_bytes : offset * word_bytes + raw.size
            ] = raw
            self.revision += 1
            return
        else:
            raise ValueError("unknown offline storage kind")
        self._touch()

    def store_voltage(self) -> None:
        """Pack the current semantic voltages back into neuron SRAM lanes."""
        if self.voltage is None:
            return
        decoded = self.decoded
        rows, lanes = decoded.voltage_locations.T
        for lane in range(VOLTAGE_LANES_PER_U64 * NEURON_SRAM_WORDS_PER_ROW):
            chosen = lanes == lane
            rr = rows[chosen]
            shift = (lane % VOLTAGE_LANES_PER_U64) * I32_BITS
            mask = np.uint64(I32_MASK << shift)
            values = self.voltage[chosen].astype(np.uint32).astype(np.uint64)
            self._neuron_sram[rr, lane // VOLTAGE_LANES_PER_U64] = (
                self._neuron_sram[rr, lane // VOLTAGE_LANES_PER_U64] & ~mask
            ) | (values << np.uint64(shift))

    def initialize(self, external: bool) -> None:
        """Reset voltage and optionally clear input/work state.

        ``external=True`` models INIT from the frame interface and clears input
        SRAM, work-step count, and the started guard.  Internal scheduler resets
        can pass ``False`` to preserve the input/work timeline.
        """
        decoded = self.decoded
        self.voltage = decoded.parameters["init_v"].astype(np.int32).copy()
        self.previous_positive = False
        if external:
            self.input_sram.fill(0)
            self.work_steps = 0
            self._started = False
        self.store_voltage()
        self.revision += 1

    def _inputs(self, tick: int) -> tuple[NDArray[np.int32], NDArray[np.intp]]:
        """Decode all physical rows consumed by one logical offline tick."""
        config = self.decoded.config
        return self.input_window.values(
            tick,
            config.get("lcn", 0),
            width=SUPPORTED_DATA_WIDTHS[config.get("input_width", 0)],
            signed=bool(config.get("input_sign", 0)),
            direct=config.get("add_potential", 0) == AddPotentialMode.DIRECT_ADD,
        )

    def step(self, tick: int) -> "StepResult":
        """Compute and commit one logical tick directly on this core.

        ``tick`` selects the physical input rows through the configured LCN.
        The result contains new voltage, output values, and sampled potentials;
        the core also stores voltage, clears consumed input rows, advances
        ``work_steps``, and increments ``revision``.  This method is a lower-
        level alternative to feeding WORK/SYNC frames through ``Simulator``.
        """
        if tick < 0:
            raise ValueError("tick must be a nonnegative integer")
        decoded = self.decoded
        if self.voltage is None:
            self.voltage = decoded.voltage.copy()
        inputs, rows = self._inputs(tick)
        result = numeric_step(decoded, inputs, self.voltage, self.previous_positive)
        self.voltage = result.voltage.copy()
        self.previous_positive = result.positive
        self.store_voltage()
        self.input_sram[rows] = 0
        self.work_steps = tick + 1
        self._started = True
        self.revision += 1
        return result

    def snapshot(self, selection: Iterable[str] | None = None) -> CoreSnapshot:
        """Return detached read-only core state.

        ``selection`` may contain ``config``, ``input``, ``neurons``,
        ``weights``, ``lut``, ``folds`` and ``voltage``.  ``None`` selects all
        parts.  Returned arrays are copies marked read-only, so inspecting a
        snapshot cannot mutate the live core.
        """
        decoded = self.decoded
        selected = snapshot_parts(selection)

        def readonly_u64(array: FrameArray) -> FrameArray:
            result = np.array(array, dtype=np.uint64, copy=True)
            result.flags.writeable = False
            return result

        def readonly_u8(array: NDArray[np.uint8]) -> NDArray[np.uint8]:
            result = np.array(array, dtype=np.uint8, copy=True)
            result.flags.writeable = False
            return result

        def readonly_i32(array: VoltageArray) -> VoltageArray:
            result = np.array(array, dtype=np.int32, copy=True)
            result.flags.writeable = False
            return result

        neurons = (
            tuple(
                NeuronRecord(
                    index=i,
                    layout=int(decoded.layouts[i]),
                    parameter_row=int(decoded.parameter_locations[i]),
                    voltage_row=int(decoded.voltage_locations[i, 0]),
                    voltage_lane=int(decoded.voltage_locations[i, 1]),
                    tick=int(decoded.ticks[i]),
                    axon=int(decoded.axons[i]),
                    output_type=OutputType(int(decoded.output_types[i])),
                )
                for i in range(decoded.voltage.size)
            )
            if "neurons" in selected
            else None
        )
        weights = (
            tuple(
                WeightRef(
                    group=group_index,
                    posts=tuple(int(value) for value in group.posts),
                    inputs=tuple(int(value) for value in group.inputs),
                    offsets=tuple(int(value) for value in group.offsets),
                )
                for group_index, group in enumerate(decoded.synapses)
            )
            if "weights" in selected
            else None
        )
        lut = (
            tuple(
                LutEntry(
                    index=i,
                    threshold=int(decoded.lut_thresholds[i]),
                    activation=int(decoded.lut_values[i]),
                    raw=int(self.lut._raw[i]),
                )
                for i in range(OFFLINE_LUT_ENTRIES)
            )
            if "lut" in selected
            else None
        )
        folds = decoded.folds if "folds" in selected else None
        return CoreSnapshot(
            revision=self.revision,
            selected=selected,
            config=core_config(decoded.config) if "config" in selected else None,
            registers=readonly_u64(self.registers) if "config" in selected else None,
            neuron_sram=readonly_u64(self._neuron_sram)
            if "neurons" in selected or "weights" in selected
            else None,
            input_sram=readonly_u8(self.input_sram) if "input" in selected else None,
            lut_sram=readonly_u64(self.lut._raw) if "lut" in selected else None,
            voltage=(
                None
                if self.voltage is None or "voltage" not in selected
                else readonly_i32(self.voltage)
            ),
            neurons=neurons,
            weights=weights,
            lut=lut,
            folds=folds,
        )


class OfflineNeuron:
    """Controlled view of one logical neuron backed by an :class:`OfflineCore`."""

    def __init__(self, core: OfflineCore, item: int) -> None:
        self._core = core
        self.index = item

    @property
    def layout(self) -> OfflineLayout:
        """Return the decoded full/half/fold layout for this neuron."""
        return OfflineLayout(int(self._core.decoded.layouts[self.index]))

    @property
    def voltage(self) -> int:
        """Return the current signed int32 membrane voltage."""
        voltage = self._core.voltage
        if voltage is None:
            self._core.decoded
            voltage = self._core.voltage
        assert voltage is not None
        return int(voltage[self.index])

    @voltage.setter
    def voltage(self, value: int) -> None:
        """Set and persist a signed int32 voltage while preserving SRAM lanes."""
        if (
            not OfflineNeuRegLimV2.THRES_NEG_MIN
            <= value
            <= OfflineNeuRegLimV2.THRES_NEG_MAX
        ):
            raise ValueError("voltage must fit signed int32")
        self._core.decoded
        assert self._core.voltage is not None
        self._core.voltage[self.index] = value
        self._core.store_voltage()
        self._core.revision += 1

    def parameter(self, name: str) -> int:
        """Return one named decoded parameter for this neuron."""
        parameters = cast(
            Mapping[str, NDArray[np.int64]], self._core.decoded.parameters
        )
        try:
            return int(parameters[name][self.index])
        except KeyError as exc:
            raise ValueError(f"unknown neuron parameter {name!r}") from exc

    def set_parameter(self, name: str, value: int) -> None:
        """Validate, encode, and persist one mutable neuron parameter."""
        self._core._touch_static_guard()
        decoded = self._core.decoded
        row = int(decoded.parameter_locations[self.index])
        low, high = map(int, self._core._neuron_sram[row])
        low_fields = {
            "lateral_inhi": ("LATERAL_INHIBITION", 0, 1),
            "leak_multi_sequence": ("LEAK_MULTI_SEQUENCE", 0, 1),
            "leak_multi_input": ("LEAK_MULTI_INPUT", 0, 1),
            "leak_multi_mode": ("LEAK_MULTI_MODE", 0, 1),
            "leak_add_mode": ("LEAK_ADD_MODE", 0, 1),
            "leak_tau": (
                "LEAK_TAU",
                OfflineNeuRegLimV2.LEAK_TAU_MIN,
                OfflineNeuRegLimV2.LEAK_TAU_MAX,
            ),
            "leak_v": (
                "LEAK_V",
                OfflineNeuRegLimV2.LEAK_V_MIN,
                OfflineNeuRegLimV2.LEAK_V_MAX,
            ),
            "init_v": (
                "VJT_INITIAL",
                OfflineNeuRegLimV2.VJT_INITIAL_MIN,
                OfflineNeuRegLimV2.VJT_INITIAL_MAX,
            ),
        }
        high_fields = {
            "reset_mode": ("RESET_MODE", 0, 2),
            "reset_v": (
                "RESET_V",
                OfflineNeuRegLimV2.RESET_V_MIN,
                OfflineNeuRegLimV2.RESET_V_MAX,
            ),
            "thres_neg_mode": ("THRESHOLD_NEG_MODE", 0, 1),
            "thres_pos_mode": ("THRESHOLD_POS_MODE", 0, 1),
            "thres_neg": (
                "THRESHOLD_NEG",
                OfflineNeuRegLimV2.THRES_NEG_MIN,
                OfflineNeuRegLimV2.THRES_NEG_MAX,
            ),
        }
        if name == "thres_pos":
            if (
                not OfflineNeuRegLimV2.THRES_POS_MIN
                <= value
                <= OfflineNeuRegLimV2.THRES_POS_MAX
            ):
                raise ValueError("thres_pos is outside its register range")
            raw = value & I32_MASK
            low = _replace(low, OffF3.Full.Word3, "THRESHOLD_POS_LOW20", raw)
            high = _replace(
                high,
                OffF3.Full.Word4,
                "THRESHOLD_POS_HIGH12",
                raw >> OffF3.Full.Word3.THRESHOLD_POS_LOW20_MASK.bit_length(),
            )
        elif name in low_fields:
            field, lower, upper = low_fields[name]
            if not lower <= value <= upper:
                raise ValueError(f"{name} is outside its register range")
            low = _replace(low, OffF3.Full.Word3, field, value)
        elif name in high_fields:
            field, lower, upper = high_fields[name]
            if not lower <= value <= upper:
                raise ValueError(f"{name} is outside its register range")
            high = _replace(high, OffF3.Full.Word4, field, value)
        else:
            raise ValueError(f"unknown neuron parameter {name!r}")
        self._core.store_voltage()
        self._core._neuron_sram[row] = np.uint64(low), np.uint64(high)
        self._core._touch_static()


class OfflineNeuronBank:
    """Indexable collection of controlled ``OfflineNeuron`` views."""

    def __init__(self, core: OfflineCore) -> None:
        self._core = core

    def __len__(self) -> int:
        """Return the number of logical neurons in the decoded core."""
        return len(self._core.decoded.voltage)

    def __getitem__(self, item: int) -> OfflineNeuron:
        """Return a neuron view, supporting Python negative indices."""
        size = len(self)
        if item < 0:
            item += size
        if not 0 <= item < size:
            raise IndexError(item)
        return OfflineNeuron(self._core, item)

    def __iter__(self) -> Iterator[OfflineNeuron]:
        return (self[i] for i in range(len(self)))


def _decode_online_config(words: FrameArray) -> dict[str, int]:
    if words.shape != (ONLINE_CONFIG_WORDS,):
        raise ValueError(
            f"online core configuration requires {ONLINE_CONFIG_WORDS} words"
        )
    w1, w2, w3, w4, w5 = map(int, words)
    return {
        "snn_ann": _field(w1, OnF1.Word1, "SNN_ANN"),
        "work_mode": _field(w1, OnF1.Word1, "WORK_MODE"),
        "input_width": _field(w1, OnF1.Word1, "INPUT_WIDTH"),
        "output_width": _field(w1, OnF1.Word1, "OUTPUT_WIDTH"),
        "neuron_number": (
            _field(w1, OnF1.Word1, "NEURON_NUMBER_HIGH10") << 3
            | _field(w2, OnF1.Word2, "NEURON_NUMBER_LOW3")
        ),
        "test_core_xy": _field(w3, OnF1.Word3, "TEST_CORE_XY"),
        "test_core_x": _field(w3, OnF1.Word3, "TEST_CORE_X"),
        "test_core_y": (
            _field(w3, OnF1.Word3, "TEST_CORE_Y_HIGH1")
            << OnF1.Word4.TEST_CORE_Y_LOW5_MASK.bit_length()
            | _field(w4, OnF1.Word4, "TEST_CORE_Y_LOW5")
        ),
        "global_send": _field(w4, OnF1.Word4, "GLOBAL_SEND"),
        "global_receive": _field(w4, OnF1.Word4, "GLOBAL_RECEIVE"),
        "thread_number": _field(w4, OnF1.Word4, "THREAD_NUMBER"),
        "tick_start": _field(w5, OnF1.Word5, "TICK_START"),
        "tick_duration": _field(w5, OnF1.Word5, "TICK_DURATION"),
        "tick_initial": _field(w5, OnF1.Word5, "TICK_INITIAL"),
    }


class OnlineCore:
    """Observable online storage with no numerical SYNC/UPDATE execution.

    CONFIG, WORK, TEST, INIT, routing, and snapshots are modeled so callers can
    inspect protocol state.  A scheduler encountering this core during SYNC
    raises ``UnsupportedFrame`` rather than pretending to compute results.
    """

    def __init__(self, addr: CoreAddr) -> None:
        if not addr.core.online:
            raise ValueError("online core requires an online coordinate")
        self.addr = addr
        self.registers = np.zeros(ONLINE_CONFIG_WORDS, np.uint64)
        self._neuron_sram = np.zeros(
            (OFFLINE_NEURON_ROWS, NEURON_SRAM_WORDS_PER_ROW), np.uint64
        )
        self.input_sram = np.zeros((ONLINE_INPUT_ROWS, ONLINE_INPUT_BYTES), np.uint8)
        self.config: dict[str, int] | None = None
        self.revision = 0
        self.lut = Lut(
            np.zeros(OFFLINE_LUT_ENTRIES, np.uint64), self._touch, online=True
        )
        self.weights = Weights(self._neuron_sram, self._touch)

    @property
    def neuron_sram(self) -> FrameArray:
        """Return online neuron/weight SRAM as a read-only uint64 array."""
        return _readonly(self._neuron_sram)

    @property
    def send(self) -> int:
        """Return global-send bits from the complete online configuration."""
        return 0 if self.config is None else self.config["global_send"]

    @property
    def receive(self) -> int:
        """Return global-receive bits from the complete online configuration."""
        return 0 if self.config is None else self.config["global_receive"]

    @property
    def thread_id(self) -> ThreadId:
        """Return the configured online thread identifier, or zero before CONFIG."""
        return ThreadId(0 if self.config is None else self.config["thread_number"])

    def _touch(self) -> None:
        self.revision += 1

    def configure(self, words: FrameArray) -> None:
        """Commit five online CONFIG register words."""
        config = _decode_online_config(words)
        if config["neuron_number"] > OFFLINE_NEURON_ROWS:
            raise ValueError("neuron_number exceeds online SRAM")
        self.registers[:] = words
        self.config = config
        self._touch()

    def write(self, kind: int, offset: int, words: FrameArray) -> None:
        """Write online LUT, neuron SRAM, or input SRAM storage by kind."""
        if kind == 0:
            if offset != 0:
                raise ValueError("core registers start at zero")
            self.configure(words)
            return
        if kind == 1:
            self.lut._raw[offset : offset + len(words)] = words
        elif kind == 2:
            self._neuron_sram.reshape(-1)[offset : offset + len(words)] = words
        elif kind == 3:
            raw = words.astype("<u8").view(np.uint8)
            word_bytes = FRAME_WORD_BITS // BYTE_BITS
            self.input_sram.reshape(-1)[
                offset * word_bytes : offset * word_bytes + raw.size
            ] = raw
        else:
            raise ValueError("unknown online storage kind")
        self._touch()

    def initialize(self) -> None:
        """Clear online input SRAM and advance the observable revision."""
        self.input_sram.fill(0)
        self._touch()


class RelayCore:
    """Control-only core that forwards global signal connectivity."""

    def __init__(
        self, addr: CoreAddr, send: int, receive: int, thread_id: ThreadId
    ) -> None:
        self.addr = addr
        self.send = send
        self.receive = receive
        self.thread_id = thread_id
        self.revision = 0


class CompletionCore:
    """Control-only core that participates in thread completion signaling."""

    def __init__(
        self, addr: CoreAddr, send: int, receive: int, thread_id: ThreadId
    ) -> None:
        self.addr = addr
        self.send = send
        self.receive = receive
        self.thread_id = thread_id
        self.revision = 0


Core: TypeAlias = OfflineCore | OnlineCore | RelayCore | CompletionCore
ComputeCore: TypeAlias = OfflineCore | OnlineCore
