"""Frame-driven orchestration over explicit chip, core, and thread objects."""

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from paicorelib.framelib.frame_defs import (
    FFV2,
    FrameHeader,
    FramePackageType,
    FrameType,
    OfflineControlFrame1FormatV2,
    OfflineControlFrame3FormatV2,
)
from paicorelib.framelib.types import FRAME_DTYPE

from ..engine.core import CompletionCore, Core, OfflineCore, OnlineCore, RelayCore
from ..engine.errors import UndefinedHardwareBehavior
from ..engine.model import FrameArray
from ..frames import ROUTE_MASK, work_fields
from ..hardware import (
    AXON_SKEW_MODULUS,
    AXON_SKEW_SIGN_BIT,
    BYTE_BITS,
    BYTE_MASK,
    OFFLINE_INPUT_BITS,
    OFFLINE_INPUT_ROWS,
    OFFLINE_PACKAGE_ALIGNMENTS,
    OFFLINE_PACKAGE_CAPACITIES,
    OFFLINE_PACKAGE_FACTORS,
    ONLINE_AXON_VALUES,
    ONLINE_INPUT_ROWS,
    ONLINE_PACKAGE_FACTORS,
)
from ..topology.board import TargetBoard, snapshot_board
from ..topology.routing import FrameRouter
from ..topology.types import ChipArray, CoreAddr, Port, ThreadId
from .errors import (
    Diagnostic,
    InvalidFrame,
    InvalidInput,
    ResourceLimit,
    SimulationFailure,
    UnsupportedFrame,
)
from .observability import (
    EventKind,
    PortDirection,
    SimEvent,
    SimEventHandler,
    TraceFilter,
)
from .scheduler import ThreadScheduler

if TYPE_CHECKING:
    from ..artifact import ArtifactView


@dataclass(slots=True)
class _Chunk:
    """One copied input array and its current word position."""

    ingress: Port
    words: FrameArray
    pos: int = 0


@dataclass(slots=True)
class _Packet:
    """Payload collected after a packet header, possibly across ``feed`` calls."""

    ingress: Port
    targets: tuple[CoreAddr, ...]
    header: FrameHeader
    raw_word: int
    kind: int
    start: int
    count: int
    payload: list[int]


class Simulator:
    """Deterministically execute PAICORE frames against one board snapshot.

    A simulator owns a detached :class:`BoardSnapshot`, runtime chips, thread
    state, and an output queue.  Construct it with a board declaration, call
    :meth:`feed` with one-dimensional ``np.uint64`` words at a declared CPU
    :class:`~paisim.topology.types.Port`, then call :meth:`run` (or bounded
    :meth:`advance`) and inspect :meth:`drain_output` or :meth:`snapshot`.

    The class never prints results.  Output frames are returned as a NumPy
    ``uint64`` array, state is returned as a detached dictionary, and optional
    events are delivered to the callback supplied at construction.  The board
    declaration is copied and validated once; changing the original board
    object later cannot mutate this simulator.
    """

    def __init__(
        self,
        board: TargetBoard,
        on_event: SimEventHandler | None = None,
        *,
        trace_filter: TraceFilter | None = None,
        max_pending_words: int = 1_000_000,
        auto_init: bool = True,
    ) -> None:
        """Create a cold simulator for ``board``.

        Args:
            board: A ``SingleBoard``, ``Array2x2Board`` or custom
                ``TargetBoard`` declaration.  Its CPU ingress and output ports
                become the only legal host endpoints.
            on_event: Optional callable receiving each selected ``SimEvent``
                after the corresponding state/output commit.
            trace_filter: Optional event selector.  It requires ``on_event``;
                an empty selector matches no events.
            max_pending_words: Host-side queue limit across all ``feed`` calls.
            auto_init: Stored for artifact construction; ordinary empty
                simulators still require configured threads before
                :meth:`initialize` can reset them.

        Raises:
            TypeError: If ``board`` is not a target board or an event filter is
                supplied without a callback.
            ValueError: If ``max_pending_words`` is less than one.
        """
        if max_pending_words < 1:
            raise ValueError("max_pending_words must be positive")

        self.board = snapshot_board(board.declaration)
        self.array = ChipArray(self.board.chips, self.board.links)
        self._cpu_addrs = frozenset(
            port.addr for port in (*self.board.ingress, *self.board.outputs)
        )
        self.on_event = on_event
        if trace_filter is not None and on_event is None:
            raise ValueError("trace_filter requires on_event")
        self.trace_filter = trace_filter
        self.auto_init = auto_init
        self.max_pending_words = max_pending_words
        self.router = FrameRouter(self.array, self.board.outputs)
        self.scheduler = ThreadScheduler(
            self.array,
            self.router,
            self._emit,
            self._complete,
            self._signal,
            self._core_step,
        )
        # Input chunks are copied at the API boundary; packet payloads may span
        # chunks, while output words remain an explicit drain-only queue.
        self._input: deque[_Chunk] = deque()
        self._packets: dict[Port, _Packet] = {}
        self._pending_words = 0
        self._outputs: list[int] = []
        self._sequence = 0
        self._prefer_thread = False
        self._artifact_roots: tuple[CoreAddr, ...] = ()
        self._artifact_view: ArtifactView | None = None

    @classmethod
    def from_artifact(
        cls,
        path: str | Path,
        board: TargetBoard,
        *,
        auto_init: bool = True,
        on_event: SimEventHandler | None = None,
        trace_filter: TraceFilter | None = None,
    ) -> "Simulator":
        """Load Protobuf configuration frames and return a ready simulator.

        The artifact is read once, fed through the normal frame path, checked
        against its thread/core metadata, and detached into ``sim.artifact``.
        ``auto_init=True`` (the default) resets every configured thread without
        emitting an internal COMPLETE frame.  Use ``auto_init=False`` to inspect
        the configured but cold state and call :meth:`initialize` yourself.

        Args:
            path: Protobuf compile-artifact file.  JSON is supported by
                :func:`paisim.artifact.read_artifact` but not by this
                convenience constructor.
            board: Explicit board declaration whose ingress and topology must
                match the artifact routes.
            auto_init: Whether to reset configured threads before returning.
            on_event: Optional callback for configuration and later execution.
            trace_filter: Optional filter applied to that callback.

        Returns:
            A simulator whose configuration frames have already been consumed.

        Raises:
            ValueError, InvalidFrame, or UnsupportedFrame when the artifact,
                metadata, routes, or board are incompatible.  No terminal
                output is printed or fabricated on failure.
        """
        from ..artifact.adapter import (
            ArtifactView,
            check_metadata,
            resolve_thread_roots,
        )
        from ..artifact.io import read_artifact

        frames, threads = read_artifact(path, format="protobuf")
        if not board.declaration.ingress:
            raise ValueError("artifact loading requires a declared CPU ingress")
        sim = cls(board, on_event, trace_filter=trace_filter, auto_init=auto_init)
        ingress = sim.board.ingress[0]
        sim.feed(frames, ingress=ingress)  # Feed the configuration frames
        sim.run()
        sim.finish_input()
        check_metadata(sim, threads, ingress=ingress)
        roots = resolve_thread_roots(sim, threads, ingress)
        sim._artifact_roots = tuple(
            root
            for _, root in sorted(
                zip(threads, roots, strict=True), key=lambda item: item[0].thread_id
            )
        )
        sim._artifact_view = ArtifactView(frames, tuple(threads))
        for root in sim._artifact_roots:
            sim.scheduler.thread(root)
        if auto_init:
            sim.initialize()
        return sim

    @property
    def artifact(self) -> ArtifactView | None:
        """Return detached artifact metadata, or ``None`` for raw-frame runs."""
        return self._artifact_view

    def initialize(self) -> None:
        """Reset every configured thread without emitting COMPLETE.

        Initialization clears dynamic input/core state and resets thread ticks.
        It is valid after ``from_artifact`` or manual CONFIG frames; an empty
        simulator raises ``ValueError`` because there is no thread to reset.
        """
        roots = self._artifact_roots or tuple(
            thread.tree.root for thread in self.scheduler.threads
        )
        if not roots:
            raise ValueError("cannot initialize a simulator with no configured threads")
        trees = [self.scheduler.thread(root).tree for root in sorted(set(roots))]
        for tree in trees:
            self.scheduler.init(tree.root, emit_complete=False)

    @property
    def pending_words(self) -> int:
        """Return the number of queued input words not yet consumed."""
        return self._pending_words

    @property
    def pending_steps(self) -> int:
        """Return the sum of remaining SYNC budgets across configured threads."""
        return sum(thread.budget for thread in self.scheduler.threads)

    def feed(self, words: FrameArray, ingress: Port) -> None:
        """Validate and queue frame words without advancing simulation state.

        ``words`` must be a one-dimensional ``np.uint64`` array.  Values are
        copied into the simulator; no reshaping or implicit scalar conversion
        is performed.  A CONFIG/WORK packet may span multiple calls, but only
        its complete payload is committed.  Call :meth:`run` or
        :meth:`advance` to consume the queued words.

        Args:
            words: One-dimensional unsigned 64-bit frame words.
            ingress: A CPU ``Port`` declared by the board.

        Raises:
            InvalidInput: For malformed values, dimensions, or dtypes.
            ResourceLimit: If the queue would exceed ``max_pending_words``.
            ValueError: If ``ingress`` is not a declared board ingress.
        """
        if ingress not in self.board.ingress:
            raise ValueError("ingress is not a declared board CPU port")
        available = self.max_pending_words - self._pending_words
        try:
            if not isinstance(words, np.ndarray):
                raise TypeError("frame array must be a 1d uint64 ndarray")
            if words.ndim != 1 or words.dtype != FRAME_DTYPE:
                raise TypeError("frame array must be a 1d uint64 ndarray")
            if words.size > available:
                raise self._resource_limit()
            data = words.copy()
        except ResourceLimit:
            raise
        except (TypeError, ValueError, OverflowError) as exc:
            raise InvalidInput(Diagnostic("invalid_input", str(exc), "api")) from exc
        if data.size:
            self._input.append(_Chunk(ingress, data))
            self._pending_words += data.size

    def advance(self, max_events: int = 1) -> int:
        """Advance input and runnable threads by at most ``max_events`` events.

        One event is either an input transaction or one deterministic thread
        step.  The method returns the number actually performed, which may be
        smaller when the simulator becomes idle.  It does not print or clear
        output frames.  To associate output with one scheduler step, call
        ``advance(1)`` and then ``drain_output()``; ``run()`` intentionally
        aggregates all output until it becomes idle.
        """
        if max_events < 0:
            raise ValueError("max_events must be nonnegative")
        done = 0
        while done < max_events:
            if self._prefer_thread:
                thread = self.scheduler.advance()
                if thread is not None:
                    self._prefer_thread = False
                    done += 1
                    self._notify("step", None, None, thread=thread.id)
                    continue
            if self._input_event():
                self._prefer_thread = self.scheduler.runnable
                done += 1
                continue
            thread = self.scheduler.advance()
            if thread is not None:
                done += 1
                self._notify("step", None, None, thread=thread.id)
                continue
            break
        return done

    def run(self, max_events: int | None = None) -> int:
        """Run until idle, or until an optional event budget is exhausted.

        Returns the number of input/compute events executed.  A bounded run
        leaves remaining input, thread budgets, and output in place so the
        caller can continue with another call.  Use :meth:`drain_output` when
        the response words are needed.  The return value and output array do
        not contain per-timestep grouping; use :meth:`advance` with a step
        event callback when that grouping is required.
        """
        if max_events is not None and max_events < 0:
            raise ValueError("max_events must be nonnegative")
        remaining = None if max_events is None else max_events
        done = 0
        while remaining is None or remaining:
            advanced = self.advance(1 if remaining is not None else 65_536)
            if not advanced:
                break
            done += advanced
            if remaining is not None:
                remaining -= advanced
        return done

    def finish_input(self) -> None:
        """Assert that no input words or partial packet remain queued.

        This is a validation barrier, not an execution step.  It raises
        ``ValueError`` when words have not been advanced and ``InvalidFrame``
        for a truncated packet; a complete input stream returns ``None``.
        """
        if self._input:
            raise ValueError("queued input has not been advanced")
        if self._packets:
            packet = next(iter(self._packets.values()))
            raise InvalidFrame(
                Diagnostic(
                    "invalid_packet",
                    f"truncated package: received {len(packet.payload)}/{packet.count}",
                    "packet",
                    raw_word=packet.raw_word,
                    recoverable=True,
                    effect="unchanged",
                )
            )

    def drain_output(self) -> FrameArray:
        """Return and clear raw response words as a one-dimensional ``uint64`` array.

        Responses are not printed and are not retained after this call.  The
        returned array is a new NumPy container, so later simulation cannot
        change its length or values.
        """
        result = np.asarray(self._outputs, dtype=FRAME_DTYPE)
        self._outputs.clear()
        return result

    def snapshot(self, selection: Iterable[CoreAddr] | None = None) -> dict[str, Any]:
        """Return a detached observation of board, threads, queues, and cores.

        ``selection`` limits the ``cores`` mapping to the given ``CoreAddr``
        values; ``None`` includes every configured core.  NumPy arrays in the
        result are copied and marked read-only.  The dictionary contains
        ``board``, ``sequence``, ``pending_words``, ``pending_steps``,
        ``threads`` and ``cores`` keys and is intended for inspection or
        serialization, not for mutating simulator state.
        """
        chosen = None if selection is None else frozenset(selection)
        cores: dict[CoreAddr, dict[str, Any]] = {}
        for chip in self.array.iter_chips():
            for coord, core in sorted(chip.cores.items()):
                addr = CoreAddr(chip.coord, coord)
                if chosen is not None and addr not in chosen:
                    continue
                item: dict[str, Any] = {
                    "type": type(core).__name__,
                    "revision": core.revision,
                    "send": core.send,
                    "receive": core.receive,
                    "thread": core.thread_id,
                }
                if isinstance(core, (OfflineCore, OnlineCore)):
                    item.update(
                        registers=self._copy(core.registers),
                        neuron_sram=self._copy(core.neuron_sram),
                        input_sram=self._copy(core.input_sram),
                        lut=self._copy(core.lut.array()),
                        config=None if core.config is None else dict(core.config),
                    )
                if isinstance(core, OfflineCore):
                    item["work_steps"] = core.work_steps
                    item["voltage"] = (
                        None if core.voltage is None else self._copy(core.voltage)
                    )
                cores[addr] = item
        threads = {
            thread.id: {
                "root": thread.tree.root,
                "tick": thread.tick,
                "budget": thread.budget,
                "busy": thread.busy,
                "done": thread.done,
                "compute": thread.tree.compute,
                "relays": thread.tree.relays,
                "completions": thread.tree.completions,
            }
            for thread in self.scheduler.threads
        }
        return {
            "board": self.board,
            "sequence": self._sequence,
            "pending_words": self.pending_words,
            "pending_steps": self.pending_steps,
            "threads": threads,
            "cores": cores,
        }

    def _input_event(self) -> bool:
        while self._input and self._input[0].pos == len(self._input[0].words):
            self._input.popleft()
        if not self._input:
            return False
        chunk = self._input[0]
        packet = self._packets.get(chunk.ingress)
        if packet is not None:
            size = min(len(chunk.words) - chunk.pos, packet.count - len(packet.payload))
            values = [int(value) for value in chunk.words[chunk.pos : chunk.pos + size]]
            packet.payload.extend(values)
            chunk.pos += size
            self._pending_words -= size
            for value in values:
                self._notify(
                    "payload",
                    chunk.ingress.addr,
                    None,
                    value,
                    port=chunk.ingress,
                    raw_word=value,
                )
            if len(packet.payload) == packet.count:
                del self._packets[chunk.ingress]
                try:
                    self._commit_packet(packet)
                except (InvalidFrame, UnsupportedFrame):
                    raise
                except UndefinedHardwareBehavior as exc:
                    raise self._unsupported(
                        str(exc),
                        "configuration",
                        packet.raw_word,
                        packet.ingress.addr,
                        packet.targets[0],
                    ) from exc
                except (TypeError, ValueError, NotImplementedError) as exc:
                    raise self._invalid(
                        str(exc),
                        "configuration",
                        packet.raw_word,
                        packet.ingress.addr,
                        packet.targets[0],
                    ) from exc
            return True
        word = int(chunk.words[chunk.pos])
        chunk.pos += 1
        self._pending_words -= 1
        self._header(chunk.ingress, word)
        return True

    def _header(self, ingress: Port, word: int) -> None:
        header = FrameHeader(
            (word >> FFV2.GENERAL_HEADER_OFFSET) & FFV2.GENERAL_HEADER_MASK
        )
        frame_type = FrameType(
            (word >> FFV2.GENERAL_FRAME_TYPE_OFFSET) & FFV2.GENERAL_FRAME_TYPE_MASK
        )
        package_type = FramePackageType(
            (word >> FFV2.GENERAL_PACKAGE_TYPE_OFFSET) & FFV2.GENERAL_PACKAGE_TYPE_MASK
        )
        try:
            targets = self.router.route(ingress.addr, word)
        except NotImplementedError as exc:
            raise self._unsupported(str(exc), "route", word, ingress.addr) from exc
        except ValueError as exc:
            raise self._invalid(str(exc), "route", word, ingress.addr) from exc
        if not targets:
            raise self._invalid(
                "frame route has no destination", "route", word, ingress.addr
            )
        if frame_type is FrameType.CONFIG:
            if package_type is not FramePackageType.CONF_TESTOUT:
                raise self._invalid(
                    "CONFIG requires package_type=0", "packet", word, ingress.addr
                )
            if self.scheduler.runnable:
                raise self._unsupported(
                    "configuration while a thread is runnable is unsupported",
                    "configuration",
                    word,
                    ingress.addr,
                )
            packet = _Packet(
                ingress,
                targets,
                header,
                word,
                header.value & FFV2.GENERAL_FRAME_SUBTYPE_MASK,
                (word >> FFV2.GENERAL_PACKAGE_NEU_START_ADDR_OFFSET)
                & FFV2.GENERAL_PACKAGE_NEU_START_ADDR_MASK,
                (word >> FFV2.GENERAL_PACKAGE_NUM_OFFSET)
                & FFV2.GENERAL_PACKAGE_NUM_MASK,
                [],
            )
            self._preflight_packet(packet)
            self._packets[ingress] = packet
            for target in targets:
                self._notify(
                    "frame", ingress.addr, target, word, port=ingress, raw_word=word
                )
            return
        if frame_type is FrameType.TEST:
            if package_type is not FramePackageType.TESTIN:
                raise self._invalid(
                    "TEST requires package_type=1", "packet", word, ingress.addr
                )
            kind = header.value & FFV2.GENERAL_FRAME_SUBTYPE_MASK
            start = (word >> FFV2.GENERAL_PACKAGE_NEU_START_ADDR_OFFSET) & (
                FFV2.GENERAL_PACKAGE_NEU_START_ADDR_MASK
            )
            count = (word >> FFV2.GENERAL_PACKAGE_NUM_OFFSET) & (
                FFV2.GENERAL_PACKAGE_NUM_MASK
            )
            for target in targets:
                self._readback(ingress.addr, target, word, kind, start, count)
            return
        if frame_type is FrameType.WORK:
            self._write_works(ingress.addr, targets, word)
            return
        if len(targets) != 1:
            raise self._unsupported(
                "multicast control roots are unsupported", "control", word, ingress.addr
            )
        root = targets[0]
        self._notify("frame", ingress.addr, root, word, port=ingress, raw_word=word)
        if header is FrameHeader.CTRL_TYPE1:
            try:
                self.scheduler.sync(
                    root,
                    (word >> OfflineControlFrame1FormatV2.N_TIMESTEP_OFFSET)
                    & OfflineControlFrame1FormatV2.N_TIMESTEP_MASK,
                )
            except NotImplementedError as exc:
                raise self._unsupported(
                    str(exc), "control", word, ingress.addr, root
                ) from exc
            except (KeyError, ValueError) as exc:
                raise self._invalid(
                    str(exc), "control", word, ingress.addr, root
                ) from exc
        elif header is FrameHeader.CTRL_TYPE2:
            try:
                self.scheduler.init(root)
            except (KeyError, ValueError, NotImplementedError) as exc:
                raise self._invalid(
                    str(exc), "control", word, ingress.addr, root
                ) from exc
        elif header is FrameHeader.CTRL_TYPE4:
            core = self._core(root)
            if isinstance(core, OnlineCore):
                raise self._unsupported(
                    "online UPDATE numerical execution is unsupported",
                    "control",
                    word,
                    ingress.addr,
                    root,
                )
            raise self._invalid(
                "UPDATE targets an offline core", "control", word, ingress.addr, root
            )
        else:
            raise self._invalid(
                "COMPLETE is not an ingress command", "control", word, ingress.addr
            )

    def _preflight_packet(self, packet: _Packet) -> None:
        for target in packet.targets:
            kind = self._target_kind(target)
            existing = self._maybe_core(target)
            if kind == "online":
                raise self._unsupported(
                    "online core execution is unsupported",
                    "configuration",
                    packet.raw_word,
                    packet.ingress.addr,
                    target,
                )
            if isinstance(existing, (RelayCore, CompletionCore)):
                raise self._unsupported(
                    "empty cores have no configurable compute storage",
                    "configuration",
                    packet.raw_word,
                    packet.ingress.addr,
                    target,
                )
            if packet.kind == 0 and isinstance(existing, OfflineCore):
                raise self._invalid(
                    "configured core cannot be replaced",
                    "configuration",
                    packet.raw_word,
                    packet.ingress.addr,
                    target,
                )
            factor = OFFLINE_PACKAGE_FACTORS[packet.kind]
            capacity = OFFLINE_PACKAGE_CAPACITIES[packet.kind]
            alignment = OFFLINE_PACKAGE_ALIGNMENTS[packet.kind]
            offset = packet.start * factor
            if packet.count < 1 or offset + packet.count > capacity:
                raise self._invalid(
                    "package range exceeds target storage",
                    "configuration",
                    packet.raw_word,
                    packet.ingress.addr,
                    target,
                )
            if packet.kind == 0 and (packet.start != 0 or packet.count != capacity):
                raise self._invalid(
                    f"{kind} core configuration requires {capacity} words at zero",
                    "configuration",
                    packet.raw_word,
                    packet.ingress.addr,
                    target,
                )
            if packet.kind in (2, 3) and packet.count % alignment:
                raise self._invalid(
                    f"storage package count must be a multiple of {alignment}",
                    "configuration",
                    packet.raw_word,
                    packet.ingress.addr,
                    target,
                )

    def _commit_packet(self, packet: _Packet) -> None:
        words = np.asarray(packet.payload, np.uint64)
        self._preflight_packet(packet)
        if packet.kind == 0:
            probe = OfflineCore(packet.targets[0])
            probe.configure(words)
        for target in packet.targets:
            core = self._ensure_core(target)
            factor = OFFLINE_PACKAGE_FACTORS[packet.kind]
            core.write(packet.kind, packet.start * factor, words)
        if packet.kind == 0:
            self.scheduler.invalidate()

    def _readback(
        self,
        source: CoreAddr,
        target: CoreAddr,
        word: int,
        kind: int,
        start: int,
        count: int,
    ) -> None:
        core = self._core(target)
        if not isinstance(core, (OfflineCore, OnlineCore)):
            raise self._unsupported(
                "empty cores have no readable SRAM", "packet", word, source, target
            )
        packet = _Packet(
            Port(source.chip, source.core),
            (target,),
            FrameHeader(
                (word >> FFV2.GENERAL_HEADER_OFFSET) & FFV2.GENERAL_HEADER_MASK
            ),
            word,
            kind,
            start,
            count,
            [],
        )
        self._preflight_packet(packet)
        bits = self._return_route(core)
        package_type_mask = (
            FFV2.GENERAL_PACKAGE_TYPE_MASK << FFV2.GENERAL_PACKAGE_TYPE_OFFSET
        )
        response = word & ~(ROUTE_MASK | package_type_mask) | bits
        storage = self._storage(core, kind)
        factor = (
            ONLINE_PACKAGE_FACTORS[kind]
            if isinstance(core, OnlineCore)
            else OFFLINE_PACKAGE_FACTORS[kind]
        )
        offset = start * factor
        for destination in self.router.route(target, bits):
            if not self.router.is_output(destination):
                raise self._unsupported(
                    "TEST response destination is not an output port",
                    "route",
                    response,
                    target,
                    destination,
                )
            delivered = self.router.delivered(response)
            self._outputs.append(delivered)
            self._notify(
                "frame",
                target,
                destination,
                delivered,
                port=self._output_port(destination),
                raw_word=response,
            )
            for payload in storage[offset : offset + count]:
                payload = int(payload)
                self._outputs.append(payload)
                self._notify(
                    "payload",
                    target,
                    destination,
                    payload,
                    port=self._output_port(destination),
                    raw_word=payload,
                )

    def _prepare_work(
        self, source: CoreAddr, target: CoreAddr, word: int
    ) -> tuple[OfflineCore | OnlineCore | None, int, int]:
        if self.router.is_output(target):
            return None, 0, 0
        core = self._core(target)
        if not isinstance(core, (OfflineCore, OnlineCore)) or core.config is None:
            raise self._invalid(
                "WORK target is not a configured compute core",
                "work",
                word,
                source,
                target,
            )
        if isinstance(core, OnlineCore):
            from paicorelib.framelib.frame_defs import OnlineWorkFrame1FormatV2

            slot = (word >> OnlineWorkFrame1FormatV2.TIMESTEP_OFFSET) & (
                OnlineWorkFrame1FormatV2.TIMESTEP_MASK
            )
            axon = (word >> OnlineWorkFrame1FormatV2.AXON_ADDR_OFFSET) & (
                OnlineWorkFrame1FormatV2.AXON_ADDR_MASK
            )
            data = (word >> OnlineWorkFrame1FormatV2.DATA_OFFSET) & (
                OnlineWorkFrame1FormatV2.DATA_MASK
            )
            voltage = FrameHeader(
                (word >> FFV2.GENERAL_HEADER_OFFSET) & FFV2.GENERAL_HEADER_MASK
            ) in (FrameHeader.WORK_TYPE3, FrameHeader.WORK_TYPE4)
        else:
            slot, axon, data, voltage = work_fields(word)
        width = BYTE_BITS if voltage else 1 << core.config["input_width"]
        skew = core.config.get("axon_skew", 0)
        skew = skew - AXON_SKEW_MODULUS if skew & AXON_SKEW_SIGN_BIT else skew
        limit = (
            OFFLINE_INPUT_BITS if isinstance(core, OfflineCore) else ONLINE_AXON_VALUES
        )
        slots = (
            OFFLINE_INPUT_ROWS if isinstance(core, OfflineCore) else ONLINE_INPUT_ROWS
        )
        address = (slot * limit + axon + skew) % (slots * limit)
        if address % width:
            raise self._invalid(
                f"unaligned {width}-bit WORK address", "work", word, source, target
            )
        byte, shift = divmod(address, BYTE_BITS)
        memory = core.input_sram.reshape(-1)
        mask = ((1 << width) - 1) << shift
        if shift + width > BYTE_BITS:
            raise self._invalid(
                "WORK field crosses an input byte", "work", word, source, target
            )
        value = (int(memory[byte]) & (BYTE_MASK ^ mask)) | ((data << shift) & mask)
        return core, byte, value

    def _write_works(
        self, source: CoreAddr, targets: tuple[CoreAddr, ...], word: int
    ) -> None:
        if len(targets) == 1:
            target = targets[0]
            core, byte, value = self._prepare_work(source, target, word)
            if core is None:
                delivered = self.router.delivered(word)
                self._outputs.append(delivered)
                self._notify(
                    "frame",
                    source,
                    target,
                    delivered,
                    port=self._output_port(target),
                    raw_word=word,
                )
            else:
                core.input_sram.reshape(-1)[byte] = value
                core.revision += 1
                self._notify("frame", source, target, word, raw_word=word)
            return
        writes = [self._prepare_work(source, target, word) for target in targets]
        for target, (core, byte, value) in zip(targets, writes, strict=True):
            if core is None:
                delivered = self.router.delivered(word)
                self._outputs.append(delivered)
                self._notify(
                    "frame",
                    source,
                    target,
                    delivered,
                    port=self._output_port(target),
                    raw_word=word,
                )
            else:
                core.input_sram.reshape(-1)[byte] = value
                core.revision += 1
                self._notify("frame", source, target, word, raw_word=word)

    def _emit(self, source: CoreAddr, word: int) -> None:
        self._write_works(source, self.router.route(source, word), word)

    def _signal(self, source: CoreAddr, target: CoreAddr, thread: ThreadId) -> None:
        self._notify("signal", source, target, thread=thread)

    def _core_step(self, core: CoreAddr, thread: ThreadId, tick: int) -> None:
        self._notify("core_step", None, None, thread=thread, tick=tick, core=core)

    def _complete(self, root: CoreAddr, thread: ThreadId) -> None:
        core = self._core(root)
        if not isinstance(core, (OfflineCore, OnlineCore)):
            raise ValueError("thread root must be a configured compute core")
        word = (
            FrameHeader.CTRL_TYPE3.value << FFV2.GENERAL_HEADER_OFFSET
            | self._return_route(core)
            | (int(thread) & OfflineControlFrame3FormatV2.THREAD_ID_MASK)
            << OfflineControlFrame3FormatV2.THREAD_ID_OFFSET
        )
        for target in self.router.route(root, word):
            if not self.router.is_output(target):
                raise ValueError("COMPLETE destination is not an output port")
            delivered = self.router.delivered(word)
            self._outputs.append(delivered)
            self._notify(
                "complete",
                root,
                target,
                delivered,
                thread,
                port=self._output_port(target),
                raw_word=word,
            )

    @staticmethod
    def _return_route(core: OfflineCore | OnlineCore) -> int:
        from paicorelib.framelib.parser_v2 import sign_magnitude_to_int

        from ..frames import route_bits

        assert core.config is not None
        values = (
            core.config["test_core_xy"],
            core.config["test_core_x"],
            core.config["test_core_y"],
        )
        values = tuple(sign_magnitude_to_int(value) for value in values)
        return route_bits(values + (0, 0, 0))

    @staticmethod
    def _storage(core: OfflineCore | OnlineCore, kind: int) -> FrameArray:
        if kind == 0:
            return core.registers.reshape(-1)
        if kind == 1:
            return core.lut.array().reshape(-1)
        if kind == 2:
            return core.neuron_sram.reshape(-1)
        return core.input_sram.reshape(-1).view("<u8")

    def _target_kind(self, addr: CoreAddr) -> Literal["offline", "online"]:
        if addr.core.cpu or addr in self._cpu_addrs:
            raise self._invalid(
                "CPU is not a core configuration target", "configuration", target=addr
            )
        return "online" if addr.core.online else "offline"

    def _ensure_core(self, addr: CoreAddr) -> OfflineCore | OnlineCore:
        current = self._maybe_core(addr)
        if current is None:
            current = OnlineCore(addr) if addr.core.online else OfflineCore(addr)
            self.array.chip(addr.chip).put(current)
        if not isinstance(current, (OfflineCore, OnlineCore)):
            raise ValueError("empty-core role was configured explicitly")
        return current

    def _maybe_core(self, addr: CoreAddr) -> Core | None:
        return self.array.chip(addr.chip).cores.get(addr.core)

    def _core(self, addr: CoreAddr) -> Core:
        try:
            return self.array.chip(addr.chip).core(addr.core)
        except KeyError as exc:
            raise self._invalid(
                f"unconfigured core {addr}", "route", target=addr
            ) from exc

    def _output_port(self, addr: CoreAddr) -> Port:
        for port in self.board.outputs:
            if port.addr == addr:
                return port
        raise ValueError(f"unregistered output port {addr}")

    def _notify(
        self,
        kind: EventKind,
        source: CoreAddr | None,
        destination: CoreAddr | None,
        word: int | None = None,
        thread: ThreadId | None = None,
        *,
        tick: int | None = None,
        core: CoreAddr | None = None,
        port: Port | None = None,
        raw_word: int | None = None,
    ) -> None:
        self._sequence += 1
        if tick is None and thread is not None:
            found = next(
                (item for item in self.scheduler.threads if item.id == thread), None
            )
            tick = None if found is None else found.tick
        direction = self._direction(source, destination, port)
        event = SimEvent(
            sequence=self._sequence,
            kind=kind,
            direction=direction,
            source=source,
            destination=destination,
            port=port,
            word=word,
            raw_word=raw_word,
            thread=thread,
            tick=tick,
            core=core,
        )
        if self.on_event is None or not self._matches(event):
            return
        try:
            self.on_event(event)
        except Exception as exc:
            effect: Literal["prefix_preserved", "step_partial"] = (
                "step_partial"
                if kind in {"core_step", "step", "complete"}
                else "prefix_preserved"
            )
            diagnostic = Diagnostic(
                "observer_error",
                f"event callback failed: {exc}",
                "observer",
                source=None if source is None else (source.core.x, source.core.y),
                target=None
                if destination is None
                else (destination.core.x, destination.core.y),
                tick=tick,
                effect=effect,
            )
            raise SimulationFailure(diagnostic) from exc

    @staticmethod
    def _direction(
        source: CoreAddr | None, destination: CoreAddr | None, port: Port | None
    ) -> PortDirection | None:
        if port is not None:
            return PortDirection.E2I if source == port.addr else PortDirection.I2E
        if source is not None and destination is not None:
            return PortDirection.I2I
        return None

    def _matches(self, event: SimEvent) -> bool:
        trace = self.trace_filter
        if trace is None:
            return True
        return trace.matches(event)

    @staticmethod
    def _copy(array: np.ndarray) -> np.ndarray:
        result = array.copy()
        result.flags.writeable = False
        return result

    @staticmethod
    def _invalid(
        message: str,
        stage: Literal[
            "api",
            "packet",
            "route",
            "configuration",
            "work",
            "control",
            "compute",
            "observer",
            "replay",
        ] = "packet",
        word: int | None = None,
        source: CoreAddr | None = None,
        target: CoreAddr | None = None,
    ) -> InvalidFrame:
        return InvalidFrame(
            Diagnostic(
                "invalid_state"
                if stage in ("control", "compute")
                else "invalid_packet",
                message,
                stage,
                raw_word=word,
                source=None if source is None else (source.core.x, source.core.y),
                target=None if target is None else (target.core.x, target.core.y),
            )
        )

    @staticmethod
    def _unsupported(
        message: str,
        stage: Literal[
            "api",
            "packet",
            "route",
            "configuration",
            "work",
            "control",
            "compute",
            "observer",
            "replay",
        ],
        word: int | None = None,
        source: CoreAddr | None = None,
        target: CoreAddr | None = None,
    ) -> UnsupportedFrame:
        return UnsupportedFrame(
            Diagnostic(
                "unsupported_behavior",
                message,
                stage,
                raw_word=word,
                source=None if source is None else (source.core.x, source.core.y),
                target=None if target is None else (target.core.x, target.core.y),
            )
        )

    @staticmethod
    def _resource_limit() -> ResourceLimit:
        return ResourceLimit(
            Diagnostic(
                "resource_limit",
                "pending frame budget exceeded",
                "api",
                recoverable=True,
                effect="unchanged",
            )
        )
