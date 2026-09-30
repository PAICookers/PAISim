"""Per-thread signal trees and deterministic cooperative scheduling."""

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from paicorelib.core_defs_v2 import AddPotentialMode, ZeroOutputMode

from ..engine.core import CompletionCore, OfflineCore, OnlineCore, RelayCore
from ..engine.model import StepResult
from ..engine.numeric import step
from ..frames import work_word
from ..hardware import (
    BYTE_BITS,
    BYTE_MASK,
    CORE_COORD_MIN,
    CORE_GRID_COORD_MAX,
    DIRECT_POTENTIAL_BITS,
    GLOBAL_SEND_DIRECTION_BITS,
    GLOBAL_SEND_ROOT_BIT,
    OFFLINE_AXON_VALUES,
    OFFLINE_WORK_TIMESTEP_COUNT,
    POTENTIAL_LANE_AXON_STRIDE,
    POTENTIAL_LANES,
    SUPPORTED_DATA_WIDTHS,
)
from ..topology.routing import FrameRouter
from ..topology.types import (
    ChipArray,
    ChipCoord,
    CoreAddr,
    CoreCoord,
    SignalLink,
    ThreadId,
)

_SIGNAL_DIRECTIONS = ((1, 1), (-1, -1), (1, 0), (-1, 0), (0, 1), (0, -1))


@dataclass(frozen=True, slots=True)
class SignalTree:
    """Physical global-signal connectivity rooted at one configured core."""

    root: CoreAddr
    nodes: tuple[CoreAddr, ...]
    compute: tuple[CoreAddr, ...]
    relays: tuple[CoreAddr, ...]
    completions: tuple[CoreAddr, ...]
    _edges: tuple[tuple[CoreAddr, CoreAddr], ...] = field(repr=False, compare=False)
    _tree_links: tuple[SignalLink, ...] = field(repr=False, compare=False)
    _array: ChipArray = field(repr=False, compare=False)

    def activate(
        self,
        on_signal: Callable[[CoreAddr, CoreAddr, ThreadId], None] | None = None,
        thread: ThreadId | None = None,
    ) -> None:
        """Notify the callback for each physical signal edge in this tree."""
        if on_signal is not None:
            for source, target in self._edges:
                if thread is None:
                    raise ValueError("signal activation requires a thread id")
                on_signal(source, target, thread)

    @classmethod
    def build(
        cls,
        array: ChipArray,
        root: CoreAddr,
        on_signal: Callable[[CoreAddr, CoreAddr, ThreadId], None] | None = None,
        *,
        activate: bool = True,
    ) -> "SignalTree":
        """Build and optionally activate a validated tree rooted at ``root``.

        The traversal follows configured global-send/receive bits across local
        cores and declared board links.  It classifies compute, relay, and
        completion members without imposing a thread-ownership filter on the
        physical signal graph.
        """
        first = array.chip(root.chip).core(root.core)
        root_thread = first.thread_id
        queue = deque([root])
        seen: set[CoreAddr] = set()
        compute: list[CoreAddr] = []
        relays: list[CoreAddr] = []
        completions: list[CoreAddr] = []
        links: list[SignalLink] = []
        edges: list[tuple[CoreAddr, CoreAddr]] = []
        while queue:
            addr = queue.popleft()
            if addr in seen:
                continue
            seen.add(addr)
            core = array.chip(addr.chip).core(addr.core)
            if core.send & GLOBAL_SEND_ROOT_BIT:
                if isinstance(core, (OfflineCore, OnlineCore)):
                    compute.append(addr)
                elif isinstance(core, CompletionCore):
                    completions.append(addr)
            if isinstance(core, RelayCore):
                relays.append(addr)
            for bit, (dx, dy) in zip(
                GLOBAL_SEND_DIRECTION_BITS, _SIGNAL_DIRECTIONS, strict=True
            ):
                if not core.send & (1 << bit):
                    continue
                x, y = addr.core.x + dx, addr.core.y + dy
                if (
                    CORE_COORD_MIN <= x <= CORE_GRID_COORD_MAX
                    and CORE_COORD_MIN <= y <= CORE_GRID_COORD_MAX
                ):
                    target = CoreAddr(addr.chip, CoreCoord(x, y))
                    try:
                        other = array.chip(target.chip).core(target.core)
                    except KeyError as exc:
                        raise ValueError(
                            f"unmatched global signal edge {addr}->{target}"
                        ) from exc
                    if not other.receive & (1 << (bit ^ 1)):
                        raise ValueError(
                            f"unmatched global signal edge {addr}->{target}"
                        )
                    edges.append((addr, target))
                    queue.append(target)
                    continue
                if dx and dy:
                    raise NotImplementedError(
                        "XY global signal cannot cross a chip boundary"
                    )
                neighbor = CoreCoord(
                    min(CORE_GRID_COORD_MAX, max(CORE_COORD_MIN, x)),
                    min(CORE_GRID_COORD_MAX, max(CORE_COORD_MIN, y)),
                )
                chip_x = addr.chip.x + (
                    1 if x > CORE_GRID_COORD_MAX else -1 if x < CORE_COORD_MIN else 0
                )
                chip_y = addr.chip.y + (
                    1 if y > CORE_GRID_COORD_MAX else -1 if y < CORE_COORD_MIN else 0
                )
                target_chip = ChipCoord(chip_x, chip_y)
                if not array.has_chip(target_chip):
                    raise ValueError(f"global signal leaves the target board at {addr}")
                link = array.link(addr.chip, target_chip)
                if link not in links:
                    links.append(link)
                source = CoreAddr(addr.chip, neighbor)
                for target in link.fanout(source):
                    try:
                        other = array.chip(target.chip).core(target.core)
                    except KeyError:
                        continue
                    if other.receive & (1 << (bit ^ 1)):
                        edges.append((addr, target))
                        queue.append(target)
        if not compute and not completions:
            raise ValueError("signal tree has no local participant")
        tree = cls(
            root,
            tuple(sorted(seen)),
            tuple(sorted(compute)),
            tuple(sorted(relays)),
            tuple(sorted(completions)),
            tuple(edges),
            tuple(links),
            array,
        )
        if activate:
            tree.activate(on_signal, root_thread)
        return tree


@dataclass(slots=True)
class CoreThread:
    """Mutable scheduler state for one root and its signal tree."""

    id: ThreadId
    tree: SignalTree
    tick: int = 0
    budget: int = 0
    busy: bool = False
    done: bool = True

    @property
    def runnable(self) -> bool:
        """Return whether this thread has positive remaining SYNC budget."""
        return self.budget > 0


@dataclass(slots=True)
class _Commit:
    core: OfflineCore
    result: StepResult
    slots: NDArray[np.int64]


class ThreadScheduler:
    """Deterministic round-robin scheduler for configured signal trees.

    Input transactions remain ordered by ``Simulator``; this class only selects
    runnable threads and commits one step at a time in stable ``ThreadId``
    order.  It keeps no event history.
    """

    def __init__(
        self,
        array: ChipArray,
        router: FrameRouter,
        deliver: Callable[[CoreAddr, int], None],
        complete: Callable[[CoreAddr, ThreadId], None],
        signal: Callable[[CoreAddr, CoreAddr, ThreadId], None] | None = None,
        core_step: Callable[[CoreAddr, ThreadId, int], None] | None = None,
    ) -> None:
        self.array = array
        self.router = router
        self._deliver = deliver
        self._complete = complete
        self._signal = signal
        self._core_step = core_step
        self._threads: dict[ThreadId, CoreThread] = {}
        self._runnable: set[ThreadId] = set()
        # Configuration revisions invalidate cached signal trees; cursor keeps
        # round-robin order stable without creating a worker thread.
        self._config_revision = 0
        self._tree_revisions: dict[CoreAddr, int] = {}
        self._cursor = 0

    @property
    def threads(self) -> tuple[CoreThread, ...]:
        """Return configured threads sorted by numeric thread identifier."""
        return tuple(self._threads[key] for key in sorted(self._threads))

    @property
    def runnable(self) -> bool:
        """Return whether at least one thread has work remaining."""
        return bool(self._runnable)

    def invalidate(self) -> None:
        """Invalidate cached trees after a configuration mutation."""
        self._config_revision += 1

    def thread(self, root: CoreAddr, tree: SignalTree | None = None) -> CoreThread:
        """Get or create the thread rooted at ``root`` and validate its identity."""
        core = self.array.chip(root.chip).core(root.core)
        ident = core.thread_id
        current = self._threads.get(ident)
        if current is not None and current.tree.root != root:
            raise ValueError(f"thread id {ident} has more than one root")
        if tree is None:
            tree = SignalTree.build(self.array, root, self._signal)
        else:
            tree.activate(self._signal, ident)
        if current is None or current.tree != tree:
            current = CoreThread(ident, tree)
            self._threads[ident] = current
            self.array.chip(root.chip).threads[ident] = current
        self._tree_revisions[root] = self._config_revision
        return current

    def sync(self, root: CoreAddr, count: int) -> None:
        """Validate a SYNC tree and add positive work to its budget."""
        if count < 0:
            raise ValueError("SYNC count must be nonnegative")
        core = self.array.chip(root.chip).core(root.core)
        current = self._threads.get(core.thread_id)
        if (
            current is not None
            and current.tree.root == root
            and self._tree_revisions.get(root) == self._config_revision
        ):
            tree = current.tree
        else:
            tree = SignalTree.build(self.array, root, activate=False)
        if count > 0 and any(
            isinstance(self.array.chip(addr.chip).core(addr.core), OnlineCore)
            for addr in tree.compute
        ):
            raise NotImplementedError("online SYNC numerical execution is unsupported")
        if current is not None and current.tree is tree:
            tree.activate(self._signal, core.thread_id)
            thread = current
        else:
            thread = self.thread(root, tree)
        if count == 0:
            self._complete(root, thread.id)
            return
        thread.budget += count
        self._runnable.add(thread.id)
        thread.busy = True
        thread.done = False

    def init(self, root: CoreAddr, *, emit_complete: bool = True) -> None:
        """Reset all compute members of one thread and optionally emit COMPLETE."""
        thread = self.thread(root)
        for addr in thread.tree.compute:
            core = self.array.chip(addr.chip).core(addr.core)
            if isinstance(core, OfflineCore):
                core.initialize(external=True)
            else:
                assert isinstance(core, OnlineCore)
                core.initialize()
        thread.tick = 0
        thread.busy = False
        thread.done = True
        if emit_complete:
            self._complete(root, thread.id)

    def advance(self) -> CoreThread | None:
        """Run one deterministic step on the next runnable thread, if any."""
        runnable = [self._threads[ident] for ident in sorted(self._runnable)]
        if not runnable:
            return None
        self._cursor %= len(runnable)
        thread = runnable[self._cursor]
        self._cursor = (self._cursor + 1) % len(runnable)
        self._step(thread)
        return thread

    def _step(self, thread: CoreThread) -> None:
        thread.tick += 1
        commits: list[_Commit] = []
        for addr in thread.tree.compute:
            core = self.array.chip(addr.chip).core(addr.core)
            assert isinstance(core, OfflineCore)
            decoded = core.decoded
            config = decoded.config
            start, duration = config["tick_start"], config["tick_duration"]
            if (
                start == 0
                or thread.tick < start
                or (duration and core.work_steps >= duration)
            ):
                continue
            initial = config["tick_initial"]
            if initial and core.work_steps and core.work_steps % initial == 0:
                core.initialize(external=False)
            core_config = core.config
            assert core_config is not None
            # Every scheduler read follows the same LCN mapping as direct core runs.
            inputs, slots = core.input_window.values(
                core.work_steps,
                core_config["lcn"],
                width=SUPPORTED_DATA_WIDTHS[core_config["input_width"]],
                signed=bool(core_config["input_sign"]),
                direct=core_config["add_potential"] == AddPotentialMode.DIRECT_ADD,
            )
            assert core.voltage is not None
            result = step(decoded, inputs, core.voltage, core.previous_positive)
            commits.append(_Commit(core, result, slots))
        deliveries: list[tuple[CoreAddr, int]] = []
        for commit in commits:
            core, result = commit.core, commit.result
            decoded, config = core.decoded, core.decoded.config
            core.voltage = result.voltage
            core.previous_positive = result.positive
            core.store_voltage()
            core.input_sram[commit.slots] = 0
            nonzero = np.flatnonzero(
                (result.output != 0)
                | (config["zero_output"] == ZeroOutputMode.ENABLE)
                | (decoded.output_types != 0)
            )
            for neuron in nonzero:
                item = int(neuron)
                route = core.output_routes[item]
                slot = (
                    core.work_steps * (1 << config["target_lcn"])
                    + int(decoded.ticks[item])
                ) % OFFLINE_WORK_TIMESTEP_COUNT
                axon = int(decoded.axons[item])
                if decoded.output_types[item]:
                    value = int(result.potential[item]) & (
                        (1 << DIRECT_POTENTIAL_BITS) - 1
                    )
                    for lane in range(POTENTIAL_LANES):
                        address = (
                            slot * OFFLINE_AXON_VALUES
                            + axon
                            + lane * POTENTIAL_LANE_AXON_STRIDE
                        )
                        deliveries.append(
                            (
                                core.addr,
                                work_word(
                                    route,
                                    (address // (OFFLINE_AXON_VALUES))
                                    % OFFLINE_WORK_TIMESTEP_COUNT,
                                    address % OFFLINE_AXON_VALUES,
                                    value >> (lane * BYTE_BITS) & BYTE_MASK,
                                    voltage=True,
                                ),
                            )
                        )
                else:
                    value = int(result.output[item]) & (
                        (1 << SUPPORTED_DATA_WIDTHS[config["output_width"]]) - 1
                    )
                    deliveries.append((core.addr, work_word(route, slot, axon, value)))
            core.work_steps += 1
            core.revision += 1
        for source, word in deliveries:
            self._deliver(source, word)
        if self._core_step is not None:
            for commit in commits:
                self._core_step(commit.core.addr, thread.id, thread.tick)
        thread.budget -= 1
        thread.busy = thread.budget > 0
        if thread.budget == 0:
            self._runnable.discard(thread.id)
            thread.done = True
            self._complete(thread.tree.root, thread.id)
