# PAISim repository guidance

## Scope

PAISim is a deterministic, frame-driven PAICORE functional simulator. The
core models raw frames, hardware state transitions, routing, and optional
observation. Tensor/application codecs, accuracy comparison, physical timing,
compiler behavior, and live-board behavior are outside the core boundary unless
the task explicitly expands it.

## Evidence and claims

- Keep conclusions at the layer actually exercised: source structure, unit
  tests, mock or synthetic input, saved artifacts, simulator execution,
  compiler output, and live hardware are different evidence classes.
- Do not infer compiler success, application accuracy, numerical equivalence,
  or hardware behavior from a simulator or saved-file result.
- Treat stage-3 boards, threads, and online numerical features as explicit work,
  not capabilities implied by application examples.

## Working safely

- Inspect callers, tests, and the current worktree before changing a contract.
- Preserve unrelated dirty files and local evidence. Do not reset, overwrite,
  or create compatibility shims without an explicit need.
- Keep shared documentation in `README.md`, `docs/`, and this file. Keep local
  profiling, logs, paths, and raw dumps untracked under `docs-local/` or
  `evidence-local/`.
- Core tests live under `tests/artifact/`, `tests/engine/`, `tests/protocol/`,
  `tests/runtime/`, and `tests/topology/`. Shared test builders belong beside
  their relevant tests in `fixtures.py`; `paisim.frames` is a source module,
  while frame/protocol tests live in `tests/protocol/`.

## Rule routing

This file is the repository-wide automatic entrypoint. For code edits, first
read `.codex/project-rules.md`; it contains implementation conventions and the
verification commands. `.codex/README.md` explains this boundary and the
session-specific routing. Neither supplemental file is automatically injected
by Codex merely because it is under `.codex/`.

## Verification

Run `uv run pytest -q` for the repository test suite. Ruff checks target `src/`
and `tests/`.
