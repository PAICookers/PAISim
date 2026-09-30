# PAISim implementation rules

## Code

- Use short, explicit `snake_case` names; avoid vague public names such as
  `*Spec` and `EventSink`.
- Do not use `from __future__ import annotations`.
- Public NumPy arrays use fixed dtypes and `numpy.typing.NDArray`.
- Use `TraceFilter`, `SimEvent`, and `SimEventHandler` for observation APIs.
  Callback failure must not pretend that simulator state rolled back.
- Library code must not print, configure global logging, or retain default
  event history.
- Search callers before changing a contract. Do not add a no-op base class,
  event bus, or compatibility layer for one implementation.
- Keep one canonical state owner. Views and mutable handles may expose it, but
  must not duplicate state beside the backing arrays or SRAM.
- Before implementing or extending an offline/online core, build a register
  width and signedness table from the authoritative hardware document and map
  every field through decode, storage, arithmetic, write-back, and frame output.
  Choose working dtypes from the modeled hardware arithmetic width, preserve
  wider temporaries only when an operation requires them, and add boundary
  tests for maximum values, signed values, and overflow without accidental
  narrowing. Treat this audit as a design gate, not a post-failure cleanup.

## Verification

Choose checks from the files touched. For implementation changes, run the
smallest focused tests first, then the full suite when shared behavior changes:

```bash
uv run pytest
uv run ruff format --check src scripts tests
uv run ruff check src scripts tests
uv build
```

For docs/config-only changes, run `git diff --check`, link/privacy scans, and
any relevant document checker; do not claim source verification was performed.

## Documentation and privacy

- Shared project facts belong in `README.md`, `docs/`, or root `AGENTS.md` and
  use repository-relative links.
- Never commit machine-specific absolute paths, usernames, private dataset
  paths, temporary directories, raw logs, credentials, or unsanitized dumps.
- Separate hardware facts, software design, test evidence, and open questions;
  mark evidence status explicitly.
- Put detailed profiling, debugging, and historical material in ignored
  `docs-local/` or `evidence-local/`.

## Codebase memory

The tracked `.codebase-memory/graph.db.zst` indexes only `src/` and `tests/`.
Use the codebase-memory MCP `index_status` first, then `search_graph`,
`search_code`, `trace_path`, or `get_code_snippet` for structural questions.
After structural source/test changes, refresh the index and check for parse
gaps. Do not treat an unavailable index as evidence that code is absent.

## Task records and git

- For a non-trivial task, record a short plan, workspace decision, allowed and
  blocked files, and verification in `tasks/todo.md`; compress stale completed
  history instead of appending indefinitely.
- After a user correction, add one reusable prevention rule to
  `tasks/lessons.md`.
- Never use destructive git commands to clean unrelated work. Keep edits
  scoped, review the diff, and report unrun checks honestly.
