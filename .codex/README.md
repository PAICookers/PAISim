# Supplemental Codex rules

The root `AGENTS.md` is the automatic repository entrypoint. Codex discovers
`AGENTS.md` files from the global directory and from the repository root to the
current working directory, then applies them in order. Files under `.codex/`
are not part of that automatic discovery.

Read `project-rules.md` explicitly before editing Python source, tests, schemas,
generated bindings, or build configuration. For a read-only or documentation-
only task, use only the relevant sections of `AGENTS.md` and do not impose
code-edit checks that cannot provide evidence for the requested change.

Keep this directory for agent-facing supplemental guidance, not secrets,
credentials, logs, generated output, or user data.
