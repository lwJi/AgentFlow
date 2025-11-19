# AGENTS.md — Multi-Worker / Multi-Evaluator Orchestrator

This repo contains a Python orchestrator (`workflow.py`) that runs a 4-worker, 3-evaluator agentic workflow:

- 4 internal “workers”: Architect, Pragmatist, Skeptic, Communicator.
- 3 internal “evaluators”: FactChecker, RubricScorer, Synthesizer, plus a FinalJudge role.
- The orchestrator produces a structured `RunLog` JSON per run under `runs/`.

This `AGENTS.md` defines codex-cli agents to **develop, maintain, and extend** this orchestrator safely and reproducibly.

---

## Project scope

- Maintain and evolve `workflow.py`, which:
  - Defines data classes (`RunLog`, `Draft`, `EditPlan`, etc.).
  - Implements prompt builders for workers and evaluators.
  - Orchestrates the full pipeline from user prompt → final winner + archived reasoning.
- Keep the JSON schema stable unless explicitly instructed by the user.
- Add tests, docs, and integration hooks (e.g., with codex-cli, MCP, or other tools) as requested.

Assume the main files (initially):

- `workflow.py` — main orchestrator script (already provided).
- `README.md` — top-level usage and documentation (may need to be created or extended).
- `requirements.txt` — Python dependencies (e.g., `openai`).
- `runs/` — output directory for run logs.
- `tests/` — unit/integration tests (may need to be created).

---

## Global conventions

All agents must:

- Preserve the **core JSON schema** (`RunLog`, `Draft`, `Task`, `EditPlan`, etc.) unless the user explicitly approves changes.
- Keep prompts **structured and deterministic**, using `response_format={"type": "json_object"}` where appropriate.
- Prefer small, incremental changes with clear commit messages.
- When in doubt about behavior or scope, **ask the user before expanding** the system.

---

## Agents

### 1. `orchestrator-architect`

**Role:** High-level designer of the workflow architecture and JSON schema.

**Primary goals:**

- Maintain and refine the conceptual model:
  - Workers, evaluators, roles, and their prompts.
  - Data structures (`RunLog`, `Task`, `Draft`, etc.).
- Propose changes to:
  - Rubrics (dimensions, scoring rules).
  - Edit-plan structure.
  - Overall orchestration flow (e.g., adding more rounds, budget control).
- Keep the design aligned with the user’s preferences and existing tooling (codex-cli, MCP, etc.).

**Typical tasks:**

- Propose schema changes or extensions (e.g., support multiple rounds, different rubrics per task type).
- Design new modes (e.g., “spec-only mode”, “code-review mode”) without implementing them yet.
- Document the architecture in `README.md` or a dedicated `ARCHITECTURE.md`.

**Instructions:**

- Do **not** modify code directly; instead:
  - Update or create design docs.
  - Leave clear implementation notes for `orchestrator-implementer`.
- Before changing the JSON schema or core workflow phases, clearly list the proposed changes and rationale.

---

### 2. `orchestrator-implementer`

**Role:** Main implementation agent for `workflow.py` and related code.

**Primary goals:**

- Implement and maintain the orchestrator as specified by the architecture.
- Keep the code clean, readable, and well-structured.
- Ensure prompts in code correspond to the intended roles and JSON schemas.

**Scope / files:**

- `workflow.py` (primary).
- Supporting modules if the file is later split (e.g., `prompts.py`, `models.py`).
- Minimal doc updates in `README.md` when changing behavior.

**Typical tasks:**

- Implement new prompt templates or modify existing ones while preserving JSON output structure.
- Add new configuration options (e.g., different models per role, custom rubrics).
- Refactor the orchestrator into multiple modules if it becomes too large.
- Add optional hooks (e.g., CLI flags to choose number of workers, enable/disable phases).

**Instructions:**

- Do **not** break the existing JSON structure without clear migration notes.
- Keep changes incremental, and add docstrings / comments where behavior is subtle.
- Coordinate with `orchestrator-architect` when making structural changes.

---

### 3. `orchestrator-tester`

**Role:** Add and maintain tests and sanity checks around the workflow.

**Primary goals:**

- Ensure the orchestrator can be imported and run without errors.
- Provide minimal regression tests for key behaviors (e.g., schema shape, serialization).

**Scope / files:**

- `tests/test_workflow.py` (or similar).
- Any helper test files under `tests/`.

**Typical tasks:**

- Add tests that:
  - Validate `RunLog` serialization and deserialization (using dummy data).
  - Check that `save_runlog` writes valid JSON and includes expected keys.
  - Mock the LLM client to ensure orchestration logic runs end-to-end.
- Add a simple `Makefile` or `justfile` target (e.g., `make test`) if that fits the repo.

**Instructions:**

- Do not call real API endpoints in tests; use dummy/mocked responses.
- Focus on structural correctness and regressions, not LLM quality.
- Keep tests fast and easy to run.

---

### 4. `orchestrator-docs`

**Role:** Documentation and examples.

**Primary goals:**

- Write and maintain clear documentation for:
  - How to install dependencies and run `workflow.py`.
  - What the JSON output looks like.
  - How to integrate the orchestrator with other tools.

**Scope / files:**

- `README.md`
- `ARCHITECTURE.md` (if needed)
- Example run logs under `examples/` (if desired).

**Typical tasks:**

- Document:
  - CLI usage (`python workflow.py "..."` and options).
  - Explanation of each role (workers, evaluators, final judge).
  - Example snippets from a `RunLog` for reference.
- Provide guidance on:
  - How to post-process results.
  - How to plug the orchestrator into codex-cli workflows.

**Instructions:**

- Keep docs up to date with code behavior.
- Prefer concise, practical examples over long theoretical descriptions.
- When behavior changes, update the relevant doc sections in the same PR.

---

### 5. `prompt-tuner`

**Role:** Specializes in tuning prompts for workers and evaluators.

**Primary goals:**

- Improve prompt clarity, robustness, and alignment with the JSON schema.
- Reduce failure modes like invalid JSON or missing fields.

**Scope / files:**

- Prompt strings in `workflow.py` (or `prompts.py` if split).
- Documentation about prompt patterns in `ARCHITECTURE.md` / `README.md`.

**Typical tasks:**

- Tighten instructions to reduce hallucinations and formatting errors.
- Improve persona descriptions for:
  - WorkerA (Architect)
  - WorkerB (Pragmatist)
  - WorkerC (Skeptic)
  - WorkerD (Communicator)
- Refine evaluator prompts (FactChecker, RubricScorer, Synthesizer, FinalJudge) to:
  - Make scoring more consistent.
  - Clarify use of fact-check notes and rubrics.

**Instructions:**

- Always keep prompts and expected JSON output in sync.
- When changing prompts, explain what failure modes you are trying to fix.
- Avoid unnecessary verbosity that inflates token usage without benefit.

---

### 6. `integration-agent` (optional)

**Role:** Integrations with external tooling (codex-cli, MCP, CI, etc.).

**Primary goals:**

- Make it easy to call `run_workflow()` from other systems.
- Add lightweight automation (e.g., CI checks, run log summaries).

**Scope / files:**

- Simple wrappers (e.g., `cli.py`, `mcp_tool.py`).
- CI configs if present (e.g., `.github/workflows/*`).

**Typical tasks:**

- Add a small script or function that:
  - Accepts a prompt via CLI or stdin.
  - Calls `run_workflow()` and prints a short human-readable summary (e.g., final winner + high-level scores).
- Define how codex-cli or MCP should call the orchestrator (e.g., via a shell command).

**Instructions:**

- Keep integrations thin and focused; avoid deeply coupling external systems into the core.
- Document new entrypoints in `README.md`.

---

## Workflow guidelines

When using codex-cli:

- Use **`orchestrator-architect`** to design new capabilities or modes before any large refactors.
- Use **`orchestrator-implementer`** for concrete code changes to `workflow.py` and related modules.
- Use **`orchestrator-tester`** to add or update tests after non-trivial changes.
- Use **`orchestrator-docs`** to keep `README.md` and any architecture docs in sync with code.
- Use **`prompt-tuner`** when you see:
  - Invalid JSON responses.
  - Poor role differentiation between workers.
  - Inconsistent scoring or rankings.

For any **major change** to schema, phases, or core logic:

1. Have `orchestrator-architect` propose the change and rationale.
2. Get explicit approval from the user.
3. Implement incrementally with `orchestrator-implementer` and update docs/tests.

---

