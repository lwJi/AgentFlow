# AgentFlow

AgentFlow orchestrates a four-worker, three-evaluator workflow that turns a single user prompt into a fully logged reasoning trace and final answer.

## Layout

- `workflow.py` – CLI entrypoint. Parse args, call `run_workflow`, and save a run log under `runs/`.
- `src/models.py` – Dataclasses defining the RunLog schema (`Task`, `Draft`, `EditPlan`, etc.).
- `src/prompts.py` – Prompt builders for workers and evaluators; keeps personas and JSON formats centralized.
- `src/llm_client.py` – Thin OpenAI Chat API wrapper that enforces JSON responses.
- `src/workflow.py` – Orchestration logic: normalization, drafting, evaluation rounds, revisions, and final decision.

## Usage

```bash
python3 workflow.py "Design a plan for..."
```

Arguments:

- `--model` (default `gpt-5.1`)
- `--temperature` (float, default `0.7`)
- `--seed` (int, optional)
- `--max-tokens` (int, optional)
- `--out-dir` (default `runs/`)

Output: `runs/run_<timestamp>_<id>.json` following the schema in `src/models.py`.

## Development hints

- Keep the JSON schema stable unless user-approved; see `AGENTS.md`.
- When editing prompts, ensure they match the expected JSON structure and keep persona intent intact.
- Add tests under `tests/` (not yet present) that import from `src.workflow`.
