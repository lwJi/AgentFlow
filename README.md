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

### Alternate LLM endpoints

By default the orchestrator uses OpenAI’s Chat Completions API. To target another OpenAI-compatible endpoint, set:

- `LLM_BASE_URL` – full URL to POST (e.g., `https://myendpoint/v1/chat/completions`)
- `LLM_API_KEY` – bearer token passed as `Authorization: Bearer ...`

When these variables are present the workflow sends the same payload structure over HTTP using `requests`, still expecting `choices[0].message.content` with JSON text.

## Development hints

- Keep the JSON schema stable unless user-approved; see `AGENTS.md`.
- When editing prompts, ensure they match the expected JSON structure and keep persona intent intact.
- Run `python -m pytest` to execute the existing tests under `tests/`, including serialization checks and the generic LLM payload test.
