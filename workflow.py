#!/usr/bin/env python3
"""CLI entrypoint for the AgentFlow orchestrator."""

import argparse

from src.workflow import run_workflow, save_runlog


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-worker, multi-evaluator agentic workflow"
    )
    parser.add_argument("prompt", help="User task prompt (string)")
    parser.add_argument("--model", default="gpt-5.1", help="Model name")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--out-dir", default="runs")
    args = parser.parse_args()

    runlog = run_workflow(
        user_prompt=args.prompt,
        model=args.model,
        temperature=args.temperature,
        seed=args.seed,
        max_tokens_per_call=args.max_tokens,
    )
    path = save_runlog(runlog, out_dir=args.out_dir)
    print(f"Saved run log to {path}")


if __name__ == "__main__":
    main()
