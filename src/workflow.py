"""Workflow orchestration helpers and CLI-facing entrypoints."""

import json
import os
import time
import uuid
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from .llm_client import LLMClient
from .models import (
    Draft,
    EditPlan,
    EvaluatorConfig,
    FactCheckIssue,
    FactCheckResult,
    FinalDecision,
    Revision,
    RubricDimensionScore,
    RubricEvaluation,
    RubricScoresForDraft,
    RunConfig,
    RunId,
    RunLog,
    SectionInstruction,
    Task,
    Uncertainty,
    WorkerConfig,
    ReuseSuggestion,
)
from .prompts import (
    build_factchecker_prompts,
    build_final_judge_prompts,
    build_revision_worker_prompts,
    build_rubric_prompts,
    build_synthesizer_prompts,
    build_task_normalizer_prompts,
    build_worker_prompts,
    worker_persona,
)


def now_run_id() -> RunId:
    ts = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    shortid = uuid.uuid4().hex[:6]
    return f"{ts}_{shortid}"


def default_workers() -> List[WorkerConfig]:
    return [
        WorkerConfig(id="WorkerA", display_name="Architect", persona=worker_persona("WorkerA")),
        WorkerConfig(id="WorkerB", display_name="Pragmatist", persona=worker_persona("WorkerB")),
        WorkerConfig(id="WorkerC", display_name="Skeptic", persona=worker_persona("WorkerC")),
        WorkerConfig(id="WorkerD", display_name="Communicator", persona=worker_persona("WorkerD")),
    ]


def default_evaluators() -> List[EvaluatorConfig]:
    return [
        EvaluatorConfig(id="Eval1", role="FactChecker"),
        EvaluatorConfig(id="Eval2", role="RubricScorer"),
        EvaluatorConfig(id="Eval3", role="Synthesizer"),
        EvaluatorConfig(id="FinalJudge", role="Final selection judge"),
    ]


def normalize_task(client: LLMClient, user_prompt: str) -> Task:
    system, user = build_task_normalizer_prompts(user_prompt)
    resp = client.chat_json(system, user)
    brief = resp.get("brief", "").strip()
    constraints = resp.get("constraints", [])
    success = resp.get("success_criteria", [])
    if not isinstance(constraints, list):
        constraints = [str(constraints)]
    if not isinstance(success, list):
        success = [str(success)]
    return Task(
        user_prompt=user_prompt,
        normalized_brief=brief,
        constraints=[str(c) for c in constraints],
        success_criteria=[str(s) for s in success],
    )


def call_workers(client: LLMClient, workers: List[WorkerConfig], task: Task) -> List[Draft]:
    drafts: List[Draft] = []
    for w in workers:
        system, user = build_worker_prompts(w, task)
        resp = client.chat_json(system, user)
        raw_uncs = resp.get("uncertainties", [])
        if not isinstance(raw_uncs, list):
            raw_uncs = []
        uncertainties = [
            Uncertainty(
                id=str(u.get("id", f"{w.id}_u{i}")),
                description=str(u.get("description", "")),
                type=str(u.get("type", "other")),
                impact=str(u.get("impact", "medium")),
            )
            for i, u in enumerate(raw_uncs)
        ]
        draft_id = f"{w.id}_v1"
        draft = Draft(
            draft_id=draft_id,
            worker_id=w.id,
            version=1,
            content=str(resp.get("draft", "")),
            uncertainties=uncertainties,
        )
        drafts.append(draft)
    return drafts


def call_factchecker(client: LLMClient,
                     evaluator: EvaluatorConfig,
                     task: Task,
                     drafts: List[Draft]) -> List[FactCheckResult]:
    system, user = build_factchecker_prompts(task, drafts)
    resp = client.chat_json(system, user)
    results = []
    for r in resp.get("results", []):
        issues_raw = r.get("issues", [])
        issues = [
            FactCheckIssue(
                severity=str(i.get("severity", "minor")),
                location_hint=str(i.get("location_hint", "")),
                description=str(i.get("description", "")),
                type=str(i.get("type", "other")),
            )
            for i in issues_raw
        ]
        results.append(
            FactCheckResult(
                evaluator_id=evaluator.id,
                draft_id=str(r.get("draft_id", "")),
                issues=issues,
                overall_confidence=float(r.get("overall_confidence", 0.0)),
                summary=str(r.get("summary", "")),
            )
        )
    return results


def parse_rubric_evaluation(evaluator_id: str, resp: Dict[str, Any]) -> RubricEvaluation:
    dims = [str(d) for d in resp.get("dimensions", [])]
    per_draft: List[RubricScoresForDraft] = []
    for rd in resp.get("per_draft", []):
        ds = [
            RubricDimensionScore(
                name=str(d.get("name", "")),
                score=float(d.get("score", 0.0)),
                justification=str(d.get("justification", "")),
            )
            for d in rd.get("dimension_scores", [])
        ]
        per_draft.append(
            RubricScoresForDraft(
                draft_id=str(rd.get("draft_id", "")),
                dimension_scores=ds,
                overall_score=float(rd.get("overall_score", 0.0)),
                summary=str(rd.get("summary", "")),
            )
        )
    ranking = [str(d) for d in resp.get("ranking", [])]
    return RubricEvaluation(
        evaluator_id=evaluator_id,
        dimensions=dims,
        per_draft=per_draft,
        ranking=ranking,
        rationale_for_ranking=str(resp.get("rationale_for_ranking", "")),
    )


def call_rubric_scorer(client: LLMClient,
                       evaluator: EvaluatorConfig,
                       task: Task,
                       drafts: List[Draft],
                       fact_checks: List[FactCheckResult],
                       phase: str) -> RubricEvaluation:
    system, user = build_rubric_prompts(task, drafts, fact_checks, phase=phase)
    resp = client.chat_json(system, user)
    return parse_rubric_evaluation(evaluator.id, resp)


def call_synthesizer(client: LLMClient,
                     evaluator: EvaluatorConfig,
                     task: Task,
                     drafts: List[Draft],
                     fact_checks: List[FactCheckResult],
                     rubric: RubricEvaluation) -> EditPlan:
    system, user = build_synthesizer_prompts(task, drafts, fact_checks, rubric)
    resp = client.chat_json(system, user)
    chosen_base = str(resp.get("chosen_base_draft", ""))
    global_strategy = str(resp.get("global_strategy", ""))
    section_instructions_raw = resp.get("section_instructions", [])
    reuse_raw = resp.get("reuse_suggestions", [])
    open_q = [str(x) for x in resp.get("open_questions", [])]

    section_instructions = [
        SectionInstruction(
            section_label=str(si.get("section_label", "")),
            base_from_draft=si.get("base_from_draft"),
            actions=[str(a) for a in si.get("actions", [])],
            notes=str(si.get("notes", "")),
        )
        for si in section_instructions_raw
    ]
    reuse_suggestions = [
        ReuseSuggestion(
            from_draft=str(rs.get("from_draft", "")),
            what_to_reuse=str(rs.get("what_to_reuse", "")),
        )
        for rs in reuse_raw
    ]
    return EditPlan(
        evaluator_id=evaluator.id,
        chosen_base_draft=chosen_base,
        global_strategy=global_strategy,
        section_instructions=section_instructions,
        reuse_suggestions=reuse_suggestions,
        open_questions=open_q,
    )


def call_revision_workers(client: LLMClient,
                          workers: List[WorkerConfig],
                          task: Task,
                          initial_drafts: List[Draft],
                          edit_plan: EditPlan) -> List[Revision]:
    draft_by_worker = {d.worker_id: d for d in initial_drafts}
    revisions: List[Revision] = []
    for w in workers:
        own = draft_by_worker[w.id]
        system, user = build_revision_worker_prompts(w, task, own, edit_plan)
        resp = client.chat_json(system, user)
        revised = str(resp.get("revised_draft", ""))
        change_summary_raw = resp.get("change_summary", [])
        if not isinstance(change_summary_raw, list):
            change_summary_raw = [str(change_summary_raw)]
        updated_uncs_raw = resp.get("updated_uncertainties", [])
        if not isinstance(updated_uncs_raw, list):
            updated_uncs_raw = []
        updated_uncs = [
            Uncertainty(
                id=str(u.get("id", f"{w.id}_v2_u{i}")),
                description=str(u.get("description", "")),
                type=str(u.get("type", "other")),
                impact=str(u.get("impact", "medium")),
            )
            for i, u in enumerate(updated_uncs_raw)
        ]
        new_draft_id = f"{w.id}_v2"
        revisions.append(
            Revision(
                draft_id=new_draft_id,
                from_draft_id=own.draft_id,
                worker_id=w.id,
                version=2,
                content=revised,
                change_summary=[str(s) for s in change_summary_raw],
                updated_uncertainties=updated_uncs,
            )
        )
    return revisions


def call_final_judge(client: LLMClient,
                     evaluator: EvaluatorConfig,
                     task: Task,
                     revisions: List[Revision]) -> FinalDecision:
    system, user = build_final_judge_prompts(task, revisions)
    resp = client.chat_json(system, user)
    rubric_eval = parse_rubric_evaluation(evaluator.id, resp)
    winner = str(resp.get("winner_draft_id", ""))
    reasoning = str(resp.get("reasoning", ""))
    return FinalDecision(
        evaluator_id=evaluator.id,
        winner_draft_id=winner,
        ranking=rubric_eval.ranking,
        reasoning=reasoning,
        rubric_evaluation=rubric_eval,
    )


def run_workflow(user_prompt: str,
                 model: str = "gpt-5.1",
                 temperature: float = 0.7,
                 seed: Optional[int] = None,
                 max_tokens_per_call: Optional[int] = None) -> RunLog:
    run_id = now_run_id()
    workers = default_workers()
    evaluators = default_evaluators()

    config = RunConfig(
        run_id=run_id,
        model=model,
        temperature=temperature,
        seed=seed,
        max_tokens_per_call=max_tokens_per_call,
        workers=workers,
        evaluators=evaluators,
    )

    client = LLMClient(model=model, temperature=temperature,
                       seed=seed, max_tokens=max_tokens_per_call)

    task = normalize_task(client, user_prompt)
    initial_drafts = call_workers(client, workers, task)

    eval1 = next(e for e in evaluators if e.id == "Eval1")
    eval2 = next(e for e in evaluators if e.id == "Eval2")
    eval3 = next(e for e in evaluators if e.id == "Eval3")
    final_eval = next(e for e in evaluators if e.id == "FinalJudge")

    fact_checks = call_factchecker(client, eval1, task, initial_drafts)
    rubric_initial = call_rubric_scorer(client, eval2, task, initial_drafts, fact_checks, phase="initial")
    edit_plan = call_synthesizer(client, eval3, task, initial_drafts, fact_checks, rubric_initial)

    revisions = call_revision_workers(client, workers, task, initial_drafts, edit_plan)
    final_decision = call_final_judge(client, final_eval, task, revisions)

    return RunLog(
        config=config,
        task=task,
        initial_drafts=initial_drafts,
        fact_checks=fact_checks,
        rubric_evaluation_initial=rubric_initial,
        edit_plan=edit_plan,
        revisions=revisions,
        final_decision=final_decision,
    )


def save_runlog(runlog: RunLog, out_dir: str = "runs") -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"run_{runlog.config.run_id}.json")

    def default(o):
        serializable = (
            RunLog, RunConfig, Task, Draft, FactCheckResult, RubricEvaluation,
            EditPlan, Revision, FinalDecision, Uncertainty, FactCheckIssue,
            RubricDimensionScore, RubricScoresForDraft, SectionInstruction, ReuseSuggestion
        )
        if isinstance(o, serializable):
            return asdict(o)
        raise TypeError(f"Type not serializable: {type(o)}")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(runlog, f, indent=2, default=default)
    return path
