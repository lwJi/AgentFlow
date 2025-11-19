"""Prompt builders for workers and evaluators."""

import json
import os
from dataclasses import asdict
from typing import List, Tuple

from .models import (
    Draft,
    EditPlan,
    EvaluatorConfig,
    FactCheckResult,
    Revision,
    RubricEvaluation,
    SectionInstruction,
    Task,
    WorkerConfig,
)


def build_task_normalizer_prompts(user_prompt: str) -> Tuple[str, str]:
    system = """You are a task normalizer. Given a raw user prompt, you must produce:
- A concise, neutral brief (what is being asked).
- A list of explicit constraints.
- A list of success criteria (what "good" looks like).

Always answer as strict JSON with keys:
  "brief": string
  "constraints": string[]
  "success_criteria": string[]

Do not include any other keys."""
    user = f"Here is the raw prompt from the user:\n\n{user_prompt}\n\nNormalize it now."
    return system, user


def worker_persona(worker_id: str) -> str:
    if worker_id == "WorkerA":
        return ("You prioritize structure, decomposition, and clear interfaces/abstractions. "
                "You propose step-by-step plans and clean designs. You care about internal consistency.")
    if worker_id == "WorkerB":
        return ("You optimize for concrete, usable output: examples, commands, code, specific steps. "
                "You care more about getting something that works than about perfect theory.")
    if worker_id == "WorkerC":
        return ("You relentlessly look for edge cases, missing assumptions, and things that could go wrong. "
                "You still produce a full draft, but you actively highlight weak spots and risks.")
    if worker_id == "WorkerD":
        return ("You optimize for clarity, pedagogy, and naming. Your draft should be easy to read "
                "for someone smart but not deeply familiar with the context.")
    return "You are a capable assistant."


def build_worker_prompts(worker: WorkerConfig, task: Task) -> Tuple[str, str]:
    persona = worker_persona(worker.id)
    system = f"""You are {worker.id}, one of several independent workers. You DO NOT see other workers' drafts.

Persona:
{persona}

You are given:
- BRIEF: {task.normalized_brief}
- CONSTRAINTS (must obey all):
{os.linesep.join(f"- {c}" for c in task.constraints)}
- SUCCESS CRITERIA (optimize for these):
{os.linesep.join(f"- {s}" for s in task.success_criteria)}

Your goals:
1. Produce your best draft solution.
2. Explicitly list your uncertainties, assumptions, and potential failure modes.

Output format (MUST be valid JSON with these exact keys):
{{
  "draft": "full draft here as markdown or plain text",
  "uncertainties": [
    {{
      "id": "short_machine_friendly_id",
      "description": "what you are unsure about",
      "type": "assumption | missing_info | ambiguity | risk | other",
      "impact": "low | medium | high"
    }}
  ]
}}"""
    user = "Create your draft and uncertainty list now."
    return system, user


def build_factchecker_prompts(task: Task, drafts: List[Draft]) -> Tuple[str, str]:
    system = f"""You are Eval1, the FactChecker.

You are given:
- BRIEF: {task.normalized_brief}
- CONSTRAINTS:
{os.linesep.join(f"- {c}" for c in task.constraints)}
- SUCCESS CRITERIA:
{os.linesep.join(f"- {s}" for s in task.success_criteria)}

You will see multiple drafts, each with:
- "draft_id"
- "content" (the full draft)
- "uncertainties" reported by the worker.

Your tasks:
1. For each draft, identify factual issues, unsupported claims, and constraint violations.
2. Summarize how trustworthy the draft is overall.

For each draft, produce:
- "issues": list of objects with keys:
    - "severity": "minor" | "moderate" | "major"
    - "location_hint": short quote or section reference
    - "description": what is wrong or risky
    - "type": "factual_error" | "unsupported_claim" | "constraint_violation" | "inconsistency" | "other"
- "overall_confidence": number from 0 to 10
- "summary": 1–3 sentences summarizing trustworthiness.

Output strict JSON of the form:
{{
  "results": [
    {{
      "draft_id": "string",
      "issues": [ ... ],
      "overall_confidence": 0,
      "summary": "..."
    }}
  ]
}}

Do not include any other keys."""
    simple_drafts = [{
        "draft_id": d.draft_id,
        "content": d.content,
        "uncertainties": [asdict(u) for u in d.uncertainties],
    } for d in drafts]
    user = "Here are the drafts (as JSON under key 'drafts'):\n\n" + json.dumps(
        {"drafts": simple_drafts}, indent=2
    )
    return system, user


def build_rubric_prompts(task: Task,
                         drafts: List[Draft],
                         fact_checks: List[FactCheckResult],
                         phase: str = "initial") -> Tuple[str, str]:
    system = f"""You are Eval2, the RubricScorer.

You are given:
- BRIEF: {task.normalized_brief}
- CONSTRAINTS:
{os.linesep.join(f"- {c}" for c in task.constraints)}
- SUCCESS CRITERIA:
{os.linesep.join(f"- {s}" for s in task.success_criteria)}

Rubric dimensions (0–10 each):
1. "correctness"
2. "coverage"
3. "clarity"
4. "practicality"
5. "risk_handling"

Definitions:
- correctness: factual / logical soundness; alignment with constraints.
- coverage: how fully it addresses the brief and success criteria.
- clarity: organization, naming, readability.
- practicality: how implementable or actionable it is.
- risk_handling: how well it handles uncertainties, risks, and edge cases.

You also receive:
- "drafts": list of drafts.
- "fact_checks": fact-check results per draft (may be empty).

Your tasks:
1. Score each draft on the 5 dimensions (0–10).
2. Provide a 1–3 sentence summary for each draft.
3. Compute an "overall_score" for each draft (0–100; you may weight dimensions equally).
4. Produce a ranking of draft_ids from best to worst, with rationale.

Output strict JSON:
{{
  "dimensions": ["correctness", "coverage", "clarity", "practicality", "risk_handling"],
  "per_draft": [
    {{
      "draft_id": "...",
      "dimension_scores": [
        {{"name": "correctness", "score": 0, "justification": "..."}},
        ...
      ],
      "overall_score": 0,
      "summary": "..."
    }}
  ],
  "ranking": ["best_draft_id", "..."],
  "rationale_for_ranking": "..."
}}"""
    simple_drafts = [{
        "draft_id": d.draft_id,
        "content": d.content,
        "uncertainties": [asdict(u) for u in d.uncertainties],
    } for d in drafts]
    simple_fact_checks = [{
        "draft_id": fc.draft_id,
        "issues": [asdict(i) for i in fc.issues],
        "overall_confidence": fc.overall_confidence,
        "summary": fc.summary,
    } for fc in fact_checks]
    user_payload = {"drafts": simple_drafts, "fact_checks": simple_fact_checks, "phase": phase}
    user = "Here are the drafts and fact-check results:\n\n" + json.dumps(user_payload, indent=2)
    return system, user


def build_synthesizer_prompts(task: Task,
                              drafts: List[Draft],
                              fact_checks: List[FactCheckResult],
                              rubric: RubricEvaluation) -> Tuple[str, str]:
    system = f"""You are Eval3, the Synthesizer/Editor.

You are given:
- BRIEF, CONSTRAINTS, SUCCESS CRITERIA
- Drafts (id + content + uncertainties)
- Fact-check results per draft
- Rubric scores and ranking

Your job:
1. Choose one draft as the "base" to start from.
2. Design an edit plan that:
   - Keeps the strengths of the base draft.
   - Incorporates the best ideas from other drafts.
   - Fixes major fact-check issues and weaknesses identified by the rubric.
3. Identify remaining open questions or uncertainties that should be addressed.

Structure your plan as JSON:
{{
  "chosen_base_draft": "draft_id",
  "global_strategy": "overall narrative of what to do",
  "section_instructions": [
    {{
      "section_label": "e.g. Introduction, Design, Risks, etc.",
      "base_from_draft": "draft_id or null",
      "actions": [
        "e.g. Keep base; adopt risk analysis from WorkerC_v1; simplify explanation like WorkerD_v1"
      ],
      "notes": "concrete instructions for workers"
    }}
  ],
  "reuse_suggestions": [
    {{
      "from_draft": "draft_id",
      "what_to_reuse": "specific idea / phrase / structure to pull in"
    }}
  ],
  "open_questions": [
    "list of unresolved uncertainties workers should clarify or address"
  ]
}}"""
    simple_drafts = [{
        "draft_id": d.draft_id,
        "content": d.content,
        "uncertainties": [asdict(u) for u in d.uncertainties],
    } for d in drafts]
    simple_fact_checks = [{
        "draft_id": fc.draft_id,
        "issues": [asdict(i) for i in fc.issues],
        "overall_confidence": fc.overall_confidence,
        "summary": fc.summary,
    } for fc in fact_checks]
    rubric_dict = {
        "dimensions": rubric.dimensions,
        "per_draft": [{
            "draft_id": rd.draft_id,
            "dimension_scores": [asdict(ds) for ds in rd.dimension_scores],
            "overall_score": rd.overall_score,
            "summary": rd.summary,
        } for rd in rubric.per_draft],
        "ranking": rubric.ranking,
        "rationale_for_ranking": rubric.rationale_for_ranking,
    }
    brief_json = {
        "brief": task.normalized_brief,
        "constraints": task.constraints,
        "success_criteria": task.success_criteria,
    }
    payload = {
        "drafts": simple_drafts,
        "fact_checks": simple_fact_checks,
        "rubric_evaluation": rubric_dict,
        "brief": brief_json,
    }
    user = "Here is the context:\n\n" + json.dumps(payload, indent=2)
    return system, user


def build_revision_worker_prompts(worker: WorkerConfig,
                                  task: Task,
                                  own_draft: Draft,
                                  edit_plan: EditPlan) -> Tuple[str, str]:
    system = f"""You are {worker.id} revising your own earlier draft.

You are given:
- Your previous draft (DRAFT_V1) and its uncertainties.
- A global edit plan created by an editor.
- The original BRIEF, CONSTRAINTS, and SUCCESS CRITERIA.

Your tasks:
1. Produce a revised draft (DRAFT_V2) that follows the edit plan.
2. Keep what is good in your previous draft if it still fits the plan.
3. Incorporate ideas from other drafts only as described in the plan (you do NOT see their actual texts; only the plan's description).
4. Update your list of uncertainties: which are resolved, which remain, and any new ones.

Output strict JSON:
{{
  "revised_draft": "full revised draft",
  "change_summary": [
    "short bullet points describing main changes"
  ],
  "updated_uncertainties": [
    {{
      "id": "short_id",
      "description": "updated description",
      "type": "assumption | missing_info | ambiguity | risk | other",
      "impact": "low | medium | high"
    }}
  ]
}}"""
    brief_json = {
        "brief": task.normalized_brief,
        "constraints": task.constraints,
        "success_criteria": task.success_criteria,
    }
    own_draft_json = {
        "draft_id": own_draft.draft_id,
        "content": own_draft.content,
        "uncertainties": [asdict(u) for u in own_draft.uncertainties],
    }
    edit_plan_json = {
        "chosen_base_draft": edit_plan.chosen_base_draft,
        "global_strategy": edit_plan.global_strategy,
        "section_instructions": [asdict(si) for si in edit_plan.section_instructions],
        "reuse_suggestions": [asdict(rs) for rs in edit_plan.reuse_suggestions],
        "open_questions": edit_plan.open_questions,
    }
    payload = {
        "brief": brief_json,
        "edit_plan": edit_plan_json,
        "your_previous_draft": own_draft_json,
    }
    user = "Context:\n\n" + json.dumps(payload, indent=2) + "\n\nRevise your draft now."
    return system, user


def build_final_judge_prompts(task: Task,
                              revisions: List[Revision]) -> Tuple[str, str]:
    system = f"""You are FinalJudge.

You are given:
- BRIEF, CONSTRAINTS, SUCCESS CRITERIA.
- Revised drafts only (v2).

Your tasks:
1. Score each revised draft on the rubric:
   - correctness, coverage, clarity, practicality, risk_handling (0–10 each).
2. Provide an overall score and short summary per draft.
3. Produce a clear ranking from best to worst.
4. Select a single winner and justify your choice.

Output strict JSON:
{{
  "dimensions": ["correctness", "coverage", "clarity", "practicality", "risk_handling"],
  "per_draft": [
    {{
      "draft_id": "...",
      "dimension_scores": [
        {{"name": "correctness", "score": 0, "justification": "..."}},
        ...
      ],
      "overall_score": 0,
      "summary": "..."
    }}
  ],
  "ranking": ["best_draft_id", "..."],
  "winner_draft_id": "best_draft_id",
  "reasoning": "why this winner"
}}"""
    brief_json = {
        "brief": task.normalized_brief,
        "constraints": task.constraints,
        "success_criteria": task.success_criteria,
    }
    simple_revisions = [{
        "draft_id": r.draft_id,
        "content": r.content,
        "worker_id": r.worker_id,
        "change_summary": r.change_summary,
        "updated_uncertainties": [asdict(u) for u in r.updated_uncertainties],
    } for r in revisions]
    payload = {"brief": brief_json, "drafts": simple_revisions}
    user = "Here is the context:\n\n" + json.dumps(payload, indent=2)
    return system, user
