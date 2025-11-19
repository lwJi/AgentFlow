#!/usr/bin/env python3
import json
import os
import time
import uuid
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional

# If you use the "openai" Python package >= 1.0:
#   pip install openai
#   export OPENAI_API_KEY=...
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # You can plug in your own client here.


# =========================
# Data model
# =========================

WorkerId = str
EvaluatorId = str
DraftId = str
RunId = str


@dataclass
class WorkerConfig:
    id: WorkerId
    display_name: str
    persona: str


@dataclass
class EvaluatorConfig:
    id: EvaluatorId
    role: str


@dataclass
class RunConfig:
    run_id: RunId
    model: str
    temperature: float
    seed: Optional[int]
    max_tokens_per_call: Optional[int]
    workers: List[WorkerConfig]
    evaluators: List[EvaluatorConfig]


@dataclass
class Task:
    user_prompt: str
    normalized_brief: str
    constraints: List[str]
    success_criteria: List[str]


@dataclass
class Uncertainty:
    id: str
    description: str
    type: str
    impact: str


@dataclass
class Draft:
    draft_id: DraftId
    worker_id: WorkerId
    version: int  # 1 = initial, 2 = revised
    content: str
    uncertainties: List[Uncertainty]


@dataclass
class FactCheckIssue:
    severity: str
    location_hint: str
    description: str
    type: str


@dataclass
class FactCheckResult:
    evaluator_id: EvaluatorId
    draft_id: DraftId
    issues: List[FactCheckIssue]
    overall_confidence: float
    summary: str


@dataclass
class RubricDimensionScore:
    name: str
    score: float
    justification: str


@dataclass
class RubricScoresForDraft:
    draft_id: DraftId
    dimension_scores: List[RubricDimensionScore]
    overall_score: float
    summary: str


@dataclass
class RubricEvaluation:
    evaluator_id: EvaluatorId
    dimensions: List[str]
    per_draft: List[RubricScoresForDraft]
    ranking: List[DraftId]
    rationale_for_ranking: str


@dataclass
class SectionInstruction:
    section_label: str
    base_from_draft: Optional[DraftId]
    actions: List[str]
    notes: str


@dataclass
class ReuseSuggestion:
    from_draft: DraftId
    what_to_reuse: str


@dataclass
class EditPlan:
    evaluator_id: EvaluatorId
    chosen_base_draft: DraftId
    global_strategy: str
    section_instructions: List[SectionInstruction]
    reuse_suggestions: List[ReuseSuggestion]
    open_questions: List[str]


@dataclass
class Revision:
    draft_id: DraftId
    from_draft_id: DraftId
    worker_id: WorkerId
    version: int  # 2
    content: str
    change_summary: List[str]
    updated_uncertainties: List[Uncertainty]


@dataclass
class FinalDecision:
    evaluator_id: EvaluatorId
    winner_draft_id: DraftId
    ranking: List[DraftId]
    reasoning: str
    rubric_evaluation: RubricEvaluation


@dataclass
class RunLog:
    config: RunConfig
    task: Task
    initial_drafts: List[Draft]
    fact_checks: List[FactCheckResult]
    rubric_evaluation_initial: RubricEvaluation
    edit_plan: EditPlan
    revisions: List[Revision]
    final_decision: FinalDecision


# =========================
# LLM client wrapper
# =========================

class LLMClient:
    """
    Thin wrapper around an OpenAI-compatible ChatCompletion API.

    Replace this with your codex-cli / MCP adapter if you prefer.
    """

    def __init__(self, model: str, temperature: float = 0.7, seed: Optional[int] = None,
                 max_tokens: Optional[int] = None):
        if OpenAI is None:
            raise RuntimeError(
                "openai package not available. Install `openai` or plug your own client into LLMClient."
            )
        self.client = OpenAI()
        self.model = model
        self.temperature = temperature
        self.seed = seed
        self.max_tokens = max_tokens

    def chat_json(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """
        Ask the model for a JSON object response. Will raise on JSON parse errors.
        """
        params: Dict[str, Any] = dict(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature,
            response_format={"type": "json_object"},
        )
        if self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens
        if self.seed is not None:
            params["seed"] = self.seed

        completion = self.client.chat.completions.create(**params)
        content = completion.choices[0].message.content
        return json.loads(content)


# =========================
# Prompt builders
# =========================

def build_task_normalizer_prompts(user_prompt: str) -> (str, str):
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


def worker_persona(worker_id: WorkerId) -> str:
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


def build_worker_prompts(worker: WorkerConfig, task: Task) -> (str, str):
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


def build_factchecker_prompts(task: Task, drafts: List[Draft]) -> (str, str):
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
                         phase: str = "initial") -> (str, str):
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
                              rubric: RubricEvaluation) -> (str, str):
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
                                  edit_plan: EditPlan) -> (str, str):
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
                              revisions: List[Revision]) -> (str, str):
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


# =========================
# Orchestration helpers
# =========================

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
    # Basic sanity
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


def call_workers(client: LLMClient,
                 workers: List[WorkerConfig],
                 task: Task) -> List[Draft]:
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
    per_draft = []
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


# =========================
# Main workflow
# =========================

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

    # 1. Normalize task
    task = normalize_task(client, user_prompt)

    # 2. Initial drafts
    initial_drafts = call_workers(client, workers, task)

    # 3. Evaluation round 1
    eval1 = next(e for e in evaluators if e.id == "Eval1")
    eval2 = next(e for e in evaluators if e.id == "Eval2")
    eval3 = next(e for e in evaluators if e.id == "Eval3")
    final_eval = next(e for e in evaluators if e.id == "FinalJudge")

    fact_checks = call_factchecker(client, eval1, task, initial_drafts)
    rubric_initial = call_rubric_scorer(client, eval2, task, initial_drafts, fact_checks, phase="initial")
    edit_plan = call_synthesizer(client, eval3, task, initial_drafts, fact_checks, rubric_initial)

    # 4. Revisions
    revisions = call_revision_workers(client, workers, task, initial_drafts, edit_plan)

    # 5. Final selection
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
        if isinstance(o, (RunLog, RunConfig, Task, Draft, FactCheckResult, RubricEvaluation,
                          EditPlan, Revision, FinalDecision,
                          Uncertainty, FactCheckIssue, RubricDimensionScore,
                          RubricScoresForDraft, SectionInstruction, ReuseSuggestion)):
            return asdict(o)
        raise TypeError(f"Type not serializable: {type(o)}")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(runlog, f, indent=2, default=default)
    return path


if __name__ == "__main__":
    import argparse

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

