"""Data models for the AgentFlow orchestrator."""

from dataclasses import dataclass
from typing import List, Optional

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
    version: int
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
    version: int
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
