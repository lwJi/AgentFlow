import json
from pathlib import Path

from src.models import (
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
    RunLog,
    SectionInstruction,
    Task,
    Uncertainty,
    WorkerConfig,
    ReuseSuggestion,
)
from src.workflow import save_runlog


def _dummy_runlog(run_id: str = "test_run") -> RunLog:
    worker = WorkerConfig(id="WorkerA", display_name="Architect", persona="Structure")
    evaluator = EvaluatorConfig(id="Eval1", role="FactChecker")
    config = RunConfig(
        run_id=run_id,
        model="dummy",
        temperature=0.0,
        seed=None,
        max_tokens_per_call=None,
        workers=[worker],
        evaluators=[evaluator],
    )
    task = Task(
        user_prompt="Prompt",
        normalized_brief="Brief",
        constraints=["c1"],
        success_criteria=["s1"],
    )
    uncertainty = Uncertainty(id="u1", description="desc", type="assumption", impact="low")
    draft = Draft(
        draft_id="WorkerA_v1",
        worker_id=worker.id,
        version=1,
        content="draft",
        uncertainties=[uncertainty],
    )
    issue = FactCheckIssue(severity="minor", location_hint="line1", description="note", type="other")
    fact_check = FactCheckResult(
        evaluator_id=evaluator.id,
        draft_id=draft.draft_id,
        issues=[issue],
        overall_confidence=5.0,
        summary="ok",
    )
    score = RubricDimensionScore(name="correctness", score=8, justification="ok")
    per_draft = RubricScoresForDraft(
        draft_id=draft.draft_id,
        dimension_scores=[score],
        overall_score=80,
        summary="solid",
    )
    rubric = RubricEvaluation(
        evaluator_id="Eval2",
        dimensions=["correctness"],
        per_draft=[per_draft],
        ranking=[draft.draft_id],
        rationale_for_ranking="best",
    )
    section = SectionInstruction(
        section_label="Intro",
        base_from_draft=draft.draft_id,
        actions=["Keep"],
        notes="None",
    )
    reuse = ReuseSuggestion(from_draft=draft.draft_id, what_to_reuse="idea")
    edit_plan = EditPlan(
        evaluator_id="Eval3",
        chosen_base_draft=draft.draft_id,
        global_strategy="Improve",
        section_instructions=[section],
        reuse_suggestions=[reuse],
        open_questions=["q1"],
    )
    revision = Revision(
        draft_id="WorkerA_v2",
        from_draft_id=draft.draft_id,
        worker_id=worker.id,
        version=2,
        content="rev",
         change_summary=["updated intro"],
        updated_uncertainties=[uncertainty],
    )
    final_decision = FinalDecision(
        evaluator_id="FinalJudge",
        winner_draft_id=revision.draft_id,
        ranking=[revision.draft_id],
        reasoning="best",
        rubric_evaluation=rubric,
    )
    return RunLog(
        config=config,
        task=task,
        initial_drafts=[draft],
        fact_checks=[fact_check],
        rubric_evaluation_initial=rubric,
        edit_plan=edit_plan,
        revisions=[revision],
        final_decision=final_decision,
    )


def test_save_runlog_round_trip(tmp_path: Path) -> None:
    runlog = _dummy_runlog()
    out_path = save_runlog(runlog, out_dir=tmp_path.as_posix())
    saved = Path(out_path)
    assert saved.exists()

    data = json.loads(saved.read_text())
    assert data["config"]["run_id"] == "test_run"
    assert data["task"]["normalized_brief"] == "Brief"
    assert data["initial_drafts"][0]["draft_id"] == "WorkerA_v1"
    assert data["final_decision"]["winner_draft_id"] == "WorkerA_v2"
