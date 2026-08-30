import json
from pathlib import Path

from app.data_repository import DataRepository
from app.health_agent import HealthAgent
from app.models import TicketInput
from app.triage_agent import TriageAgent

from app.evaluation.cases import HEALTH_CASES, TRIAGE_CASES
from app.evaluation.scorer import score_health, score_triage


def run(output_path: str = "eval_report.json") -> dict:
    """
    Run the complete Phase 3 evaluation.

    The evaluation covers:
    - Triage agent cases
    - Account health agent cases

    Each agent produces its normal production result.
    The scorer then compares that result against the expected
    evaluation case.

    A JSON report is written to output_path.
    """

    if len(TRIAGE_CASES) < 5 or len(HEALTH_CASES) < 5:
        raise RuntimeError(
            "Evaluation requires at least five cases per task: "
            f"triage={len(TRIAGE_CASES)}, health={len(HEALTH_CASES)}."
        )

    # ---------------------------------------------------------
    # STEP 1: Load the dataset once
    # ---------------------------------------------------------

    repo = DataRepository()

    # ---------------------------------------------------------
    # STEP 2: Create the agents
    # ---------------------------------------------------------

    triage = TriageAgent()

    health = HealthAgent(
        repo=repo,
    )

    # ---------------------------------------------------------
    # STEP 3: Evaluate the exact fixture payload
    # ---------------------------------------------------------

    # A case owns both its expected result and its input text. Fixture and
    # dataset text can legitimately diverge as adversarial coverage evolves.

    # ---------------------------------------------------------
    # STEP 4: Evaluate Triage cases
    # ---------------------------------------------------------

    results = []

    for case in TRIAGE_CASES:
        try:
            result = triage.triage(
                TicketInput(
                    subject=case.subject,
                    body=case.body,
                )
            )
            scored_result = score_triage(result, case)
        except Exception as exc:
            scored_result = {
                "case_id": case.ticket_id,
                "case_name": case.name,
                "adversarial": getattr(case, "adversarial", False),
                "passed": False,
                "quality_score": 0.0,
                "error": f"{type(exc).__name__}: {exc}",
            }

        results.append(scored_result)

    # ---------------------------------------------------------
    # STEP 5: Evaluate Account Health cases
    # ---------------------------------------------------------

    for case in HEALTH_CASES:
        try:
            result = health.summarise(case.account_id)
            scored_result = score_health(result, case)
        except Exception as exc:
            scored_result = {
                "case_id": case.account_id,
                "case_name": case.name,
                "adversarial": getattr(case, "adversarial", False),
                "passed": False,
                "quality_score": 0.0,
                "error": f"{type(exc).__name__}: {exc}",
            }

        results.append(scored_result)

    # ---------------------------------------------------------
    # STEP 6: Calculate overall evaluation metrics
    # ---------------------------------------------------------

    total_cases = len(results)

    passed_cases = sum(
        result["passed"]
        for result in results
    )

    overall_quality_score = (
        sum(
            result["quality_score"]
            for result in results
        )
        / total_cases
        if total_cases
        else 0.0
    )

    # ---------------------------------------------------------
    # STEP 7: Build final evaluation report
    # ---------------------------------------------------------

    report = {
        "dataset_as_of": repo.dataset_as_of.isoformat(),

        "total_cases": total_cases,

        "passed_cases": passed_cases,

        "overall_quality_score": round(
            overall_quality_score,
            3,
        ),

        "adversarial_cases": [
            result
            for result in results
            if result["adversarial"]
        ],

        "results": results,
    }

    # ---------------------------------------------------------
    # STEP 8: Persist report to disk
    # ---------------------------------------------------------

    Path(output_path).write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    return report


if __name__ == "__main__":
    report = run()

    print(
        json.dumps(
            report,
            indent=2,
        )
    )
