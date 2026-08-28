import json

from app.evaluation.runner import run


print("===== RUNNING PHASE 3 EVALUATION =====")

report = run()

print("\n===== EVALUATION SUMMARY =====")
print(f"Dataset as-of       : {report['dataset_as_of']}")
print(f"Total cases         : {report['total_cases']}")
print(f"Passed cases        : {report['passed_cases']}")
print(
    f"Overall quality    : "
    f"{report['overall_quality_score']:.1%}"
)

print("\n===== ADVERSARIAL CASES =====")

for case in report["adversarial_cases"]:
    print(
        f"- {case['case_id']} | "
        f"passed={case['passed']} | "
        f"quality={case['quality_score']}"
    )

print("\n===== FULL REPORT =====")
print(json.dumps(report, indent=2))