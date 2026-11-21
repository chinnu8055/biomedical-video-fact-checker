"""Evaluate entity/negation/qualifier behavior on fixed claim-evidence pairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from biomed_video_fc.domain.models import Claim, Evidence, Verdict
from biomed_video_fc.evaluation import classification_metrics
from biomed_video_fc.reasoning.local_classifier import HealthFCClassifierReasoner
from biomed_video_fc.retrieval.entity_validation import (
    audit_entity_consistency,
    entity_labels,
)


def _load_cases(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Challenge set does not exist: {path}")
    cases: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            required = {"id", "claim", "title", "abstract", "expected_verdict", "focus"}
            if not required.issubset(row):
                missing = sorted(required - set(row))
                raise ValueError(f"Challenge case {line_number} is missing: {missing}")
            row["expected_verdict"] = Verdict(str(row["expected_verdict"]))
            cases.append(row)
    return cases


def evaluate_challenge(
    challenge_path: Path,
    model_dir: Path,
    output_path: Path,
) -> dict[str, Any]:
    cases = _load_cases(challenge_path)
    reasoner = HealthFCClassifierReasoner(model_dir)
    expected: list[Verdict] = []
    predicted: list[Verdict] = []
    rows: list[dict[str, Any]] = []

    for case in cases:
        claim_text = str(case["claim"])
        document = f"{case['title']}. {case['abstract']}"
        entity_audit = audit_entity_consistency(claim_text, document)
        if not entity_audit.passes:
            verdict = Verdict.NEI
            confidence = 0.0
            decision = "entity_filter"
            assessments: list[dict[str, Any]] = []
        else:
            evidence = Evidence(
                pmid=f"challenge-{case['id']}",
                title=str(case["title"]),
                abstract=str(case["abstract"]),
                url=f"https://example.invalid/{case['id']}",
                relevance_score=1.0,
                term_coverage=1.0,
                entity_match=True,
                matched_entities=entity_labels(entity_audit.matched),
                additional_entities=entity_labels(entity_audit.additional),
            )
            verification = reasoner.verify(Claim(text=claim_text), [evidence])
            verdict = verification.verdict
            confidence = verification.confidence
            decision = verification.decision_source
            assessments = [
                item.model_dump(mode="json") for item in verification.evidence_assessments
            ]

        gold = case["expected_verdict"]
        expected.append(gold)
        predicted.append(verdict)
        rows.append(
            {
                "id": case["id"],
                "focus": case["focus"],
                "claim": claim_text,
                "expected": gold.value,
                "predicted": verdict.value,
                "correct": verdict == gold,
                "confidence": confidence,
                "decision": decision,
                "required_entities": entity_labels(entity_audit.required),
                "matched_entities": entity_labels(entity_audit.matched),
                "missing_entities": entity_labels(entity_audit.missing),
                "assessments": assessments,
            }
        )

    report = {
        "challenge_set": str(challenge_path),
        "model_dir": str(model_dir),
        "case_count": len(cases),
        "metrics": classification_metrics(expected, predicted),
        "predictions": rows,
        "warning": "This focused challenge set is a diagnostic, not the final test dataset.",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the fixed precision challenge set")
    parser.add_argument(
        "--challenge",
        type=Path,
        default=Path("data/challenge_sets/vitamin_c_precision.jsonl"),
    )
    parser.add_argument("--model-dir", type=Path, default=Path("artifacts/multisource_classifier"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/evaluation/vitamin_c_precision_report.json"),
    )
    args = parser.parse_args()
    report = evaluate_challenge(args.challenge, args.model_dir, args.output)
    print(json.dumps(report["metrics"], indent=2))


if __name__ == "__main__":
    main()
