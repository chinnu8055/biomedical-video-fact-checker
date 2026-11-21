"""Provider-independent classification metrics and HealthFC label mapping."""

from __future__ import annotations

from typing import Any

from biomed_video_fc.domain.models import Verdict

LABEL_MAP = {
    "0": Verdict.TRUE,
    "supported": Verdict.TRUE,
    "true": Verdict.TRUE,
    "1": Verdict.NEI,
    "nei": Verdict.NEI,
    "not enough information": Verdict.NEI,
    "2": Verdict.FALSE,
    "refuted": Verdict.FALSE,
    "false": Verdict.FALSE,
}


def normalize_healthfc_label(value: object) -> Verdict:
    key = str(value).strip().casefold()
    if key not in LABEL_MAP:
        raise ValueError(f"Unknown HealthFC label: {value!r}")
    return LABEL_MAP[key]


def classification_metrics(gold: list[Verdict], predicted: list[Verdict]) -> dict[str, Any]:
    if len(gold) != len(predicted):
        raise ValueError("gold and predicted lengths differ")
    labels = [Verdict.TRUE, Verdict.FALSE, Verdict.NEI]
    matrix = {actual.value: {guess.value: 0 for guess in labels} for actual in labels}
    for actual, guess in zip(gold, predicted, strict=True):
        matrix[actual.value][guess.value] += 1

    per_class: dict[str, dict[str, float]] = {}
    for label in labels:
        tp = matrix[label.value][label.value]
        fp = sum(matrix[other.value][label.value] for other in labels if other is not label)
        fn = sum(matrix[label.value][other.value] for other in labels if other is not label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label.value] = {"precision": precision, "recall": recall, "f1": f1}

    total = len(gold)
    return {
        "examples": total,
        "accuracy": sum(actual is guess for actual, guess in zip(gold, predicted, strict=True))
        / total
        if total
        else 0.0,
        "macro_f1": sum(values["f1"] for values in per_class.values()) / len(labels),
        "nei_prediction_rate": sum(item is Verdict.NEI for item in predicted) / total
        if total
        else 0.0,
        "per_class": per_class,
        "confusion_matrix": matrix,
    }
