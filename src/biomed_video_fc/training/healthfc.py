"""Fine-tune a local biomedical sequence classifier on the official HealthFC CSV."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from biomed_video_fc.domain.models import Verdict
from biomed_video_fc.evaluation import classification_metrics, normalize_healthfc_label
from biomed_video_fc.exceptions import ConfigurationError, ModelUnavailableError

logger = logging.getLogger(__name__)

VERDICT_TO_ID = {Verdict.TRUE: 0, Verdict.NEI: 1, Verdict.FALSE: 2}
ID_TO_VERDICT = {identifier: verdict for verdict, identifier in VERDICT_TO_ID.items()}


@dataclass(frozen=True)
class HealthFCExample:
    row_index: int
    claim: str
    evidence: str
    label: int
    source_url: str = ""


def load_healthfc(path: Path) -> list[HealthFCExample]:
    if not path.is_file():
        raise ConfigurationError(f"HealthFC CSV does not exist: {path}")
    examples: list[HealthFCExample] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"en_claim", "en_top_sentences", "label"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ConfigurationError(
                f"HealthFC requires columns {sorted(required)}; found {reader.fieldnames}"
            )
        for row_index, row in enumerate(reader):
            claim = str(row.get("en_claim") or "").strip()
            evidence = str(row.get("en_top_sentences") or "").strip()
            if not claim or not evidence:
                continue
            verdict = normalize_healthfc_label(row.get("label"))
            examples.append(
                HealthFCExample(
                    row_index=row_index,
                    claim=claim,
                    evidence=evidence,
                    label=VERDICT_TO_ID[verdict],
                    source_url=str(row.get("url") or "").strip(),
                )
            )
    if not examples:
        raise ConfigurationError("No trainable HealthFC examples were found")
    return examples


def _local_dependencies() -> tuple[Any, ...]:
    try:
        import torch
        from sklearn.model_selection import StratifiedGroupKFold
        from torch.utils.data import DataLoader, Dataset
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            get_linear_schedule_with_warmup,
        )
    except ImportError as exc:
        raise ModelUnavailableError(
            "Local training requires PyTorch, Transformers, and scikit-learn. "
            "Install: pip install -e '.[local-ml]'"
        ) from exc
    return (
        torch,
        StratifiedGroupKFold,
        DataLoader,
        Dataset,
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )


def train(
    dataset_path: Path,
    output_dir: Path,
    base_model: str,
    epochs: int = 5,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    max_length: int = 512,
    seed: int = 42,
    patience: int = 2,
) -> dict[str, Any]:
    (
        torch,
        stratified_group_kfold,
        data_loader_class,
        dataset_class,
        model_class,
        tokenizer_class,
        scheduler_factory,
    ) = _local_dependencies()
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    examples = load_healthfc(dataset_path)
    labels = [item.label for item in examples]
    groups = [item.source_url or f"row:{item.row_index}" for item in examples]
    outer = stratified_group_kfold(n_splits=7, shuffle=True, random_state=seed)
    train_validation_indices, test_indices = next(outer.split(examples, labels, groups))
    train_and_validation = [examples[index] for index in train_validation_indices]
    test_examples = [examples[index] for index in test_indices]
    inner_labels = [item.label for item in train_and_validation]
    inner_groups = [item.source_url or f"row:{item.row_index}" for item in train_and_validation]
    inner = stratified_group_kfold(n_splits=6, shuffle=True, random_state=seed)
    train_indices, validation_indices = next(
        inner.split(train_and_validation, inner_labels, inner_groups)
    )
    train_examples = [train_and_validation[index] for index in train_indices]
    validation_examples = [train_and_validation[index] for index in validation_indices]

    tokenizer = tokenizer_class.from_pretrained(base_model)

    class EncodedDataset(dataset_class):
        def __init__(self, items: list[HealthFCExample]) -> None:
            self.items = items

        def __len__(self) -> int:
            return len(self.items)

        def __getitem__(self, index: int) -> dict[str, Any]:
            item = self.items[index]
            encoded = tokenizer(
                item.claim,
                item.evidence,
                truncation="only_second",
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
            batch = {name: value.squeeze(0) for name, value in encoded.items()}
            batch["labels"] = torch.tensor(item.label, dtype=torch.long)
            return batch

    train_loader = data_loader_class(
        EncodedDataset(train_examples), batch_size=batch_size, shuffle=True
    )
    validation_loader = data_loader_class(
        EncodedDataset(validation_examples), batch_size=batch_size, shuffle=False
    )
    test_loader = data_loader_class(
        EncodedDataset(test_examples), batch_size=batch_size, shuffle=False
    )

    id2label = {0: Verdict.TRUE.value, 1: "NEI", 2: Verdict.FALSE.value}
    model = model_class.from_pretrained(
        base_model,
        num_labels=3,
        id2label=id2label,
        label2id={label: identifier for identifier, label in id2label.items()},
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    label_counts = Counter(item.label for item in train_examples)
    class_weights = torch.tensor(
        [len(train_examples) / (3 * label_counts[label]) for label in range(3)],
        dtype=torch.float,
        device=device,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    total_steps = max(1, epochs * len(train_loader))
    scheduler = scheduler_factory(
        optimizer, num_warmup_steps=max(1, int(total_steps * 0.1)), num_training_steps=total_steps
    )

    def evaluate(loader: Any) -> dict[str, Any]:
        model.eval()
        gold_ids: list[int] = []
        predicted_ids: list[int] = []
        losses: list[float] = []
        with torch.no_grad():
            for batch in loader:
                labels_tensor = batch.pop("labels").to(device)
                inputs = {name: value.to(device) for name, value in batch.items()}
                logits = model(**inputs).logits
                loss = torch.nn.functional.cross_entropy(
                    logits, labels_tensor, weight=class_weights
                )
                losses.append(float(loss.item()))
                gold_ids.extend(labels_tensor.cpu().tolist())
                predicted_ids.extend(torch.argmax(logits, dim=-1).cpu().tolist())
        metrics = classification_metrics(
            [ID_TO_VERDICT[item] for item in gold_ids],
            [ID_TO_VERDICT[item] for item in predicted_ids],
        )
        metrics["loss"] = sum(losses) / len(losses) if losses else 0.0
        return metrics

    output_dir.mkdir(parents=True, exist_ok=True)
    best_macro_f1 = -1.0
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses: list[float] = []
        for batch in train_loader:
            labels_tensor = batch.pop("labels").to(device)
            inputs = {name: value.to(device) for name, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            logits = model(**inputs).logits
            loss = torch.nn.functional.cross_entropy(logits, labels_tensor, weight=class_weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            train_losses.append(float(loss.item()))
        validation_metrics = evaluate(validation_loader)
        epoch_record = {
            "epoch": epoch,
            "train_loss": sum(train_losses) / len(train_losses),
            "validation": validation_metrics,
        }
        history.append(epoch_record)
        logger.info(
            "Epoch %d: train loss %.4f, validation macro-F1 %.4f",
            epoch,
            epoch_record["train_loss"],
            validation_metrics["macro_f1"],
        )
        if validation_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = validation_metrics["macro_f1"]
            epochs_without_improvement = 0
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                logger.info("Early stopping after epoch %d", epoch)
                break

    best_model = model_class.from_pretrained(output_dir).to(device)
    model = best_model
    test_metrics = evaluate(test_loader)
    report = {
        "dataset": dataset_path.name,
        "base_model": base_model,
        "seed": seed,
        "split_sizes": {
            "train": len(train_examples),
            "validation": len(validation_examples),
            "test": len(test_examples),
        },
        "label_distribution": dict(Counter(labels)),
        "class_weights": class_weights.detach().cpu().tolist(),
        "history": history,
        "test": test_metrics,
        "split_row_indices": {
            "train": [item.row_index for item in train_examples],
            "validation": [item.row_index for item in validation_examples],
            "test": [item.row_index for item in test_examples],
        },
        "split_source_groups": {
            "train": sorted({item.source_url for item in train_examples if item.source_url}),
            "validation": sorted(
                {item.source_url for item in validation_examples if item.source_url}
            ),
            "test": sorted({item.source_url for item in test_examples if item.source_url}),
        },
        "split_strategy": "source-grouped stratified folds (approximately 71/14/14)",
    }
    (output_dir / "training_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the local classifier on HealthFC")
    parser.add_argument("dataset", type=Path, help="Path to HealthFC.csv")
    parser.add_argument("--output", type=Path, default=Path("artifacts/healthfc_classifier"))
    parser.add_argument(
        "--base-model",
        default="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    report = train(
        args.dataset,
        args.output,
        args.base_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        seed=args.seed,
    )
    print(json.dumps(report["test"], indent=2))


if __name__ == "__main__":
    main()
