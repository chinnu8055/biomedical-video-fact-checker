"""Fine-tune BiomedBERT on prepared HealthFC, BioASQ-7b, and SciFact data."""

from __future__ import annotations

import argparse
import contextlib
import importlib.metadata
import json
import logging
import math
import platform
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from biomed_video_fc.domain.models import Verdict
from biomed_video_fc.evaluation import classification_metrics
from biomed_video_fc.exceptions import ConfigurationError, ModelUnavailableError
from biomed_video_fc.training.augmentation import build_hard_negatives
from biomed_video_fc.training.datasets import VerificationExample

logger = logging.getLogger(__name__)

VERDICT_TO_ID = {Verdict.TRUE: 0, Verdict.NEI: 1, Verdict.FALSE: 2}
ID_TO_VERDICT = {identifier: verdict for verdict, identifier in VERDICT_TO_ID.items()}


@dataclass(frozen=True)
class ExperimentSplits:
    train: list[VerificationExample]
    validation: list[VerificationExample]
    test: list[VerificationExample]


def _read_jsonl(path: Path) -> list[VerificationExample]:
    if not path.is_file():
        raise ConfigurationError(
            f"Prepared dataset does not exist: {path}. Run biofc-prepare-datasets first."
        )
    examples: list[VerificationExample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                examples.append(
                    VerificationExample(
                        example_id=str(row["example_id"]),
                        claim=str(row["claim"]),
                        evidence=str(row["evidence"]),
                        label=Verdict(str(row["label"])),
                        dataset=str(row["dataset"]),
                        split=str(row["split"]),
                        group_id=str(row["group_id"]),
                        document_ids=tuple(row.get("document_ids", [])),
                        rationale_sentence_indices=tuple(row.get("rationale_sentence_indices", [])),
                        metadata=dict(row.get("metadata", {})),
                    )
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ConfigurationError(
                    f"Invalid prepared example at {path}:{line_number}"
                ) from exc
    return examples


def _grouped_fold(
    examples: list[VerificationExample],
    splitter_class: Any,
    n_splits: int,
    seed: int,
) -> tuple[list[VerificationExample], list[VerificationExample]]:
    if not examples:
        return [], []
    labels = [VERDICT_TO_ID[item.label] for item in examples]
    groups = [f"{item.dataset}:{item.group_id}" for item in examples]
    splitter = splitter_class(n_splits=n_splits, shuffle=True, random_state=seed)
    retained_indices, held_out_indices = next(splitter.split(examples, labels, groups))
    return (
        [examples[index] for index in retained_indices],
        [examples[index] for index in held_out_indices],
    )


def build_splits(
    prepared_dir: Path,
    splitter_class: Any,
    seed: int,
) -> ExperimentSplits:
    """Build leakage-resistant splits while honoring official evaluation partitions."""

    healthfc = _read_jsonl(prepared_dir / "healthfc.jsonl")
    bioasq = _read_jsonl(prepared_dir / "bioasq7b.jsonl")
    scifact = _read_jsonl(prepared_dir / "scifact.jsonl")

    health_train_validation, health_test = _grouped_fold(
        healthfc, splitter_class, n_splits=7, seed=seed
    )
    health_train, health_validation = _grouped_fold(
        health_train_validation, splitter_class, n_splits=6, seed=seed
    )

    bio_train_official = [item for item in bioasq if item.split == "train"]
    bio_test = [item for item in bioasq if item.split == "test"]
    bio_train, bio_validation = _grouped_fold(
        bio_train_official, splitter_class, n_splits=7, seed=seed
    )

    sci_train_official = [item for item in scifact if item.split == "train"]
    sci_test = [item for item in scifact if item.split == "validation"]
    sci_train, sci_validation = _grouped_fold(
        sci_train_official, splitter_class, n_splits=7, seed=seed
    )

    result = ExperimentSplits(
        train=[*health_train, *bio_train, *sci_train],
        validation=[*health_validation, *bio_validation, *sci_validation],
        test=[*health_test, *bio_test, *sci_test],
    )
    train_groups = {f"{item.dataset}:{item.group_id}" for item in result.train}
    validation_groups = {f"{item.dataset}:{item.group_id}" for item in result.validation}
    test_groups = {f"{item.dataset}:{item.group_id}" for item in result.test}
    if (
        train_groups & validation_groups
        or train_groups & test_groups
        or validation_groups & test_groups
    ):
        raise ConfigurationError("A source group appears in more than one experiment split")

    normalized_claims = [
        {" ".join(item.claim.casefold().split()).strip(" .!?") for item in split}
        for split in (result.train, result.validation, result.test)
    ]
    if (
        normalized_claims[0] & normalized_claims[1]
        or normalized_claims[0] & normalized_claims[2]
        or normalized_claims[1] & normalized_claims[2]
    ):
        raise ConfigurationError("An exact claim appears in more than one experiment split")
    return result


def _dependencies() -> tuple[Any, ...]:
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
            "Multisource training requires PyTorch, Transformers, and scikit-learn. "
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


def _metrics_by_dataset(items: list[VerificationExample], predictions: list[int]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for dataset in sorted({item.dataset for item in items}):
        positions = [index for index, item in enumerate(items) if item.dataset == dataset]
        output[dataset] = classification_metrics(
            [items[index].label for index in positions],
            [ID_TO_VERDICT[predictions[index]] for index in positions],
        )
    return output


def _distribution(items: list[VerificationExample]) -> dict[str, Any]:
    by_dataset = Counter(item.dataset for item in items)
    by_label = Counter(item.label.value for item in items)
    dataset_labels = {
        dataset: dict(Counter(item.label.value for item in items if item.dataset == dataset))
        for dataset in sorted(by_dataset)
    }
    return {
        "by_dataset": dict(by_dataset),
        "by_label": dict(by_label),
        "by_dataset_and_label": dataset_labels,
    }


def train(
    prepared_dir: Path,
    output_dir: Path,
    base_model: str,
    epochs: int = 5,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    max_length: int = 512,
    seed: int = 42,
    patience: int = 2,
    gradient_accumulation_steps: int = 1,
    fp16: bool = False,
    hard_negative_ratio: float = 0.25,
) -> dict[str, Any]:
    if gradient_accumulation_steps < 1:
        raise ConfigurationError("gradient_accumulation_steps must be at least 1")
    if hard_negative_ratio < 0 or hard_negative_ratio > 1:
        raise ConfigurationError("hard_negative_ratio must be between 0 and 1")
    (
        torch,
        splitter_class,
        data_loader_class,
        dataset_class,
        model_class,
        tokenizer_class,
        scheduler_factory,
    ) = _dependencies()
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    splits = build_splits(prepared_dir, splitter_class, seed)
    original_train_size = len(splits.train)
    hard_negatives = build_hard_negatives(
        splits.train,
        max_ratio=hard_negative_ratio,
        seed=seed,
    )
    splits = ExperimentSplits(
        train=[*splits.train, *hard_negatives],
        validation=splits.validation,
        test=splits.test,
    )
    manifest_path = prepared_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ConfigurationError(f"Prepared dataset manifest does not exist: {manifest_path}")
    try:
        dataset_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigurationError(f"Prepared dataset manifest is invalid: {manifest_path}") from exc
    tokenizer = tokenizer_class.from_pretrained(base_model)

    class EncodedDataset(dataset_class):
        def __init__(self, items: list[VerificationExample]) -> None:
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
            batch["labels"] = torch.tensor(VERDICT_TO_ID[item.label], dtype=torch.long)
            return batch

    loaders = {
        "train": data_loader_class(
            EncodedDataset(splits.train), batch_size=batch_size, shuffle=True
        ),
        "validation": data_loader_class(
            EncodedDataset(splits.validation), batch_size=batch_size, shuffle=False
        ),
        "test": data_loader_class(
            EncodedDataset(splits.test), batch_size=batch_size, shuffle=False
        ),
    }
    id2label = {0: Verdict.TRUE.value, 1: "NEI", 2: Verdict.FALSE.value}
    model = model_class.from_pretrained(
        base_model,
        num_labels=3,
        id2label=id2label,
        label2id={label: identifier for identifier, label in id2label.items()},
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    use_fp16 = fp16 and device.type == "cuda"
    if fp16 and not use_fp16:
        logger.warning("FP16 was requested but CUDA is unavailable; using FP32")
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)

    def autocast_context():
        if use_fp16:
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return contextlib.nullcontext()

    train_counts = Counter(VERDICT_TO_ID[item.label] for item in splits.train)
    class_weights = torch.tensor(
        [len(splits.train) / (3 * train_counts[label]) for label in range(3)],
        dtype=torch.float,
        device=device,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    optimizer_steps_per_epoch = math.ceil(len(loaders["train"]) / gradient_accumulation_steps)
    total_steps = max(1, epochs * optimizer_steps_per_epoch)
    scheduler = scheduler_factory(
        optimizer,
        num_warmup_steps=max(1, int(total_steps * 0.1)),
        num_training_steps=total_steps,
    )

    def evaluate(split_name: str) -> tuple[dict[str, Any], list[int]]:
        model.eval()
        items = getattr(splits, split_name)
        gold_ids: list[int] = []
        predicted_ids: list[int] = []
        losses: list[float] = []
        with torch.no_grad():
            for batch in loaders[split_name]:
                labels = batch.pop("labels").to(device)
                inputs = {name: value.to(device) for name, value in batch.items()}
                with autocast_context():
                    logits = model(**inputs).logits
                    loss = torch.nn.functional.cross_entropy(logits, labels, weight=class_weights)
                losses.append(float(loss.item()))
                gold_ids.extend(labels.cpu().tolist())
                predicted_ids.extend(torch.argmax(logits, dim=-1).cpu().tolist())
        metrics = classification_metrics(
            [ID_TO_VERDICT[item] for item in gold_ids],
            [ID_TO_VERDICT[item] for item in predicted_ids],
        )
        metrics["loss"] = sum(losses) / len(losses) if losses else 0.0
        metrics["by_dataset"] = _metrics_by_dataset(items, predicted_ids)
        return metrics, predicted_ids

    output_dir.mkdir(parents=True, exist_ok=True)
    best_macro_f1 = -1.0
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses: list[float] = []
        optimizer.zero_grad(set_to_none=True)
        for batch_index, batch in enumerate(loaders["train"], start=1):
            labels = batch.pop("labels").to(device)
            inputs = {name: value.to(device) for name, value in batch.items()}
            with autocast_context():
                logits = model(**inputs).logits
                full_loss = torch.nn.functional.cross_entropy(logits, labels, weight=class_weights)
                loss = full_loss / gradient_accumulation_steps
            scaler.scale(loss).backward()
            should_step = batch_index % gradient_accumulation_steps == 0 or batch_index == len(
                loaders["train"]
            )
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
            train_losses.append(float(full_loss.item()))
        validation_metrics, _ = evaluate("validation")
        record = {
            "epoch": epoch,
            "train_loss": sum(train_losses) / len(train_losses),
            "validation": validation_metrics,
        }
        history.append(record)
        logger.info(
            "Epoch %d: train loss %.4f, validation macro-F1 %.4f",
            epoch,
            record["train_loss"],
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
                break

    model = model_class.from_pretrained(output_dir).to(device)
    test_metrics, predictions = evaluate("test")
    prediction_path = output_dir / "test_predictions.jsonl"
    with prediction_path.open("w", encoding="utf-8", newline="\n") as handle:
        for item, predicted_id in zip(splits.test, predictions, strict=True):
            handle.write(
                json.dumps(
                    {
                        "example_id": item.example_id,
                        "dataset": item.dataset,
                        "group_id": item.group_id,
                        "gold": item.label.value,
                        "predicted": ID_TO_VERDICT[predicted_id].value,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    report = {
        "base_model": base_model,
        "seed": seed,
        "epochs_requested": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "max_length": max_length,
        "patience": patience,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "fp16_requested": fp16,
        "fp16_used": use_fp16,
        "hard_negative_ratio": hard_negative_ratio,
        "original_train_size": original_train_size,
        "hard_negative_count": len(hard_negatives),
        "device": str(device),
        "python": platform.python_version(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "scikit-learn")
        },
        "split_sizes": {
            "train": len(splits.train),
            "validation": len(splits.validation),
            "test": len(splits.test),
        },
        "split_distributions": {
            "train": _distribution(splits.train),
            "validation": _distribution(splits.validation),
            "test": _distribution(splits.test),
        },
        "dataset_manifest": dataset_manifest,
        "split_examples": {
            "train": [item.example_id for item in splits.train],
            "validation": [item.example_id for item in splits.validation],
            "test": [item.example_id for item in splits.test],
        },
        "history": history,
        "test": test_metrics,
        "test_predictions": str(prediction_path),
    }
    (output_dir / "training_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "dataset_manifest.json").write_text(
        json.dumps(dataset_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the multisource BiomedBERT verifier")
    parser.add_argument("--prepared-dir", type=Path, default=Path("artifacts/prepared_datasets"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/multisource_classifier"))
    parser.add_argument(
        "--base-model",
        default="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use CUDA mixed precision when a compatible GPU is available",
    )
    parser.add_argument(
        "--hard-negative-ratio",
        type=float,
        default=0.25,
        help="Maximum synthetic entity/negation/qualifier examples as a fraction of training data",
    )
    args = parser.parse_args()
    report = train(
        args.prepared_dir,
        args.output,
        args.base_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        seed=args.seed,
        patience=args.patience,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fp16=args.fp16,
        hard_negative_ratio=args.hard_negative_ratio,
    )
    print(json.dumps(report["test"], indent=2))


if __name__ == "__main__":
    main()
