"""Prepare HealthFC, BioASQ-7b, and SciFact without erasing source provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from biomed_video_fc.training.datasets import load_available_datasets, summarize


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare(data_dir: Path, output_dir: Path) -> dict[str, Any]:
    examples = load_available_datasets(data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    by_dataset: dict[str, list] = {}
    for item in examples:
        by_dataset.setdefault(item.dataset, []).append(item)

    files: dict[str, str] = {}
    summaries: dict[str, Any] = {}
    for dataset, items in sorted(by_dataset.items()):
        target = output_dir / f"{dataset}.jsonl"
        with target.open("w", encoding="utf-8", newline="\n") as handle:
            for item in items:
                row = asdict(item)
                row["label"] = item.label.value
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        files[dataset] = str(target)
        summaries[dataset] = asdict(summarize(items, dataset))

    source_files = [
        data_dir / "HealthFC.csv",
        data_dir / "BioASQ-training7b.zip",
        data_dir / "Task7BGoldenEnriched.zip",
        data_dir / "data.tar.gz",
    ]
    manifest = {
        "format_version": 1,
        "total_examples": len(examples),
        "summaries": summaries,
        "prepared_files": files,
        "source_sha256": {path.name: _sha256(path) for path in source_files},
        "notes": [
            "HealthFC explanations are excluded to avoid verdict leakage.",
            "Only BioASQ yes/no records are included.",
            "Duplicate BioASQ questions are merged; conflicting duplicate labels are excluded.",
            "SciFact public test claims are excluded because their labels are not public.",
            "Exact SciFact claims repeated in train and development are removed from train.",
            "Dataset source and original split are preserved; files are not blindly concatenated.",
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize biomedical verification datasets")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/prepared_datasets"))
    args = parser.parse_args()
    report = prepare(args.data_dir, args.output)
    print(json.dumps(report["summaries"], indent=2))


if __name__ == "__main__":
    main()
