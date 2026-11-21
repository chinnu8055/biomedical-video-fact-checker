"""Load supported biomedical verification datasets into one auditable schema.

The loaders read the original archives directly.  They never silently reinterpret
unlabelled test data as NEI, and every example keeps its source, split, and source
document/group identifier so experiments can avoid leakage.
"""

from __future__ import annotations

import csv
import hashlib
import json
import tarfile
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from biomed_video_fc.domain.models import Verdict
from biomed_video_fc.evaluation import normalize_healthfc_label
from biomed_video_fc.exceptions import ConfigurationError


@dataclass(frozen=True)
class VerificationExample:
    example_id: str
    claim: str
    evidence: str
    label: Verdict
    dataset: str
    split: str
    group_id: str
    document_ids: tuple[str, ...] = ()
    rationale_sentence_indices: tuple[int, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DatasetSummary:
    dataset: str
    split_counts: dict[str, int]
    label_counts: dict[str, int]


def summarize(examples: list[VerificationExample], dataset: str) -> DatasetSummary:
    return DatasetSummary(
        dataset=dataset,
        split_counts=dict(Counter(item.split for item in examples)),
        label_counts=dict(Counter(item.label.value for item in examples)),
    )


def load_healthfc(path: Path, split: str = "unsplit") -> list[VerificationExample]:
    """Read HealthFC without using its explanation field, which leaks the verdict."""

    if not path.is_file():
        raise ConfigurationError(f"HealthFC CSV does not exist: {path}")
    examples: list[VerificationExample] = []
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
            source_url = str(row.get("url") or "").strip()
            examples.append(
                VerificationExample(
                    example_id=f"healthfc:{row_index}",
                    claim=claim,
                    evidence=evidence,
                    label=normalize_healthfc_label(row.get("label")),
                    dataset="healthfc",
                    split=split,
                    group_id=source_url or f"healthfc-row:{row_index}",
                    document_ids=(source_url,) if source_url else (),
                    metadata={"row_index": row_index, "source_url": source_url},
                )
            )
    if not examples:
        raise ConfigurationError("No trainable HealthFC examples were found")
    return examples


def _read_bioasq_questions(archive: Path) -> list[dict[str, Any]]:
    if not archive.is_file():
        raise ConfigurationError(f"BioASQ archive does not exist: {archive}")
    try:
        with zipfile.ZipFile(archive) as handle:
            json_names = [name for name in handle.namelist() if name.casefold().endswith(".json")]
            questions: list[dict[str, Any]] = []
            for name in sorted(json_names):
                payload = json.loads(handle.read(name))
                questions.extend(payload.get("questions", []))
            return questions
    except (OSError, zipfile.BadZipFile, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"Cannot read BioASQ archive: {archive}") from exc


def _bioasq_answer(value: Any) -> Verdict:
    if isinstance(value, list):
        value = value[0] if value else ""
    normalized = str(value).strip().casefold().rstrip(".")
    if normalized == "yes":
        return Verdict.TRUE
    if normalized == "no":
        return Verdict.FALSE
    raise ConfigurationError(f"Unknown BioASQ yes/no answer: {value!r}")


def load_bioasq_7b(
    training_archive: Path,
    golden_archive: Path,
) -> list[VerificationExample]:
    """Load the official BioASQ-7b yes/no subset and evidence snippets.

    Factoid, list, and summary questions are deliberately excluded because they
    do not map to the verifier's TRUE/FALSE/NEI output space. Exact duplicate
    questions with the same answer are merged; conflicting duplicate labels are
    excluded instead of teaching the model both verdicts for the same claim.
    """

    raw_examples: list[VerificationExample] = []
    for split, archive in (("train", training_archive), ("test", golden_archive)):
        for row in _read_bioasq_questions(archive):
            if str(row.get("type") or "").casefold() != "yesno":
                continue
            snippets = row.get("snippets") or []
            evidence_parts: list[str] = []
            document_ids: list[str] = []
            seen_text: set[str] = set()
            for snippet in snippets:
                text = " ".join(str(snippet.get("text") or "").split())
                if text and text.casefold() not in seen_text:
                    seen_text.add(text.casefold())
                    evidence_parts.append(text)
                document = str(snippet.get("document") or "").strip()
                if document and document not in document_ids:
                    document_ids.append(document)
            if not evidence_parts:
                continue
            question_id = str(row.get("id") or "").strip()
            raw_examples.append(
                VerificationExample(
                    example_id=f"bioasq7b:{split}:{question_id}",
                    claim=" ".join(str(row.get("body") or "").split()),
                    evidence="\n".join(evidence_parts),
                    label=_bioasq_answer(row.get("exact_answer")),
                    dataset="bioasq7b",
                    split=split,
                    group_id=question_id,
                    document_ids=tuple(document_ids),
                    metadata={"question_type": "yesno", "question_id": question_id},
                )
            )
    grouped: dict[tuple[str, str], list[VerificationExample]] = {}
    for item in raw_examples:
        grouped.setdefault((item.split, _normalized_claim(item.claim)), []).append(item)

    examples: list[VerificationExample] = []
    for items in grouped.values():
        labels = {item.label for item in items}
        if len(labels) != 1:
            continue
        first = items[0]
        evidence_parts: list[str] = []
        seen_evidence: set[str] = set()
        document_ids: list[str] = []
        question_ids: list[str] = []
        for item in items:
            question_ids.append(str(item.metadata["question_id"]))
            for passage in item.evidence.splitlines():
                key = passage.casefold()
                if passage and key not in seen_evidence:
                    seen_evidence.add(key)
                    evidence_parts.append(passage)
            for document_id in item.document_ids:
                if document_id not in document_ids:
                    document_ids.append(document_id)
        examples.append(
            VerificationExample(
                example_id=first.example_id,
                claim=first.claim,
                evidence="\n".join(evidence_parts),
                label=first.label,
                dataset=first.dataset,
                split=first.split,
                group_id=_claim_group_id(first.claim),
                document_ids=tuple(document_ids),
                metadata={
                    "question_type": "yesno",
                    "question_id": question_ids[0],
                    "merged_question_ids": question_ids,
                    "merged_duplicate_count": len(items),
                },
            )
        )
    if not examples:
        raise ConfigurationError("No labelled BioASQ-7b yes/no examples were found")
    return examples


def _read_jsonl_member(handle: tarfile.TarFile, name: str) -> list[dict[str, Any]]:
    member = handle.extractfile(name)
    if member is None:
        raise ConfigurationError(f"SciFact archive is missing {name}")
    return [json.loads(line) for line in member if line.strip()]


def _scifact_document_text(document: dict[str, Any]) -> str:
    title = " ".join(str(document.get("title") or "").split())
    abstract = " ".join(str(part).strip() for part in document.get("abstract", []) if part)
    return f"{title}. {abstract}".strip(" .")


def _normalized_claim(text: str) -> str:
    return " ".join(text.casefold().split()).strip(" .!?")


def _claim_group_id(claim: str) -> str:
    digest = hashlib.sha256(_normalized_claim(claim).encode("utf-8")).hexdigest()[:16]
    return f"claim:{digest}"


def load_scifact(archive: Path, include_nei: bool = True) -> list[VerificationExample]:
    """Load labelled SciFact train/dev pairs plus conservative NEI negatives.

    The public test split is intentionally not returned because it has no public
    evidence labels. Exact claim duplicates in train and development are removed
    from train so the held-out score is not inflated. For SUPPORT/CONTRADICT
    examples, only annotated rationale sentences are used as evidence. At most
    one cited, non-evidence document is retained per claim as an NEI pair to avoid
    overwhelming the other classes.
    """

    if not archive.is_file():
        raise ConfigurationError(f"SciFact archive does not exist: {archive}")
    try:
        with tarfile.open(archive, "r:gz") as handle:
            corpus_rows = _read_jsonl_member(handle, "data/corpus.jsonl")
            corpus = {str(row["doc_id"]): row for row in corpus_rows}
            split_rows = {
                "train": _read_jsonl_member(handle, "data/claims_train.jsonl"),
                "validation": _read_jsonl_member(handle, "data/claims_dev.jsonl"),
            }
    except (OSError, tarfile.TarError, json.JSONDecodeError, KeyError) as exc:
        raise ConfigurationError(f"Cannot read SciFact archive: {archive}") from exc

    examples: list[VerificationExample] = []
    held_out_claims = {
        _normalized_claim(str(row.get("claim") or "")) for row in split_rows["validation"]
    }
    for split, claims in split_rows.items():
        for row in claims:
            claim_id = str(row.get("id"))
            claim = " ".join(str(row.get("claim") or "").split())
            if split == "train" and _normalized_claim(claim) in held_out_claims:
                continue
            claim_group_id = _claim_group_id(claim)
            annotated_doc_ids: set[str] = set()
            for doc_id, groups in (row.get("evidence") or {}).items():
                doc_key = str(doc_id)
                annotated_doc_ids.add(doc_key)
                document = corpus.get(doc_key)
                if document is None:
                    continue
                sentences = [str(item) for item in document.get("abstract", [])]
                for group_index, group in enumerate(groups):
                    indices = tuple(int(index) for index in group.get("sentences", []))
                    rationale = " ".join(
                        sentences[index].strip()
                        for index in indices
                        if 0 <= index < len(sentences) and sentences[index].strip()
                    )
                    if not rationale:
                        continue
                    label_name = str(group.get("label") or "").strip().upper()
                    label_map = {"SUPPORT": Verdict.TRUE, "CONTRADICT": Verdict.FALSE}
                    if label_name not in label_map:
                        raise ConfigurationError(f"Unknown SciFact label: {label_name!r}")
                    examples.append(
                        VerificationExample(
                            example_id=(f"scifact:{split}:{claim_id}:{doc_key}:{group_index}"),
                            claim=claim,
                            evidence=rationale,
                            label=label_map[label_name],
                            dataset="scifact",
                            split=split,
                            group_id=claim_group_id,
                            document_ids=(doc_key,),
                            rationale_sentence_indices=indices,
                            metadata={"claim_id": claim_id, "document_id": doc_key},
                        )
                    )

            if include_nei:
                negative_doc_ids = [
                    str(item)
                    for item in row.get("cited_doc_ids", [])
                    if str(item) not in annotated_doc_ids and str(item) in corpus
                ]
                if negative_doc_ids:
                    doc_key = negative_doc_ids[0]
                    evidence = _scifact_document_text(corpus[doc_key])
                    if evidence:
                        examples.append(
                            VerificationExample(
                                example_id=f"scifact:{split}:{claim_id}:{doc_key}:nei",
                                claim=claim,
                                evidence=evidence,
                                label=Verdict.NEI,
                                dataset="scifact",
                                split=split,
                                group_id=claim_group_id,
                                document_ids=(doc_key,),
                                metadata={
                                    "claim_id": claim_id,
                                    "document_id": doc_key,
                                    "negative_sampling": "cited_non_evidence_document",
                                },
                            )
                        )
    if not examples:
        raise ConfigurationError("No labelled SciFact examples were found")
    return examples


def load_available_datasets(data_dir: Path) -> list[VerificationExample]:
    """Load all three datasets from their documented filenames."""

    return [
        *load_healthfc(data_dir / "HealthFC.csv"),
        *load_bioasq_7b(
            data_dir / "BioASQ-training7b.zip",
            data_dir / "Task7BGoldenEnriched.zip",
        ),
        *load_scifact(data_dir / "data.tar.gz"),
    ]
