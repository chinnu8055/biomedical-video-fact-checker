"""Local BiomedBERT biomedical verdict classifier and evidence aggregator."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from biomed_video_fc.domain.models import (
    Claim,
    Evidence,
    EvidenceAssessment,
    EvidenceStance,
    Verdict,
    Verification,
)
from biomed_video_fc.exceptions import ModelUnavailableError
from biomed_video_fc.reasoning.constraints import audit_stance_constraints

logger = logging.getLogger(__name__)

DEFAULT_ID_TO_VERDICT = {0: Verdict.TRUE, 1: Verdict.NEI, 2: Verdict.FALSE}


class HealthFCClassifierReasoner:
    def __init__(self, model_dir: Path, max_length: int = 512) -> None:
        self.model_dir = Path(model_dir)
        self.max_length = max_length
        self._model: Any = None
        self._tokenizer: Any = None
        self._torch: Any = None

    def _load(self) -> None:
        if self._model is not None:
            return
        if not (self.model_dir / "config.json").is_file():
            raise ModelUnavailableError(
                "The verifier checkpoint is unavailable. Train with biofc-train-multisource "
                "or configure BIOFC_CLASSIFIER_DIR."
            )
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise ModelUnavailableError(
                "PyTorch and Transformers are required. Install: pip install -e '.[local-ml]'"
            ) from exc
        logger.info("Loading local HealthFC classifier from %s", self.model_dir)
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(device)
        self._model.eval()

    @staticmethod
    def _verdict_for(model: Any, label_id: int) -> Verdict:
        configured = model.config.id2label.get(label_id, f"LABEL_{label_id}")
        normalized = str(configured).strip().upper().replace("_", " ")
        aliases = {
            "SUPPORTED": Verdict.TRUE,
            "TRUE": Verdict.TRUE,
            "REFUTED": Verdict.FALSE,
            "FALSE": Verdict.FALSE,
            "NEI": Verdict.NEI,
            "NOT ENOUGH INFORMATION": Verdict.NEI,
        }
        return aliases.get(normalized, DEFAULT_ID_TO_VERDICT[label_id])

    @staticmethod
    def _terms(text: str) -> set[str]:
        stop = {
            "a",
            "an",
            "and",
            "are",
            "as",
            "at",
            "be",
            "by",
            "for",
            "from",
            "in",
            "is",
            "it",
            "of",
            "on",
            "or",
            "that",
            "the",
            "this",
            "to",
            "was",
            "were",
            "with",
        }
        return {
            item
            for item in re.findall(r"[a-z0-9]+", text.casefold())
            if item not in stop and (len(item) > 1 or item == "c")
        }

    @classmethod
    def _best_passage(cls, claim: str, evidence: Evidence, window: int = 2) -> str:
        sentences = [
            item.strip() for item in re.split(r"(?<=[.!?])\s+", evidence.abstract) if item.strip()
        ]
        if not sentences:
            return f"{evidence.title}. {evidence.abstract}".strip()
        claim_terms = cls._terms(claim)
        candidates: list[tuple[float, str]] = []
        for start in range(len(sentences)):
            abstract_window = " ".join(sentences[start : start + window])
            passage = f"{evidence.title}. {abstract_window}".strip()
            passage_terms = cls._terms(passage)
            overlap = len(claim_terms & passage_terms) / max(len(claim_terms), 1)
            candidates.append((overlap, passage))
        return max(candidates, key=lambda item: item[0])[1]

    def _predict_pair(self, claim: str, passage: str) -> tuple[Verdict, float, bool]:
        device = next(self._model.parameters()).device
        untruncated = self._tokenizer(claim, passage, add_special_tokens=True)
        truncated = len(untruncated.get("input_ids", [])) > self.max_length
        encoded = self._tokenizer(
            claim,
            passage,
            return_tensors="pt",
            truncation="only_second",
            max_length=self.max_length,
        )
        encoded = {name: value.to(device) for name, value in encoded.items()}
        with self._torch.no_grad():
            logits = self._model(**encoded).logits[0]
            probabilities = self._torch.softmax(logits, dim=-1)
        label_id = int(self._torch.argmax(probabilities).item())
        return (
            self._verdict_for(self._model, label_id),
            float(probabilities[label_id].item()),
            truncated,
        )

    @staticmethod
    def _aggregate(
        assessments: list[EvidenceAssessment],
    ) -> tuple[Verdict, float, bool]:
        if not assessments:
            return Verdict.NEI, 0.0, True
        totals = {stance: 0.0 for stance in EvidenceStance}
        for item in assessments:
            relevance = item.relevance_score if item.relevance_score is not None else 0.5
            totals[item.stance] += item.confidence * max(relevance, 0.05)

        support = totals[EvidenceStance.SUPPORTS]
        contradict = totals[EvidenceStance.CONTRADICTS]
        neutral = totals[EvidenceStance.NEUTRAL]
        total = support + contradict + neutral
        if support == 0 and contradict == 0:
            neutral_confidence = max(
                (item.confidence for item in assessments if item.stance is EvidenceStance.NEUTRAL),
                default=0.0,
            )
            return Verdict.NEI, neutral_confidence, True

        stance_confidence = {
            stance: max(
                (item.confidence for item in assessments if item.stance is stance),
                default=0.0,
            )
            for stance in EvidenceStance
        }
        if support > 0 and contradict > 0:
            disagreement = abs(support - contradict) / max(support, contradict)
            if disagreement < 0.25:
                leading_stance = (
                    EvidenceStance.SUPPORTS if support >= contradict else EvidenceStance.CONTRADICTS
                )
                confidence = (max(support, contradict) / total) * stance_confidence[leading_stance]
                return Verdict.NEI, confidence, True
        winning_stance = (
            EvidenceStance.SUPPORTS if support > contradict else EvidenceStance.CONTRADICTS
        )
        verdict = Verdict.TRUE if winning_stance is EvidenceStance.SUPPORTS else Verdict.FALSE
        winning = max(support, contradict)
        review_required = neutral > winning or min(support, contradict) >= winning * 0.75
        confidence = (winning / total) * stance_confidence[winning_stance]
        if review_required:
            neutral_confidence = (neutral / total) * stance_confidence[EvidenceStance.NEUTRAL]
            return Verdict.NEI, max(confidence, neutral_confidence), True
        return verdict, confidence, False

    def verify(self, claim: Claim, evidence: list[Evidence]) -> Verification:
        if not evidence:
            return Verification(
                verdict=Verdict.NEI,
                confidence=0.0,
                reasoning="The PubMed search returned no usable evidence for this claim.",
                limitations=["No local evidence was retrieved."],
                decision_source="no_evidence",
                review_required=True,
            )
        self._load()
        assessments: list[EvidenceAssessment] = []
        for item in evidence:
            passage = self._best_passage(claim.text, item)
            pair_verdict, pair_confidence, truncated = self._predict_pair(claim.text, passage)
            stance = {
                Verdict.TRUE: EvidenceStance.SUPPORTS,
                Verdict.FALSE: EvidenceStance.CONTRADICTS,
                Verdict.NEI: EvidenceStance.NEUTRAL,
            }[pair_verdict]
            constraint_audit = audit_stance_constraints(
                claim.text,
                passage,
                stance,
                missing_entities=item.missing_entities,
            )
            assessments.append(
                EvidenceAssessment(
                    pmid=item.pmid,
                    stance=constraint_audit.final_stance,
                    model_stance=stance,
                    confidence=pair_confidence,
                    passage=passage,
                    relevance_score=item.relevance_score,
                    model_input_truncated=truncated,
                    constraint_warnings=list(constraint_audit.warnings),
                )
            )

        verdict, confidence, review_required = self._aggregate(assessments)
        stance_counts = {
            stance: sum(item.stance is stance for item in assessments) for stance in EvidenceStance
        }
        adjusted_count = sum(
            item.model_stance is not None and item.model_stance is not item.stance
            for item in assessments
        )
        leading = max(
            assessments,
            key=lambda item: item.confidence * (item.relevance_score or 0.5),
        )
        excerpt = " ".join(leading.passage.split())[:420]
        reasoning = (
            f"The local BiomedBERT verifier assessed {len(assessments)} PubMed passages "
            f"independently: {stance_counts[EvidenceStance.SUPPORTS]} support, "
            f"{stance_counts[EvidenceStance.CONTRADICTS]} contradict, and "
            f"{stance_counts[EvidenceStance.NEUTRAL]} are neutral. "
            f"Constraint checks adjusted {adjusted_count} model stance(s). "
            "The strongest assessed passage, "
            f"PMID {leading.pmid}, states: {excerpt}"
        )
        return Verification(
            verdict=verdict,
            confidence=confidence,
            reasoning=reasoning,
            supporting_pmids=[
                item.pmid for item in assessments if item.stance is EvidenceStance.SUPPORTS
            ],
            contradicting_pmids=[
                item.pmid for item in assessments if item.stance is EvidenceStance.CONTRADICTS
            ],
            limitations=[
                "The explanation is extractive; no generative API or external LLM was used.",
                "The prediction depends on retrieval quality and abstract-level evidence.",
                "Entity, qualifier, quantity, and negation checks can conservatively change a "
                "model stance to neutral while retaining the related evidence for review.",
                "Training datasets and live PubMed passages differ; this domain shift must be "
                "measured during evaluation.",
            ],
            evidence_assessments=assessments,
            decision_source="model_aggregation",
            review_required=review_required,
        )
