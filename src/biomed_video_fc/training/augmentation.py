"""Create deterministic hard examples for entity and negation sensitivity."""

from __future__ import annotations

import hashlib
import random
import re

from biomed_video_fc.domain.models import Verdict
from biomed_video_fc.retrieval.entity_validation import (
    ENTITY_ALIASES,
    ENTITY_LABELS,
    find_biomedical_entities,
    replace_first_entity,
)
from biomed_video_fc.training.datasets import VerificationExample

_NEGATION = re.compile(r"\b(?:no|not|never|without|cannot|can't|doesn't|isn't|aren't)\b", re.I)
_AUXILIARY = re.compile(
    r"\b(is|are|was|were|has|have|had|can|could|will|would|should|may|might|must)\b",
    re.IGNORECASE,
)
_SIGN_RELATION = re.compile(
    r"\b(?P<copula>is|are|was|were)\s+(?:an?|the)\s+"
    r"(?:(?:early|common|initial)\s+)?"
    r"(?P<noun>sign|symptom|manifestation)s?\s+of\b",
    re.IGNORECASE,
)
_SIGN_QUESTION = re.compile(
    r"^\s*is\s+(?P<subject>.+?)\s+(?:one\s+of\s+)?(?:(?:a|an|the)\s+)?"
    r"(?:cardinal\s+)?(?P<noun>sign|symptom|manifestation)s?\s+of\s+"
    r"(?P<condition>.+?)\?\s*$",
    re.IGNORECASE,
)
_FIRST_QUALIFIER = re.compile(
    r"\b(?:first|earliest|initial|initially|presenting|presentation)\b", re.IGNORECASE
)


def _augmented_example(
    source: VerificationExample,
    claim: str,
    label: Verdict,
    augmentation: str,
) -> VerificationExample:
    fingerprint = hashlib.sha256(
        f"{source.example_id}\n{augmentation}\n{claim}".encode()
    ).hexdigest()[:12]
    metadata = dict(source.metadata)
    metadata.update(
        {
            "augmentation": augmentation,
            "source_example_id": source.example_id,
            "generated_label": label.value,
        }
    )
    return VerificationExample(
        example_id=f"{source.example_id}:aug:{fingerprint}",
        claim=claim,
        evidence=source.evidence,
        label=label,
        dataset=f"{source.dataset}_augmented",
        split="train",
        group_id=source.group_id,
        document_ids=source.document_ids,
        rationale_sentence_indices=source.rationale_sentence_indices,
        metadata=metadata,
    )


def _entity_swap(source: VerificationExample) -> VerificationExample | None:
    claim_entities = sorted(find_biomedical_entities(source.claim))
    evidence_entities = find_biomedical_entities(source.evidence)
    if not claim_entities:
        return None
    entity = claim_entities[0]
    is_vitamin = entity.startswith("vitamin_")
    alternatives = [
        candidate
        for candidate in ENTITY_ALIASES
        if candidate != entity
        and candidate not in evidence_entities
        and candidate.startswith("vitamin_") == is_vitamin
    ]
    if not alternatives:
        return None
    replacement_entity = alternatives[0]
    swapped = replace_first_entity(source.claim, entity, ENTITY_LABELS[replacement_entity])
    if not swapped or swapped == source.claim:
        return None
    return _augmented_example(source, swapped, Verdict.NEI, "wrong_entity_hard_negative")


def _negated_claim(source: VerificationExample) -> VerificationExample | None:
    if source.claim.rstrip().endswith("?") or _NEGATION.search(source.claim):
        return None
    match = _AUXILIARY.search(source.claim)
    if match is None:
        return None
    auxiliary = match.group(0)
    replacement = "cannot" if auxiliary.casefold() == "can" else f"{auxiliary} not"
    negated = source.claim[: match.start()] + replacement + source.claim[match.end() :]
    return _augmented_example(source, negated, Verdict.FALSE, "negation_flip")


def _unsupported_first_qualifier(source: VerificationExample) -> VerificationExample | None:
    if _FIRST_QUALIFIER.search(source.evidence):
        return None
    match = _SIGN_RELATION.search(source.claim)
    if match is not None:
        replacement = f"{match.group('copula')} the first {match.group('noun')} of"
        transformed = source.claim[: match.start()] + replacement + source.claim[match.end() :]
    else:
        question = _SIGN_QUESTION.match(source.claim)
        if question is None:
            return None
        transformed = (
            f"{question.group('subject')} is the first {question.group('noun')} "
            f"of {question.group('condition')}."
        )
    return _augmented_example(source, transformed, Verdict.NEI, "unsupported_first_qualifier")


def build_hard_negatives(
    examples: list[VerificationExample],
    max_ratio: float = 0.25,
    seed: int = 42,
) -> list[VerificationExample]:
    """Generate a capped, reproducible set from supported training pairs only."""

    if max_ratio < 0:
        raise ValueError("max_ratio cannot be negative")
    limit = int(len(examples) * max_ratio)
    if limit == 0:
        return []
    candidates: list[VerificationExample] = []
    seen: set[tuple[str, str]] = set()
    for source in examples:
        if source.label is not Verdict.TRUE:
            continue
        for generated in (
            _entity_swap(source),
            _negated_claim(source),
            _unsupported_first_qualifier(source),
        ):
            if generated is None:
                continue
            key = (generated.claim.casefold(), generated.evidence.casefold())
            if key not in seen:
                seen.add(key)
                candidates.append(generated)
    random.Random(seed).shuffle(candidates)
    return candidates[:limit]
