"""Conservative decomposition of compound biomedical claims.

The component expands only structures whose meaning can be preserved without a
generative model.  Ambiguous sentences remain intact rather than being rewritten
speculatively.
"""

from __future__ import annotations

import re
from typing import Protocol

from biomed_video_fc.domain.models import Claim, TranscriptBundle


class _BaseExtractor(Protocol):
    def extract(self, transcript: TranscriptBundle) -> list[Claim]: ...


_SIGN_LIST = re.compile(
    r"^(?:the\s+)?(?P<descriptor>first|early|common|initial)?\s*"
    r"(?P<noun>signs?|symptoms?|manifestations?)\s+of\s+"
    r"(?P<condition>.+?)\s+(?:is|are|include|includes)\s+(?P<items>.+?)[.!?]?$",
    re.IGNORECASE,
)
_OBJECT_LIST = re.compile(
    r"^(?P<subject>.+?)\s+"
    r"(?P<predicate>causes?|prevents?|reduces?|increases?|improves?|treats?|cures?|"
    r"produces?|promotes?|leads?\s+to)\s+(?P<items>.+?)[.!?]?$",
    re.IGNORECASE,
)


def _split_items(text: str) -> list[str]:
    clean = re.sub(r"\s+", " ", text).strip(" .")
    parts = [
        part.strip(" ,")
        for part in re.split(r"\s*,\s*|\s+and\s+|\s+or\s+", clean, flags=re.IGNORECASE)
        if part.strip(" ,")
    ]
    return parts if len(parts) > 1 else []


def _sentence(text: str) -> str:
    clean = text.strip().rstrip(".!?")
    return (clean[:1].upper() + clean[1:] if clean else clean) + "."


def decompose_text(text: str) -> list[str]:
    """Return faithful atomic statements, or the original statement if unsure."""

    original = _sentence(text)
    sign_match = _SIGN_LIST.match(original)
    if sign_match:
        items = _split_items(sign_match.group("items"))
        if items:
            descriptor = (sign_match.group("descriptor") or "").casefold()
            noun = sign_match.group("noun").casefold()
            condition = sign_match.group("condition").strip()
            relation = f"{descriptor} {noun[:-1] if noun.endswith('s') else noun}".strip()
            atomic: list[str] = []
            for item in items:
                negative = item.casefold().startswith("not ")
                subject = re.sub(r"^not\s+", "", item, flags=re.IGNORECASE).strip()
                plural = subject.casefold().endswith("s") and not subject.casefold().startswith(
                    ("getting ", "taking ", "using ")
                )
                copula = (
                    "are not"
                    if plural and negative
                    else "are"
                    if plural
                    else "is not"
                    if negative
                    else "is"
                )
                article = (
                    "the"
                    if descriptor == "first"
                    else "an"
                    if descriptor in {"early", "initial"}
                    else "a"
                )
                atomic.append(_sentence(f"{subject} {copula} {article} {relation} of {condition}"))
            return atomic

    object_match = _OBJECT_LIST.match(original)
    if object_match:
        items = _split_items(object_match.group("items"))
        if items:
            subject = object_match.group("subject").strip()
            predicate = object_match.group("predicate").strip()
            return [_sentence(f"{subject} {predicate} {item}") for item in items]

    return [original]


class DecomposingClaimExtractor:
    """Decorate an existing claim extractor with conservative atomic splitting."""

    def __init__(self, base: _BaseExtractor, max_atomic_claims: int = 12) -> None:
        self.base = base
        self.max_atomic_claims = max_atomic_claims

    def extract(self, transcript: TranscriptBundle) -> list[Claim]:
        output: list[Claim] = []
        for parent in self.base.extract(transcript):
            atomic_texts = decompose_text(parent.text)
            if len(atomic_texts) == 1:
                output.append(parent.model_copy(update={"is_atomic": True}))
                continue
            for index, text in enumerate(atomic_texts):
                output.append(
                    Claim(
                        text=text,
                        source=parent.source,
                        start_seconds=parent.start_seconds,
                        end_seconds=parent.end_seconds,
                        extraction_confidence=parent.extraction_confidence,
                        parent_claim_id=parent.id,
                        parent_text=parent.text,
                        atomic_index=index,
                        is_atomic=True,
                    )
                )
                if len(output) >= self.max_atomic_claims:
                    return output
        return output[: self.max_atomic_claims]
