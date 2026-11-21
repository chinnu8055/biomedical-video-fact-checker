"""Time-align speech and OCR while collapsing repeated on-screen text."""

from __future__ import annotations

import re
from difflib import SequenceMatcher

from biomed_video_fc.domain.models import TextSource, TimedText, TranscriptBundle


def _normalized(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()


_NEGATIONS = {"no", "not", "never", "none", "without", "cannot", "can't", "isn't", "aren't"}
_QUANTITY = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|g|kg|ml|l|%|percent|hours?|days?|weeks?|months?)?\b",
    re.IGNORECASE,
)


def _protected_meaning(text: str) -> tuple[frozenset[str], tuple[str, ...]]:
    words = frozenset(re.findall(r"[a-z]+(?:n't)?", text.casefold()))
    negations = frozenset(words & _NEGATIONS)
    quantities = tuple(
        match.group(0).replace(" ", "").casefold() for match in _QUANTITY.finditer(text)
    )
    return negations, quantities


class TimeAlignedTranscriptMerger:
    def __init__(self, dedup_similarity: float = 0.88, max_gap_seconds: float = 2.5) -> None:
        self.dedup_similarity = dedup_similarity
        self.max_gap_seconds = max_gap_seconds

    def _deduplicate_ocr(self, segments: list[TimedText]) -> list[TimedText]:
        deduplicated: list[TimedText] = []
        for segment in sorted(segments, key=lambda item: item.start_seconds):
            if not deduplicated:
                deduplicated.append(segment)
                continue
            previous = deduplicated[-1]
            gap = segment.start_seconds - previous.end_seconds
            similarity = SequenceMatcher(
                None, _normalized(previous.text), _normalized(segment.text)
            ).ratio()
            same_protected_meaning = _protected_meaning(previous.text) == _protected_meaning(
                segment.text
            )
            if (
                similarity >= self.dedup_similarity
                and same_protected_meaning
                and gap <= self.max_gap_seconds
            ):
                best = (
                    segment if (segment.confidence or 0) > (previous.confidence or 0) else previous
                )
                deduplicated[-1] = best.model_copy(
                    update={
                        "start_seconds": previous.start_seconds,
                        "end_seconds": max(previous.end_seconds, segment.end_seconds),
                    }
                )
            else:
                deduplicated.append(segment)
        return deduplicated

    def merge(self, speech: list[TimedText], ocr: list[TimedText]) -> TranscriptBundle:
        if any(item.source is not TextSource.SPEECH for item in speech):
            raise ValueError("speech list contains a non-speech segment")
        if any(item.source is not TextSource.OCR for item in ocr):
            raise ValueError("ocr list contains a non-OCR segment")
        clean_ocr = self._deduplicate_ocr(ocr)
        combined = sorted([*speech, *clean_ocr], key=lambda item: (item.start_seconds, item.source))
        return TranscriptBundle(
            speech_segments=sorted(speech, key=lambda item: item.start_seconds),
            ocr_segments=clean_ocr,
            combined_segments=combined,
        )
