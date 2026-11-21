"""Fully local biomedical claim detection using a Transformers zero-shot classifier."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from difflib import SequenceMatcher
from typing import Any

from biomed_video_fc.domain.models import Claim, TextSource, TimedText, TranscriptBundle
from biomed_video_fc.exceptions import ModelUnavailableError

logger = logging.getLogger(__name__)

ScoreFunction = Callable[[list[str]], list[float]]
TARGET_LABEL = "a checkable biomedical factual claim"
OTHER_LABEL = "not a biomedical factual claim"

_PREDICATE = re.compile(
    r"\b(?:is|are|was|were|has|have|had|can|could|may|might|must|will|should|"
    r"cause[sd]?|support[sd]?|help[sd]?|prevent[sd]?|reduce[sd]?|increase[sd]?|"
    r"improve[sd]?|promote[sd]?|produce[sd]?|require[sd]?|lower[sd]?|raise[sd]?|"
    r"lead[sd]?|contain[sd]?)\b",
    re.IGNORECASE,
)
_INCOMPLETE_START = re.compile(
    r"^(?:and|or|but|because|which|that|when\s+it|it(?:'s|\s+is)|these|those)\b",
    re.IGNORECASE,
)
_INCOMPLETE_END = re.compile(
    r"\b(?:a|an|the|and|or|but|because|for|from|of|that|to|with|your)\s*[.!?]?$",
    re.IGNORECASE,
)
_OCR_CITATION_MARKERS = re.compile(
    r"\b(?:article|open access|published|show auth|authors?|journal|doi|pmid)\b",
    re.IGNORECASE,
)
_OCR_BOILERPLATE = re.compile(
    r"^(?:doctor|sethi|follow|for|more|share|gastroenterology|saurabh.*)$",
    re.IGNORECASE,
)


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()


def _meaning_markers(text: str) -> tuple[bool, tuple[str, ...]]:
    negative = bool(
        re.search(r"\b(?:no|not|never|none|without|cannot|can't|isn't|aren't)\b", text, re.I)
    )
    quantities = tuple(
        match.group(0).replace(" ", "").casefold()
        for match in re.finditer(
            r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|g|kg|ml|l|%|percent|hours?|days?|weeks?|months?)?\b",
            text,
            re.I,
        )
    )
    return negative, quantities


def _looks_like_complete_claim(text: str) -> bool:
    words = text.split()
    return (
        4 <= len(words) <= 80
        and any(character.isalpha() for character in text)
        and _PREDICATE.search(text) is not None
        and _INCOMPLETE_START.search(text) is None
        and _INCOMPLETE_END.search(text) is None
    )


def _sentences_with_timing(segments: list[TimedText]) -> list[TimedText]:
    """Rebuild sentences before splitting so Whisper boundaries do not create fragments."""
    if not segments:
        return []

    text_parts: list[str] = []
    spans: list[tuple[int, int, TimedText]] = []
    cursor = 0
    for segment in sorted(segments, key=lambda item: item.start_seconds):
        clean = re.sub(r"\s+", " ", segment.text).strip()
        if not clean:
            continue
        if text_parts:
            cursor += 1
        start = cursor
        text_parts.append(clean)
        cursor += len(clean)
        spans.append((start, cursor, segment))

    full_text = " ".join(text_parts)
    sentences: list[TimedText] = []
    for match in re.finditer(r"\S.*?(?:[.!?](?=\s|$)|$)", full_text):
        sentence = match.group(0).strip()
        overlapping = [
            segment for start, end, segment in spans if match.start() < end and match.end() > start
        ]
        if not sentence or not overlapping:
            continue
        confidence_values = [item.confidence for item in overlapping if item.confidence is not None]
        sentences.append(
            TimedText(
                text=sentence,
                start_seconds=min(item.start_seconds for item in overlapping),
                end_seconds=max(item.end_seconds for item in overlapping),
                source=TextSource.SPEECH,
                confidence=(
                    sum(confidence_values) / len(confidence_values) if confidence_values else None
                ),
            )
        )
    return sentences


def _resolve_speech_context(sentences: list[TimedText]) -> list[TimedText]:
    """Join simple correction pairs and resolve nearby vitamin references."""
    resolved: list[TimedText] = []
    index = 0
    last_vitamin: str | None = None
    correction = re.compile(
        r"^(?P<subject>.+?)\s+(?P<verb>is|are)\s+not\s+(?P<excluded>.+?)[.!?]$",
        re.IGNORECASE,
    )
    replacement = re.compile(r"^It(?:'s|\s+is)\s+(?P<actual>.+?)[.!?]$", re.IGNORECASE)

    while index < len(sentences):
        current = sentences[index]
        text = current.text
        vitamin_matches = re.findall(r"\bvitamin\s+[a-z0-9]+\b", text, re.IGNORECASE)
        if vitamin_matches:
            last_vitamin = vitamin_matches[-1]

        if index + 1 < len(sentences):
            first_match = correction.match(text)
            second_match = replacement.match(sentences[index + 1].text)
            if first_match and second_match:
                following = sentences[index + 1]
                pair_confidences = [
                    value
                    for value in [current.confidence, following.confidence]
                    if value is not None
                ]
                text = (
                    f"{first_match.group('subject')} {first_match.group('verb')} "
                    f"{second_match.group('actual')}, not {first_match.group('excluded')}."
                )
                current = current.model_copy(
                    update={
                        "text": text,
                        "end_seconds": following.end_seconds,
                        "confidence": min(pair_confidences) if pair_confidences else None,
                    }
                )
                index += 1

        if last_vitamin:
            text = re.sub(
                r"^When it(?:'s|\s+is) low\b",
                f"When {last_vitamin} is low",
                current.text,
                flags=re.IGNORECASE,
            )
            current = current.model_copy(update={"text": text})
        resolved.append(current)
        index += 1
    return resolved


def _ocr_candidates(
    segments: list[TimedText], speech_corpus: str, speech_sentences: list[TimedText]
) -> list[TimedText]:
    candidates: list[TimedText] = []
    for segment in sorted(segments, key=lambda item: item.start_seconds):
        if _OCR_CITATION_MARKERS.search(segment.text):
            continue
        lines = [
            clean
            for part in re.split(r"\s*\|\s*|[\r\n]+", segment.text)
            if (clean := re.sub(r"\s+", " ", part).strip(" -–—:;,."))
            and _OCR_BOILERPLATE.match(clean) is None
        ]
        text = " ".join(lines).strip()
        if not _looks_like_complete_claim(text):
            continue
        normalized = _normalize(text)
        if normalized and normalized in speech_corpus:
            matches_complete_speech = any(
                SequenceMatcher(None, normalized, _normalize(item.text)).ratio() >= 0.90
                for item in speech_sentences
            )
            if not matches_complete_speech:
                continue
        candidates.append(segment.model_copy(update={"text": text}))
    return candidates


class LocalBiomedicalClaimExtractor:
    def __init__(
        self,
        model_name: str,
        max_claims: int = 5,
        min_confidence: float = 0.55,
        device: int = -1,
        score_function: ScoreFunction | None = None,
    ) -> None:
        self.model_name = model_name
        self.max_claims = max_claims
        self.min_confidence = min_confidence
        self.device = device
        self.score_function = score_function
        self._classifier: Any = None

    def _load_classifier(self) -> Any:
        if self._classifier is not None:
            return self._classifier
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise ModelUnavailableError(
                "Transformers is required for local claim detection. "
                "Install: pip install -e '.[local-ml]'"
            ) from exc
        logger.info("Loading local claim detector %s", self.model_name)
        self._classifier = pipeline(
            "zero-shot-classification", model=self.model_name, device=self.device
        )
        return self._classifier

    def _score(self, texts: list[str]) -> list[float]:
        if self.score_function is not None:
            return self.score_function(texts)
        raw = self._load_classifier()(
            texts,
            candidate_labels=[TARGET_LABEL, OTHER_LABEL],
            hypothesis_template="This text is {}.",
            multi_label=False,
        )
        results = raw if isinstance(raw, list) else [raw]
        scores: list[float] = []
        for result in results:
            by_label = dict(zip(result["labels"], result["scores"], strict=False))
            scores.append(float(by_label.get(TARGET_LABEL, 0.0)))
        return scores

    @staticmethod
    def _candidates(transcript: TranscriptBundle) -> list[tuple[str, TimedText]]:
        speech_segments = [
            item for item in transcript.combined_segments if item.source is TextSource.SPEECH
        ]
        ocr_segments = [
            item for item in transcript.combined_segments if item.source is TextSource.OCR
        ]
        speech_sentences = _resolve_speech_context(_sentences_with_timing(speech_segments))
        speech_corpus = _normalize(" ".join(item.text for item in speech_segments))
        complete = [item for item in speech_sentences if _looks_like_complete_claim(item.text)]
        complete.extend(_ocr_candidates(ocr_segments, speech_corpus, complete))
        return [(item.text.strip(" -–—:;,."), item) for item in complete]

    @staticmethod
    def _source(segment: TimedText) -> str:
        return "speech" if segment.source is TextSource.SPEECH else "ocr"

    def extract(self, transcript: TranscriptBundle) -> list[Claim]:
        candidates = self._candidates(transcript)
        if not candidates:
            return []
        scores = self._score([text for text, _ in candidates])
        accepted: list[Claim] = []
        for (text, segment), score in zip(candidates, scores, strict=True):
            if score < self.min_confidence:
                continue
            duplicate_index = next(
                (
                    index
                    for index, existing in enumerate(accepted)
                    if SequenceMatcher(None, _normalize(existing.text), _normalize(text)).ratio()
                    >= 0.90
                    and _meaning_markers(existing.text) == _meaning_markers(text)
                ),
                None,
            )
            if duplicate_index is None:
                accepted.append(
                    Claim(
                        text=text.rstrip(".") + ".",
                        source=self._source(segment),
                        start_seconds=segment.start_seconds,
                        end_seconds=segment.end_seconds,
                        extraction_confidence=score,
                    )
                )
                continue
            existing = accepted[duplicate_index]
            source = existing.source
            if source != self._source(segment):
                source = "both"
            accepted[duplicate_index] = existing.model_copy(
                update={
                    "source": source,
                    "start_seconds": min(existing.start_seconds or 0, segment.start_seconds),
                    "end_seconds": max(existing.end_seconds or 0, segment.end_seconds),
                    "extraction_confidence": max(existing.extraction_confidence or 0, score),
                }
            )
        accepted.sort(key=lambda item: item.extraction_confidence or 0, reverse=True)
        selected = accepted[: self.max_claims]
        selected.sort(key=lambda item: item.start_seconds or 0)
        logger.info(
            "Local claim detector accepted %d/%d candidates", len(selected), len(candidates)
        )
        return selected
