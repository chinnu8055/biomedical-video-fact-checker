"""Typed domain models for traceable fact-checking results."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator


class Verdict(StrEnum):
    TRUE = "TRUE"
    FALSE = "FALSE"
    NEI = "NOT ENOUGH INFORMATION"


class EvidenceStance(StrEnum):
    SUPPORTS = "SUPPORTS"
    CONTRADICTS = "CONTRADICTS"
    NEUTRAL = "NEUTRAL"


class TextSource(StrEnum):
    SPEECH = "speech"
    OCR = "ocr"


class TimedText(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str = Field(min_length=1)
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(ge=0)
    source: TextSource
    confidence: float | None = Field(default=None, ge=0, le=1)
    frame_path: Path | None = None

    @model_validator(mode="after")
    def validate_time_range(self) -> TimedText:
        if self.end_seconds < self.start_seconds:
            raise ValueError("end_seconds cannot be before start_seconds")
        return self


class TranscriptBundle(BaseModel):
    speech_segments: list[TimedText] = Field(default_factory=list)
    ocr_segments: list[TimedText] = Field(default_factory=list)
    combined_segments: list[TimedText] = Field(default_factory=list)

    @property
    def speech_text(self) -> str:
        return " ".join(item.text for item in self.speech_segments).strip()

    @property
    def ocr_text(self) -> str:
        return " ".join(item.text for item in self.ocr_segments).strip()

    @property
    def combined_text(self) -> str:
        return "\n".join(
            f"[{item.start_seconds:07.2f}s][{item.source.value.upper()}] {item.text}"
            for item in self.combined_segments
        )


class Claim(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex[:12])
    text: str = Field(min_length=5)
    source: Literal["speech", "ocr", "both", "combined"] = "combined"
    start_seconds: float | None = Field(default=None, ge=0)
    end_seconds: float | None = Field(default=None, ge=0)
    extraction_confidence: float | None = Field(default=None, ge=0, le=1)
    parent_claim_id: str | None = None
    parent_text: str | None = None
    atomic_index: int | None = Field(default=None, ge=0)
    is_atomic: bool = False


class Evidence(BaseModel):
    pmid: str
    title: str
    abstract: str
    journal: str | None = None
    publication_date: str | None = None
    doi: str | None = None
    url: HttpUrl
    relevance_score: float | None = Field(default=None, ge=0, le=1)
    term_coverage: float | None = Field(default=None, ge=0, le=1)
    entity_match: bool | None = None
    matched_entities: list[str] = Field(default_factory=list)
    missing_entities: list[str] = Field(default_factory=list)
    additional_entities: list[str] = Field(default_factory=list)
    retrieval_queries: list[str] = Field(default_factory=list)
    selection_reason: str | None = None


class EvidenceAssessment(BaseModel):
    pmid: str
    stance: EvidenceStance
    model_stance: EvidenceStance | None = None
    confidence: float = Field(ge=0, le=1)
    passage: str = Field(min_length=1)
    relevance_score: float | None = Field(default=None, ge=0, le=1)
    model_input_truncated: bool = False
    constraint_warnings: list[str] = Field(default_factory=list)


class Verification(BaseModel):
    verdict: Verdict
    confidence: float = Field(ge=0, le=1)
    reasoning: str = Field(min_length=1)
    supporting_pmids: list[str] = Field(default_factory=list)
    contradicting_pmids: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    evidence_assessments: list[EvidenceAssessment] = Field(default_factory=list)
    decision_source: Literal["model_aggregation", "no_evidence", "error"] = "model_aggregation"
    review_required: bool = False


class ClaimResult(BaseModel):
    claim: Claim
    evidence: list[Evidence]
    verification: Verification


class StageTiming(BaseModel):
    stage: str
    seconds: float = Field(ge=0)


class AnalysisResult(BaseModel):
    analysis_id: str = Field(default_factory=lambda: uuid4().hex)
    video_name: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    transcript: TranscriptBundle
    claim_results: list[ClaimResult]
    ocr_enabled: bool = True
    timings: list[StageTiming] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @property
    def claims(self) -> list[Claim]:
        return [item.claim for item in self.claim_results]
