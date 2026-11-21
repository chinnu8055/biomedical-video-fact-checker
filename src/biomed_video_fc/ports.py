"""Protocols that decouple orchestration from external tools and model vendors."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol

from biomed_video_fc.domain.models import Claim, Evidence, TimedText, TranscriptBundle, Verification

ProgressCallback = Callable[[str, float], None]


class AudioExtractor(Protocol):
    def extract(self, video_path: Path, output_path: Path) -> Path: ...


class FrameExtractor(Protocol):
    def extract(self, video_path: Path, output_dir: Path) -> list[tuple[Path, float]]: ...


class SpeechTranscriber(Protocol):
    def transcribe(self, audio_path: Path) -> list[TimedText]: ...


class OCRReader(Protocol):
    def read_frames(self, frames: Sequence[tuple[Path, float]]) -> list[TimedText]: ...


class TranscriptMerger(Protocol):
    def merge(self, speech: list[TimedText], ocr: list[TimedText]) -> TranscriptBundle: ...


class ClaimExtractor(Protocol):
    def extract(self, transcript: TranscriptBundle) -> list[Claim]: ...


class EvidenceRetriever(Protocol):
    def retrieve(self, claim: Claim) -> list[Evidence]: ...


class ClaimReasoner(Protocol):
    def verify(self, claim: Claim, evidence: list[Evidence]) -> Verification: ...
