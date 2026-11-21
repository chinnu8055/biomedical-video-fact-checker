"""Application orchestrator; every processing stage is injected and independently testable."""

from __future__ import annotations

import logging
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

from biomed_video_fc.domain.models import (
    AnalysisResult,
    ClaimResult,
    StageTiming,
    TimedText,
    Verdict,
    Verification,
)
from biomed_video_fc.exceptions import BioFactCheckError, MediaProcessingError
from biomed_video_fc.ports import (
    AudioExtractor,
    ClaimExtractor,
    ClaimReasoner,
    EvidenceRetriever,
    FrameExtractor,
    OCRReader,
    ProgressCallback,
    SpeechTranscriber,
    TranscriptMerger,
)

logger = logging.getLogger(__name__)


class BiomedicalFactCheckPipeline:
    def __init__(
        self,
        audio_extractor: AudioExtractor,
        frame_extractor: FrameExtractor,
        transcriber: SpeechTranscriber,
        ocr_reader: OCRReader,
        transcript_merger: TranscriptMerger,
        claim_extractor: ClaimExtractor,
        evidence_retriever: EvidenceRetriever,
        reasoner: ClaimReasoner,
    ) -> None:
        self.audio_extractor = audio_extractor
        self.frame_extractor = frame_extractor
        self.transcriber = transcriber
        self.ocr_reader = ocr_reader
        self.transcript_merger = transcript_merger
        self.claim_extractor = claim_extractor
        self.evidence_retriever = evidence_retriever
        self.reasoner = reasoner

    @staticmethod
    def _run_timed(name: str, timings: list[StageTiming], operation: Callable):
        started = time.perf_counter()
        try:
            return operation()
        finally:
            timings.append(StageTiming(stage=name, seconds=time.perf_counter() - started))

    def analyze(
        self,
        video_path: str | Path,
        progress: ProgressCallback | None = None,
        include_ocr: bool = True,
    ) -> AnalysisResult:
        path = Path(video_path)
        if not path.is_file():
            raise MediaProcessingError(f"Video does not exist: {path}")
        if path.suffix.casefold() != ".mp4":
            raise MediaProcessingError("Only .mp4 video input is supported")

        report = progress or (lambda _message, _fraction: None)
        timings: list[StageTiming] = []
        warnings: list[str] = []
        speech: list[TimedText] = []
        ocr: list[TimedText] = []

        logger.info("Starting analysis for %s", path.name)
        with tempfile.TemporaryDirectory(prefix="biofc-") as temp_dir:
            workspace = Path(temp_dir)
            audio_path = workspace / "audio.wav"
            frames_dir = workspace / "frames"

            report("Extracting audio", 0.05)
            try:
                self._run_timed(
                    "audio_extraction",
                    timings,
                    lambda: self.audio_extractor.extract(path, audio_path),
                )
                report("Transcribing speech", 0.18)
                speech = self._run_timed(
                    "speech_transcription", timings, lambda: self.transcriber.transcribe(audio_path)
                )
            except BioFactCheckError as exc:
                message = f"Speech processing was skipped: {exc}"
                warnings.append(message)
                logger.warning(message)

            if include_ocr:
                report("Sampling video frames", 0.32)
                try:
                    frames = self._run_timed(
                        "frame_extraction",
                        timings,
                        lambda: self.frame_extractor.extract(path, frames_dir),
                    )
                    report("Reading on-screen text", 0.48)
                    ocr = self._run_timed(
                        "ocr", timings, lambda: self.ocr_reader.read_frames(frames)
                    )
                    ocr = [item.model_copy(update={"frame_path": None}) for item in ocr]
                except BioFactCheckError as exc:
                    message = f"On-screen text processing was skipped: {exc}"
                    warnings.append(message)
                    logger.warning(message)
            else:
                report("On-screen text extraction disabled", 0.48)
                logger.info("Skipping frame extraction and OCR because OCR is disabled")

            if not speech and not ocr:
                if not include_ocr:
                    raise MediaProcessingError(
                        "No speech could be extracted and on-screen text extraction was disabled. "
                        "Enable OCR and analyze the video again."
                    )
                raise MediaProcessingError(
                    "Neither speech nor readable on-screen text could be extracted from the video"
                )

            report("Combining speech and on-screen text", 0.60)
            transcript = self._run_timed(
                "transcript_merge", timings, lambda: self.transcript_merger.merge(speech, ocr)
            )
            report("Detecting biomedical claims", 0.68)
            claims = self._run_timed(
                "claim_extraction", timings, lambda: self.claim_extractor.extract(transcript)
            )

            claim_results: list[ClaimResult] = []
            for index, claim in enumerate(claims):
                fraction = 0.72 + (0.24 * index / max(len(claims), 1))
                report(f"Checking claim {index + 1} of {len(claims)}", fraction)
                evidence = []
                try:
                    evidence = self._run_timed(
                        f"retrieval:{claim.id}",
                        timings,
                        lambda current=claim: self.evidence_retriever.retrieve(current),
                    )
                    verification = self._run_timed(
                        f"reasoning:{claim.id}",
                        timings,
                        lambda current=claim, docs=evidence: self.reasoner.verify(current, docs),
                    )
                except BioFactCheckError as exc:
                    message = f"Claim {claim.id} requires review because verification failed: {exc}"
                    warnings.append(message)
                    logger.warning(message)
                    verification = Verification(
                        verdict=Verdict.NEI,
                        confidence=0.0,
                        reasoning=message,
                        limitations=[
                            "This is a processing failure, not an evidence-based NEI result."
                        ],
                        decision_source="error",
                        review_required=True,
                    )
                claim_results.append(
                    ClaimResult(claim=claim, evidence=evidence, verification=verification)
                )

        report("Analysis complete", 1.0)
        logger.info("Completed %s with %d claims", path.name, len(claim_results))
        return AnalysisResult(
            video_name=path.name,
            transcript=transcript,
            claim_results=claim_results,
            ocr_enabled=include_ocr,
            timings=timings,
            warnings=warnings,
        )
