"""PaddleOCR and EasyOCR adapters with normalized, timestamped output."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from biomed_video_fc.domain.models import TextSource, TimedText
from biomed_video_fc.exceptions import MediaProcessingError, ModelUnavailableError

logger = logging.getLogger(__name__)


class PaddleOCRReader:
    def __init__(
        self, language: str = "en", min_confidence: float = 0.55, frame_duration: float = 1.0
    ) -> None:
        self.language = language
        self.min_confidence = min_confidence
        self.frame_duration = frame_duration
        self._reader: Any = None

    def _load(self) -> Any:
        if self._reader is not None:
            return self._reader
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise ModelUnavailableError(
                "PaddleOCR is not installed. Run: pip install -e '.[ocr-paddle]'"
            ) from exc
        logger.info("Loading PaddleOCR language=%s", self.language)
        try:
            self._reader = PaddleOCR(use_angle_cls=True, lang=self.language, show_log=False)
        except (TypeError, ValueError) as exc:
            if isinstance(exc, ValueError) and "Unknown argument" not in str(exc):
                raise
            # PaddleOCR 3.x replaced the legacy angle-classifier arguments.
            logger.info("Using the PaddleOCR 3.x initialization interface")
            self._reader = PaddleOCR(
                lang=self.language,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=True,
                enable_mkldnn=False,
            )
        return self._reader

    @staticmethod
    def _parse_v2(result: Any) -> list[tuple[str, float]]:
        entries: list[tuple[str, float]] = []
        pages = result if isinstance(result, list) else [result]
        for page in pages:
            if not isinstance(page, list):
                continue
            for line in page:
                if (
                    isinstance(line, (list, tuple))
                    and len(line) >= 2
                    and isinstance(line[1], (list, tuple))
                    and len(line[1]) >= 2
                ):
                    entries.append((str(line[1][0]).strip(), float(line[1][1])))
        return entries

    @staticmethod
    def _parse_v3(result: Any) -> list[tuple[str, float]]:
        entries: list[tuple[str, float]] = []
        for item in result if isinstance(result, list) else [result]:
            payload = getattr(item, "json", item)
            if callable(payload):
                payload = payload()
            if not isinstance(payload, dict):
                continue
            payload = payload.get("res", payload)
            texts = payload.get("rec_texts", [])
            scores = payload.get("rec_scores", [])
            entries.extend(
                (str(text).strip(), float(score))
                for text, score in zip(texts, scores, strict=False)
            )
        return entries

    def read_frames(self, frames: Sequence[tuple[Path, float]]) -> list[TimedText]:
        reader = self._load()
        output: list[TimedText] = []
        for frame_path, timestamp in frames:
            try:
                if hasattr(reader, "predict"):
                    raw = reader.predict(str(frame_path))
                    entries = self._parse_v3(raw)
                else:
                    entries = self._parse_v2(reader.ocr(str(frame_path), cls=True))
            except Exception as exc:
                raise MediaProcessingError(f"OCR failed for {frame_path.name}: {exc}") from exc
            accepted = [
                (text, score) for text, score in entries if text and score >= self.min_confidence
            ]
            if accepted:
                output.append(
                    TimedText(
                        text=" | ".join(text for text, _ in accepted),
                        start_seconds=timestamp,
                        end_seconds=timestamp + self.frame_duration,
                        source=TextSource.OCR,
                        confidence=sum(score for _, score in accepted) / len(accepted),
                        frame_path=frame_path,
                    )
                )
        logger.info("PaddleOCR found text in %d/%d sampled frames", len(output), len(frames))
        return output


class EasyOCRReader:
    def __init__(
        self, language: str = "en", min_confidence: float = 0.55, frame_duration: float = 1.0
    ) -> None:
        self.language = language
        self.min_confidence = min_confidence
        self.frame_duration = frame_duration
        self._reader: Any = None

    def _load(self) -> Any:
        if self._reader is not None:
            return self._reader
        try:
            import easyocr
        except ImportError as exc:
            raise ModelUnavailableError(
                "EasyOCR is not installed. Run: pip install -e '.[ocr-easy]'"
            ) from exc
        self._reader = easyocr.Reader([self.language])
        return self._reader

    def read_frames(self, frames: Sequence[tuple[Path, float]]) -> list[TimedText]:
        output: list[TimedText] = []
        for frame_path, timestamp in frames:
            try:
                raw = self._load().readtext(str(frame_path), detail=1, paragraph=False)
            except Exception as exc:
                raise MediaProcessingError(f"OCR failed for {frame_path.name}: {exc}") from exc
            accepted = [
                (str(text).strip(), float(score))
                for _, text, score in raw
                if str(text).strip() and float(score) >= self.min_confidence
            ]
            if accepted:
                output.append(
                    TimedText(
                        text=" | ".join(text for text, _ in accepted),
                        start_seconds=timestamp,
                        end_seconds=timestamp + self.frame_duration,
                        source=TextSource.OCR,
                        confidence=sum(score for _, score in accepted) / len(accepted),
                        frame_path=frame_path,
                    )
                )
        logger.info("EasyOCR found text in %d/%d sampled frames", len(output), len(frames))
        return output
