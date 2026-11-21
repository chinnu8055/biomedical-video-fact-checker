"""Lazy-loading Whisper adapter that preserves segment timestamps."""

from __future__ import annotations

import logging
import wave
from pathlib import Path
from typing import Any

import numpy as np

from biomed_video_fc.domain.models import TextSource, TimedText
from biomed_video_fc.exceptions import MediaProcessingError, ModelUnavailableError

logger = logging.getLogger(__name__)


class WhisperTranscriber:
    def __init__(self, model_name: str = "small", device: str = "auto", language: str = "") -> None:
        self.model_name = model_name
        self.device = device
        self.language = language or None
        self._model: Any = None

    @staticmethod
    def _load_extracted_wav(audio_path: Path) -> np.ndarray | str:
        """Load the pipeline's PCM WAV without making Whisper spawn FFmpeg."""
        if audio_path.suffix.casefold() != ".wav":
            return str(audio_path)

        with wave.open(str(audio_path), "rb") as wav_file:
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            sample_rate = wav_file.getframerate()
            frames = wav_file.readframes(wav_file.getnframes())

        # FFmpegAudioExtractor always produces this format. Keep Whisper's normal
        # file loader as a fallback for WAV files supplied from elsewhere.
        if channels != 1 or sample_width != 2 or sample_rate != 16_000:
            return str(audio_path)

        return np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            import whisper
        except ImportError as exc:
            raise ModelUnavailableError(
                "Whisper is not installed. Run: pip install -e '.[speech]'"
            ) from exc
        kwargs = {} if self.device == "auto" else {"device": self.device}
        logger.info("Loading Whisper model %s", self.model_name)
        self._model = whisper.load_model(self.model_name, **kwargs)
        return self._model

    def transcribe(self, audio_path: Path) -> list[TimedText]:
        try:
            audio = self._load_extracted_wav(audio_path)
            result = self._load_model().transcribe(
                audio, language=self.language, fp16=self.device not in {"cpu"}
            )
        except Exception as exc:
            raise MediaProcessingError(f"Whisper transcription failed: {exc}") from exc

        segments: list[TimedText] = []
        for raw in result.get("segments", []):
            text = str(raw.get("text", "")).strip()
            if not text:
                continue
            no_speech = float(raw.get("no_speech_prob", 0.0))
            segments.append(
                TimedText(
                    text=text,
                    start_seconds=max(0.0, float(raw.get("start", 0.0))),
                    end_seconds=max(0.0, float(raw.get("end", raw.get("start", 0.0)))),
                    source=TextSource.SPEECH,
                    confidence=max(0.0, min(1.0, 1.0 - no_speech)),
                )
            )
        logger.info("Whisper returned %d speech segments", len(segments))
        return segments
