"""Configuration loading from TOML plus BIOFC_* environment overrides."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class AppConfig(BaseModel):
    name: str = "Biomedical Video Fact Checker"
    max_upload_mb: int = Field(default=500, gt=0)
    allowed_extensions: list[str] = [".mp4"]


class MediaConfig(BaseModel):
    ffmpeg_binary: str = "ffmpeg"
    frame_interval_seconds: float = Field(default=1.0, gt=0)
    audio_sample_rate: int = Field(default=16000, gt=0)
    max_video_seconds: int = Field(default=1800, gt=0)


class SpeechConfig(BaseModel):
    model: str = "small"
    device: str = "auto"
    language: str = ""


class OCRConfig(BaseModel):
    engine: str = "paddle"
    language: str = "en"
    min_confidence: float = Field(default=0.55, ge=0, le=1)
    dedup_similarity: float = Field(default=0.88, ge=0, le=1)


class ClaimConfig(BaseModel):
    max_claims: int = Field(default=5, gt=0, le=25)
    max_atomic_claims: int = Field(default=12, gt=0, le=50)
    model: str = "MoritzLaurer/deberta-v3-large-zeroshot-v1.1-all-33"
    min_confidence: float = Field(default=0.55, ge=0, le=1)
    device: int = -1


class RetrievalConfig(BaseModel):
    max_results: int = Field(default=3, gt=0, le=20)
    candidate_results: int = Field(default=50, gt=0, le=100)
    max_queries: int = Field(default=3, gt=0, le=5)
    email: str = ""
    api_key: str = ""
    tool: str = "biomedical-video-fact-checker"
    timeout_seconds: float = Field(default=20.0, gt=0, le=120)
    retries: int = Field(default=3, ge=0, le=8)
    semantic_rerank: bool = True
    embedding_model: str = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
    minimum_relevance: float = Field(default=0.30, ge=0, le=1)
    minimum_term_coverage: float = Field(default=0.20, ge=0, le=1)


class ClassifierConfig(BaseModel):
    model_dir: Path = Path("artifacts/healthfc_classifier")
    base_model: str = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
    max_length: int = Field(default=512, gt=32, le=4096)


class LoggingConfig(BaseModel):
    level: str = "INFO"
    json_logs: bool = Field(default=False, alias="json")


class Settings(BaseModel):
    app: AppConfig = AppConfig()
    media: MediaConfig = MediaConfig()
    speech: SpeechConfig = SpeechConfig()
    ocr: OCRConfig = OCRConfig()
    claims: ClaimConfig = ClaimConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    classifier: ClassifierConfig = ClassifierConfig()
    logging: LoggingConfig = LoggingConfig()


_ENV_MAP: dict[str, tuple[str, str, type]] = {
    "BIOFC_WHISPER_MODEL": ("speech", "model", str),
    "BIOFC_OCR_ENGINE": ("ocr", "engine", str),
    "BIOFC_FRAME_INTERVAL_SECONDS": ("media", "frame_interval_seconds", float),
    "BIOFC_MAX_CLAIMS": ("claims", "max_claims", int),
    "BIOFC_MAX_ATOMIC_CLAIMS": ("claims", "max_atomic_claims", int),
    "BIOFC_CLAIM_MODEL": ("claims", "model", str),
    "BIOFC_PUBMED_RESULTS": ("retrieval", "max_results", int),
    "BIOFC_PUBMED_CANDIDATES": ("retrieval", "candidate_results", int),
    "BIOFC_PUBMED_QUERIES": ("retrieval", "max_queries", int),
    "BIOFC_MIN_EVIDENCE_RELEVANCE": ("retrieval", "minimum_relevance", float),
    "BIOFC_MIN_EVIDENCE_TERM_COVERAGE": ("retrieval", "minimum_term_coverage", float),
    "BIOFC_NCBI_EMAIL": ("retrieval", "email", str),
    "BIOFC_NCBI_API_KEY": ("retrieval", "api_key", str),
    "BIOFC_CLASSIFIER_DIR": ("classifier", "model_dir", Path),
    "BIOFC_LOG_LEVEL": ("logging", "level", str),
}


def load_settings(path: str | Path = "config/default.toml") -> Settings:
    config_path = Path(path)
    data: dict[str, Any] = {}
    if config_path.exists():
        with config_path.open("rb") as handle:
            data = tomllib.load(handle)

    for env_name, (section, key, converter) in _ENV_MAP.items():
        value = os.getenv(env_name)
        if value is not None and value != "":
            data.setdefault(section, {})[key] = converter(value)
    return Settings.model_validate(data)
