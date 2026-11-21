"""Composition root for the production pipeline."""

from __future__ import annotations

from biomed_video_fc.claims.decomposer import DecomposingClaimExtractor
from biomed_video_fc.claims.extractor import LocalBiomedicalClaimExtractor
from biomed_video_fc.config import Settings
from biomed_video_fc.exceptions import ConfigurationError
from biomed_video_fc.media.processors import FFmpegAudioExtractor, OpenCVFrameExtractor
from biomed_video_fc.ocr.readers import EasyOCRReader, PaddleOCRReader
from biomed_video_fc.pipeline import BiomedicalFactCheckPipeline
from biomed_video_fc.reasoning.local_classifier import HealthFCClassifierReasoner
from biomed_video_fc.retrieval.pubmed_api import PubMedAPIRetriever
from biomed_video_fc.speech.whisper_transcriber import WhisperTranscriber
from biomed_video_fc.text.merger import TimeAlignedTranscriptMerger


def build_pipeline(settings: Settings) -> BiomedicalFactCheckPipeline:
    ocr_kwargs = {
        "language": settings.ocr.language,
        "min_confidence": settings.ocr.min_confidence,
        "frame_duration": settings.media.frame_interval_seconds,
    }
    if settings.ocr.engine == "paddle":
        ocr_reader = PaddleOCRReader(**ocr_kwargs)
    elif settings.ocr.engine == "easy":
        ocr_reader = EasyOCRReader(**ocr_kwargs)
    else:
        raise ConfigurationError("ocr.engine must be 'paddle' or 'easy'")

    return BiomedicalFactCheckPipeline(
        audio_extractor=FFmpegAudioExtractor(
            settings.media.ffmpeg_binary, settings.media.audio_sample_rate
        ),
        frame_extractor=OpenCVFrameExtractor(
            settings.media.frame_interval_seconds, settings.media.max_video_seconds
        ),
        transcriber=WhisperTranscriber(
            settings.speech.model, settings.speech.device, settings.speech.language
        ),
        ocr_reader=ocr_reader,
        transcript_merger=TimeAlignedTranscriptMerger(
            settings.ocr.dedup_similarity,
            max_gap_seconds=settings.media.frame_interval_seconds * 1.5,
        ),
        claim_extractor=DecomposingClaimExtractor(
            LocalBiomedicalClaimExtractor(
                model_name=settings.claims.model,
                max_claims=settings.claims.max_claims,
                min_confidence=settings.claims.min_confidence,
                device=settings.claims.device,
            ),
            max_atomic_claims=settings.claims.max_atomic_claims,
        ),
        evidence_retriever=PubMedAPIRetriever(
            email=settings.retrieval.email,
            api_key=settings.retrieval.api_key,
            tool=settings.retrieval.tool,
            max_results=settings.retrieval.max_results,
            candidate_results=settings.retrieval.candidate_results,
            max_queries=settings.retrieval.max_queries,
            timeout_seconds=settings.retrieval.timeout_seconds,
            retries=settings.retrieval.retries,
            semantic_rerank=settings.retrieval.semantic_rerank,
            embedding_model=settings.retrieval.embedding_model,
            minimum_relevance=settings.retrieval.minimum_relevance,
            minimum_term_coverage=settings.retrieval.minimum_term_coverage,
        ),
        reasoner=HealthFCClassifierReasoner(
            settings.classifier.model_dir, settings.classifier.max_length
        ),
    )
