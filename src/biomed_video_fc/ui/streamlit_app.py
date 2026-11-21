"""Streamlit dashboard for end-to-end video analysis."""

from __future__ import annotations

import contextlib
import hashlib
import os
import tempfile
from pathlib import Path

import streamlit as st

from biomed_video_fc.config import Settings, load_settings
from biomed_video_fc.domain.models import AnalysisResult, Verdict
from biomed_video_fc.exceptions import BioFactCheckError
from biomed_video_fc.factory import build_pipeline
from biomed_video_fc.logging_config import configure_logging
from biomed_video_fc.performance import PerformanceProfile, apply_performance_profile
from biomed_video_fc.pipeline import BiomedicalFactCheckPipeline
from biomed_video_fc.ui.presentation import (
    best_evidence,
    build_summary_rows,
    compact_text,
    prioritized_claim_results,
)


def _pipeline_signature(settings: Settings) -> str:
    """Identify the active configuration without storing credentials in plain text."""

    return hashlib.sha256(settings.model_dump_json().encode("utf-8")).hexdigest()


def _get_session_pipeline(settings: Settings) -> BiomedicalFactCheckPipeline:
    """Reuse loaded local models for later analyses in this browser session."""

    signature = _pipeline_signature(settings)
    if st.session_state.get("pipeline_signature") != signature:
        st.session_state["analysis_pipeline"] = build_pipeline(settings)
        st.session_state["pipeline_signature"] = signature
    return st.session_state["analysis_pipeline"]


def _render_transcripts(result: AnalysisResult, expanded: bool) -> None:
    with st.expander("Transcripts and extracted on-screen text", expanded=expanded):
        speech_tab, ocr_tab, combined_tab = st.tabs(
            ["Speech transcript", "On-screen text (OCR)", "Combined transcript"]
        )
        with speech_tab:
            st.text_area(
                "Speech", result.transcript.speech_text or "No speech detected.", height=220
            )
        with ocr_tab:
            st.text_area(
                "OCR",
                (
                    result.transcript.ocr_text or "No on-screen text detected."
                    if result.ocr_enabled
                    else "OCR was disabled for this analysis."
                ),
                height=220,
            )
        with combined_tab:
            st.text_area(
                "Time-aligned combined transcript", result.transcript.combined_text, height=300
            )


def _render_detailed_claims(result: AnalysisResult) -> None:
    for number, item in enumerate(result.claim_results, start=1):
        verdict = item.verification.verdict
        with st.expander(f"Claim {number}: {item.claim.text}", expanded=False):
            claim_time = item.claim.start_seconds if item.claim.start_seconds is not None else "—"
            if verdict is Verdict.NEI:
                left, right = st.columns([1, 2])
                left.metric("Prediction", verdict.value)
                right.markdown(f"**Source:** {item.claim.source}  \n**Time:** {claim_time}s")
            else:
                left, middle, right = st.columns([1, 1, 2])
                left.metric("Prediction", verdict.value)
                middle.metric("Confidence", f"{item.verification.confidence:.0%}")
                right.markdown(f"**Source:** {item.claim.source}  \n**Time:** {claim_time}s")
            if item.claim.parent_text and item.claim.parent_text != item.claim.text:
                st.caption(f"Atomic claim from: {item.claim.parent_text}")

            st.markdown("#### Reasoning")
            st.write(item.verification.reasoning)
            if item.verification.limitations:
                st.caption("Limitations: " + " • ".join(item.verification.limitations))
            if not item.evidence:
                st.info(
                    "Evidence is not available in the retrieved PubMed resources to support "
                    "or contradict this claim."
                )
                continue

            st.markdown("#### Retrieved PubMed evidence")
            assessments = {
                assessment.pmid: assessment for assessment in item.verification.evidence_assessments
            }
            for evidence_number, evidence in enumerate(item.evidence, start=1):
                assessment = assessments.get(evidence.pmid)
                suffix = f" — {assessment.stance.value}" if assessment else ""
                with st.expander(f"Evidence {evidence_number}: {evidence.title}{suffix}"):
                    if assessment:
                        st.markdown(f"**Final evidence stance:** {assessment.stance.value}")
                        if verdict is not Verdict.NEI:
                            st.markdown(f"**Raw model confidence:** {assessment.confidence:.0%}")
                        if (
                            assessment.model_stance is not None
                            and assessment.model_stance is not assessment.stance
                        ):
                            st.caption(
                                f"Raw model stance {assessment.model_stance.value} was changed to "
                                f"{assessment.stance.value} by claim-detail checks."
                            )
                        for warning in assessment.constraint_warnings:
                            st.warning(warning)
                        st.markdown("**Passage actually assessed by BiomedBERT**")
                        st.write(assessment.passage)
                    st.write(evidence.abstract)
                    metadata = " · ".join(
                        value
                        for value in [
                            evidence.journal,
                            evidence.publication_date,
                            f"PMID {evidence.pmid}",
                            (
                                f"Semantic similarity {evidence.relevance_score:.0%}"
                                if evidence.relevance_score is not None
                                else None
                            ),
                            (
                                f"Claim-term coverage {evidence.term_coverage:.0%}"
                                if evidence.term_coverage is not None
                                else None
                            ),
                            (
                                "Matched entities: " + ", ".join(evidence.matched_entities)
                                if evidence.matched_entities
                                else None
                            ),
                            (
                                "Other entities in paper: "
                                + ", ".join(evidence.additional_entities)
                                if evidence.additional_entities
                                else None
                            ),
                        ]
                        if value
                    )
                    st.caption(metadata)
                    if evidence.selection_reason:
                        st.caption(f"Selection: {evidence.selection_reason}")
                    st.link_button("Open in PubMed", str(evidence.url))


def _render_summary_claims(result: AnalysisResult) -> None:
    st.dataframe(
        build_summary_rows(result.claim_results),
        use_container_width=True,
        hide_index=True,
        column_config={
            "#": st.column_config.NumberColumn(width="small"),
            "Claim": st.column_config.TextColumn(width="large"),
            "Verdict": st.column_config.TextColumn(width="medium"),
            "Confidence": st.column_config.TextColumn(width="small"),
            "Best evidence": st.column_config.TextColumn(width="large"),
        },
    )
    st.caption(
        "TRUE and FALSE results are listed first; NOT ENOUGH INFORMATION follows. "
        "Claim numbers still match their original video order."
    )
    st.markdown("#### Evidence highlights")
    for number, item in prioritized_claim_results(result.claim_results):
        verdict = item.verification.verdict
        label = f"Claim {number} · {verdict.value}: {compact_text(item.claim.text, 110)}"
        with st.expander(label, expanded=False):
            if verdict is Verdict.NEI:
                st.metric("Prediction", verdict.value)
            else:
                left, middle = st.columns(2)
                left.metric("Prediction", verdict.value)
                middle.metric("Confidence", f"{item.verification.confidence:.0%}")
            if item.claim.parent_text and item.claim.parent_text != item.claim.text:
                st.caption(f"Atomic claim from: {compact_text(item.claim.parent_text, 220)}")

            st.markdown("**Why this verdict**")
            st.write(compact_text(item.verification.reasoning, 320))

            evidence, assessment = best_evidence(item)
            if evidence is None:
                st.info("No matching PubMed evidence was retrieved for this claim.")
                continue

            st.markdown("**Best retrieved evidence**")
            if assessment:
                st.caption(f"{assessment.stance.value} · PMID {evidence.pmid} · {evidence.title}")
                if assessment.constraint_warnings:
                    st.warning(compact_text(assessment.constraint_warnings[0], 260))
                st.write(assessment.passage)
            else:
                st.caption(f"PMID {evidence.pmid} · {evidence.title}")
            st.link_button("Open best paper in PubMed", str(evidence.url))
            if len(item.evidence) > 1:
                st.caption(
                    f"{len(item.evidence) - 1} additional paper(s) are available in Detailed view."
                )


def _render_processing_details(result: AnalysisResult) -> None:
    with st.expander("Processing details"):
        total_seconds = sum(timing.seconds for timing in result.timings)
        if result.timings:
            slowest = max(result.timings, key=lambda timing: timing.seconds)
            first, second = st.columns(2)
            first.metric("Total processing time", f"{total_seconds:.1f}s")
            second.metric("Slowest stage", f"{slowest.stage} ({slowest.seconds:.1f}s)")
        st.dataframe(
            [
                {"Stage": timing.stage, "Seconds": round(timing.seconds, 3)}
                for timing in result.timings
            ],
            use_container_width=True,
            hide_index=True,
        )
        st.download_button(
            "Download complete result (JSON)",
            result.model_dump_json(indent=2),
            file_name=f"{result.analysis_id}.json",
            mime="application/json",
        )


def _render_results(result: AnalysisResult, detailed: bool) -> None:
    if result.warnings:
        for warning in result.warnings:
            st.warning(warning)
    _render_transcripts(result, expanded=detailed)
    st.subheader(f"Verified atomic claims ({len(result.claim_results)})")
    if not result.claim_results:
        st.info("No checkable biomedical claims were detected in the video.")
    elif detailed:
        _render_detailed_claims(result)
    else:
        _render_summary_claims(result)
    _render_processing_details(result)


def main() -> None:
    settings = load_settings()
    configure_logging(settings.logging.level, settings.logging.json_logs)
    st.set_page_config(page_title=settings.app.name, page_icon="🧬", layout="wide")
    st.title("Biomedical Video Fact Checker")
    st.caption(
        "No OpenAI or Gemini: all inference and classification run locally. Current scientific "
        "abstracts are retrieved from PubMed through the official NCBI API. "
        "This is research decision support, not medical advice."
    )

    classifier_ready = all(
        (settings.classifier.model_dir / name).is_file()
        for name in ("config.json", "model.safetensors", "tokenizer.json")
    )
    with st.sidebar:
        st.subheader("System status")
        (st.success if classifier_ready else st.error)(
            "Verifier model ready" if classifier_ready else "Train or configure the verifier model"
        )
        ncbi_email = st.text_input(
            "NCBI contact email",
            value=settings.retrieval.email,
            help="NCBI requests a contact email from software using E-utilities.",
        ).strip()
        ncbi_api_key = st.text_input(
            "NCBI API key (optional)",
            value=settings.retrieval.api_key,
            type="password",
            help="Leave empty to use NCBI's standard request limit.",
        ).strip()
        pubmed_ready = bool(ncbi_email)
        (st.success if pubmed_ready else st.error)(
            "PubMed API configured" if pubmed_ready else "Enter your NCBI contact email"
        )
        if pubmed_ready:
            key_status = "API key enabled" if ncbi_api_key else "Standard rate limit"
            st.caption(f"NCBI E-utilities · {key_status}")
        st.caption("Only claim searches and PubMed abstracts are sent through NCBI.")
        st.divider()
        st.subheader("Performance")
        include_ocr = st.toggle(
            "Extract on-screen text (OCR)",
            value=True,
            help=(
                "Turn this off when every claim is spoken. Frame sampling and PaddleOCR will be "
                "skipped, which can substantially reduce analysis time."
            ),
        )
        if include_ocr:
            st.caption("Speech and visible text will both be checked.")
        else:
            st.caption("Speech only: frame extraction and PaddleOCR will be skipped.")
        profile_label = st.selectbox(
            "Analysis profile",
            ["Balanced", "Fast", "Thorough"],
            index=0,
            help=(
                "Balanced samples frames every 2 seconds and searches fewer PubMed candidates. "
                "Fast reduces those limits further. Thorough uses every configured setting."
            ),
        )
        profiles: dict[str, PerformanceProfile] = {
            "Balanced": "balanced",
            "Fast": "fast",
            "Thorough": "thorough",
        }
        profile = profiles[profile_label]
        if profile == "fast":
            st.caption("Fast: 3 s frames, up to 3 claims / 6 atomic claims, 1 PubMed query.")
        elif profile == "balanced":
            st.caption("Balanced: 2 s frames, up to 5 claims / 10 atomic claims, 2 queries.")
        else:
            st.caption("Thorough: full configured frame, claim, and PubMed search limits.")
        st.caption("Loaded local models are reused until these settings change.")
        st.divider()
        st.subheader("Display")
        display_mode = st.radio(
            "Result detail",
            ["Summary", "Detailed"],
            horizontal=True,
            help="Summary is concise; Detailed preserves every abstract and audit field.",
        )
        if display_mode == "Summary":
            st.caption("One result row and one best evidence passage per atomic claim.")
        else:
            st.caption("Full transcripts, PubMed abstracts, model stances, and audit metadata.")

    uploaded = st.file_uploader("Upload one MP4 health video", type=["mp4"])
    if uploaded is None:
        st.info("Upload an MP4 to begin. Speech and visible text will both be analyzed.")
        return
    if uploaded.size > settings.app.max_upload_mb * 1024 * 1024:
        st.error(f"The upload exceeds the {settings.app.max_upload_mb} MB configured limit.")
        return

    upload_bytes = uploaded.getvalue()
    upload_fingerprint = hashlib.sha256(upload_bytes).hexdigest()
    if st.session_state.get("upload_fingerprint") != upload_fingerprint:
        st.session_state.pop("analysis_result", None)
        st.session_state["upload_fingerprint"] = upload_fingerprint

    st.video(uploaded)
    if st.button("Analyze video", type="primary", use_container_width=True):
        temp_path: Path | None = None
        progress_bar = st.progress(0.0, text="Preparing analysis")

        def progress(message: str, fraction: float) -> None:
            progress_bar.progress(fraction, text=message)

        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as handle:
                handle.write(upload_bytes)
                temp_path = Path(handle.name)
            runtime_settings = settings.model_copy(deep=True)
            runtime_settings.retrieval.email = ncbi_email
            runtime_settings.retrieval.api_key = ncbi_api_key
            runtime_settings = apply_performance_profile(runtime_settings, profile)
            pipeline = _get_session_pipeline(runtime_settings)
            result = pipeline.analyze(temp_path, progress=progress, include_ocr=include_ocr)
            result = result.model_copy(update={"video_name": uploaded.name})
            st.session_state["analysis_result"] = result
        except BioFactCheckError as exc:
            st.session_state.pop("analysis_result", None)
            st.error(str(exc))
        except Exception as exc:
            st.session_state.pop("analysis_result", None)
            st.exception(exc)
        finally:
            progress_bar.empty()
            if temp_path is not None:
                with contextlib.suppress(OSError):
                    os.unlink(temp_path)

    result = st.session_state.get("analysis_result")
    if isinstance(result, AnalysisResult):
        _render_results(result, detailed=display_mode == "Detailed")


if __name__ == "__main__":
    main()
