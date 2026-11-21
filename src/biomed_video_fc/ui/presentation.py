"""Pure presentation helpers for concise, auditable result summaries."""

from __future__ import annotations

import re

from biomed_video_fc.domain.models import ClaimResult, Evidence, EvidenceAssessment, Verdict


def compact_text(text: str, limit: int = 220) -> str:
    """Collapse whitespace and shorten text at a readable boundary."""

    normalized = re.sub(r"\s+", " ", text).strip()
    if len(normalized) <= limit:
        return normalized
    sentence_end = normalized.rfind(". ", 0, limit)
    if sentence_end >= limit // 2:
        return normalized[: sentence_end + 1]
    word_end = normalized.rfind(" ", 0, limit - 1)
    cutoff = word_end if word_end >= limit // 2 else limit - 1
    return normalized[:cutoff].rstrip(" ,;:-") + "…"


def prioritized_claim_results(
    results: list[ClaimResult],
) -> list[tuple[int, ClaimResult]]:
    """Place conclusive truth decisions before unresolved claims."""

    verdict_priority = {Verdict.TRUE: 0, Verdict.FALSE: 1, Verdict.NEI: 2}
    numbered = list(enumerate(results, start=1))
    return sorted(
        numbered,
        key=lambda pair: (
            verdict_priority[pair[1].verification.verdict],
            pair[0],
        ),
    )


def best_evidence(
    result: ClaimResult,
) -> tuple[Evidence | None, EvidenceAssessment | None]:
    """Return the highest-ranked paper and its assessment, when available."""

    if not result.evidence:
        return None, None
    evidence = result.evidence[0]
    assessments = {
        assessment.pmid: assessment for assessment in result.verification.evidence_assessments
    }
    return evidence, assessments.get(evidence.pmid)


def evidence_summary(result: ClaimResult) -> str:
    """Create one short evidence explanation for the summary table."""

    evidence, assessment = best_evidence(result)
    if evidence is None:
        return "No matching PubMed evidence"
    if assessment and assessment.constraint_warnings:
        return "Related evidence; " + compact_text(assessment.constraint_warnings[0], 130)
    if assessment:
        return f"{assessment.stance.value}: {compact_text(evidence.title, 100)}"
    return compact_text(evidence.title, 120)


def build_summary_rows(results: list[ClaimResult]) -> list[dict[str, object]]:
    """Build compact rows while preserving each claim's original number."""

    rows: list[dict[str, object]] = []
    for number, item in prioritized_claim_results(results):
        rows.append(
            {
                "#": number,
                "Claim": item.claim.text,
                "Verdict": item.verification.verdict.value,
                "Confidence": (
                    ""
                    if item.verification.verdict is Verdict.NEI
                    else f"{item.verification.confidence:.0%}"
                ),
                "Best evidence": evidence_summary(item),
            }
        )
    return rows
