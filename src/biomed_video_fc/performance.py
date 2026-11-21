"""Safe runtime profiles that trade search breadth for analysis speed."""

from __future__ import annotations

from typing import Literal

from biomed_video_fc.config import Settings

PerformanceProfile = Literal["fast", "balanced", "thorough"]


def apply_performance_profile(settings: Settings, profile: PerformanceProfile) -> Settings:
    """Return an independent settings copy tuned for the selected runtime profile.

    All profiles retain the entity-consistency and claim-constraint safety checks. Faster
    profiles sample fewer frames and limit work rather than bypassing those checks.
    """

    tuned = settings.model_copy(deep=True)
    if profile == "thorough":
        return tuned

    if profile == "balanced":
        tuned.media.frame_interval_seconds = max(tuned.media.frame_interval_seconds, 2.0)
        tuned.claims.max_claims = min(tuned.claims.max_claims, 5)
        tuned.claims.max_atomic_claims = min(tuned.claims.max_atomic_claims, 10)
        tuned.retrieval.max_results = min(tuned.retrieval.max_results, 3)
        tuned.retrieval.candidate_results = min(tuned.retrieval.candidate_results, 30)
        tuned.retrieval.max_queries = min(tuned.retrieval.max_queries, 2)
    elif profile == "fast":
        tuned.media.frame_interval_seconds = max(tuned.media.frame_interval_seconds, 3.0)
        tuned.claims.max_claims = min(tuned.claims.max_claims, 3)
        tuned.claims.max_atomic_claims = min(tuned.claims.max_atomic_claims, 6)
        tuned.retrieval.max_results = min(tuned.retrieval.max_results, 2)
        tuned.retrieval.candidate_results = min(tuned.retrieval.candidate_results, 18)
        tuned.retrieval.max_queries = 1
    else:
        raise ValueError(f"Unknown performance profile: {profile}")

    tuned.retrieval.candidate_results = max(
        tuned.retrieval.candidate_results, tuned.retrieval.max_results
    )
    return tuned
