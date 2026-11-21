"""Safety checks for claim details that generic NLI models often overlook."""

from __future__ import annotations

import re
from dataclasses import dataclass

from biomed_video_fc.domain.models import EvidenceStance

_NEGATION = re.compile(
    r"\b(?:no|not|never|neither|without|cannot|can't|doesn't|does\s+not|"
    r"isn't|is\s+not|aren't|are\s+not|unlikely|fails?\s+to)\b",
    re.IGNORECASE,
)

_QUALIFIER_GROUPS: dict[str, re.Pattern[str]] = {
    "first/earliest": re.compile(
        r"\b(?:first|earliest|initial|initially|presenting|presentation)\b", re.IGNORECASE
    ),
    "early": re.compile(r"\b(?:early|earliest|initial|initially|first)\b", re.IGNORECASE),
    "only/exclusive": re.compile(
        r"\b(?:only|sole|solely|exclusive|exclusively|unique)\b", re.IGNORECASE
    ),
    "universal": re.compile(r"\b(?:all|every|always|never|none)\b", re.IGNORECASE),
    "maximum/minimum": re.compile(
        r"\b(?:most|least|highest|lowest|maximum|minimum)\b", re.IGNORECASE
    ),
}

_CLAIM_QUALIFIER_TRIGGERS: dict[str, re.Pattern[str]] = {
    "first/earliest": re.compile(r"\b(?:first|earliest)\b", re.IGNORECASE),
    "early": re.compile(r"\bearly\b", re.IGNORECASE),
    "only/exclusive": re.compile(r"\b(?:only|sole|solely|exclusive|exclusively)\b", re.IGNORECASE),
    "universal": re.compile(r"\b(?:all|every|always|never|none)\b", re.IGNORECASE),
    "maximum/minimum": re.compile(
        r"\b(?:most|least|highest|lowest|maximum|minimum)\b", re.IGNORECASE
    ),
}

_QUANTITY = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:%|mg|g|kg|mcg|µg|ug|ml|l|iu|"
    r"minutes?|hours?|days?|weeks?|months?|years?)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ConstraintAudit:
    final_stance: EvidenceStance
    warnings: tuple[str, ...]


def _quantities(text: str) -> set[str]:
    return {
        re.sub(r"\s+", "", item.casefold()).replace("µ", "u") for item in _QUANTITY.findall(text)
    }


def audit_stance_constraints(
    claim: str,
    passage: str,
    model_stance: EvidenceStance,
    missing_entities: list[str] | None = None,
) -> ConstraintAudit:
    """Downgrade a stance when the passage does not address material claim details.

    The evidence remains visible as related information. Only its ability to
    support or contradict the *complete* claim is restricted.
    """

    if model_stance is EvidenceStance.NEUTRAL:
        return ConstraintAudit(final_stance=model_stance, warnings=())

    warnings: list[str] = []
    if missing_entities:
        warnings.append(
            "Evidence does not mention required entities: " + ", ".join(missing_entities)
        )

    for name, trigger in _CLAIM_QUALIFIER_TRIGGERS.items():
        if trigger.search(claim) and not _QUALIFIER_GROUPS[name].search(passage):
            warnings.append(f"Evidence does not address the claim qualifier '{name}'.")

    claim_quantities = _quantities(claim)
    if claim_quantities:
        passage_quantities = _quantities(passage)
        absent = sorted(claim_quantities - passage_quantities)
        if absent:
            warnings.append("Evidence does not address claim quantity: " + ", ".join(absent))

    if (
        model_stance is EvidenceStance.SUPPORTS
        and _NEGATION.search(claim)
        and not _NEGATION.search(passage)
    ):
        warnings.append("Supporting evidence does not preserve the claim's negation.")

    return ConstraintAudit(
        final_stance=EvidenceStance.NEUTRAL if warnings else model_stance,
        warnings=tuple(warnings),
    )
