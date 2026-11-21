"""Deterministic biomedical entity checks for retrieval precision.

Semantic similarity alone can rank an abstract about the wrong micronutrient.
This module protects explicit nutrient/mineral identities while keeping the
retrieval stage independent from truth/negation decisions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

ENTITY_ALIASES: dict[str, tuple[str, ...]] = {
    "vitamin_a": ("vitamin a", "retinol", "retinoic acid", "beta carotene"),
    "vitamin_b1": ("vitamin b1", "thiamine", "thiamin"),
    "vitamin_b2": ("vitamin b2", "riboflavin"),
    "vitamin_b3": ("vitamin b3", "niacin", "nicotinic acid"),
    "vitamin_b5": ("vitamin b5", "pantothenic acid"),
    "vitamin_b6": ("vitamin b6", "pyridoxine", "pyridoxal"),
    "vitamin_b7": ("vitamin b7", "biotin"),
    "vitamin_b9": ("vitamin b9", "folate", "folic acid"),
    "vitamin_b12": ("vitamin b12", "cobalamin", "cyanocobalamin"),
    "vitamin_c": ("vitamin c", "ascorbic acid", "ascorbate", "scurvy"),
    "vitamin_d": ("vitamin d", "cholecalciferol", "ergocalciferol", "calcitriol"),
    "vitamin_e": ("vitamin e", "tocopherol", "tocotrienol"),
    "vitamin_k": ("vitamin k", "phylloquinone", "menaquinone"),
    "calcium": ("calcium",),
    "iron": ("iron",),
    "magnesium": ("magnesium",),
    "zinc": ("zinc",),
    "selenium": ("selenium",),
    "iodine": ("iodine", "iodide"),
}

ENTITY_LABELS = {
    name: name.replace("_", " ").title().replace("Vitamin B", "Vitamin B")
    for name in ENTITY_ALIASES
}


def _alias_pattern(alias: str) -> re.Pattern[str]:
    escaped = re.escape(alias).replace(r"\ ", r"[\s-]+")
    return re.compile(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])", re.IGNORECASE)


_ENTITY_PATTERNS = {
    entity: tuple(_alias_pattern(alias) for alias in aliases)
    for entity, aliases in ENTITY_ALIASES.items()
}


@dataclass(frozen=True)
class EntityAudit:
    required: tuple[str, ...]
    matched: tuple[str, ...]
    missing: tuple[str, ...]
    additional: tuple[str, ...]

    @property
    def passes(self) -> bool:
        return not self.missing


def find_biomedical_entities(text: str) -> set[str]:
    """Return protected nutrient/mineral concepts explicitly present in text."""

    return {
        entity
        for entity, patterns in _ENTITY_PATTERNS.items()
        if any(pattern.search(text) for pattern in patterns)
    }


def normalize_biomedical_entities(text: str) -> str:
    """Replace protected aliases with stable canonical terms for lexical scoring."""

    normalized = text
    for entity, patterns in _ENTITY_PATTERNS.items():
        canonical = entity.replace("_", " ")
        for pattern in patterns:
            normalized = pattern.sub(canonical, normalized)
    return normalized


def audit_entity_consistency(claim: str, document: str) -> EntityAudit:
    """Require every protected claim entity to occur in the candidate evidence."""

    required = find_biomedical_entities(claim)
    present = find_biomedical_entities(document)
    matched = required & present
    return EntityAudit(
        required=tuple(sorted(required)),
        matched=tuple(sorted(matched)),
        missing=tuple(sorted(required - present)),
        additional=tuple(sorted(present - required)),
    )


def entity_labels(entities: tuple[str, ...] | set[str]) -> list[str]:
    return [ENTITY_LABELS.get(entity, entity.replace("_", " ")) for entity in sorted(entities)]


def replace_first_entity(text: str, entity: str, replacement: str) -> str | None:
    """Replace the first explicit alias of one protected entity."""

    for pattern in _ENTITY_PATTERNS.get(entity, ()):
        if pattern.search(text):
            return pattern.sub(replacement, text, count=1)
    return None
