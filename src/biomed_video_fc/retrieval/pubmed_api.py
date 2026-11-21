"""Live PubMed evidence retrieval through the official NCBI E-utilities API."""

from __future__ import annotations

import json
import logging
import math
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import OrderedDict
from collections.abc import Callable
from typing import Any

from biomed_video_fc.domain.models import Claim, Evidence
from biomed_video_fc.exceptions import ConfigurationError, ExternalServiceError
from biomed_video_fc.retrieval.entity_validation import (
    ENTITY_ALIASES,
    audit_entity_consistency,
    entity_labels,
    find_biomedical_entities,
    normalize_biomedical_entities,
)
from biomed_video_fc.retrieval.pubmed_xml import parse_pubmed_article

logger = logging.getLogger(__name__)

EUTILS_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
RequestFunction = Callable[[str, float], bytes]

_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "because",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "was",
    "were",
    "when",
    "with",
    "your",
}
_NEGATIONS = {"cannot", "never", "no", "none", "not", "without"}

_SYNONYM_GROUPS: tuple[tuple[str, ...], ...] = (
    ("bleeding gums", "gingival bleeding", "gingivitis", "swollen gums"),
    ("dry skin", "xerosis"),
    ("easy bruising", "bruising", "ecchymosis", "purpura"),
    ("common colds", "more colds", "frequent colds", "respiratory infections"),
)
_QUERY_NOISE = {
    "first",
    "early",
    "common",
    "sign",
    "symptom",
    "getting",
    "more",
    "not",
}


def _term(token: str) -> str:
    if len(token) > 5 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith("es"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _content_terms(text: str) -> set[str]:
    normalized = normalize_biomedical_entities(text).casefold()
    normalized = normalized.replace("scurvy", "vitamin c deficiency")
    normalized = normalized.replace("gingival bleeding", "bleeding gums")
    normalized = normalized.replace("gingivitis", "bleeding gums")
    normalized = normalized.replace("xerosis", "dry skin")
    normalized = normalized.replace("ecchymosis", "easy bruising")
    normalized = normalized.replace("purpura", "easy bruising")
    normalized = re.sub(
        r"\bcollagen\s+(?:biosynthesis|production)\b", "collagen formation", normalized
    )
    return {
        _term(token)
        for token in re.findall(r"[a-z0-9]+", normalized)
        if token not in _STOP_WORDS and (len(token) > 1 or token == "c")
    }


def _claim_coverage(claim: str, document: str) -> float:
    claim_terms = _content_terms(claim)
    if not claim_terms:
        return 0.0
    document_terms = _content_terms(document)
    return len(claim_terms & document_terms) / len(claim_terms)


def _negation_is_addressed(claim: str, document: str) -> bool:
    """Deprecated compatibility helper; negation must be assessed as stance, not relevance."""
    claim_terms = _content_terms(claim)
    if not (claim_terms & _NEGATIONS):
        return True
    return bool(_content_terms(document) & _NEGATIONS)


def _pubmed_group(variants: tuple[str, ...]) -> str:
    return "(" + " OR ".join(f'"{item}"[Title/Abstract]' for item in variants) + ")"


def build_pubmed_queries(claim: str, max_queries: int = 3) -> list[str]:
    """Create concise alternative PubMed queries from one atomic claim."""

    normalized = " ".join(re.findall(r"[a-z0-9]+", claim.casefold()))
    matched = [ENTITY_ALIASES[entity] for entity in sorted(find_biomedical_entities(claim))]
    matched.extend(group for group in _SYNONYM_GROUPS if any(term in normalized for term in group))
    queries: list[str] = []
    if len(matched) >= 2:
        queries.append(" AND ".join(_pubmed_group(group) for group in matched))
    elif matched:
        covered = set().union(*(_content_terms(term) for term in matched[0]))
        remaining = [
            term
            for term in _content_terms(claim)
            if term not in covered and term not in _QUERY_NOISE and term not in _NEGATIONS
        ]
        query = _pubmed_group(matched[0])
        if remaining:
            query += (
                " AND ("
                + " OR ".join(f'"{term}"[Title/Abstract]' for term in sorted(remaining))
                + ")"
            )
        queries.append(query)

    clean_claim = re.sub(r"[^A-Za-z0-9% ]+", " ", claim)
    clean_claim = re.sub(r"\s+", " ", clean_claim).strip()
    if clean_claim:
        queries.append(clean_claim)
    broad_terms = [
        term
        for term in sorted(_content_terms(claim))
        if term not in _QUERY_NOISE and term not in _NEGATIONS
    ]
    if broad_terms:
        queries.append(" OR ".join(f'"{term}"[Title/Abstract]' for term in broad_terms))

    unique: list[str] = []
    for query in queries:
        if query and query not in unique:
            unique.append(query)
    return unique[:max_queries]


class PubMedAPIRetriever:
    """Retrieve PubMed abstracts and optionally rerank them with a local encoder."""

    def __init__(
        self,
        email: str,
        api_key: str = "",
        tool: str = "biomedical-video-fact-checker",
        max_results: int = 3,
        candidate_results: int = 50,
        max_queries: int = 3,
        timeout_seconds: float = 20.0,
        retries: int = 3,
        semantic_rerank: bool = True,
        embedding_model: str = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
        minimum_relevance: float = 0.30,
        minimum_term_coverage: float = 0.20,
        request_function: RequestFunction | None = None,
    ) -> None:
        if not email.strip():
            raise ConfigurationError(
                "PubMed requires a contact email. Set BIOFC_NCBI_EMAIL before analysis."
            )
        if max_results > candidate_results:
            raise ConfigurationError("retrieval.max_results cannot exceed candidate_results")
        self.email = email.strip()
        self.api_key = api_key.strip()
        self.tool = tool.strip() or "biomedical-video-fact-checker"
        self.max_results = max_results
        self.candidate_results = candidate_results
        self.max_queries = max_queries
        self.timeout_seconds = timeout_seconds
        self.retries = retries
        self.semantic_rerank = semantic_rerank
        self.embedding_model = embedding_model
        self.minimum_relevance = minimum_relevance
        self.minimum_term_coverage = minimum_term_coverage
        self._request_function = request_function or self._default_request
        self._request_lock = threading.Lock()
        self._last_request_at = 0.0
        self._encoder: Any = None
        self._result_cache: OrderedDict[str, tuple[Evidence, ...]] = OrderedDict()
        self._cache_lock = threading.Lock()
        self._cache_size = 128

    def retrieve(self, claim: Claim) -> list[Evidence]:
        cache_key = " ".join(claim.text.casefold().split())
        with self._cache_lock:
            cached = self._result_cache.get(cache_key)
            if cached is not None:
                self._result_cache.move_to_end(cache_key)
                logger.info("Using cached PubMed evidence for claim %s", claim.id)
                return [item.model_copy(deep=True) for item in cached]

        queries = build_pubmed_queries(claim.text, max_queries=self.max_queries)
        if not self.semantic_rerank:
            queries = [claim.text]
        pmids: list[str] = []
        query_by_pmid: dict[str, list[str]] = {}
        per_query = max(5, math.ceil(self.candidate_results / max(len(queries), 1)))
        for query in queries:
            for pmid in self._search(query, retmax=per_query):
                query_by_pmid.setdefault(pmid, []).append(query)
                if pmid not in pmids and len(pmids) < self.candidate_results:
                    pmids.append(pmid)
        if not pmids:
            logger.info("PubMed returned no records for claim %s", claim.id)
            self._cache_result(cache_key, [])
            return []

        records = self._fetch(pmids)
        if not records:
            logger.info("PubMed records for claim %s contained no usable abstracts", claim.id)
            self._cache_result(cache_key, [])
            return []

        order = {pmid: position for position, pmid in enumerate(pmids)}
        records.sort(key=lambda item: order.get(item["pmid"], len(order)))
        evidence = [
            self._to_evidence(item).model_copy(
                update={"retrieval_queries": query_by_pmid.get(item["pmid"], [])}
            )
            for item in records
        ]
        if self.semantic_rerank:
            evidence = self._rerank(claim.text, evidence)
        candidate_count = len(evidence)
        scored: list[Evidence] = []
        for item in evidence:
            document = f"{item.title}. {item.abstract}"
            coverage = _claim_coverage(claim.text, document)
            entity_audit = audit_entity_consistency(claim.text, document)
            semantic_status = (
                f"semantic score {item.relevance_score:.3f}"
                if item.relevance_score is not None
                else "PubMed relevance order"
            )
            entity_status = (
                "protected entities matched"
                if entity_audit.passes
                else "missing protected entities: " + ", ".join(entity_labels(entity_audit.missing))
            )
            scored.append(
                item.model_copy(
                    update={
                        "term_coverage": coverage,
                        "entity_match": entity_audit.passes,
                        "matched_entities": entity_labels(entity_audit.matched),
                        "missing_entities": entity_labels(entity_audit.missing),
                        "additional_entities": entity_labels(entity_audit.additional),
                        "selection_reason": (
                            f"{semantic_status}; biomedical-term coverage {coverage:.3f}; "
                            f"{entity_status}"
                        ),
                    }
                )
            )
        evidence = [item for item in scored if item.entity_match is not False]
        if self.semantic_rerank:
            evidence = [item for item in evidence if self._addresses_claim(claim.text, item)]
        if candidate_count and not evidence:
            logger.info(
                "Discarded %d PubMed abstracts that did not adequately address claim %s",
                candidate_count,
                claim.id,
            )
        selected = evidence[: self.max_results]
        self._cache_result(cache_key, selected)
        logger.info("PubMed returned %d abstracts for claim %s", len(selected), claim.id)
        return [item.model_copy(deep=True) for item in selected]

    def _cache_result(self, key: str, evidence: list[Evidence]) -> None:
        """Keep recent claim results in memory for repeated videos or duplicate claims."""

        with self._cache_lock:
            self._result_cache[key] = tuple(item.model_copy(deep=True) for item in evidence)
            self._result_cache.move_to_end(key)
            while len(self._result_cache) > self._cache_size:
                self._result_cache.popitem(last=False)

    def _search(self, query: str, retmax: int | None = None) -> list[str]:
        payload = self._request(
            "esearch.fcgi",
            {
                "db": "pubmed",
                "term": query,
                "retmode": "json",
                "retmax": str(retmax or self.candidate_results),
                "sort": "relevance",
            },
        )
        try:
            result = json.loads(payload.decode("utf-8"))["esearchresult"]
            return [str(pmid) for pmid in result.get("idlist", []) if str(pmid).isdigit()]
        except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ExternalServiceError("PubMed returned an invalid search response") from exc

    def _fetch(self, pmids: list[str]) -> list[dict[str, str]]:
        payload = self._request(
            "efetch.fcgi",
            {
                "db": "pubmed",
                "id": ",".join(pmids),
                "retmode": "xml",
            },
        )
        try:
            root = ET.fromstring(payload)
        except ET.ParseError as exc:
            raise ExternalServiceError("PubMed returned invalid article XML") from exc

        records: list[dict[str, str]] = []
        for item in root.iter():
            if item.tag.rsplit("}", 1)[-1] != "PubmedArticle":
                continue
            record = parse_pubmed_article(item)
            if record is not None:
                records.append(record)
        return records

    def _request(self, endpoint: str, parameters: dict[str, str]) -> bytes:
        common = {"tool": self.tool, "email": self.email}
        if self.api_key:
            common["api_key"] = self.api_key
        url = f"{EUTILS_BASE_URL}/{endpoint}?{urllib.parse.urlencode(parameters | common)}"

        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            self._respect_rate_limit()
            try:
                return self._request_function(url, self.timeout_seconds)
            except urllib.error.HTTPError as exc:
                last_error = exc
                if exc.code != 429 and exc.code < 500:
                    raise ExternalServiceError(
                        f"PubMed rejected the request with HTTP {exc.code}"
                    ) from exc
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_error = exc
            if attempt < self.retries:
                time.sleep(min(2**attempt, 8))

        raise ExternalServiceError(
            "PubMed is temporarily unavailable after repeated attempts"
        ) from last_error

    def _respect_rate_limit(self) -> None:
        minimum_interval = 0.11 if self.api_key else 0.34
        with self._request_lock:
            elapsed = time.monotonic() - self._last_request_at
            if elapsed < minimum_interval:
                time.sleep(minimum_interval - elapsed)
            self._last_request_at = time.monotonic()

    def _default_request(self, url: str, timeout_seconds: float) -> bytes:
        request = urllib.request.Request(
            url,
            headers={"User-Agent": f"{self.tool}/0.5-dev ({self.email})"},
        )
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            return response.read()

    @staticmethod
    def _to_evidence(record: dict[str, str]) -> Evidence:
        pmid = record["pmid"]
        return Evidence(
            pmid=pmid,
            title=record["title"],
            abstract=record["abstract"],
            journal=record.get("journal") or None,
            publication_date=record.get("publication_date") or None,
            doi=record.get("doi") or None,
            url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            relevance_score=None,
        )

    def _addresses_claim(self, claim: str, evidence: Evidence) -> bool:
        document = f"{evidence.title}. {evidence.abstract}"
        entity_match = (
            evidence.entity_match
            if evidence.entity_match is not None
            else audit_entity_consistency(claim, document).passes
        )
        coverage = (
            evidence.term_coverage
            if evidence.term_coverage is not None
            else _claim_coverage(claim, document)
        )
        return (
            entity_match
            and (
                evidence.relevance_score is None
                or evidence.relevance_score >= self.minimum_relevance
            )
            and coverage >= self.minimum_term_coverage
        )

    def _rerank(self, query: str, evidence: list[Evidence]) -> list[Evidence]:
        try:
            import numpy as np

            if self._encoder is None:
                from sentence_transformers import SentenceTransformer

                logger.info("Loading local PubMed reranker %s", self.embedding_model)
                self._encoder = SentenceTransformer(self.embedding_model)
            query_vector = self._encoder.encode(
                [query], convert_to_numpy=True, normalize_embeddings=True
            )[0]
            document_vectors = self._encoder.encode(
                [f"{item.title}. {item.abstract}" for item in evidence],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            similarities = np.dot(document_vectors, query_vector)
            ranked: list[Evidence] = []
            for index in np.argsort(-similarities).tolist():
                score = max(0.0, min(1.0, float(similarities[index])))
                ranked.append(evidence[index].model_copy(update={"relevance_score": score}))
            return ranked
        except (ImportError, RuntimeError, ValueError, TypeError) as exc:
            logger.warning(
                "Local semantic reranking failed; preserving PubMed relevance order: %s",
                type(exc).__name__,
            )
            return evidence
