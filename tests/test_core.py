from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
import urllib.parse
import wave
from pathlib import Path
from unittest.mock import patch

import numpy as np

from biomed_video_fc.claims.decomposer import decompose_text
from biomed_video_fc.claims.extractor import LocalBiomedicalClaimExtractor
from biomed_video_fc.config import load_settings
from biomed_video_fc.domain.models import (
    Claim,
    ClaimResult,
    Evidence,
    EvidenceAssessment,
    EvidenceStance,
    TextSource,
    TimedText,
    TranscriptBundle,
    Verdict,
    Verification,
)
from biomed_video_fc.evaluation import classification_metrics, normalize_healthfc_label
from biomed_video_fc.exceptions import ExternalServiceError
from biomed_video_fc.media.processors import FFmpegAudioExtractor
from biomed_video_fc.ocr.readers import PaddleOCRReader
from biomed_video_fc.performance import apply_performance_profile
from biomed_video_fc.pipeline import BiomedicalFactCheckPipeline
from biomed_video_fc.reasoning.constraints import audit_stance_constraints
from biomed_video_fc.reasoning.local_classifier import HealthFCClassifierReasoner
from biomed_video_fc.retrieval.entity_validation import audit_entity_consistency
from biomed_video_fc.retrieval.pubmed_api import (
    PubMedAPIRetriever,
    _claim_coverage,
    _negation_is_addressed,
    build_pubmed_queries,
)
from biomed_video_fc.retrieval.pubmed_xml import prepare_csv
from biomed_video_fc.speech.whisper_transcriber import WhisperTranscriber
from biomed_video_fc.text.merger import TimeAlignedTranscriptMerger
from biomed_video_fc.training.augmentation import build_hard_negatives
from biomed_video_fc.training.datasets import VerificationExample, load_bioasq_7b, load_scifact
from biomed_video_fc.training.healthfc import load_healthfc
from biomed_video_fc.training.multisource import build_splits
from biomed_video_fc.ui.presentation import build_summary_rows, compact_text


class CoreTests(unittest.TestCase):
    def test_config_environment_override(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "BIOFC_FRAME_INTERVAL_SECONDS": "2.5",
                "BIOFC_MAX_CLAIMS": "7",
                "BIOFC_MAX_ATOMIC_CLAIMS": "11",
                "BIOFC_NCBI_EMAIL": "researcher@example.com",
                "BIOFC_PUBMED_QUERIES": "2",
            },
            clear=False,
        ):
            settings = load_settings("missing.toml")
        self.assertEqual(settings.media.frame_interval_seconds, 2.5)
        self.assertEqual(settings.claims.max_claims, 7)
        self.assertEqual(settings.claims.max_atomic_claims, 11)
        self.assertEqual(settings.retrieval.email, "researcher@example.com")
        self.assertEqual(settings.retrieval.max_queries, 2)

    def test_performance_profiles_reduce_work_without_mutating_settings(self) -> None:
        settings = load_settings("config/default.toml")
        fast = apply_performance_profile(settings, "fast")
        balanced = apply_performance_profile(settings, "balanced")
        thorough = apply_performance_profile(settings, "thorough")

        self.assertEqual(settings.media.frame_interval_seconds, 1.0)
        self.assertEqual(fast.media.frame_interval_seconds, 3.0)
        self.assertEqual(fast.claims.max_claims, 3)
        self.assertEqual(fast.claims.max_atomic_claims, 6)
        self.assertEqual(fast.retrieval.max_queries, 1)
        self.assertEqual(fast.retrieval.candidate_results, 18)
        self.assertEqual(balanced.media.frame_interval_seconds, 2.0)
        self.assertEqual(balanced.retrieval.max_queries, 2)
        self.assertEqual(thorough, settings)

    def test_compact_summary_lists_true_false_before_nei_and_hides_nei_confidence(self) -> None:
        supported = ClaimResult(
            claim=Claim(text="Vitamin C is required for collagen formation."),
            evidence=[],
            verification=Verification(
                verdict=Verdict.TRUE,
                confidence=0.9,
                reasoning="The selected evidence directly supports the claim.",
            ),
        )
        evidence = Evidence(
            pmid="41374045",
            title="Clinical characteristics of pediatric vitamin C deficiency",
            abstract="Bleeding gums were a common clinical sign.",
            url="https://pubmed.ncbi.nlm.nih.gov/41374045/",
        )
        unresolved = ClaimResult(
            claim=Claim(text="Bleeding gums are the first sign of vitamin C deficiency."),
            evidence=[evidence],
            verification=Verification(
                verdict=Verdict.NEI,
                confidence=0.8,
                reasoning="The paper is related but does not establish temporal ordering.",
                review_required=True,
                evidence_assessments=[
                    EvidenceAssessment(
                        pmid="41374045",
                        stance=EvidenceStance.NEUTRAL,
                        confidence=0.91,
                        passage="Bleeding gums were a common clinical sign.",
                        constraint_warnings=[
                            "Evidence does not establish the first/earliest sign."
                        ],
                    )
                ],
            ),
        )
        refuted = ClaimResult(
            claim=Claim(text="Vitamin C deficiency never causes bruising."),
            evidence=[],
            verification=Verification(
                verdict=Verdict.FALSE,
                confidence=0.84,
                reasoning="The evidence contradicts the claim.",
            ),
        )

        rows = build_summary_rows([unresolved, refuted, supported])

        self.assertEqual([row["Verdict"] for row in rows], ["TRUE", "FALSE", Verdict.NEI.value])
        self.assertEqual(rows[0]["Confidence"], "90%")
        self.assertEqual(rows[1]["Confidence"], "84%")
        self.assertEqual(rows[2]["Confidence"], "")
        self.assertNotIn("Review", rows[2])
        self.assertIn("Related evidence", str(rows[2]["Best evidence"]))
        self.assertTrue(compact_text("word " * 100, 40).endswith("…"))

    def test_merger_deduplicates_persistent_ocr(self) -> None:
        speech = [
            TimedText(
                text="Vitamin D supports bone health.",
                start_seconds=0,
                end_seconds=2,
                source=TextSource.SPEECH,
            )
        ]
        ocr = [
            TimedText(
                text="Vitamin D prevents every fracture",
                start_seconds=0,
                end_seconds=1,
                source=TextSource.OCR,
                confidence=0.8,
            ),
            TimedText(
                text="Vitamin D prevents every fracture!",
                start_seconds=1,
                end_seconds=2,
                source=TextSource.OCR,
                confidence=0.9,
            ),
        ]
        result = TimeAlignedTranscriptMerger(0.85).merge(speech, ocr)
        self.assertEqual(len(result.ocr_segments), 1)
        self.assertEqual(result.ocr_segments[0].end_seconds, 2)
        self.assertIn("[OCR]", result.combined_text)

    def test_merger_preserves_negation_dosage_and_distant_occurrences(self) -> None:
        ocr = [
            TimedText(
                text="Drug A works at 10 mg", start_seconds=0, end_seconds=1, source=TextSource.OCR
            ),
            TimedText(
                text="Drug A does not work at 10 mg",
                start_seconds=1,
                end_seconds=2,
                source=TextSource.OCR,
            ),
            TimedText(
                text="Drug A works at 100 mg", start_seconds=2, end_seconds=3, source=TextSource.OCR
            ),
            TimedText(
                text="Drug A works at 10 mg",
                start_seconds=100,
                end_seconds=101,
                source=TextSource.OCR,
            ),
        ]
        result = TimeAlignedTranscriptMerger(0.75, max_gap_seconds=2).merge([], ocr)
        self.assertEqual(len(result.ocr_segments), 4)

    def test_reasoner_returns_nei_without_evidence(self) -> None:
        reasoner = HealthFCClassifierReasoner(Path("missing-model"))
        result = reasoner.verify(Claim(text="A medicine cures all cancers."), [])
        self.assertEqual(result.verdict, Verdict.NEI)
        self.assertEqual(result.confidence, 0.0)
        self.assertTrue(result.review_required)
        self.assertEqual(result.decision_source, "no_evidence")

    def test_evidence_aggregation_preserves_conflict(self) -> None:
        assessments = [
            EvidenceAssessment(
                pmid="1",
                stance=EvidenceStance.SUPPORTS,
                confidence=0.9,
                passage="Support passage.",
                relevance_score=0.9,
            ),
            EvidenceAssessment(
                pmid="2",
                stance=EvidenceStance.CONTRADICTS,
                confidence=0.9,
                passage="Contradict passage.",
                relevance_score=0.9,
            ),
        ]
        verdict, _confidence, review = HealthFCClassifierReasoner._aggregate(assessments)
        self.assertEqual(verdict, Verdict.NEI)
        self.assertTrue(review)

    def test_single_evidence_does_not_inflate_confidence_to_one(self) -> None:
        assessments = [
            EvidenceAssessment(
                pmid="1",
                stance=EvidenceStance.SUPPORTS,
                confidence=0.72,
                passage="Support passage.",
                relevance_score=0.8,
            )
        ]
        verdict, confidence, review = HealthFCClassifierReasoner._aggregate(assessments)
        self.assertEqual(verdict, Verdict.TRUE)
        self.assertAlmostEqual(confidence, 0.72)
        self.assertFalse(review)

    def test_unaddressed_first_qualifier_changes_support_to_neutral(self) -> None:
        audit = audit_stance_constraints(
            "Bleeding gums are the first sign of vitamin C deficiency.",
            "Common clinical signs included bleeding gums in vitamin C deficiency.",
            EvidenceStance.SUPPORTS,
        )
        self.assertEqual(audit.final_stance, EvidenceStance.NEUTRAL)
        self.assertTrue(any("first/earliest" in warning for warning in audit.warnings))

    def test_support_for_negative_claim_must_preserve_negation(self) -> None:
        audit = audit_stance_constraints(
            "Getting more colds is not the first sign of vitamin C deficiency.",
            "Respiratory infections were reported in vitamin C deficiency.",
            EvidenceStance.SUPPORTS,
        )
        self.assertEqual(audit.final_stance, EvidenceStance.NEUTRAL)
        self.assertTrue(any("negation" in warning for warning in audit.warnings))

    def test_reasoner_keeps_related_passage_but_neutralizes_missing_qualifier(self) -> None:
        reasoner = HealthFCClassifierReasoner(Path("unused-model"))
        evidence = Evidence(
            pmid="41374045",
            title="Pediatric vitamin C deficiency",
            abstract=(
                "Common clinical signs included limb pain and bleeding gums in children with "
                "vitamin C deficiency."
            ),
            url="https://pubmed.ncbi.nlm.nih.gov/41374045/",
            relevance_score=0.9,
            entity_match=True,
            matched_entities=["Vitamin C"],
        )
        with (
            patch.object(reasoner, "_load"),
            patch.object(
                reasoner,
                "_predict_pair",
                return_value=(Verdict.TRUE, 0.94, False),
            ),
        ):
            result = reasoner.verify(
                Claim(text="Bleeding gums are the first sign of vitamin C deficiency."),
                [evidence],
            )

        self.assertEqual(result.verdict, Verdict.NEI)
        self.assertEqual(result.evidence_assessments[0].model_stance, EvidenceStance.SUPPORTS)
        self.assertEqual(result.evidence_assessments[0].stance, EvidenceStance.NEUTRAL)
        self.assertIn("bleeding gums", result.evidence_assessments[0].passage.casefold())
        self.assertTrue(result.evidence_assessments[0].constraint_warnings)

    def test_local_claim_detector_merges_speech_and_ocr_duplicates(self) -> None:
        transcript = TranscriptBundle(
            combined_segments=[
                TimedText(
                    text="Vitamin D prevents bone fractures.",
                    start_seconds=0,
                    end_seconds=2,
                    source=TextSource.SPEECH,
                ),
                TimedText(
                    text="Vitamin D prevents bone fractures",
                    start_seconds=1,
                    end_seconds=3,
                    source=TextSource.OCR,
                ),
            ]
        )
        extractor = LocalBiomedicalClaimExtractor(
            "unused", score_function=lambda texts: [0.91 for _ in texts]
        )
        claims = extractor.extract(transcript)
        self.assertEqual(len(claims), 1)
        self.assertEqual(claims[0].source, "both")

    def test_claim_candidates_reconstruct_cross_segment_vitamin_c_claims(self) -> None:
        transcript = TranscriptBundle(
            combined_segments=[
                TimedText(
                    text=(
                        "The first sign of vitamin C deficiency is not getting more colds. "
                        "It's bleeding gums, dry skin,"
                    ),
                    start_seconds=0,
                    end_seconds=6.72,
                    source=TextSource.SPEECH,
                ),
                TimedText(
                    text=(
                        "and easy bruising. Vitamin C is essential for collagen formation, "
                        "the stuff that keeps your"
                    ),
                    start_seconds=6.72,
                    end_seconds=12.48,
                    source=TextSource.SPEECH,
                ),
                TimedText(
                    text=(
                        "skin firm and your blood vessels strong. When it's low, tiny "
                        "capillaries leak easily,"
                    ),
                    start_seconds=12.48,
                    end_seconds=18.8,
                    source=TextSource.SPEECH,
                ),
                TimedText(
                    text="causing bruises even from minor bumps.",
                    start_seconds=18.8,
                    end_seconds=21,
                    source=TextSource.SPEECH,
                ),
                TimedText(
                    text=(
                        "Article | Open access | Vitamin C-squalene bioconjugate | "
                        "promotes epidermal thickening and | collagen production in human skin"
                    ),
                    start_seconds=9,
                    end_seconds=12,
                    source=TextSource.OCR,
                ),
            ]
        )

        candidates = [
            text for text, _segment in LocalBiomedicalClaimExtractor._candidates(transcript)
        ]

        self.assertIn(
            "Vitamin C is essential for collagen formation, the stuff that keeps your skin "
            "firm and your blood vessels strong",
            candidates,
        )
        self.assertTrue(
            any(
                item.casefold() == "when vitamin c is low, tiny capillaries leak easily, "
                "causing bruises even from minor bumps"
                for item in candidates
            )
        )
        self.assertTrue(
            any("bleeding gums" in item and "not getting more colds" in item for item in candidates)
        )
        self.assertFalse(any(item == "collagen production in human skin" for item in candidates))

    def test_compound_vitamin_c_claim_is_decomposed(self) -> None:
        atomic = decompose_text(
            "The first sign of vitamin C deficiency is bleeding gums, dry skin, "
            "and easy bruising, not getting more colds."
        )
        self.assertEqual(len(atomic), 4)
        self.assertIn("Bleeding gums are the first sign of vitamin C deficiency.", atomic)
        self.assertIn("Getting more colds is not the first sign of vitamin C deficiency.", atomic)

    def test_healthfc_label_mapping_and_metrics(self) -> None:
        gold = [
            normalize_healthfc_label("0"),
            normalize_healthfc_label("2"),
            normalize_healthfc_label("1"),
        ]
        metrics = classification_metrics(gold, [Verdict.TRUE, Verdict.FALSE, Verdict.NEI])
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["macro_f1"], 1.0)

    def test_uploaded_healthfc_dataset_is_readable(self) -> None:
        examples = load_healthfc(Path("data/HealthFC.csv"))
        self.assertGreaterEqual(len(examples), 700)
        self.assertEqual({item.label for item in examples}, {0, 1, 2})

    def test_added_bioasq_and_scifact_archives_are_readable(self) -> None:
        bioasq = load_bioasq_7b(
            Path("data/BioASQ-training7b.zip"), Path("data/Task7BGoldenEnriched.zip")
        )
        scifact = load_scifact(Path("data/data.tar.gz"))
        self.assertEqual(sum(item.split == "train" for item in bioasq), 735)
        self.assertEqual(sum(item.split == "test" for item in bioasq), 139)
        bioasq_claims = [" ".join(item.claim.casefold().split()).strip(" .!?") for item in bioasq]
        self.assertEqual(len(bioasq_claims), len(set(bioasq_claims)))
        self.assertTrue(any(item.label is Verdict.TRUE for item in scifact))
        self.assertTrue(any(item.label is Verdict.FALSE for item in scifact))
        scifact_train_claims = {item.claim.casefold() for item in scifact if item.split == "train"}
        scifact_validation_claims = {
            item.claim.casefold() for item in scifact if item.split == "validation"
        }
        self.assertFalse(scifact_train_claims & scifact_validation_claims)

    def test_pubmed_xml_can_be_prepared_for_optional_offline_experiments(self) -> None:
        xml = """<PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>123</PMID>
        <Article><ArticleTitle>Example trial</ArticleTitle><Abstract>
        <AbstractText Label="RESULTS">Treatment improved the measured outcome.</AbstractText>
        </Abstract><Journal><Title>Example Journal</Title><JournalIssue><PubDate>
        <Year>2025</Year></PubDate></JournalIssue></Journal></Article></MedlineCitation>
        <PubmedData><ArticleIdList><ArticleId IdType="doi">10.1/example</ArticleId>
        </ArticleIdList></PubmedData></PubmedArticle></PubmedArticleSet>"""
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "sample.xml"
            output = Path(temp_dir) / "pubmed.csv"
            source.write_text(xml, encoding="utf-8")
            count = prepare_csv(source, output)
            prepared = output.read_text(encoding="utf-8")
        self.assertEqual(count, 1)
        self.assertIn("Example trial", prepared)
        self.assertIn("10.1/example", prepared)

    def test_pubmed_api_searches_fetches_and_preserves_relevance_order(self) -> None:
        calls: list[str] = []
        xml = b"""<PubmedArticleSet>
        <PubmedArticle><MedlineCitation><PMID>111</PMID><Article>
        <ArticleTitle>Second result</ArticleTitle><Abstract>
        <AbstractText>Second abstract.</AbstractText>
        </Abstract><Journal><Title>Journal B</Title></Journal></Article></MedlineCitation>
        <PubmedData><ArticleIdList /></PubmedData></PubmedArticle>
        <PubmedArticle><MedlineCitation><PMID>222</PMID><Article>
        <ArticleTitle>Most relevant result</ArticleTitle><Abstract>
        <AbstractText Label="RESULTS">First abstract.</AbstractText></Abstract>
        <Journal><Title>Journal A</Title></Journal></Article></MedlineCitation>
        <PubmedData><ArticleIdList><ArticleId IdType="doi">10.1/test</ArticleId>
        </ArticleIdList></PubmedData></PubmedArticle></PubmedArticleSet>"""

        def fake_request(url: str, _timeout: float) -> bytes:
            calls.append(url)
            if "esearch.fcgi" in url:
                return json.dumps({"esearchresult": {"idlist": ["222", "111"]}}).encode()
            return xml

        retriever = PubMedAPIRetriever(
            email="researcher@example.com",
            max_results=2,
            candidate_results=5,
            semantic_rerank=False,
            request_function=fake_request,
        )
        with patch.object(retriever, "_respect_rate_limit"):
            evidence = retriever.retrieve(Claim(text="Treatment improves an outcome."))

        self.assertEqual([item.pmid for item in evidence], ["222", "111"])
        self.assertEqual(evidence[0].doi, "10.1/test")
        search_parameters = urllib.parse.parse_qs(urllib.parse.urlparse(calls[0]).query)
        self.assertEqual(search_parameters["db"], ["pubmed"])
        self.assertEqual(search_parameters["retmax"], ["5"])
        self.assertEqual(search_parameters["email"], ["researcher@example.com"])
        self.assertIn("id=222%2C111", calls[1])

        cached = retriever.retrieve(Claim(text="Treatment improves an outcome."))
        self.assertEqual([item.pmid for item in cached], ["222", "111"])
        self.assertEqual(len(calls), 2)

    def test_pubmed_relevance_does_not_require_matching_negation_word(self) -> None:
        claim = "The first sign of vitamin C deficiency is not getting more colds."
        abstract = (
            "Three patients had scurvy caused by dietary vitamin C deficiency. "
            "Cutaneous findings and treatment response were described."
        )
        evidence = Evidence(
            pmid="11908167",
            title="Three cases of scurvy",
            abstract=abstract,
            url="https://pubmed.ncbi.nlm.nih.gov/11908167/",
            relevance_score=0.90,
        )
        retriever = PubMedAPIRetriever(email="researcher@example.com", minimum_term_coverage=0.10)

        self.assertGreater(_claim_coverage(claim, abstract), 0)
        self.assertFalse(_negation_is_addressed(claim, abstract))
        self.assertTrue(retriever._addresses_claim(claim, evidence))

    def test_pubmed_query_expands_vitamin_c_and_bleeding_gum_synonyms(self) -> None:
        queries = build_pubmed_queries("Bleeding gums are an early sign of vitamin C deficiency.")
        self.assertIn("scurvy", queries[0])
        self.assertIn("gingival bleeding", queries[0])

    def test_wrong_micronutrient_is_rejected_even_with_high_similarity(self) -> None:
        claim = "Vitamin C deficiency causes bleeding gums."
        evidence = Evidence(
            pmid="999",
            title="Clinical findings in vitamin A deficiency",
            abstract="Vitamin A deficiency can cause dry skin and ocular symptoms.",
            url="https://pubmed.ncbi.nlm.nih.gov/999/",
            relevance_score=0.99,
            term_coverage=0.80,
        )
        retriever = PubMedAPIRetriever(email="researcher@example.com")

        audit = audit_entity_consistency(claim, f"{evidence.title}. {evidence.abstract}")
        self.assertFalse(audit.passes)
        self.assertIn("vitamin_c", audit.missing)
        self.assertFalse(retriever._addresses_claim(claim, evidence))

    def test_wrong_micronutrient_is_removed_from_retrieval_results(self) -> None:
        xml = b"""<PubmedArticleSet><PubmedArticle><MedlineCitation>
        <PMID>999</PMID><Article><ArticleTitle>Vitamin A deficiency</ArticleTitle>
        <Abstract><AbstractText>Vitamin A deficiency causes dry skin.</AbstractText></Abstract>
        </Article></MedlineCitation><PubmedData><ArticleIdList /></PubmedData>
        </PubmedArticle></PubmedArticleSet>"""

        def fake_request(url: str, _timeout: float) -> bytes:
            if "esearch.fcgi" in url:
                return json.dumps({"esearchresult": {"idlist": ["999"]}}).encode()
            return xml

        class FakeEncoder:
            def encode(self, texts: list[str], **_kwargs: object) -> np.ndarray:
                if len(texts) == 1:
                    return np.array([[1.0, 0.0]])
                return np.array([[0.99, 0.1] for _text in texts])

        retriever = PubMedAPIRetriever(
            email="researcher@example.com",
            request_function=fake_request,
            minimum_term_coverage=0.0,
        )
        retriever._encoder = FakeEncoder()
        with patch.object(retriever, "_respect_rate_limit"):
            results = retriever.retrieve(Claim(text="Vitamin C deficiency causes dry skin."))

        self.assertEqual(results, [])

    def test_target_entity_is_accepted_when_an_abstract_compares_vitamins(self) -> None:
        audit = audit_entity_consistency(
            "Vitamin C deficiency causes bleeding gums.",
            "Vitamin A and vitamin C deficiencies have different clinical presentations.",
        )
        self.assertTrue(audit.passes)
        self.assertIn("vitamin_c", audit.matched)
        self.assertIn("vitamin_a", audit.additional)

    def test_vitamin_c_pediatric_abstract_survives_retrieval_filters(self) -> None:
        xml = b"""<PubmedArticleSet><PubmedArticle><MedlineCitation>
        <PMID>41374045</PMID><Article>
        <ArticleTitle>Clinical Characteristics and Outcomes of Pediatric
        Vitamin C Deficiency</ArticleTitle>
        <Abstract><AbstractText>Common clinical signs included limb pain, refusal to walk,
        and bleeding gums in children with vitamin C deficiency.</AbstractText></Abstract>
        <Journal><Title>Nutrients</Title></Journal></Article></MedlineCitation>
        <PubmedData><ArticleIdList /></PubmedData></PubmedArticle></PubmedArticleSet>"""

        def fake_request(url: str, _timeout: float) -> bytes:
            if "esearch.fcgi" in url:
                return json.dumps({"esearchresult": {"idlist": ["41374045"]}}).encode()
            return xml

        class FakeEncoder:
            def encode(self, texts: list[str], **_kwargs: object) -> np.ndarray:
                if len(texts) == 1:
                    return np.array([[1.0, 0.0]])
                return np.array([[0.8, 0.6] for _text in texts])

        retriever = PubMedAPIRetriever(
            email="researcher@example.com",
            request_function=fake_request,
            minimum_term_coverage=0.20,
        )
        retriever._encoder = FakeEncoder()
        with patch.object(retriever, "_respect_rate_limit"):
            evidence = retriever.retrieve(
                Claim(text="Bleeding gums are an early sign of vitamin C deficiency.")
            )

        self.assertEqual([item.pmid for item in evidence], ["41374045"])
        self.assertGreater(evidence[0].term_coverage or 0, 0.20)
        self.assertTrue(evidence[0].retrieval_queries)

    def test_pubmed_relevance_accepts_direct_collagen_evidence(self) -> None:
        claim = "Vitamin C is required for collagen formation."
        evidence = Evidence(
            pmid="123",
            title="Vitamin C and collagen biosynthesis",
            abstract="Vitamin C is required for collagen production in human tissues.",
            url="https://pubmed.ncbi.nlm.nih.gov/123/",
            relevance_score=0.72,
        )
        retriever = PubMedAPIRetriever(email="researcher@example.com")

        self.assertTrue(retriever._addresses_claim(claim, evidence))

    def test_semantic_reranker_scores_even_one_abstract(self) -> None:
        class FakeEncoder:
            def encode(self, texts: list[str], **_kwargs: object) -> np.ndarray:
                if len(texts) == 1 and texts[0] == "Vitamin C supports collagen formation.":
                    return np.array([[1.0, 0.0]])
                return np.array([[0.6, 0.8] for _text in texts])

        retriever = PubMedAPIRetriever(email="researcher@example.com")
        retriever._encoder = FakeEncoder()
        evidence = Evidence(
            pmid="123",
            title="Collagen study",
            abstract="Vitamin C supports collagen formation.",
            url="https://pubmed.ncbi.nlm.nih.gov/123/",
        )

        ranked = retriever._rerank("Vitamin C supports collagen formation.", [evidence])

        self.assertAlmostEqual(ranked[0].relevance_score or 0, 0.6)

    def test_paddleocr_v3_initialization_fallback(self) -> None:
        calls: list[dict[str, object]] = []
        loaded_reader = object()

        def fake_paddleocr(**kwargs: object) -> object:
            calls.append(kwargs)
            if "show_log" in kwargs:
                raise ValueError("Unknown argument: show_log")
            return loaded_reader

        fake_module = types.ModuleType("paddleocr")
        fake_module.PaddleOCR = fake_paddleocr
        with patch.dict(sys.modules, {"paddleocr": fake_module}):
            result = PaddleOCRReader()._load()

        self.assertIs(result, loaded_reader)
        self.assertEqual(len(calls), 2)
        self.assertNotIn("show_log", calls[1])
        self.assertTrue(calls[1]["use_textline_orientation"])
        self.assertFalse(calls[1]["enable_mkldnn"])

    def test_ffmpeg_uses_bundled_binary_when_system_binary_is_missing(self) -> None:
        fake_module = types.ModuleType("imageio_ffmpeg")
        fake_module.get_ffmpeg_exe = lambda: str(Path(__file__))
        with (
            patch("biomed_video_fc.media.processors.shutil.which", return_value=None),
            patch.dict(sys.modules, {"imageio_ffmpeg": fake_module}),
        ):
            binary = FFmpegAudioExtractor()._resolve_binary()
        self.assertEqual(binary, str(Path(__file__)))

    def test_whisper_loads_extracted_wav_without_spawning_ffmpeg(self) -> None:
        samples = np.array([0, 8192, -8192], dtype="<i2")
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "audio.wav"
            with wave.open(str(audio_path), "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(16_000)
                wav_file.writeframes(samples.tobytes())

            loaded = WhisperTranscriber._load_extracted_wav(audio_path)

        self.assertIsInstance(loaded, np.ndarray)
        np.testing.assert_allclose(loaded, [0.0, 0.25, -0.25])


class FakeAudio:
    def extract(self, video_path: Path, output_path: Path) -> Path:
        output_path.write_bytes(b"audio")
        return output_path


class FakeFrames:
    def extract(self, video_path: Path, output_dir: Path) -> list[tuple[Path, float]]:
        output_dir.mkdir(parents=True)
        frame = output_dir / "frame.jpg"
        frame.write_bytes(b"frame")
        return [(frame, 1.0)]


class FakeSpeech:
    def transcribe(self, audio_path: Path) -> list[TimedText]:
        return [
            TimedText(
                text="Drug A lowers fever.",
                start_seconds=0,
                end_seconds=2,
                source=TextSource.SPEECH,
            )
        ]


class FakeOCR:
    def read_frames(self, frames: list[tuple[Path, float]]) -> list[TimedText]:
        return [
            TimedText(
                text="Works in 10 minutes",
                start_seconds=1,
                end_seconds=2,
                source=TextSource.OCR,
                frame_path=frames[0][0],
            )
        ]


class FakeClaims:
    def extract(self, transcript: TranscriptBundle) -> list[Claim]:
        return [Claim(text="Drug A lowers fever in 10 minutes.", source="both")]


class FakeRetriever:
    def retrieve(self, claim: Claim) -> list[Evidence]:
        return []


class FailingRetriever:
    def retrieve(self, claim: Claim) -> list[Evidence]:
        raise ExternalServiceError("PubMed is unavailable")


class FakeReasoner:
    def verify(self, claim: Claim, evidence: list[Evidence]) -> Verification:
        return Verification(verdict=Verdict.NEI, confidence=0.9, reasoning="No evidence.")


class PipelineTests(unittest.TestCase):
    def test_pipeline_orchestrates_all_stages(self) -> None:
        pipeline = BiomedicalFactCheckPipeline(
            FakeAudio(),
            FakeFrames(),
            FakeSpeech(),
            FakeOCR(),
            TimeAlignedTranscriptMerger(),
            FakeClaims(),
            FakeRetriever(),
            FakeReasoner(),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            video = Path(temp_dir) / "sample.mp4"
            video.write_bytes(b"video")
            result = pipeline.analyze(video)
        self.assertEqual(len(result.claim_results), 1)
        self.assertEqual(result.claim_results[0].claim.source, "both")
        self.assertIn("Works in 10 minutes", result.transcript.ocr_text)
        self.assertTrue(result.ocr_enabled)
        self.assertTrue(result.timings)

    def test_pipeline_skips_frame_extraction_and_ocr_when_disabled(self) -> None:
        frames = FakeFrames()
        ocr = FakeOCR()
        pipeline = BiomedicalFactCheckPipeline(
            FakeAudio(),
            frames,
            FakeSpeech(),
            ocr,
            TimeAlignedTranscriptMerger(),
            FakeClaims(),
            FakeRetriever(),
            FakeReasoner(),
        )
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            patch.object(frames, "extract", side_effect=AssertionError("frames should be skipped")),
            patch.object(ocr, "read_frames", side_effect=AssertionError("OCR should be skipped")),
        ):
            video = Path(temp_dir) / "sample.mp4"
            video.write_bytes(b"video")
            result = pipeline.analyze(video, include_ocr=False)

        self.assertFalse(result.ocr_enabled)
        self.assertEqual(result.transcript.ocr_text, "")
        stages = {timing.stage for timing in result.timings}
        self.assertNotIn("frame_extraction", stages)
        self.assertNotIn("ocr", stages)

    def test_retrieval_failure_is_review_required_not_a_truth_prediction(self) -> None:
        pipeline = BiomedicalFactCheckPipeline(
            FakeAudio(),
            FakeFrames(),
            FakeSpeech(),
            FakeOCR(),
            TimeAlignedTranscriptMerger(),
            FakeClaims(),
            FailingRetriever(),
            FakeReasoner(),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            video = Path(temp_dir) / "sample.mp4"
            video.write_bytes(b"video")
            result = pipeline.analyze(video)
        verification = result.claim_results[0].verification
        self.assertEqual(verification.verdict, Verdict.NEI)
        self.assertEqual(verification.confidence, 0.0)
        self.assertEqual(verification.decision_source, "error")
        self.assertTrue(verification.review_required)
        self.assertIn("PubMed is unavailable", result.warnings[0])


class FakeStratifiedGroupKFold:
    def __init__(self, n_splits: int, shuffle: bool, random_state: int) -> None:
        self.n_splits = n_splits

    def split(self, examples, labels, groups):
        unique_groups = list(dict.fromkeys(groups))
        held_groups = set(unique_groups[:: self.n_splits])
        retained = [index for index, group in enumerate(groups) if group not in held_groups]
        held = [index for index, group in enumerate(groups) if group in held_groups]
        yield retained, held


class MultisourceSplitTests(unittest.TestCase):
    @staticmethod
    def _record(dataset: str, split: str, index: int) -> dict[str, object]:
        return {
            "example_id": f"{dataset}:{split}:{index}",
            "claim": f"{dataset} claim number {index}",
            "evidence": f"Evidence number {index}",
            "label": [Verdict.TRUE.value, Verdict.NEI.value, Verdict.FALSE.value][index % 3],
            "dataset": dataset,
            "split": split,
            "group_id": f"group-{index}",
        }

    @staticmethod
    def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
        path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")

    def test_multisource_splits_keep_source_groups_separate(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            prepared = Path(temp_dir)
            self._write_jsonl(
                prepared / "healthfc.jsonl",
                [self._record("healthfc", "unsplit", index) for index in range(14)],
            )
            self._write_jsonl(
                prepared / "bioasq7b.jsonl",
                [self._record("bioasq7b", "train", index) for index in range(7)]
                + [self._record("bioasq7b", "test", 100)],
            )
            self._write_jsonl(
                prepared / "scifact.jsonl",
                [self._record("scifact", "train", index) for index in range(7)]
                + [self._record("scifact", "validation", 100)],
            )
            splits = build_splits(prepared, FakeStratifiedGroupKFold, seed=42)

        grouped = [
            {f"{item.dataset}:{item.group_id}" for item in split}
            for split in (splits.train, splits.validation, splits.test)
        ]
        self.assertFalse(grouped[0] & grouped[1])
        self.assertFalse(grouped[0] & grouped[2])
        self.assertFalse(grouped[1] & grouped[2])
        self.assertTrue(any(item.example_id == "bioasq7b:test:100" for item in splits.test))
        self.assertTrue(any(item.example_id == "scifact:validation:100" for item in splits.test))


class HardNegativeTests(unittest.TestCase):
    def test_hard_negatives_target_entity_negation_and_unsupported_first(self) -> None:
        examples = [
            VerificationExample(
                example_id="source:1",
                claim="Vitamin C is required for collagen formation.",
                evidence="Vitamin C is required for collagen biosynthesis.",
                label=Verdict.TRUE,
                dataset="fixture",
                split="train",
                group_id="one",
            ),
            VerificationExample(
                example_id="source:2",
                claim="Bleeding gums are a sign of vitamin C deficiency.",
                evidence="Bleeding gums are a common sign of vitamin C deficiency.",
                label=Verdict.TRUE,
                dataset="fixture",
                split="train",
                group_id="two",
            ),
        ]
        generated = build_hard_negatives(examples, max_ratio=3, seed=42)
        kinds = {str(item.metadata["augmentation"]) for item in generated}

        self.assertIn("wrong_entity_hard_negative", kinds)
        self.assertIn("negation_flip", kinds)
        self.assertIn("unsupported_first_qualifier", kinds)
        self.assertTrue(all(item.split == "train" for item in generated))
        self.assertTrue(all(item.label in {Verdict.FALSE, Verdict.NEI} for item in generated))


if __name__ == "__main__":
    unittest.main()
