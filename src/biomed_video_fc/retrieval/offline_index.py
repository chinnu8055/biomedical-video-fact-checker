"""CER-style offline PubMed FAISS index construction and retrieval."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any

from biomed_video_fc.domain.models import Claim, Evidence
from biomed_video_fc.exceptions import ConfigurationError, ModelUnavailableError

logger = logging.getLogger(__name__)

INDEX_FILE = "pubmed.faiss"
METADATA_FILE = "metadata.jsonl"
MANIFEST_FILE = "manifest.json"


class OfflinePubMedIndexBuilder:
    """Build an index from a local CSV containing PMID, title, and abstract columns."""

    def __init__(self, embedding_model: str, batch_size: int = 64) -> None:
        self.embedding_model = embedding_model
        self.batch_size = batch_size

    @staticmethod
    def _dependencies() -> tuple[Any, Any]:
        try:
            import faiss
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ModelUnavailableError(
                "FAISS and Sentence Transformers are required. Install: "
                "pip install -e '.[offline-index]'"
            ) from exc
        return faiss, SentenceTransformer

    def build(
        self,
        corpus_path: Path,
        output_dir: Path,
        pmid_column: str = "pmid",
        title_column: str = "title",
        abstract_column: str = "abstract",
        overwrite: bool = False,
    ) -> dict[str, Any]:
        if not corpus_path.is_file():
            raise ConfigurationError(f"PubMed corpus CSV does not exist: {corpus_path}")
        output_dir.mkdir(parents=True, exist_ok=True)
        targets = [output_dir / INDEX_FILE, output_dir / METADATA_FILE, output_dir / MANIFEST_FILE]
        if not overwrite and any(path.exists() for path in targets):
            raise ConfigurationError(
                f"Index files already exist in {output_dir}; pass --overwrite to replace them"
            )

        faiss, sentence_transformer = self._dependencies()
        encoder = sentence_transformer(self.embedding_model)
        index: Any = None
        metadata_count = 0
        pending_texts: list[str] = []
        pending_metadata: list[dict[str, Any]] = []

        def flush(metadata_handle: Any) -> None:
            nonlocal index, metadata_count
            if not pending_texts:
                return
            vectors = encoder.encode(
                pending_texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            ).astype("float32")
            if index is None:
                index = faiss.IndexFlatIP(vectors.shape[1])
            index.add(vectors)
            for record in pending_metadata:
                metadata_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            metadata_count += len(pending_metadata)
            pending_texts.clear()
            pending_metadata.clear()

        with corpus_path.open("r", encoding="utf-8-sig", newline="") as source_handle:
            reader = csv.DictReader(source_handle)
            required = {pmid_column, title_column, abstract_column}
            if not required.issubset(set(reader.fieldnames or [])):
                raise ConfigurationError(
                    f"Corpus must contain columns {sorted(required)}; found {reader.fieldnames}"
                )
            with (output_dir / METADATA_FILE).open("w", encoding="utf-8") as metadata_handle:
                for row in reader:
                    pmid = str(row.get(pmid_column) or "").strip()
                    title = str(row.get(title_column) or "").strip()
                    abstract = str(row.get(abstract_column) or "").strip()
                    if not (pmid and title and abstract):
                        continue
                    record = {
                        "pmid": pmid,
                        "title": title,
                        "abstract": abstract,
                        "journal": str(row.get("journal") or "").strip() or None,
                        "publication_date": str(row.get("publication_date") or "").strip() or None,
                        "doi": str(row.get("doi") or "").strip() or None,
                    }
                    pending_texts.append(f"{title}. {abstract}")
                    pending_metadata.append(record)
                    if len(pending_texts) >= self.batch_size:
                        flush(metadata_handle)
                flush(metadata_handle)

        if index is None or metadata_count == 0:
            raise ConfigurationError("No valid PubMed rows were found in the corpus")
        faiss.write_index(index, str(output_dir / INDEX_FILE))
        manifest = {
            "format_version": 1,
            "embedding_model": self.embedding_model,
            "documents": metadata_count,
            "source_file": corpus_path.name,
        }
        (output_dir / MANIFEST_FILE).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info("Built offline PubMed index with %d abstracts", metadata_count)
        return manifest


class OfflinePubMedRetriever:
    def __init__(self, index_dir: Path, embedding_model: str, max_results: int = 8) -> None:
        self.index_dir = Path(index_dir)
        self.embedding_model = embedding_model
        self.max_results = max_results
        self._index: Any = None
        self._metadata: list[dict[str, Any]] = []
        self._encoder: Any = None

    def _load(self) -> None:
        if self._index is not None:
            return
        index_path = self.index_dir / INDEX_FILE
        metadata_path = self.index_dir / METADATA_FILE
        manifest_path = self.index_dir / MANIFEST_FILE
        if not all(path.is_file() for path in [index_path, metadata_path, manifest_path]):
            raise ModelUnavailableError(
                "The offline PubMed index is missing. Build it with: "
                "biofc-build-index data/pubmed_abstracts.csv"
            )
        try:
            import faiss
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ModelUnavailableError(
                "FAISS and Sentence Transformers are required. Install: "
                "pip install -e '.[offline-index]'"
            ) from exc
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        model_name = str(manifest.get("embedding_model") or self.embedding_model)
        self._index = faiss.read_index(str(index_path))
        with metadata_path.open("r", encoding="utf-8") as handle:
            self._metadata = [json.loads(line) for line in handle if line.strip()]
        if self._index.ntotal != len(self._metadata):
            raise ConfigurationError("FAISS index and metadata have different document counts")
        logger.info("Loading offline embedding model %s", model_name)
        self._encoder = SentenceTransformer(model_name)

    def retrieve(self, claim: Claim) -> list[Evidence]:
        self._load()
        query = self._encoder.encode(
            [claim.text], convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32")
        count = min(self.max_results, int(self._index.ntotal))
        scores, indices = self._index.search(query, count)
        evidence: list[Evidence] = []
        for score, index_position in zip(scores[0].tolist(), indices[0].tolist(), strict=True):
            if index_position < 0:
                continue
            item = self._metadata[index_position]
            pmid = str(item["pmid"])
            evidence.append(
                Evidence(
                    pmid=pmid,
                    title=item["title"],
                    abstract=item["abstract"],
                    journal=item.get("journal"),
                    publication_date=item.get("publication_date"),
                    doi=item.get("doi"),
                    url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                    relevance_score=max(0.0, min(1.0, (float(score) + 1.0) / 2.0)),
                )
            )
        logger.info("Offline index returned %d abstracts for claim %s", len(evidence), claim.id)
        return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a local FAISS index from a PubMed CSV")
    parser.add_argument("corpus", type=Path, help="CSV with pmid, title, and abstract columns")
    parser.add_argument("--output", type=Path, default=Path("artifacts/pubmed_index"))
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    manifest = OfflinePubMedIndexBuilder(args.embedding_model, args.batch_size).build(
        args.corpus, args.output, overwrite=args.overwrite
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
