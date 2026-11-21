"""Stream NCBI PubMed baseline XML files into the CSV consumed by the offline index builder."""

from __future__ import annotations

import argparse
import csv
import gzip
import logging
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from pathlib import Path
from typing import Any, BinaryIO

from biomed_video_fc.exceptions import ConfigurationError

logger = logging.getLogger(__name__)

FIELDNAMES = ["pmid", "title", "abstract", "journal", "publication_date", "doi"]


def _text(element: ET.Element | None) -> str:
    return "" if element is None else "".join(element.itertext()).strip()


def parse_pubmed_article(item: ET.Element) -> dict[str, str] | None:
    citation = item.find("MedlineCitation")
    article = item.find(".//Article")
    pmid = _text(citation.find("PMID") if citation is not None else None)
    title = _text(article.find("ArticleTitle") if article is not None else None)
    abstract_parts: list[str] = []
    if article is not None:
        for part in article.findall(".//Abstract/AbstractText"):
            value = _text(part)
            label = part.attrib.get("Label", "").strip()
            if value:
                abstract_parts.append(f"{label}: {value}" if label else value)
    abstract = " ".join(abstract_parts)
    if not (pmid and title and abstract):
        return None
    journal = _text(article.find(".//Journal/Title") if article is not None else None)
    pub_date = article.find(".//JournalIssue/PubDate") if article is not None else None
    publication_date = " ".join(
        value
        for value in [
            _text(pub_date.find("Year") if pub_date is not None else None),
            _text(pub_date.find("Month") if pub_date is not None else None),
            _text(pub_date.find("Day") if pub_date is not None else None),
            _text(pub_date.find("MedlineDate") if pub_date is not None else None),
        ]
        if value
    )
    doi = ""
    for article_id in item.findall(".//PubmedData/ArticleIdList/ArticleId"):
        if article_id.attrib.get("IdType") == "doi":
            doi = _text(article_id)
            break
    return {
        "pmid": pmid,
        "title": title,
        "abstract": abstract,
        "journal": journal,
        "publication_date": publication_date,
        "doi": doi,
    }


def iter_pubmed_records(handle: BinaryIO) -> Iterator[dict[str, str]]:
    for _event, element in ET.iterparse(handle, events=("end",)):
        if element.tag.rsplit("}", 1)[-1] != "PubmedArticle":
            continue
        record = parse_pubmed_article(element)
        if record is not None:
            yield record
        element.clear()


def _source_files(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    if source.is_dir():
        files = sorted([*source.glob("*.xml.gz"), *source.glob("*.xml")])
        if files:
            return files
    raise ConfigurationError(f"No .xml or .xml.gz PubMed files found at {source}")


def prepare_csv(source: Path, output: Path, max_records: int | None = None) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8", newline="") as output_handle:
        writer = csv.DictWriter(output_handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for source_file in _source_files(source):
            opener: Any = gzip.open if source_file.suffix == ".gz" else open
            logger.info("Parsing %s", source_file.name)
            with opener(source_file, "rb") as input_handle:
                for record in iter_pubmed_records(input_handle):
                    writer.writerow(record)
                    count += 1
                    if max_records is not None and count >= max_records:
                        return count
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert PubMed baseline XML into a local CSV")
    parser.add_argument(
        "source", type=Path, help="An XML/XML.GZ file or directory of baseline files"
    )
    parser.add_argument("--output", type=Path, default=Path("data/pubmed_abstracts.csv"))
    parser.add_argument("--max-records", type=int)
    args = parser.parse_args()
    count = prepare_csv(args.source, args.output, args.max_records)
    print(f"Wrote {count} PubMed abstracts to {args.output}")


if __name__ == "__main__":
    main()
