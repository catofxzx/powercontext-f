# Copyright (c) 2026 OceanBase.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Capture one PDF document as page-level evidence Sources."""

from __future__ import annotations

import hashlib
import io
import math
import re
from collections import Counter
from typing import TYPE_CHECKING
from urllib.parse import quote

from pydantic import BaseModel
from pypdf import PdfReader

from powercontext.artifacts.models import MemoryCitation
from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources.definitions import AdapterSourceDefinition
from powercontext.sources.models import Source, SourceMaterialization
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY, TextEvidence

if TYPE_CHECKING:
    from powercontext.sources.definitions import SourceDefinition

PDF_PAGE_SOURCE_NAME = "pdf-page"
PDF_PAGE_SOURCE_VERSION = "1"
_PDF_IDENTITY_PREFIX = "pdf:"
_PDF_MAGIC = b"%PDF-"
_QUERY_TOKEN = re.compile(r"[a-z]+|\d+")
_CANDIDATES_PER_QUERY = 60
# The evidence pages of a financial question are rarely written in the wording
# of the question: a question about "revenue" is answered by a page that says
# "net sales", and a question about "days payable outstanding" is answered by a
# balance sheet. The bridge below maps the wording of a question onto the
# vocabulary a financial statement actually uses, so that the lexical search
# reaches the pages that carry the answer.
_FINANCE_ALIASES: dict[str, tuple[str, ...]] = {
    "revenue": ("net sales", "net product sales", "total revenues", "net revenue"),
    "revenues": ("net sales", "net product sales", "total revenues", "net revenue"),
    "sales": ("net sales", "net product sales", "revenues"),
    "margin": ("gross profit", "operating income", "cost of sales"),
    "gross": ("gross profit", "cost of sales"),
    "operating": ("operating income", "income from operations"),
    "profit": ("gross profit", "net income"),
    "income": ("net income", "operating income"),
    "earnings": ("net earnings", "net income"),
    "payable": ("accounts payable", "trade payables"),
    "payables": ("accounts payable",),
    "receivable": ("accounts receivable",),
    "capital": ("capital expenditures", "property plant and equipment", "net sales"),
    "capex": ("capital expenditures", "purchases of property"),
    "expenditure": ("capital expenditures", "purchases of property"),
    "expenditures": ("capital expenditures", "purchases of property"),
    "asset": ("total assets", "current assets"),
    "assets": ("total assets", "current assets"),
    "liabilities": ("total liabilities", "current liabilities"),
    "equity": ("stockholders equity", "shareholders equity"),
    "cash": ("cash and cash equivalents", "cash flows"),
    "tax": ("income taxes", "effective tax rate"),
    "taxes": ("income taxes", "effective tax rate"),
    "debt": ("long term debt", "short term debt"),
    "depreciation": ("depreciation and amortization",),
    "amortization": ("depreciation and amortization",),
    "ebitda": ("operating income", "depreciation and amortization"),
    "customer": ("customer concentration", "significant portion", "substantial portion", "revenues"),
    "customers": ("customer concentration", "significant portion", "substantial portion", "revenues"),
    "concentration": ("customer concentration", "accounted for", "of our revenue"),
    "segment": ("business segment", "reportable segment"),
    "segments": ("business segment", "reportable segment"),
    "geographic": ("north america", "latin america", "europe", "asia pacific", "middle east"),
    "geographies": ("north america", "latin america", "europe", "asia pacific", "middle east"),
    "geography": ("north america", "latin america", "europe", "asia pacific", "middle east"),
    "product": ("products", "product categories"),
    "products": ("products", "product categories"),
    "category": ("categories", "disaggregation"),
    "categories": ("category", "disaggregation"),
    "disaggregation": ("disaggregated",),
    "legal": ("legal proceedings", "litigation", "contingencies"),
    "litigation": ("legal proceedings", "contingencies"),
    "lawsuit": ("legal proceedings",),
    "lawsuits": ("legal proceedings",),
    "battle": ("legal proceedings",),
    "battles": ("legal proceeding",),
    "proceeding": ("legal proceedings",),
    "restructuring": ("restructuring", "restructuring charges"),
    "cyclical": ("cyclicality", "seasonality", "cyclical"),
    "cyclicality": ("cyclical", "seasonality"),
    "production": ("production rate",),
    "rate": ("production rate", "tax rate"),
    "cost": ("cost of sales", "cost of goods sold"),
    "costs": ("cost of sales", "cost of goods sold"),
    "dpo": ("accounts payable", "days payable"),
    "quick": ("cash and cash equivalents", "marketable securities"),
    "liquidity": ("current assets", "current liabilities"),
    "working": ("working capital",),
    "growth": ("increase", "net sales"),
    "increase": ("increase",),
    "decrease": ("decrease",),
    "acquisition": ("acquisitions", "business combination"),
    "merger": ("acquisitions", "business combination"),
    "share": ("per share", "shares"),
    "dividend": ("dividends",),
    "interest": ("interest expense",),
    "risk": ("risk factors",),
    "volume": ("unit sales", "sales volume"),
    "price": ("pricing", "average selling price"),
    "supply": ("supplier", "suppliers"),
    "supplier": ("suppliers",),
    "suppliers": ("supplier",),
}
# Words that carry no evidence of their own in a financial report.
_EXPANSION_STOPWORDS = frozenset(
    {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "do", "does", "did", "what",
        "which", "who", "whom", "whose", "how", "why", "when", "where", "of", "in", "on", "at", "to", "for",
        "from", "by", "with", "without", "and", "or", "if", "then", "than", "that", "this", "these", "those",
        "there", "here", "it", "its", "as", "not", "no", "nor", "but", "so", "such", "can", "could", "should",
        "would", "may", "might", "will", "shall", "have", "has", "had", "having", "about", "into", "over",
        "under", "again", "further", "once", "more", "most", "other", "some", "any", "each", "few", "own",
        "same", "too", "very", "just", "don", "now", "please", "based", "state", "explain", "define",
        "calculation", "calculate", "according", "information", "company", "companies", "business", "fy",
        "year", "years", "quarter", "annual", "metric", "useful",
    }
)
# A question that asks for a figure is answered by a statement page, which
# carries the numbers; a question that asks for a narrative is answered by the
# discussion pages, which do not.
_METRIC_QUESTION_CUES = (
    "what is",
    "calculate",
    "how much",
    "defined as",
    "compute",
    "ratio",
    "outstanding",
    "percent",
    "margin",
    "in units of",
)
_SEARCH_PLAN_LIMIT = 20
_RERANK_POOL_LIMIT = 400
_BM25_K1 = 2.2
_BM25_B = 0.75
_COVERAGE_WEIGHT = 20.0
_NUMERIC_WEIGHT = 40.0

__all__ = [
    "PDF_PAGE_SOURCE_DEFINITION",
    "PDF_PAGE_SOURCE_NAME",
    "PDF_PAGE_SOURCE_VERSION",
    "PDFDocumentInput",
    "PDFPageInput",
    "PDFPageSource",
    "parse_pdf_document",
    "search_pdf_pages",
]


class PDFDocumentInput(BaseModel):
    """One complete PDF document."""

    document_id: str
    content: bytes


class PDFPageInput(BaseModel):
    """One physical page of a PDF document."""

    document_id: str
    document_sha256: str
    page_number: int
    text: str


class PDFPageSource(Source):
    """One captured PDF page kept as retrievable evidence."""

    source_id: str
    document_id: str
    document_sha256: str
    page_number: int
    text: str


class PDFPageSourceAdapter:
    """Resolve and read one captured PDF page."""

    input_class = PDFPageInput
    name = PDF_PAGE_SOURCE_NAME
    source_class = PDFPageSource

    async def resolve(self, value: PDFPageInput, /) -> PDFPageSource:
        source_id = pdf_page_source_id(value.document_id, value.document_sha256, value.page_number)
        return PDFPageSource(
            name=source_id,
            source_id=source_id,
            document_id=value.document_id,
            document_sha256=value.document_sha256,
            page_number=value.page_number,
            text=value.text,
            materialization=SourceMaterialization.CAPTURED,
        )

    async def read(self, source: PDFPageSource, /) -> PDFPageInput:
        return PDFPageInput(
            document_id=source.document_id,
            document_sha256=source.document_sha256,
            page_number=source.page_number,
            text=source.text,
        )


class PDFPageTextEvidenceProjection:
    """Expose one captured PDF page as text evidence."""

    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = PDFPageSource
    output_class: type[BaseModel] = TextEvidence

    def project(self, source: PDFPageSource, /) -> TextEvidence:
        return TextEvidence(
            source_type=PDF_PAGE_SOURCE_NAME,
            source_id=source.source_id,
            content=f"{source.document_id}\n{source.text}",
            metadata={
                "document_id": source.document_id,
                "document_sha256": source.document_sha256,
                "page_number": source.page_number,
            },
        )


PDF_PAGE_SOURCE_DEFINITION: SourceDefinition[PDFPageInput, PDFPageSource, PDFPageInput] = AdapterSourceDefinition(
    PDFPageSourceAdapter(),
    version=PDF_PAGE_SOURCE_VERSION,
    projections=(PDFPageTextEvidenceProjection(),),
)


def pdf_page_source_id(document_id: str, document_sha256: str, page_number: int) -> str:
    """Return the stable identity of one captured PDF page."""

    _validate_document_id(document_id)
    identity = (
        f"{_PDF_IDENTITY_PREFIX}{quote(document_id, safe='')}:{document_sha256}:{page_number}"
    )
    if len(identity) > MAX_SOURCE_ID_LENGTH:
        raise ValueError("PDF page source identity must not exceed the maximum source id length")  # noqa: TRY003
    return identity


def parse_pdf_document(value: PDFDocumentInput) -> tuple[PDFPageInput, ...]:
    """Parse every physical page of one PDF document.

    The whole document is parsed before any page is returned, so a damaged file
    never yields partial pages.
    """

    document_id = _validate_document_id(value.document_id)
    content = value.content
    if not content:
        raise ValueError("PDF document must not be empty")  # noqa: TRY003
    if not content.startswith(_PDF_MAGIC):
        raise ValueError("PDF document must start with the PDF header")  # noqa: TRY003
    digest = hashlib.sha256(content).hexdigest()
    pages = _read_pages(content)
    if not pages:
        raise ValueError("PDF document must contain at least one page")  # noqa: TRY003
    return tuple(
        PDFPageInput(
            document_id=document_id,
            document_sha256=digest,
            page_number=page_number,
            text=text,
        )
        for page_number, text in pages
    )


async def search_pdf_pages(memory, question: str, limit: int = 10) -> tuple[MemoryCitation, ...]:
    """Return the exact Memory citations of the pages that answer one question.

    The question is first expanded into short lexical queries, because one long
    query would demand that a page carries a quarter of its words and would
    therefore drop the statement pages that carry the answer. The pages the
    queries return are then reranked offline over the question, its financial
    wording and the shape of each page, and the best distinct pages are
    returned. Every returned citation points at one distinct physical page.
    """

    from powercontext.builtin.runtime.models import GetMemoryEntryRequest, SearchMemoryRequest

    if limit < 1:
        raise ValueError("page search limit must be positive")  # noqa: TRY003
    pool: dict[tuple[str, str], object] = {}
    best_rank: dict[tuple[str, str], int] = {}
    for query in _search_plan(question):
        page = await memory.search(SearchMemoryRequest(query=query, mode="fts", limit=_CANDIDATES_PER_QUERY))
        for rank, hit in enumerate(page.hits, start=1):
            key = (hit.entry_id, hit.entry_version_id)
            pool.setdefault(key, hit)
            best_rank[key] = min(best_rank.get(key, rank), rank)
    if not pool:
        return ()
    entries: dict[tuple[str, str], object] = {}
    ranked_keys = sorted(pool, key=lambda key: (best_rank[key], key))
    for key in ranked_keys[:_RERANK_POOL_LIMIT]:
        hit = pool[key]
        citation = MemoryCitation(
            memory_ref=hit.memory_ref,
            entry_id=hit.entry_id,
            entry_version_id=hit.entry_version_id,
        )
        entries[key] = await memory.get(GetMemoryEntryRequest(citation=citation))
    scores = _score_pages(question, {key: entry.entry.text for key, entry in entries.items()})
    ordered = sorted(entries, key=lambda key: (-scores.get(key, 0.0), key))
    citations: list[MemoryCitation] = []
    seen_sources: set[str] = set()
    for key in ordered:
        if len(citations) >= limit:
            break
        entry = entries[key]
        source_ids = {reference.source_id for reference in entry.entry.sources}
        if not source_ids or source_ids & seen_sources:
            continue
        seen_sources |= source_ids
        hit = pool[key]
        citations.append(
            MemoryCitation(
                memory_ref=hit.memory_ref,
                entry_id=hit.entry_id,
                entry_version_id=hit.entry_version_id,
            )
        )
    return tuple(citations)


def _search_plan(question: str) -> tuple[str, ...]:
    """Return the short queries that collect the candidate pages of one question.

    Every query is one or two words, so a page only has to carry one of them to
    enter the candidate set instead of a quarter of the whole question.
    """

    queries: list[str] = []
    for term in _question_terms(question):
        if term not in queries:
            queries.append(term)
        for alias in _finance_aliases(term):
            for word in alias.split():
                if word not in queries:
                    queries.append(word)
    for term in _question_terms(question):
        for alias in _finance_aliases(term):
            if len(alias.split()) > 1 and alias not in queries:
                queries.append(alias)
    return tuple(queries[:_SEARCH_PLAN_LIMIT])


def _question_terms(question: str) -> tuple[str, ...]:
    """Return the content words of one question."""

    seen: set[str] = set()
    terms: list[str] = []
    for token in _QUERY_TOKEN.findall(question.lower()):
        if len(token) < 3 or token in _EXPANSION_STOPWORDS or token in seen:
            continue
        seen.add(token)
        terms.append(token)
    return tuple(terms)


def _finance_aliases(term: str) -> tuple[str, ...]:
    """Return the statement vocabulary a question word stands for."""

    return _FINANCE_ALIASES.get(term, _FINANCE_ALIASES.get(_stem(term), ()))


def _stem(word: str) -> str:
    """Reduce a word to the form a financial statement uses for it."""

    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith("sses"):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        word = word[:-1]
    for suffix in ("ically", "ility", "ation", "ment", "ally", "ance", "ence", "ings"):
        if len(word) > len(suffix) + 3 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def _expanded_terms(question: str) -> tuple[str, ...]:
    """Return the stemmed terms that the page text is matched against."""

    seen: set[str] = set()
    terms: list[str] = []
    for term in _question_terms(question):
        for value in (term, *_finance_aliases(term)):
            for word in value.split():
                key = _stem(word)
                if key and key not in seen:
                    seen.add(key)
                    terms.append(key)
    return tuple(terms)


def _is_metric_question(question: str) -> bool:
    """Report whether the question asks for a computed figure."""

    lowered = question.lower()
    return any(cue in lowered for cue in _METRIC_QUESTION_CUES)


def _numeric_ratio(text: str) -> float:
    """Return the share of the page that is written as figures."""

    tokens = _QUERY_TOKEN.findall(text.lower())
    if not tokens:
        return 0.0
    figures = sum(1 for token in tokens if token.isdigit())
    return figures / len(tokens)


def _score_pages(question: str, texts: dict[tuple[str, str], str]) -> dict[tuple[str, str], float]:
    """Rank the candidate pages of one question.

    The score combines the lexical relevance of the page for the question and
    its financial wording, the share of the question the page covers, and the
    share of the page that is written as figures.
    """

    terms = _expanded_terms(question)
    if not terms:
        return dict.fromkeys(texts, 0.0)
    counters = {key: Counter(_stem_tokens(text)) for key, text in texts.items()}
    lengths = {key: max(1, sum(counter.values())) for key, counter in counters.items()}
    average = sum(lengths.values()) / max(1, len(lengths))
    document_frequency: Counter[str] = Counter()
    for counter in counters.values():
        document_frequency.update(counter.keys())
    total = len(counters)
    unique = tuple(dict.fromkeys(terms))
    weights = {
        term: math.log(1 + (total - document_frequency.get(term, 0) + 0.5) / (document_frequency.get(term, 0) + 0.5))
        for term in unique
    }
    quantitative = _is_metric_question(question)
    scores: dict[tuple[str, str], float] = {}
    for key, counter in counters.items():
        length = lengths[key]
        relevance = 0.0
        for term in unique:
            frequency = counter.get(term, 0)
            if not frequency:
                continue
            relevance += (
                weights[term]
                * (frequency * _BM25_K1)
                / (frequency + _BM25_K1 * (1 - _BM25_B + _BM25_B * length / average))
            )
        covered = sum(1 for term in unique if counter.get(term))
        score = relevance + _COVERAGE_WEIGHT * (covered / len(unique))
        if quantitative:
            score += _NUMERIC_WEIGHT * _numeric_ratio(texts[key])
        scores[key] = score
    return scores


def _stem_tokens(text: str) -> list[str]:
    """Return the stemmed word and number tokens of one text."""

    return [_stem(token) for token in _QUERY_TOKEN.findall(text.lower())]


def _read_pages(content: bytes) -> list[tuple[int, str]]:
    try:
        reader = PdfReader(io.BytesIO(content))
        if reader.is_encrypted and not reader.decrypt(""):
            raise ValueError("PDF document cannot be opened with an empty password")  # noqa: TRY003
        pages: list[tuple[int, str]] = []
        for page_number, page in enumerate(reader.pages, start=1):
            pages.append((page_number, page.extract_text() or ""))
    except ValueError:
        raise
    except Exception as error:
        raise ValueError(f"PDF document cannot be parsed: {error}") from error
    return pages


def _validate_document_id(document_id: str) -> str:
    if not isinstance(document_id, str) or not document_id or document_id != document_id.strip():
        raise ValueError("document id must be a non-empty trimmed string")  # noqa: TRY003
    return document_id
