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
import re
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
_QUERY_STOPWORDS = frozenset(
    {
        "a", "an", "and", "any", "are", "as", "at", "be", "been", "but", "by", "can", "did", "do", "does",
        "for", "from", "has", "have", "how", "into", "its", "not", "our", "should", "than", "that", "the",
        "their", "then", "there", "these", "they", "this", "was", "were", "what", "when", "where", "which",
        "who", "why", "will", "with", "would", "you", "your",
    }
)
_RECIPROCAL_RANK_CONSTANT = 60
_CANDIDATES_PER_QUERY = 60
_QUERY_VARIANT_LIMIT = 12

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

    The search runs through the public Memory API. The question is expanded into
    its distinctive terms and the per-query rankings are fused offline, so a page
    that matches one term of the question can still reach the delivered set.
    Every returned citation points at one distinct physical page.
    """

    from powercontext.builtin.runtime.models import GetMemoryEntryRequest, SearchMemoryRequest

    if limit < 1:
        raise ValueError("page search limit must be positive")  # noqa: TRY003
    scores: dict[tuple[str, str], float] = {}
    hits: dict[tuple[str, str], object] = {}
    first_seen: dict[tuple[str, str], int] = {}
    for query in _query_variants(question):
        page = await memory.search(
            SearchMemoryRequest(query=query, mode="auto", limit=_CANDIDATES_PER_QUERY)
        )
        for rank, hit in enumerate(page.hits, start=1):
            key = (hit.entry_id, hit.entry_version_id)
            scores[key] = scores.get(key, 0.0) + 1.0 / (_RECIPROCAL_RANK_CONSTANT + rank)
            hits.setdefault(key, hit)
            first_seen.setdefault(key, len(first_seen))
    ordered = sorted(scores, key=lambda key: (-scores[key], first_seen[key]))
    citations: list[MemoryCitation] = []
    seen_sources: set[str] = set()
    for key in ordered:
        if len(citations) >= limit:
            break
        hit = hits[key]
        citation = MemoryCitation(
            memory_ref=hit.memory_ref,
            entry_id=hit.entry_id,
            entry_version_id=hit.entry_version_id,
        )
        entry = await memory.get(GetMemoryEntryRequest(citation=citation))
        source_ids = {reference.source_id for reference in entry.entry.sources}
        if not source_ids or source_ids & seen_sources:
            continue
        seen_sources |= source_ids
        citations.append(citation)
    return tuple(citations)


def _query_variants(question: str) -> tuple[str, ...]:
    """Return the question and the distinctive terms it is built from."""

    variants = [question]
    seen = {question}
    for term in _QUERY_TOKEN.findall(question.lower()):
        if len(term) < 3 or term in _QUERY_STOPWORDS or term in seen:
            continue
        seen.add(term)
        variants.append(term)
        if len(variants) > _QUERY_VARIANT_LIMIT:
            break
    return tuple(variants)


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
