from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from typing import Protocol

from .models import Document, DocumentChunk, RetrievalHit

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _tokens(text: str) -> list[str]:
    return [match.group(0).lower() for match in _TOKEN_RE.finditer(text)]


def chunk_documents(
    documents: Sequence[Document],
    *,
    max_words_per_chunk: int = 180,
    overlap_lines: int = 2,
    max_chunks: int = 5_000,
) -> list[DocumentChunk]:
    if max_words_per_chunk < 1:
        raise ValueError("max_words_per_chunk must be positive")
    if overlap_lines < 0:
        raise ValueError("overlap_lines cannot be negative")

    chunks: list[DocumentChunk] = []
    for document in documents:
        lines = document.text.splitlines() or [document.text]
        start = 0
        sequence = 0

        while start < len(lines):
            word_count = 0
            end = start
            selected: list[str] = []

            while end < len(lines):
                line = lines[end]
                line_words = max(1, len(_tokens(line)))
                if selected and word_count + line_words > max_words_per_chunk:
                    break
                selected.append(line)
                word_count += line_words
                end += 1

            text = "\n".join(selected).strip()
            if text:
                sequence += 1
                chunks.append(
                    DocumentChunk(
                        chunk_id=f"{document.source_id}:{sequence}",
                        source_id=document.source_id,
                        text=text,
                        start_line=start + 1,
                        end_line=end,
                    )
                )
                if len(chunks) > max_chunks:
                    raise ValueError(f"Corpus exceeds the {max_chunks} chunk limit")

            if end >= len(lines):
                break
            start = max(start + 1, end - overlap_lines)

    return chunks


class Retriever(Protocol):
    @property
    def size(self) -> int: ...

    def index(self, chunks: Sequence[DocumentChunk]) -> None: ...

    def search(self, query: str, *, top_k: int) -> list[RetrievalHit]: ...


class BM25Retriever:
    """Deterministic, dependency-light lexical baseline."""

    def __init__(self, *, k1: float = 1.5, b: float = 0.75, max_chunks: int = 5_000) -> None:
        self._k1 = k1
        self._b = b
        self._max_chunks = max_chunks
        self._chunks: list[DocumentChunk] = []
        self._term_frequencies: list[Counter[str]] = []
        self._document_frequency: Counter[str] = Counter()
        self._document_lengths: list[int] = []
        self._average_length = 0.0

    @property
    def size(self) -> int:
        return len(self._chunks)

    def index(self, chunks: Sequence[DocumentChunk]) -> None:
        if len(chunks) > self._max_chunks:
            raise ValueError(f"Corpus exceeds the {self._max_chunks} chunk limit")

        self._chunks = list(chunks)
        self._term_frequencies = []
        self._document_frequency = Counter()
        self._document_lengths = []

        for chunk in self._chunks:
            tokens = _tokens(chunk.text)
            frequencies = Counter(tokens)
            self._term_frequencies.append(frequencies)
            self._document_lengths.append(len(tokens))
            self._document_frequency.update(frequencies.keys())

        self._average_length = (
            sum(self._document_lengths) / len(self._document_lengths)
            if self._document_lengths
            else 0.0
        )

    def search(self, query: str, *, top_k: int) -> list[RetrievalHit]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if not self._chunks:
            return []

        query_terms = list(dict.fromkeys(_tokens(query)))
        if not query_terms:
            return []

        count = len(self._chunks)
        scored: list[tuple[float, int]] = []
        for index, frequencies in enumerate(self._term_frequencies):
            score = 0.0
            document_length = self._document_lengths[index]
            for term in query_terms:
                frequency = frequencies.get(term, 0)
                if frequency == 0:
                    continue

                document_frequency = self._document_frequency[term]
                inverse_document_frequency = math.log(
                    1.0 + (count - document_frequency + 0.5) / (document_frequency + 0.5)
                )
                normalizer = frequency + self._k1 * (
                    1.0
                    - self._b
                    + self._b * document_length / max(self._average_length, 1.0)
                )
                score += inverse_document_frequency * (
                    frequency * (self._k1 + 1.0) / normalizer
                )

            if score > 0.0:
                scored.append((score, index))

        scored.sort(key=lambda item: (-item[0], self._chunks[item[1]].chunk_id))
        return [
            RetrievalHit(chunk=self._chunks[index], score=score)
            for score, index in scored[:top_k]
        ]
