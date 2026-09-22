from __future__ import annotations

from collections.abc import Sequence

from .models import Citation, Document, GenerationRequest, GroundedAnswer, QueryRequest
from .providers import LLMProvider
from .retrieval import Retriever, chunk_documents


class GroundingViolation(RuntimeError):
    """The provider produced an answer that violates the grounding contract."""


class GroundedAnswerService:
    def __init__(
        self,
        *,
        retriever: Retriever,
        provider: LLMProvider,
        minimum_retrieval_score: float = 0.01,
    ) -> None:
        if minimum_retrieval_score < 0:
            raise ValueError("minimum_retrieval_score cannot be negative")
        self._retriever = retriever
        self._provider = provider
        self._minimum_retrieval_score = minimum_retrieval_score

    @property
    def indexed_chunks(self) -> int:
        return self._retriever.size

    def replace_documents(
        self,
        documents: Sequence[Document],
        *,
        max_words_per_chunk: int = 180,
        overlap_lines: int = 2,
    ) -> int:
        chunks = chunk_documents(
            documents,
            max_words_per_chunk=max_words_per_chunk,
            overlap_lines=overlap_lines,
        )
        self._retriever.index(chunks)
        return len(chunks)

    async def answer(self, request: QueryRequest) -> GroundedAnswer:
        hits = self._retriever.search(request.query, top_k=request.top_k)
        if not hits or hits[0].score < self._minimum_retrieval_score:
            return GroundedAnswer(
                answer="I don't have enough retrieved evidence to answer that.",
                citations=[],
                confidence=0.0,
                abstained=True,
                retrieval_count=len(hits),
            )

        evidence = [hit.chunk for hit in hits]
        draft = await self._provider.generate(
            GenerationRequest(query=request.query, evidence=evidence)
        )

        if draft.abstained:
            return GroundedAnswer(
                answer=draft.answer,
                citations=[],
                confidence=draft.confidence,
                abstained=True,
                retrieval_count=len(hits),
            )

        allowed = {chunk.chunk_id: chunk for chunk in evidence}
        citation_ids = list(dict.fromkeys(draft.cited_chunk_ids))
        if not citation_ids:
            raise GroundingViolation("Non-abstained model output did not cite retrieved evidence")

        invalid = [chunk_id for chunk_id in citation_ids if chunk_id not in allowed]
        if invalid:
            raise GroundingViolation("Model cited evidence that was not supplied for this request")

        citations = [
            Citation(
                chunk_id=chunk_id,
                source_id=allowed[chunk_id].source_id,
                start_line=allowed[chunk_id].start_line,
                end_line=allowed[chunk_id].end_line,
            )
            for chunk_id in citation_ids
        ]
        return GroundedAnswer(
            answer=draft.answer,
            citations=citations,
            confidence=draft.confidence,
            abstained=False,
            retrieval_count=len(hits),
        )
