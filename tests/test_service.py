import pytest

from grounded_llm.models import Document, GenerationRequest, ModelDraft, QueryRequest
from grounded_llm.retrieval import BM25Retriever
from grounded_llm.service import GroundedAnswerService, GroundingViolation


class StubProvider:
    def __init__(self, draft: ModelDraft) -> None:
        self.draft = draft
        self.calls = 0
        self.last_request: GenerationRequest | None = None

    async def generate(self, request: GenerationRequest) -> ModelDraft:
        self.calls += 1
        self.last_request = request
        return self.draft


@pytest.mark.asyncio
async def test_service_abstains_without_evidence_and_skips_provider():
    provider = StubProvider(
        ModelDraft(answer="unused", cited_chunk_ids=[], confidence=0.0, abstained=True)
    )
    service = GroundedAnswerService(retriever=BM25Retriever(), provider=provider)
    service.replace_documents(
        [Document(source_id="manual", text="Emergency stop disables drive motion.")]
    )

    response = await service.answer(QueryRequest(query="What is the database password?"))

    assert response.abstained is True
    assert response.citations == []
    assert provider.calls == 0


@pytest.mark.asyncio
async def test_service_accepts_only_citations_from_supplied_evidence():
    provider = StubProvider(
        ModelDraft(
            answer="It disables drive motion.",
            cited_chunk_ids=["manual:1"],
            confidence=0.91,
            abstained=False,
        )
    )
    service = GroundedAnswerService(retriever=BM25Retriever(), provider=provider)
    service.replace_documents(
        [Document(source_id="manual", text="Emergency stop disables drive motion.")]
    )

    response = await service.answer(QueryRequest(query="What does emergency stop disable?"))

    assert response.abstained is False
    assert response.citations[0].source_id == "manual"
    assert provider.calls == 1


@pytest.mark.asyncio
async def test_service_rejects_invented_citation():
    provider = StubProvider(
        ModelDraft(
            answer="Unsupported answer",
            cited_chunk_ids=["invented:99"],
            confidence=0.8,
            abstained=False,
        )
    )
    service = GroundedAnswerService(retriever=BM25Retriever(), provider=provider)
    service.replace_documents(
        [Document(source_id="manual", text="Emergency stop disables drive motion.")]
    )

    with pytest.raises(GroundingViolation, match="not supplied"):
        await service.answer(QueryRequest(query="What does emergency stop disable?"))
