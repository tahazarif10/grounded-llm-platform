import httpx
import pytest

from grounded_llm.api import create_app
from grounded_llm.models import GenerationRequest, ModelDraft
from grounded_llm.retrieval import BM25Retriever
from grounded_llm.service import GroundedAnswerService


class ApiProvider:
    async def generate(self, request: GenerationRequest) -> ModelDraft:
        return ModelDraft(
            answer="Motion permission is removed.",
            cited_chunk_ids=[request.evidence[0].chunk_id],
            confidence=0.9,
            abstained=False,
        )


@pytest.mark.asyncio
async def test_index_then_query_roundtrip():
    service = GroundedAnswerService(retriever=BM25Retriever(), provider=ApiProvider())
    transport = httpx.ASGITransport(app=create_app(service))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        index_response = await client.post(
            "/v1/index",
            json={
                "documents": [
                    {
                        "source_id": "manual",
                        "text": "Emergency stop removes motion permission.",
                    }
                ]
            },
        )
        assert index_response.status_code == 200
        assert index_response.json()["indexed_chunks"] == 1

        query_response = await client.post(
            "/v1/query",
            json={"query": "What does emergency stop remove?", "top_k": 5},
        )

    assert query_response.status_code == 200
    body = query_response.json()
    assert body["abstained"] is False
    assert body["citations"][0]["source_id"] == "manual"
