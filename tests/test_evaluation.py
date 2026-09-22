import pytest

from grounded_llm.evaluation import EvalCase, run_evaluation
from grounded_llm.models import Document, GenerationRequest, ModelDraft
from grounded_llm.retrieval import BM25Retriever
from grounded_llm.service import GroundedAnswerService


class EchoCitationProvider:
    async def generate(self, request: GenerationRequest) -> ModelDraft:
        return ModelDraft(
            answer="Grounded answer",
            cited_chunk_ids=[request.evidence[0].chunk_id],
            confidence=0.8,
            abstained=False,
        )


@pytest.mark.asyncio
async def test_eval_reports_grounding_and_abstention_metrics():
    service = GroundedAnswerService(
        retriever=BM25Retriever(),
        provider=EchoCitationProvider(),
    )
    service.replace_documents(
        [Document(source_id="safety", text="Emergency stop removes motion permission.")]
    )

    summary = await run_evaluation(
        service,
        [
            EvalCase(
                case_id="grounded",
                query="What does emergency stop remove?",
                expected_source_ids=["safety"],
            ),
            EvalCase(
                case_id="abstain",
                query="What is the database password?",
                should_abstain=True,
            ),
        ],
    )

    assert summary.cases == 2
    assert summary.task_success_rate == 1.0
    assert summary.abstention_accuracy == 1.0
    assert summary.citation_validity_rate == 1.0
    assert summary.latency_p95_ms >= 0.0
