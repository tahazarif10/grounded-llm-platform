from grounded_llm.benchmark import RetrievalEvalCase, evaluate_retrieval
from grounded_llm.models import Document
from grounded_llm.retrieval import BM25Retriever, chunk_documents


def _retriever() -> BM25Retriever:
    retriever = BM25Retriever()
    retriever.index(
        chunk_documents(
            [
                Document(source_id="motion", text="servo enable motion permission axis"),
                Document(source_id="network", text="ethernet packet counters diagnostics"),
            ]
        )
    )
    return retriever


def test_retrieval_benchmark_reports_rank_metrics_and_negative_zero_hits():
    summary = evaluate_retrieval(
        _retriever(),
        [
            RetrievalEvalCase(
                case_id="answerable",
                query="servo motion permission",
                expected_source_ids=["motion"],
            ),
            RetrievalEvalCase(
                case_id="negative",
                query="wpa3 sae pmkid",
                should_abstain=True,
            ),
        ],
        top_k=2,
    )

    assert summary.cases == 2
    assert summary.answerable_cases == 1
    assert summary.negative_cases == 1
    assert summary.recall_at_k == 1.0
    assert summary.mean_reciprocal_rank == 1.0
    assert summary.ndcg_at_k == 1.0
    assert summary.negative_zero_hit_rate == 1.0
    assert summary.per_case[0].miss_category == "hit"
    assert summary.per_case[1].miss_category == "correct_zero_hit"


def test_retrieval_benchmark_rejects_ambiguous_case_contract():
    try:
        evaluate_retrieval(
            _retriever(),
            [
                RetrievalEvalCase(
                    case_id="bad",
                    query="servo",
                    expected_source_ids=["motion"],
                    should_abstain=True,
                )
            ],
        )
    except ValueError as exc:
        assert "cannot both abstain" in str(exc)
    else:
        raise AssertionError("Ambiguous evaluation case was accepted")
