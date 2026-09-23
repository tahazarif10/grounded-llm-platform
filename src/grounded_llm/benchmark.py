from __future__ import annotations

import math
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from .retrieval import Retriever


class RetrievalEvalCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(min_length=1, max_length=120)
    query: str = Field(min_length=1, max_length=4_000)
    expected_source_ids: list[str] = Field(default_factory=list)
    should_abstain: bool = False


class RetrievalCaseResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    query: str
    retrieved_source_ids: list[str]
    expected_source_ids: list[str]
    recall_at_k: float | None
    reciprocal_rank: float | None
    ndcg_at_k: float | None
    miss_category: str


class RetrievalBenchmarkSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cases: int
    answerable_cases: int
    negative_cases: int
    top_k: int
    recall_at_k: float
    mean_reciprocal_rank: float
    ndcg_at_k: float
    negative_zero_hit_rate: float
    per_case: list[RetrievalCaseResult]


def _dedupe_sources(source_ids: Sequence[str], *, limit: int) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for source_id in source_ids:
        if source_id in seen:
            continue
        seen.add(source_id)
        unique.append(source_id)
        if len(unique) == limit:
            break
    return unique


def _dcg(retrieved: Sequence[str], relevant: set[str], *, top_k: int) -> float:
    score = 0.0
    for rank, source_id in enumerate(retrieved[:top_k], start=1):
        if source_id in relevant:
            score += 1.0 / math.log2(rank + 1.0)
    return score


def evaluate_retrieval(
    retriever: Retriever,
    cases: Sequence[RetrievalEvalCase],
    *,
    top_k: int = 5,
) -> RetrievalBenchmarkSummary:
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if not cases:
        raise ValueError("At least one retrieval evaluation case is required")

    results: list[RetrievalCaseResult] = []
    recalls: list[float] = []
    reciprocal_ranks: list[float] = []
    ndcgs: list[float] = []
    negative_zero_hits = 0
    negative_cases = 0

    for case in cases:
        if case.should_abstain and case.expected_source_ids:
            raise ValueError(
                f"Case {case.case_id!r} cannot both abstain and declare expected sources"
            )
        if not case.should_abstain and not case.expected_source_ids:
            raise ValueError(
                f"Case {case.case_id!r} must declare expected sources or set should_abstain"
            )

        hits = retriever.search(case.query, top_k=max(top_k * 4, top_k))
        sources = _dedupe_sources(
            [hit.chunk.source_id for hit in hits],
            limit=top_k,
        )

        if case.should_abstain:
            negative_cases += 1
            zero_hit = not sources
            if zero_hit:
                negative_zero_hits += 1
            results.append(
                RetrievalCaseResult(
                    case_id=case.case_id,
                    query=case.query,
                    retrieved_source_ids=sources,
                    expected_source_ids=[],
                    recall_at_k=None,
                    reciprocal_rank=None,
                    ndcg_at_k=None,
                    miss_category="correct_zero_hit" if zero_hit else "unexpected_lexical_hit",
                )
            )
            continue

        relevant = set(case.expected_source_ids)
        retrieved_relevant = relevant.intersection(sources)
        recall = len(retrieved_relevant) / len(relevant)

        first_rank = next(
            (rank for rank, source_id in enumerate(sources, start=1) if source_id in relevant),
            None,
        )
        reciprocal_rank = 0.0 if first_rank is None else 1.0 / first_rank

        ideal_count = min(len(relevant), top_k)
        ideal_dcg = sum(1.0 / math.log2(rank + 1.0) for rank in range(1, ideal_count + 1))
        ndcg = _dcg(sources, relevant, top_k=top_k) / ideal_dcg

        if retrieved_relevant:
            miss_category = "hit"
        elif not sources:
            miss_category = "no_hits"
        else:
            miss_category = "relevant_source_not_retrieved"

        recalls.append(recall)
        reciprocal_ranks.append(reciprocal_rank)
        ndcgs.append(ndcg)
        results.append(
            RetrievalCaseResult(
                case_id=case.case_id,
                query=case.query,
                retrieved_source_ids=sources,
                expected_source_ids=case.expected_source_ids,
                recall_at_k=recall,
                reciprocal_rank=reciprocal_rank,
                ndcg_at_k=ndcg,
                miss_category=miss_category,
            )
        )

    answerable_cases = len(recalls)
    if answerable_cases == 0:
        raise ValueError("At least one answerable retrieval case is required")

    return RetrievalBenchmarkSummary(
        cases=len(cases),
        answerable_cases=answerable_cases,
        negative_cases=negative_cases,
        top_k=top_k,
        recall_at_k=sum(recalls) / answerable_cases,
        mean_reciprocal_rank=sum(reciprocal_ranks) / answerable_cases,
        ndcg_at_k=sum(ndcgs) / answerable_cases,
        negative_zero_hit_rate=(
            negative_zero_hits / negative_cases if negative_cases else 0.0
        ),
        per_case=results,
    )
