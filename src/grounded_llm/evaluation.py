from __future__ import annotations

import math
import time

from pydantic import BaseModel, ConfigDict, Field

from .models import QueryRequest
from .service import GroundedAnswerService


class EvalCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(min_length=1, max_length=120)
    query: str = Field(min_length=1, max_length=4_000)
    should_abstain: bool = False
    expected_source_ids: list[str] = Field(default_factory=list)


class EvalSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cases: int
    task_success_rate: float
    abstention_accuracy: float
    citation_validity_rate: float
    latency_p50_ms: float
    latency_p95_ms: float


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[rank]


async def run_evaluation(
    service: GroundedAnswerService,
    cases: list[EvalCase],
) -> EvalSummary:
    if not cases:
        raise ValueError("At least one evaluation case is required")

    task_successes = 0
    abstention_matches = 0
    citation_valid_cases = 0
    latency_ms: list[float] = []

    for case in cases:
        started = time.perf_counter()
        response = await service.answer(QueryRequest(query=case.query))
        latency_ms.append((time.perf_counter() - started) * 1_000.0)

        if response.abstained == case.should_abstain:
            abstention_matches += 1

        returned_sources = {citation.source_id for citation in response.citations}
        expected_sources = set(case.expected_source_ids)

        if response.abstained:
            citations_valid = not response.citations
        elif expected_sources:
            citations_valid = bool(returned_sources & expected_sources)
        else:
            citations_valid = bool(response.citations)

        if citations_valid:
            citation_valid_cases += 1

        success = response.abstained if case.should_abstain else (
            not response.abstained and citations_valid
        )
        if success:
            task_successes += 1

    count = len(cases)
    return EvalSummary(
        cases=count,
        task_success_rate=task_successes / count,
        abstention_accuracy=abstention_matches / count,
        citation_validity_rate=citation_valid_cases / count,
        latency_p50_ms=_percentile(latency_ms, 0.50),
        latency_p95_ms=_percentile(latency_ms, 0.95),
    )
