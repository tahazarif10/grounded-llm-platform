from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException

from .models import GroundedAnswer, IndexRequest, QueryRequest
from .providers import OpenAICompatibleProvider, ProviderError
from .retrieval import BM25Retriever
from .service import GroundedAnswerService, GroundingViolation


def _service_from_environment() -> GroundedAnswerService:
    base_url = os.environ.get("GLLM_BASE_URL", "").strip()
    model = os.environ.get("GLLM_MODEL", "").strip()
    if not base_url or not model:
        raise RuntimeError("GLLM_BASE_URL and GLLM_MODEL are required")

    provider = OpenAICompatibleProvider(
        base_url=base_url,
        model=model,
        api_key=os.environ.get("GLLM_API_KEY") or None,
        timeout_seconds=float(os.environ.get("GLLM_TIMEOUT_SECONDS", "30")),
        max_output_tokens=int(os.environ.get("GLLM_MAX_OUTPUT_TOKENS", "700")),
    )
    return GroundedAnswerService(retriever=BM25Retriever(), provider=provider)


def create_app(service: GroundedAnswerService | None = None) -> FastAPI:
    active_service = service or _service_from_environment()
    app = FastAPI(
        title="Grounded LLM Platform",
        version="0.1.0",
        description="Grounded RAG service with bounded retrieval and strict citation validation.",
    )

    @app.get("/health")
    async def health() -> dict[str, int | str]:
        return {"status": "ok", "indexed_chunks": active_service.indexed_chunks}

    @app.post("/v1/index")
    async def index(request: IndexRequest) -> dict[str, int]:
        try:
            count = active_service.replace_documents(
                request.documents,
                max_words_per_chunk=request.max_words_per_chunk,
                overlap_lines=request.overlap_lines,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"indexed_chunks": count}

    @app.post("/v1/query", response_model=GroundedAnswer)
    async def query(request: QueryRequest) -> GroundedAnswer:
        try:
            return await active_service.answer(request)
        except ProviderError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        except GroundingViolation as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    return app
