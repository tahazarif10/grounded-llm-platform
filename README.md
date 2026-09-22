# Grounded LLM Platform

A production-oriented reference implementation for **retrieval-augmented generation (RAG)** with explicit grounding contracts, structured model outputs, deterministic retrieval baselines, citation validation, evaluation, and bounded failure handling.

This is intentionally not a chatbot demo. The engineering surface is the product: retrieval quality, prompt/data trust boundaries, strict output validation, abstention, provider failures, reproducible evals, and API serving.

## Current scope — v0.1 foundation

- deterministic in-memory BM25 retrieval baseline
- bounded text ingestion with source and line provenance
- provider abstraction plus an OpenAI-compatible HTTP implementation
- strict JSON model-output validation with Pydantic
- citation allow-listing against the exact retrieved evidence
- fail-closed abstention when retrieval has no usable evidence
- prompt-injection boundary: retrieved documents are untrusted data, never instructions
- FastAPI index/query surface
- deterministic evaluation harness for grounding, abstention, and latency
- unit/integration tests, Ruff, mypy, GitHub Actions, and a non-root Docker image

No model-quality benchmark claim is made yet. A public-corpus benchmark and dense-retrieval comparison are separate evidence gates.

## Architecture

```text
documents -> bounded chunking + provenance -> retriever -> top-k evidence
                                                     |
                                                     v
                                              grounding service
                                                /          \
                              insufficient evidence          LLM provider
                                      |                         |
                                   abstain                 structured draft
                                                                |
                                                     citation allow-list
                                                                |
                                                      grounded response
```

The baseline starts with BM25 on purpose. Dense embeddings, hybrid retrieval, reranking, persistence, and public-corpus evaluation are added only behind measured milestones.

## Quick start

Python 3.11+:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
ruff check .
mypy src
pytest
```

Run against an OpenAI-compatible endpoint:

```bash
export GLLM_BASE_URL=http://127.0.0.1:8001
export GLLM_MODEL=your-model
export GLLM_API_KEY=
uvicorn grounded_llm.api:create_app --factory --host 127.0.0.1 --port 8080
```

Index a document:

```bash
curl -X POST http://127.0.0.1:8080/v1/index \
  -H "content-type: application/json" \
  -d '{"documents":[{"source_id":"manual-1","text":"Emergency stop removes motion permission before drive commands are accepted."}]}'
```

Query it:

```bash
curl -X POST http://127.0.0.1:8080/v1/query \
  -H "content-type: application/json" \
  -d '{"query":"What does the emergency stop do?","top_k":5}'
```

## Engineering contracts

1. **Evidence before generation.** The model sees only the evidence selected for the request.
2. **Retrieved text is untrusted.** Instructions embedded in documents never outrank the system contract.
3. **No invented citations.** Returned citations must map to chunks supplied to the provider.
4. **Structured output is mandatory.** Malformed output is a typed failure, not best-effort prose parsing.
5. **Abstention is first-class.** No usable evidence means no provider call.
6. **Resources are bounded.** Document, corpus, query, top-k, model-output, and timeout limits are explicit.
7. **Measure before claiming.** Synthetic tests validate software behavior, not model quality.

## Repository map

```text
src/grounded_llm/   domain, retrieval, provider, service, eval, API
tests/              deterministic regression tests
docs/               architecture, evaluation protocol, roadmap
.github/workflows/  CI
```

## Evidence boundary

The repository currently demonstrates **LLM systems engineering**, not general model intelligence. Until the public-corpus evaluation milestone is completed, it makes no claim about real-document answer accuracy, hallucination rate, or production readiness.

See [Architecture](docs/ARCHITECTURE.md), [Evaluation](docs/EVALUATION.md), and [Roadmap](docs/ROADMAP.md).

## License

MIT
