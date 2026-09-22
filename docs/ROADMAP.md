# Roadmap

The roadmap is evidence-gated. A milestone closes only when its behavior and claims are reproducible.

## M0 — Engineering foundation

- typed domain contracts
- deterministic BM25 baseline
- source/line provenance
- OpenAI-compatible provider boundary
- strict JSON output validation
- citation allow-listing
- abstention path
- FastAPI adapter
- deterministic tests and CI
- non-root container

Exit gate: CI green on the exact merge candidate.

## M1 — Public-corpus RAG benchmark

- choose a redistributable technical-document corpus
- check in corpus manifest/hashes, not ambiguous downloads
- build labeled retrieval/eval cases
- report BM25 retrieval and grounded-answer baseline
- record model/runtime/prompt versions
- publish reproducible metrics and limitations

Exit gate: a third party can reproduce the benchmark from documented inputs.

## M2 — Dense + hybrid retrieval

- embedding-provider protocol
- local embedding implementation
- bounded persistent vector index
- hybrid BM25+dense fusion
- optional reranking behind a narrow interface
- compare against M1 on the exact same cases

Exit gate: adoption requires measured benefit, not novelty.

## M3 — Robustness and adversarial evaluation

- retrieved prompt-injection cases
- conflicting-source cases
- stale/outdated-source cases
- oversized/malformed provider responses
- provider timeout/disconnect/retry policy
- explicit grounding/abstention error analysis

Exit gate: publish failure taxonomy and measured regressions.

## M4 — Operability

- request IDs and structured metadata-only telemetry
- bounded concurrency/backpressure
- health/readiness semantics
- container smoke test
- load/latency benchmark
- optional persistence with schema/version migration

Authentication and Internet-facing hardening are separate deployment work and must not be implied by a local development benchmark.
