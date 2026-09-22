# Evaluation

## Principle

An LLM application needs two different kinds of evidence:

1. **software evidence** — contracts, error handling, deterministic tests, resource bounds;
2. **model/system quality evidence** — retrieval success, grounded answer success, abstention, latency, and failure behavior on a declared dataset.

v0.1 establishes the first category and the evaluation machinery for the second. It does not publish model-quality numbers yet.

## Current eval contract

Each `EvalCase` records:

- stable case ID
- query
- whether the system should abstain
- optional expected source IDs

`run_evaluation()` reports:

- task success rate
- abstention accuracy
- citation validity rate
- latency p50
- latency p95

These metrics are intentionally simple. They are sufficient to gate architecture changes without pretending to measure every aspect of answer quality.

## Next evidence gate

The first public benchmark should use a redistributable technical-document corpus and a checked-in case set. It must record:

- corpus source/version/hash
- ingestion configuration
- retriever configuration
- provider/model/runtime/version
- prompt version
- eval-case version
- exact commit
- retrieval hit rate / MRR or nDCG where labels support it
- grounded task success
- abstention precision/recall
- citation validity
- latency p50/p95
- failure counts

A dense or hybrid retriever is not accepted as "better" until it improves declared metrics on the same cases.

## Non-claims

Passing unit tests or synthetic fixtures does not establish real-document retrieval quality, factual accuracy, low hallucination rate, prompt-injection resistance, production latency, or production readiness.
