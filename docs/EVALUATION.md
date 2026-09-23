# Evaluation

## Principle

An LLM application needs two different kinds of evidence:

1. **software evidence** — contracts, error handling, deterministic tests, resource bounds;
2. **model/system quality evidence** — retrieval success, grounded answer success, abstention,
   latency, and failure behavior on a declared dataset.

v0.1 establishes the first category and the machinery required to collect the second.

## Synthetic contract evaluation

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

These fixtures verify software behavior. They do not establish real model quality.

## External-corpus retrieval baseline

M1 begins with `benchmarks/zephyr_bt_v1`, a five-document subset of the Zephyr Bluetooth shell
documentation pinned to commit `70be2ff0b565a3313128f5577f51cfeb3ebcf602`.

Reproducibility properties:

- exact upstream repository and commit are recorded;
- exact source paths and Git blob SHAs are recorded;
- every downloaded file is verified with SHA-256 before use;
- the external corpus is not silently vendored or refreshed;
- chunking and BM25 parameters are versioned in the manifest;
- cases are checked in as JSONL with stable IDs;
- machine-readable output records manifest/case hashes and the exact evaluated Git commit.

The retrieval report includes:

- Recall@k
- mean reciprocal rank (MRR)
- nDCG@k
- negative zero-hit rate
- per-case retrieved sources and miss category

The cases are authored from the same public documents and are visible to developers. This makes
the suite appropriate for deterministic regression and retriever comparison, but not a hidden
generalization benchmark.

## Remaining M1 evidence gate

A model-backed public-corpus run must still record:

- provider/model/runtime/version
- prompt version
- exact commit
- grounded task success
- abstention precision/recall
- citation validity
- provider/schema failure counts
- latency p50/p95
- machine-readable per-case results

Dense or hybrid retrieval is not accepted as "better" until it improves declared metrics on the
same versioned retrieval cases.

## Non-claims

Passing unit tests, synthetic fixtures, or the development-visible retrieval benchmark does not
establish factual answer accuracy, low hallucination rate, prompt-injection resistance, production
latency, or production readiness.
