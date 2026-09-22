# Architecture

## Goal

Build a grounded LLM system whose important claims can be tested independently: retrieval, generation, grounding, abstention, and serving.

## Component boundaries

### Ingestion

Input documents are bounded and converted to chunks carrying source ID and line provenance. Chunk identifiers are stable within one deterministic ingestion configuration.

### Retrieval

`Retriever` is a narrow protocol. v0.1 ships an in-memory BM25 implementation as a deterministic lexical baseline. Dense or hybrid retrieval must compete against this baseline in the same evaluation harness.

### Provider

`LLMProvider` accepts a typed `GenerationRequest` and returns a validated `ModelDraft`. The first concrete provider implements a strict subset of an OpenAI-compatible chat-completions endpoint.

The provider boundary owns transport timeout and envelope/schema failure classification. It does not own grounding policy.

### Grounding service

`GroundedAnswerService` owns the request-level invariant:

1. retrieve evidence;
2. abstain before generation if no usable evidence exists;
3. call the provider with only selected evidence;
4. reject a non-abstained response without citations;
5. reject citations outside the exact retrieved-evidence allow-list;
6. attach source/line provenance from trusted application state, not model-provided source metadata.

The model is never trusted to manufacture provenance.

### API

FastAPI is a transport adapter. Domain behavior remains usable without HTTP so tests and evaluation do not need a running server.

## Prompt-injection boundary

Retrieved documents are data controlled by the corpus, not by the application. The provider system contract explicitly states that instructions inside evidence are untrusted and must not override generation rules.

This is one layer, not a complete defense. Later milestones should add adversarial public-corpus cases and measure attack success rather than claim prompt injection is solved.

## Why no agent framework in v0.1

The current problem needs retrieval, one generation boundary, and validation. Adding a graph/agent framework before these contracts are measured would hide behavior without adding capability.

Framework adoption is allowed later when a concrete orchestration requirement justifies it.

## Failure policy

- no retrieval evidence -> abstain, no model call
- provider timeout -> typed provider failure
- HTTP failure -> typed provider failure
- malformed provider JSON -> typed provider failure
- schema-invalid provider output -> typed provider failure
- missing citations on non-abstained output -> grounding violation
- invented citation -> grounding violation

No path silently downgrades to unvalidated model prose.
