# Zephyr Bluetooth Shell retrieval benchmark v1

This benchmark is the first external-corpus retrieval gate for the project.

## Corpus

The corpus is a five-document subset of the Zephyr Project Bluetooth shell documentation,
pinned to commit `70be2ff0b565a3313128f5577f51cfeb3ebcf602`.

The source repository is Apache-2.0 licensed. The corpus text is **not vendored** here.
`manifest.json` records the exact upstream paths, Git blob SHAs, raw URLs, and SHA-256
digests. `scripts/prepare_benchmark.py` downloads the pinned files and fails closed if any
digest differs.

## Cases

`cases.jsonl` contains ten answerable retrieval cases and two negative lexical cases. The
answerable cases use document-level relevance labels. The negative cases are expected to
produce no lexical hit.

The cases were authored from the same public documents by the project maintainer. They are
therefore a **development benchmark**, not a hidden or independently curated test set. Results
must not be presented as general RAG quality.

## Run

```bash
python scripts/prepare_benchmark.py
python scripts/run_retrieval_benchmark.py
```

The report records the corpus and case hashes, chunking parameters, BM25 parameters, exact Git
commit when available, per-case results, Recall@k, MRR, nDCG@k, and negative zero-hit rate.

## Regression gate

The first measured run established a perfect score on this small development-visible suite.
`manifest.json` therefore records a 1.0 floor for Recall@5, MRR, nDCG@5, and negative zero-hit
rate. The gate is a regression contract for this exact corpus/case version, not a claim that BM25
has perfect retrieval quality in general.
