# Contributing

This project prefers small, evidence-backed changes over broad framework or architecture churn.

## Development gate

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
ruff check .
mypy src
pytest
```

A pull request should keep one technical purpose, include regression coverage for behavior changes, and state exactly what was and was not validated.

## Design rules

- Keep provider, retrieval, orchestration, and API boundaries separable.
- Do not add an agent framework to avoid implementing a small explicit contract.
- Do not accept free-form provider output when a typed schema is available.
- Preserve source provenance through ingestion and retrieval.
- Treat retrieved content as untrusted data.
- Do not weaken abstention or citation validation to make a demo look successful.
- New quality/performance claims require checked-in evaluation methodology and reproducible evidence.

## Commit and PR scope

Prefer conventional, imperative subjects such as:

- `feat: add hybrid retrieval scoring`
- `test: cover malformed provider output`
- `docs: record public-corpus benchmark protocol`

Avoid unrelated cleanup in a behavior-bearing PR.
