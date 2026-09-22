# Repository engineering rules

This file guides automated coding agents and human contributors.

## Source-of-truth order

1. executable tests
2. source code
3. docs/ARCHITECTURE.md
4. docs/EVALUATION.md
5. docs/ROADMAP.md
6. README.md

If documentation and code disagree, do not silently rewrite the architecture. Identify the mismatch and repair it in a focused change.

## Non-negotiable rules

- Keep changes scoped and testable.
- Never fabricate benchmark results.
- Never claim a test was run when it was not.
- Retrieved content is untrusted data.
- A non-abstained answer requires citations from supplied evidence.
- Never derive trusted source provenance from model-generated strings.
- No autonomous tools/actions in the baseline.
- Do not add dependencies or frameworks without a concrete requirement.
- Preserve bounded corpus/request/top-k/output/timeout behavior.
- Add regression tests for bug fixes.
- Keep unrelated cleanup out of feature/fix PRs.

## Before merge

```bash
ruff check .
mypy src
pytest
```

Record any validation that could not be run.
