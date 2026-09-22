## Scope

What single technical problem does this PR solve?

## Change

Describe the smallest behavior or contract change.

## Validation

- [ ] `ruff check .`
- [ ] `mypy src`
- [ ] `pytest`
- [ ] container build when Docker/runtime inputs changed
- [ ] no secrets, local paths, datasets, model weights, or generated artifacts committed

List any validation that could not be run.

## Evidence boundary

State which claims this PR supports and which it does **not** support.

## Architecture check

- [ ] retrieved content remains untrusted data
- [ ] provenance is not trusted from model-generated output
- [ ] non-abstained answers remain citation-grounded
- [ ] resource/timeout bounds are not weakened
- [ ] no unrelated framework, refactor, or dependency churn
