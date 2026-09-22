# Security

## Reporting

Avoid publishing credentials or sensitive exploit details in a public issue. Use GitHub's private vulnerability-reporting channel if it is enabled for the repository.

## Trust boundaries

All indexed and retrieved document text is treated as **untrusted data**. Retrieved content must never override the system instruction contract.

Current security invariants:

- API keys come from process environment, never repository files.
- Provider responses are schema-validated before use.
- Citations are allow-listed against the exact evidence supplied to the provider.
- Corpus, request, retrieval, response, and timeout bounds are explicit.
- The baseline has no tool execution, shell execution, autonomous action, browser control, or model-directed filesystem write path.
- Provider errors fail closed rather than falling back to unvalidated prose.

## Out of scope for v0.1

Authentication, multi-tenant isolation, encrypted persistence, rate limiting, and Internet-facing deployment hardening are not implemented yet. Do not expose the development API directly to an untrusted network.
