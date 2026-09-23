from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

from grounded_llm.benchmark import RetrievalEvalCase, evaluate_retrieval
from grounded_llm.models import Document
from grounded_llm.retrieval import BM25Retriever, chunk_documents


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit() -> str:
    github_sha = os.environ.get("GITHUB_SHA")
    if github_sha:
        return github_sha

    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return "unknown"
    return completed.stdout.strip()


def _quality_gate_failures(summary: dict[str, object], gate: dict[str, float]) -> list[str]:
    checks = {
        "recall_at_k": "min_recall_at_k",
        "mean_reciprocal_rank": "min_mean_reciprocal_rank",
        "ndcg_at_k": "min_ndcg_at_k",
        "negative_zero_hit_rate": "min_negative_zero_hit_rate",
    }
    failures: list[str] = []
    for metric, gate_name in checks.items():
        value = float(summary[metric])
        minimum = float(gate[gate_name])
        if value < minimum:
            failures.append(f"{metric}={value:.6f} is below {gate_name}={minimum:.6f}")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("benchmarks/zephyr_bt_v1/manifest.json"),
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path("benchmarks/zephyr_bt_v1/cases.jsonl"),
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=Path(".benchmark-cache/zephyr_bt_v1"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark-results/zephyr_bt_v1.json"),
    )
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    cases = [
        RetrievalEvalCase.model_validate_json(line)
        for line in args.cases.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    documents = []
    for entry in manifest["documents"]:
        path = args.corpus_dir / entry["local_name"]
        if not path.is_file():
            raise RuntimeError(
                f"Missing corpus document {path}; run scripts/prepare_benchmark.py first"
            )
        documents.append(
            Document(
                source_id=entry["source_id"],
                text=path.read_text(encoding="utf-8"),
                metadata={"upstream_path": entry["path"]},
            )
        )

    chunking = manifest["chunking"]
    retriever_config = manifest["retriever"]
    chunks = chunk_documents(
        documents,
        max_words_per_chunk=chunking["max_words_per_chunk"],
        overlap_lines=chunking["overlap_lines"],
    )
    retriever = BM25Retriever(
        k1=retriever_config["k1"],
        b=retriever_config["b"],
    )
    retriever.index(chunks)

    summary = evaluate_retrieval(retriever, cases, top_k=args.top_k)
    summary_json = summary.model_dump(mode="json")
    quality_gate = manifest["quality_gate"]
    gate_failures = _quality_gate_failures(summary_json, quality_gate)

    report = {
        "benchmark": manifest["name"],
        "git_commit": _git_commit(),
        "corpus": {
            "upstream_repository": manifest["upstream_repository"],
            "upstream_commit": manifest["upstream_commit"],
            "license": manifest["license"],
            "manifest_sha256": _sha256(args.manifest),
        },
        "evaluation_set": {
            "cases_sha256": _sha256(args.cases),
            "case_count": len(cases),
        },
        "chunking": chunking,
        "retriever": retriever_config,
        "quality_gate": {
            "thresholds": quality_gate,
            "passed": not gate_failures,
            "failures": gate_failures,
        },
        "summary": summary_json,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    print(json.dumps(report["quality_gate"], indent=2, sort_keys=True))
    print(f"wrote {args.output}")

    if gate_failures:
        raise SystemExit("Retrieval quality gate failed")


if __name__ == "__main__":
    main()
