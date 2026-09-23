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
        "summary": summary.model_dump(mode="json"),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
