from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

MAX_DOCUMENT_BYTES = 1_000_000


def _download(url: str) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "grounded-llm-platform-benchmark/1"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        data = response.read(MAX_DOCUMENT_BYTES + 1)
    if len(data) > MAX_DOCUMENT_BYTES:
        raise RuntimeError(f"Benchmark document exceeds {MAX_DOCUMENT_BYTES} bytes: {url}")
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("benchmarks/zephyr_bt_v1/manifest.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".benchmark-cache/zephyr_bt_v1"),
    )
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for document in manifest["documents"]:
        data = _download(document["raw_url"])
        digest = hashlib.sha256(data).hexdigest()
        if digest != document["sha256"]:
            raise RuntimeError(
                f"SHA-256 mismatch for {document['source_id']}: "
                f"expected {document['sha256']}, got {digest}"
            )

        target = args.output_dir / document["local_name"]
        target.write_bytes(data)
        print(f"verified {document['source_id']}: {digest}")

    print(
        f"prepared {len(manifest['documents'])} documents from "
        f"{manifest['upstream_repository']}@{manifest['upstream_commit']}"
    )


if __name__ == "__main__":
    main()
