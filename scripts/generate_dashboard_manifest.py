#!/usr/bin/env python3
"""Generate docs/manifest.json for static dashboard browsing on GitHub Pages."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _isoformat(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).replace(microsecond=0).isoformat()


def _read_optional_meta(meta_path: Path) -> dict[str, Any]:
    if not meta_path.exists():
        return {}

    try:
        with meta_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
            return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        # Ignore malformed metadata to keep manifest generation robust.
        return {}


def build_manifest(docs_dir: Path) -> dict[str, Any]:
    runs_dir = docs_dir / "runs"
    records: list[dict[str, Any]] = []

    for html_path in runs_dir.glob("*/display.html"):
        if not html_path.is_file():
            continue

        run_dir = html_path.parent
        stats = html_path.stat()
        meta = _read_optional_meta(run_dir / "meta.json")
        run_id = run_dir.name

        record = {
            "id": run_id,
            "title": meta.get("title", run_id),
            "path": str(html_path.relative_to(docs_dir)).replace("\\", "/"),
            "updated_at": _isoformat(stats.st_mtime),
            "description": meta.get("description", ""),
            "tags": meta.get("tags", []),
        }

        for key in ("model_path", "config_path"):
            value = meta.get(key)
            if isinstance(value, str) and value:
                record[key] = value

        records.append(record)

    records.sort(key=lambda item: item["updated_at"], reverse=True)

    return {
        "generated_at": datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat(),
        "runs": records,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--docs-dir",
        default="docs",
        help="Path to the docs directory that contains runs/ (default: docs)",
    )
    parser.add_argument(
        "--output",
        default="manifest.json",
        help="Output file path relative to docs-dir (default: manifest.json)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    docs_dir = Path(args.docs_dir).resolve()
    docs_dir.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest(docs_dir)

    output_path = docs_dir / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")

    print(f"Wrote {output_path}")
    print(f"Indexed {len(manifest['runs'])} runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
