from __future__ import annotations
from src import AudioDeduplicator

import argparse
import json
from pathlib import Path

def _discover_audio_files(input_dir: Path, extension: str, recursive: bool) -> list[str]:
    ext = extension.lower().lstrip(".")
    pattern = f"*.{ext}"
    files = input_dir.rglob(pattern) if recursive else input_dir.glob(pattern)
    return sorted(str(p) for p in files if p.is_file())


def _write_jsonl(paths: list[str], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for path in paths:
            f.write(json.dumps({"path": path}, ensure_ascii=True) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run audio deduplication on a directory of files.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/audios"),
        help="Directory containing audio files.",
    )
    parser.add_argument(
        "--extension",
        default="flac",
        help="Audio file extension to scan for (default: flac).",
    )
    parser.add_argument(
        "--non-recursive",
        action="store_true",
        help="Only scan the input directory (do not recurse).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.88,
        help="Duplicate similarity threshold in [0, 1].",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Worker processes for fingerprint extraction.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of files to process.",
    )
    parser.add_argument(
        "--removed-out",
        type=Path,
        default=Path("data/manifests/dedup_removed.jsonl"),
        help="Path to write removed duplicate paths as JSONL.",
    )
    parser.add_argument(
        "--keep-out",
        type=Path,
        default=Path("data/manifests/dedup_keep.jsonl"),
        help="Path to write kept paths as JSONL.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be between 0.0 and 1.0")

    if args.max_workers <= 0:
        parser.error("--max-workers must be a positive integer")

    recursive = not args.non_recursive
    input_dir: Path = args.input_dir
    if not input_dir.exists() or not input_dir.is_dir():
        parser.error(f"Input directory does not exist or is not a directory: {input_dir}")

    paths = _discover_audio_files(input_dir, args.extension, recursive)
    if args.limit is not None:
        if args.limit <= 0:
            parser.error("--limit must be a positive integer")
        paths = paths[: args.limit]

    if not paths:
        print("No matching audio files found.")
        return

    deduplicator = AudioDeduplicator(
        threshold=args.threshold,
        max_workers=args.max_workers,
    )
    keep, removed = deduplicator.deduplicate(paths, show_progress=not args.no_progress)

    removed_sorted = sorted(removed)
    _write_jsonl(keep, args.keep_out)
    _write_jsonl(removed_sorted, args.removed_out)

    print(f"input_files={len(paths)}")
    print(f"kept={len(keep)}")
    print(f"removed={len(removed_sorted)}")
    print(f"keep_manifest={args.keep_out}")
    print(f"removed_manifest={args.removed_out}")


if __name__ == "__main__":
    main()
