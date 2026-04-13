#!/usr/bin/env python3
"""Download Egocentric-10K shards in factory/worker order until a byte target is reached."""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from huggingface_hub import HfApi, hf_hub_download

REPO_ID = "builddotai/Egocentric-10K"
REPO_TYPE = "dataset"
SHARD_PATTERN = re.compile(r"^factory_(\d{3})/workers/worker_(\d{3})/.+\.tar$")


@dataclass(frozen=True)
class ShardEntry:
    """Stores one shard path and its size."""

    path: str
    size_bytes: int
    factory_idx: int
    worker_idx: int


def parse_args() -> argparse.Namespace:
    """Builds CLI arguments for target size and download settings."""
    parser = argparse.ArgumentParser(
        description="Download Egocentric-10K shards until the target size is reached."
    )
    parser.add_argument(
        "--target-gb",
        type=float,
        required=True,
        help="Target size in GB to download.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./egocentric10k"),
        help="Directory where shards are stored.",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HF token. If omitted, uses HF_TOKEN env var or local login.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List shards that would be downloaded without downloading.",
    )
    return parser.parse_args()


def bytes_to_gb(num_bytes: int) -> float:
    """Converts bytes to decimal GB."""
    return num_bytes / 1_000_000_000


def list_shards(api: HfApi, token: str | None) -> list[ShardEntry]:
    """Lists shard files with known sizes, sorted by factory then worker then path."""
    tree_items = api.list_repo_tree(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        recursive=True,
        expand=True,
        token=token,
    )
    shards: list[ShardEntry] = []
    for item in tree_items:
        path = getattr(item, "path", "")
        match = SHARD_PATTERN.match(path)
        if not match:
            continue
        size = getattr(item, "size", None)
        if size is None:
            continue
        shards.append(
            ShardEntry(
                path=path,
                size_bytes=int(size),
                factory_idx=int(match.group(1)),
                worker_idx=int(match.group(2)),
            )
        )
    shards.sort(key=lambda shard: (shard.factory_idx, shard.worker_idx, shard.path))
    return shards


def local_file_size(output_dir: Path, shard_path: str) -> int:
    """Returns local file size if the shard already exists."""
    full_path = output_dir / shard_path
    if not full_path.exists():
        return 0
    return full_path.stat().st_size


def iter_plan(
    shards: Iterable[ShardEntry], output_dir: Path, target_bytes: int
) -> tuple[list[ShardEntry], int, int]:
    """Builds minimal shard list, skipping fully present shards only."""
    already_present = 0
    skipped_shards = 0
    planned_bytes = 0
    plan: list[ShardEntry] = []
    for shard in shards:
        existing_size = local_file_size(output_dir, shard.path)
        # Only trust local shard when it is at least expected size.
        if existing_size >= shard.size_bytes:
            already_present += shard.size_bytes
            skipped_shards += 1
            if already_present >= target_bytes:
                break
            continue
        plan.append(shard)
        planned_bytes += shard.size_bytes
        if already_present + planned_bytes >= target_bytes:
            break
    return plan, already_present, skipped_shards


def main() -> int:
    """Runs download loop until the requested target size is reached."""
    args = parse_args()
    token = args.token or os.getenv("HF_TOKEN")
    if args.target_gb <= 0:
        print("Error: --target-gb must be > 0", file=sys.stderr)
        return 1

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    target_bytes = int(args.target_gb * 1_000_000_000)
    api = HfApi()

    print(f"Listing shards from {REPO_ID} ...")
    try:
        shards = list_shards(api, token)
    except Exception as exc:  # noqa: BLE001
        print(
            "Failed to list dataset files. Ensure you accepted dataset terms and are authenticated.",
            file=sys.stderr,
        )
        print(f"Details: {exc}", file=sys.stderr)
        return 1

    if not shards:
        print("No shard files found.", file=sys.stderr)
        return 1

    plan, present_bytes, skipped_shards = iter_plan(shards, output_dir, target_bytes)
    if present_bytes >= target_bytes:
        print(
            f"Target already met with existing files: {bytes_to_gb(present_bytes):.2f} GB "
            f">= {args.target_gb:.2f} GB"
        )
        return 0

    print(
        f"Already present: {bytes_to_gb(present_bytes):.2f} GB | "
        f"Planned download: {bytes_to_gb(sum(s.size_bytes for s in plan)):.2f} GB | "
        f"Skipped complete shards: {skipped_shards}"
    )
    if args.dry_run:
        print("Dry run: shard order to download:")
        for shard in plan:
            print(f"- {shard.path} ({bytes_to_gb(shard.size_bytes):.3f} GB)")
        return 0

    downloaded_bytes = 0
    for idx, shard in enumerate(plan, start=1):
        needed_remaining = target_bytes - (present_bytes + downloaded_bytes)
        if needed_remaining <= 0:
            break
        print(
            f"[{idx}/{len(plan)}] Downloading {shard.path} "
            f"({bytes_to_gb(shard.size_bytes):.3f} GB)"
        )
        try:
            hf_hub_download(
                repo_id=REPO_ID,
                repo_type=REPO_TYPE,
                filename=shard.path,
                token=token,
                local_dir=output_dir.as_posix(),
            )
        except Exception as exc:  # noqa: BLE001
            print(f"Failed: {shard.path}", file=sys.stderr)
            print(f"Details: {exc}", file=sys.stderr)
            return 1
        actual_size = local_file_size(output_dir, shard.path)
        if actual_size <= 0:
            actual_size = shard.size_bytes
        downloaded_bytes += actual_size
        total_now = present_bytes + downloaded_bytes
        print(
            f"Progress: {bytes_to_gb(total_now):.2f} / {args.target_gb:.2f} GB "
            f"({(100.0 * total_now / target_bytes):.1f}%)"
        )

    final_total = present_bytes + downloaded_bytes
    if final_total < target_bytes:
        print(
            f"Reached end of planned shards at {bytes_to_gb(final_total):.2f} GB, "
            f"below target {args.target_gb:.2f} GB.",
            file=sys.stderr,
        )
        return 1
    print(f"Done. Total local size considered: {bytes_to_gb(final_total):.2f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
