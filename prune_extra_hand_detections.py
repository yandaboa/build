#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path


FRAME_ARTIFACT_PATTERN = re.compile(r"^frame_(\d{6})_(\d+)(?:_pose)?\.(obj|png|npz)$")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for duplicate hand-artifact cleanup."""
    parser = argparse.ArgumentParser(
        description="Delete per-frame hand artifacts with detection index >= 2."
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=Path(
            "/workspace/build/egocentric10k_hamer_meshes/factory_001/workers/worker_001/"
            "factory001_worker001_part00/factory001_worker001_00000"
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    """Remove duplicate hand outputs and keep only index 0/1 per frame."""
    args = parse_args()
    if not args.target_dir.exists():
        raise FileNotFoundError(f"Target directory not found: {args.target_dir}")

    removed = 0
    for path in sorted(args.target_dir.iterdir()):
        if not path.is_file():
            continue
        match = FRAME_ARTIFACT_PATTERN.match(path.name)
        if match is None:
            continue
        hand_index = int(match.group(2))
        if hand_index < 2:
            continue
        print(f"delete: {path}")
        if not args.dry_run:
            path.unlink()
        removed += 1

    print(f"removed_files={removed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
