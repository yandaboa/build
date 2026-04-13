#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from tar_video_loader import TarVideoLoader


POSE_FILE_PATTERN = re.compile(r"frame_(\d{6})_(\d+)_pose\.npz$")


def parse_args() -> argparse.Namespace:
    """Parse command-line options for sampled-frame export."""
    parser = argparse.ArgumentParser(
        description="Save source-video frames that correspond to extracted HaMeR pose files."
    )
    parser.add_argument("--data-root", type=Path, default=Path("/workspace/build/egocentric10k_data"))
    parser.add_argument("--action-root", type=Path, default=Path("/workspace/build/egocentric10k_hamer_meshes"))
    parser.add_argument("--work-root", type=Path, default=Path("/workspace/build/egocentric10k_hamer_work"))
    parser.add_argument("--frame-dir-name", type=str, default="state_frames")
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_pose_requests(video_out_dir: Path) -> dict[int, int | None]:
    """Collect sampled-frame indices and source-frame indices from pose metadata."""
    sampled_to_source: dict[int, int | None] = {}
    for pose_path in sorted(video_out_dir.glob("*_pose.npz")):
        match = POSE_FILE_PATTERN.match(pose_path.name)
        if match is None:
            continue
        sampled_idx = int(match.group(1))
        source_idx: int | None = None
        with np.load(pose_path, allow_pickle=False) as pose_data:
            if "source_frame_index" in pose_data.files:
                source_idx = int(pose_data["source_frame_index"].item())
        sampled_to_source[sampled_idx] = source_idx
    return sampled_to_source


def infer_source_indices(sampled_to_source: dict[int, int | None], source_fps: float, target_fps: float) -> dict[int, int]:
    """Fill missing source-frame indices using the extraction sampling rule."""
    # Uses the same sampling math as demo.py and hamer_inference_pipeline.py.
    sample_every = 1 if target_fps <= 0 else max(int(round(source_fps / target_fps)), 1)
    resolved: dict[int, int] = {}
    for sampled_idx, source_idx in sampled_to_source.items():
        resolved[sampled_idx] = source_idx if source_idx is not None else sampled_idx * sample_every
    return resolved


def save_requested_frames(video_path: Path, out_dir: Path, sampled_to_source: dict[int, int], overwrite: bool) -> int:
    """Decode and save only the requested source frames."""
    out_dir.mkdir(parents=True, exist_ok=True)
    source_to_sampled = {source_idx: sampled_idx for sampled_idx, source_idx in sampled_to_source.items()}
    if not source_to_sampled:
        return 0

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return 0
    saved = 0
    frame_idx = 0
    max_source_idx = max(source_to_sampled.keys())
    while frame_idx <= max_source_idx:
        success, frame = capture.read()
        if not success:
            break
        sampled_idx = source_to_sampled.get(frame_idx)
        if sampled_idx is not None:
            output_path = out_dir / f"frame_{sampled_idx:06d}.jpg"
            if overwrite or not output_path.exists():
                cv2.imwrite(str(output_path), frame)
                saved += 1
        frame_idx += 1
    capture.release()
    return saved


def main() -> int:
    """Export state frames next to action data for policy training."""
    args = parse_args()
    if not args.data_root.exists():
        raise FileNotFoundError(f"Data root not found: {args.data_root}")
    if not args.action_root.exists():
        raise FileNotFoundError(f"Action root not found: {args.action_root}")
    args.work_root.mkdir(parents=True, exist_ok=True)

    loader = TarVideoLoader(data_root=args.data_root, work_root=args.work_root)
    pose_dirs = sorted({path.parent for path in args.action_root.rglob("*_pose.npz")})
    if not pose_dirs:
        print(f"No pose files found under: {args.action_root}")
        return 1

    processed = 0
    skipped = 0
    for video_out_dir in tqdm(pose_dirs, desc="videos", unit="video"):
        relative = video_out_dir.relative_to(args.action_root)
        if len(relative.parts) < 2:
            skipped += 1
            continue
        tar_stem = relative.parts[-2]
        video_stem = relative.parts[-1]
        tar_rel_parent = Path(*relative.parts[:-2]) if len(relative.parts) > 2 else Path(".")
        tar_path = args.data_root / tar_rel_parent / f"{tar_stem}.tar"
        if not tar_path.exists():
            skipped += 1
            continue

        sampled_to_source = parse_pose_requests(video_out_dir)
        if not sampled_to_source:
            skipped += 1
            continue

        with loader.open_shard(tar_path) as shard:
            matches = [member for member in shard.video_members() if member.safe_stem == video_stem]
            if len(matches) != 1:
                skipped += 1
                continue
            member = matches[0]
            temp_work_dir = args.work_root / "_frame_extract_tmp" / tar_rel_parent / tar_stem / video_stem
            extracted_video_path = shard.extract_video(member, temp_work_dir)

        capture = cv2.VideoCapture(str(extracted_video_path))
        source_fps = capture.get(cv2.CAP_PROP_FPS)
        capture.release()
        if source_fps <= 1e-6:
            source_fps = 30.0

        sampled_to_source_resolved = infer_source_indices(sampled_to_source, source_fps, args.fps)
        frame_out_dir = video_out_dir / args.frame_dir_name
        save_requested_frames(extracted_video_path, frame_out_dir, sampled_to_source_resolved, args.overwrite)

        mapping_path = video_out_dir / f"{args.frame_dir_name}_metadata.json"
        mapping_payload = {
            "source_fps": source_fps,
            "target_fps": args.fps,
            "sampled_to_source_frame_index": {
                f"{sampled_idx:06d}": source_idx for sampled_idx, source_idx in sampled_to_source_resolved.items()
            },
        }
        mapping_path.write_text(json.dumps(mapping_payload, indent=2), encoding="utf-8")
        shutil.rmtree(temp_work_dir, ignore_errors=True)
        processed += 1

    print(f"processed={processed}, skipped={skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
