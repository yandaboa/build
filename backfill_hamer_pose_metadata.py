#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
import re

import numpy as np
from tqdm import tqdm

from tar_video_loader import TarVideoLoader

try:
    import cv2
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "OpenCV (cv2) is required for backfill_hamer_pose_metadata.py. "
        "Run this script with the hand env python:\n"
        "/workspace/build/.venv_hand/bin/python /workspace/build/backfill_hamer_pose_metadata.py ..."
    ) from exc


POSE_FILE_PATTERN = re.compile(r"frame_(\d{6})_(\d+)_pose\.npz$")


def parse_args() -> argparse.Namespace:
    """Parse arguments for one-off pose metadata backfill."""
    parser = argparse.ArgumentParser(
        description="Backfill frame metadata and state frames for existing HaMeR pose outputs."
    )
    parser.add_argument("--data-root", type=Path, default=Path("/workspace/build/egocentric10k_data"))
    parser.add_argument("--action-root", type=Path, default=Path("/workspace/build/egocentric10k_hamer_meshes"))
    parser.add_argument("--work-root", type=Path, default=Path("/workspace/build/egocentric10k_hamer_work"))
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--frame-dir-name", type=str, default="state_frames")
    parser.add_argument("--overwrite-metadata", action="store_true")
    parser.add_argument("--overwrite-frames", action="store_true")
    parser.add_argument("--save-debug-images", action="store_true")
    parser.add_argument("--max-debug-images", type=int, default=5)
    parser.add_argument("--debug-dir-name", type=str, default="debug_overlays")
    return parser.parse_args()


def infer_sampled_indices(video_out_dir: Path) -> list[int]:
    """Collect sampled frame indices from pose filenames."""
    sampled_indices: set[int] = set()
    for pose_path in video_out_dir.glob("*_pose.npz"):
        match = POSE_FILE_PATTERN.match(pose_path.name)
        if match is None:
            continue
        sampled_indices.add(int(match.group(1)))
    return sorted(sampled_indices)


def sample_every_for_fps(source_fps: float, target_fps: float) -> int:
    """Compute sampling stride using the same rule as extraction."""
    return 1 if target_fps <= 0 else max(int(round(source_fps / target_fps)), 1)


def read_source_fps(video_path: Path) -> float:
    """Read video FPS and return fallback if unavailable."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return 30.0
    source_fps = capture.get(cv2.CAP_PROP_FPS)
    capture.release()
    if source_fps <= 1e-6:
        return 30.0
    return float(source_fps)


def backfill_pose_file(
    pose_path: Path,
    sampled_idx: int,
    source_idx: int,
    source_fps: float,
    sample_every: int,
    overwrite_metadata: bool,
) -> None:
    """Write inferred frame metadata into one pose npz."""
    with np.load(pose_path, allow_pickle=False) as pose_data:
        payload = {key: pose_data[key] for key in pose_data.files}
    if not overwrite_metadata and "source_frame_index" in payload and "state_frame_name" in payload:
        return
    payload["sampled_frame_index"] = np.int64(sampled_idx)
    payload["source_frame_index"] = np.int64(source_idx)
    payload["source_fps"] = np.float32(source_fps)
    payload["sample_every"] = np.int64(sample_every)
    payload["source_timestamp_sec"] = np.float32(source_idx / source_fps)
    payload["state_frame_name"] = np.array(f"frame_{sampled_idx:06d}.jpg")
    np.savez(pose_path, **payload)


def save_frames_for_sampled_indices(
    video_path: Path,
    frame_dir: Path,
    sampled_indices: list[int],
    sample_every: int,
    overwrite_frames: bool,
) -> int:
    """Decode and save sampled state frames for given indices."""
    frame_dir.mkdir(parents=True, exist_ok=True)
    source_to_sampled = {sampled_idx * sample_every: sampled_idx for sampled_idx in sampled_indices}
    if not source_to_sampled:
        return 0
    max_source_idx = max(source_to_sampled.keys())

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return 0
    saved = 0
    source_idx = 0
    while source_idx <= max_source_idx:
        success, frame = capture.read()
        if not success:
            break
        sampled_idx = source_to_sampled.get(source_idx)
        if sampled_idx is not None:
            frame_path = frame_dir / f"frame_{sampled_idx:06d}.jpg"
            if overwrite_frames or not frame_path.exists():
                cv2.imwrite(str(frame_path), frame)
                saved += 1
        source_idx += 1
    capture.release()
    return saved


def save_pose_debug_overlay(state_frame_path: Path, pose_path: Path, debug_path: Path) -> bool:
    """Save one pose-derived overlay image for pairing verification."""
    image = cv2.imread(str(state_frame_path), cv2.IMREAD_COLOR)
    if image is None:
        return False
    with np.load(pose_path, allow_pickle=False) as pose_data:
        if "pred_keypoints_2d" not in pose_data.files:
            return False
        keypoints_2d = np.asarray(pose_data["pred_keypoints_2d"], dtype=np.float32)
    if keypoints_2d.ndim != 2 or keypoints_2d.shape[1] != 2:
        return False

    h, w = image.shape[:2]
    valid_mask = (
        np.isfinite(keypoints_2d[:, 0])
        & np.isfinite(keypoints_2d[:, 1])
        & (keypoints_2d[:, 0] >= 0)
        & (keypoints_2d[:, 0] < w)
        & (keypoints_2d[:, 1] >= 0)
        & (keypoints_2d[:, 1] < h)
    )
    valid_points = keypoints_2d[valid_mask]
    if valid_points.shape[0] < 3:
        return False

    points_int = np.round(valid_points).astype(np.int32).reshape(-1, 1, 2)
    overlay = image.copy()
    hull = cv2.convexHull(points_int)
    cv2.fillConvexPoly(overlay, hull, color=(70, 170, 245))
    blended = cv2.addWeighted(overlay, 0.25, image, 0.75, 0.0)

    full_points = np.round(keypoints_2d).astype(np.int32)
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20),
    ]
    for start, end in edges:
        if valid_mask[start] and valid_mask[end]:
            cv2.line(blended, tuple(full_points[start]), tuple(full_points[end]), (0, 255, 0), 2, cv2.LINE_AA)
    for idx, point in enumerate(full_points):
        if valid_mask[idx]:
            cv2.circle(blended, tuple(point), 3, (0, 0, 255), -1, cv2.LINE_AA)

    debug_path.parent.mkdir(parents=True, exist_ok=True)
    return cv2.imwrite(str(debug_path), blended)


def main() -> int:
    """Backfill pose metadata and state frames into existing outputs."""
    args = parse_args()
    if not args.data_root.exists():
        raise FileNotFoundError(f"Data root not found: {args.data_root}")
    if not args.action_root.exists():
        raise FileNotFoundError(f"Action root not found: {args.action_root}")
    args.work_root.mkdir(parents=True, exist_ok=True)

    loader = TarVideoLoader(data_root=args.data_root, work_root=args.work_root)
    pose_dirs = sorted({path.parent for path in args.action_root.rglob("*_pose.npz")})
    if not pose_dirs:
        print(f"No pose files found under {args.action_root}")
        return 1

    processed_videos = 0
    skipped_videos = 0
    saved_debug_images = 0

    for video_out_dir in tqdm(pose_dirs, desc="videos", unit="video"):
        rel = video_out_dir.relative_to(args.action_root)
        if len(rel.parts) < 2:
            skipped_videos += 1
            continue

        tar_stem = rel.parts[-2]
        video_stem = rel.parts[-1]
        tar_rel_parent = Path(*rel.parts[:-2]) if len(rel.parts) > 2 else Path(".")
        tar_path = args.data_root / tar_rel_parent / f"{tar_stem}.tar"
        if not tar_path.exists():
            skipped_videos += 1
            continue

        sampled_indices = infer_sampled_indices(video_out_dir)
        if not sampled_indices:
            skipped_videos += 1
            continue

        temp_work_dir = args.work_root / "_metadata_backfill_tmp" / tar_rel_parent / tar_stem / video_stem
        temp_work_dir.mkdir(parents=True, exist_ok=True)

        with loader.open_shard(tar_path) as shard:
            members = [member for member in shard.video_members() if member.safe_stem == video_stem]
            if len(members) != 1:
                shutil.rmtree(temp_work_dir, ignore_errors=True)
                skipped_videos += 1
                continue
            extracted_video_path = shard.extract_video(members[0], temp_work_dir)

        source_fps = read_source_fps(extracted_video_path)
        sample_every = sample_every_for_fps(source_fps, args.fps)

        save_frames_for_sampled_indices(
            video_path=extracted_video_path,
            frame_dir=video_out_dir / args.frame_dir_name,
            sampled_indices=sampled_indices,
            sample_every=sample_every,
            overwrite_frames=args.overwrite_frames,
        )

        for pose_path in video_out_dir.glob("*_pose.npz"):
            match = POSE_FILE_PATTERN.match(pose_path.name)
            if match is None:
                continue
            sampled_idx = int(match.group(1))
            person_id = int(match.group(2))
            source_idx = sampled_idx * sample_every
            backfill_pose_file(
                pose_path=pose_path,
                sampled_idx=sampled_idx,
                source_idx=source_idx,
                source_fps=source_fps,
                sample_every=sample_every,
                overwrite_metadata=args.overwrite_metadata,
            )
            if args.save_debug_images and saved_debug_images < args.max_debug_images:
                state_frame_path = video_out_dir / args.frame_dir_name / f"frame_{sampled_idx:06d}.jpg"
                debug_file = (
                    video_out_dir
                    / args.debug_dir_name
                    / f"frame_{sampled_idx:06d}_{person_id}_debug.jpg"
                )
                if save_pose_debug_overlay(state_frame_path, pose_path, debug_file):
                    saved_debug_images += 1

        shutil.rmtree(temp_work_dir, ignore_errors=True)
        processed_videos += 1

    print(f"processed={processed_videos}, skipped={skipped_videos}, debug_images={saved_debug_images}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
