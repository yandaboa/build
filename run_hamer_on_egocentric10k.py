#!/usr/bin/env python3
"""Run HaMeR mesh extraction over videos stored in tar shards."""

import argparse
import shutil
import subprocess
from pathlib import Path

from tqdm import tqdm

from hamer_inference_pipeline import HamerInferencePipeline
from tar_video_loader import TarVideoLoader


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the batch pipeline."""
    parser = argparse.ArgumentParser(
        description="Extract videos from tar shards and run HaMeR to export hand meshes."
    )
    parser.add_argument("--data-root", type=Path, default=Path("/workspace/build/egocentric10k_data"))
    parser.add_argument("--hamer-root", type=Path, default=Path("/workspace/build/Hand-Texture-Module"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/workspace/build/Hand-Texture-Module/_DATA/hamer_ckpts/checkpoints/new_hamer_weights.ckpt"),
    )
    parser.add_argument("--output-root", type=Path, default=Path("/workspace/build/egocentric10k_hamer_meshes"))
    parser.add_argument("--work-root", type=Path, default=Path("/workspace/build/egocentric10k_hamer_work"))
    parser.add_argument("--python-bin", type=str, default="/workspace/build/.venv_hand/bin/python")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--body-detector", type=str, default="regnety", choices=["vitdet", "regnety"])
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--max-seconds", type=float, default=0.0)
    parser.add_argument("--limit-videos", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-work", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--export-overlay-video", action="store_true")
    parser.add_argument("--save-pose", action="store_true")
    parser.add_argument(
        "--save-pose-metadata",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save source/sampled frame metadata into each *_pose.npz file.",
    )
    parser.add_argument("--save-images", action="store_true")
    parser.add_argument(
        "--stream-video-inference",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run demo.py directly on video stream to avoid writing intermediate frame JPGs.",
    )
    return parser.parse_args()


def main() -> int:
    """Execute full tar-to-mesh batch processing with resume markers."""
    args = parse_args()

    if not args.data_root.exists():
        raise FileNotFoundError(f"Data root not found: {args.data_root}")
    if not args.hamer_root.exists():
        raise FileNotFoundError(f"HaMeR root not found: {args.hamer_root}")
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not Path(args.python_bin).exists():
        raise FileNotFoundError(f"Python binary not found: {args.python_bin}")

    if args.export_overlay_video and not args.save_images:
        print("Enabling --save-images because --export-overlay-video needs *_all.jpg frames.")
        args.save_images = True

    args.output_root.mkdir(parents=True, exist_ok=True)
    args.work_root.mkdir(parents=True, exist_ok=True)

    loader = TarVideoLoader(data_root=args.data_root, work_root=args.work_root)
    pipeline = HamerInferencePipeline(
        python_bin=args.python_bin,
        hamer_root=args.hamer_root,
        checkpoint=args.checkpoint,
        batch_size=args.batch_size,
        body_detector=args.body_detector,
        save_pose=args.save_pose,
        save_pose_metadata=args.save_pose_metadata,
        save_images=args.save_images,
        fps=args.fps,
        max_frames=args.max_frames,
        max_seconds=args.max_seconds,
    )
    pipeline.validate_assets()

    tar_paths = loader.list_tar_paths()
    if not tar_paths:
        print(f"No .tar files found under {args.data_root}")
        return 1

    processed_videos = 0
    skipped_videos = 0

    for tar_path in tar_paths:
        rel_tar_parent = tar_path.parent.relative_to(args.data_root)
        tar_tag = tar_path.stem
        print(f"[tar] {tar_path}")

        with loader.open_shard(tar_path) as shard:
            members = shard.video_members()
            for member in tqdm(members, desc=f"videos:{tar_tag}", unit="video"):
                video_out_dir = args.output_root / rel_tar_parent / tar_tag / member.safe_stem
                done_marker = video_out_dir / "_done.txt"
                has_existing_meshes = video_out_dir.exists() and any(video_out_dir.glob("*.obj"))
                if not args.overwrite and (done_marker.exists() or has_existing_meshes):
                    print(f"  ! warning: found existing processed data for {member.name}, skipping")
                    skipped_videos += 1
                    continue

                print(f"  [video] {member.name}")
                video_work_dir = args.work_root / rel_tar_parent / tar_tag / member.safe_stem
                extracted_video_path = shard.extract_video(member, video_work_dir)

                try:
                    if args.stream_video_inference:
                        pipeline.run_on_video(extracted_video_path, video_out_dir)
                    else:
                        frame_dir = video_work_dir / "frames"
                        frame_count = pipeline.extract_frames(extracted_video_path, frame_dir)
                        if frame_count == 0:
                            print("    ! no frames decoded, skipping")
                            if not args.keep_work:
                                shutil.rmtree(video_work_dir, ignore_errors=True)
                            continue
                        pipeline.run_on_frames(frame_dir, video_out_dir)
                except subprocess.CalledProcessError as exc:
                    print(f"    ! HaMeR failed for {member.name}: {exc}")
                    if not args.continue_on_error:
                        print("Stopping because --continue-on-error was not set.")
                        return 2
                    continue

                done_marker.write_text("ok\n", encoding="utf-8")
                if args.export_overlay_video and not pipeline.export_overlay_video(video_out_dir):
                    print("    ! overlay video was not created (no overlay frames or writer failed)")
                processed_videos += 1

                if not args.keep_work:
                    shutil.rmtree(video_work_dir, ignore_errors=True)

                if args.limit_videos > 0 and processed_videos >= args.limit_videos:
                    print(f"Reached --limit-videos={args.limit_videos}, stopping early.")
                    print(f"processed={processed_videos}, skipped={skipped_videos}, output={args.output_root}")
                    return 0

    print(f"processed={processed_videos}, skipped={skipped_videos}, output={args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
