from __future__ import annotations

import subprocess
from pathlib import Path

import cv2


class HamerInferencePipeline:
    """Preprocessing and inference runner for HaMeR demo.py."""

    def __init__(
        self,
        *,
        python_bin: str,
        hamer_root: Path,
        checkpoint: Path | None,
        batch_size: int,
        body_detector: str,
        save_pose: bool,
        save_images: bool,
        fps: float,
        max_frames: int,
        max_seconds: float,
    ):
        self.python_bin = python_bin
        self.hamer_root = hamer_root
        self.checkpoint = checkpoint
        self.batch_size = batch_size
        self.body_detector = body_detector
        self.save_pose = save_pose
        self.save_images = save_images
        self.fps = fps
        self.max_frames = max_frames
        self.max_seconds = max_seconds

    def validate_assets(self) -> None:
        if self.checkpoint is None:
            raise FileNotFoundError("Checkpoint path must be provided")
        model_cfg = self.checkpoint.parent.parent / "model_config.yaml"
        mano_pkl = self.hamer_root / "_DATA/data/mano/MANO_RIGHT.pkl"
        vitpose_cfg_dir = self.hamer_root / "third-party/ViTPose/configs"
        vitpose_ckpt = self.hamer_root / "_DATA/vitpose_ckpts/vitpose+_huge/wholebody.pth"

        missing = []
        for path in [self.checkpoint, model_cfg, mano_pkl, vitpose_cfg_dir, vitpose_ckpt]:
            if not path.exists():
                missing.append(str(path))
        if missing:
            joined = "\n- ".join(missing)
            raise FileNotFoundError(
                "Missing required HaMeR assets:\n"
                f"- {joined}\n"
                "Run setup/download steps before starting batch inference."
            )

    def extract_frames(self, video_path: Path, frame_dir: Path) -> int:
        frame_dir.mkdir(parents=True, exist_ok=True)
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            return 0

        source_fps = capture.get(cv2.CAP_PROP_FPS)
        if source_fps <= 1e-6:
            source_fps = 30.0
        sample_every = 1 if self.fps <= 0 else max(int(round(source_fps / self.fps)), 1)
        max_source_frames = int(self.max_seconds * source_fps) if self.max_seconds > 0 else 0

        frame_index = 0
        saved = 0
        while True:
            if max_source_frames > 0 and frame_index >= max_source_frames:
                break
            success, frame = capture.read()
            if not success:
                break
            if frame_index % sample_every == 0:
                out_path = frame_dir / f"frame_{saved:06d}.jpg"
                cv2.imwrite(str(out_path), frame)
                saved += 1
                if self.max_frames > 0 and saved >= self.max_frames:
                    break
            frame_index += 1
        capture.release()
        return saved

    def run_on_frames(self, frame_dir: Path, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        command = [
            self.python_bin,
            "demo.py",
            "--img_folder",
            str(frame_dir),
            "--out_folder",
            str(out_dir),
            "--batch_size",
            str(self.batch_size),
            "--body_detector",
            self.body_detector,
            "--full_frame",
            "--save_mesh",
            "--file_type",
            "*.jpg",
        ]
        if self.save_pose:
            command.append("--save_pose")
        if self.save_images:
            command.append("--save_vis")
        if self.checkpoint is not None:
            command[2:2] = ["--checkpoint", str(self.checkpoint)]
        subprocess.run(command, cwd=str(self.hamer_root), check=True)

    def run_on_video(self, video_path: Path, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        command = [
            self.python_bin,
            "demo.py",
            "--video_path",
            str(video_path),
            "--video_fps",
            str(self.fps),
            "--video_max_frames",
            str(self.max_frames),
            "--video_max_seconds",
            str(self.max_seconds),
            "--out_folder",
            str(out_dir),
            "--batch_size",
            str(self.batch_size),
            "--body_detector",
            self.body_detector,
            "--full_frame",
            "--save_mesh",
        ]
        if self.save_pose:
            command.append("--save_pose")
        if self.save_images:
            command.append("--save_vis")
        if self.checkpoint is not None:
            command[2:2] = ["--checkpoint", str(self.checkpoint)]
        subprocess.run(command, cwd=str(self.hamer_root), check=True)

    def export_overlay_video(self, out_dir: Path) -> bool:
        overlay_frames = sorted(out_dir.glob("*_all.jpg"))
        if not overlay_frames:
            return False
        first_frame = cv2.imread(str(overlay_frames[0]))
        if first_frame is None:
            return False

        height, width = first_frame.shape[:2]
        output_video = out_dir / "overlay.mp4"
        output_fps = self.fps if self.fps > 0 else 30.0
        writer = cv2.VideoWriter(
            str(output_video),
            cv2.VideoWriter_fourcc(*"mp4v"),
            output_fps,
            (width, height),
        )
        if not writer.isOpened():
            return False

        try:
            for frame_path in overlay_frames:
                frame = cv2.imread(str(frame_path))
                if frame is None:
                    continue
                if frame.shape[:2] != (height, width):
                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                writer.write(frame)
        finally:
            writer.release()
        return output_video.exists()
