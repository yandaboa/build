# ACTION_EXTRACT

This document describes:
- the input/output format for `build/run_hamer_on_egocentric10k.py`
- commands to run the pipeline
- how extracted hand information maps to robot action-space usage

## 1) Pipeline Input Format

The runner expects tar shards under a data root (default: `/workspace/build/egocentric10k_data`).

### Expected directory structure

The script recursively scans for `*.tar` files under `--data-root`.

Example:

```text
/workspace/build/egocentric10k_data/
  factory_001/
    workers/
      worker_001/
        factory001_worker001_part00.tar
        factory001_worker001_part01.tar
```

### Expected tar contents

Each tar should contain video files (`.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`, `.m4v`).
If JSON/meta files are present, they are ignored by this script.

Example tar members:

```text
factory001_worker001_00000.mp4
factory001_worker001_00000.json
factory001_worker001_00001.mp4
factory001_worker001_00001.json
```

## 2) Pipeline Output Format

For each input video inside each tar, outputs are written to:

```text
{output_root}/{relative_tar_parent}/{tar_stem}/{video_stem}/
```

With defaults:
- `output_root`: `/workspace/build/egocentric10k_hamer_meshes`
- `work_root`: `/workspace/build/egocentric10k_hamer_work`

### Per-video artifacts

- `_done.txt`  
  Marker file for resume/skip logic.

- `frame_XXXXXX_<hand_idx>.obj`  
  Hand mesh per detected hand in that sampled frame.

- `frame_XXXXXX_<hand_idx>.png`  
  Patch + rendered hand visualization.

- `frame_XXXXXX_all.jpg`  
  Full-frame overlay visualization.

- `overlay.mp4` (if `--export-overlay-video`)  
  Video assembled from `*_all.jpg`.

- `frame_XXXXXX_<hand_idx>_pose.npz` (if `--save-pose`)  
  Pose and camera outputs per detected hand.

- `state_frames/frame_XXXXXX.jpg` (after running frame export/backfill utilities)  
  Observation image paired to that sampled frame index.

## 3) Pose NPZ Data Format

Each `*_pose.npz` contains:

- `global_orient`: shape `(1, 3, 3)`  
  Wrist/root orientation rotation matrix.

- `hand_pose`: shape `(15, 3, 3)`  
  MANO finger-joint rotation matrices (15 joints).

- `betas`: shape `(10,)`  
  MANO shape parameters (identity/shape), not motion.

- `pred_keypoints_3d`: shape `(21, 3)`  
  3D hand keypoints/joints.

- `pred_keypoints_2d`: shape `(21, 2)`  
  2D image keypoints.

- `pred_cam`: shape `(3,)`  
  Internal weak-perspective camera parameters.

- `pred_cam_t_full`: shape `(3,)`  
  Full-frame camera translation estimate.

- `is_right`: scalar-like value  
  Handedness flag.

When `--save-pose-metadata` is enabled (default in `run_hamer_on_egocentric10k.py`), pose files also include:

- `sampled_frame_index`: scalar int  
  Sampled frame number used in output naming (`frame_XXXXXX`).

- `source_frame_index`: scalar int  
  Original source-video frame index before FPS sampling.

- `source_fps`: scalar float  
  FPS reported by the video decoder.

- `sample_every`: scalar int  
  Sampling stride used (`source_frame_index % sample_every == 0`).

- `source_timestamp_sec`: scalar float  
  `source_frame_index / source_fps`.

- `state_frame_name`: scalar string  
  Expected paired observation filename (typically `frame_XXXXXX.jpg`).

## 4) Required Assets

Current setup expects:

- checkpoint  
  `/workspace/build/Hand-Texture-Module/_DATA/hamer_ckpts/checkpoints/new_hamer_weights.ckpt`

- model config  
  `/workspace/build/Hand-Texture-Module/_DATA/hamer_ckpts/model_config.yaml`

- ViTPose weights  
  `/workspace/build/Hand-Texture-Module/_DATA/vitpose_ckpts/vitpose+_huge/wholebody.pth`

- MANO model  
  `/workspace/build/Hand-Texture-Module/_DATA/data/mano/MANO_RIGHT.pkl`

## 5) Commands

### Full run (recommended)

```bash
/workspace/build/.venv_hand/bin/python /workspace/build/run_hamer_on_egocentric10k.py \
  --data-root /workspace/build/egocentric10k_data \
  --output-root /workspace/build/egocentric10k_hamer_meshes \
  --work-root /workspace/build/egocentric10k_hamer_work \
  --body-detector regnety \
  --fps 5 \
  --batch-size 8 \
  --continue-on-error \
  --export-overlay-video \
  --save-pose \
  --save-pose-metadata
```

### Small smoke test

```bash
/workspace/build/.venv_hand/bin/python /workspace/build/run_hamer_on_egocentric10k.py \
  --limit-videos 1 \
  --max-frames 16 \
  --fps 2 \
  --batch-size 1 \
  --body-detector regnety \
  --overwrite \
  --export-overlay-video \
  --save-pose \
  --save-pose-metadata \
  --output-root /workspace/build/egocentric10k_hamer_meshes_test2 \
  --work-root /workspace/build/egocentric10k_hamer_work_test2
```

### Inspect a pose file quickly

```bash
/workspace/build/.venv_hand/bin/python - <<'PY'
import numpy as np
p = "/workspace/build/egocentric10k_hamer_meshes_test2/factory_001/workers/worker_001/factory001_worker001_part00/factory001_worker001_00000/frame_000000_0_pose.npz"
d = np.load(p)
print(d.files)
print("global_orient", d["global_orient"].shape)
print("hand_pose", d["hand_pose"].shape)
print("betas", d["betas"].shape)
print("pred_keypoints_3d", d["pred_keypoints_3d"].shape)
print("state_frame_name", d.get("state_frame_name", "<missing>"))
PY
```

### Save paired observation frames (without re-running HaMeR)

Use this when pose files already exist and you want `state_frames/frame_XXXXXX.jpg` next to them.

```bash
/workspace/build/.venv_hand/bin/python /workspace/build/save_hamer_action_frames.py \
  --data-root /workspace/build/egocentric10k_data \
  --action-root /workspace/build/egocentric10k_hamer_meshes \
  --work-root /workspace/build/egocentric10k_hamer_work \
  --fps 5
```

### Backfill metadata + paired frames for old pose files

Use this for older outputs that were extracted before metadata existed.

```bash
/workspace/build/.venv_hand/bin/python /workspace/build/backfill_hamer_pose_metadata.py \
  --data-root /workspace/build/egocentric10k_data \
  --action-root /workspace/build/egocentric10k_hamer_meshes \
  --work-root /workspace/build/egocentric10k_hamer_work \
  --fps 5 \
  --save-debug-images \
  --max-debug-images 5
```

Debug outputs are saved under `debug_overlays/` (max 5 by default), showing pose-derived overlays on paired state images for quick consistency checks.

### One-off cleanup for duplicate detections in a directory

If a directory was re-generated repeatedly and contains too many hand indices per frame, keep only index `0` and `1`:

```bash
python /workspace/build/prune_extra_hand_detections.py --dry-run
python /workspace/build/prune_extra_hand_detections.py
```

Default target is:
`/workspace/build/egocentric10k_hamer_meshes/factory_001/workers/worker_001/factory001_worker001_part00/factory001_worker001_00000`

## 6) Robot Action Space vs Extracted Signals

This is the practical mapping from extracted terms to robot control relevance.

- `global_orient`  
  **Use for robot action?** Often yes.  
  Good wrist/end-effector orientation target (after frame alignment).

- `hand_pose`  
  **Use for robot action?** Yes for dexterous hands.  
  Requires retargeting from MANO joints to robot joint DoFs/limits.

- `betas`  
  **Use for robot action?** No.  
  Shape/identity term, not time-varying action.

- `pred_keypoints_3d`  
  **Use for robot action?** Indirectly yes.  
  Useful as IK or imitation targets; usually not direct low-level action.

- `pred_keypoints_2d`  
  **Use for robot action?** Usually no.  
  Primarily visual/perception/debug signal.

- `pred_cam`, `pred_cam_t_full`  
  **Use for robot action?** Usually no direct action use.  
  Useful for camera/world alignment and reprojection consistency.

- `is_right`  
  **Use for robot action?** Metadata only.  
  Useful for selecting left/right retargeting profile.

## 7) Recommended Action-Space Design by Robot Type

- Arm + simple gripper:
  - Use wrist target pose (orientation from `global_orient`, position from keypoint/mesh pipeline).
  - Derive one scalar gripper command (open/close).

- Dexterous hand robot:
  - Use `hand_pose` + `global_orient` as imitation targets.
  - Solve retargeting to robot joints each frame.

- Policy-learning observations:
  - Use `pred_keypoints_3d` or retargeted joint targets as observations.
  - Output robot-native commands (joint position/velocity/torque), not raw MANO params directly.

## 8) Training Dataset Format (State/Action Pairing)

`build/egocentric10k_action_dataset.py` now returns frame-level samples:

- `state`  
  RGB tensor from `state_frames/frame_XXXXXX.jpg`.

- `action`  
  Concatenated tensor as:
  `[left_hand_action_dims, right_hand_action_dims]`

Per-hand action dims are flattened from configured keys (default: `global_orient`, `hand_pose`).

If only one hand exists for a frame:
- `is_right == 0`: fills left action, right action is zero-padded.
- `is_right == 1`: fills right action, left action is zero-padded.
