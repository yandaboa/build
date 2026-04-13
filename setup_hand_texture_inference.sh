#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="${ROOT_DIR}/.venv_hand"
SUBMODULE_DIR="${ROOT_DIR}/Hand-Texture-Module"
CHECKPOINT_DIR="${SUBMODULE_DIR}/_DATA/hamer_ckpts/checkpoints"
MANO_DIR="${SUBMODULE_DIR}/_DATA/data/mano"
TEXTURE_CKPT_URL="https://forthgr-my.sharepoint.com/:u:/g/personal/gkarv_ics_forth_gr/IQDIDDN3B6e9RZFraetL-wClAdBcbeMSbRHsjWORwZf5irM?download=1"

if [[ ! -d "${SUBMODULE_DIR}" ]]; then
  echo "Missing submodule directory: ${SUBMODULE_DIR}" >&2
  echo "Run: git submodule update --init --recursive" >&2
  exit 1
fi

python3 -m venv "${VENV_PATH}"
source "${VENV_PATH}/bin/activate"

python -m pip install --upgrade pip setuptools wheel
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio
pip install \
  gdown opencv-python pyrender pytorch-lightning scikit-image smplx==0.1.28 yacs \
  timm einops xtcocotools pandas hydra-core hydra-submitit-launcher hydra-colorlog \
  pyrootutils rich webdataset
pip install --no-build-isolation mmcv==1.3.9
pip install --no-build-isolation \
  "chumpy @ git+https://github.com/mattloper/chumpy" \
  "detectron2 @ git+https://github.com/facebookresearch/detectron2" \
  "git+https://github.com/facebookresearch/pytorch3d.git"

mkdir -p "${CHECKPOINT_DIR}" "${MANO_DIR}"

(
  cd "${SUBMODULE_DIR}"
  bash fetch_demo_data.sh
)

CKPT_PATH="${CHECKPOINT_DIR}/texture_supervised_hamer_weights.ckpt"
if curl -L "${TEXTURE_CKPT_URL}" -o "${CKPT_PATH}"; then
  if [[ ! -s "${CKPT_PATH}" ]] || [[ "$(wc -c < "${CKPT_PATH}")" -lt 1048576 ]]; then
    rm -f "${CKPT_PATH}"
    echo "Checkpoint download appears blocked. Please download manually from SharePoint."
  fi
fi

cat <<'EOF'
Setup complete.

Important manual step:
- Download MANO_RIGHT.pkl from https://mano.is.tue.mpg.de
- Place it at: Hand-Texture-Module/_DATA/data/mano/MANO_RIGHT.pkl

Example run:
source .venv_hand/bin/activate
python Hand-Texture-Module/demo.py \
  --checkpoint Hand-Texture-Module/_DATA/hamer_ckpts/checkpoints/texture_supervised_hamer_weights.ckpt \
  --img_folder /path/to/your/images \
  --out_folder /path/to/output \
  --batch_size 8 \
  --full_frame
EOF
