# HaMeR Setup (Concise)

This sets up inference for `build/Hand-Texture-Module` in this workspace.

## 1) Initialize repo + submodule

```bash
cd /workspace/build
git init
git submodule add https://github.com/gkarv/Hand-Texture-Module.git Hand-Texture-Module
```

If the submodule already exists:

```bash
git submodule update --init --recursive
```

## 2) Create environment

```bash
cd /workspace/build
python3 -m venv .venv_hand
source .venv_hand/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

## 3) Install dependencies

```bash
# PyTorch (CPU build)
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio

# Core libs
pip install -r requirements-hand-inference.txt

# Extra packages required by this repo
pip install --no-build-isolation mmcv==1.3.9
pip install --no-build-isolation \
  "chumpy @ git+https://github.com/mattloper/chumpy" \
  "detectron2 @ git+https://github.com/facebookresearch/detectron2" \
  "git+https://github.com/facebookresearch/pytorch3d.git"
```

## 4) Download gated assets and place them

Create dirs:

```bash
mkdir -p /workspace/build/Hand-Texture-Module/_DATA/data/mano
mkdir -p /workspace/build/Hand-Texture-Module/_DATA/hamer_ckpts/checkpoints
```

Manual downloads:

- `MANO_RIGHT.pkl` from [MANO](https://mano.is.tue.mpg.de)
- HaMeR checkpoint (example used here): `new_hamer_weights.ckpt`

Move them:

```bash
mv /workspace/build/MANO_RIGHT.pkl \
  /workspace/build/Hand-Texture-Module/_DATA/data/mano/MANO_RIGHT.pkl

mv /workspace/build/new_hamer_weights.ckpt \
  /workspace/build/Hand-Texture-Module/_DATA/hamer_ckpts/checkpoints/new_hamer_weights.ckpt
```

## 5) Verify required files

```bash
ls -lh /workspace/build/Hand-Texture-Module/_DATA/data/mano/MANO_RIGHT.pkl
ls -lh /workspace/build/Hand-Texture-Module/_DATA/hamer_ckpts/model_config.yaml
ls -lh /workspace/build/Hand-Texture-Module/_DATA/vitpose_ckpts/vitpose+_huge/wholebody.pth
ls -lh /workspace/build/Hand-Texture-Module/_DATA/hamer_ckpts/checkpoints/new_hamer_weights.ckpt
```

## 6) Run inference

```bash
cd /workspace/build
source .venv_hand/bin/activate

python Hand-Texture-Module/demo.py \
  --checkpoint Hand-Texture-Module/_DATA/hamer_ckpts/checkpoints/new_hamer_weights.ckpt \
  --img_folder /path/to/frames \
  --out_folder /workspace/build/hamer_out \
  --batch_size 8 \
  --full_frame \
  --save_mesh
```

