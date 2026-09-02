#!/usr/bin/env bash
set -euo pipefail

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found. Install Miniconda/Anaconda first." >&2
  exit 1
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

conda create -n joint python=3.10 -y
conda activate joint

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install nnunetv2
pip install -U pip setuptools wheel ninja
pip install iopath fvcore
pip install -U iopath fvcore ninja
pip install --no-deps --extra-index-url https://miropsota.github.io/torch_packages_builder  "pytorch3d==0.7.9+d9839a9pt2.5.1cu121"
pip install trimesh
pip install edt
pip install numba
pip install pyvista
pip install "fastmorph[spherical]"
pip install open3d
pip install potpourri3d
pip install numpy scipy tqdm jinja2 pyparsing psutil requests aiohttp fsspec xxhash
pip install --no-deps "torch-geometric==2.6.1"
pip install --no-deps pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv  -f https://data.pyg.org/whl/torch-2.5.0+cu121.html

echo "Done. Activate with: conda activate joint"
