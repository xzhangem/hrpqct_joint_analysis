Reproducible setup for the `joint` Conda environment: **PyTorch 2.5.1 + CUDA 12.1**, **nnU-Net v2**, **PyTorch3D**, and **PyTorch Geometric**, plus 3D / medical imaging utilities.

> Do **not** install from a raw `pip freeze` dump. `torch`, `pytorch3d`, and the PyG CUDA extensions must come from their own wheel indexes.

## Requirements

- Linux x86_64
- NVIDIA GPU with a driver compatible with **CUDA 12.1**
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda
- Python **3.10**

CPU-only machines and macOS cannot reproduce this environment with the commands below.

## Installation

### 1. Create the environment

```bash
conda create -n joint python=3.10 -y
conda activate joint
python -m pip install -U pip setuptools wheel ninja
```

### 2. Install PyTorch (CUDA 12.1)

```bash
pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu121
```

Verify:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Expected output:

```text
2.5.1+cu121 True
```

### 3. Install PyTorch3D

```bash
pip install iopath fvcore
pip install --no-deps \
  --extra-index-url https://miropsota.github.io/torch_packages_builder \
  "pytorch3d==0.7.9+d9839a9pt2.5.1cu121"
```

This wheel is **not** on official PyPI. It is a prebuilt binary that must match `torch==2.5.1+cu121`.

If the third-party index is unavailable, build from source instead:

```bash
pip install iopath fvcore
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
```

### 4. Install PyTorch Geometric

```bash
pip install --no-deps "torch-geometric==2.6.1"
pip install --no-deps \
  pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
  -f https://data.pyg.org/whl/torch-2.5.0+cu121.html
```

Use `torch-2.5.0+cu121` in the PyG URL (the index name for the 2.5.x series). Do not change it to `2.5.1`.

### 5. Install nnU-Net v2 and remaining packages

```bash
pip install nnunetv2
pip install trimesh edt numba pyvista "fastmorph[spherical]" open3d potpourri3d
pip install numpy scipy tqdm jinja2 pyparsing psutil requests aiohttp fsspec xxhash
```

`nnunetv2` will pull in `batchgenerators`, `batchgeneratorsv2`, `dynamic_network_architectures`, and `acvl_utils`. You do not need to install those by hand.

### 6. Verify

```bash
python -c "import torch, pytorch3d, torch_geometric, nnunetv2, open3d, pyvista; print('ok')"
```

## One-shot script

```bash
bash scripts/install.sh
```

## Notes

- Install **PyTorch before** PyTorch3D, PyG extensions, and nnU-Net.
- Pinning every transitive dependency (`pip freeze`) will break installs on other machines. Keep only the commands above.
- `open3d` / `pyvista` / `vtk` may fail on unusual system libraries. Treat them as optional if they are not required for your entry script.
- Recreating the env later:

  ```bash
  conda deactivate
  conda env remove -n npj
  bash scripts/install.sh
  ```

## Stack

| Component | Version / source |
| --- | --- |
| Python | 3.10 |
| PyTorch / torchvision / torchaudio | 2.5.1 + cu121 ([PyTorch wheel index](https://download.pytorch.org/whl/cu121)) |
| PyTorch3D | `0.7.9+d9839a9pt2.5.1cu121` |
| PyTorch Geometric | 2.6.1 + `torch-2.5.0+cu121` extensions |
| nnU-Net | `nnunetv2` |
