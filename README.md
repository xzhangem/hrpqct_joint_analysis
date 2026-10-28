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

### One-shot script

```bash
bash scripts/install.sh
```


### Stack

| Component | Version / source |
| --- | --- |
| Python | 3.10 |
| PyTorch / torchvision / torchaudio | 2.5.1 + cu121 ([PyTorch wheel index](https://download.pytorch.org/whl/cu121)) |
| PyTorch3D | `0.7.9+d9839a9pt2.5.1cu121` |
| PyTorch Geometric | 2.6.1 + `torch-2.5.0+cu121` extensions |
| nnU-Net | `nnunetv2` |

# Run

The analysis entry point is `run_microhai.py`. It expects the source scripts
(`joint_segment_torch.py`, `gesm_match_adv.py`, `export_abnormality_ply.py`,
`ply_to_nii.py`) in the same directory, plus:

- `nnUNet_results/Dataset101_JointCT/nnUNetTrainer__nnUNetPlans__3d_fullres`
- healthy templates `mean_mcp2.ply` and `mean_mcp3.ply`

The joint is chosen from the file name. `MCP_2` and `MCP2` use
`mean_mcp2.ply`; `MCP_3` and `MCP3` use `mean_mcp3.ply`. 

## One volume

```bash
conda activate joint
cd /path/to/src
python run_microhai.py xxx_MCP_2.nii
```

Optional arguments:

```bash
python run_microhai.py xxx_MCP_2.nii \
  --out-dir results/xxx_MCP_2 \
  --template-dir . \
  --percentile 75 \
  --min-points 20
```

`--percentile` is the signed-distance threshold used to keep candidate
abnormal points. `--min-points` drops clusters smaller than this size
(the paper uses 20).

## A folder of volumes

Put the NIfTI files in `joint_nii_file`, next to `batch_run_microhai.py`.

```bash
python batch_run_microhai.py
python batch_run_microhai.py --skip-existing
python batch_run_microhai.py --input joint_nii_file --output joint_nii_results
```

Each volume is written to its own folder under `joint_nii_results/<stem>/`.
A failed case is logged and the batch continues. `--skip-existing` skips a
case whose `*_microhai_labels.nii.gz` is already present. Per-case time is
printed and stored in `joint_nii_results/batch_log.csv`.

# Results

For an input `538_2134_MCP_2.nii` the case folder contains:

| File | What it is |
|---|---|
| `xxx_MCP_2_up.ply` | Proximal cortical surface |
| `xxx_MCP_2_down.ply` | Distal cortical surface |
| `xxx_MCP_2_up.ply` | Healthy template registered to the proximal surface |
| `xxx_MCP_2_jsw.json` | Joint-space metrics from the segmentation step |
| `abnormality_ply/` | One PLY per abnormality cluster, plus `*_kinds.json` |
| `xxx_MCP_2_microhai_labels.nii.gz` | Label volume aligned to the source CT |
| `xxx_MCP_2_microhai_labels_labels.json` | Record of every label and the JSW metrics |
| `xxx_MCP_2_microhai_labels_itksnap.txt` | ITK-SNAP colours and names |

Cortical PLY files are in the padded voxel-index frame used by meshing
(transpose to x, y, z, pad 5, origin 0, spacing 1), not in NIfTI millimetres.
`reg_*.ply` is in the same patient-surface frame and is the registered
template, not an abnormality cloud.

## Labels

In `*_microhai_labels.nii.gz`:

- 1: proximal cortex
- 2: distal cortex
- 3 and above: one label per abnormality cluster

An abnormality is the volume between the patient surface and the registered
healthy template, restricted to the cluster. Erosion is the missing bone
(inside the template, outside the patient bone). Proliferation is the extra
bone (outside the template, inside the patient bone).

## ITK-SNAP

1. **File → Open Image** and open the source CT, for example `538_2134_MCP_2.nii`.
2. **Segmentation → Open Segmentation** and choose `*_microhai_labels.nii.gz`.
   Do not add it as another image.
3. **Segmentation → Import Label Descriptions** and choose `*_itksnap.txt`.
4. Update the 3D mesh if it is not already shown.

Cortex 1 is red and cortex 2 is green. Each abnormality has its own
high-contrast colour. The transparency column in the description file is
`1.00`; a value of `255` makes ITK-SNAP drop the labels on import.

## Metrics

`*_microhai_labels_labels.json` starts with the source path and the JSW
block:

```json
{
  "reference": ".../xxx_MCP_2.nii",
  "jsw": {
    "JSV": ---,
    "JSW_Mean": ---,
    "JSW_Min": ---,
    "JSW_Max": ---,
    "JSW_SD": ---,
    "JSW_Asymm": ---,
    "n_voxels": ---
  }
}
```

JSW follows the SPECTRA consensus local-thickness definition and is reported
in millimetres at the HR-pQCT resolution used for that run (0.082 mm here).
`JSV` is the joint-space volume in mm³. `JSW_Asymm` is max / min.

Each abnormality object in the same file also stores:

- `volume_mm3`: convex-hull volume of the cluster, in mm³ at 82 µm
- `shape_index`: Löwner–John shape index ρ_d of the patient cluster
- `shape_index_healthy`: the same index ρ_h on the closest template points
- `relative_change_of_shape_index`: r = max((ρ_h − ρ_d) / ρ_d, 0)
- `over_flat`: true when ρ_d > 10, which the paper treats as not a lesion

These three quantities are computed on the cluster points, not on the filled
CT volume. The same values are printed while the case runs.



