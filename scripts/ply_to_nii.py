#!/usr/bin/env python3
"""Map micro-HAI PLY results back onto the source HR-pQCT NIfTI grid.

The cortical meshes written by ``scripts/joint_segment_torch.py`` (``nii2mesh``)
are *not* in the NIfTI physical frame:

1. SimpleITK ``GetArrayFromImage`` is ``(z, y, x)`` and is transposed to
   ``(x, y, z)`` before meshing.
2. The binary joint volume is padded by 5 voxels on every side.
3. ``volume_to_smoothed_mesh`` builds a PyVista ``ImageData`` with
   ``origin=(0, 0, 0)`` and ``spacing=(1, 1, 1)``. Vertex coordinates are
   therefore padded voxel indices, not millimetres.

Abnormality clusters in ``scripts/vis_distance.py`` are subsets of mesh A
(the patient surface, e.g. ``*_up.ply``), so they live in the same frame.
Registered template meshes (``reg_*.ply``) are in *template* space and must
not be mapped with this convention.

The label volume copies size, spacing, origin and direction from the
reference NIfTI, so ITK-SNAP can load it as a segmentation on the CT.

Labels
------
0  background
1  cortical surface (first ``--cortex`` file, then 1, 2, ... if several)
N  abnormality regions start after the cortex labels (default: one shared
   label, or one label per file with ``--label-each``)

Examples
--------
python ply_to_nii.py \\
    --reference 538_2134_MCP_2.nii \\
    --cortex 538_2134_MCP_2_up.ply 538_2134_MCP_2_down.ply \\
    --abnormality cluster0.ply cluster1.ply \\
    --output 538_2134_MCP_2_microhai_labels.nii.gz

ITK-SNAP: File -> Open Main Image (the original .nii), then Segmentation ->
Open Segmentation (the label .nii.gz). Geometry is identical, so the overlay
is aligned without resampling.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from typing import Iterable, Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# PLY loading (ASCII + binary). Optional trimesh / open3d if present.
# ---------------------------------------------------------------------------

def _parse_ply_header(f):
    if f.read(4) != b"ply\n" and False:
        pass
    f.seek(0)
    first = f.readline()
    if first.strip() not in (b"ply", b"ply\r"):
        raise ValueError("Not a PLY file")
    fmt = None
    elements = []
    current = None
    while True:
        line = f.readline()
        if not line:
            raise ValueError("PLY header ended before end_header")
        text = line.decode("ascii", errors="replace").strip()
        if text.startswith("format "):
            fmt = text.split()[1]
        elif text.startswith("element "):
            parts = text.split()
            current = {"name": parts[1], "count": int(parts[2]), "props": []}
            elements.append(current)
        elif text.startswith("property "):
            parts = text.split()
            if parts[1] == "list":
                current["props"].append(
                    {"list": True, "count_type": parts[2], "item_type": parts[3], "name": parts[4]}
                )
            else:
                current["props"].append({"list": False, "type": parts[1], "name": parts[2]})
        elif text == "end_header":
            break
    if fmt is None:
        raise ValueError("PLY header missing format")
    return fmt, elements, f.tell()


_PLY_DTYPE = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "<i2",
    "int16": "<i2",
    "ushort": "<u2",
    "uint16": "<u2",
    "int": "<i4",
    "int32": "<i4",
    "uint": "<u4",
    "uint32": "<u4",
    "float": "<f4",
    "float32": "<f4",
    "double": "<f8",
    "float64": "<f8",
}


def _ply_scalar(type_name: str, endian: str) -> np.dtype:
    dt = np.dtype(_PLY_DTYPE[type_name])
    if endian == ">":
        dt = dt.newbyteorder(">")
    return dt


def load_ply(path: str):
    """Return (vertices [N,3] float64, faces [M,3] int64 or empty)."""
    try:
        import trimesh

        mesh = trimesh.load(path, process=False)
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int64) if hasattr(mesh, "faces") else np.zeros((0, 3), np.int64)
        return vertices, faces
    except Exception:
        pass

    with open(path, "rb") as f:
        fmt, elements, data_start = _parse_ply_header(f)
        f.seek(data_start)
        blob = f.read()

    if fmt == "ascii":
        text = blob.decode("ascii", errors="replace").splitlines()
        return _load_ply_ascii(text, elements)
    if fmt == "binary_little_endian":
        return _load_ply_binary(blob, elements, "<")
    if fmt == "binary_big_endian":
        return _load_ply_binary(blob, elements, ">")
    raise ValueError(f"Unsupported PLY format: {fmt}")


def _load_ply_ascii(lines, elements):
    cursor = 0
    vertices = None
    faces = np.zeros((0, 3), dtype=np.int64)
    for el in elements:
        rows = lines[cursor : cursor + el["count"]]
        cursor += el["count"]
        if el["name"] == "vertex":
            names = [p["name"] for p in el["props"]]
            cols = []
            for row in rows:
                cols.append([float(x) for x in row.split()[: len(names)]])
            arr = np.asarray(cols, dtype=np.float64)
            ix, iy, iz = names.index("x"), names.index("y"), names.index("z")
            vertices = arr[:, [ix, iy, iz]]
        elif el["name"] == "face":
            tri = []
            for row in rows:
                vals = [int(float(x)) for x in row.split()]
                n = vals[0]
                idx = vals[1 : 1 + n]
                if n == 3:
                    tri.append(idx)
                elif n > 3:
                    for k in range(1, n - 1):
                        tri.append([idx[0], idx[k], idx[k + 1]])
            if tri:
                faces = np.asarray(tri, dtype=np.int64)
    if vertices is None:
        raise ValueError("PLY has no vertex element")
    return vertices, faces


def _load_ply_binary(blob: bytes, elements, endian: str):
    offset = 0
    vertices = None
    faces = np.zeros((0, 3), dtype=np.int64)
    for el in elements:
        if el["name"] == "vertex" and all(not p["list"] for p in el["props"]):
            dtype = np.dtype([(p["name"], _ply_scalar(p["type"], endian)) for p in el["props"]])
            count = el["count"]
            arr = np.frombuffer(blob, dtype=dtype, count=count, offset=offset)
            offset += dtype.itemsize * count
            vertices = np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(np.float64)
            continue
        if el["name"] == "face":
            tri = []
            prop = el["props"][0]
            count_dt = _ply_scalar(prop["count_type"], endian)
            item_dt = _ply_scalar(prop["item_type"], endian)
            for _ in range(el["count"]):
                n = int(np.frombuffer(blob, dtype=count_dt, count=1, offset=offset)[0])
                offset += count_dt.itemsize
                idx = np.frombuffer(blob, dtype=item_dt, count=n, offset=offset).astype(np.int64)
                offset += item_dt.itemsize * n
                if n == 3:
                    tri.append(idx)
                elif n > 3:
                    for k in range(1, n - 1):
                        tri.append([idx[0], idx[k], idx[k + 1]])
            if tri:
                faces = np.asarray(tri, dtype=np.int64)
            continue
        raise ValueError(f"Cannot skip binary PLY element '{el['name']}'")
    if vertices is None:
        raise ValueError("PLY has no vertex element")
    return vertices, faces


# ---------------------------------------------------------------------------
# Coordinate mapping
# ---------------------------------------------------------------------------

def vertices_to_voxel_index(
    vertices: np.ndarray,
    *,
    space: str,
    pad: float,
    mesh_spacing: Sequence[float],
    mesh_origin: Sequence[float],
    image_spacing: Sequence[float],
    image_origin: Sequence[float],
    image_direction: np.ndarray,
) -> np.ndarray:
    """Map PLY vertices to continuous ITK voxel indices (x, y, z).

    space='pipeline'  padded index frame used by nii2mesh (default)
    space='voxel'     already unpadded (x, y, z) indices of the reference
    space='physical'  millimetres in the reference NIfTI physical space
    """
    v = np.asarray(vertices, dtype=np.float64)
    if v.ndim != 2 or v.shape[1] != 3:
        raise ValueError(f"vertices must be (N, 3), got {v.shape}")
    spacing = np.asarray(mesh_spacing, dtype=np.float64)
    origin = np.asarray(mesh_origin, dtype=np.float64)
    if space == "pipeline":
        return (v - origin) / spacing - float(pad)
    if space == "voxel":
        return (v - origin) / spacing
    if space == "physical":
        direction = np.asarray(image_direction, dtype=np.float64).reshape(3, 3)
        spacing_img = np.asarray(image_spacing, dtype=np.float64)
        origin_img = np.asarray(image_origin, dtype=np.float64)
        # physical = origin + direction @ (spacing * index)
        delta = v - origin_img
        scaled = np.linalg.solve(direction, delta.T).T
        return scaled / spacing_img
    raise ValueError(f"Unknown space '{space}'")


def _sample_triangles(vertices: np.ndarray, faces: np.ndarray, step: float) -> np.ndarray:
    """Dense samples on triangle faces, in the same frame as ``vertices``."""
    if faces.size == 0:
        return vertices
    tri = vertices[faces]  # (M, 3, 3)
    edges = [
        (tri[:, 1] - tri[:, 0], tri[:, 0]),
        (tri[:, 2] - tri[:, 1], tri[:, 1]),
        (tri[:, 0] - tri[:, 2], tri[:, 2]),
    ]
    chunks = [vertices]
    # Edge samples. Step is in the same unit as vertices (voxel indices).
    lengths = [np.linalg.norm(e, axis=1) for e, _ in edges]
    max_len = float(max(x.max() if x.size else 0.0 for x in lengths))
    n_edge = max(int(np.ceil(max_len / step)), 1)
    t = (np.arange(n_edge + 1, dtype=np.float64) / n_edge)[:, None]
    for direction, start in edges:
        pts = start[None, :, :] + t[:, None, :] * direction[None, :, :]
        chunks.append(pts.reshape(-1, 3))
    # Face samples on a barycentric lattice. Cap per-triangle samples.
    n_face = max(int(np.ceil(max_len / step)), 1)
    n_face = min(n_face, 24)
    bary = []
    for i in range(n_face + 1):
        for j in range(n_face + 1 - i):
            a = i / n_face
            b = j / n_face
            bary.append((a, b, 1.0 - a - b))
    bary = np.asarray(bary, dtype=np.float64)
    face_pts = (
        bary[:, 0, None, None] * tri[:, 0][None, :, :]
        + bary[:, 1, None, None] * tri[:, 1][None, :, :]
        + bary[:, 2, None, None] * tri[:, 2][None, :, :]
    )
    chunks.append(face_pts.reshape(-1, 3))
    return np.concatenate(chunks, axis=0)


def rasterize_indices(
    indices: np.ndarray,
    shape_xyz: Sequence[int],
    *,
    radius: float,
) -> np.ndarray:
    """Mark voxels. ``shape_xyz`` is (nx, ny, nz). Returns bool array (z, y, x)."""
    nx, ny, nz = (int(s) for s in shape_xyz)
    mask = np.zeros((nz, ny, nx), dtype=bool)
    if indices.size == 0:
        return mask
    r = float(radius)
    if r <= 0.0:
        rounded = np.rint(indices).astype(np.int64)
        ok = (
            (rounded[:, 0] >= 0)
            & (rounded[:, 0] < nx)
            & (rounded[:, 1] >= 0)
            & (rounded[:, 1] < ny)
            & (rounded[:, 2] >= 0)
            & (rounded[:, 2] < nz)
        )
        sel = rounded[ok]
        mask[sel[:, 2], sel[:, 1], sel[:, 0]] = True
        return mask

    rad = int(np.ceil(r))
    offsets = [
        (i, j, k)
        for i in range(-rad, rad + 1)
        for j in range(-rad, rad + 1)
        for k in range(-rad, rad + 1)
        if i * i + j * j + k * k <= r * r + 1e-8
    ]
    base = np.rint(indices).astype(np.int64)
    for di, dj, dk in offsets:
        pts = base + np.array([di, dj, dk], dtype=np.int64)
        ok = (
            (pts[:, 0] >= 0)
            & (pts[:, 0] < nx)
            & (pts[:, 1] >= 0)
            & (pts[:, 1] < ny)
            & (pts[:, 2] >= 0)
            & (pts[:, 2] < nz)
        )
        sel = pts[ok]
        if sel.size:
            mask[sel[:, 2], sel[:, 1], sel[:, 0]] = True
    return mask


def dilate6(mask: np.ndarray, iterations: int) -> np.ndarray:
    out = mask
    for _ in range(int(iterations)):
        grown = out.copy()
        grown[1:] |= out[:-1]
        grown[:-1] |= out[1:]
        grown[:, 1:] |= out[:, :-1]
        grown[:, :-1] |= out[:, 1:]
        grown[:, :, 1:] |= out[:, :, :-1]
        grown[:, :, :-1] |= out[:, :, 1:]
        out = grown
    return out


# ---------------------------------------------------------------------------
# NIfTI I/O. SimpleITK is preferred (same stack as the pipeline).
# ---------------------------------------------------------------------------

def read_reference(path: str):
    try:
        import SimpleITK as sitk

        image = sitk.ReadImage(path)
        array = sitk.GetArrayFromImage(image)
        meta = {
            "backend": "sitk",
            "image": image,
            "size_xyz": tuple(int(x) for x in image.GetSize()),
            "spacing": tuple(float(x) for x in image.GetSpacing()),
            "origin": tuple(float(x) for x in image.GetOrigin()),
            "direction": np.array(image.GetDirection(), dtype=np.float64).reshape(3, 3),
            "array_shape_zyx": tuple(int(x) for x in array.shape),
        }
        return meta
    except ImportError:
        pass
    import nibabel as nib

    img = nib.load(path)
    shape = tuple(int(x) for x in img.shape[:3])
    aff = img.affine
    # column k of the 3x3 affine is spacing_k * direction_k
    direction = np.zeros((3, 3), dtype=np.float64)
    spacing = []
    for k in range(3):
        col = aff[:3, k]
        sp = float(np.linalg.norm(col))
        spacing.append(sp if sp > 0 else 1.0)
        direction[:, k] = col / spacing[-1]
    meta = {
        "backend": "nibabel",
        "image": img,
        "size_xyz": shape,
        "spacing": tuple(spacing),
        "origin": tuple(float(x) for x in aff[:3, 3]),
        "direction": direction,
        "array_shape_zyx": (shape[2], shape[1], shape[0]),
        "affine": aff,
    }
    return meta


def write_label(path: str, label_zyx: np.ndarray, meta: dict) -> None:
    label_zyx = np.ascontiguousarray(label_zyx)
    if meta["backend"] == "sitk":
        import SimpleITK as sitk

        out = sitk.GetImageFromArray(label_zyx)
        out.CopyInformation(meta["image"])
        # Integer labels. ITK-SNAP treats this as a segmentation overlay.
        out = sitk.Cast(out, sitk.sitkUInt16 if label_zyx.max() > 255 else sitk.sitkUInt8)
        sitk.WriteImage(out, path, useCompression=True)
        return
    import nibabel as nib

    # nibabel array is (x, y, z); our mask is (z, y, x)
    data = np.transpose(label_zyx, (2, 1, 0))
    dtype = np.uint16 if int(data.max()) > 255 else np.uint8
    img = nib.Nifti1Image(data.astype(dtype), meta["affine"])
    img.header.set_xyzt_units("mm")
    img.header["scl_slope"] = 1
    img.header["scl_inter"] = 0
    img.header.set_intent("label", name="micro-HAI labels")
    nib.save(img, path)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def write_points_ply(path: str, points: np.ndarray) -> None:
    """Write an ASCII point-cloud PLY. Coordinates are left unchanged."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {pts.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("end_header\n")
        for p in pts:
            f.write(f"{p[0]:.8g} {p[1]:.8g} {p[2]:.8g}\n")


def export_cluster_plys(cluster_point_sets, out_dir: str, prefix: str = "abnormality") -> list:
    """Save abnormality clusters as PLY in the same frame as the patient surface.

    ``cluster_point_sets`` is a list of (N, 3) arrays, typically
    ``cluster_stats[i]['points_A_cluster']`` from ``vis_distance.py``.
    Those points are subsets of mesh A (*_up.ply), so ``--space pipeline`` applies.
    Do not export ``corresp_points_B`` or ``reg_*.ply`` vertices.
    """
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for i, pts in enumerate(cluster_point_sets):
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
        if pts.shape[0] == 0:
            continue
        path = os.path.join(out_dir, f"{prefix}_{i:02d}.ply")
        write_points_ply(path, pts)
        paths.append(path)
    return paths


def _load_jsw(output: str, cortex: Sequence[str]) -> dict | None:
    """JSW written by nii2mesh as <prefix>_jsw.json next to the cortical PLY."""
    candidates = []
    out_dir = os.path.dirname(os.path.abspath(output))
    stem = os.path.basename(output)
    for suffix in ("_microhai_labels.nii.gz", "_microhai_labels.nii", ".nii.gz", ".nii"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    candidates.append(os.path.join(out_dir, stem + "_jsw.json"))
    for path in cortex:
        candidates.append(os.path.splitext(path)[0].replace("_up", "").replace("_down", "") + "_jsw.json")
        candidates.append(os.path.join(os.path.dirname(os.path.abspath(path)), stem + "_jsw.json"))
    for path in candidates:
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as f:
                return json.load(f)
    return None


def _load_abnormality_kinds(paths: Sequence[str]) -> dict:
    kinds = {}
    dirs = {os.path.dirname(os.path.abspath(p)) for p in paths}
    for folder in dirs:
        if not folder or not os.path.isdir(folder):
            continue
        for name in os.listdir(folder):
            if not name.endswith("_kinds.json"):
                continue
            try:
                with open(os.path.join(folder, name), encoding="utf-8") as f:
                    records = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue
            for rec in records:
                kinds[os.path.abspath(rec.get("path", ""))] = rec
    return kinds
    kinds = {}
    dirs = {os.path.dirname(os.path.abspath(p)) for p in paths}
    for folder in dirs:
        if not folder or not os.path.isdir(folder):
            continue
        for name in os.listdir(folder):
            if not name.endswith("_kinds.json"):
                continue
            try:
                with open(os.path.join(folder, name), encoding="utf-8") as f:
                    records = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue
            for rec in records:
                kinds[os.path.abspath(rec.get("path", ""))] = rec
    return kinds


def write_itksnap_label_descriptions(path: str, objects: list) -> None:
    """ITK-SNAP label description file.

    Cortex is red and green. Each abnormality gets a distinct high-contrast
    color so it stays visible on the black ITK-SNAP background.

    Alpha must be 0.00-1.00. Writing 255 makes ITK-SNAP drop the labels
    and the surface mesh on import.
    """
    lines = [
        "################################################",
        "# ITK-SnAP Label Description File",
        "# File format:",
        "# IDX   -R-  -G-  -B-  -A--  VIS MSH  LABEL",
        "# Fields:",
        "#    IDX:   Zero-based index",
        "#    -R-:   Red color component (0..255)",
        "#    -G-:   Green color component (0..255)",
        "#    -B-:   Blue color component (0..255)",
        "#    -A-:   Label transparency (0.00 .. 1.00)",
        "#    VIS:   Label visibility (0 or 1)",
        "#    MSH:   Label mesh visibility (0 or 1)",
        "#  LABEL:   Label description",
        "################################################",
        '     0     0    0    0        0  0  0    "Clear Label"',
    ]
    cortex_colors = [(255, 40, 40), (40, 220, 40)]
    abnormal_colors = [
        (40, 140, 255),
        (255, 200, 0),
        (180, 40, 255),
        (0, 230, 230),
        (255, 120, 0),
        (255, 80, 180),
        (140, 255, 0),
        (80, 80, 255),
        (255, 255, 80),
        (0, 180, 255),
        (255, 0, 120),
        (160, 255, 200),
    ]
    cortex_i = 0
    abnormal_i = 0
    for item in objects:
        label = int(item["label"])
        kind = item.get("abnormality_kind") or item["kind"]
        if item["kind"] == "cortex":
            color = cortex_colors[cortex_i % len(cortex_colors)]
            cortex_i += 1
            name = f"cortex {label}"
        else:
            color = abnormal_colors[abnormal_i % len(abnormal_colors)]
            abnormal_i += 1
            name = f"{kind} {label}" if kind else f"abnormality {label}"
        lines.append(
            '{:>6} {:>5} {:>4} {:>4}     1.00  1  1    "{}"'.format(
                label, color[0], color[1], color[2], name
            )
        )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _require_files(paths: Sequence[str], flag: str) -> None:
    missing = [p for p in paths if not os.path.isfile(p)]
    if not missing:
        return
    cwd = os.getcwd()
    nearby = sorted(
        name for name in os.listdir(cwd) if name.lower().endswith(".ply")
    )[:30]
    raise FileNotFoundError(
        f"{flag} file(s) not found: {missing}\n"
        f"cwd: {cwd}\n"
        f"PLY files in cwd: {nearby or '(none)'}\n"
        "cluster0.ply / cluster1.ply in the example command are placeholders. "
        "vis_distance.py does not write abnormality PLY files. "
        "Omit --abnormality to map the cortical surfaces only, or export "
        "points_A_cluster with export_cluster_plys() and pass those paths."
    )


def map_ply_to_label(
    reference: str,
    cortex: Sequence[str],
    abnormality: Sequence[str],
    output: str,
    *,
    space: str = "pipeline",
    pad: float = 5.0,
    mesh_spacing: Sequence[float] = (1.0, 1.0, 1.0),
    mesh_origin: Sequence[float] = (0.0, 0.0, 0.0),
    sample_step: float = 0.5,
    point_radius: float = 1.0,
    thickness: int = 0,
    label_each: bool = False,
    cortex_label0: int = 1,
) -> dict:
    meta = read_reference(reference)
    nx, ny, nz = meta["size_xyz"]
    if meta["array_shape_zyx"] != (nz, ny, nx):
        raise RuntimeError(
            f"Reference array shape {meta['array_shape_zyx']} != (z,y,x)={(nz, ny, nx)}"
        )
    label = np.zeros((nz, ny, nx), dtype=np.uint16)
    jsw = _load_jsw(output, cortex)
    report = {
        "reference": os.path.abspath(reference),
        "jsw": jsw,
        "output": os.path.abspath(output),
        "space": space,
        "pad": pad,
        "size_xyz": [nx, ny, nz],
        "spacing": list(meta["spacing"]),
        "origin": list(meta["origin"]),
        "direction": meta["direction"].reshape(-1).tolist(),
        "objects": [],
    }
    next_label = int(cortex_label0)
    kinds = _load_abnormality_kinds(abnormality)

    def _paint(path: str, value: int, kind: str) -> None:
        vertices, faces = load_ply(path)
        index = vertices_to_voxel_index(
            vertices,
            space=space,
            pad=pad,
            mesh_spacing=mesh_spacing,
            mesh_origin=mesh_origin,
            image_spacing=meta["spacing"],
            image_origin=meta["origin"],
            image_direction=meta["direction"],
        )
        inside = (
            (index[:, 0] >= -0.5)
            & (index[:, 0] < nx - 0.5)
            & (index[:, 1] >= -0.5)
            & (index[:, 1] < ny - 0.5)
            & (index[:, 2] >= -0.5)
            & (index[:, 2] < nz - 0.5)
        )
        if faces.size and kind == "cortex":
            samples = _sample_triangles(index, faces, sample_step)
        else:
            samples = index
        info = kinds.get(os.path.abspath(path), {})
        ab_kind = info.get("kind", "")
        radius = 0.0 if (faces.size and kind == "cortex") else point_radius
        if kind == "abnormality" and ab_kind in ("erosion", "proliferation"):
            radius = max(radius, 0.7)
        mask = rasterize_indices(samples, (nx, ny, nz), radius=radius)
        if thickness > 0:
            mask = dilate6(mask, thickness)
        label[mask] = value
        n_in = int(inside.sum())
        report["objects"].append(
            {
                "path": os.path.abspath(path),
                "kind": kind,
                "label": int(value),
                "n_vertices": int(vertices.shape[0]),
                "n_faces": int(faces.shape[0]),
                "vertices_inside_frac": float(n_in / max(vertices.shape[0], 1)),
                "vertex_min": vertices.min(axis=0).tolist(),
                "vertex_max": vertices.max(axis=0).tolist(),
                "n_voxels": int(mask.sum()),
                "abnormality_kind": ab_kind,
                "volume_mm3": info.get("volume_mm3"),
                "filled_volume_mm3": info.get("filled_volume_mm3"),
                "shape_index": info.get("shape_index"),
                "shape_index_healthy": info.get("shape_index_healthy"),
                "relative_change_of_shape_index": info.get("relative_change_of_shape_index"),
                "over_flat": info.get("over_flat"),
                "voxel_size_mm": info.get("voxel_size_mm"),
            }
        )
        print(
            f"[{kind}] {os.path.basename(path)} -> label {value}"
            f"{(' ' + ab_kind) if ab_kind else ''}: "
            f"{vertices.shape[0]} verts, {faces.shape[0]} faces, "
            f"{n_in / max(vertices.shape[0], 1):.1%} inside volume, "
            f"{int(mask.sum())} voxels"
        )
        if ab_kind:
            vol = info.get("volume_mm3")
            rho = info.get("shape_index")
            rel = info.get("relative_change_of_shape_index")
            print(
                "  volume {} mm^3   shape index {}   relative change {}".format(
                    "nan" if vol is None else f"{vol:.4f}",
                    "nan" if rho is None else f"{rho:.3f}",
                    "nan" if rel is None else f"{rel:.3f}",
                )
            )
        if n_in == 0:
            print(
                "  WARNING: no vertex landed in the reference volume. "
                "Check --pad (pipeline default 5) and --space. "
                "reg_*.ply is in template space and cannot use --space pipeline."
            )

    for path in cortex:
        _paint(path, next_label, "cortex")
        next_label += 1
    if abnormality:
        if label_each:
            for path in abnormality:
                _paint(path, next_label, "abnormality")
                next_label += 1
        else:
            ab_label = next_label
            for path in abnormality:
                _paint(path, ab_label, "abnormality")
            next_label += 1

    os.makedirs(os.path.dirname(os.path.abspath(output)) or ".", exist_ok=True)
    write_label(output, label, meta)
    sidecar = os.path.splitext(output)[0]
    if sidecar.endswith(".nii"):
        sidecar = sidecar[:-4]
    sidecar = sidecar + "_labels.json"
    report["label_names"] = {
        str(item["label"]): (
            f"{item.get('abnormality_kind') or item['kind']}:{os.path.basename(item['path'])}"
        )
        for item in report["objects"]
    }
    with open(sidecar, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    desc_path = sidecar.replace("_labels.json", "_itksnap.txt")
    write_itksnap_label_descriptions(desc_path, report["objects"])
    print(f"Wrote {output}")
    print(f"Wrote {sidecar}")
    print(f"Wrote {desc_path}")
    print(
        "ITK-SNAP: open the reference NIfTI as the main image, then "
        "Segmentation -> Open Segmentation and choose the label file."
    )
    return report


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Rasterize micro-HAI PLY surfaces/point clouds into a NIfTI label map aligned to the source CT."
    )
    p.add_argument("--reference", required=True, help="Original MCP .nii/.nii.gz used by nii2mesh")
    p.add_argument("--cortex", nargs="*", default=[], help="Cortical surface PLY files (*_up.ply, *_down.ply)")
    p.add_argument("--abnormality", nargs="*", default=[], help="Abnormality region PLY point clouds or meshes")
    p.add_argument(
        "--abnormality-dir",
        default=None,
        help="Directory of abnormality PLYs. Every *.ply is mapped, each with its own label, sorted by name.",
    )
    p.add_argument("--output", required=True, help="Output label NIfTI (.nii or .nii.gz)")
    p.add_argument(
        "--space",
        choices=("pipeline", "voxel", "physical"),
        default="pipeline",
        help="pipeline: nii2mesh frame (pad + unit spacing). voxel: unpadded indices. physical: NIfTI mm.",
    )
    p.add_argument("--pad", type=float, default=5.0, help="Voxel pad applied before meshing (nii2mesh uses 5)")
    p.add_argument("--mesh-spacing", type=float, nargs=3, default=(1.0, 1.0, 1.0))
    p.add_argument("--mesh-origin", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    p.add_argument("--sample-step", type=float, default=0.5, help="Triangle sampling step in voxels")
    p.add_argument(
        "--point-radius",
        type=float,
        default=1.0,
        help="Splat radius in voxels for point clouds (abnormality PLY without faces)",
    )
    p.add_argument("--thickness", type=int, default=0, help="Extra 6-connected dilation in voxels")
    p.add_argument("--label-each", action="store_true", help="Give each abnormality PLY its own label")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argparser().parse_args(argv)
    abnormality = list(args.abnormality)
    label_each = args.label_each
    if args.abnormality_dir:
        if not os.path.isdir(args.abnormality_dir):
            print(f"--abnormality-dir is not a directory: {args.abnormality_dir}", file=sys.stderr)
            return 2
        found = sorted(
            os.path.join(args.abnormality_dir, name)
            for name in os.listdir(args.abnormality_dir)
            if name.lower().endswith(".ply")
        )
        if not found:
            print(f"No .ply files in {args.abnormality_dir}", file=sys.stderr)
            return 2
        abnormality.extend(found)
        label_each = True
        print(f"abnormality dir: {len(found)} files, one label each")
        for i, path in enumerate(found):
            print(f"  label {1 + len(args.cortex) + i}: {path}")
    if not args.cortex and not abnormality:
        print("Provide at least one of --cortex, --abnormality, or --abnormality-dir", file=sys.stderr)
        return 2
    _require_files([args.reference], "--reference")
    _require_files(args.cortex, "--cortex")
    _require_files(abnormality, "--abnormality")
    map_ply_to_label(
        args.reference,
        args.cortex,
        abnormality,
        args.output,
        space=args.space,
        pad=args.pad,
        mesh_spacing=args.mesh_spacing,
        mesh_origin=args.mesh_origin,
        sample_step=args.sample_step,
        point_radius=args.point_radius,
        thickness=args.thickness,
        label_each=label_each,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
