#!/usr/bin/env python3
"""Export abnormality clusters as PLY in the patient-surface frame.

vis_distance.py only visualizes clusters; it does not write
abnormality_ply/*.ply. This script does that step, using the same
mesh-A / mesh-B comparison:

  mesh A = patient cortical surface, e.g. 538_2134_MCP_2_up.ply
  mesh B = registered template,     e.g. reg_538_2134_MCP_2_up.ply

Cluster points are subsets of mesh A, so ply_to_nii.py --space pipeline
can map them. Do not pass reg_*.ply itself as an abnormality cloud.

Example
-------
python export_abnormality_ply.py \
  --mesh-a 538_2134_MCP_2_up.ply \
  --mesh-b reg_538_2134_MCP_2_up.ply \
  --out-dir abnormality_ply \
  --prefix 538_2134_MCP_2_up

python ply_to_nii.py \
  --reference 538_2134_MCP_2.nii \
  --cortex 538_2134_MCP_2_up.ply 538_2134_MCP_2_down.ply \
  --abnormality abnormality_ply/538_2134_MCP_2_up_00.ply \
  --output 538_2134_MCP_2_microhai_labels.nii.gz
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

from ply_to_nii import write_points_ply


def signed_distance_to_mesh(points: np.ndarray, mesh_b) -> np.ndarray:
    """Signed distance from points to mesh B. Requires Open3D."""
    import open3d as o3d

    mesh_b_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh_b)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh_b_t)
    query = o3d.core.Tensor(np.asarray(points, dtype=np.float32))
    return scene.compute_signed_distance(query).numpy().reshape(-1)


def cluster_large_deviations(
    points: np.ndarray,
    signed_dists: np.ndarray,
    percentile: float = 75.0,
    min_points: int = 20,
):
    """DBSCAN on points whose |signed distance| exceeds the percentile.

    eps is 1.5 times the mean nearest-neighbor distance, matching
    vis_distance.py. Clusters smaller than min_points are dropped
    (the paper filter uses 20).
    """
    import open3d as o3d

    abs_d = np.abs(signed_dists)
    threshold = float(np.percentile(abs_d, percentile))
    mask = abs_d > threshold
    pts = points[mask]
    print(f"deviation percentile {percentile:.1f}% -> threshold {threshold:.4f}")
    print(f"large-deviation points: {pts.shape[0]} / {points.shape[0]}")
    if pts.shape[0] < min_points:
        print("too few large-deviation points; no clusters written")
        return []

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    tree = o3d.geometry.KDTreeFlann(pcd)
    nn = []
    for i in range(len(pcd.points)):
        _, _, dist2 = tree.search_knn_vector_3d(pcd.points[i], 2)
        if len(dist2) >= 2:
            nn.append(np.sqrt(dist2[1]))
    eps = float(np.mean(nn) * 1.5) if nn else 1.0
    print(f"DBSCAN eps = {eps:.4f} (1.5 x mean nearest-neighbor)")
    labels = np.asarray(pcd.cluster_dbscan(eps=eps, min_points=5, print_progress=False))
    clusters = []
    n_clusters = int(labels.max()) + 1 if labels.size and labels.max() >= 0 else 0
    for cid in range(n_clusters):
        sel = pts[labels == cid]
        if sel.shape[0] < min_points:
            continue
        clusters.append(sel)
    clusters.sort(key=lambda x: x.shape[0], reverse=True)
    print(f"clusters kept (>={min_points} points): {len(clusters)}")
    for i, sel in enumerate(clusters):
        print(f"  {i:02d}: {sel.shape[0]} points")
    return clusters


VOXEL_SIZE_MM = 0.082  # HR-pQCT isotropic resolution, 82 um


def convex_hull_volume_mm3(points: np.ndarray) -> float | None:
    """Volume of the cluster convex hull, converted at 82 um. Paper vol."""
    from scipy.spatial import ConvexHull

    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if pts.shape[0] < 4:
        return None
    try:
        hull = ConvexHull(pts)
    except Exception:
        return None
    return float(hull.volume) * VOXEL_SIZE_MM ** 3


def lowner_shape_index(points: np.ndarray, tol: float = 1e-4, max_iter: int = 200) -> float | None:
    """Shape index rho = longest / shortest semi-axis of the Loewner-John ellipsoid.

    Khachiyan minimum-volume enclosing ellipsoid, matching the paper.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    pts = pts[np.isfinite(pts).all(axis=1)]
    n = pts.shape[0]
    if n < 4:
        return None
    q = np.concatenate([pts, np.ones((n, 1))], axis=1)
    u = np.full(n, 1.0 / n)
    err = 1.0
    for _ in range(max_iter):
        x = q.T * u
        try:
            m = np.linalg.inv(x @ q)
        except np.linalg.LinAlgError:
            return None
        score = np.einsum("ij,jk,ik->i", q, m, q)
        j = int(np.argmax(score))
        step = (score[j] - 4.0) / (4.0 * (score[j] - 1.0))
        new_u = (1.0 - step) * u
        new_u[j] += step
        err = np.linalg.norm(new_u - u)
        u = new_u
        if err <= tol:
            break
    center = u @ pts
    cov = (pts - center).T * u @ (pts - center)
    try:
        eigvals = np.linalg.eigvalsh(cov)
    except np.linalg.LinAlgError:
        return None
    eigvals = eigvals[eigvals > 1e-12]
    if eigvals.size < 2:
        return None
    semi = np.sqrt(eigvals)
    return float(semi.max() / semi.min())


def closest_on_mesh(points: np.ndarray, mesh) -> np.ndarray:
    import open3d as o3d

    mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh_t)
    query = o3d.core.Tensor(np.ascontiguousarray(points, dtype=np.float32))
    info = scene.compute_closest_points(query)
    return np.asarray(info["points"].numpy(), dtype=np.float64).reshape(-1, 3)


def cluster_metrics(patient_pts: np.ndarray, template_pts: np.ndarray, n_voxels: int) -> dict:
    """Paper metrics on the cluster points, not on the filled CT volume.

    vol is the convex-hull volume. rho is the Loewner longest/shortest axis.
    r = max((rho_h - rho_d) / rho_d, 0). Regions with rho_d > 10 are over-flat.
    """
    rho_d = lowner_shape_index(patient_pts)
    rho_h = lowner_shape_index(template_pts)
    if rho_d not in (None, 0.0) and rho_h is not None:
        relative = float(max((rho_h - rho_d) / rho_d, 0.0))
    else:
        relative = None
    return {
        "voxel_size_mm": VOXEL_SIZE_MM,
        "volume_mm3": convex_hull_volume_mm3(patient_pts),
        "filled_volume_mm3": float(n_voxels) * VOXEL_SIZE_MM ** 3,
        "shape_index": rho_d,
        "shape_index_healthy": rho_h,
        "relative_change_of_shape_index": relative,
        "over_flat": bool(rho_d is not None and rho_d > 10.0),
    }


def classify_cluster(signed_dists: np.ndarray) -> str:
    """Open3D signed distance is negative inside the template mesh.

    Patient points inside the healthy template are recessed bone: erosion.
    Patient points outside the template stick out: proliferation.
    """
    return "erosion" if float(np.mean(signed_dists)) < 0.0 else "proliferation"


def points_in_convex_hull(points: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    from scipy.spatial import Delaunay

    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    candidates = np.asarray(candidates, dtype=np.float64).reshape(-1, 3)
    if points.shape[0] < 4:
        return np.ones(candidates.shape[0], dtype=bool)
    try:
        hull = Delaunay(points)
    except Exception:
        return np.ones(candidates.shape[0], dtype=bool)
    return hull.find_simplex(candidates) >= 0


def fill_abnormal_volume(cluster_pts: np.ndarray, mesh_a, mesh_b, kind: str, margin: float = 2.0) -> np.ndarray:
    """Voxel centers of the abnormal volume inside the cluster hull.

    erosion: missing bone, inside the healthy template and outside the patient bone.
    proliferation: extra bone, outside the template and inside the patient bone.
    Coordinates are the pipeline voxel-index frame of the cortical PLY.
    """
    cluster_pts = np.asarray(cluster_pts, dtype=np.float64).reshape(-1, 3)
    lo = np.floor(cluster_pts.min(axis=0) - margin).astype(int)
    hi = np.ceil(cluster_pts.max(axis=0) + margin).astype(int)
    axes = [np.arange(lo[i], hi[i] + 1) for i in range(3)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3).astype(np.float64)
    inside = points_in_convex_hull(cluster_pts, grid)
    candidates = grid[inside]
    if candidates.shape[0] == 0:
        return cluster_pts
    sd_patient = signed_distance_to_mesh(candidates, mesh_a)
    sd_template = signed_distance_to_mesh(candidates, mesh_b)
    if kind == "erosion":
        keep = (sd_template < 0.0) & (sd_patient > 0.0)
    else:
        keep = (sd_template > 0.0) & (sd_patient < 0.0)
    filled = candidates[keep]
    return filled if filled.shape[0] else cluster_pts


def export_from_meshes(mesh_a: str, mesh_b: str, out_dir: str, prefix: str, percentile: float, min_points: int):
    import open3d as o3d

    mesh_A = o3d.io.read_triangle_mesh(mesh_a)
    mesh_B = o3d.io.read_triangle_mesh(mesh_b)
    if len(mesh_A.vertices) == 0 or len(mesh_B.triangles) == 0:
        raise RuntimeError("mesh A needs vertices and mesh B needs faces for signed distance")
    points = np.asarray(mesh_A.vertices, dtype=np.float64)
    signed = signed_distance_to_mesh(points, mesh_B)
    clusters = cluster_large_deviations(points, signed, percentile=percentile, min_points=min_points)

    os.makedirs(out_dir, exist_ok=True)
    paths = []
    records = []
    for i, sel in enumerate(clusters):
        tree_pts = points
        # signed distance of the cluster points, matched back to the full surface
        from scipy.spatial import cKDTree

        dist, idx = cKDTree(tree_pts).query(sel, k=1)
        cluster_signed = signed[idx]
        kind = classify_cluster(cluster_signed)
        out_pts = fill_abnormal_volume(sel, mesh_A, mesh_B, kind)
        template_pts = closest_on_mesh(sel, mesh_B)
        metrics = cluster_metrics(sel, template_pts, out_pts.shape[0])
        path = os.path.join(out_dir, f"{prefix}_{i:02d}.ply")
        write_points_ply(path, out_pts)
        paths.append(path)
        records.append(
            {
                "path": os.path.abspath(path),
                "kind": kind,
                "mean_signed_distance": float(np.mean(cluster_signed)),
                "n_surface_points": int(sel.shape[0]),
                "n_written_points": int(out_pts.shape[0]),
                **metrics,
            }
        )
        rel = metrics["relative_change_of_shape_index"]
        print(
            f"wrote {path}  {kind}  "
            f"mean signed {records[-1]['mean_signed_distance']:.3f}  "
            f"{records[-1]['n_surface_points']} surface -> {records[-1]['n_written_points']} voxels"
        )
        print(
            "  vol {vol} mm^3   rho_d {rho}   rho_h {rho_h}   r {rel}{flat}".format(
                vol="nan" if metrics["volume_mm3"] is None else f"{metrics['volume_mm3']:.4f}",
                rho="nan" if metrics["shape_index"] is None else f"{metrics['shape_index']:.3f}",
                rho_h="nan" if metrics["shape_index_healthy"] is None else f"{metrics['shape_index_healthy']:.3f}",
                rel="nan" if metrics["relative_change_of_shape_index"] is None else f"{metrics['relative_change_of_shape_index']:.3f}",
                flat="   over-flat" if metrics["over_flat"] else "",
            )
        )
    sidecar = os.path.join(out_dir, prefix + "_kinds.json")
    with open(sidecar, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    if not paths:
        print("no abnormality PLY written")
    return paths


def main() -> int:
    p = argparse.ArgumentParser(description="Write abnormality cluster PLYs from mesh A vs registered mesh B.")
    p.add_argument("--mesh-a", required=True, help="Patient surface, e.g. 538_2134_MCP_2_up.ply")
    p.add_argument("--mesh-b", required=True, help="Registered template, e.g. reg_538_2134_MCP_2_up.ply")
    p.add_argument("--out-dir", default="abnormality_ply")
    p.add_argument("--prefix", default="abnormality")
    p.add_argument("--percentile", type=float, default=75.0, help="vis_distance.py hardcoded default is 75")
    p.add_argument("--min-points", type=int, default=20)
    args = p.parse_args()
    export_from_meshes(args.mesh_a, args.mesh_b, args.out_dir, args.prefix, args.percentile, args.min_points)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
