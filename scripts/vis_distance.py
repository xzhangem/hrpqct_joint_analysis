"""
vis_distance.py

Wrap the entire abnormal_gui.py script into a function.
Only mesh_A_filename / mesh_B_filename are inputs; everything else uses the hardcoded defaults from abnormal_gui.py.
Logic, thresholds, clustering, and visualization are identical to abnormal_gui.py; no additional changes.
"""

import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering


def trimesh_to_open3d_mesh(tm):
    """
    Convert a trimesh.Trimesh to an open3d.geometry.TriangleMesh.
    Supports vertices, faces, normals, and vertex colors (if present).
    """
    vertices = np.asarray(tm.vertices, dtype=np.float64)
    faces = np.asarray(tm.faces, dtype=np.int32)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    return mesh


def khachiyan(a, tol):
    """
    Approximate Löwner ellipsoid of a centrally symmetric set.
    Corresponds to the khachiyan function in MATLAB.

    Parameters:
        a:   (n, m) array, input points (as column vectors)
        tol: positive number, tolerance (controls relative volume error)

    Returns:
        E:   ellipsoid matrix such that all points satisfy dot(a_i, E @ a_i) <= 1
    """
    n, m = a.shape
    if n < 2:
        raise ValueError("Input must be in two dimensions or higher.")
    if not np.all(np.isreal(a)):
        raise ValueError("Inputs must be real.")
    if not (np.isreal(tol) and tol > 0):
        raise ValueError("Tolerance must be positive.")

    # Initialization
    invA = n * np.linalg.inv(a @ a.T)
    w = np.sum(a * (invA @ a), axis=0)   # dot(a, invA @ a, 1)

    iter_count = 0
    max_iter = 10000  # prevent infinite loop

    while True:
        iter_count += 1
        if iter_count > max_iter:
            print("Warning: reached max iteration in khachiyan")
            break

        # Find the index and value of the maximum w
        r = np.argmax(w)
        w_r = w[r]

        f = w_r / n
        epsilon = f - 1.0

        if epsilon <= tol:
            break

        g = epsilon / ((n - 1) * f)
        h = 1 + g
        g = g / f

        # Update invA
        b = invA @ a[:, r]
        invA = h * invA - g * np.outer(b, b)

        # Update w
        bTa = b @ a
        w = h * w - g * (bTa * bTa)

    # Final ellipsoid matrix (scale once to account for numerical error)
    E = invA / w_r

    # Force all points to be covered (handle floating-point error)
    dots = np.sum(a * (E @ a), axis=0)
    max_dot = np.max(dots)
    if max_dot > 1.0:
        E = E / max_dot

    return E


def lowner(a, tol=1e-3):
    """
    Approximate Löwner ellipsoid of a general point set.
    Corresponds to the lowner function in MATLAB.

    Parameters:
        a:   (n, m) array, point cloud (each column is a point, n=dimension, m=number of points)
        tol: tolerance

    Returns:
        E:   ellipsoid matrix (n×n)
        c:   ellipsoid center (n,)
             such that for all points x, (x - c).T @ E @ (x - c) <= 1
    """
    n, m = a.shape
    if n < 1:
        raise ValueError("Input must be in one dimension or higher.")

    # Lift dimension: append a row of 1s to construct a centrally symmetric version
    aug = np.vstack([a, np.ones(m)])

    # Call khachiyan on the lifted ellipsoid
    F = khachiyan(aug, tol)

    # Take the top-left n×n block and the last column
    A = F[:n, :n]
    b = F[:n, -1]

    # Solve for center c = -A⁻¹ b
    try:
        c = -np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        raise RuntimeError("Singular matrix in lowner: cannot solve for center.")

    # Compute E
    E = A / (1 - c @ b - F[-1, -1])

    # Force all points to be covered again (most robust approach)
    ac = a - c[:, np.newaxis]
    dots = np.sum(ac * (E @ ac), axis=0)
    max_dot = np.max(dots)
    if max_dot > 1.0 + 1e-10:
        E = E / max_dot

    return E, c


def compute_min_vol_ellipsoid_metrics(points, tol=1e-4):
    """
    Compute the minimum-volume enclosing ellipsoid (Löwner-John ellipsoid)
    for an input 3D or 2D point cloud and return the aspect ratio and ellipsoid volume.

    Parameters:
        points : np.ndarray, shape (N, d) or (d, N)
                 point cloud coordinates, d is usually 2 or 3
        tol    : float, tolerance (default 1e-4)

    Returns:
        dict containing the following keys:
        - 'aspect_ratio' : float, longest semi-axis / shortest semi-axis (np.nan on failure)
        - 'volume'       : float, ellipsoid volume (np.nan on failure)
        - 'center'       : (d,) array, ellipsoid center
        - 'semi_axes'    : array, lengths of semi-axes (sorted)
        - 'E'            : (d,d) array, ellipsoid matrix
        - 'success'      : bool, whether the computation succeeded
    """
    # Unify to (d, N) format
    points = np.asarray(points)
    if points.ndim != 2:
        raise ValueError("points must be a 2D array")

    if points.shape[0] > points.shape[1]:
        points = points.T  # convert to (d, N)

    d, N = points.shape

    if N < d + 3:
        return {
            'aspect_ratio': np.nan,
            'volume': np.nan,
            'center': np.full(d, np.nan),
            'semi_axes': np.full(d, np.nan),
            'E': None,
            'success': False
        }

    try:
        # Call lowner directly (no convex hull)
        E, c = lowner(points, tol=tol)

        # Compute eigenvalues → semi-axis lengths = 1 / sqrt(eigenvalues)
        eigvals = np.linalg.eigvalsh(E)
        valid = eigvals > 1e-12
        if not np.any(valid):
            raise ValueError("No valid eigenvalues")

        semi_axes = 1.0 / np.sqrt(eigvals[valid])
        semi_axes = np.sort(semi_axes)

        aspect_ratio = semi_axes[-1] / semi_axes[0] if len(semi_axes) >= 2 else np.nan

        # Compute d-dimensional ellipsoid volume
        # Volume = (π^{d/2} / Γ(d/2 + 1)) * ∏(semi-axis lengths)
        if d == 2:
            volume = np.pi * np.prod(semi_axes)
        elif d == 3:
            volume = (4/3) * np.pi * np.prod(semi_axes)
        else:
            from scipy.special import gamma
            vol_factor = np.pi**(d/2) / gamma(d/2 + 1)
            volume = vol_factor * np.prod(semi_axes)

    except Exception as e:
        print(f"Computation failed: {str(e)}")
        return {
            'aspect_ratio': np.nan,
            'volume': np.nan,
            'center': np.full(d, np.nan),
            'semi_axes': np.full(d, np.nan),
            'E': None,
            'success': False
        }

    return {
        'aspect_ratio': aspect_ratio,
        'volume': volume,
        'center': c,
        'semi_axes': semi_axes,
        'E': E,
        'success': True
    }


def visualize_registration_deviation(mesh_A_filename, mesh_B_filename):
    """
    Function wrapper for the entire abnormal_gui.py script.

    Parameters:
        mesh_A_filename: query mesh path (.ply), corresponds to mesh_A in the original script
        mesh_B_filename: reference mesh path (.ply), corresponds to mesh_B in the original script

    All other settings use the hardcoded defaults from abnormal_gui.py:
        percentile_threshold = 75.0
        eps_dbscan = avg_nn * 1.5 (fallback to 25.0 if unavailable)
        min_points = 5
        Always perform DBSCAN visualization, per-cluster GUI, and signed-distance coloring
    """
    # ====================== 1. Load models & compute signed distance ======================
    mesh_A = o3d.io.read_triangle_mesh(mesh_A_filename)
    mesh_B = o3d.io.read_triangle_mesh(mesh_B_filename)

    # Convert to tensor and build RaycastingScene
    mesh_B_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh_B)
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(mesh_B_t)

    points_A = np.asarray(mesh_A.vertices, dtype=np.float32)
    query = o3d.core.Tensor(points_A)
    signed_dists = scene.compute_signed_distance(query).numpy().flatten()   # positive and negative
    dists_abs = np.abs(signed_dists)

    print(f"Total query points: {len(points_A)}")
    print(f"Max absolute deviation: {dists_abs.max():.4f}   Mean: {dists_abs.mean():.4f}")

    # ====================== 2. Automatically recommend percentile_threshold ======================
    def auto_recommend_percentile(dists_abs):
        """
        Automatically recommend a reasonable percentile based on the coefficient of variation (CV) of the deviation distribution
        - High CV (large variation in deviations) → stricter, take only the most extreme 2–3%
        - Low CV (more uniform deviations) → slightly looser, take 6–7%
        """
        if len(dists_abs) < 100:
            return 95.0

        mean_d = np.mean(dists_abs)
        std_d = np.std(dists_abs)
        cv = std_d / mean_d if mean_d > 0 else 0.0

        if cv > 2.0:
            target_frac = 0.02 * 3     # top 2%
        elif cv > 1.2:
            target_frac = 0.03 * 3     # top 3%
        elif cv > 0.6:
            target_frac = 0.05 * 3      # top 5% (most common)
        else:
            target_frac = 0.07 * 3      # top 7%

        p = 100 * (1 - target_frac)
        return np.clip(p, 90.0, 99.0)

    recommended_p = auto_recommend_percentile(dists_abs)
    print(f"\n=== Automatic recommendation ===")
    print(f"Based on coefficient of variation CV = {np.std(dists_abs)/np.mean(dists_abs):.3f}")
    print(f"Recommended percentile_threshold = {recommended_p:.1f}%  (most deviant {100-recommended_p:.1f}% points)")

    # ====================== 3. User can manually adjust (defaults to the automatic recommendation) ======================
    #percentile_threshold = recommended_p          # ←←← change manually here, e.g. to 96.5
    percentile_threshold = 75.0                 # example: force use of 95%

    d_threshold = np.percentile(dists_abs, percentile_threshold)
    print(f"Actual threshold used = {d_threshold:.4f} ({percentile_threshold}% percentile)")

    # ====================== 4. Filter large-deviation points ======================
    mask_large = dists_abs > d_threshold
    points_large = points_A[mask_large]
    signed_large = signed_dists[mask_large]

    n_large = len(points_large)
    print(f"Filtered {n_large} large-deviation points ({n_large/len(points_A)*100:.2f}% of total points)")

    if n_large < 30:
        print("Warning: too few large-deviation points; consider lowering percentile_threshold or checking registration quality")
        # Visualize original deviation coloring and exit
    else:
        # ====================== 5. DBSCAN region clustering ======================
        # === Adaptive eps_dbscan: use “average nearest-neighbor distance of each point in the cloud” instead of the manually set 25 ===
        # First create the point cloud (needed for later DBSCAN)
        pcd_large = o3d.geometry.PointCloud()
        pcd_large.points = o3d.utility.Vector3dVector(points_large)

        print("\n=== Adaptive eps_dbscan computation ===")
        # Use KDTree to compute nearest-neighbor distance of each point (k=2, excluding self)
        kdtree = o3d.geometry.KDTreeFlann(pcd_large)
        nn_dists = []
        for i in range(len(pcd_large.points)):
            _, _, d = kdtree.search_knn_vector_3d(pcd_large.points[i], 2)
            if len(d) >= 2:
                nn_dists.append(np.sqrt(d[1]))   # Note: Open3D KDTreeFlann returns squared distance, so take square root

        if nn_dists:
            avg_nn_dist = np.mean(nn_dists)
            eps_dbscan = avg_nn_dist * 1.5       # ←←← multiplier can be adjusted (recommended 2–4)
                                             # Reason: eps needs to be slightly larger than average point spacing for effective clustering
            print(f"Average nearest-neighbor distance of each point in the cloud: {avg_nn_dist:.4f}")
            print(f"Adaptive eps_dbscan = {eps_dbscan:.4f} (avg_nn * 3)")
        else:
            eps_dbscan = 25.0
            print("Unable to compute nearest-neighbor distance, falling back to default eps=25.0")

        min_points = 5       # minimum points per cluster (keep original value)

        labels = np.array(pcd_large.cluster_dbscan(
            eps=eps_dbscan,
            min_points=min_points,
            print_progress=True
        ))

        n_clusters = int(labels.max()) + 1 if labels.max() >= 0 else 0
        print(f"DBSCAN found {n_clusters} valid clusters (-1 is noise)")

        # ====================== 6. Compute convex-hull volume for each cluster ======================
        volumes = []
        cluster_stats = []

        for cid in range(n_clusters):
            mask = (labels == cid)
            pts = points_large[mask]
            if len(pts) < 4:
                continue

            pcd_cluster = o3d.geometry.PointCloud()
            pcd_cluster.points = o3d.utility.Vector3dVector(pts)

            try:
                hull, _ = pcd_cluster.compute_convex_hull(joggle_inputs=True)
                vol = hull.get_volume()
                query_t = o3d.core.Tensor(pts.astype(np.float32))
                closest_info = scene.compute_closest_points(query_t)
                points_B_corresp = closest_info['points'].numpy()

                metrics_B = compute_min_vol_ellipsoid_metrics(points_B_corresp)

                ellipsoid_metrics = compute_min_vol_ellipsoid_metrics(pts, tol=1e-4)
                volumes.append(vol)
                cluster_stats.append({
                    "cluster_id": cid,
                    "n_points": len(pts),
                    "volume": vol,
                    "ellipsoid_volume": ellipsoid_metrics['volume'],
                    "aspect_ratio": ellipsoid_metrics['aspect_ratio'],
                    "points_A_cluster": pts,
                    "corresp_points_B": points_B_corresp,
                    "corresp_ellipsoid_volume": metrics_B['volume'],
                    "corresp_aspect_ratio": metrics_B['aspect_ratio'],
                    "max_dev": np.max(np.abs(signed_large[mask])),
                    "mean_dev": np.mean(np.abs(signed_large[mask]))
                })
            except Exception as e:
                print(f"Cluster {cid} convex hull computation failed (degenerate geometry): {e}")

        # Output statistics
        if volumes:
            print("\n=== Cluster convex-hull volume results ===")
            print(f"Number of valid clusters      : {len(volumes)}")
            print(f"Total volume                  : {sum(volumes):.6f}")
            print(f"Maximum single-cluster volume : {max(volumes):.6f}")
            print(f"Minimum single-cluster volume : {min(volumes):.6f}")

            # Print sorted by volume
            for info in sorted(cluster_stats, key=lambda x: x["volume"], reverse=True)[:8]:
                print(f"  Cluster {info['cluster_id']:2d} | "
                      f"points {info['n_points']:5d} | "
                      f"volume {info['volume']:8.6f} | "
                      f"ellipsoid volume {info['ellipsoid_volume']:8.6f} | "
                      f"aspect ratio {info['aspect_ratio']:6.2f} | "
                      f"B-side ellipsoid volume {info['corresp_ellipsoid_volume']:8.6f} | "
                      f"B-side aspect ratio {info['corresp_aspect_ratio']:6.2f} | "
                      f"max_dev {info['max_dev']:6.3f} | "
                      f"mean_dev {info['mean_dev']:6.3f}")
        else:
            print("No valid convex-hull volumes computed")

        # ====================== 7. Visualization 1: clustering results (different colors) ======================
        if n_clusters > 0:
            cmap = plt.get_cmap("tab20")
            colors = np.zeros((len(points_large), 3))
            for cid in range(n_clusters):
                mask = (labels == cid)
                colors[mask] = cmap(cid % 20)[:3]
                colors[labels == -1] = [0.35, 0.35, 0.35]   # noise in gray

            pcd_vis = o3d.geometry.PointCloud()
            pcd_vis.points = o3d.utility.Vector3dVector(points_large)
            pcd_vis.colors = o3d.utility.Vector3dVector(colors)

            mesh_B_vis = o3d.geometry.TriangleMesh(mesh_A)
            mesh_B_vis.paint_uniform_color([0.85, 0.85, 0.9])
            mesh_B_vis.compute_vertex_normals()

            o3d.visualization.draw_geometries(
                [pcd_vis, mesh_B_vis],
                window_name=f"DBSCAN clustering (percentile={percentile_threshold:.1f}%)",
                zoom=0.8, front=[0.42, 0.74, 0.53],
                lookat=[0, 0, 0], up=[-0.15, -0.20, 0.97]
            )

        print("\n=== Visualize corresponding points of each cluster on mesh_B separately (sorted by volume descending) ===")

        app = gui.Application.instance
        app.initialize()

        main_win = app.create_window("All cluster original points on mesh_B (concentrated view)", 1200, 800)
        em = main_win.theme.font_size
        layout = gui.ScrollableVert(spacing=em, margins=gui.Margins(em))
        main_win.add_child(layout)

        if len(mesh_B.vertex_normals) == 0:
            mesh_B.compute_vertex_normals()

        sorted_stats = sorted(cluster_stats, key=lambda x: x["volume"], reverse=True)[:8]

        for idx, info in enumerate(sorted_stats):
            if 'points_A_cluster' not in info or len(info['points_A_cluster']) < 20:
                print(f"Skipping cluster {info['cluster_id']} (insufficient points)")
                continue

            scene_widget = gui.SceneWidget()
            scene_widget.scene = rendering.Open3DScene(main_win.renderer)
            mesh_mat = rendering.MaterialRecord()
            mesh_mat.shader = "defaultLit"
            mesh_mat.base_color = [0.85, 0.85, 0.9, 1.0]
            scene_widget.scene.add_geometry(f"mesh_bg_{idx}", mesh_B, mesh_mat)

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(info['points_A_cluster'])
            pcd_mat = rendering.MaterialRecord()
            pcd_mat.shader = "defaultUnlit"
            pcd_mat.point_size = 2.0
            pcd_mat.base_color = [1.0, 0.0, 0.0, 1.0]
            scene_widget.scene.add_geometry(f"pcd_cluster_{idx}", pcd, pcd_mat)

            bbox = scene_widget.scene.bounding_box
            center = bbox.get_center()
            extent = bbox.get_extent().max() if not bbox.is_empty() else 100.0
            eye = center + np.array([0.0, 0.0, extent * 2.5])   # dynamically pull back
            scene_widget.setup_camera(60.0, bbox, eye)

            label = gui.Label(f"Cluster {info['cluster_id']} | volume {info['volume']:.6f} | points {info['n_points']}")
            label.text_color = gui.Color(1, 1, 1, 1)
            panel = gui.Vert(spacing=em/2)
            panel.add_child(label)
            panel.add_child(scene_widget)
            layout.add_child(panel)

        app.run()

    # ====================== 8. Visualization 2: original signed-distance coloring (coolwarm) ======================
    # (always shown for comparison)
    cmap = plt.get_cmap("coolwarm")
    d_max = np.percentile(dists_abs, 95)
    norm = np.clip(dists_abs / d_max, 0, 1)
    colors_orig = cmap(norm)[:, :3]

    pcd_dev = o3d.geometry.PointCloud()
    pcd_dev.points = mesh_A.vertices
    pcd_dev.colors = o3d.utility.Vector3dVector(colors_orig)

    mesh_A_vis = o3d.geometry.TriangleMesh(mesh_A)
    mesh_A_vis.paint_uniform_color([0.7, 0.7, 0.7])
    mesh_A_vis.compute_vertex_normals()

    o3d.visualization.draw_geometries(
        [pcd_dev, mesh_A_vis, mesh_B],
        window_name="Original signed distance coloring visualization",
        zoom=0.8, front=[0.42, 0.74, 0.53],
        lookat=[0, 0, 0], up=[-0.15, -0.20, 0.97]
    )


# if __name__ == "__main__":
#     visualize_registration_deviation(
#         mesh_A_filename="689_2849_LEFT_MCP_2_up.ply",
#         mesh_B_filename="clasif_test.ply",
#     )
