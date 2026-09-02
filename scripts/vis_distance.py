"""
vis_distance.py

将 abnormal_gui.py 整段脚本包装成一个函数。
输入只有 mesh_A_filename / mesh_B_filename，其余全部使用 abnormal_gui.py 里的硬编码默认值。
逻辑、阈值、聚类、可视化与 abnormal_gui.py 一致，不额外改写。
"""

import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering


def trimesh_to_open3d_mesh(tm):
    """
    将 trimesh.Trimesh 转换为 open3d.geometry.TriangleMesh
    支持顶点、面、法向量、顶点颜色（如果有）
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
    对应 MATLAB 中的 khachiyan 函数。

    参数:
        a:   (n, m) array，输入点（列向量形式）
        tol: 正数，容差（相对体积误差控制）

    返回:
        E:   椭球矩阵，使得所有点满足 dot(a_i, E @ a_i) <= 1
    """
    n, m = a.shape
    if n < 2:
        raise ValueError("Input must be in two dimensions or higher.")
    if not np.all(np.isreal(a)):
        raise ValueError("Inputs must be real.")
    if not (np.isreal(tol) and tol > 0):
        raise ValueError("Tolerance must be positive.")

    # 初始化
    invA = n * np.linalg.inv(a @ a.T)
    w = np.sum(a * (invA @ a), axis=0)   # dot(a, invA @ a, 1)

    iter_count = 0
    max_iter = 10000  # 防止死循环

    while True:
        iter_count += 1
        if iter_count > max_iter:
            print("Warning: reached max iteration in khachiyan")
            break

        # 找到最大 w 的索引和值
        r = np.argmax(w)
        w_r = w[r]

        f = w_r / n
        epsilon = f - 1.0

        if epsilon <= tol:
            break

        g = epsilon / ((n - 1) * f)
        h = 1 + g
        g = g / f

        # 更新 invA
        b = invA @ a[:, r]
        invA = h * invA - g * np.outer(b, b)

        # 更新 w
        bTa = b @ a
        w = h * w - g * (bTa * bTa)

    # 最终椭球矩阵（考虑数值误差后做一次缩放）
    E = invA / w_r

    # 强制所有点被覆盖（处理浮点误差）
    dots = np.sum(a * (E @ a), axis=0)
    max_dot = np.max(dots)
    if max_dot > 1.0:
        E = E / max_dot

    return E


def lowner(a, tol=1e-3):
    """
    Approximate Löwner ellipsoid of a general point set.
    对应 MATLAB 中的 lowner 函数。

    参数:
        a:   (n, m) array，点云（每列是一个点，n=维度，m=点数）
        tol: 容差

    返回:
        E:   椭球矩阵 (n×n)
        c:   椭球中心 (n,)
             满足：对于所有点 x，(x - c).T @ E @ (x - c) <= 1
    """
    n, m = a.shape
    if n < 1:
        raise ValueError("Input must be in one dimension or higher.")

    # 升维：附加一行 1，构造中心对称版本
    aug = np.vstack([a, np.ones(m)])

    # 调用 khachiyan 求解升维后的椭球
    F = khachiyan(aug, tol)

    # 取前 n 行 n 列 和最后一列
    A = F[:n, :n]
    b = F[:n, -1]

    # 求中心 c = -A⁻¹ b
    try:
        c = -np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        raise RuntimeError("Singular matrix in lowner: cannot solve for center.")

    # 计算 E
    E = A / (1 - c @ b - F[-1, -1])

    # 再次强制所有点被覆盖（最保险的做法）
    ac = a - c[:, np.newaxis]
    dots = np.sum(ac * (E @ ac), axis=0)
    max_dot = np.max(dots)
    if max_dot > 1.0 + 1e-10:
        E = E / max_dot

    return E, c


def compute_min_vol_ellipsoid_metrics(points, tol=1e-4):
    """
    計算輸入 3D 或 2D 點雲的最小體積包圍椭球 (Löwner-John ellipsoid)
    並返回長短軸比 (aspect ratio) 與椭球體積

    參數:
        points : np.ndarray, shape (N, d) 或 (d, N)
                 點雲座標，d 通常為 2 或 3
        tol    : float, 容差 (預設 1e-4)

    返回:
        dict 包含以下鍵值:
        - 'aspect_ratio' : float, 最長半軸 / 最短半軸 (若失敗則 np.nan)
        - 'volume'       : float, 椭球體積 (若失敗則 np.nan)
        - 'center'       : (d,) array, 椭球中心
        - 'semi_axes'    : array, 各半軸長度 (排序後)
        - 'E'            : (d,d) array, 椭球矩陣
        - 'success'      : bool, 是否成功計算
    """
    # 統一轉成 (d, N) 格式
    points = np.asarray(points)
    if points.ndim != 2:
        raise ValueError("points 必須是 2維陣列")

    if points.shape[0] > points.shape[1]:
        points = points.T  # 轉成 (d, N)

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
        # 直接呼叫 lowner （不做凸包）
        E, c = lowner(points, tol=tol)

        # 計算特徵值 → 半軸長 = 1 / sqrt(特徵值)
        eigvals = np.linalg.eigvalsh(E)
        valid = eigvals > 1e-12
        if not np.any(valid):
            raise ValueError("無有效特徵值")

        semi_axes = 1.0 / np.sqrt(eigvals[valid])
        semi_axes = np.sort(semi_axes)

        aspect_ratio = semi_axes[-1] / semi_axes[0] if len(semi_axes) >= 2 else np.nan

        # 計算 d 維椭球體積
        # 體積 = (π^{d/2} / Γ(d/2 + 1)) * ∏(半軸長)
        if d == 2:
            volume = np.pi * np.prod(semi_axes)
        elif d == 3:
            volume = (4/3) * np.pi * np.prod(semi_axes)
        else:
            from scipy.special import gamma
            vol_factor = np.pi**(d/2) / gamma(d/2 + 1)
            volume = vol_factor * np.prod(semi_axes)

    except Exception as e:
        print(f"計算失敗: {str(e)}")
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
    abnormal_gui.py 全脚本的函数包装。

    参数:
        mesh_A_filename: 查询网格路径 (.ply)，对应原脚本 mesh_A
        mesh_B_filename: 参考网格路径 (.ply)，对应原脚本 mesh_B

    其余全部使用 abnormal_gui.py 硬编码默认值:
        percentile_threshold = 75.0
        eps_dbscan = avg_nn * 1.5（算不出则 25.0）
        min_points = 5
        始终做 DBSCAN 可视化、分簇 GUI、signed distance 着色
    """
    # ====================== 1. 加载模型 & 计算 signed distance ======================
    mesh_A = o3d.io.read_triangle_mesh(mesh_A_filename)
    mesh_B = o3d.io.read_triangle_mesh(mesh_B_filename)

    # 转为 tensor 并构建 RaycastingScene
    mesh_B_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh_B)
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(mesh_B_t)

    points_A = np.asarray(mesh_A.vertices, dtype=np.float32)
    query = o3d.core.Tensor(points_A)
    signed_dists = scene.compute_signed_distance(query).numpy().flatten()   # 有正负
    dists_abs = np.abs(signed_dists)

    print(f"总查询点数: {len(points_A)}")
    print(f"最大绝对偏差: {dists_abs.max():.4f}   平均: {dists_abs.mean():.4f}")

    # ====================== 2. 自动推荐 percentile_threshold ======================
    def auto_recommend_percentile(dists_abs):
        """
        根据偏差分布的变异系数 (CV) 自动推荐合理的 percentile
        - CV 高（偏差差异大）→ 更严格，只取最极端的 2~3%
        - CV 低（偏差较均匀）→ 稍宽松，取 6~7%
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
            target_frac = 0.05 * 3      # top 5%（最常用）
        else:
            target_frac = 0.07 * 3      # top 7%

        p = 100 * (1 - target_frac)
        return np.clip(p, 90.0, 99.0)

    recommended_p = auto_recommend_percentile(dists_abs)
    print(f"\n=== 自动推荐 ===")
    print(f"基于变异系数 CV = {np.std(dists_abs)/np.mean(dists_abs):.3f}")
    print(f"推荐 percentile_threshold = {recommended_p:.1f}%  （最偏的 {100-recommended_p:.1f}% 点）")

    # ====================== 3. 用户可手动微调（默认使用自动推荐值） ======================
    #percentile_threshold = recommended_p          # ←←← 这里手动改即可，例如改成 96.5
    percentile_threshold = 75.0                 # 示例：强制使用 95%

    d_threshold = np.percentile(dists_abs, percentile_threshold)
    print(f"实际采用阈值 = {d_threshold:.4f}（{percentile_threshold}% 分位数）")

    # ====================== 4. 筛选大偏差点 ======================
    mask_large = dists_abs > d_threshold
    points_large = points_A[mask_large]
    signed_large = signed_dists[mask_large]

    n_large = len(points_large)
    print(f"筛选出 {n_large} 个大偏差点（占总点数 {n_large/len(points_A)*100:.2f}%）")

    if n_large < 30:
        print("警告：大偏差点过少，建议降低 percentile_threshold 或检查配准质量")
        # 可视化原始偏差着色后退出
    else:
        # ====================== 5. DBSCAN 区域聚类 ======================
        # === 自适应 eps_dbscan：用“点云内每个点的平均最近邻距离”代替人工设定的 25 ===
        # 先创建点云（后续 DBSCAN 需要）
        pcd_large = o3d.geometry.PointCloud()
        pcd_large.points = o3d.utility.Vector3dVector(points_large)

        print("\n=== 自适应 eps_dbscan 计算 ===")
        # 使用 KDTree 计算每个点的最近邻距离（k=2，排除自身）
        kdtree = o3d.geometry.KDTreeFlann(pcd_large)
        nn_dists = []
        for i in range(len(pcd_large.points)):
            _, _, d = kdtree.search_knn_vector_3d(pcd_large.points[i], 2)
            if len(d) >= 2:
                nn_dists.append(np.sqrt(d[1]))   # 注意：Open3D KDTreeFlann 返回的是 squared distance，故需开方

        if nn_dists:
            avg_nn_dist = np.mean(nn_dists)
            eps_dbscan = avg_nn_dist * 1.5       # ←←← 倍数可自行调整（推荐 2~4）
                                             # 理由：eps 需略大于平均点间距才能有效聚类
            print(f"点云内每个点的平均最近邻距离: {avg_nn_dist:.4f}")
            print(f"自适应 eps_dbscan = {eps_dbscan:.4f}（avg_nn * 3）")
        else:
            eps_dbscan = 25.0
            print("无法计算最近邻距离，退回默认 eps=25.0")

        min_points = 5       # 最小簇点数（保持原值）

        labels = np.array(pcd_large.cluster_dbscan(
            eps=eps_dbscan,
            min_points=min_points,
            print_progress=True
        ))

        n_clusters = int(labels.max()) + 1 if labels.max() >= 0 else 0
        print(f"DBSCAN 找到 {n_clusters} 个有效聚类（-1 为噪声）")

        # ====================== 6. 每个聚类计算凸包体积 ======================
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
                print(f"簇 {cid} 凸包计算失败（退化几何）: {e}")

        # 输出统计
        if volumes:
            print("\n=== 聚类凸包体积结果 ===")
            print(f"有效聚类数      : {len(volumes)}")
            print(f"总体积          : {sum(volumes):.6f}")
            print(f"最大单簇体积    : {max(volumes):.6f}")
            print(f"最小单簇体积    : {min(volumes):.6f}")

            # 按体积排序打印
            for info in sorted(cluster_stats, key=lambda x: x["volume"], reverse=True)[:8]:
                print(f"  簇 {info['cluster_id']:2d} | "
                      f"点数 {info['n_points']:5d} | "
                      f"体积 {info['volume']:8.6f} | "
                      f"椭球体积 {info['ellipsoid_volume']:8.6f} | "
                      f"长短轴比 {info['aspect_ratio']:6.2f} | "
                      f"B侧椭球体积 {info['corresp_ellipsoid_volume']:8.6f} | "
                      f"B侧长短轴比 {info['corresp_aspect_ratio']:6.2f} | "
                      f"max_dev {info['max_dev']:6.3f} | "
                      f"mean_dev {info['mean_dev']:6.3f}")
        else:
            print("未计算到有效凸包体积")

        # ====================== 7. 可视化 1：聚类结果（不同颜色） ======================
        if n_clusters > 0:
            cmap = plt.get_cmap("tab20")
            colors = np.zeros((len(points_large), 3))
            for cid in range(n_clusters):
                mask = (labels == cid)
                colors[mask] = cmap(cid % 20)[:3]
                colors[labels == -1] = [0.35, 0.35, 0.35]   # 噪声灰色

            pcd_vis = o3d.geometry.PointCloud()
            pcd_vis.points = o3d.utility.Vector3dVector(points_large)
            pcd_vis.colors = o3d.utility.Vector3dVector(colors)

            mesh_B_vis = o3d.geometry.TriangleMesh(mesh_A)
            mesh_B_vis.paint_uniform_color([0.85, 0.85, 0.9])
            mesh_B_vis.compute_vertex_normals()

            o3d.visualization.draw_geometries(
                [pcd_vis, mesh_B_vis],
                window_name=f"DBSCAN 聚类 (percentile={percentile_threshold:.1f}%)",
                zoom=0.8, front=[0.42, 0.74, 0.53],
                lookat=[0, 0, 0], up=[-0.15, -0.20, 0.97]
            )

        print("\n=== 单独可视化每个聚类在 mesh_B 上的对应点（按体积降序） ===")

        app = gui.Application.instance
        app.initialize()

        main_win = app.create_window("所有聚类原始点 on mesh_B (集中视图)", 1200, 800)
        em = main_win.theme.font_size
        layout = gui.ScrollableVert(spacing=em, margins=gui.Margins(em))
        main_win.add_child(layout)

        if len(mesh_B.vertex_normals) == 0:
            mesh_B.compute_vertex_normals()

        sorted_stats = sorted(cluster_stats, key=lambda x: x["volume"], reverse=True)[:8]

        for idx, info in enumerate(sorted_stats):
            if 'points_A_cluster' not in info or len(info['points_A_cluster']) < 20:
                print(f"跳过簇 {info['cluster_id']}（点不足）")
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
            eye = center + np.array([0.0, 0.0, extent * 2.5])   # 动态拉远
            scene_widget.setup_camera(60.0, bbox, eye)

            label = gui.Label(f"Cluster {info['cluster_id']} | 体积 {info['volume']:.6f} | 点数 {info['n_points']}")
            label.text_color = gui.Color(1, 1, 1, 1)
            panel = gui.Vert(spacing=em/2)
            panel.add_child(label)
            panel.add_child(scene_widget)
            layout.add_child(panel)

        app.run()

    # ====================== 8. 可视化 2：原始带符号偏差着色（coolwarm） ======================
    # （始终显示，方便对比）
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
        window_name="原始 signed distance 着色可视化",
        zoom=0.8, front=[0.42, 0.74, 0.53],
        lookat=[0, 0, 0], up=[-0.15, -0.20, 0.97]
    )


# if __name__ == "__main__":
#     visualize_registration_deviation(
#         mesh_A_filename="689_2849_LEFT_MCP_2_up.ply",
#         mesh_B_filename="clasif_test.ply",
#     )
