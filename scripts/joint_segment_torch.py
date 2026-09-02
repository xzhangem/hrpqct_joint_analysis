import SimpleITK as sitk
import numpy as np
import math

import scipy
from PIL import Image as image
import pathlib
import skimage as ski
import scipy.ndimage as ndimg
import trimesh
import utils

from skimage import segmentation
import edt

from scipy import ndimage as ndi
from numba import njit, prange
from joblib import Parallel, delayed
import pyvista as pv
import fastmorph
from skimage import segmentation as ski_seg

import time
import torch
import torch.nn.functional as F


import gc
from pathlib import Path
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


_PREDICTOR_CACHE = {}

def release_nnunet():
    global _PREDICTOR_CACHE
    _PREDICTOR_CACHE.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
        gc.collect()
        torch.cuda.empty_cache()


def nnunet_predict(
    nii_path,
    model_dir,
    fold=0,
    gpu=0,
    checkpoint="checkpoint_final.pth",
    keep_model=False,
):
    """
    输入
        nii_path  : 待分割的 .nii
        model_dir : 训练好的 nnU-Net 目录（含 fold_0/ 的那一层）
    输出
        mask : uint8 {0,1}，(z, y, x)，在 CPU 上
    keep_model=False 时推理完立即卸载 nnU-Net，把显存还给 level set。
    """
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO

    nii_path = str(Path(nii_path).resolve())
    model_dir = str(Path(model_dir).resolve())
    if not Path(nii_path).is_file():
        raise FileNotFoundError(nii_path)
    if not Path(model_dir).is_dir():
        raise FileNotFoundError(model_dir)

    key = (model_dir, int(fold), int(gpu), checkpoint)
    predictor = _PREDICTOR_CACHE.get(key)
    if predictor is None:
        fold_dir = Path(model_dir) / "fold_{}".format(fold)
        ckpt_path = fold_dir / checkpoint
        if not ckpt_path.is_file():
            ckpt_path = fold_dir / "checkpoint_latest.pth"
            if not ckpt_path.is_file():
                raise FileNotFoundError(fold_dir / checkpoint)
            checkpoint = "checkpoint_latest.pth"

        device = (
            torch.device("cuda", int(gpu))
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,
            perform_everything_on_device=False,
            device=device,
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=False,
        )
        predictor.initialize_from_trained_model_folder(
            model_dir,
            use_folds=(int(fold),),
            checkpoint_name=checkpoint,
        )
        _PREDICTOR_CACHE[key] = predictor

    img, props = SimpleITKIO().read_images([nii_path])
    seg = predictor.predict_single_npy_array(img, props, None, None, False)
    seg = np.asarray(seg)
    if seg.ndim == 4:
        seg = seg[0]
    mask = np.ascontiguousarray((seg > 0).astype(np.uint8))

    ref = sitk.GetArrayFromImage(sitk.ReadImage(nii_path))
    if tuple(mask.shape) != tuple(ref.shape):
        raise RuntimeError("seg shape {} != ct shape {}".format(mask.shape, ref.shape))

    del img, seg, ref
    if not keep_model:
        release_nnunet()
    else:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return mask

def Div2d(x, y):
    x = to_tensor(x)
    y = to_tensor(y)

    x_x = torch.gradient(x, dim=0)[0]  # ∂x/∂x (dim=0)
    y_y = torch.gradient(y, dim=1)[0]  # ∂y/∂y (dim=1)

    div = x_x + y_y
    return div

#def Div2d(x, y):
#    x_x = np.gradient(x, edge_order=1)[0]
#    y_y = np.gradient(y, edge_order=1)[1]
#    return x_x + y_y


def gaussian_kernel_1d(sigma, size=None):
    """生成 1D 高斯核 (可分離)"""
    if size is None:
        size = int(4 * sigma + 1)   # 通常 4~6 sigma 涵蓋足夠
        size = size if size % 2 == 1 else size + 1
    x = torch.arange(-size//2 + 1, size//2 + 1, dtype=torch.float32)
    kernel = torch.exp(-x**2 / (2 * sigma**2))
    kernel = kernel / kernel.sum()          # 歸一化
    return kernel


def gaussian_blur3d(input_vol, sigma, kernel_size):
    """
    對 3D 體積做高斯模糊 (可分離 1D 卷積 ×3，速度快很多)
    
    input_vol: shape [B, C, D, H, W] 或 [C, D, H, W] 或 [D, H, W]
    sigma: 可以是單一 float 或 (sigma_d, sigma_h, sigma_w)
    """
    if isinstance(sigma, (int, float)):
        sigma = (sigma, sigma, sigma)
    
    # 為每個維度生成 1D 核
    kd = gaussian_kernel_1d(sigma[0], kernel_size)
    kh = gaussian_kernel_1d(sigma[1], kernel_size)
    kw = gaussian_kernel_1d(sigma[2], kernel_size)
    
    # 轉成 1D → 需要 reshape 成 conv3d 需要的形狀
    kd = kd.view(1, 1, -1, 1, 1)   # [1,1,kd,1,1]
    kh = kh.view(1, 1, 1, -1, 1)   # [1,1,1,kh,1]
    kw = kw.view(1, 1, 1, 1, -1)   # [1,1,1,1,kw]
    
    # 補齊 batch & channel 維度
    if input_vol.dim() == 3:
        input_vol = input_vol.unsqueeze(0).unsqueeze(0)  # → [1,1,D,H,W]
    elif input_vol.dim() == 4:
        input_vol = input_vol.unsqueeze(0)               # → [1,C,D,H,W]
    
    padding_d = (kd.shape[2] - 1) // 2
    padding_h = (kh.shape[3] - 1) // 2
    padding_w = (kw.shape[4] - 1) // 2
    
    # 可分離卷積：三次 3D conv，但每次只在一個軸上作用
    out = F.conv3d(input_vol, kd, padding=(padding_d, 0, 0), groups=input_vol.shape[1])
    out = F.conv3d(out,       kh, padding=(0, padding_h, 0), groups=input_vol.shape[1])
    out = F.conv3d(out,       kw, padding=(0, 0, padding_w), groups=input_vol.shape[1])
    
    return out.squeeze(0) if input_vol.shape[0] == 1 else out



def to_tensor(arr):
    if isinstance(arr, np.ndarray):
        return torch.from_numpy(arr).to(device=device, dtype=torch.float32)
    elif isinstance(arr, torch.Tensor):
        return arr.to(device=device, dtype=torch.float32)
    else:
        raise TypeError("Input x and y must be numpy.ndarray or torch.Tensor")



def del3d(p):
    p = torch.as_tensor(p, dtype=torch.float32, device=device)
    p_x, p_y, p_z = torch.gradient(p, dim=(0,1,2))
    p_xx = torch.gradient(p_x, dim=0)[0]
    p_yy = torch.gradient(p_y, dim=1)[0]
    p_zz = torch.gradient(p_z, dim=2)[0]
    return p_xx + p_yy + p_zz



def Div(x, y, z):
    x = to_tensor(x)
    y = to_tensor(y)
    z = to_tensor(z)

    x_x = torch.gradient(x, dim=0)[0]  # ∂x/∂x (dim=0)
    y_y = torch.gradient(y, dim=1)[0]  # ∂y/∂y (dim=1)
    z_z = torch.gradient(z, dim=2)[0]

    div = x_x + y_y + z_z
    return div


def DistReg(phi):

    if isinstance(phi, np.ndarray):
        phi = torch.from_numpy(phi).to(device=device, dtype=torch.float32)
    elif isinstance(phi, torch.Tensor):
        phi = phi.to(device=device, dtype=torch.float32)
    else:
        raise TypeError("Input phi must be numpy.ndarray or torch.Tensor")

    # 计算梯度 (使用 torch.gradient)
    phi_x, phi_y, phi_z = torch.gradient(phi, dim=(0, 1, 2))

    # 计算梯度模长
    s = torch.sqrt(phi_x**2 + phi_y**2 + phi_z**2 + 1e-10)  # 防止除零

    # 定义势阱区域
    a = (s >= 0) & (s <= 1)          # 0 ≤ s ≤ 1
    b = s > 1                         # s > 1

    # ps 函数（双势阱的导数相关项）
    ps = torch.zeros_like(s)
    ps[a] = torch.sin(2 * torch.pi * s[a]) / (2 * torch.pi)
    ps[b] = s[b] - 1

    # dps/ds （对 s 的导数）
    # 当 ps == 0 时特殊处理，避免除零
    dps = torch.zeros_like(s)
    nonzero_ps = (ps != 0)
    zero_ps = ~nonzero_ps

    dps[nonzero_ps] = ps[nonzero_ps]
    # 当 ps ≈ 0 且 s ≠ 0 时，dps/ds ≈ 1；s=0 时可设为 0 或其他平滑值
    dps[zero_ps] = 1.0 * (s[zero_ps] != 0)

    # 计算向量场： dps * ∇φ - ∇φ = (dps - 1) * ∇φ
    vec_x = dps * phi_x - phi_x
    vec_y = dps * phi_y - phi_y
    vec_z = dps * phi_z - phi_z

    # Div( (dps-1) ∇φ ) + 4 * Laplacian(φ)
    div_term = Div(vec_x, vec_y, vec_z)
    lap_term = del3d(phi) * 4.0

    # 最终正则化项
    f = div_term + lap_term

    return f


def SmoothDirac(phi, epsilon, device=None):
    if isinstance(phi, np.ndarray):
        phi = torch.from_numpy(phi).to(device=device, dtype=torch.float32)
    elif isinstance(phi, torch.Tensor):
        phi = phi.to(device=device, dtype=torch.float32)
    else:
        raise TypeError("Input phi must be numpy.ndarray or torch.Tensor")

    # 计算核心部分：(1/(2ε)) * (1 + cos(π phi / ε))
    cos_term = torch.cos(torch.pi * phi / epsilon)
    f = (1.0 / (2.0 * epsilon)) * (1.0 + cos_term)

    # 掩码：只在 [-epsilon, epsilon] 区间内非零
    mask = (phi >= -epsilon) & (phi <= epsilon)

    # 应用掩码（区间外置零）
    f = f * mask.float()   # mask 是 bool，转为 float (0/1)

    return f


def SmoothHeavi(phi, epsilon):
    """
    光滑的 Heaviside 阶跃函数（常用于 level set 方法中的区域指示函数）

    当 |phi| <= epsilon 时：
        H(phi) ≈ 0.5 * (1 + phi/ε + (1/π) sin(π phi / ε))
    当 phi > epsilon 时：  H(phi) = 1
    当 phi < -epsilon 时： H(phi) = 0

    参数:
        phi: 输入水平集函数，可以是 numpy.ndarray 或 torch.Tensor
        epsilon: 光滑参数（宽度），通常为 1.0 ~ 2.0
        device: 可选，指定计算设备；若不传则自动检测

    返回:
        torch.Tensor，与 phi 相同形状的光滑 Heaviside 函数值
    """
    if isinstance(phi, np.ndarray):
        phi = torch.from_numpy(phi).to(device=device, dtype=torch.float32)
    elif isinstance(phi, torch.Tensor):
        phi = phi.to(device=device, dtype=torch.float32)
    else:
        raise TypeError("Input phi must be numpy.ndarray or torch.Tensor")

    # 计算过渡区间的光滑部分
    sin_term = torch.sin(torch.pi * phi / epsilon)
    f = 0.5 * (1.0 + phi / epsilon + (1.0 / torch.pi) * sin_term)

    # 定义三个区域的掩码（互斥）
    inside  = (phi >= -epsilon) & (phi <= epsilon)     # 过渡区
    outside_pos = phi > epsilon                         # 正外部 → 1
    outside_neg = phi < -epsilon                        # 负外部 → 0

    # 组合结果
    #heavi = torch.zeros_like(phi)
    #heavi[inside]      = f[inside]
    #heavi[outside_pos] = 1.0
    #heavi[outside_neg] = 0.0

    # 或者使用更简洁的广播写法（数值等价）：
    heavi = f * inside.float() + 1.0 * outside_pos.float() + 0.0 * outside_neg.float()

    return heavi


def NeumannBoundCond_np(f):
    """
    更簡潔的寫法，利用廣播和同時賦值
    """
    g = f.copy()

    g[[0, -1], :, :] = g[[1, -2], :, :]
    g[:, [0, -1], :] = g[:, [1, -2], :]
    g[:, :, [0, -1]] = g[:, :, [1, -2]]

    return g



def NeumannBoundCond(f):
    """
    应用 Neumann 边界条件（零梯度边界，也称反射边界）
    通过镜像填充边界像素，使梯度计算在边界处为零。

    参数:
        f: 输入 3D 数组，可以是 numpy.ndarray 或 torch.Tensor
        device: 可选，指定计算设备；若不传则自动检测

    返回:
        torch.Tensor，与输入 f 相同形状，已应用 Neumann 边界条件的数组
    """
    f = f.unsqueeze(0).unsqueeze(0)
    f_padded = torch.nn.functional.pad(f, pad=(1, 1, 1, 1, 1, 1), mode='replicate')

    f_padded = f_padded.squeeze(0).squeeze(0)
    # 切回原始形状（去掉填充层）
    g = f_padded[1:-1, 1:-1, 1:-1]

    return g


def LogVol(x):
    x = torch.as_tensor(x, dtype=torch.float32, device=device)

    return torch.where(x > 0, torch.log(x), torch.zeros_like(x))

def MaskCent(mask):
    return scipy.ndimage.center_of_mass(mask)

def SmoothDirac_atan(phi, epsilon):
    phi = to_tensor(phi)
    return (epsilon / torch.pi) / (epsilon ** 2 + phi ** 2)


def SmoothHeavi_atan(phi, epsilon):
    phi = to_tensor(phi)
    return 0.5 * (1.0 + (2.0 / torch.pi) * torch.atan(phi / epsilon))


@torch.no_grad()
def SurfaceShrink_cmig(vol_0, vol, res, mu, lambd, alpha, epsilon, timestep, iter_num):
    """
    MATLAB:
      ConReg_vol = SurfaceShrink(ConReg_vol, raw_seg, 7, 0.4, 1, 0, 1, 0.5, 5)
    vol_0: 被 shrink 的模板 (level set)
    vol:   原始二值 (nnU-Net)
    """
    small = 1e-9
    res = abs(float(res))
    vol_0 = to_tensor(vol_0)
    vol = to_tensor(vol)
    print(vol_0.shape)
    print(vol.shape)
    g = 1.0 - vol
    phi = SignDistance(vol_0)
    #phi = vol_0

    for k in range(int(iter_num)):
        phi = NeumannBoundCond(phi)

        dist_regu = DistReg(phi)

        phi_x, phi_y, phi_z = torch.gradient(phi, dim=(0, 1, 2))
        s = torch.sqrt(phi_x ** 2 + phi_y ** 2 + phi_z ** 2)
        Nx = phi_x / (s + small)
        Ny = phi_y / (s + small)
        Nz = phi_z / (s + small)
        #del phi_x, phi_y, phi_z, s
        curvature = Div(Nx, Ny, Nz)
        #del Nx, Ny, Nz

        dirac_phi = SmoothDirac_atan(phi, epsilon)
        heavi_phi = SmoothHeavi_atan(phi, epsilon)
        edge_term = dirac_phi * (curvature + res) * g
        data_term = dirac_phi * (heavi_phi - vol_0)
        #del dirac_phi, heavi_phi

        phi = phi + timestep * g * (curvature + res) #* (mu * dist_regu + lambd * edge_term - alpha * data_term)
        #del dist_regu, edge_term, data_term, curvature
        #if torch.cuda.is_available():
        #    torch.cuda.empty_cache()

    mask = (phi > 0).to(dtype=torch.float32).cpu().numpy()
    del phi
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    #for i in range(mask.shape[2]):
    #    mask[:, :, i] = ndi.binary_fill_holes(mask[:, :, i]).astype(np.float32)
    return to_tensor(mask)


@torch.no_grad()
def SurfaceShrink(U, J, eps_gap=1.0, gamma=0.4, epsilon=1.0, timestep=0.5, iter_num=5):
    small = 1e-9
    U = (to_tensor(U) > 0).float()
    J = (to_tensor(J) > 0).float()
    w = (1.0 + float(eps_gap) - J).clamp(min=small)
    wx, wy, wz = torch.gradient(w, dim=(0, 1, 2))
    phi = SignDistance(U)

    for k in range(int(iter_num)):
        phi = NeumannBoundCond(phi)
        dist_regu = DistReg(phi)

        phi_x, phi_y, phi_z = torch.gradient(phi, dim=(0, 1, 2))
        s = torch.sqrt(phi_x ** 2 + phi_y ** 2 + phi_z ** 2)
        Nx = phi_x / (s + small)
        Ny = phi_y / (s + small)
        Nz = phi_z / (s + small)
        del phi_x, phi_y, phi_z, s
        curvature = Div(Nx, Ny, Nz)
        div_wN = wx * Nx + wy * Ny + wz * Nz + w * curvature
        del Nx, Ny, Nz, curvature

        dirac_phi = SmoothDirac_atan(phi, epsilon)
        edge_term = dirac_phi * div_wN
        del dirac_phi, div_wN

        phi = phi + timestep * (gamma * dist_regu + edge_term)
        del dist_regu, edge_term
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    mask = (phi > 0).to(dtype=torch.float32).cpu().numpy()
    del phi
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    #for i in range(mask.shape[2]):
    #    mask[:, :, i] = ndi.binary_fill_holes(mask[:, :, i]).astype(np.float32)
    return to_tensor(mask)

@torch.no_grad()
def LSEdgeSmooth(phi, vol, mu, lambd, alpha, epsilon, timestep, iter_num):
    small = 1e-8
    #phi = NeumannBoundCond(phi)
    for k in range(iter_num):
        phi = NeumannBoundCond(phi)
        phi_x, phi_y, phi_z = torch.gradient(phi, dim=(0,1,2))
        s = torch.sqrt(phi_x**2 + phi_y**2 + phi_z**2 + small)
        Nx, Ny, Nz = phi_x/s, phi_y/s, phi_z/s
        curvature = Div(Nx, Ny, Nz)
        dirac_phi = SmoothDirac(phi, epsilon)
        heavi_phi = SmoothHeavi(phi, epsilon)
        edge_term = dirac_phi * curvature
        data_term = dirac_phi * (heavi_phi - vol)
        phi = phi + timestep * (lambd * edge_term - alpha * data_term)

    return phi


@torch.no_grad()
def gamma_ls_energy_original_param(
    phi,
    vol,
    mu=0.0,
    lambd=6.0,
    alpha=1.5,
    beta=1.0,
    gamma_v=1.0,
    epsilon=1.0,
    timestep=1.0,
    iter_num=15
):
    """
    使用你原版代码中完全相同的 Gamma 参数估计方式（k, theta, modify_k, modify_theta）
    """
    phi = NeumannBoundCond(phi)
    small = 1e-7

    for i in range(iter_num):
        phi_x, phi_y, phi_z = torch.gradient(phi, dim=(0,1,2))
        s = torch.sqrt(phi_x**2 + phi_y**2 + phi_z**2 + small)
        Nx, Ny, Nz = phi_x/s, phi_y/s, phi_z/s
        curvature = Div(Nx, Ny, Nz)
        del Nx, Ny, Nz
        #dist_regu = DistReg(phi)

        dirac_phi = SmoothDirac(phi, epsilon)
        heavi_phi = SmoothHeavi(phi, epsilon)

    # ────────────────────────────────────────────────
    # 边缘 + 面积项
    # ────────────────────────────────────────────────
        edge_term = dirac_phi * curvature
        area_term = dirac_phi
        del curvature

    # ────────────────────────────────────────────────
    # 外部 Gaussian 项（基本保持原逻辑）
    # ────────────────────────────────────────────────
        inner_shift = vol * heavi_phi
        outside = 1.0 - heavi_phi
    
        sum_out = (vol - inner_shift).sum()
        area_out = outside.sum() + small
        c_ex = sum_out / area_out
        err_ex = vol - c_ex
        sigma_ex_sq = (err_ex**2 * outside).sum() / area_out + small
        ext_term = ((err_ex**2) / (2 * sigma_ex_sq) + 0.5 * torch.log(sigma_ex_sq))
        del outside, err_ex

    # ────────────────────────────────────────────────
    # 内部 Gamma 项 —— 使用原版参数估计公式
    # ────────────────────────────────────────────────
        inner_shift = inner_shift.min() - 0.5
        vol_shifted = vol - inner_shift

        log_vol = torch.where(vol_shifted > 0, torch.log(vol_shifted), torch.zeros_like(vol_shifted))

        sum_heavi     = heavi_phi.sum() + small
        sum_vol       = (vol_shifted * heavi_phi).sum()
        sum_logvol    = (log_vol     * heavi_phi).sum()
        sum_vol_logvol= (vol_shifted * log_vol * heavi_phi).sum()

    # 原版 k 的分子和分母
        numerator_k   = sum_vol
        denominator_k = sum_vol_logvol - (sum_logvol * sum_vol / sum_heavi)
        k = numerator_k / (denominator_k + small)
        print(k)

    # 原版 theta
        #theta = sum_logvol / sum_heavi - (sum_vol * sum_logvol / (sum_heavi ** 2))
        theta = sum_vol_logvol / sum_heavi - (sum_vol * sum_logvol / (sum_heavi ** 2))
        print(theta)

    # 原版 modify_k 的修正项
        correction = (3 * k - 2 * k / (3 + 3*k) - 4 * k / (5 + 10*k + 5*k**2)) / sum_heavi
        modify_k = k - correction

    # 原版 modify_theta
        modify_theta = (sum_heavi / (sum_heavi - 1 + small)) * theta

    # Gamma 负对数似然核心部分（使用 modify_k 和 modify_theta）
        inter_term = (
            -(modify_k - 1) * log_vol +
            vol_shifted / modify_theta +
            modify_k * torch.log(modify_theta + small) +
            torch.lgamma(modify_k + small)
        )

        del log_vol, vol_shifted, heavi_phi

        phi = phi + timestep * (lambd * edge_term - alpha * area_term - beta * inter_term + gamma_v * ext_term)
        print('edge_term: {}; area_ter: {}; inter_term: {}; ext_term: {}'.format(edge_term.max(), area_term.max(), inter_term.max(), ext_term.mean()))

        del dirac_phi, edge_term, area_term, inter_term, ext_term
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return phi

def SignDistance_slow_wnp(vol):
    vx, vy, vz = vol.shape
    contour = ski.segmentation.find_boundaries(vol, mode='inner').astype(np.uint8)
    distance_vol = ndimg.distance_transform_edt(1 - contour)
    for i in range(vx):
        for j in range(vy):
            for k in range(vz):
                if vol[i, j, k] == 0:
                    distance_vol[i, j, k] = -distance_vol[i, j, k]

    return distance_vol



def SignDistance_wnp(vol):
    if vol.dtype != np.uint8 and vol.dtype != np.bool_:
        vol = vol.astype(np.uint8)
    boundary = ski_seg.find_boundaries(vol, mode='inner').astype(np.uint8)
    dist = ndi.distance_transform_edt(~boundary)
    return np.where(vol, dist, -dist).astype(np.float32)



def SignDistance_np(vol):
    # 假設 vol 是 binary array，1=inside(物件)，0=outside
    anisotropy=(1,1,1)
    dist = edt.edt(vol, anisotropy=anisotropy)
    sdf = np.where(vol, -dist, dist)
    return sdf

def SignDistance(vol):
    vol_np = torch.as_tensor(vol).detach().cpu().numpy().astype(np.float32)
    sdf = SignDistance_wnp(vol_np)
    sdf = to_tensor(sdf)
    return sdf



def LevelSetSegmentation_torch(
    ori_vol,                  # 可以是 numpy 或 torch.Tensor
    iter_num,
    input_PDE_step=None,      # 原代码中未实际使用，可保留作占位
    device=None,
    # GammaLS_Autograd 的参数（可根据需要调整）
    gamma_ls_lr=0.02,
    gamma_ls_max_iter=20,
    reinitialize_every=0,
    use_dist_reg=False
):
    """
    PyTorch 版本的 LevelSetSegmentation
    使用 autograd 优化的 GammaLS_Segmentation 作为核心
    """
    vol = ori_vol - np.min(ori_vol) + 1
    #print(np.min(vol))
    vol_smooth = ndimg.gaussian_filter(vol, sigma=0.2, mode='nearest', radius=5).astype(float)
    Vx, Vy, Vz = np.gradient(vol_smooth, edge_order=1)
    f = Vx * Vx + Vy * Vy + Vz * Vz
    g = 1.0 / (1.0 + f)

    c0 = 2
    vx, vy, vz = vol.shape
    initialLSF = - c0 * np.ones((vx, vy, vz))
    coe = 0.15
    vx_cut = np.floor(vx * coe).astype(int)
    vy_cut = np.floor(vy * coe).astype(int)
    vz_cut = np.floor(vz * coe).astype(int)
    initialLSF[vx_cut : vx-vx_cut, vy_cut : vy-vy_cut, vz_cut : vz-vz_cut] = c0
    phi = initialLSF

    phi = to_tensor(phi)
    vol = to_tensor(vol)

    phi_final = gamma_ls_energy_original_param(
            phi = phi,
            vol = vol,
            mu=0.0,
            lambd=6.0,
            alpha=1.5,
            beta=1.0,
            gamma_v=1.0,
            epsilon=1.0,
            timestep=1.0,
            iter_num=15
            )

    # 6. 生成分割结果
    seg_result = (phi_final > 0).float()   # 保持 float 类型，与原版 .astype(float) 一致

    return seg_result


def fill_slice(vol, sl):
    slice_2d = vol[:, :, sl]
    # 如果 vol 是 float，先轉 bool 處理比較穩
    filled = ndi.binary_fill_holes(slice_2d)  # 或 slice_2d.astype(bool)
    return filled.astype(np.float32)


def morphological_closing_fastmorph(vol: np.ndarray, radius: int = 12):
    """CPU 最快版（binary volume 直接用）"""
    vol = vol.astype(np.uint32)  # fastmorph 推荐 uint* 类型

    closed = fastmorph.spherical_close(
            vol,
            radius=radius,
            parallel=0,
            anisotropy=(1.0, 1.0, 1.0)
            )
    
    vx, vy, vz = closed.shape
    for sl in range(vz):
        tem = closed[:, :, sl]
        tem = scipy.ndimage.binary_fill_holes(tem)
        closed[:,:,sl] = tem

    return closed.astype(np.float32)


def InitialHeight(input_vol):
    vx, vy, vz = input_vol.shape
    origin_vol = input_vol
    vol = input_vol
    sep_tag = 0
    sep_check = 1
    while sep_tag == 0:
        if sep_check == 1:
            test_vol = vol
        else:
            sq = 2 * sep_check - 1
            test_vol = fastmorph.spherical_open(vol.astype(np.uint32), radius=sep_check, anisotropy=(1,1,1), parallel=0).astype(int)
        test_label, test_num = scipy.ndimage.label(test_vol)
        areas = np.bincount(test_label.ravel())[1:]
        if len(areas) <= 2:
            test_ind = np.arange(len(areas))
        else:
            test_ind = np.argpartition(-areas, 2)[:2]
        if len(test_ind) == 1:
            sep_check += 1
        else:
            test_bone_1 = (test_label == (test_ind[0]+1)).astype(int)
            test_bone_2 = (test_label == (test_ind[1]+1)).astype(int)
            z_sum_1 = np.sum(test_bone_1, axis=(0, 1))
            z_sum_2 = np.sum(test_bone_2, axis=(0, 1))
            zerosum_1 = (z_sum_1 == 0).astype(int)
            zerosum_2 = (z_sum_2 == 0).astype(int)
            if ((np.sum(zerosum_1) == 0) or (np.sum(zerosum_2) == 0)):
                sep_check += 1
            else:
                _, _, test_z_1 = MaskCent(test_bone_1)
                _, _, test_z_2 = MaskCent(test_bone_2)
                if test_z_1 > test_z_2:
                    bone_up = test_bone_1
                    bone_down = test_bone_2
                else:
                    bone_up = test_bone_2
                    bone_down = test_bone_1
                sep_tag = 1

    max_h = np.zeros((vx, vy))
    min_h = np.zeros((vx, vy))
    for i in range(vx):
        for j in range(vy):
            max_h[i, j] = np.max(bone_down[i,j,:] * np.array(range(0, vz)))
            upz_vec = (bone_up[i, j, :]) * np.array(range(0, vz))
            if np.sum(upz_vec) == 0:
                min_h[i, j] = vz
            else:
                upz_vec = upz_vec[np.nonzero(upz_vec)]
                min_h[i,j] = np.min(upz_vec)

    #max_mask = (max_h > 0).astype(int)
    #contour = ski.segmentation.find_boundaries(vol, mode='inner').astype(np.uint8)
    #_, idx = ndimg.distance_transform_edt(contour, return_indices=True)
    #row, col = idx[0], idx[1]

    #valid = (max_h > 0).astype(np.uint8)
    #dist, idx = ndi.distance_transform_edt(~valid, return_indices=True)
    #row_valid, col_valid = idx[0], idx[1]
    #nozero_maxh = max_h.copy()

    valid = (max_h > 0).astype(np.uint8)
    dist, coords = ndi.distance_transform_edt(~valid, return_indices=True)
    filled_values = max_h[coords[0], coords[1]]
    nozero_maxh = np.where(valid, max_h, filled_values)

    #print("max_h.shape           :", max_h.shape)
    #print("valid.shape           :", valid.shape)
    #print("row_valid.shape       :", row_valid.shape)
    #print("col_valid.shape       :", col_valid.shape)

    # 看最大索引值是否超界
    #print("row_valid[~valid].max()  :", row_valid[~valid].max())
    #print("col_valid[~valid].max()  :", col_valid[~valid].max())
    #nozero_maxh[~valid] = max_h[row_valid[~valid], col_valid[~valid]]

    #nozero_maxh = max_h
    #for i in range(vx):
    #    for j in range(vy):
    #        if max_h[i, j] == 0:
    #            x_ind = int(row[i, j])
    #            y_ind = int(col[i, j])
    #            nozero_maxh[i, j] = max_h[x_ind, y_ind]
    res_h = min_h - max_h
    min_gap = math.ceil(np.min(res_h) / 2)
    nozero_maxh = nozero_maxh + np.min((2, min_gap))
    print("initial height finished")
    return nozero_maxh


def HeightLevel(init_h, seg, mu, lambd, timestep, iter_num):
    init_h = to_tensor(init_h)
    seg = to_tensor(seg)
    h_map = init_h.clone()

    vx, vy, vz = seg.shape
    gz = torch.gradient(seg, dim=2)[0]
    small_num = 1e-9
    for it in range(iter_num):
        h_map = torch.clamp(h_map, 0.0, float(vz - 1))
        gh = torch.gradient(h_map, dim=(0,1))
        gh_x = gh[0]
        gh_y = gh[1]

        s = torch.sqrt(gh_x**2 + gh_y**2 + small_num)
        N_ghx = gh_x / s
        N_ghy = gh_y / s
        curvature = Div2d(N_ghx, N_ghy)

        floor_idx = torch.floor(h_map).long()
        ceil_idx  = torch.ceil(h_map).long()
        frac = h_map - floor_idx.float()
        w_floor  = frac
        w_ceil = 1.0 - frac


        idx_floor = floor_idx.unsqueeze(-1)          # (H, W) → (H, W, 1)
        idx_ceil  = ceil_idx.unsqueeze(-1)

        # 4. gather 结果也是 (H, W, 1)
        gz_floor  = torch.gather(gz,   dim=2, index=idx_floor)  # → (H, W, 1)
        gz_ceil   = torch.gather(gz,   dim=2, index=idx_ceil)
        seg_floor = torch.gather(seg,  dim=2, index=idx_floor)
        seg_ceil  = torch.gather(seg,  dim=2, index=idx_ceil)

        # 5. 乘法前让权重有最后一维 1，强制广播
        w_floor = w_floor.unsqueeze(-1)              # (H, W) → (H, W, 1)
        w_ceil  = w_ceil.unsqueeze(-1)

        # 6. 计算（所有都是 (H, W, 1)）
        gz_h  = w_floor * gz_floor + w_ceil * gz_ceil
        seg_h = w_floor * seg_floor + w_ceil * seg_ceil

        # 7. 去掉最后一维
        gz_h  = gz_h.squeeze(-1)   # → (H, W)
        seg_h = seg_h.squeeze(-1)

        #x_idx = torch.arange(vx, device=h_map.device)[:, None]
        #y_idx = torch.arange(vy, device=h_map.device)[None, :]

        #gz_h = (w_floor * torch.gather(gz, dim=2, index=floor_idx.unsqueeze(-1)) + w_ceil  * torch.gather(gz, dim=2, index=ceil_idx.unsqueeze(-1))).squeeze(-1)

        #seg_h = (w_floor * torch.gather(seg, dim=2, index=floor_idx.unsqueeze(-1)) + w_ceil  * torch.gather(seg, dim=2, index=ceil_idx.unsqueeze(-1))).squeeze(-1)

        force = mu * (h_map * gz_h + seg_h) - lambd * curvature
        h_map = h_map - timestep * force

    h_map = torch.clamp(h_map, 0.0, float(vz - 1))
    h_map = h_map.cpu().numpy()

    return h_map


def IsoDiffusionFill(vol, timestep, iter_num):

    pad_v = math.ceil(timestep) + 2
    vol = np.pad(vol, ((pad_v,pad_v), (pad_v,pad_v), (pad_v,pad_v)), 'symmetric').astype(np.float32)
    vx, vy, vz = vol.shape
    for i in range(iter_num):
        a_time = time.time()
        isosurface = SignDistance_wnp(vol)
        print(np.max(isosurface))
        print(np.min(isosurface))
        b_time = time.time()
        print("Sign dist time: {}".format(b_time - a_time))
        #isosurface = NeumannBoundCond_np(isosurface)
        a_time = time.time()
        [g_x, g_y, g_z] = np.gradient(isosurface, edge_order=1)
        norm_mat = np.hypot(g_x, g_y, g_z)
        b_time = time.time()
        print("evolve time: {}".format(b_time - a_time))
        isosurface = isosurface + timestep * norm_mat

        vol = (isosurface > 0)#.astype(np.float32)
        a_time = time.time()
        #filled_slices = Parallel(n_jobs=-1, backend='threading')(delayed(fill_slice)(vol, sl) for sl in range(vz))
        #vol[:] = np.stack(filled_slices, axis=2)
        #vol = scipy.ndimage.binary_fill_holes(vol).astype(float)
        for sl in range(vz):
            tem = vol[:, :, sl]
            tem = scipy.ndimage.binary_fill_holes(tem).astype(float)
            vol[:,:,sl] = tem
        b_time = time.time()
        print("fill time: {}".format(b_time - a_time))

        isosurface = SignDistance_wnp(vol)
        [g_x, g_y, g_z] = np.gradient(isosurface, edge_order=1)
        norm_mat = np.hypot(g_x, g_y, g_z)
        isosurface = isosurface - timestep * norm_mat
        vol = (isosurface > 0)#.astype(np.float32)
        #vol = scipy.ndimage.binary_fill_holes(vol).astype(float)
        for sl in range(vz):
            tem = vol[:, :, sl]
            tem = scipy.ndimage.binary_fill_holes(tem).astype(float)
            vol[:, :, sl] = tem

    result_vol = vol[pad_v:vx-pad_v, pad_v:vy-pad_v, pad_v:vz-pad_v]
    return result_vol


def get_largest_component(vol):
    """
    返回体积中体素数最大的连通域（mask）和其大小
    vol: 二值 3D numpy 数组 (0/1)
    """
    if not np.any(vol):
        return np.zeros_like(vol, dtype=bool), 0

    labels, num = ndi.label(vol)                  # 26-连通
    if num == 0:
        return np.zeros_like(vol, dtype=bool), 0

    # 计算每个连通域的体素数
    areas = np.bincount(labels.ravel())[1:]       # [标签1, 标签2, ...] 的体积

    # 找到最大连通域的标签
    max_label = np.argmax(areas) + 1

    # 生成 mask
    largest = (labels == max_label)

    return largest, areas[max_label-1]


def split_by_height_where_numpy(vol, h_map):
    z = np.arange(vol.shape[2])[None, None, :]
    
    h_round = np.round(h_map)[:, :, None]
    h_floor = np.floor(h_map)[:, :, None]
    
    # 上半部分
    up_crop   = np.where(z >= h_round, vol, 0)
    
    # 下半部分
    down_crop = np.where(z <  h_floor, vol, 0)
    
    return up_crop, down_crop


def SepJoint(vol):
    vx, vy, vz = vol.shape
    CONNECT_BONE = 0

    con_lab, con_num = scipy.ndimage.label(vol)
    areas = np.bincount(con_lab.ravel())[1:]
    #print("area num: {}:".format(len(areas)))
    #print(np.max(areas))

    if len(areas) <= 2:
        ind = np.arange(len(areas))
    else:
        # 只找前 2 大，平均情況比完整 argsort 快 2–5 倍
        ind = np.argpartition(-areas, 2)[:2]
    #print(areas[ind[0]])
    #print(areas[ind[1]])

    #area_list = []
    #for i in range(con_num):
    #    area = scipy.ndimage.sum_labels(vol, con_lab, i+1)
    #    area_list.append(area)

    #ind = np.argsort(-np.array(area_list))
    if len(ind) < 2:
        CONNECT_BONE = 1
    else:
        part_bone_1 = (con_lab == (ind[0]+1)).astype(int)
        #print(np.sum(part_bone_1))
        part_bone_2 = (con_lab == (ind[1]+1)).astype(int)
        z_sum_1 = np.sum(part_bone_1, axis=(0,1))
        z_sum_2 = np.sum(part_bone_2, axis=(0,1))
        zerosum_1 = (z_sum_1 == 0).astype(int)
        zerosum_2 = (z_sum_2 == 0).astype(int)
        if ((np.sum(zerosum_1) == 0) or (np.sum(zerosum_2) == 0)):
            CONNECT_BONE = 1

    if CONNECT_BONE == 0:
        _, _, part_z_1 = MaskCent(part_bone_1)
        _, _, part_z_2 = MaskCent(part_bone_2)
        if part_z_1 > part_z_2:
            bone_up = part_bone_1
            bone_down = part_bone_2
        else:
            bone_up = part_bone_2
            bone_down = part_bone_1
        up_crop = bone_up
        down_crop = bone_down
    else:
        #fill_vol = IsoDiffusionFill(vol, 2, 1)
        fill_vol = morphological_closing_fastmorph(vol, 1)
        init_hmap = InitialHeight(fill_vol)
        a_time = time.time()
        h_map = HeightLevel(init_hmap, fill_vol, 1, 8, 0.1, 10000)
        b_time = time.time()
        print("height time comsume: {}".format(b_time - a_time))

        a_time = time.time()
        up_crop, down_crop = split_by_height_where_numpy(vol, h_map)
        b_time = time.time()
        print("crop time: {}".format(b_time - a_time))

    return up_crop, down_crop


def volume_to_smoothed_mesh(
    volume: np.ndarray,
    iso_value: float = 1.0,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),    # voxel spacing (x,y,z) in mm or your unit
    target_reduction: float = 0.98,                           # 0.0~1.0，越大简化越多（0.7 ≈ 保留约30%面数）
    smooth_iters: int = 8,                                   # Taubin 平滑迭代次数
    smooth_pass_band: float = 0.05,                           # 越小越平滑
    feature_angle: float = 120.0,                             # 保护特征的角度阈值（度）
    method: str = "marching_cubes" #"flying_edges"                              # "marching_cubes" 或 "flying_edges"（后者通常更快）
):
    """
    从 binary volume (0/1) 生成平滑且简化后的 triangular mesh。
    
    参数说明：
    - volume          : shape (nz, ny, nx) 或 (nx, ny, nz) 的 numpy 数组，值只有 0 和 1
    - iso_value       : 等值面阈值，对于 0/1 binary 通常设为 1.0
    - spacing         : 体素真实物理间距 (dx, dy, dz)，影响 mesh 的尺度
    - target_reduction: decimate 强度，0=不简化，0.9=去掉90%面数
    - smooth_iters    : Taubin 平滑迭代次数（保体积效果好）
    - smooth_pass_band: Taubin 平滑的频率参数（越小越平滑）
    - feature_angle   : 保护尖锐特征的角度（越大越保留棱角）
    - method          : 提取表面算法，"flying_edges" 通常比 marching_cubes 更快且质量相近

    返回：
    - pv.PolyData 对象（vertices + faces 的 mesh），可直接 .save() 或进一步处理
    """
    if not np.all(np.isin(volume, [0, 1])):
        raise ValueError("输入 volume 应为 binary，只包含 0 和 1")

    # 确保 volume 是 float32，避免后续问题
    volume = volume.astype(np.float32)

    # PyVista 的 ImageData 期望 dimensions=(nx, ny, nz)
    # 如果你的 volume 是 (nz, ny, nx) 这种医学常见顺序，需要转置
    # 这里假设输入已经是 (nx, ny, nz)，如不是请提前转置
    dims = volume.shape  # (nx, ny, nz)

    grid = pv.ImageData(
        dimensions=dims,
        spacing=spacing,
        origin=(0.0, 0.0, 0.0)
    )

    # 塞入数据，必须用 Fortran order (与 VTK 内存布局一致)
    grid.point_data["values"] = volume.ravel(order="F")

    # 提取表面
    mesh = grid.contour(
        isosurfaces=[iso_value],
        scalars="values",
        method=method,
        compute_scalars=False,   # 不需要保留标量
        compute_normals=True     # 生成法向量，便于后续渲染/平滑
    )

    if mesh.n_faces == 0:
        raise RuntimeError("没有提取到任何表面，可能 iso_value 设置不当或 volume 全 0/全 1")

    # 可选：清理孤立小片（视情况开启）
    # mesh = mesh.clean(tolerance=0.001)

    # 简化网格（downsample）
    if target_reduction > 0.0 and target_reduction < 1.0:
        mesh = mesh.decimate(target_reduction=target_reduction)

    # Taubin 平滑（推荐，保体积）
    if smooth_iters > 0:
        mesh = mesh.smooth_taubin(
            n_iter=smooth_iters,
            pass_band=smooth_pass_band,
            feature_angle=feature_angle,
            boundary_smoothing=False
        )

    # 可选：额外轻度 Laplacian 平滑（如果还不够光滑）
    # mesh = mesh.smooth(n_iter=5, relaxation_factor=0.2)

    # 清理可能的退化三角形
    #mesh = mesh.clean(tolerance=1e-6)

    return mesh



def nii2mesh(nii_name, prefix_name, if_return=False):
    total_time_a = time.time()
    #nii_name = "741_6918_RIGHT_MCP_2_reg.nii"

    nnunet_result = nnunet_predict(nii_path=nii_name,
            model_dir="./nnUNet_results/Dataset101_JointCT/nnUNetTrainer__nnUNetPlans__3d_fullres",
            )

    gc.collect()
    torch.cuda.empty_cache()
    unet_vol = to_tensor(nnunet_result.transpose(2, 1, 0).astype(np.float32))

    nii_vol = sitk.ReadImage(nii_name)
    nii_vol = sitk.GetArrayFromImage(nii_vol).astype(float)
    nii_vol = nii_vol.transpose(2, 1, 0)

    s_time = time.time()
    result = LevelSetSegmentation_torch(nii_vol, 15)

    result = result.cpu().numpy()
    e_time = time.time()
    print("segtime: {}".format(e_time - s_time))


    #result = SurfaceShrink_cmig(unet_vol, result, 2.0, 0.4, 1, 0, 1, 0.5, 5)
    result = SurfaceShrink(U=unet_vol, J=result, eps_gap=0.2, gamma=0.4, epsilon=1.0, timestep=0.5, iter_num=5)

    result = result.cpu().numpy()

    #mask = (result > 0).astype(np.uint8)
    #out = sitk.GetImageFromArray(np.ascontiguousarray(mask.transpose(2, 1, 0)))
    #sitk.WriteImage(out, "seg_test.nii")

    #result = unet_vol.cpu().numpy()
    print("start seperate")

    up_vol, down_vol = SepJoint(result)

    a_time = time.time()
    up_vol = up_vol.astype(float)
    down_vol = down_vol.astype(float)

    #up_vol = (IsoDiffusionFill(up_vol, 15, 1)).astype(float)
    #up_vol = morphological_closing_fastmorph(up_vol, 1)
    #down_vol = morphological_closing_fastmorph(down_vol, 1)

    up_vol,_ = get_largest_component(up_vol)
    down_vol,_ = get_largest_component(down_vol)

    #up_vol = morphological_closing_fastmorph(up_vol, 7)
    #down_vol = morphological_closing_fastmorph(down_vol, 7)

    b_time = time.time()
    #down_vol = (IsoDiffusionFill(down_vol, 15, 1)).astype(float)
    #print("iso diffus fill: {}".format(b_time - a_time))


    #a_time = time.time()
    #up_phi = SignDistance_wnp(up_vol)
    #up_phi = to_tensor(up_phi)
    #up_vol = to_tensor(up_vol)
    #up_phi = LSEdgeSmooth(up_phi, up_vol, 0.04, 10, 3, 2, 1.5, 15)
    #up_phi = up_phi.cpu().numpy()
    #up_vol = (up_phi > 0).astype(float)
    up_vol = np.pad(up_vol, 5, 'constant')


    #down_phi = SignDistance_wnp(down_vol)
    #down_phi = to_tensor(down_phi)
    #down_vol = to_tensor(down_vol)
    #down_phi = LSEdgeSmooth(down_phi, down_vol, 0.04, 10, 3, 2, 1.5, 15)
    #down_phi = down_phi.cpu().numpy()
    #down_vol = (down_phi > 0).astype(float)
    down_vol = np.pad(down_vol, 5, 'constant')
    b_time = time.time()
    #print("smooth time: {}".format(b_time - a_time))

    a_time = time.time()
    up_mesh = volume_to_smoothed_mesh(up_vol)
    up_mesh.save(prefix_name + "_up.ply")

    down_mesh = volume_to_smoothed_mesh(down_vol)
    down_mesh.save(prefix_name + "_down.ply")

    b_time = time.time()
    print("meshing time: {}".format(b_time - a_time))
    total_time_b = time.time()
    print("Total time for cortical surface reconstruction: {}".format(total_time_b - total_time_a))
    
    if if_return == True:
        return up_mesh, down_mesh


#nii_path = './SDL1220/585/585_2333_RIGHT_MCP_2.nii'
#nii2mesh(nii_path, 'test_s')
#'''
#main_folder = Path('./SDL1220')   # 如果是 SDLL12200 则改这里


#seg_result_file = "./segreult_file/"
# 获取所有直接子文件夹（排除 . 和 ..）
#sub_dirs = [p for p in main_folder.iterdir() if p.is_dir()]

# 用于存储符合条件的文件完整路径
#selected_files = []
#file_list = []           # 只存文件名（如果你还需要这个列表）

'''
for sub_dir in sub_dirs:
    # 获取该子文件夹下所有的 .nii 文件
    nii_files = list(sub_dir.glob('*.nii'))

    for nii_path in nii_files:
        file_name = nii_path.name                # 例如：xxx_aaa_bbb.nii
        base_name = nii_path.stem                # 去掉 .nii 后缀 → xxx_aaa_bbb

        print("Sample name: {}".format(base_name))
        # 用下划线分割
        tokens = base_name.split('_')

        # 如果任何一个部分是 'MASK' 或 'ER'，就跳过
        if 'MASK' in tokens or 'ER' in tokens:
            continue

        # 保留的文件
        selected_files.append(str(nii_path))     # 完整路径（字符串形式）
        file_list.append(base_name)              # 仅文件名
        prefix_name = seg_result_file + base_name
        nii2mesh(str(nii_path), prefix_name)

        torch.cuda.empty_cache()
        gc.collect()
'''        

#nii_name = "741_6918_RIGHT_MCP_2_reg.nii"
#prefix_name = "test_seg"
#nii2mesh(nii_name, prefix_name)
#'''
