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

from slice_close_fill import slice_close_fill

import gc
import json
from pathlib import Path
from typing import Dict
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


_PREDICTOR_CACHE = {}


def _as_numpy_mask(vol) -> np.ndarray:
    if isinstance(vol, torch.Tensor):
        vol = vol.detach().cpu().numpy()
    return np.asarray(vol) > 0


def build_outer_mask(J, close_radius: int = 2) -> np.ndarray:
    """Seal only the smallest pores, then fill. This is J_out, not the raw level set.

    close_radius is in voxels. 2 voxels is about 0.12-0.16 mm, at or below the
    SPECTRA minimum cortical break of 2 consecutive slices, so qualifying
    erosions are not closed before the boundary is extracted.
    """
    mask = _as_numpy_mask(J)
    if not np.any(mask):
        return mask
    radius = int(close_radius)
    if radius > 0:
        try:
            import fastmorph

            closed = fastmorph.spherical_close(
                mask.astype(np.uint32),
                radius=radius,
                parallel=0,
                anisotropy=(1.0, 1.0, 1.0),
            ) > 0
        except Exception:
            structure = ndi.generate_binary_structure(3, 1)
            closed = ndi.binary_closing(mask, structure=structure, iterations=radius)
    else:
        closed = mask
    return ndi.binary_fill_holes(closed)


def _fill_components(mask: np.ndarray, min_voxels: int = 50) -> np.ndarray:
    """Fill holes inside each bone, but keep every component.

    Up and down are separated later by SepJoint. Keeping only the largest
    component drops the smaller bone whenever the two sides are disconnected.
    """
    mask = mask.astype(bool)
    if not np.any(mask):
        return mask
    labels, n = ndi.label(mask)
    if n == 0:
        return mask
    counts = np.bincount(labels.ravel())
    out = np.zeros(mask.shape, dtype=bool)
    for lab in range(1, n + 1):
        if counts[lab] < min_voxels:
            continue
        out |= ndi.binary_fill_holes(labels == lab)
    return out


@torch.no_grad()
def SurfaceShrinkOuter(
    U,
    J,
    close_radius: int = 2,
    band: float = 4.0,
    eps_gap: float = 0.2,
    gamma: float = 0.4,
    epsilon: float = 1.0,
    timestep: float = 0.5,
    iter_num: int = 8,
):
    """Align dU to the outer cortical edge of J without opening the interior.

    Same update as SurfaceShrink, with three changes:
      1. J is replaced by J_out = Fill(Close(J)).
      2. The update is applied only where |phi| <= band, so trabecular
         gradients and the opposite bone cannot move the contour.
      3. Voxels deeper than `band` inside U are forced on, then holes are
         filled. Closure is therefore enforced, not hoped for.

    phi follows SignDistance: positive inside, negative outside.
    Returns a float tensor mask on the same device as to_tensor().
    """
    small = 1e-9
    U_np = _as_numpy_mask(U)
    J_out = build_outer_mask(J, close_radius=close_radius)

    # Deep interior of the UNet mask is topology, not something to evolve.
    dist_in = ndi.distance_transform_edt(U_np)
    deep = dist_in > float(band)

    w_np = (1.0 + float(eps_gap) - J_out.astype(np.float32)).clip(min=small)
    w = to_tensor(w_np)
    wx, wy, wz = torch.gradient(w, dim=(0, 1, 2))
    phi = SignDistance(U_np.astype(np.float32))
    band_t = float(band)

    for _ in range(int(iter_num)):
        phi = NeumannBoundCond(phi)
        dist_regu = DistReg(phi)

        phi_x, phi_y, phi_z = torch.gradient(phi, dim=(0, 1, 2))
        s = torch.sqrt(phi_x ** 2 + phi_y ** 2 + phi_z ** 2)
        Nx = phi_x / (s + small)
        Ny = phi_y / (s + small)
        Nz = phi_z / (s + small)
        curvature = Div(Nx, Ny, Nz)
        div_wN = wx * Nx + wy * Ny + wz * Nz + w * curvature
        dirac_phi = SmoothDirac_atan(phi, epsilon)
        update = gamma * dist_regu + dirac_phi * div_wN

        free = phi.abs() <= band_t
        phi = torch.where(free, phi + float(timestep) * update, phi)

    mask = (phi > 0).detach().cpu().numpy().astype(bool)
    mask[deep] = True
    # A voxel may only turn on inside the band if J_out says so. This blocks
    # the front from walking into the joint space or the other bone.
    outside = ~U_np
    band_out = outside & (ndi.distance_transform_edt(outside) <= float(band))
    mask[band_out] = J_out[band_out]
    mask[deep] = True
    mask = _fill_components(mask)
    return to_tensor(mask.astype(np.float32))


def fuse_outer_numpy(U, J, close_radius: int = 2, band: float = 4.0) -> np.ndarray:
    """Non-iterative form of the same constraint, useful as a debug baseline.

    Deep interior of U stays on. Within `band` voxels of dU, the label is
    copied from J_out. Everything farther outside U stays off.
    """
    U_np = _as_numpy_mask(U)
    J_out = build_outer_mask(J, close_radius=close_radius)
    dist_in = ndi.distance_transform_edt(U_np)
    dist_out = ndi.distance_transform_edt(~U_np)
    deep = dist_in > float(band)
    band_mask = np.where(U_np, dist_in, dist_out) <= float(band)
    fused = np.zeros(U_np.shape, dtype=bool)
    fused[deep] = True
    fused[band_mask] = J_out[band_mask]
    fused[deep] = True
    return _fill_components(fused)


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
    Input
        nii_path  : NIfTI file to segment
        model_dir : trained nnU-Net directory (the layer containing fold_0/)
    Output
        mask : uint8 {0,1}, shape (z, y, x), on CPU
    When keep_model=False, unload nnU-Net immediately after inference to free GPU memory for the level set.
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
    """Generate a 1D Gaussian kernel (separable)."""
    if size is None:
        size = int(4 * sigma + 1)   # usually 4~6 sigma covers enough
        size = size if size % 2 == 1 else size + 1
    x = torch.arange(-size//2 + 1, size//2 + 1, dtype=torch.float32)
    kernel = torch.exp(-x**2 / (2 * sigma**2))
    kernel = kernel / kernel.sum()          # normalize
    return kernel


def gaussian_blur3d(input_vol, sigma, kernel_size):
    """
    Apply Gaussian blur to a 3D volume (separable 1D convolutions ×3, much faster).
    
    input_vol: shape [B, C, D, H, W] or [C, D, H, W] or [D, H, W]
    sigma: a single float or (sigma_d, sigma_h, sigma_w)
    """
    if isinstance(sigma, (int, float)):
        sigma = (sigma, sigma, sigma)
    
    # Generate 1D kernel for each dimension
    kd = gaussian_kernel_1d(sigma[0], kernel_size)
    kh = gaussian_kernel_1d(sigma[1], kernel_size)
    kw = gaussian_kernel_1d(sigma[2], kernel_size)
    
    # Reshape to the form required by conv3d
    kd = kd.view(1, 1, -1, 1, 1)   # [1,1,kd,1,1]
    kh = kh.view(1, 1, 1, -1, 1)   # [1,1,1,kh,1]
    kw = kw.view(1, 1, 1, 1, -1)   # [1,1,1,1,kw]
    
    # Add batch & channel dimensions if needed
    if input_vol.dim() == 3:
        input_vol = input_vol.unsqueeze(0).unsqueeze(0)  # → [1,1,D,H,W]
    elif input_vol.dim() == 4:
        input_vol = input_vol.unsqueeze(0)               # → [1,C,D,H,W]
    
    padding_d = (kd.shape[2] - 1) // 2
    padding_h = (kh.shape[3] - 1) // 2
    padding_w = (kw.shape[4] - 1) // 2
    
    # Separable convolution: three 3D convs, each acting on one axis
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

    # Compute gradient (using torch.gradient)
    phi_x, phi_y, phi_z = torch.gradient(phi, dim=(0, 1, 2))

    # Compute gradient magnitude
    s = torch.sqrt(phi_x**2 + phi_y**2 + phi_z**2 + 1e-10)  # prevent division by zero

    # Define potential well regions
    a = (s >= 0) & (s <= 1)          # 0 ≤ s ≤ 1
    b = s > 1                         # s > 1

    # ps function (related to the derivative of the double-well potential)
    ps = torch.zeros_like(s)
    ps[a] = torch.sin(2 * torch.pi * s[a]) / (2 * torch.pi)
    ps[b] = s[b] - 1

    # dps/ds (derivative w.r.t. s)
    # Special handling when ps == 0 to avoid division by zero
    dps = torch.zeros_like(s)
    nonzero_ps = (ps != 0)
    zero_ps = ~nonzero_ps

    dps[nonzero_ps] = ps[nonzero_ps]
    # When ps ≈ 0 and s ≠ 0, dps/ds ≈ 1; when s=0 it can be set to 0 or another smooth value
    dps[zero_ps] = 1.0 * (s[zero_ps] != 0)

    # Compute vector field: dps * ∇φ - ∇φ = (dps - 1) * ∇φ
    vec_x = dps * phi_x - phi_x
    vec_y = dps * phi_y - phi_y
    vec_z = dps * phi_z - phi_z

    # Div( (dps-1) ∇φ ) + 4 * Laplacian(φ)
    div_term = Div(vec_x, vec_y, vec_z)
    lap_term = del3d(phi) * 4.0

    # Final regularization term
    f = div_term + lap_term

    return f


def SmoothDirac(phi, epsilon, device=None):
    if isinstance(phi, np.ndarray):
        phi = torch.from_numpy(phi).to(device=device, dtype=torch.float32)
    elif isinstance(phi, torch.Tensor):
        phi = phi.to(device=device, dtype=torch.float32)
    else:
        raise TypeError("Input phi must be numpy.ndarray or torch.Tensor")

    # Core part: (1/(2ε)) * (1 + cos(π phi / ε))
    cos_term = torch.cos(torch.pi * phi / epsilon)
    f = (1.0 / (2.0 * epsilon)) * (1.0 + cos_term)

    # Mask: non-zero only inside [-epsilon, epsilon]
    mask = (phi >= -epsilon) & (phi <= epsilon)

    # Apply mask (zero outside the interval)
    f = f * mask.float()   # mask is bool, convert to float (0/1)

    return f


def SmoothHeavi(phi, epsilon):
    """
    Smooth Heaviside step function (commonly used as a region indicator in level-set methods).

    When |phi| <= epsilon:
        H(phi) ≈ 0.5 * (1 + phi/ε + (1/π) sin(π phi / ε))
    When phi > epsilon:  H(phi) = 1
    When phi < -epsilon: H(phi) = 0

    Parameters:
        phi: input level-set function, numpy.ndarray or torch.Tensor
        epsilon: smoothing parameter (width), typically 1.0 ~ 2.0
        device: optional, compute device; auto-detected if not given

    Returns:
        torch.Tensor of the same shape as phi, smooth Heaviside values
    """
    if isinstance(phi, np.ndarray):
        phi = torch.from_numpy(phi).to(device=device, dtype=torch.float32)
    elif isinstance(phi, torch.Tensor):
        phi = phi.to(device=device, dtype=torch.float32)
    else:
        raise TypeError("Input phi must be numpy.ndarray or torch.Tensor")

    # Smooth part in the transition interval
    sin_term = torch.sin(torch.pi * phi / epsilon)
    f = 0.5 * (1.0 + phi / epsilon + (1.0 / torch.pi) * sin_term)

    # Define three mutually exclusive region masks
    inside  = (phi >= -epsilon) & (phi <= epsilon)     # transition region
    outside_pos = phi > epsilon                         # positive exterior → 1
    outside_neg = phi < -epsilon                        # negative exterior → 0

    # Combine results
    #heavi = torch.zeros_like(phi)
    #heavi[inside]      = f[inside]
    #heavi[outside_pos] = 1.0
    #heavi[outside_neg] = 0.0

    # Or the more concise broadcast form (numerically equivalent):
    heavi = f * inside.float() + 1.0 * outside_pos.float() + 0.0 * outside_neg.float()

    return heavi


def NeumannBoundCond_np(f):
    """
    More concise implementation using broadcasting and simultaneous assignment.
    """
    g = f.copy()

    g[[0, -1], :, :] = g[[1, -2], :, :]
    g[:, [0, -1], :] = g[:, [1, -2], :]
    g[:, :, [0, -1]] = g[:, :, [1, -2]]

    return g



def NeumannBoundCond(f):
    """
    Apply Neumann boundary condition (zero-gradient / reflective boundary).
    Mirror-pad the boundary so that the gradient is zero at the boundary.

    Parameters:
        f: input 3D array, numpy.ndarray or torch.Tensor
        device: optional compute device; auto-detected if not given

    Returns:
        torch.Tensor of the same shape as f with Neumann boundary condition applied
    """
    f = f.unsqueeze(0).unsqueeze(0)
    f_padded = torch.nn.functional.pad(f, pad=(1, 1, 1, 1, 1, 1), mode='replicate')

    f_padded = f_padded.squeeze(0).squeeze(0)
    # Crop back to original shape (remove padding)
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
    vol_0: template being shrunk (level set)
    vol:   original binary mask (nnU-Net)
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
    Uses exactly the same Gamma parameter estimation as the original code (k, theta, modify_k, modify_theta).
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
    # Edge + area terms
    # ────────────────────────────────────────────────
        edge_term = dirac_phi * curvature
        area_term = dirac_phi
        del curvature

    # ────────────────────────────────────────────────
    # External Gaussian term (basically original logic)
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
    # Internal Gamma term — using the original parameter estimation formulas
    # ────────────────────────────────────────────────
        inner_shift = inner_shift.min() - 0.5
        vol_shifted = vol - inner_shift

        log_vol = torch.where(vol_shifted > 0, torch.log(vol_shifted), torch.zeros_like(vol_shifted))

        sum_heavi     = heavi_phi.sum() + small
        sum_vol       = (vol_shifted * heavi_phi).sum()
        sum_logvol    = (log_vol     * heavi_phi).sum()
        sum_vol_logvol= (vol_shifted * log_vol * heavi_phi).sum()

    # Original numerator and denominator for k
        numerator_k   = sum_vol
        denominator_k = sum_vol_logvol - (sum_logvol * sum_vol / sum_heavi)
        k = numerator_k / (denominator_k + small)
        print(k)

    # Original theta
        #theta = sum_logvol / sum_heavi - (sum_vol * sum_logvol / (sum_heavi ** 2))
        theta = sum_vol_logvol / sum_heavi - (sum_vol * sum_logvol / (sum_heavi ** 2))
        print(theta)

    # Original correction term for modify_k
        correction = (3 * k - 2 * k / (3 + 3*k) - 4 * k / (5 + 10*k + 5*k**2)) / sum_heavi
        modify_k = k - correction

    # Original modify_theta
        modify_theta = (sum_heavi / (sum_heavi - 1 + small)) * theta

    # Core of the Gamma negative log-likelihood (using modify_k and modify_theta)
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
    # Assume vol is a binary array, 1=inside (object), 0=outside
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
    ori_vol,                  # can be numpy or torch.Tensor
    initial_vol,
    iter_num,
    input_PDE_step=None,      # not actually used in the original code, kept as a placeholder
    device=None,
    # Parameters for GammaLS_Autograd (adjustable as needed)
    gamma_ls_lr=0.02,
    gamma_ls_max_iter=20,
    reinitialize_every=0,
    use_dist_reg=False
):
    """
    PyTorch version of LevelSetSegmentation.
    Uses the autograd-optimized GammaLS_Segmentation as the core.
    """
    vol = ori_vol - np.min(ori_vol) + 1
    #print(np.min(vol))
    vol_smooth = ndimg.gaussian_filter(vol, sigma=0.2, mode='nearest', radius=5).astype(float)
    Vx, Vy, Vz = np.gradient(vol_smooth, edge_order=1)
    f = Vx * Vx + Vy * Vy + Vz * Vz
    g = 1.0 / (1.0 + f)

    c0 = 2
    vx, vy, vz = vol.shape
    #initialLSF = - c0 * np.ones((vx, vy, vz))
    initialLSF = initial_vol.cpu().numpy()
    #coe = 0.15
    #vx_cut = np.floor(vx * coe).astype(int)
    #vy_cut = np.floor(vy * coe).astype(int)
    #vz_cut = np.floor(vz * coe).astype(int)
    #initialLSF[vx_cut : vx-vx_cut, vy_cut : vy-vy_cut, vz_cut : vz-vz_cut] = c0
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
            iter_num=iter_num
            )

    # 6. Generate segmentation result
    seg_result = (phi_final > 0).float()   # keep float type, consistent with the original .astype(float)

    return seg_result


def fill_slice(vol, sl):
    slice_2d = vol[:, :, sl]
    # If vol is float, convert to bool first for more stable processing
    filled = ndi.binary_fill_holes(slice_2d)  # or slice_2d.astype(bool)
    return filled.astype(np.float32)


def morphological_closing_fastmorph(vol: np.ndarray, radius: int = 12):
    """Fastest CPU version (directly on binary volume)"""
    vol = vol.astype(np.uint32)  # fastmorph recommends uint* types

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

    # Check whether the maximum index exceeds bounds
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

        # 4. gather result is also (H, W, 1)
        gz_floor  = torch.gather(gz,   dim=2, index=idx_floor)  # → (H, W, 1)
        gz_ceil   = torch.gather(gz,   dim=2, index=idx_ceil)
        seg_floor = torch.gather(seg,  dim=2, index=idx_floor)
        seg_ceil  = torch.gather(seg,  dim=2, index=idx_ceil)

        # 5. Add a trailing dimension of 1 to the weights for broadcasting
        w_floor = w_floor.unsqueeze(-1)              # (H, W) → (H, W, 1)
        w_ceil  = w_ceil.unsqueeze(-1)

        # 6. Compute (all are (H, W, 1))
        gz_h  = w_floor * gz_floor + w_ceil * gz_ceil
        seg_h = w_floor * seg_floor + w_ceil * seg_ceil

        # 7. Remove the last dimension
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
    Return the largest connected component (mask) in the volume and its size.
    vol: binary 3D numpy array (0/1)
    """
    if not np.any(vol):
        return np.zeros_like(vol, dtype=bool), 0

    labels, num = ndi.label(vol)                  # 26-connectivity
    if num == 0:
        return np.zeros_like(vol, dtype=bool), 0

    # Compute voxel count of each connected component
    areas = np.bincount(labels.ravel())[1:]       # volumes of [label1, label2, ...]

    # Find the label of the largest connected component
    max_label = np.argmax(areas) + 1

    # Generate mask
    largest = (labels == max_label)

    return largest, areas[max_label-1]


def split_by_height_where_numpy(vol, h_map):
    z = np.arange(vol.shape[2])[None, None, :]
    
    h_round = np.round(h_map)[:, :, None]
    h_floor = np.floor(h_map)[:, :, None]
    
    # Upper half
    up_crop   = np.where(z >= h_round, vol, 0)
    
    # Lower half
    down_crop = np.where(z <  h_floor, vol, 0)
    
    return up_crop, down_crop


def sep_joint_unet(vol, unet_vol):
    init_hmap = InitialHeight(unet_vol)
    a_time = time.time()
    h_map = HeightLevel(init_hmap, unet_vol, 1, 8, 0.1, 10000)
    b_time = time.time()
    print("height time comsume: {}".format(b_time - a_time))

    a_time = time.time()
    up_crop, down_crop = split_by_height_where_numpy(vol, h_map)
    b_time = time.time()
    print("crop time: {}".format(b_time - a_time))

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
        # Only find the top-2 largest; on average 2–5× faster than a full argsort
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
    target_reduction: float = 0.98,                           # 0.0~1.0, larger means more simplification (0.7 ≈ keep ~30% faces)
    smooth_iters: int = 8,                                   # Taubin smoothing iterations
    smooth_pass_band: float = 0.05,                           # smaller means smoother
    feature_angle: float = 120.0,                             # angle threshold (degrees) for protecting features
    method: str = "marching_cubes" #"flying_edges"                              # "marching_cubes" or "flying_edges" (the latter is usually faster)
):
    """
    Generate a smoothed and simplified triangular mesh from a binary volume (0/1).
    
    Parameters:
    - volume          : numpy array of shape (nz, ny, nx) or (nx, ny, nz), values only 0 and 1
    - iso_value       : isosurface threshold, usually 1.0 for 0/1 binary
    - spacing         : physical voxel spacing (dx, dy, dz), affects mesh scale
    - target_reduction: decimation strength, 0=no simplification, 0.9=remove 90% of faces
    - smooth_iters    : Taubin smoothing iterations (good volume preservation)
    - smooth_pass_band: frequency parameter of Taubin smoothing (smaller = smoother)
    - feature_angle   : angle for protecting sharp features (larger = keep more edges)
    - method          : surface extraction algorithm, "flying_edges" is usually faster than marching_cubes with similar quality

    Returns:
    - pv.PolyData object (vertices + faces mesh), can be .save()'d or further processed
    """
    if not np.all(np.isin(volume, [0, 1])):
        raise ValueError("Input volume must be binary, containing only 0 and 1")

    # Ensure volume is float32 to avoid later issues
    volume = volume.astype(np.float32)

    # PyVista ImageData expects dimensions=(nx, ny, nz)
    # If your volume is (nz, ny, nx) (common in medical imaging), transpose it first
    # Here we assume the input is already (nx, ny, nz); transpose beforehand if not
    dims = volume.shape  # (nx, ny, nz)

    grid = pv.ImageData(
        dimensions=dims,
        spacing=spacing,
        origin=(0.0, 0.0, 0.0)
    )

    # Insert data, must use Fortran order (consistent with VTK memory layout)
    grid.point_data["values"] = volume.ravel(order="F")

    # Extract surface
    mesh = grid.contour(
        isosurfaces=[iso_value],
        scalars="values",
        method=method,
        compute_scalars=False,   # no need to keep scalars
        compute_normals=True     # generate normals for later rendering/smoothing
    )

    if mesh.n_faces == 0:
        raise RuntimeError("No surface extracted, possibly wrong iso_value or volume is all 0 / all 1")

    # Optional: clean isolated small pieces (enable as needed)
    # mesh = mesh.clean(tolerance=0.001)

    # Simplify mesh (downsample)
    if target_reduction > 0.0 and target_reduction < 1.0:
        mesh = mesh.decimate(target_reduction=target_reduction)

    # Taubin smoothing (recommended, volume-preserving)
    if smooth_iters > 0:
        mesh = mesh.smooth_taubin(
            n_iter=smooth_iters,
            pass_band=smooth_pass_band,
            feature_angle=feature_angle,
            boundary_smoothing=False
        )

    # Optional: extra light Laplacian smoothing (if still not smooth enough)
    # mesh = mesh.smooth(n_iter=5, relaxation_factor=0.2)

    # Clean possible degenerate triangles
    #mesh = mesh.clean(tolerance=1e-6)

    return mesh


def compute_spectra_jsw(
    up_vol: np.ndarray,
    down_vol: np.ndarray,
    voxel_size_mm: float = 0.061,
    dilation_voxels: int = 20,
    erosion_voxels: int = 26,
    min_joint_voxels: int = 100,
) -> Dict[str, float]:
    """
    Compute 3D JSW following the Stok et al. 2020 SPECTRA consensus method.
    
    Parameters
    ----
    up_vol, down_vol : binary arrays (0/1 or bool) of the same shape, distal (upper) and proximal (lower)
    voxel_size_mm    : isotropic voxel size (mm). XtremeCT II commonly 0.061, XtremeCT I commonly 0.082
    dilation_voxels  : dilation iterations (consensus default 20, for ~82 µm)
    erosion_voxels   : erosion iterations (consensus default 26, more than dilation, more conservative)
    min_joint_voxels : treat as failure if joint-space voxels are too few
    
    Returns
    ----
    dict: JSV, JSW_Mean, JSW_Min, JSW_Max, JSW_SD, JSW_Asymm
    """
    up = (up_vol > 0).astype(bool)
    down = (down_vol > 0).astype(bool)
    if up.shape != down.shape:
        raise ValueError("up_vol and down_vol must have the same shape")
    if not (np.any(up) and np.any(down)):
        raise ValueError("One of the bone masks is empty")

    bones = up | down

    # ---------- 1. Morphologically define joint-space VOI (key consensus step) ----------
    # Prefer fastmorph (faster, spherical kernel); fall back to scipy on failure
    try:
        # fastmorph spherical_close is a combination of dilation+erosion
        # Here we do them separately for precise control of different dilation/erosion
        dilated = fastmorph.spherical_dilate(
            bones.astype(np.uint32),
            radius=dilation_voxels,
            parallel=0,
            anisotropy=(1.0, 1.0, 1.0),
        ) > 0
        closed = fastmorph.spherical_erode(
            dilated.astype(np.uint32),
            radius=erosion_voxels,
            parallel=0,
            anisotropy=(1.0, 1.0, 1.0),
        ) > 0
    except Exception:
        structure = ndi.generate_binary_structure(3, 1)  # 6-connectivity cube approximation
        dilated = ndi.binary_dilation(bones, structure=structure, iterations=dilation_voxels)
        closed = ndi.binary_erosion(dilated, structure=structure, iterations=erosion_voxels)

    joint_space = closed & ~bones

    # Remove possible residual small components (non-joint space)
    labels, n = ndi.label(joint_space)
    if n == 0:
        raise RuntimeError("No joint space volume was created. Try larger dilation or check masks.")
    sizes = np.bincount(labels.ravel())
    largest = sizes[1:].argmax() + 1
    joint_space = (labels == largest)

    n_voxels = int(joint_space.sum())
    if n_voxels < min_joint_voxels:
        raise RuntimeError(f"Joint space too small ({n_voxels} voxels). Check dilation/erosion or masks.")

    # ---------- 2. Local thickness (distance transform + sphere-diameter approximation) ----------
    # edt returns Euclidean distance to nearest background (in voxel units)
    # Local thickness ≈ 2 * distance (diameter), a common sphere-fitting approximation in the literature
    dt = edt.edt(joint_space, anisotropy=(voxel_size_mm, voxel_size_mm, voxel_size_mm), parallel=0)
    local_thickness = 2.0 * dt  # mm

    # Statistics only inside the joint space (light erosion can exclude unstable boundary values; kept consistent with consensus)
    thickness_values = local_thickness[joint_space]
    thickness_values = thickness_values[thickness_values > 0]

    if thickness_values.size == 0:
        raise RuntimeError("No valid thickness values")

    # ---------- 3. Metrics ----------
    voxel_vol = voxel_size_mm ** 3
    jsv = n_voxels * voxel_vol

    jsw_mean = float(np.mean(thickness_values))
    jsw_min  = float(np.min(thickness_values))
    jsw_max  = float(np.max(thickness_values))
    jsw_sd   = float(np.std(thickness_values))
    jsw_asymm = jsw_max / jsw_min if jsw_min > 1e-6 else np.inf

    return {
        "JSV": jsv,
        "JSW_Mean": jsw_mean,
        "JSW_Min": jsw_min,
        "JSW_Max": jsw_max,
        "JSW_SD": jsw_sd,
        "JSW_Asymm": jsw_asymm,
        "n_voxels": n_voxels,
    }


def nii2mesh(nii_name, prefix_name, if_return=False):
    total_time_a = time.time()
    #nii_name = "741_6918_RIGHT_MCP_2_reg.nii"

    nnunet_result = nnunet_predict(nii_path=nii_name,
            model_dir="/home/xzhangem/npj/OAReg/src/nnUNet_results/Dataset101_JointCT/nnUNetTrainer__nnUNetPlans__3d_fullres",
            )

    gc.collect()
    torch.cuda.empty_cache()
    unet_vol = to_tensor(nnunet_result.transpose(2, 1, 0).astype(np.float32))

    nii_vol = sitk.ReadImage(nii_name)
    nii_vol = sitk.GetArrayFromImage(nii_vol).astype(float)
    nii_vol = nii_vol.transpose(2, 1, 0)

    s_time = time.time()
    result = LevelSetSegmentation_torch(nii_vol, unet_vol, 3)

    result = result.cpu().numpy()
    e_time = time.time()
    print("segtime: {}".format(e_time - s_time))

    up_vol, down_vol = sep_joint_unet(result, unet_vol.cpu().numpy())
    up_unet_vol, down_unet_vol = sep_joint_unet(unet_vol.cpu().numpy(), unet_vol.cpu().numpy())

    up_vol = up_vol.astype(float)
    down_vol = down_vol.astype(float)

    up_unet_vol = up_unet_vol.astype(float)
    down_unet_vol = down_unet_vol.astype(float)

    up_vol,_ = get_largest_component(up_vol)
    down_vol,_ = get_largest_component(down_vol)

    up_unet_vol,_ = get_largest_component(up_unet_vol)
    down_unet_vol,_ = get_largest_component(down_unet_vol)

    up_vol = slice_close_fill(up_vol, up_unet_vol, axis=2, min_radius=4, max_radius=12, boundary_ratio_min=0.8, boundary_ratio_max=1.2)
    down_vol = slice_close_fill(down_vol, down_unet_vol, axis=2, min_radius=4, max_radius=12, boundary_ratio_min=0.8, boundary_ratio_max=1.2)
    print(up_vol.shape)

    up_vol = (up_vol > 0).astype(np.uint8)
    #up_vol = sitk.GetImageFromArray(np.ascontiguousarray(up_vol.transpose(2, 1, 0)))
    #sitk.WriteImage(up_vol, "up_vol_test.nii")

    down_vol = (down_vol > 0).astype(np.uint8)
    #down_vol = sitk.GetImageFromArray(np.ascontiguousarray(down_vol.transpose(2, 1, 0)))
    #sitk.WriteImage(down_vol, "down_vol_test.nii")

    lvl_vol = ((up_vol + down_vol) > 0).astype(np.uint8)
    lvl_vol = sitk.GetImageFromArray(np.ascontiguousarray(lvl_vol.transpose(2, 1, 0)))
    sitk.WriteImage(lvl_vol, "lvl_test.nii")

    unet_vol = unet_vol.cpu().numpy()
    unet_vol = (unet_vol > 0).astype(np.uint8)
    unet_vol = sitk.GetImageFromArray(np.ascontiguousarray(unet_vol.transpose(2, 1, 0)))
    sitk.WriteImage(unet_vol, 'unet_test.nii')


    #result = SurfaceShrink_cmig(unet_vol, result, 2.0, 0.4, 1, 0, 1, 0.5, 5)
    #result = SurfaceShrink(U=unet_vol, J=result, eps_gap=0.2, gamma=0.4, epsilon=1.0, timestep=0.5, iter_num=5)
    ##### result = SurfaceShrinkOuter(U=unet_vol,J=result,close_radius=4, band=4.0,eps_gap=0.2, gamma=0.4, epsilon=1.0, timestep=0.5, iter_num=8)

    #lvlset_result = result
    #lvlset_mask = (lvlset_result > 0).astype(np.uint8)
    #lvlset_out = sitk.GetImageFromArray(np.ascontiguousarray(lvlset_mask.transpose(2, 1, 0)))
    #sitk.WriteImage(lvlset_out, "seg_lvlset_test.nii")

    #result = unet_vol

    #result = result.cpu().numpy()

    #mask = (result > 0).astype(np.uint8)
    #out = sitk.GetImageFromArray(np.ascontiguousarray(mask.transpose(2, 1, 0)))
    #sitk.WriteImage(out, "seg_test.nii")

    #result = unet_vol.cpu().numpy()
    print("start seperate")

    #up_vol, down_vol = SepJoint(result)

    a_time = time.time()
    #up_vol = up_vol.astype(float)
    #down_vol = down_vol.astype(float)

    #up_vol = (IsoDiffusionFill(up_vol, 15, 1)).astype(float)
    #up_vol = morphological_closing_fastmorph(up_vol, 1)
    #down_vol = morphological_closing_fastmorph(down_vol, 1)

    #up_vol,_ = get_largest_component(up_vol)
    #down_vol,_ = get_largest_component(down_vol)

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

    jsw_metrics = compute_spectra_jsw(up_vol, down_vol, voxel_size_mm=0.082, dilation_voxels=20, erosion_voxels=26)
    print(jsw_metrics)
    jsw_path = prefix_name + "_jsw.json"
    with open(jsw_path, "w", encoding="utf-8") as f:
        json.dump(jsw_metrics, f, indent=2)
    print("Wrote {}".format(jsw_path))
    
    up_vol = np.pad(up_vol, 5, 'constant')
    up_unet_vol = np.pad(up_unet_vol, 5, 'constant')


    #down_phi = SignDistance_wnp(down_vol)
    #down_phi = to_tensor(down_phi)
    #down_vol = to_tensor(down_vol)
    #down_phi = LSEdgeSmooth(down_phi, down_vol, 0.04, 10, 3, 2, 1.5, 15)
    #down_phi = down_phi.cpu().numpy()
    #down_vol = (down_phi > 0).astype(float)
    down_vol = np.pad(down_vol, 5, 'constant')
    down_unet_vol = np.pad(down_unet_vol, 5, 'constant')
    b_time = time.time()
    #print("smooth time: {}".format(b_time - a_time))

    a_time = time.time()
    up_mesh = volume_to_smoothed_mesh(up_vol)
    up_mesh.save(prefix_name + "_up.ply")
    up_unet_mesh = volume_to_smoothed_mesh(up_unet_vol)
    up_unet_mesh.save(prefix_name + "_unet_up.ply")

    down_mesh = volume_to_smoothed_mesh(down_vol)
    down_mesh.save(prefix_name + "_down.ply")
    down_unet_mesh = volume_to_smoothed_mesh(down_unet_vol)
    down_unet_mesh.save(prefix_name + "_unet_down.ply")

    b_time = time.time()
    print("meshing time: {}".format(b_time - a_time))
    total_time_b = time.time()
    print("Total time for cortical surface reconstruction: {}".format(total_time_b - total_time_a))
    
    if if_return == True:
        return up_mesh, down_mesh


#nii_path = './SDL1220/585/585_2333_RIGHT_MCP_2.nii'
#nii2mesh(nii_path, 'test_s')
#'''
#main_folder = Path('./SDL1220')   # change here if it is SDLL12200


#seg_result_file = "./segreult_file/"
# Get all immediate subfolders (exclude . and ..)
#sub_dirs = [p for p in main_folder.iterdir() if p.is_dir()]

# Store full paths of files that meet the criteria
#selected_files = []
#file_list = []           # store only filenames (if you still need this list)

'''
for sub_dir in sub_dirs:
    # Get all .nii files in this subfolder
    nii_files = list(sub_dir.glob('*.nii'))

    for nii_path in nii_files:
        file_name = nii_path.name                # e.g. xxx_aaa_bbb.nii
        base_name = nii_path.stem                # remove .nii suffix → xxx_aaa_bbb

        print("Sample name: {}".format(base_name))
        # Split by underscore
        tokens = base_name.split('_')

        # Skip if any part is 'MASK' or 'ER'
        if 'MASK' in tokens or 'ER' in tokens:
            continue

        # Keep the file
        selected_files.append(str(nii_path))     # full path (as string)
        file_list.append(base_name)              # filename only
        prefix_name = seg_result_file + base_name
        nii2mesh(str(nii_path), prefix_name)

        torch.cuda.empty_cache()
        gc.collect()
'''        

#nii_name = "741_6918_RIGHT_MCP_2_reg.nii"
#prefix_name = "test_seg"
#nii2mesh(nii_name, prefix_name)
#'''
