import os
import open3d as o3d
import trimesh
import numpy as np
from scipy.spatial import KDTree

import glob
from tqdm import tqdm
import argparse
import time
import sys
sys.path.append("../")
import torch
from torch import nn
import torch.nn.functional as F
import pytorch3d
from pytorch3d.io import load_ply
from utils2.normalize_pointcloud import normalize_ply,normalize_ply_file
from utils2.loss_functions import correntropy_chamfer_distance
from model.model import Siren

from utils2.pc_DDG import estimate_velocity_gradient_torch, LocalPCEnergy, LocalPCEnergy_Grass, LocalPCEnergy_Grass_knn

from pytorch3d.ops import knn_points

from pytorch3d.loss import chamfer_distance
from pytorch3d.ops import estimate_pointcloud_normals

from pytorch3d.ops.points_alignment import corresponding_points_alignment, _apply_similarity_transform, iterative_closest_point

from collections import namedtuple
import potpourri3d as pp3d

import robust_laplacian

from torch.optim.lr_scheduler import _LRScheduler



from utils2.mesh_DDG import mesh_gesm
from utils2.normalize_mesh import normalize_mesh_file

from torch_geometric.nn import PointNetConv, global_max_pool, radius_graph
from torch_geometric.data import Data

torch.cuda.set_device(0)
DEVICE = 'cuda'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:32"

# Default checkpoint: same directory as this file (pointnet2_best.pth).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_POINTNET_CKPT = os.path.join(_THIS_DIR, "pointnet2_best.pth")


# ---------------------------------------------------------------------------
# Frozen PointNet++ discriminator (keep in sync with train.py)
# Label 0 = healthy, 1 = diseased. Forward returns a single diseased-class logit.
# ---------------------------------------------------------------------------
class VariablePointNetPlusPlusBinary(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()

        self.sa1 = PointNetConv(
            local_nn=nn.Sequential(
                nn.Linear(3 + 3, 64), nn.BatchNorm1d(64), nn.ReLU(),
                nn.Linear(64, 128), nn.BatchNorm1d(128), nn.ReLU()
            ),
            global_nn=nn.Sequential(
                nn.Linear(128, 256), nn.BatchNorm1d(256), nn.ReLU()
            )
        )

        self.sa2 = PointNetConv(
            local_nn=nn.Sequential(
                nn.Linear(3 + 256, 256), nn.BatchNorm1d(256), nn.ReLU(),
                nn.Linear(256, 512), nn.BatchNorm1d(512), nn.ReLU()
            ),
            global_nn=nn.Sequential(
                nn.Linear(512, 1024), nn.BatchNorm1d(1024), nn.ReLU()
            )
        )

        self.fc1 = nn.Linear(1024, 512)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, 1)
        self.bn1 = nn.LayerNorm(512)
        self.bn2 = nn.LayerNorm(256)

    def forward(self, data):
        pos, batch = data.pos, data.batch

        edge_index1 = radius_graph(pos, r=0.5, batch=batch, max_num_neighbors=64, loop=False)
        x1 = self.sa1(pos, pos, edge_index=edge_index1)

        edge_index2 = radius_graph(pos, r=1.0, batch=batch, max_num_neighbors=128, loop=False)
        x2 = self.sa2(x1, pos, edge_index=edge_index2)

        x = global_max_pool(x2, batch)

        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        logit = self.fc3(x)
        return logit


def load_pointnet_classifier(ckpt_path=DEFAULT_POINTNET_CKPT, device=DEVICE):
    """Load a frozen PointNet++ binary classifier. Weights are not optimized."""
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            "PointNet++ checkpoint not found: {}. "
            "Place pointnet2_best.pth next to gesm_match.py, or pass ckpt_path.".format(ckpt_path)
        )

    classifier = VariablePointNetPlusPlusBinary()
    try:
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(ckpt_path, map_location=device)

    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}

    classifier.load_state_dict(state)
    classifier.to(device)
    classifier.eval()
    for p in classifier.parameters():
        p.requires_grad = False
    return classifier


def _vertices_to_pyg(verts):
    """Pack a single [N, 3] (or [1, N, 3]) cloud as a PyG Data object."""
    pos = verts.reshape(-1, 3).contiguous()
    batch = torch.zeros(pos.shape[0], dtype=torch.long, device=pos.device)
    return Data(pos=pos, batch=batch)


def healthy_class_prob(classifier, verts):
    """
    P(C(verts) = healthy). Differentiable w.r.t. vertex coordinates.

    train.py uses BCEWithLogitsLoss on a single logit for the diseased class,
    so P(healthy) = sigmoid(-logit).
    """
    logit = classifier(_vertices_to_pyg(verts))
    return torch.sigmoid(-logit).reshape(())


class ICP(nn.Module):
    def __init__(self, correspondence):
        """ ICP alignment implementation in torch.
        """
        super(ICP, self).__init__()
        self.correspondence = correspondence

    def forward(self, from_vertices, to_vertices):
        full_source = from_vertices.clone()
        start_shape = full_source.shape

        # Apply correct initial alignment
        if self.correspondence:
            R, T, s = corresponding_points_alignment(
                from_vertices,
                to_vertices,
                weights=None,
                estimate_scale=False,
                allow_reflection=False,
            )

            from_vertices = _apply_similarity_transform(from_vertices, R, T, s)

        icp = iterative_closest_point(from_vertices, to_vertices, relative_rmse_thr=1e-5)
        from_vertices = icp.Xt

        assert start_shape == from_vertices.shape, "Shape mismatch"
        return from_vertices



def deform_mesh(model, v_src=None, f_src=None, v_trg=None, f_trg=None,
        n_steps=200, sigma2=1.0, init_lr=1.0e-4,
        esm_weight=1.0e2, dist_weight=1.0e4, eval_every_nth_step=100, scale=1,
        classifier=None, mu=0.2):
    """
    Test-time SIREN optimization of the displacement field D.

    L = a E_shr + b E_bld + c E_smth + λ Chamfer(S̄ + D, S)
        - μ P(C(S̄ + D) = healthy)     # if classifier is provided

    The classifier is a frozen pretrained PointNet++; only SIREN parameters
    are updated. μ = 0.2 matches Eq. (e::adversial_esm).
    """

    model = model.train()
    optm = torch.optim.Adam(model.parameters(), lr=init_lr, betas=(0.9, 0.999), eps=1.0e-8)
    schedm = torch.optim.lr_scheduler.ReduceLROnPlateau(optm, verbose=True, patience=1)

    dist_loss_total = 0.0
    deform_loss_total = 0.0
    class_loss_total = 0.0
    total_loss = 0.0
    n_r = 0

    for step in range(0, n_steps):
        loss = 0.0
        v_deformed = v_src.clone()
        for s in range(scale):
            vel_field = model(v_deformed)
            deform_loss = esm_weight * mesh_gesm(v_deformed, f_src, vel_field, 0.0, 500.0, 0.0, 500.0, 0.0, 500.0) #/ (3**scale)
            deform_loss_total = deform_loss_total + deform_loss
            v_deformed = v_deformed.detach().clone() + vel_field

        #vel_field = model(v_src)
        #deform_loss = esm_weight * mesh_gesm(v_src, f_src, vel_field, 1.0, 100.0, 1.0, 10.0, 1.0, 50.0)
        #deform_loss_total = deform_loss_total + deform_loss
        #v_deformed = v_src + vel_field
        loss += deform_loss

        dist_loss = dist_weight * correntropy_chamfer_distance(v_deformed.unsqueeze(0),v_trg.unsqueeze(0),step,x_normals=None, y_normals=None, sigma2=sigma2)

        dist_loss_total += dist_loss

        loss += dist_loss

        # GAN-style healthy prior: min L_total - μ P(healthy)
        prob_healthy = None
        class_term = 0.0
        if classifier is not None and mu != 0.0:
            prob_healthy = healthy_class_prob(classifier, v_deformed)
            class_term = -mu * prob_healthy
            class_loss_total = class_loss_total + class_term
            loss = loss + class_term

        total_loss += loss
        n_r += 1

        optm.zero_grad()

        loss.backward()

        optm.step()

        if step % eval_every_nth_step == 0:
            dist_loss_total /= n_r
            deform_loss_total /= n_r
            total_loss /= n_r
            if classifier is not None:
                class_loss_total = class_loss_total / n_r
                p_h = float(prob_healthy.detach()) if prob_healthy is not None else float("nan")
                print("step {:03d} | deform {:.6f} | dist {:.6f} | cls {:.6f} | p_healthy {:.4f} | loss {:.6f}".format(
                    step, float(deform_loss_total), float(dist_loss_total),
                    float(class_loss_total), p_h, float(total_loss)))
            else:
                print("step {:03d} | deform {:.6f} | dist {:.6f} | loss {:.6f}".format(
                    step, float(deform_loss_total), float(dist_loss_total), float(total_loss)))

            schedm.step(float(total_loss))

            dist_loss_total = 0
            deform_loss_total = 0
            class_loss_total = 0.0
            total_loss = 0
            n_r = 0

    #dist_loss_total /= n_r
    #deform_loss_total /= n_r
    #total_loss /= n_r


def gesm_matching(name, v_src, f_src, v_trg, f_trg, target_normal_scale, target_normal_center, n_steps=200, sigma2=1.0, init_lr=1.0e-4, esm_weight=1.0e2,
        dist_weight=1.0e3, scale=1, classifier=None, mu=0.1):

    model = Siren(in_features=3,
                    hidden_features=128,
                    hidden_layers=3,
                    out_features=3, outermost_linear=True,
                    first_omega_0=30, hidden_omega_0=30.).to(DEVICE).train()

    deform_mesh(model, v_src, f_src, v_trg, f_trg, n_steps=n_steps, sigma2=sigma2, init_lr=init_lr,
            esm_weight=esm_weight, dist_weight=dist_weight, eval_every_nth_step=100, scale=scale,
            classifier=classifier, mu=mu)

    
    model.eval()
    state_num = scale
    #print("state num for testing: {}".format(state_num))
    inter_result = []
    vpred = v_src
    for l in range(state_num):
        vpred = vpred + model(vpred).detach().clone()
        vpred_np = target_normal_scale * vpred.cpu().numpy() + target_normal_center
        inter_result.append(vpred_np)

    return inter_result


def neural_GESM(source_name, target_name, save_name, if_return=False,
                use_classifier=True, ckpt_path=None, mu=0.5,
                n_steps=300, init_lr=1.0e-4,
                esm_weight=2.0e3, dist_weight=1.0e2, sigma2=50.0, scale=1):
    """
    Deform the healthy template (source) toward the sample (target).

    With use_classifier=True this optimizes Eq. (e::adversial_esm):
        min_D  L_total(S_bar, S; D) - μ P(C(S_bar + D) = healthy)
    using a frozen PointNet++ loaded from pointnet2_best.pth (same directory).

    Set use_classifier=False to recover the original ESSM-only matching.
    """
    a_time = time.time()
    #source_name = './704_2924_L_MCP_2_down.ply' #'./pancreas_dataset/pancreas_001.ply'
    #target_name = './741_3090_MCP_2_BL_down.ply' #'./pancreas_dataset/pancreas_078.ply'

    icp_no_corr = ICP(correspondence=False).to(DEVICE)

    nl_src_vert, src_face, src_center, src_scale = normalize_mesh_file(source_name)
    nl_trg_vert, trg_face, trg_center, trg_scale = normalize_mesh_file(target_name)

    nl_src_vert = torch.from_numpy(nl_src_vert).to(DEVICE)
    nl_trg_vert = torch.from_numpy(nl_trg_vert).to(DEVICE)

    nl_src_vert = icp_no_corr(nl_src_vert.unsqueeze(0), nl_trg_vert.unsqueeze(0))
    nl_src_vert = nl_src_vert.squeeze(0)

    src_face = torch.from_numpy(src_face).to(DEVICE)
    trg_face = torch.from_numpy(trg_face).to(DEVICE)

    src_num = nl_src_vert.shape[0]
    trg_num = nl_trg_vert.shape[0]

    state_num = 1

    classifier = None
    if use_classifier:
        if ckpt_path is None:
            ckpt_path = DEFAULT_POINTNET_CKPT
        classifier = load_pointnet_classifier(ckpt_path, device=DEVICE)
        print("Loaded frozen PointNet++ from {} (mu={})".format(ckpt_path, mu))

    results = gesm_matching(None, nl_src_vert, src_face, nl_trg_vert, trg_face, trg_scale, trg_center,
            n_steps=n_steps, sigma2=sigma2,
            init_lr=init_lr, esm_weight=esm_weight, dist_weight=dist_weight, scale=scale,
            classifier=classifier, mu=mu)

    print(len(results))
    print(results[-1].shape)

    result_tensor = torch.from_numpy(results[-1]).to(DEVICE)
    target_tensor = nl_trg_vert.cpu().numpy() * trg_scale + trg_center
    target_tensor = torch.from_numpy(target_tensor).to(DEVICE)
    loss, _ = chamfer_distance(result_tensor.unsqueeze(0), target_tensor.unsqueeze(0), point_reduction=None, batch_reduction=None)
    chamfer_dist = (0.5 * (loss[0].sqrt().mean(dim=1) + loss[1].sqrt().mean(dim=1))).cpu().numpy()
    print(chamfer_dist)
    save_v = results[-1]
    save_p = src_face.cpu().numpy()
    #save_name = "reg_up_smoothed_nb1_down.ply"
    result_mesh = trimesh.Trimesh(vertices=save_v, faces=save_p)
    result_mesh.export(save_name)
    b_time = time.time()
    print("matching time: {}".format(b_time - a_time))
    if if_return == True:
        return result_mesh
