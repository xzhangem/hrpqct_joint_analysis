import torch
import torch.nn as nn
import robust_laplacian
import numpy as np
import os

def get_one_form(vert, face):
    nF = face.shape[0]
    alpha = torch.zeros((nF, 3, 2))
    v0, v1, v2 = vert.index_select(0, face[:,0]), vert.index_select(0, face[:,1]), vert.index_select(0, face[:,2])
    #print("v0 shape: {}".format(v0.shape))
    #print("v1 shape: {}".format(v1.shape))
    #print("v2 shape: {}".format(v2.shape))
    alpha[:,:,0] = v1 - v0
    alpha[:,:,1] = v2 - v0
    return alpha


def get_tangent_info(vert, face):
    nF = face.shape[0]

    alpha = torch.zeros((nF, 3, 2))
    v0, v1, v2 = vert.index_select(0, face[:,0]), vert.index_select(0, face[:,1]), vert.index_select(0, face[:,2])

    alpha[:,:,0] = v1 - v0
    alpha[:,:,1] = v2 - v0

    metric_tensor = torch.matmul(alpha.transpose(1,2),alpha)

    #A = (v1 - v2).norm(dim=1)
    #B = (v0 - v2).norm(dim=1)
    #C = (v0 - v1).norm(dim=1)
    #s = 0.5 * (A + B + C)
    #area = (s * (s - A) * (s - B) * (s - C)).clamp_(min=1e-6).sqrt()

    face_norm = 0.5 * torch.cross(v1-v0, v2-v0)
    area = face_norm.norm(dim=1)
    ### The output is the FACE-WISE tangent basis, metric tensor, area and norm vector
    return alpha, metric_tensor, area, face_norm

def mesh_gesm(vert_s, face, diff, para_len=1.0, para_shear=1.0, para_scale=1.0, para_bend=1.0, para_slfr=1.0, 
        para_lapc=1.0, torchdeviceId=torch.device('cuda:0')):

    torchdtype = torch.float32

    vert_s = vert_s.to(dtype=torch.float32, device=torchdeviceId)
    face = face.to(dtype=torch.long, device=torchdeviceId)
    diff = diff.to(dtype=torch.float32, device=torchdeviceId)

    alpha, mt, area, fnorm = get_tangent_info(vert_s, face)
    vert_np = vert_s.cpu().detach().numpy()
    face_np = face.cpu().detach().numpy()
    L, M = robust_laplacian.mesh_laplacian(vert_np, face_np)

    rows, cols = L.nonzero()
    indices = torch.from_numpy(np.vstack((rows, cols))).to(dtype=torch.long, device=torchdeviceId)
    values = torch.from_numpy(L.data).to(dtype=torch.float32, device=torchdeviceId)
    L = torch.sparse_coo_tensor(indices, values, L.shape, device=torchdeviceId)
    #L = L.coalesce()
    #L = L.values()

    rows, cols = M.nonzero()
    indices = torch.from_numpy(np.vstack((rows, cols))).to(dtype=torch.long, device=torchdeviceId)
    values = torch.from_numpy(M.data).to(dtype=torch.float32, device=torchdeviceId)
    M = torch.sparse_coo_tensor(indices, values, M.shape, device=torchdeviceId)
    M = M.coalesce()
    M = M.values()
    #print(type(M))
    #print(M.shape)
    #print(M)
    #print(M.shape)
    #M = torch.from_numpy(np.diag(M.toarray())).to(dtype=torchdtype, device=torchdeviceId)
    #L = torch.from_numpy(L.toarray()).to(dtype=torchdtype, device=torchdeviceId)

    inv_mt = torch.inverse(mt)
    of_diff = get_one_form(diff, face)

    dq_Uq_dh = torch.matmul(torch.matmul(alpha, inv_mt), of_diff.transpose(1,2))
    dq_Uq_dq = torch.matmul(torch.matmul(alpha, inv_mt), alpha.transpose(1,2))

    qUqh = torch.matmul(dq_Uq_dq, of_diff)
    qUhq = torch.matmul(dq_Uq_dh, alpha)
    scale_p = torch.einsum('nii->n', dq_Uq_dh)

    of_diff_scl = 0.5 * torch.einsum("n, nij -> nij", scale_p, alpha)
    of_diff_shr = 0.5 * (qUqh + qUhq) - of_diff_scl
    of_diff_nrm = of_diff - qUqh
    of_diff_slfr = 0.5 * (qUqh - qUhq)

    nrm_loss = torch.matmul(torch.matmul(of_diff_nrm, inv_mt), of_diff_nrm.transpose(1,2))
    scl_loss = torch.matmul(torch.matmul(of_diff_scl, inv_mt), of_diff_scl.transpose(1,2))
    shr_loss = torch.matmul(torch.matmul(of_diff_shr, inv_mt), of_diff_shr.transpose(1,2))
    slfr_loss = torch.matmul(torch.matmul(of_diff_slfr, inv_mt), of_diff_slfr.transpose(1,2))

    nrm_loss = torch.einsum('nii->n', nrm_loss).to(torchdeviceId)
    scl_loss = torch.einsum('nii->n', scl_loss).to(torchdeviceId)
    shr_loss = torch.einsum('nii->n', shr_loss).to(torchdeviceId)
    slfr_loss = torch.einsum('nii->n', slfr_loss).to(torchdeviceId)

    #print(nrm_loss.shape)
    lapc_loss = torch.abs(torch.sum(diff * (L @ diff), dim=1))

    loss = para_shear * shr_loss + para_scale * scl_loss + para_bend * nrm_loss + para_slfr * slfr_loss 
    #print(loss.shape)
    loss = torch.mean(loss.mul(area))

    if para_lapc > 0.0:
        loss = loss + para_lapc * torch.mean(lapc_loss.mul(M))

    return loss
