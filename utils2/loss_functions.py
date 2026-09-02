import torch
from pytorch3d.ops.knn import knn_points
from pytorch3d.loss.chamfer import chamfer_distance,_handle_pointcloud_input
from pytorch3d.loss.chamfer import _handle_pointcloud_input

from pytorch3d.ops import ball_query


from .pairwise_exp import pairwise_l1_exp_kernel

def correntropy_chamfer_distance(
        x,
        y,
        it_num,
        x_lengths=None,
        y_lengths=None,
        x_normals=None,
        y_normals=None,
        norm=1,
        sigma2=1.0
):
    """
    Correntropy Chamfer distance between two pointclouds x and y.

    Args:
        x: FloatTensor of shape (N, P1, D) or a Pointclouds object representing
            a batch of point clouds with at most P1 points in each batch element,
            batch size N and feature dimension D.
        y: FloatTensor of shape (N, P2, D) or a Pointclouds object representing
            a batch of point clouds with at most P2 points in each batch element,
            batch size N and feature dimension D.
        x_lengths: Optional LongTensor of shape (N,) giving the number of points in each
            cloud in x.
        y_lengths: Optional LongTensor of shape (N,) giving the number of points in each
            cloud in y.
        x_normals: Optional FloatTensor of shape (N, P1, D).
        y_normals: Optional FloatTensor of shape (N, P2, D).
        weights: Optional FloatTensor of shape (N,) giving weights for
            batch elements for reduction operation.
        batch_reduction: Reduction operation to apply for the loss across the
            batch, can be one of ["mean", "sum"] or None.
        point_reduction: Reduction operation to apply for the loss across the
            points, can be one of ["mean", "sum"].

    Returns:
        2-element tuple containing

        - **loss**: Tensor giving the reduced distance between the pointclouds
          in x and the pointclouds in y.
        - **loss_normals**: Tensor giving the reduced cosine distance of normals
          between pointclouds in x and pointclouds in y. Returns None if
          x_normals and y_normals are None.
    """
    norm=1

    x, x_lengths, x_normals = _handle_pointcloud_input(x, x_lengths, x_normals)
    y, y_lengths, y_normals = _handle_pointcloud_input(y, y_lengths, y_normals)

    N, P1, D = x.shape
    P2 = y.shape[1]

    x_mask = (
            torch.arange(P1, device=x.device)[None] >= x_lengths[:, None]
    )  # shape [N, P1] 

    y_mask = (
            torch.arange(P2, device=y.device)[None] >= y_lengths[:, None]
    )  # shape [N, P2]
    #x_mask = x_mask.unsqueeze(-1).expand(-1, -1, 15)
    #y_mask = y_mask.unsqueeze(-1).expand(-1, -1, 15)


    if y.shape[0] != N or y.shape[2] != D:
        raise ValueError("y does not have the correct shape.")

    x_nn = knn_points(x, y, lengths1=x_lengths, lengths2=y_lengths, norm=norm, K=1)
    y_nn = knn_points(y, x, lengths1=y_lengths, lengths2=x_lengths, norm=norm, K=1)

    '''
    dists_x, idx_x,_ = ball_query(x, y, radius=0.05, K=20, return_nn=False)
    dists_y, idx_y,_ = ball_query(y, x, radius=0.05, K=20, return_nn=False)
    #dists_x = [dists_x[0, i][idx_x[0, i] != -1] for i in range(dists_x.shape[1])]
    #dists_y = [dists_y[0, i][idx_y[0, i] != -1] for i in range(dists_y.shape[1])]
    #print(dists_x[0].shape)

    exp_dists_x = torch.exp(-1*dists_x/(sigma2))
    exp_dists_x = exp_dists_x.sum(2)
    exp_dists_x = exp_dists_x.sum(1)
    exp_dists_x = exp_dists_x / (x_lengths)
    exp_dists_x = exp_dists_x.sum()

    exp_dists_y = torch.exp(-1*dists_y/(sigma2))
    exp_dists_y = exp_dists_y.sum(2)
    exp_dists_y = exp_dists_y.sum(1)
    exp_dists_y = exp_dists_y / (y_lengths)
    exp_dists_y = exp_dists_y.sum()
    '''

    cham_x = x_nn.dists[..., 0]  # (N, P1)
    cham_y = y_nn.dists[..., 0]  # (N, P2)
    trunc_x=0.2   
    trunc_y=0.2  
    x_mask[cham_x >= trunc_x] = True
    y_mask[cham_y >= trunc_y] = True
    cham_x[x_mask] = 0.0
    cham_y[y_mask] = 0.0

    #  correntropy criterion
    exp_cham_x=torch.exp(-1*cham_x/(sigma2))
    exp_cham_y=torch.exp(-1*cham_y/(sigma2))

    x_nn_idx = x_nn.idx[..., 0]  # (N, P1), indices in y for each x point
    y_nn_idx = y_nn.idx[..., 0]  # (N, P2), indices in x for each y point
    '''

    if (x_normals is not None) and (y_normals is not None):
        if it_num < 300:
            it_k = 0.0
        else:
            it_k = 0.001 #* it_num
        n_nn_y = torch.gather(y_normals, 1, x_nn_idx.unsqueeze(-1).expand(-1, -1, D))  # (N, P1, D)
        n_nn_x = torch.gather(x_normals, 1, y_nn_idx.unsqueeze(-1).expand(-1, -1, D))
        weight_x = (1 + it_k * torch.sum(x_normals * n_nn_y, dim=-1))  # (N, P1)
        weight_y = (1 + it_k * torch.sum(y_normals * n_nn_x, dim=-1))  # (N, P2)
        weight_x[x_mask] = 0.0
        weight_y[y_mask] = 0.0
        exp_cham_x = exp_cham_x * weight_x
        exp_cham_y = exp_cham_y * weight_y

    '''
    cham_x=exp_cham_x 
    cham_y=exp_cham_y


    cham_x = cham_x.sum(1)
    cham_y = cham_y.sum(1)

    cham_x /= (x_lengths)
    cham_y /= (y_lengths)
        

    cham_x = cham_x.sum()
    cham_y = cham_y.sum()


    cham_dist = -1.0*(cham_x + cham_y)
    #src_pair_mat = pairwise_l1_exp_kernel(x, x, sigma2)
    #cham_dist = src_pair_mat.sum() / (P1 * P1) + cham_dist
    #cham_dist = -1.0 * (exp_dists_x + exp_dists_y)

    return cham_dist
