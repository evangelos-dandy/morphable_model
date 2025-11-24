import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import h5py

import cvxpy as cp
from scipy.spatial import cKDTree

from scipy.spatial.distance import pdist, squareform
import copy

import open3d as o3d



def return_mesh(hdf5_dataset, indices, index, align_to_canonical=True):
    crown_parent = hdf5_dataset["data/tooth_8/" + indices[index]]
    crown = hdf5_dataset["data/tooth_8/" + indices[index] + "/crown_model"]
    P = crown["vertices"][()].T
    F = crown["faces"][()].T
    if align_to_canonical:
        R = crown_parent["canonical_transformation"]["orientation"][()]
        T = crown_parent["canonical_transformation"]["translation"][()]
        P = P
        P_canonical = R @ (P)  # - T.reshape(3, 1)
        P_canonical -= P_canonical.mean(axis=1).reshape(3, 1)
        return P_canonical, F

    return P, F


def plot_pts(Ps, start_new_plot=True):
    if start_new_plot:
        plt.subplots(1, 2, figsize=(10, 5))
    plt.subplot(1, 2, 1)
    for P in Ps:
        plt.plot(P[0, :], P[1, :], ".", alpha=0.1)
        plt.xlabel("x")
        plt.ylabel("y")
    plt.subplot(1, 2, 2)
    for P in Ps:
        plt.plot(P[2, :], P[1, :], ".", alpha=0.1)
        plt.xlabel("z")
        plt.ylabel("y")


def nearest_neighbor_indices(A: np.ndarray, B: np.ndarray):
    """
    A: (N, 3) source points
    B: (M, 3) target points
    Returns:
        idx:  (N,) int array, idx[i] is index in B of nearest neighbor of A[i]
        dists: (N,) float array, Euclidean distance to that neighbor
    """
    tree = cKDTree(B)
    dists, idx = tree.query(A, k=1)
    return idx, dists

import numpy as np





import numpy as np
import torch


def chamfer_distance_gpu(pts1: np.ndarray,
                     pts2: np.ndarray,
                     device: str = "cuda") -> float:
    """
    Compute symmetric Chamfer distance (Euclidean) between two point clouds.
    Input is NumPy, computation is on GPU via PyTorch.

    pts1: [N, D] NumPy array
    pts2: [M, D] NumPy array
    returns: float (Euclidean Chamfer distance)
    """

    # choose device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    # convert numpy → torch
    x = torch.tensor(pts1, dtype=torch.float64, device=device)
    y = torch.tensor(pts2, dtype=torch.float64, device=device)

    # x: [N, D], y: [M, D]
    # pairwise squared distances: [N, M]
    x2 = (x ** 2).sum(1, keepdim=True)          # [N, 1]
    y2 = (y ** 2).sum(1, keepdim=True).T        # [1, M]
    d2 = x2 - 2 * (x @ y.T) + y2                # [N, M]

    # ensure numerical safety
    d2 = torch.clamp(d2, min=0.0)

    # nearest neighbor squared distances
    min_x_to_y = d2.min(dim=1).values           # [N]
    min_y_to_x = d2.min(dim=0).values           # [M]

    # convert to Euclidean
    eps = 1e-9
    min_x_to_y = torch.sqrt(min_x_to_y + eps)
    min_y_to_x = torch.sqrt(min_y_to_x + eps)

    # symmetric Chamfer
    cd = min_x_to_y.mean() + min_y_to_x.mean()

    return float(cd.item())

import numpy as np
from sklearn.neighbors import KDTree
import open3d as o3d

def chamfer_distance(pcd1: o3d.geometry.PointCloud,
                               pcd2: o3d.geometry.PointCloud):

    P = np.asarray(pcd1.points)  # (N,3)
    Q = np.asarray(pcd2.points)  # (M,3)

    tree_P = KDTree(P)
    tree_Q = KDTree(Q)

    # Euclidean nearest neighbors
    d_p_to_q, _ = tree_Q.query(P, k=1)  # (N,1)
    d_q_to_p, _ = tree_P.query(Q, k=1)  # (M,1)

    chamfer = d_p_to_q.mean() + d_q_to_p.mean()

    return chamfer



from sklearn.neighbors import KDTree

def chamfer_distance_sklearn(pcd1: o3d.geometry.PointCloud,
                             pcd2: o3d.geometry.PointCloud):

    P = np.asarray(pcd1.points)   # (N,3)
    Q = np.asarray(pcd2.points)   # (M,3)

    tree_P = KDTree(P)
    tree_Q = KDTree(Q)

    # Nearest neighbor distances
    d1, _ = tree_Q.query(P, k=1)  # distances from P → Q
    d2, _ = tree_P.query(Q, k=1)  # distances from Q → P

    # squared chamfer
    chamfer = np.mean(d1**2) + np.mean(d2**2)

    return chamfer




def ICP(source_pts, target_pts, voxel_size=0.3, scaling=True):
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(source_pts)

    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(target_pts)

    # Optional: downsample for speed & robustness
    source_ds = source.voxel_down_sample(voxel_size)
    target_ds = target.voxel_down_sample(voxel_size)

    print(np.asarray(source_ds.points).shape, np.asarray(target_ds.points).shape)

    # Initial transform (identity if you have no guess)
    init_transform = np.eye(4)

    # Max correspondence distance (very important!)
    # Should be a bit larger than average nearest-neighbor distance.
    max_corr_dist = 2.5

    result = o3d.pipelines.registration.registration_icp(
        source_ds,
        target_ds,
        max_corr_dist,
        init_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(
            with_scaling=scaling
        ),
    )

    print("Fitness:", result.fitness)
    #    print("Inlier RMSE:", result.inlier_rmse)
    # print("Transformation:\n", result.transformation)

    # plot_pts([np.asarray(source_ds.points).T, np.asarray(target_ds.points).T])

    # Apply to the original high-res source
    print(chamfer_distance(source, target))
    source_transformed = source.transform(result.transformation)
    CD = chamfer_distance(source_transformed, target)
    print(CD)
    return np.asarray(source_transformed.points).T, CD, result.transformation

def align(R, Glmks, lmk_indices):

    def solve_optimization_problem(D, Dl, b):
        np.random.seed(1907)
        m = Dl.shape[0]
        n = Dl.shape[1]
        bmax = np.max(np.abs(b))
        #e2max = 0.2*emax;       
        
        x = np.linalg.solve(Dl, b)
        return x
        
    lmk_indices = np.array(lmk_indices)
    
    Dx = squareform(pdist(R)).T
    Dx = Dx[:,lmk_indices]
    
    for j in range(Dx.shape[1]):
        Dx[:,j] = 1-Dx[:,j]/Dx[:,j].max()
    
    Dx = Dx
    Dy = Dx
    Dz = Dx
    
    Dxl = copy.deepcopy(Dx)[lmk_indices,:]
    Dyl = copy.deepcopy(Dy)[lmk_indices,:]
    Dzl = copy.deepcopy(Dz)[lmk_indices,:]
    
    bxl = Glmks[:,0]-R[lmk_indices,0]
    byl = Glmks[:,1]-R[lmk_indices,1]
    bzl = Glmks[:,2]-R[lmk_indices,2]
            
    dxu = solve_optimization_problem(Dx, Dxl, bxl)
    dyu = solve_optimization_problem(Dy, Dyl, byl)
    dzu = solve_optimization_problem(Dz, Dzl, bzl)
    
    N = R.shape[0]
    R2 = np.zeros((N,3))
    
    R2[:,0] = R[:,0] + Dx@dxu
    R2[:,1] = R[:,1] + Dy@dyu
    R2[:,2] = R[:,2] + Dz@dzu
    
    return R2



def rigid_align(A, B):
    """
    Rigidly aligns A to B (no scale).
    A, B: (N, 3) numpy arrays with 1-to-1 correspondence.
    
    Returns:
        R: (3,3) rotation
        t: (3,) translation
        T: (4,4) homogeneous transform
    """
    import numpy as np
    assert A.shape == B.shape
    A = A.astype(np.float64)
    B = B.astype(np.float64)

    # Centroids
    muA = A.mean(axis=0)
    muB = B.mean(axis=0)

    # Center
    X = A - muA
    Y = B - muB

    # Covariance
    H = X.T @ Y

    # SVD
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Reflection fix 
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Translation
    t = muB - R @ muA

    # Homogeneous transform
    T = np.eye(4)
    T[:3,:3] = R
    T[:3, 3] = t

    return (R @ A.T).T + t, R, t, T

    import numpy as np


def anisotropic_align(A, B, n_iters: int = 10):
    """
    Align A to B with rotation + per-axis scaling + translation.
    A, B: (N,3) arrays with 1-to-1 point correspondence.
    faces: optional (M,3) int array of triangle indices (shared topology).

    Model: B ≈ R diag(s) A + t

    Returns:
        R: (3,3) rotation matrix
        s: (3,) per-axis scale (sx, sy, sz)
        t: (3,) translation vector
        T: (4,4) homogeneous transform with R@diag(s) in the top-left
        A_aligned: (N,3) aligned vertex array
        aligned_mesh: (A_aligned, faces) if faces is not None, else None
    """
    assert A.shape == B.shape and A.shape[1] == 3
    A = A.astype(np.float64)
    B = B.astype(np.float64)

    # Centroids
    muA = A.mean(axis=0)
    muB = B.mean(axis=0)

    # Centered coordinates
    X = A - muA
    Y = B - muB

    # Initialize
    s = np.ones(3, dtype=np.float64)
    R = np.eye(3, dtype=np.float64)

    for _ in range(n_iters):
        # --- 1) given s, solve for R (Kabsch on scaled A vs B) ---
        X_scaled = X * s  # (N,3), per-axis scale
        H = X_scaled.T @ Y
        U, Svals, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        # Enforce proper rotation (det = +1)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        # --- 2) given R, solve for s (per-axis LS) ---
        # Bring Y into A's frame: Y' = R^T Y
        Y_in_A = Y @ R.T  # (N,3)
        # For each axis j: s_j = (Y'_j · X_j) / (X_j · X_j)
        denom = np.sum(X * X, axis=0)  # (3,)
        denom = np.where(denom == 0, 1e-12, denom)  # avoid division by zero
        s = np.sum(Y_in_A * X, axis=0) / denom

    # Final translation so that centroids match under R, s
    # muB = R diag(s) muA + t  =>  t = muB - R (muA * s)
    t = muB - R @ (muA * s)

    # Homogeneous transform with anisotropic scale baked into the linear part
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R @ np.diag(s)
    T[:3, 3] = t

    # Apply to full vertex set
    A_scaled = A * s
    A_aligned = (R @ A_scaled.T).T + t  # (N,3)

    return A_aligned, R, s, t, T

