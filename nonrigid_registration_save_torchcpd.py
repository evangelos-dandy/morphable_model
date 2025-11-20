# %%
import argparse
import copy
import numpy as np
import open3d as o3d
import h5py
import matplotlib.pyplot as plt
import torch

hdf5_filepath = "/home/evangelos.sariyanidi/data/smaller_context_canonical_anterior_audits_1946.hdf5"
hdf5_dataset = h5py.File(hdf5_filepath, "r")

indices = list(hdf5_dataset["data/tooth_8"])

parser = argparse.ArgumentParser()
parser.add_argument("crown_index", type=int, default=80)
args = parser.parse_args()

crown_index = args.crown_index

def return_mesh(index, align_to_canonical=True):
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
    #
    # plt.show()


# Create the Mesh3d trace
P0, F0 = return_mesh(0)
P1, F1 = return_mesh(crown_index)


def chamfer_distance(pcd1, pcd2):
    pts1 = np.asarray(pcd1.points)  # convert to numpy
    pts2 = np.asarray(pcd2.points)

    # KD-tree in Open3D
    tree1 = o3d.geometry.KDTreeFlann(pcd1)
    tree2 = o3d.geometry.KDTreeFlann(pcd2)

    d1 = []
    for p in pts1:
        _, idx, _ = tree2.search_knn_vector_3d(p, 1)
        nn = pts2[idx[0]]
        d1.append(np.sum((p - nn) ** 2))

    d2 = []
    for p in pts2:
        _, idx, _ = tree1.search_knn_vector_3d(p, 1)
        nn = pts1[idx[0]]
        d2.append(np.sum((p - nn) ** 2))

    return np.mean(d1) + np.mean(d2)


# %%
source_faces = F0.T
target_pts = P1.T
target_faces = F1.T


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


P0, F0 = return_mesh(0)
P1, F1 = return_mesh(crown_index)

std_ratio = P0.std(axis=1).reshape(3, 1) / P1.std(axis=1).reshape(3, 1)
P1 *= std_ratio
P1r, CD, rigid_transformation = ICP(P1.T, P0.T, scaling=True, voxel_size=0.3)



#
#plot_pts([P0, P1])
#plot_pts([P0, P1r])

# %%


source = o3d.geometry.TriangleMesh()
source.vertices = o3d.utility.Vector3dVector(P1.T)
source.triangles = o3d.utility.Vector3iVector(F1.T)


target = o3d.geometry.TriangleMesh()
target.vertices = o3d.utility.Vector3dVector(P0.T)
target.triangles = o3d.utility.Vector3iVector(F0.T)
# %%


def o3d_to_torch(pc):
    return torch.tensor(np.asarray(pc.points), dtype=torch.float32, device="cpu")


def torch_to_o3d(t):
    # t is a torch tensor (cuda or cpu)
    arr = t.detach().cpu().numpy().astype(np.float64)  # Open3D needs float64
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(arr)
    return pcd


# Optional: ensure normals etc.
source.compute_vertex_normals()
target.compute_vertex_normals()

# --- Sample meshes to point clouds for CPD ---
# adjust number_of_points as needed
o3d.utility.random.seed(42)

indices_source = np.random.choice(len(source.vertices), size=10000, replace=False)
indices_target = np.random.choice(len(target.vertices), size=10000, replace=False)

source_pcd = o3d.geometry.PointCloud()
source_pcd.points = o3d.utility.Vector3dVector(np.asarray(source.vertices)[indices_source])
source_pcd.normals = o3d.utility.Vector3dVector(np.asarray(source.vertex_normals)[indices_source])

target_pcd = o3d.geometry.PointCloud()
target_pcd.points = o3d.utility.Vector3dVector(np.asarray(target.vertices)[indices_target])
target_pcd.normals = o3d.utility.Vector3dVector(np.asarray(target.vertex_normals)[indices_target])

"""
source_pcd = o3d.geometry.PointCloud()
source_pcd.points = source.vertices
source_pcd.normals = source.vertex_normals
target_pcd = o3d.geometry.PointCloud()
target_pcd.points = target.vertices
target_pcd.normals = target.vertex_normals

"""
print(np.asarray(source_pcd.points))
# %%
"""
source_pcd = o3d.geometry.PointCloud()
source_pcd.points = source.vertices
source_pcd.normals = source.vertex_normals
target_pcd = o3d.geometry.PointCloud()
target_pcd.points = target.vertices
target_pcd.normals = target.vertex_normals
"""
from torchcpd import DeformableRegistration

# (optional) remove NaNs / infs
# source_pcd.remove_non_finite_points()
# target_pcd.remove_non_finite_points()
import time

t0 = time.time()
# --- Run non-rigid CPD (probreg handles Open3D <-> numpy) ---
# beta, lmd are the usual CPD non-rigid parameters

X = o3d_to_torch(target_pcd)  # fixed
Y = o3d_to_torch(source_pcd)  # moving

reg = DeformableRegistration(
    X=X.numpy(),
    Y=Y.numpy(),
    alpha=2.0,  # smoothness
    beta=2.0,  # kernel width
    w=0.0,  # outlier weight
    max_iterations=75,
    tolerance=1e-5,
)  # .cuda()

TY, (G, W) = reg.register()

warped = torch_to_o3d(TY)

t1 = time.time()
print(f"Time taken: {t1 - t0} seconds")

# res.transformation is a probreg.NonRigidTransformation
# It can directly transform Open3D point data:
aligned_source_pcd = copy.deepcopy(warped)
# %%
# aligned_source_pcd.points = res.transformation.transform(source_pcd.points)
plot_pts([np.asarray(aligned_source_pcd.points).T, np.asarray(target_pcd.points).T])
plt.savefig(f"/home/evangelos.sariyanidi/output/cpd/nonrigid_registration_save_{crown_index}.png")


out = {
    'std_ratio': std_ratio,
    'rigid_transformation': rigid_transformation,
    'source_rigid_aligned': P1r.T,
    'indices_source': indices_source,
    'indices_target': indices_target,
    'source_warped': TY
}

np.save(f"/home/evangelos.sariyanidi/output/cpd/nonrigid_registration_save_{crown_index}.npy", out)



# --- (Optional) visualize ---
# o3d.visualization.draw_geometries(
#    [
#        aligned_source_pcd.paint_uniform_color([1, 0, 0]),
#        target_pcd.paint_uniform_color([0, 1, 0]),
#    ]
# )

# %%

# %%
