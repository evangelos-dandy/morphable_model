# %%

import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("crown_index", type=int, default=80)

args = parser.parse_args()
crown_index = args.crown_index
target_out = f"/home/evangelos.sariyanidi/output/cpd/nonrigid_registration_save_{crown_index}.npy"

if os.path.exists(target_out):
    print(f"File {target_out} already exists")
    exit()


import copy
import numpy as np
import open3d as o3d
import h5py
import matplotlib.pyplot as plt
import torch
from torchcpd import DeformableRegistration
import time

from common import return_mesh, plot_pts, ICP

hdf5_filepath = "/home/evangelos.sariyanidi/data/smaller_context_canonical_anterior_audits_1946.hdf5"
hdf5_dataset = h5py.File(hdf5_filepath, "r")

indices = list(hdf5_dataset["data/tooth_8"])


# Create the Mesh3d trace
P0, F0 = return_mesh(hdf5_dataset, indices, 0)
P1, F1 = return_mesh(hdf5_dataset, indices, crown_index)
# %%
source_faces = F0.T
target_pts = P1.T
target_faces = F1.T


P0, F0 = return_mesh(hdf5_dataset, indices, 0)
P1, F1 = return_mesh(hdf5_dataset, indices, crown_index)

std_ratio = P0.std(axis=1).reshape(3, 1) / P1.std(axis=1).reshape(3, 1)
P1 *= std_ratio
P1r, CD, rigid_transformation = ICP(P1.T, P0.T, scaling=True, voxel_size=0.3)


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

# (optional) remove NaNs / infs
# source_pcd.remove_non_finite_points()
# target_pcd.remove_non_finite_points()

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

np.save(target_out, out)



# --- (Optional) visualize ---
# o3d.visualization.draw_geometries(
#    [
#        aligned_source_pcd.paint_uniform_color([1, 0, 0]),
#        target_pcd.paint_uniform_color([0, 1, 0]),
#    ]
# )

# %%

# %%
