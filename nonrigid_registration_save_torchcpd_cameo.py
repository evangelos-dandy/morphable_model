# %%

import os
import copy
import argparse
import torch
import time
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from common import plot_pts, chamfer_distance_gpu
from torchcpd import DeformableRegistration
from common import return_mesh, plot_pts, ICP, align, plot_mesh

import trimesh
from trimesh.proximity import ProximityQuery

reference_crown_uuid = '00354548-ea39-45f4-be82-28ce814e47e7'

input_dir = "/data/tooth_8_cameos_and_line_angles"
output_dir = "/data/tooth_8_cameos_and_line_angles_common_topology_with_la"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

uuids = list([x.split(".")[0] for x in os.listdir(input_dir)])

def parse_data(uuid):
    data_path = os.path.join(input_dir, f"{uuid}.npy")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file {data_path} not found")
    out = np.load(data_path, allow_pickle=True).item()
    return out['vertices'], out['faces'], out['cameo_perceived_mask']

def o3d_to_torch(pc):
    return torch.tensor(np.asarray(pc.points), dtype=torch.float32, device="cpu")


def torch_to_o3d(t):
    # t is a torch tensor (cuda or cpu)
    arr = t.detach().cpu().numpy().astype(np.float64)  # Open3D needs float64
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(arr)
    return pcd

def point_to_triangle_correspondence_with_barycentric(
    M1: trimesh.Trimesh,
    M2: trimesh.Trimesh
):
    """
    For each vertex of M2, find the closest point on the surface of M1
    (point-to-triangle), and return barycentric coordinates of that
    closest point on the corresponding triangle of M1.

    Returns
    -------
    closest_points : (n, 3) float
        Closest points on M1 surface for each vertex of M2.
    distances : (n,) float
        Euclidean distances from each M2 vertex to M1 surface.
    triangle_ids : (n,) int
        Index in M1.faces of the triangle on which the closest point lies.
    bary_coords : (n, 3) float
        Barycentric coordinates (w0, w1, w2) of each closest point
        with respect to the triangle M1.faces[triangle_ids[i]].
    """
    # 1) Closest point on the surface of M1 for each vertex of M2
    query = ProximityQuery(M1)
    points = M2.vertices                   # (n, 3)

    closest_points, distances, triangle_ids = query.on_surface(points)

    # 2) Extract the corresponding triangles (vertex positions) from M1
    #    M1.triangles has shape (num_faces, 3, 3)
    triangles = M1.triangles[triangle_ids]  # (n, 3, 3)

    # 3) Compute barycentric coordinates of closest_points in those triangles
    #    trimesh has a helper for this:
    bary_coords = trimesh.triangles.points_to_barycentric(triangles, closest_points)
    # shape: (n, 3), each row [w0, w1, w2] for triangle vertices (v0, v1, v2)

    return closest_points, distances, triangle_ids, bary_coords




V0, F0, L0 = parse_data(reference_crown_uuid)
"""

parser = argparse.ArgumentParser()
parser.add_argument("crown_relative_index", type=int, default=80)

args = parser.parse_args()
crown_relative_index = args.crown_relative_index
"""
# %%

def return_new_mask(V1, F1, M1, V2, F2):
    
    # Original mesh
    mesh1 = trimesh.Trimesh(vertices=V1, faces=F1, process=False)
    M1 = M1.astype(bool)  # shape (n_faces_1,)

    # New mesh
    mesh2 = trimesh.Trimesh(vertices=V2, faces=F2, process=False)

    # Build proximity query on mesh1
    pq = trimesh.proximity.ProximityQuery(mesh1)

    # Barycenters of faces in mesh2
    tri_verts2 = mesh2.vertices[mesh2.faces]      # (nF2, 3, 3)
    c2 = tri_verts2.mean(axis=1)                  # (nF2, 3)

    # Project barycenters onto mesh1
    closest_pts, distance, face_ids = pq.on_surface(c2)  # face_ids: (nF2,)
    M2 = M1[face_ids]
    return M2


def single_crown_registration(crown_relative_index):
    target_out = os.path.join(output_dir, f"{uuids[crown_relative_index]}.npy")
    print(f"Processing {uuids[crown_relative_index]}", end="\r")

    if os.path.exists(target_out):
        print(f"File {target_out} already exists")
        return

    t0 = time.time()

    V, F, L = parse_data(uuids[crown_relative_index])

    #std_ratio = V0.std(axis=0).reshape(1, 3) / V.std(axis=0).reshape(1, 3)
    #V *= std_ratio

    Vr, CD_before, CD_after, rigid_transformation = ICP(V, V0, scaling=False, voxel_size=0.3)

    source = o3d.geometry.TriangleMesh()
    source.vertices = o3d.utility.Vector3dVector(V)
    source.triangles = o3d.utility.Vector3iVector(F)

    target = o3d.geometry.TriangleMesh()
    target.vertices = o3d.utility.Vector3dVector(V0)
    target.triangles = o3d.utility.Vector3iVector(F0)

    # Optional: ensure normals etc.
    source.compute_vertex_normals()
    target.compute_vertex_normals()

    # --- Sample meshes to point clouds for CPD ---
    # adjust number_of_points as needed
    o3d.utility.random.seed(42)

    indices_source = np.random.choice(len(source.vertices), size=4000, replace=False)
    indices_target = np.random.choice(len(target.vertices), size=4000, replace=False)

    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(np.asarray(source.vertices)[indices_source])
    source_pcd.normals = o3d.utility.Vector3dVector(np.asarray(source.vertex_normals)[indices_source])

    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(np.asarray(target.vertices)[indices_target])
    target_pcd.normals = o3d.utility.Vector3dVector(np.asarray(target.vertex_normals)[indices_target])



    # (optional) remove NaNs / infs
    # source_pcd.remove_non_finite_points()
    # target_pcd.remove_non_finite_points()

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


    # res.transformation is a probreg.NonRigidTransformation
    # It can directly transform Open3D point data:
    aligned_source_pcd = copy.deepcopy(warped)

    Glmks = TY[::2].cpu().numpy()
    R = Vr

    t1 = time.time()
    Rwarped = align(R, Glmks, indices_source[::2])


    M1 = trimesh.Trimesh(vertices=Rwarped, faces=F, process=False)
    M2 = trimesh.Trimesh(vertices=V0, faces=F0, process=False)
    # Example usage:
    # M1 = trimesh.load("mesh1.ply")   # 10000 vertices
    # M2 = trimesh.load("mesh2.ply")   # 5000 vertices
    closest_pts, dists, tri_ids, bary_coords = point_to_triangle_correspondence_with_barycentric(M1, M2)

    data = V        # or any (V, k) data
    face_data = data[M1.faces[tri_ids]]    # (N,3,k)
    interp_data = (bary_coords[...,None] * face_data).sum(axis=1)

    Lnew = L[tri_ids]

    out = {
        'V_common_topology': interp_data,
        'L_common_topology': return_new_mask(V, F, L, interp_data, F0)

    }

    np.save(target_out, out)

    t1 = time.time()
    print(f"Time taken: {t1 - t0} seconds")



# %%
for crown_relative_index in range(len(uuids)):
    try:
        single_crown_registration(crown_relative_index)
    except Exception as e:
        print(f"Error processing {uuids[crown_relative_index]}: {e}")
        continue
#%%
