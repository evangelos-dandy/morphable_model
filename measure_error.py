#%%
import numpy as np
import open3d as o3d
import h5py
import seaborn as sns
from common import return_mesh, chamfer_distance_gpu

hdf5_filepath = "/home/evangelos.sariyanidi/data/smaller_context_canonical_anterior_audits_1946.hdf5"
hdf5_dataset = h5py.File(hdf5_filepath, "r")
indices = list(hdf5_dataset["data/tooth_8"])

template_crown_path = "/home/evangelos.sariyanidi/output/meshes_common_topology/template_crown.npy"
template_crown = np.load(template_crown_path, allow_pickle=True).item()
Ft = template_crown['faces']

def return_crown_template_topology(crown_index):
    fp = f"/home/evangelos.sariyanidi/output/meshes_common_topology/source_crown_{crown_index:05d}.npy"
    crown_path_template_topology = np.load(fp, allow_pickle=True).item()
    Pt = crown_path_template_topology['vertices']
    return Pt
#%%



for crown_index in range(1, 10):
    print(f"Processing crown {crown_index}")
    Pt = return_crown_template_topology(crown_index)
    P, F = return_mesh(hdf5_dataset, indices, crown_index)
    try:
        P = P.T
        F = F.T
    except:
        print(f'{crown_index} is not found')
        continue

    error = chamfer_distance_gpu(P, Pt, "cuda")
    print(f"Error: {error}")
    #error = chamfer_distance(Po3d, Pt3d)
    #print(f"Error: {error}")
    


#%%

import torch
chamfer_distance_gpu(P, Pt, "cuda")
#%%
Po3d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
Pt3d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(Pt))
print(chamfer_distance(Po3d, Pt3d))

#%%




