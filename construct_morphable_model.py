#%%
# import argparse
import numpy as np
import open3d as o3d
import h5py
import matplotlib.pyplot as plt
import torch
from torchcpd import DeformableRegistration
import time
import os
#%%
from common import return_mesh, plot_pts, align, rigid_align, anisotropic_align
hdf5_filepath = "/home/evangelos.sariyanidi/data/smaller_context_canonical_anterior_audits_1946.hdf5"
hdf5_dataset = h5py.File(hdf5_filepath, "r")

indices = list(hdf5_dataset["data/tooth_8"])

template_crown_index = 0


template_crown_path = "/home/evangelos.sariyanidi/output/meshes_common_topology/template_crown.npy"
template_crown = np.load(template_crown_path, allow_pickle=True).item()
P0 = template_crown["vertices"]
#%%
alignment_method = "rigid"
#alignment_method = "anisotropic"
#alignment_method = "rigid+anisotropic"

reg_meshes = [P0]
for crown_index in range(1, 490): # 490
    print(f"\rProcessing crown {crown_index}", end="")
    source_crown_path = f"/home/evangelos.sariyanidi/output/meshes_common_topology/source_crown_{crown_index:05d}.npy"

    if not os.path.exists(source_crown_path):
        continue
    
    Pi = np.load(source_crown_path, allow_pickle=True).item()['vertices']
    if alignment_method == "rigid":
        Pir = rigid_align(Pi, P0)[0]
    elif alignment_method == "anisotropic":
        Pir = anisotropic_align(Pi, P0)[0]
    elif alignment_method == "rigid+anisotropic":
        Pir = anisotropic_align(rigid_align(Pi, P0)[0], P0)[0]
    reg_meshes.append(Pir)

print()
#plot_pts([Pi.T, P0.T])
plot_pts([Pir.T, P0.T])
print(np.mean(np.sqrt((Pir-P0)**2)))

# %%

from sklearn.decomposition import PCA
X = np.stack(reg_meshes[:-10])
X = X.reshape(X.shape[0], -1)
num_points = P0.shape[0]


pca = PCA(n_components=0.99, svd_solver="full")
pca.fit(X)

# ------------------------------
# 4. Project meshes into PCA space
# ------------------------------
# latents.shape == (num_meshes, n_components)
latents = pca.transform(X)
print("Latent code shape:", latents.shape)
#%%

# ------------------------------
# 5. Reconstruct a mesh from PCA
# ------------------------------
mesh_index = 10  # index of mesh to reconstruct
reconstructed_flat = pca.inverse_transform(latents[mesh_index:mesh_index+1])  # (1, num_points * 3)
reconstructed_mesh = reconstructed_flat.reshape(num_points, 3)

plot_pts([reconstructed_mesh.T, reg_meshes[mesh_index].T])

#%%
test_mesh_index = 2
test_meshes = reg_meshes[-10:]
X_test = np.stack(test_meshes)
X_test = X_test.reshape(X_test.shape[0], -1)

latents_test = pca.transform(X_test)
reconstructed_flat_test = pca.inverse_transform(latents_test[test_mesh_index])
reconstructed_mesh_test = reconstructed_flat_test.reshape(num_points, 3)

plot_pts([reconstructed_mesh_test.T, test_meshes[test_mesh_index].T])

#%%