#%%
import numpy as np
import open3d as o3d
import h5py
import os
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from common import return_mesh, chamfer_distance_gpu, plot_mesh, plot_pts, Open3DRenderer, rigid_align, ICP

import plotly.io as pio

cameo_dir = "/data/tooth_8_cameos_and_line_angles"
cameo_common_topology_dir = "/data/tooth_8_cameos_and_line_angles_common_topology"

reference_crown_uuid = '00354548-ea39-45f4-be82-28ce814e47e7'

ref = np.load(os.path.join("/data/tooth_8_cameos_and_line_angles", f"{reference_crown_uuid}.npy"), allow_pickle=True).item()
V0 = ref['vertices']
F0 = ref['faces']


def parse_orig_cameo(uuid):
    data_path = os.path.join(cameo_dir, f"{uuid}.npy")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"D ata file {data_path} not found")
    out = np.load(data_path, allow_pickle=True).item()
    return out['vertices'], out['faces'], out['cameo_perceived_mask']


def parse_cameo_common_topology(uuid):
    data_path = os.path.join(cameo_common_topology_dir, f"{uuid}.npy")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file {data_path} not found")
    out = np.load(data_path, allow_pickle=True).item()
    return out['V_common_topology']


V0, F0, L0 = parse_orig_cameo(reference_crown_uuid)
uuids = list([x.split(".")[0] for x in os.listdir(cameo_common_topology_dir)])
uuids.sort()

try:
    del renderer
except:
    pass

renderer = Open3DRenderer()

imdir = "images"
if not os.path.exists(imdir):
    os.makedirs(imdir)
    
from sklearn.decomposition import PCA

pca_path = "saved_models/pca_model_cameos_1501scans_var0.999.pkl"
if os.path.exists(pca_path):
    pca = joblib.load(pca_path)    
    Ntra = len(pca.tra_uuids)

#%%
CDs_new_topology = []
CDs_reconstructed = []
for relative_crown_index in range(Ntra, Ntra+50):#len(uuids)):
    if uuids[relative_crown_index] in pca.tra_uuids:
        print(f"Skipping {uuids[relative_crown_index]} because it is in the training set")
        continue
    Vorig, Forig, _= parse_orig_cameo(uuids[relative_crown_index])
    V = parse_cameo_common_topology(uuids[relative_crown_index])
    V = rigid_align(V, V0)[0]
    Vorig = ICP(Vorig, V, scaling=False, voxel_size=0.1)[0]
    Vrec = pca.inverse_transform(pca.transform(V.reshape(1, -1))).reshape(V.shape[0], 3)
    CDs_new_topology.append(chamfer_distance_gpu(Vorig, V, "cuda"))
    CDs_reconstructed.append(chamfer_distance_gpu(Vrec, V, "cuda"))
    print(f"\r {relative_crown_index}: {np.mean(CDs_new_topology)}, {np.mean(CDs_reconstructed)}", end="")
    continue
#%%

plt.hist(CDs_reconstructed)
print(np.median(CDs_reconstructed))

#%%


#%%np.median(CDs_reconstructed)
plt.hist(CDs_new_topology)

#%%
imdir = "images_3fold"
if not os.path.exists(imdir):
    os.makedirs(imdir)

for relative_crown_index in range(Ntra, Ntra+50):#len(uuids)):
    if uuids[relative_crown_index] in pca.tra_uuids:
        print(f"Skipping {uuids[relative_crown_index]} because it is in the training set")
        continue
    Vorig, Forig, _= parse_orig_cameo(uuids[relative_crown_index])
    V = parse_cameo_common_topology(uuids[relative_crown_index])
    V = rigid_align(V, V0)[0]
    Vorig = ICP(Vorig, V, scaling=False, voxel_size=0.1)[0]
    Vrec = pca.inverse_transform(pca.transform(V.reshape(1, -1))).reshape(V.shape[0], 3)

    mesh_orig = o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(Vorig), triangles=o3d.utility.Vector3iVector(Forig))
    mesh_common_topology = o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(V), triangles=o3d.utility.Vector3iVector(F0))
    mesh_reconstructed = o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(Vrec), triangles=o3d.utility.Vector3iVector(F0))
    renderer.render_to_image_multi([mesh_orig, mesh_common_topology, mesh_reconstructed], os.path.join(imdir, f"V{relative_crown_index:04d}_combined.png"))
    continue

# %%


from sklearn.decomposition import PCA
pca_path = "saved_models/pca_model_cameos_{X.shape[0]}scans.pkl"
if os.path.exists(pca_path):
    pca = joblib.load(pca_path)

# ------------------------------
# 5. Reconstruct a mesh from PCA
# ------------------------------
mesh_index = 10  # index of mesh to reconstruct
reconstructed_flat = pca.inverse_transform(latents[mesh_index:mesh_index+1])  # (1, num_points * 3)
reconstructed_mesh = reconstructed_flat.reshape(num_points, 3)

plot_pts([reconstructed_mesh.T, reg_meshes[mesh_index].T])


test_mesh_index = 2
test_meshes = reg_meshes[-10:]
X_test = np.stack(test_meshes)
X_test = X_test.reshape(X_test.shape[0], -1)

latents_test = pca.transform(X_test)
reconstructed_flat_test = pca.inverse_transform(latents_test[test_mesh_index])
reconstructed_mesh_test = reconstructed_flat_test.reshape(num_points, 3)

plot_pts([reconstructed_mesh_test.T, test_meshes[test_mesh_index].T])

