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
from common import rigid_align, anisotropic_align, plot_pts

reference_crown_uuid = '00354548-ea39-45f4-be82-28ce814e47e7'

ref = np.load(os.path.join("/data/tooth_8_cameos_and_line_angles", f"{reference_crown_uuid}.npy"), allow_pickle=True).item()
V0 = ref['vertices']
F0 = ref['faces']
cameo_perceived_mask = ref['cameo_perceived_mask']

#%%

input_dir = '/data/tooth_8_cameos_and_line_angles_common_topology'
uuids = list([x.split(".")[0] for x in os.listdir(input_dir)])
    

#%%

alignment_method = "rigid"
#alignment_method = "anisotropic"
#alignment_method = "rigid+anisotropic"

reg_meshes = [V0]
for crown_relative_index in range(len(uuids)):
    print(f"\rProcessing crown {crown_relative_index}", end="")
    fp = os.path.join(input_dir, f"{uuids[crown_relative_index]}.npy")
    if not os.path.exists(fp):
        continue

    out = np.load(fp, allow_pickle=True).item()
    V = out['V_common_topology']

    if alignment_method == "rigid":
        Vr = rigid_align(V, V0)[0]
    elif alignment_method == "anisotropic":
        Vr = anisotropic_align(V, V0)[0]
    elif alignment_method == "rigid+anisotropic":
        Vr = anisotropic_align(rigid_align(V, V0)[0], V0)[0]

    reg_meshes.append(Vr)

print()
#plot_pts([Pi.T, P0.T])
plot_pts([V, V0])
#plot_pts([Vr, V0]

#%%
import joblib

if not os.path.exists("saved_models"):
    os.makedirs("saved_models")

X = np.stack(reg_meshes[:-10])
X = X.reshape(X.shape[0], -1)

from sklearn.decomposition import PCA
pca_path = "saved_models/pca_model_cameos.pkl"
if os.path.exists(pca_path):
    pca = joblib.load(pca_path)
else:
    pca = PCA(n_components=0.99, svd_solver="full")
    pca.fit(X)
    joblib.dump(pca, pca_path)

latents = pca.transform(X)
num_points = V0.shape[0]

#%%

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

#%%

Pvis, Fvis = return_mesh(hdf5_dataset, indices, 0)
mesh = o3d.geometry.TriangleMesh()
mesh.vertices = o3d.utility.Vector3dVector(Pvis.T)
mesh.triangles = o3d.utility.Vector3iVector(Fvis.T)
mesh.compute_vertex_normals()


#%%


# --------------------------------------------------------------------- #
# Main viewer class
# --------------------------------------------------------------------- #
class SklearnPCAMeshViewer:
    """
    Viewer for a sklearn.decomposition.PCA trained on flattened mesh vertices.

    PCA must be trained on vectors of length 3N:
        [x1, y1, z1, x2, y2, z2, ..., xN, yN, zN]

    Parameters
    ----------
    pca : sklearn.decomposition.PCA
        Fitted PCA instance.
    faces : (F,3) int array
        Triangle indices of the template mesh.
    """

    def __init__(self, pca, faces):
        self.pca = pca
        self.faces = np.asarray(faces, dtype=np.int32)

        mean_flat = np.asarray(pca.mean_, dtype=np.float64)  # (3N,)
        if mean_flat.ndim != 1 or mean_flat.size % 3 != 0:
            raise ValueError(
                "pca.mean_ must be a flat vector of length 3N (x,y,z per vertex)."
            )

        self.N = mean_flat.size // 3
        self.mean_vertices = mean_flat.reshape(self.N, 3)  # (N,3)

        self.pcs = np.asarray(pca.components_, dtype=np.float64)  # (K,3N)
        self.K = self.pcs.shape[0]
        if self.pcs.shape[1] != 3 * self.N:
            raise ValueError(
                f"pca.components_ shape mismatch: got {self.pcs.shape}, "
                f"expected (K, 3N) with N={self.N}"
            )

        self.eigenvalues = np.asarray(pca.explained_variance_, dtype=np.float64)
        if self.eigenvalues.shape[0] != self.K:
            raise ValueError(
                "explained_variance_ length must match n_components (K)"
            )

    # ----------------------------------------------------------------- #
    # Core deformation
    # ----------------------------------------------------------------- #
    def deformed_vertices(self, k, alpha=3.0, use_sqrt_eig=True):
        """
        Return vertices for PCA mode k at amplitude alpha.

        Parameters
        ----------
        k : int
            PCA mode index (0 <= k < K).
        alpha : float
            Scaling in units of std if use_sqrt_eig=True.
        use_sqrt_eig : bool
            If True, deformation = alpha * sqrt(lambda_k) * pc_k.
            If False, deformation = alpha * pc_k.

        Returns
        -------
        verts : (N,3) ndarray
        """
        pc_k_flat = self.pcs[k]              # (3N,)
        pc_k = pc_k_flat.reshape(self.N, 3)  # (N,3)

        if use_sqrt_eig:
            sigma = np.sqrt(float(self.eigenvalues[k]))
            deformation = alpha * sigma * pc_k
        else:
            deformation = alpha * pc_k

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(self.mean_vertices + deformation)
        mesh.triangles = o3d.utility.Vector3iVector(self.faces)
        mesh.compute_vertex_normals()

        return mesh




template_crown_path = "/home/evangelos.sariyanidi/output/meshes_common_topology/template_crown.npy"
template_crown = np.load(template_crown_path, allow_pickle=True).item()
faces = template_crown["faces"]



#%%
# --- 2. Create an offscreen renderer ---

try:
    del renderer
    del scene
except:
    pass
width, height = 920, 1080
renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
scene = renderer.scene

# Background color (RGBA)
scene.set_background([1.0, 1.0, 1.0, 1.0])  # white

# --- 3. Add geometry to the scene with a basic material ---

material = o3d.visualization.rendering.MaterialRecord()
material.shader = "defaultLitTransparency"   # uses vertex normals & lighting
material.base_color = (1.0 , 1.0, 1.0, 0.8)   # red-ish, 50% transparent
scene.add_geometry("mesh", mesh, material)

color     = np.array([1.0, 1.0, 1.0], dtype=np.float32)
position  = np.array([0.0, 1.0, 10.5], dtype=np.float32)   # above & in front
direction = np.array([0.0, -0.5, 1.0], dtype=np.float32) # pointing down to origin

# --- 4. Set up camera from the mesh bounding box ---

bbox = mesh.get_axis_aligned_bounding_box()
center = bbox.get_center()
extent = bbox.get_extent()
radius = np.linalg.norm(extent) * 0.5
# Place camera back along +Z, looking at the object
cam_pos = center + np.array([0, 0, radius * 2])
up_vec = np.array([0, 1, 0])

# Make sure these are float32 for Open3D
center = center.astype(np.float32)
cam_pos = cam_pos.astype(np.float32)
up_vec = up_vec.astype(np.float32)

# ---- FIX IS HERE: pass center, eye, up (no bbox) ----
renderer.setup_camera(
    60.0,         # vertical FOV in degrees
    center+np.array([-1, 0, 0]),       # look-at center
    cam_pos,      # camera position (eye)
    up_vec,       # up vector
    -1.0,         # near clip (auto if -1)
    -1.0,         # far clip (auto if -1)
)

# You no longer need scene.camera.look_at() – setup_camera already sets it up.
# So you can delete this line:
# scene.camera.look_at(center, cam_pos, up_vec)

img = renderer.render_to_image()
o3d.io.write_image("render.png", img)
print("Saved render to render.png")


#%%
viewer = SklearnPCAMeshViewer(pca, faces)

import imageio.v2 as imageio
import numpy as np
import cv2

vids_dir = "vids"
if not os.path.exists(vids_dir):
    os.makedirs(vids_dir)

fps = 30
T = 30
for k in range(10):
    pos_alphas = np.concatenate([np.linspace(0, 2, T), np.linspace(2, 0, T)])
    neg_alphas = np.concatenate([np.linspace(0, -2, T), np.linspace(-2, 0, T)])
    for ai, alphas in enumerate([pos_alphas, neg_alphas]):
        with imageio.get_writer(os.path.join(vids_dir, f"pca_mode_{k}_{ai}.mp4"), fps=fps) as writer:
            for alpha in alphas:
                scene.clear_geometry()      # removes all meshes / pcds
                mesh = viewer.deformed_vertices(k, alpha=alpha)
                scene.add_geometry("mesh", mesh, material)
                img1 = renderer.render_to_image()

                R = mesh.get_rotation_matrix_from_xyz((0, np.deg2rad(-90), 0))
                mesh.rotate(R, center=mesh.get_center())
                scene.clear_geometry()      # removes all meshes / pcds
                scene.add_geometry("mesh", mesh, material)
                img2 = renderer.render_to_image()

                img = np.concatenate([np.asarray(img1), np.asarray(img2)], axis=1)
                writer.append_data(np.asarray(img))

                if alpha == 2 or alpha == -2:
                    cv2.imwrite(os.path.join(vids_dir, f"pca_mode_{k}_{ai}_{alpha}.png"), img)
