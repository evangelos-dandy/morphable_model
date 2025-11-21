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

alignment_method = "rigid"
#alignment_method = "anisotropic"
#alignment_method = "rigid+anisotropic"

reg_meshes = [P0]
for crown_index in range(1, 710): # 490
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


import joblib

if not os.path.exists("saved_models"):
    os.makedirs("saved_models")

X = np.stack(reg_meshes[:-10])
X = X.reshape(X.shape[0], -1)

from sklearn.decomposition import PCA
pca_path = "saved_models/pca_model.pkl"
if os.path.exists(pca_path):
    pca = joblib.load(pca_path)
else:
    pca = PCA(n_components=0.99, svd_solver="full")
    pca.fit(X)
    joblib.dump(pca, pca_path)

latents = pca.transform(X)
num_points = P0.shape[0]
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


#%%
# Example reshaping from sklearn PCA
# pca.components_.shape == (K, N*3)
mean_vertices = viewer.mean_vertices.astype(float)
K = pca.components_.shape[0]
N = mean_vertices.shape[0]

pc_modes = pca.components_.reshape(K, N, 3)  # (K, N, 3)


import numpy as np
import plotly.graph_objects as go
import ipywidgets as widgets
from IPython.display import display

# ---- Your PCA model data here ----
# mean_vertices: (N, 3)
# pc_modes:      (K, N, 3)
# faces:         (F, 3)

pc_modes = pc_modes.astype(float)
faces = faces.astype(int)

K = pc_modes.shape[0]
N = mean_vertices.shape[0]

# Limit number of PCs with sliders if you have many
MAX_PCS_WITH_SLIDERS = 8
k_show = min(K, MAX_PCS_WITH_SLIDERS)

# ---- Create initial mesh figure ----
def make_mesh(vertices):
    x, y, z = vertices.T
    i, j, k = faces.T
    mesh = go.Mesh3d(
        x=x, y=y, z=z,
        i=i, j=j, k=k,
        opacity=1.0,
        flatshading=True,
        lighting=dict(ambient=0.5, diffuse=0.7, specular=0.2),
        showscale=False,
    )

    fig = go.Figure(data=[mesh])
    fig.update_layout(
        scene=dict(
            aspectmode='data',
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
    )
    return fig

fig = make_mesh(mean_vertices)
fig_widget = go.FigureWidget(fig)
display(fig_widget)

# ---- Create sliders for each PC ----
sliders = [
    widgets.FloatSlider(
        value=0.0,
        min=-3.0,
        max=3.0,
        step=0.1,
        description=f'PC{i+1}',
        continuous_update=True,
        readout_format='.1f',
        layout=widgets.Layout(width='400px')
    )
    for i in range(k_show)
]

# Optional: button to reset sliders
reset_button = widgets.Button(description="Reset", button_style='warning')

# ---- Update function ----
def update_mesh(*alphas):
    # alphas is a tuple of slider values (a1, a2, ..., ak_show)
    vertices = mean_vertices.copy()
    for a, mode in zip(alphas, pc_modes[:k_show]):
        vertices += a * mode

    x, y, z = vertices.T

    with fig_widget.batch_update():
        fig_widget.data[0].x = x
        fig_widget.data[0].y = y
        fig_widget.data[0].z = z

def on_reset_clicked(b):
    for s in sliders:
        s.value = 0.0  # triggers update_mesh via observer

# ---- Wire up callbacks ----
for s in sliders:
    s.observe(lambda change: update_mesh(*(sl.value for sl in sliders)), names='value')

reset_button.on_click(on_reset_clicked)

# Initial call
update_mesh(*(s.value for s in sliders))

# Display controls
ui = widgets.VBox(sliders + [reset_button])
display(ui)


