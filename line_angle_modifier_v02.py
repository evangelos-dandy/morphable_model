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
from common import rigid_align, anisotropic_align, plot_pts, plot_mesh, plot_meshes

weight_beta = 1.0
weight_gamma = 3.0

reference_crown_uuid = '00354548-ea39-45f4-be82-28ce814e47e7'

def parse_data(uuid):
    data_path = os.path.join("/data/tooth_8_cameos_and_line_angles", f"{uuid}.npy")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file {data_path} not found")
    out = np.load(data_path, allow_pickle=True).item()
    return out['vertices'], out['faces'], out['cameo_perceived_mask']
#%%
ref = np.load(os.path.join("/data/tooth_8_cameos_and_line_angles", f"{reference_crown_uuid}.npy"), allow_pickle=True).item()
V0 = ref['vertices']
F0 = ref['faces']
cameo_perceived_mask = ref['cameo_perceived_mask']

#%%

input_dir = '/data/tooth_8_cameos_and_line_angles_common_topology_with_la'
uuids = list([x.split(".")[0] for x in os.listdir(input_dir)])


#%%

Vs = []

for crown_relative_index in range(50):#range(len(uuids)):
    V = np.load(os.path.join(input_dir, f"{uuids[crown_relative_index]}.npy"), allow_pickle=True).item()['V_common_topology']
    V = rigid_align(V, V0)[0]
    Vs.append(V)

from scipy.spatial.distance import pdist, squareform
dist_matrix = squareform(pdist(np.array(Vs).reshape(len(Vs), -1)))
dist_matrix[np.diag_indices_from(dist_matrix)] = np.max(dist_matrix)
plt.imshow(dist_matrix)
idx = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape) # 9 and 46 are good
#%%

V1 = np.load(os.path.join(input_dir, f"{uuids[idx[0]]}.npy"), allow_pickle=True).item()['V_common_topology']
L1 = np.load(os.path.join(input_dir, f"{uuids[idx[0]]}.npy"), allow_pickle=True).item()['L_common_topology']
V2 = np.load(os.path.join(input_dir, f"{uuids[idx[1]]}.npy"), allow_pickle=True).item()['V_common_topology']
L2 = np.load(os.path.join(input_dir, f"{uuids[idx[1]]}.npy"), allow_pickle=True).item()['L_common_topology']


#%%

import numpy as np

def boundary_edges_from_mask(faces, mask=None, mask_indices=None):
    """
    faces: (M, 3) int array of all triangle indices
    mask:  (M,) bool array, True for faces inside the region  (optional)
    mask_indices: (K,) int array of indices into faces for region (optional)

    Returns:
        boundary_edges: (B, 2) int array of vertex index pairs
    """
    if mask is None and mask_indices is None:
        raise ValueError("Provide either mask or mask_indices")

    if mask is not None:
        Fm = faces[mask]
    else:
        Fm = faces[mask_indices]

    # All edges from masked faces: (3*K, 2)
    edges = np.vstack([
        Fm[:, [0, 1]],
        Fm[:, [1, 2]],
        Fm[:, [2, 0]],
    ])

    # Sort each edge so (min, max), so we treat (i,j) and (j,i) as the same
    edges = np.sort(edges, axis=1)

    # Find unique edges and how many times they appear
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)

    # Boundary edges appear only once among the masked faces
    boundary_edges = unique_edges[counts == 1]

    return boundary_edges

#%%

def separate_parts(V, F, mask):
    boundary_edges = boundary_edges_from_mask(F, mask)
    boundary_vertices = np.unique(boundary_edges)
    boundary_coords = V[boundary_vertices]
    boundary_coords = boundary_coords[boundary_coords[:,0]>0]

    diff = V[:, None, :] - boundary_coords[None, :, :]  # (N, B, 3)
    D = np.linalg.norm(diff, axis=2)                           # (N, B)
    weights = 1./(np.min(D,axis=1).flatten()+np.mean(D,axis=1).flatten())
    weights = (weights-weights.min())/(weights.max()-weights.min())
    V_part1 = V*weights[:, None]
    V_part2 = V*(1-weights)[:, None]
    return V_part1, V_part2, boundary_coords

def return_weights(V, F, mask):
    boundary_edges = boundary_edges_from_mask(F, mask)
    boundary_vertices = np.unique(boundary_edges)
    boundary_coords = V[boundary_vertices]
    boundary_coords = boundary_coords[boundary_coords[:,0]>0]

    diff = V[:, None, :] - boundary_coords[None, :, :]  # (N, B, 3)
    D = np.linalg.norm(diff, axis=2)                           # (N, B)
    weights = 1./(np.min(D,axis=1).flatten()+weight_beta*np.mean(D,axis=1).flatten())
    weights = (weights-weights.min())/(weights.max()-weights.min())
    return weights

def separate_parts_with_weights(V, weights):
    V_part1 = V*weights[:, None]
    V_part2 = V*(1-weights)[:, None]
    return V_part1, V_part2

weights1 = return_weights(V1, F0, L1)
weights1 = (weights1-weights1.min())/(weights1.max()-weights1.min())
weights1 = np.power(weights1, weight_gamma)
#weights2 = return_weights(V2, F0, L1)
V1_part1, V1_part2 = separate_parts_with_weights(V1, weights1)
V2_part1, V2_part2 = separate_parts_with_weights(V2, weights1)

plt.scatter(V1[:,0], V1[:,1], c=weights1, cmap='viridis')
plt.colorbar()
plt.show()

#%%

plot_mesh(V1_part1, F0)

plot_pts([V1,V2])
V2 = rigid_align(V2, V1)[0]
plot_pts([V1,V2])


def euler_matrix(rx, ry, rz, order="xyz"):
    """
    Build a 3x3 rotation matrix from Euler angles (radians).
    order: string with some permutation of 'x','y','z', e.g. 'xyz', 'zyx', ...
           Rotations are applied in the order given: first order[0], then order[1], then order[2].
    """
    cx, cy, cz = np.cos(rx), np.cos(ry), np.cos(rz)
    sx, sy, sz = np.sin(rx), np.sin(ry), np.sin(rz)

    Rx = np.array([[1,  0,   0],
                   [0, cx,  -sx],
                   [0, sx,   cx]])

    Ry = np.array([[ cy, 0, sy],
                   [  0, 1,  0],
                   [-sy, 0, cy]])

    Rz = np.array([[cz, -sz, 0],
                   [sz,  cz, 0],
                   [ 0,   0, 1]])

    mapping = {'x': Rx, 'y': Ry, 'z': Rz}

    R = np.eye(3)
    # apply in given order: first order[0], then order[1], then order[2]
    for ax in order:
        R = mapping[ax] @ R
    return R
def rotate_mesh_euler(V, rx, ry, rz, order="xyz", center=None):
    """
    V: (N, 3) vertices
    rx, ry, rz: Euler angles in radians
    order: e.g. 'xyz', 'zyx', etc.
    center: optional (3,) point to rotate around. If None, rotate about origin.
    """
    V = np.asarray(V)
    R = euler_matrix(np.deg2rad(rx), np.deg2rad(ry), np.deg2rad(rz), order=order)

    if center is not None:
        center = np.asarray(center)
        V_rot = (V - center) @ R.T + center
    else:
        V_rot = V @ R.T

    return V_rot
#%%
from common import ICP
Vcomb1 = V1_part1 + V1_part2
Vcomb2 = V1_part1 + V2_part2#V1_part1 + V2_part2
Vcomb2[:,0] *= -1
Vcomb2 = rotate_mesh_euler(Vcomb2, 0, 0, -10, center=np.mean(Vcomb2, axis=0))
Vcomb2 = rotate_mesh_euler(Vcomb2, 0, -3,0, center=np.mean(Vcomb2, axis=0))
Vcomb1 = rotate_mesh_euler(Vcomb1, 0, 5,0, center=np.mean(Vcomb1, axis=0))
Vcomb2 = Vcomb2+np.array([8.0,-0.95,0.1])
#Vcomb2 = ICP(Vcomb2, Vcomb1)[0]+np.array([10,0,0])
plot_meshes([(Vcomb1, F0), (Vcomb2, F0)], 1.0)
#%%

meshes = []
for alpha in np.linspace(0, 1, 20):
    Vcomb2_alpha = V2_part2 + alpha*V1_part1 + (1-alpha)*V2_part1
    Vcomb2_alpha[:,0] *= -1
    Vcomb2_alpha = rotate_mesh_euler(Vcomb2_alpha, 0, 0, -10, center=np.mean(Vcomb2_alpha, axis=0))
    Vcomb2_alpha = rotate_mesh_euler(Vcomb2_alpha, 0, -5, 0, center=np.mean(Vcomb2_alpha, axis=0))
    Vcomb2_alpha = Vcomb2_alpha+np.array([8.0,-0.95,0])
    meshes.append((Vcomb2_alpha, F0))

package = {
    'Vcomb1': Vcomb1,
    'F0': F0,
    'meshes': meshes,
}

outdir = 'packages'
if not os.path.exists(outdir):
    os.makedirs(outdir)

path = os.path.join(outdir, f"package_{weight_beta}_{weight_gamma}.npy")
np.save(path, package)
#%%

opacity=1.0

import plotly.graph_objects as go
import ipywidgets as widgets
from IPython.display import display
# -------------------------------------------------
# 2. Helper: Open3D mesh -> Plotly Mesh3d kwargs
# -------------------------------------------------
def mesh_to_plotly_kwargs(mesh):
    verts = mesh
    faces = F0#np.asarray(mesh.triangles)

    return dict(
        x = verts[:, 0],
        y = verts[:, 1],
        z = verts[:, 2],
        i = faces[:, 0],
        j = faces[:, 1],
        k = faces[:, 2],
    )

# -------------------------------------------------
# 3. Create a single-Scene FigureWidget with 2 traces
# -------------------------------------------------
kwargs_A0 = mesh_to_plotly_kwargs(Vcomb1)
kwargs_B0 = mesh_to_plotly_kwargs(meshes[0][0])

trace_A = go.Mesh3d(
    **kwargs_A0,
    name="Mesh A",
    color="lightblue",
    # --- LIGHTING ---
    lighting=dict(
        ambient=0.4,
        diffuse=0.9,
        specular=0.5,
        roughness=0.3,
        fresnel=0.1,
    ),
    lightposition=dict(
        x=0,
        y=0,
        z=-1.0  # light from above
    ),
    flatshading=False,
    opacity=opacity,
    showscale=False,
)


trace_B = go.Mesh3d(
    **kwargs_B0,
    name="Mesh B",
    color="lightblue",

    # --- LIGHTING ---
    lighting=dict(
        ambient=0.4,
        diffuse=0.9,
        specular=0.5,
        roughness=0.3,
        fresnel=0.1,
    ),
    lightposition=dict(
        x=0,
        y=0,
        z=-1.0  # light from above
    ),
    flatshading=False,
    opacity=opacity,
                showscale=False,
)


fig = go.FigureWidget(
    data=[trace_A, trace_B],
    layout=go.Layout(
        width=400,
        height=400,
        scene=dict(
            aspectmode="data",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
        ),
        margin=dict(l=0, r=0, t=0, b=0),
            scene_camera=dict(
            eye=dict(x=-0.002, y=-0.018, z=2.135),
            center=dict(x=0, y=0, z=0),
            up=dict(x=0, y=0, z=1),
        )
    )
)


# -------------------------------------------------
# 4. Slider + callback updating BOTH meshes
# -------------------------------------------------
slider = widgets.IntSlider(
    value=0,
    min=0,
    max=len(meshes) - 1,
    step=1,
    description="Step",
    continuous_update=True,
)

def on_slider_change(change):
    idx = change["new"]

    #kA = mesh_to_plotly_kwargs(meshes_A[idx])
    kB = mesh_to_plotly_kwargs(meshes[idx][0])

    with fig.batch_update():
        # Mesh A (trace 0)
        """
        fig.data[0].x = kA["x"]
        fig.data[0].y = kA["y"]
        fig.data[0].z = kA["z"]
        fig.data[0].i = kA["i"]
        fig.data[0].j = kA["j"]
        fig.data[0].k = kA["k"]
        """

        # Mesh B (trace 1)
        fig.data[1].x = kB["x"]
        fig.data[1].y = kB["y"]
        fig.data[1].z = kB["z"]
        fig.data[1].i = kB["i"]
        fig.data[1].j = kB["j"]
        fig.data[1].k = kB["k"]

        fig.layout.title = f"Step {idx}"

slider.observe(on_slider_change, names="value")

display(widgets.VBox([slider, fig]))
# %%
