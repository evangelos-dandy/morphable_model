# %%
import h5py
import matplotlib.pyplot as plt

hdf5_filepath = "/home/evangelos.sariyanidi/data/smaller_context_canonical_anterior_audits_1946.hdf5"
hdf5_dataset = h5py.File(hdf5_filepath, "r")

indices = list(hdf5_dataset["data/tooth_8"])


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
    return
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
    plt.show()


import numpy as np
import open3d as o3d

# Create the Mesh3d trace
P0, F0 = return_mesh(0)
P1, F1 = return_mesh(1)


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


def ICP(source_pts, target_pts, voxel_size=0.2, scaling=True):
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
    return np.asarray(source_transformed.points).T, CD


P0, F0 = return_mesh(0)
P1, F1 = return_mesh(8)


P1 *= P0.std(axis=1).reshape(3, 1) / P1.std(axis=1).reshape(3, 1)
P1r, CD = ICP(P1.T, P0.T, scaling=True, voxel_size=0.3)


#
plot_pts([P0, P1])
plot_pts([P0, P1r])

# %%
P1r = ICP(P1.T, P0.T)


# %%
def compute_performance(voxel_size, scaling=True):
    CDs = []
    for i in range(10):
        Pi, Fi = return_mesh(i)
        Pir, CD = ICP(Pi.T, P0.T, voxel_size, scaling)
        CDs.append(CD)
    return np.mean(CDs)


# %%
def compute_performance_multiple_voxel_sizes(voxel_sizes, scalings):
    CDs = []
    for i in range(1, 20):
        curCDs = []
        for voxel_size in voxel_sizes:
            Pi, Fi = return_mesh(i)
            Pi *= P0.std(axis=1).reshape(3, 1) / Pi.std(axis=1).reshape(3, 1)

            for scaling in scalings:
                Pir, CD = ICP(Pi.T, P0.T, voxel_size, scaling)
                curCDs.append(CD)
        CDs.append(min(curCDs))
    return np.mean(CDs)


# %%


voxel_sizes = [0.2]
scalings = [False]
perf_wrt_voxel_size = {}
for voxel_size in np.linspace(0.1, 1.5, 7):
    perf_wrt_voxel_size[voxel_size] = compute_performance_multiple_voxel_sizes(
        [voxel_size], [False]
    )

print(perf_wrt_voxel_size)

# %%
CDs = []
CDrs = []
CDrrs = []
for i in range(1, 200):
    Pi, Fi = return_mesh(i)
    print(i)
    CDs.append(
        chamfer_distance(
            o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(Pi.T)),
            o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(P0.T)),
        )
    )
    Pir, CDr = ICP(Pi.T, P0.T, voxel_size=0.35, scaling=False)
    CDrs.append(CDr)
    Pir *= P0.std(axis=1).reshape(3, 1) / Pir.std(axis=1).reshape(3, 1)
    Pirr, CDrr = ICP(Pir.T, P0.T, voxel_size=0.35, scaling=False)
    CDrrs.append(CDrr)
    plot_pts([P0, Pi, Pir, Pirr])
    print(np.mean(CDs), np.mean(CDrs), np.mean(CDrrs))


# %%
plot_pts([P1, P0])
plot_pts([P1r, P0])
# plot_pts([P1r, P1])
# plot_pts(P1r, False)
# plot_pts([P1, P1r])
# plot_pts(P1r, False)

# %%
