import numpy as np
import open3d as o3d
from pathlib import Path
from PIL import Image


SCENE_DIR = Path("./data/my_leg_scene")
DA3_FILE = Path("./da3_outputs/my_leg_scene/da3_output.npz")
OBJECT_NAME = "leg"
OUTPUT_DIR = Path("./mac_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


from PIL import Image

def load_alpha_mask(path: Path, target_hw=None) -> np.ndarray:
    img = Image.open(path).convert("RGBA")
    alpha = np.array(img)[:, :, 3]

    if target_hw is not None:
        target_h, target_w = target_hw
        alpha_img = Image.fromarray(alpha)
        alpha_img = alpha_img.resize((target_w, target_h), Image.NEAREST)
        alpha = np.array(alpha_img)

    return alpha > 0


def to_homogeneous(points: np.ndarray) -> np.ndarray:
    ones = np.ones((points.shape[0], 1), dtype=points.dtype)
    return np.concatenate([points, ones], axis=1)


def transform_points(points: np.ndarray, extrinsic: np.ndarray, mode: str) -> np.ndarray:
    """
    mode='w2c'  : extrinsic is world->camera, so convert camera->world by inverse
    mode='c2w'  : extrinsic is camera->world, apply directly
    """
    if extrinsic.shape == (3, 4):
        ext4 = np.eye(4, dtype=np.float32)
        ext4[:3, :4] = extrinsic
    elif extrinsic.shape == (4, 4):
        ext4 = extrinsic.astype(np.float32)
    else:
        raise ValueError(f"Unexpected extrinsic shape: {extrinsic.shape}")

    pts_h = to_homogeneous(points)

    if mode == "w2c":
        T = np.linalg.inv(ext4)
    elif mode == "c2w":
        T = ext4
    else:
        raise ValueError("mode must be 'w2c' or 'c2w'")

    pts_world = (T @ pts_h.T).T[:, :3]
    return pts_world


def collect_points(mode: str):
    data = np.load(DA3_FILE)

    pointmaps = data["pointmaps"]      # (N, H, W, 3)
    extrinsics = data["extrinsics"]    # (N, 3, 4) or (N, 4, 4)

    all_points = []

    for i in range(pointmaps.shape[0]):
        mask_path = SCENE_DIR / OBJECT_NAME / f"{i}.png"

        pts = pointmaps[i]  # (H, W, 3)
        h, w = pts.shape[:2]

        mask = load_alpha_mask(mask_path, target_hw=(h, w))
        obj_pts = pts[mask]  # (K, 3)

        # remove invalid points
        valid = np.isfinite(obj_pts).all(axis=1)
        obj_pts = obj_pts[valid]

        if len(obj_pts) == 0:
            continue

        obj_pts_world = transform_points(obj_pts, extrinsics[i], mode=mode)
        all_points.append(obj_pts_world)

    if not all_points:
        raise RuntimeError("No points collected")

    return np.vstack(all_points)


def build_mesh_from_points(points: np.ndarray, tag: str):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # clean up
    pcd = pcd.voxel_down_sample(voxel_size=0.01)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    # normals
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30)
    )
    pcd.orient_normals_consistent_tangent_plane(20)

    # save point cloud
    pcd_path = OUTPUT_DIR / f"leg_pointcloud_{tag}.ply"
    o3d.io.write_point_cloud(str(pcd_path), pcd)

    # mesh via poisson
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=8
    )

    densities = np.asarray(densities)
    keep = densities > np.quantile(densities, 0.05)
    mesh.remove_vertices_by_mask(~keep)

    mesh.compute_vertex_normals()

    mesh_path = OUTPUT_DIR / f"leg_mesh_{tag}.ply"
    o3d.io.write_triangle_mesh(str(mesh_path), mesh)

    print(f"Saved point cloud: {pcd_path}")
    print(f"Saved mesh:       {mesh_path}")


def main():
    mode = "c2w"
    print(f"\n=== Using extrinsic mode: {mode} ===")
    points = collect_points(mode=mode)
    print("Collected points:", points.shape)
    build_mesh_from_points(points, tag=mode)


if __name__ == "__main__":
    main()