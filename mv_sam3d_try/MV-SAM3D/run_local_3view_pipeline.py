from pathlib import Path
import numpy as np
import open3d as o3d
from PIL import Image


BASE_DIR = Path(__file__).resolve().parent

SCENE_NAME = "my_leg_scene"
OBJECT_NAME = "leg"

SCENE_DIR = BASE_DIR / "data" / SCENE_NAME
DA3_FILE = BASE_DIR / "da3_outputs" / SCENE_NAME / "da3_output.npz"

OUTPUT_DIR = BASE_DIR / "visualization" / SCENE_NAME / OBJECT_NAME / "local_3view_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODE = "w2c"

BASE_VIEW = 0
SECOND_VIEW = 1
THIRD_VIEW = 2

VOXEL_SIZE = 0.003
NB_NEIGHBORS = 30
STD_RATIO = 1.2

ROTATE_RESULT = True
ROTATE_AXIS = "x"
ROTATE_DEG = 180.0


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


def points_to_pcd(points: np.ndarray) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    return pcd


def preprocess_pcd(
    pcd: o3d.geometry.PointCloud,
    voxel_size: float = VOXEL_SIZE,
    nb_neighbors: int = NB_NEIGHBORS,
    std_ratio: float = STD_RATIO,
) -> o3d.geometry.PointCloud:
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    pcd, _ = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio
    )
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_size * 5,
            max_nn=30
        )
    )
    return pcd


def rotate_cloud_around_center(
    pcd: o3d.geometry.PointCloud,
    angle_deg: float,
    axis: str = "x"
) -> o3d.geometry.PointCloud:
    rotated = o3d.geometry.PointCloud(pcd)
    angle_rad = np.deg2rad(angle_deg)

    if axis == "x":
        R = o3d.geometry.get_rotation_matrix_from_xyz((angle_rad, 0.0, 0.0))
    elif axis == "y":
        R = o3d.geometry.get_rotation_matrix_from_xyz((0.0, angle_rad, 0.0))
    elif axis == "z":
        R = o3d.geometry.get_rotation_matrix_from_xyz((0.0, 0.0, angle_rad))
    else:
        raise ValueError("axis must be 'x', 'y', or 'z'")

    center = rotated.get_center()
    rotated.rotate(R, center=center)
    return rotated


def keep_largest_cluster(pcd: o3d.geometry.PointCloud, eps=0.01, min_points=30):
    labels = np.array(
        pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False)
    )

    valid = labels >= 0
    if not np.any(valid):
        return pcd

    largest_label = np.bincount(labels[valid]).argmax()
    idx = np.where(labels == largest_label)[0]
    return pcd.select_by_index(idx)


def build_view_pointcloud(view_index: int, mode: str) -> o3d.geometry.PointCloud:
    data = np.load(DA3_FILE)

    pointmaps = data["pointmaps"]
    extrinsics = data["extrinsics"]

    if view_index < 0 or view_index >= pointmaps.shape[0]:
        raise ValueError(f"view_index {view_index} is out of range")

    mask_path = SCENE_DIR / OBJECT_NAME / f"{view_index}.png"
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask not found: {mask_path}")

    pts = pointmaps[view_index]
    h, w = pts.shape[:2]

    mask = load_alpha_mask(mask_path, target_hw=(h, w))
    obj_pts = pts[mask]

    valid = np.isfinite(obj_pts).all(axis=1)
    obj_pts = obj_pts[valid]

    if len(obj_pts) == 0:
        raise RuntimeError(f"No valid points for view {view_index}")

    obj_pts_world = transform_points(obj_pts, extrinsics[view_index], mode=mode)

    pcd = points_to_pcd(obj_pts_world)
    pcd = preprocess_pcd(pcd)
    pcd = keep_largest_cluster(pcd, eps=0.015, min_points=50)

    if ROTATE_RESULT:
        pcd = rotate_cloud_around_center(pcd, ROTATE_DEG, axis=ROTATE_AXIS)

    return pcd


def centroid_init(source: o3d.geometry.PointCloud, target: o3d.geometry.PointCloud) -> np.ndarray:
    src_center = np.asarray(source.get_center())
    tgt_center = np.asarray(target.get_center())

    T = np.eye(4)
    T[:3, 3] = tgt_center - src_center
    return T


def run_icp(source: o3d.geometry.PointCloud, target: o3d.geometry.PointCloud, threshold: float = 0.02):
    init = centroid_init(source, target)

    result = o3d.pipelines.registration.registration_icp(
        source,
        target,
        threshold,
        init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )

    aligned = o3d.geometry.PointCloud(source)
    aligned.transform(result.transformation)

    print("ICP fitness:", result.fitness)
    print("ICP inlier RMSE:", result.inlier_rmse)
    print("ICP transformation:\n", result.transformation)

    return aligned


def merge_pointclouds(pcds):
    merged = o3d.geometry.PointCloud()
    for p in pcds:
        merged += p

    merged = preprocess_pcd(merged, voxel_size=0.004, nb_neighbors=20, std_ratio=2.0)
    merged = keep_largest_cluster(merged, eps=0.015, min_points=50)
    return merged


def main():
    print(f"Building local 3-view reconstruction: base={BASE_VIEW}, second={SECOND_VIEW}, third={THIRD_VIEW}")

    base_pcd = build_view_pointcloud(BASE_VIEW, MODE)
    second_pcd = build_view_pointcloud(SECOND_VIEW, MODE)
    third_pcd = build_view_pointcloud(THIRD_VIEW, MODE)

    second_pcd = rotate_cloud_around_center(second_pcd, 30, axis="y")

    aligned_second = run_icp(second_pcd, base_pcd, threshold=0.02)
    merged_pair = merge_pointclouds([base_pcd, aligned_second])

    third_pcd = rotate_cloud_around_center(third_pcd, 10, axis="y")
    aligned_third = run_icp(third_pcd, merged_pair, threshold=0.01)
    aligned_third = rotate_cloud_around_center(aligned_third, 10, axis="y")

    merged_final = merge_pointclouds([merged_pair, aligned_third])

    o3d.io.write_point_cloud(str(OUTPUT_DIR / "base_view.ply"), base_pcd)
    o3d.io.write_point_cloud(str(OUTPUT_DIR / "second_aligned.ply"), aligned_second)
    o3d.io.write_point_cloud(str(OUTPUT_DIR / "third_aligned.ply"), aligned_third)
    o3d.io.write_point_cloud(str(OUTPUT_DIR / "merged_final.ply"), merged_final)

    print(f"Saved outputs to: {OUTPUT_DIR}")
    print("Done!")


if __name__ == "__main__":
    main()