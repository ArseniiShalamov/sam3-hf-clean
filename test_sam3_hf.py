from pathlib import Path
import numpy as np
import open3d as o3d
from PIL import Image


BASE_DIR = Path(__file__).resolve().parent

SCENE_DIR = BASE_DIR / "data" / "my_leg_scene"
DA3_FILE = BASE_DIR / "da3_outputs" / "my_leg_scene" / "da3_output.npz"
OBJECT_NAME = "leg"

OUTPUT_DIR = BASE_DIR.parent.parent.parent / "data" / "answer_test_3d"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Choose which image/view to reconstruct
VIEW_INDEX = 0

# Choose extrinsic mode
MODE = "w2c"

# Global rotation fix for upside-down result
ROTATE_RESULT = True
ROTATE_AXIS = "x"
ROTATE_DEG = 180.0

# Basic cleanup
VOXEL_SIZE = 0.003
NB_NEIGHBORS = 30
STD_RATIO = 1.2


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
    mode='w2c': extrinsic is world-to-camera, so convert camera-to-world by inverse
    mode='c2w': extrinsic is camera-to-world, apply directly
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


def build_single_view_pointcloud(view_index: int, mode: str) -> o3d.geometry.PointCloud:
    data = np.load(DA3_FILE)

    pointmaps = data["pointmaps"]
    extrinsics = data["extrinsics"]

    if view_index < 0 or view_index >= pointmaps.shape[0]:
        raise ValueError(f"VIEW_INDEX {view_index} is out of range")

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
        raise RuntimeError("No valid points after masking")

    obj_pts_world = transform_points(obj_pts, extrinsics[view_index], mode=mode)

    pcd = points_to_pcd(obj_pts_world)
    pcd = preprocess_pcd(pcd)
    pcd = keep_largest_cluster(pcd, eps=0.015, min_points=50)

    if ROTATE_RESULT:
        pcd = rotate_cloud_around_center(pcd, ROTATE_DEG, axis=ROTATE_AXIS)

    return pcd


def main():
    print(f"Building single-view point cloud")
    print(f"VIEW_INDEX = {VIEW_INDEX}")
    print(f"MODE = {MODE}")

    pcd = build_single_view_pointcloud(VIEW_INDEX, MODE)

    pointcloud_path = OUTPUT_DIR / f"leg_single_view_{VIEW_INDEX}_{MODE}.ply"
    o3d.io.write_point_cloud(str(pointcloud_path), pcd)

    print(f"Saved point cloud: {pointcloud_path}")
    print("Done!")


if __name__ == "__main__":
    main()
















# import os
# import sys
# from pathlib import Path
# import numpy as np
#
# sys.path.insert(0, os.path.abspath("."))
#
# from PIL import Image, ImageOps
# import torch
# from sam3.model_builder import build_sam3_image_model
# from sam3.model.sam3_image_processor import Sam3Processor
#
#
# INPUT_DIR = Path("data/my_leg")
# OUTPUT_DIR = Path("data/answer_test_sam3_hf")
# OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
#
#
# def pick_device():
#     if torch.cuda.is_available():
#         return "cuda"
#     if torch.backends.mps.is_available():
#         return "mps"
#     return "cpu"
#
#
# def select_best_mask(mask_tensor):
#     """
#     Select the largest mask by area.
#     Expected shape is usually: (N, 1, H, W)
#     """
#     if len(mask_tensor.shape) == 4:
#         areas = [mask_tensor[i].sum().item() for i in range(mask_tensor.shape[0])]
#         best_idx = int(np.argmax(areas))
#         best_mask = mask_tensor[best_idx]
#     else:
#         best_mask = mask_tensor
#
#     if len(best_mask.shape) == 3:
#         best_mask = best_mask[0]
#
#     return best_mask
#
#
# def save_overlay_and_mask(image, mask_tensor, overlay_path, mask_path):
#     print("Mask shape:", mask_tensor.shape)
#
#     mask = select_best_mask(mask_tensor)
#
#     if torch.is_tensor(mask):
#         mask = mask.cpu().numpy()
#
#     mask = (mask > 0).astype(np.uint8) * 255
#
#     # Save binary mask
#     mask_img = Image.fromarray(mask)
#     mask_img = mask_img.resize(image.size, Image.NEAREST)
#     mask_np = np.array(mask_img)
#     Image.fromarray(mask_np).save(mask_path)
#     print(f"Saved binary mask to {mask_path}")
#
#     # Save overlay
#     image_np = np.array(image)
#     overlay = image_np.copy()
#     overlay[mask_np == 255] = [0, 200, 0]
#
#     alpha = 0.5
#     result = (image_np * (1 - alpha) + overlay * alpha).astype(np.uint8)
#
#     Image.fromarray(result).save(overlay_path)
#     print(f"Saved overlay to {overlay_path}")
#
#
# def main():
#     image_paths = sorted(
#         list(INPUT_DIR.glob("*.png")) +
#         list(INPUT_DIR.glob("*.jpg")) +
#         list(INPUT_DIR.glob("*.jpeg"))
#     )
#
#     if not image_paths:
#         raise FileNotFoundError(
#             f"No input images found in {INPUT_DIR}\n"
#             "Put images there, for example:\n"
#             "data/my_leg/0.png\n"
#             "data/my_leg/1.png\n"
#             "data/my_leg/2.png"
#         )
#
#     device = pick_device()
#
#     if device == "mps":
#         print("MPS detected, switching to CPU-safe mode")
#         device = "cpu"
#
#     print(f"Using device: {device}")
#     print("Loading model...")
#
#     model = build_sam3_image_model()
#     model = model.to(device)
#     model = model.float()
#     model.eval()
#
#     processor = Sam3Processor(model, device=device)
#
#     for image_path in image_paths:
#         print(f"\nProcessing: {image_path}")
#
#         # Apply EXIF orientation so Mac/iPhone photos are loaded correctly
#         image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
#
#         prompts = ["leg", "human leg", "person leg"]
#         output = None
#
#         for p in prompts:
#             print(f"Trying prompt: {p}")
#             state = processor.set_image(image)
#
#             if device == "cpu":
#                 with torch.autocast(device_type="cpu", enabled=False):
#                     output = processor.set_text_prompt(
#                         prompt=p,
#                         state=state,
#                     )
#             else:
#                 output = processor.set_text_prompt(
#                     prompt=p,
#                     state=state,
#                 )
#
#             if len(output["masks"]) > 0:
#                 print(f"Found mask with prompt: {p}")
#                 break
#
#         if output is None:
#             print(f"No output generated for {image_path}")
#             continue
#
#         print("Output keys:", output.keys())
#
#         if len(output["masks"]) == 0:
#             print(f"No masks found for {image_path}")
#             continue
#
#         base_name = image_path.stem
#         overlay_path = OUTPUT_DIR / f"{base_name}_overlay.png"
#         mask_path = OUTPUT_DIR / f"{base_name}_mask.png"
#
#         save_overlay_and_mask(image, output["masks"], overlay_path, mask_path)
#
#     print("\nDone processing all images.")
#
#
# if __name__ == "__main__":
#     main()