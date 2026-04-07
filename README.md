# SAM3 + MV-SAM3D 3D Reconstruction Pipeline

## Overview

This project builds a 3D reconstruction pipeline from multiple 2D images using:

- SAM3 (segmentation)
- Depth Anything 3 (depth + pointmaps)
- Custom reconstruction backend (Mac-compatible)

Pipeline:

Images → Segmentation (SAM3) → Masks → Depth (DA3) → Pointmaps → 3D Reconstruction → Mesh (.ply)

---

## ⚙️ Features

- Works on Mac (CPU-only)
- Modular pipeline:
  - segmentation
  - depth estimation
  - reconstruction
- Supports multiple reconstruction backends

---

## 🧠 Mac-compatible Reconstruction

Since the original MV-SAM3D pipeline depends on PyTorch3D (GPU/Linux),
a custom reconstruction backend was implemented using Open3D.

File:
mac_reconstruct.py

This backend:
- loads masks and DA3 pointmaps
- merges multi-view 3D points
- builds a point cloud
- reconstructs a mesh using Poisson reconstruction
- exports `.ply`

---

## ▶️ How to Run (Mac)

### 1. Segmentation

python preprocessing/build_mvsam3d_dataset.py --input data/my_leg_scene --objects leg

---

### 2. Depth (DA3)

python scripts/run_da3.py \
  --image_dir ./data/my_leg_scene/images \
  --output_dir ./da3_outputs/my_leg_scene \
  --device cpu \
  --no_vis

---

### 3. 3D Reconstruction (Mac)

python mac_reconstruct.py

---

## 📦 Output

Results are saved in:

mac_outputs/

Files:
- leg_pointcloud_c2w.ply
- leg_mesh_c2w.ply

---

## 🧪 Notes

- Two extrinsic modes were tested: w2c and c2w
- c2w produced correct geometry and is used as default

---

## 🔮 Future Work

- Add GPU backend (PyTorch3D)
- Improve mesh quality
- Multi-object reconstruction
- Integration into unified pipeline interface

---

## 👨‍💻 Author

Arsenii Shalamov