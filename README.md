# ⚡ Quick Start

git clone <your-repo>
cd sam3_hf_clean

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

# Add your images to:
# data/my_leg/

# Run segmentation
python test_sam3_hf.py

# Run 3D pipeline
python mv_sam3d_try/MV-SAM3D/run_mac_pipeline.py


# 🦵 3D Leg Reconstruction Pipeline (SAM3 + Depth + Open3D)

This project reconstructs a **3D model of a human leg** from multiple 2D images.

Pipeline:
1. Image → Segmentation (SAM3)
2. Segmentation → Depth (DA3)
3. Depth → Point Cloud → 3D Mesh (Open3D)

---

# 📁 Project Structure

data/
├── my_leg/                  
├── answer_test_sam3_hf/     
└── answer_test_3d/          

---

# 📸 Step 1 — Add your images

Put your images here:

data/my_leg/

Example:

data/my_leg/0.png  
data/my_leg/1.png  
data/my_leg/2.png  

---

# ⚠️ IMPORTANT — Supported format

👉 The pipeline expects **PNG images**

If your images are not PNG, convert them.

---

# 🍏 Convert HEIC (Mac)

Example:

IMG_7064.HEIC  
IMG_7065.HEIC  
IMG_7066.HEIC  

Run:

sips -s format png data/raw/IMG_7064.HEIC --out data/my_leg/0.png  
sips -s format png data/raw/IMG_7065.HEIC --out data/my_leg/1.png  
sips -s format png data/raw/IMG_7066.HEIC --out data/my_leg/2.png  

---

# 📷 Convert JPEG / JPG

If you have:

image1.jpg  
image2.jpeg  

Run:

sips -s format png image1.jpg --out data/my_leg/0.png  
sips -s format png image2.jpeg --out data/my_leg/1.png  

---

# ✅ Check images

file data/my_leg/0.png  
file data/my_leg/1.png  
file data/my_leg/2.png  

Expected:

PNG image data  

---

# 🧠 Step 2 — Run segmentation

python test_sam3_hf.py

Result:

data/answer_test_sam3_hf/

Files:

0_mask.png  
0_overlay.png  
1_mask.png  
1_overlay.png  
2_mask.png  
2_overlay.png  

---

# 🧊 Step 3 — 3D reconstruction

python mv_sam3d_try/MV-SAM3D/run_mac_pipeline.py  

---

# 📦 Result

ls data/answer_test_3d  

Files:

leg_pointcloud_w2c.ply  
leg_mesh_w2c.ply  

---

# 👁️ View 3D

python -c "import open3d as o3d; p=o3d.io.read_point_cloud('data/answer_test_3d/leg_pointcloud_w2c.ply'); o3d.visualization.draw_geometries([p])"

---

# 📦 Requirements

- Python 3.10+
- macOS / Linux (tested on Mac)
- CPU supported (no GPU required)

Install:

pip install -r requirements.txt

---

# ❗ Troubleshooting

### Image not found
Make sure files exist in:
data/my_leg/

---

### PIL.UnidentifiedImageError
Your file is not a real PNG  
Convert using:

sips -s format png input.jpg --out output.png

---

### Open3D window stuck
Press:
q

---

### No output in answer_test_3d
Make sure pipeline finished successfully

---

# 🔄 Pipeline Overview

Images → SAM3 → Masks → Depth → Point Cloud → Mesh

---

# 🎯 Goal

Detect medical risks (e.g. swelling, edema) using 3D leg reconstruction.

---

# 🚀 Next

- Improve mesh  
- Combine views  
- Add ML model  