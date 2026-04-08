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

# 🧠 Step 2 — Run segmentation

python test_sam3_hf.py

Result:

data/answer_test_sam3_hf/

---

# 🍏 If you have HEIC (Mac)

sips -s format png data/raw/IMG_7064.HEIC --out data/my_leg/0.png  
sips -s format png data/raw/IMG_7065.HEIC --out data/my_leg/1.png  
sips -s format png data/raw/IMG_7066.HEIC --out data/my_leg/2.png  

---

# ✅ Check images

file data/my_leg/0.png  
file data/my_leg/1.png  
file data/my_leg/2.png  

---

# 🧊 Step 3 — 3D reconstruction

python experiments/mv_sam3d_try/MV-SAM3D/run_mac_pipeline.py  

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

# 🎯 Goal

Detect medical risks (e.g. swelling, edema) using 3D leg reconstruction.

---

# 🚀 Next

- Improve mesh
- Combine views
- Add ML model