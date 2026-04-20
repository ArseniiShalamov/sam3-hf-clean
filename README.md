# 🧠 SAM3 + MV-SAM3D (Mac Adapted)

This project is a Mac-compatible pipeline for generating 3D objects from images using:

- SAM3 (segmentation)
- Depth-Anything-3 (depth estimation)
- MV-SAM3D (multi-view 3D reconstruction)

---

## 🚀 What this project does

You can:

1. Take 1–3 images of an object  
2. Run the pipeline  
3. Get a 3D reconstruction (mesh / scene)

Works on Mac (Apple Silicon) without CUDA.

---

## 📂 Project structure

```
sam3_hf_clean/
│
├── data/                  # Input images
├── sam3/                  # SAM3 model
├── mv_sam3d_try/
│   ├── MV-SAM3D/          # 3D reconstruction pipeline
│   └── Depth-Anything-3/  # Depth model
│
├── test_sam3_hf.py        # Single-image test
└── README.md
```

---

## ⚠️ IMPORTANT — Supported format

👉 The pipeline expects PNG images

If your images are not PNG — convert them first.

---

## 🍏 Convert HEIC (Mac)

```
sips -s format png input.HEIC --out output.png
```

---

## 🛠️ Setup

### 1. Create virtual environment

```
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

---

## ▶️ Run pipeline (Mac)

```
python mv_sam3d_try/MV-SAM3D/run_mac_pipeline.py
```

---

## 📸 Input images

Place your images here:

```
data/my_leg/
```

Example:

```
data/my_leg/0.png
data/my_leg/1.png
data/my_leg/2.png
```

---

## 🧪 Single image test

```
python test_sam3_hf.py
```

---

## ⚙️ Notes

- Designed for CPU / Apple Silicon
- No CUDA required
- Some parts are adapted for Mac compatibility

---

## 📌 TODO

- Improve reconstruction quality
- Add automatic preprocessing
- Support more than 3 views

---

## 👤 Author

Arsenii Shalamov