# SAM3 Multi-View 3D Project

This repository is a single organized project built around the following pipeline:

1. **2D segmentation with SAM3**
2. **Mask generation and visualization**
3. **Multi-view scene preparation**
4. **MV-SAM3D preprocessing experiments**
5. **Future 3D reconstruction stage**

The project is currently being developed and tested on **Mac**, with a structure designed to support both:
- local SAM3 experiments
- future multi-view 3D reconstruction workflows

---

## Project Goals

The main goal of this project is to reconstruct a **3D model from multiple images** of the same object.

Current test case:
- 3 images of a human leg
- segmentation with SAM3
- binary masks for each view
- multi-view scene preparation for MV-SAM3D

---

## Current Status

### Working
- SAM3 runs locally
- text-guided segmentation works
- binary masks are generated
- overlay visualization is generated
- project structure was reorganized
- MV-SAM3D preprocessing was adapted to run on Mac in CPU-safe mode
- multi-view scene preprocessing works on Mac

### In Progress
- depth stage / DA3 exploration on Mac
- full 3D reconstruction stage

---

## Repository Structure

```text
sam3_hf_clean/
├── README.md
├── requirements.txt
├── test_sam3_hf.py
├── sam3/
│
├── data/
│   ├── raw/         # input images
│   ├── masks/       # binary masks
│   ├── overlay/     # mask visualization
│   └── scenes/      # prepared scene folders
│
├── outputs/         # optional outputs
├── docs/            # project notes / docs
│
└── experiments/
    └── mv_sam3d_try/
        └── MV-SAM3D/