import os
import subprocess

SCENE = "my_leg_scene"
OBJECT = "leg"

print("\n=== STEP 1: Segmentation (SAM3) ===")
subprocess.run([
    "python", "preprocessing/build_mvsam3d_dataset.py",
    "--input", f"data/{SCENE}",
    "--objects", OBJECT
], check=True)

print("\n=== STEP 2: Depth (DA3) ===")
subprocess.run([
    "python", "scripts/run_da3.py",
    "--image_dir", f"./data/{SCENE}/images",
    "--output_dir", f"./da3_outputs/{SCENE}",
    "--model_path", "depth-anything/DA3NESTED-GIANT-LARGE",
    "--device", "cpu",
    "--no_vis"
], check=True)

print("\n=== STEP 3: 3D Reconstruction (Mac) ===")
subprocess.run([
    "python", "mac_reconstruct.py"
], check=True)

print("\n✅ Pipeline finished successfully!")