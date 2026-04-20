"""
Image organization module - move raw images into images/ directory
Mac-friendly version without cv2
"""

from pathlib import Path
from typing import Dict
from PIL import Image
from loguru import logger


def organize_images(scene_dir: Path) -> Dict:
    """
    Organize scattered images into images/ directory

    Args:
        scene_dir: scene directory (contains 0.jpg, 1.jpg, etc.)

    Returns:
        Dict with status and info
    """
    logger.info(f"Organizing images in: {scene_dir}")

    image_files = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
        image_files.extend(list(scene_dir.glob(ext)))

    # exclude files already inside images/
    image_files = [f for f in image_files if "images" not in f.parts]

    if not image_files:
        images_dir = scene_dir / "images"
        if images_dir.exists():
            existing_images = list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg"))
            if existing_images:
                logger.info(f"  Images already organized ({len(existing_images)} files)")
                return {
                    "success": True,
                    "num_images": len(existing_images),
                    "already_organized": True,
                }

        logger.error("  No image files found!")
        return {"success": False, "error": "No images found"}

    def natural_sort_key(p):
        try:
            return int(p.stem)
        except Exception:
            return p.stem

    image_files = sorted(image_files, key=natural_sort_key)
    logger.info(f"  Found {len(image_files)} images")

    images_dir = scene_dir / "images"
    images_dir.mkdir(exist_ok=True)

    for i, img_path in enumerate(image_files):
        output_path = images_dir / f"{i}.png"

        try:
            img = Image.open(img_path).convert("RGB")
            img.save(output_path, format="PNG")
            logger.info(f"  {img_path.name} → {output_path.name}")
        except Exception as e:
            logger.warning(f"  Failed to process {img_path}: {e}")

    logger.success(f"✓ Organized {len(image_files)} images to {images_dir}")

    return {
        "success": True,
        "num_images": len(image_files),
        "images_dir": images_dir,
        "already_organized": False,
    }