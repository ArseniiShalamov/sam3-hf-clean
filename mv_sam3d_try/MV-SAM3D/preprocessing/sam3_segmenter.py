"""
SAM3 multi-object segmentation module
Mac-friendly version
"""

import sys
from pathlib import Path
from typing import Dict
import torch
import numpy as np
from PIL import Image
from loguru import logger


class SAM3MultiObjectSegmenter:
    """SAM3 multi-object segmenter"""

    def __init__(self, checkpoint_path: Path = None, confidence_threshold: float = 0.1):
        """
        Initialize SAM3 model

        Args:
            checkpoint_path: optional checkpoint path
            confidence_threshold: mask confidence threshold
        """
        self.confidence_threshold = confidence_threshold

        # -------- device selection --------
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        # Mac-safe fallback
        if self.device == "mps":
            logger.info("MPS detected, switching to CPU-safe mode")
            self.device = "cpu"

        logger.info(f"Using device: {self.device}")

        # -------- locate local SAM3 repo --------
        # MV-SAM3D/
        #   preprocessing/
        #   ...
        # ../../.. -> sam3_hf_clean/
        project_root = Path(__file__).resolve().parents[4]
        local_sam3_path = project_root / "sam3"

        if not local_sam3_path.exists():
            raise FileNotFoundError(
                f"Local SAM3 source not found at: {local_sam3_path}\n"
                f"Expected your project structure to contain a local 'sam3/' folder."
            )

        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        logger.info(f"Loading local SAM3 from: {local_sam3_path}")

        if checkpoint_path is not None:
            model = build_sam3_image_model(
                checkpoint_path=str(checkpoint_path),
                load_from_HF=False,
            )
        else:
            model = build_sam3_image_model()

        model = model.to(self.device)
        model = model.float()
        model.eval()

        self.processor = Sam3Processor(
            model,
            device=self.device,
            confidence_threshold=confidence_threshold,
        )

        logger.success("✓ SAM3 model loaded successfully")

    def segment_object_multiview(
        self,
        images_dir: Path,
        object_name: str,
        text_prompt: str,
        output_dir: Path,
    ) -> Dict:
        """
        Segment the same object across multiple views

        Args:
            images_dir: directory containing images
            object_name: object folder name
            text_prompt: text prompt for SAM3
            output_dir: output root directory

        Returns:
            Dict with segmentation status
        """
        logger.info(f"\n[Segmenting: {object_name}]")
        logger.info(f"  Prompt: '{text_prompt}'")

        def natural_sort_key(p):
            try:
                return (0, int(p.stem), p.stem)
            except ValueError:
                return (1, 0, p.stem)

        image_files = sorted(
            list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg")),
            key=natural_sort_key,
        )

        if not image_files:
            return {"success": False, "error": "No images found"}

        logger.info(f"  Processing {len(image_files)} views...")

        mask_dir = output_dir / object_name
        mask_dir.mkdir(parents=True, exist_ok=True)

        success_count = 0
        failed_views = []

        fallback_prompts = [
            text_prompt,
            f"human {text_prompt}",
            f"person {text_prompt}",
        ]

        for i, img_path in enumerate(image_files):
            try:
                image = Image.open(img_path).convert("RGB")
                output = None

                for prompt in fallback_prompts:
                    state = self.processor.set_image(image)

                    if self.device == "cpu":
                        with torch.autocast(device_type="cpu", enabled=False):
                            output = self.processor.set_text_prompt(
                                state=state,
                                prompt=prompt,
                            )
                    else:
                        output = self.processor.set_text_prompt(
                            state=state,
                            prompt=prompt,
                        )

                    if len(output["masks"]) > 0:
                        logger.info(f"  View {i}: found mask with prompt '{prompt}'")
                        break

                if output is None or len(output["masks"]) == 0:
                    logger.warning(f"  View {i}: No mask generated")
                    failed_views.append(i)
                    continue

                masks = output["masks"]
                scores = output["scores"]

                best_idx = scores.argmax().item()
                best_mask = masks[best_idx]
                best_score = scores[best_idx].item()

                if torch.is_tensor(best_mask):
                    mask_np = best_mask.squeeze(0).cpu().numpy()
                else:
                    mask_np = best_mask.squeeze(0)

                if mask_np.shape != (image.size[1], image.size[0]):
                    mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8))
                    mask_pil = mask_pil.resize(image.size, Image.NEAREST)
                    mask_np = np.array(mask_pil) / 255.0

                image_np = np.array(image)
                mask_bool = mask_np > 0.5

                # RGBA mask expected by MV-SAM3D:
                # RGB = original image on foreground
                # A = 255 on foreground, 0 on background
                rgba_mask = np.zeros((image_np.shape[0], image_np.shape[1], 4), dtype=np.uint8)
                rgba_mask[mask_bool, :3] = image_np[mask_bool]
                rgba_mask[mask_bool, 3] = 255
                rgba_mask[~mask_bool, :3] = 0
                rgba_mask[~mask_bool, 3] = 0

                mask_path = mask_dir / f"{img_path.stem}.png"
                Image.fromarray(rgba_mask, "RGBA").save(mask_path)

                area_ratio = np.sum(rgba_mask[:, :, 3] > 0) / (
                    rgba_mask.shape[0] * rgba_mask.shape[1]
                )

                logger.info(
                    f"  View {i}: ✓ (area={area_ratio*100:.1f}%, score={best_score:.3f})"
                )
                success_count += 1

            except Exception as e:
                logger.error(f"  View {i}: Failed - {e}")
                failed_views.append(i)

        logger.info(f"  Result: {success_count}/{len(image_files)} views segmented")
        if failed_views:
            logger.warning(f"  Failed views: {failed_views}")

        return {
            "success": success_count > 0,
            "object_name": object_name,
            "total_views": len(image_files),
            "success_views": success_count,
            "failed_views": failed_views,
            "mask_dir": mask_dir,
        }