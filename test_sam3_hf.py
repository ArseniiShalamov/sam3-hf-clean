import os
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, os.path.abspath("."))

from PIL import Image
import torch
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


INPUT_DIR = Path("data/my_leg")
OUTPUT_DIR = Path("data/answer_test_sam3_hf")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def save_overlay_and_mask(image, mask_tensor, overlay_path, mask_path):
    mask = mask_tensor

    print("Mask shape:", mask.shape)

    mask = mask[0]
    if len(mask.shape) == 4:
        mask = mask[0]
    if len(mask.shape) == 3:
        mask = mask[0]

    if torch.is_tensor(mask):
        mask = mask.cpu().numpy()

    mask = (mask > 0).astype(np.uint8) * 255

    # save binary mask
    mask_img = Image.fromarray(mask)
    mask_img = mask_img.resize(image.size, Image.NEAREST)
    mask_np = np.array(mask_img)
    Image.fromarray(mask_np).save(mask_path)
    print(f"Saved binary mask to {mask_path}")

    # save overlay
    image_np = np.array(image)
    overlay = image_np.copy()
    overlay[mask_np == 255] = [0, 200, 0]

    alpha = 0.5
    result = (image_np * (1 - alpha) + overlay * alpha).astype(np.uint8)

    Image.fromarray(result).save(overlay_path)
    print(f"Saved overlay to {overlay_path}")


def main():
    image_paths = sorted(
        list(INPUT_DIR.glob("*.png")) +
        list(INPUT_DIR.glob("*.jpg")) +
        list(INPUT_DIR.glob("*.jpeg"))
    )

    if not image_paths:
        raise FileNotFoundError(
            f"No input images found in {INPUT_DIR}\n"
            "Put images there, for example:\n"
            "data/my_leg/0.png\n"
            "data/my_leg/1.png\n"
            "data/my_leg/2.png"
        )

    device = pick_device()

    if device == "mps":
        print("MPS detected, switching to CPU-safe mode")
        device = "cpu"

    print(f"Using device: {device}")
    print("Loading model...")

    model = build_sam3_image_model()
    model = model.to(device)
    model = model.float()
    model.eval()

    processor = Sam3Processor(model, device=device)

    for image_path in image_paths:
        print(f"\nProcessing: {image_path}")
        image = Image.open(image_path).convert("RGB")

        prompts = ["leg", "human leg", "person leg"]
        output = None

        for p in prompts:
            print(f"Trying prompt: {p}")
            state = processor.set_image(image)

            if device == "cpu":
                with torch.autocast(device_type="cpu", enabled=False):
                    output = processor.set_text_prompt(
                        prompt=p,
                        state=state,
                    )
            else:
                output = processor.set_text_prompt(
                    prompt=p,
                    state=state,
                )

            if len(output["masks"]) > 0:
                print(f"Found mask with prompt: {p}")
                break

        print("Output keys:", output.keys())

        if len(output["masks"]) == 0:
            print(f"No masks found for {image_path}")
            continue

        base_name = image_path.stem
        overlay_path = OUTPUT_DIR / f"{base_name}_overlay.png"
        mask_path = OUTPUT_DIR / f"{base_name}_mask.png"

        save_overlay_and_mask(image, output["masks"], overlay_path, mask_path)

    print("\nDone processing all images.")


if __name__ == "__main__":
    main()