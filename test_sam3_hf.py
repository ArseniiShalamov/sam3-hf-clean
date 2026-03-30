import sys
import os

sys.path.insert(0, os.path.abspath("."))

from PIL import Image
import torch
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


device = pick_device()

# Safe fallback for Mac because some SAM3 ops are not stable on MPS yet.
if device == "mps":
    print("MPS detected, switching to CPU-safe mode")
    device = "cpu"

print(f"Using device: {device}")

image_path = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"

if not os.path.exists(image_path):
    raise FileNotFoundError(
        f"Image not found: {image_path}\n"
        "Put an image in the project folder and run:\n"
        "python test_sam3_hf.py your_image.jpg"
    )

print("Loading model...")
model = build_sam3_image_model()
model = model.to(device)
model = model.float()
model.eval()

processor = Sam3Processor(model, device=device)

print("Loading image...")
image = Image.open(image_path).convert("RGB")

if device == "cpu":
    with torch.autocast(device_type="cpu", enabled=False):
        state = processor.set_image(image)

        print("Running prompt...")
        output = processor.set_text_prompt(
            prompt="leg",
            state=state,
        )
else:
    state = processor.set_image(image)

    print("Running prompt...")
    output = processor.set_text_prompt(
        prompt="leg",
        state=state,
    )

print("Done!")
print(output.keys())



import numpy as np
from PIL import Image

mask = output["masks"]

print("Mask shape:", mask.shape)

mask = mask[0]
if len(mask.shape) == 4:
    mask = mask[0]
if len(mask.shape) == 3:
    mask = mask[0]

mask = mask.cpu().numpy()
mask = (mask > 0).astype(np.uint8) * 255

# resize mask to match image
mask = Image.fromarray(mask)
mask = mask.resize(image.size)
mask = np.array(mask)

image_np = np.array(image)

overlay = image_np.copy()
overlay[mask == 255] = [0, 200, 0]

alpha = 0.5
result = (image_np * (1 - alpha) + overlay * alpha).astype(np.uint8)

Image.fromarray(result).save("output.png")

print("Saved result to output.png")