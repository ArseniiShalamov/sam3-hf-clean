import sys
import os

sys.path.insert(0, os.path.abspath("."))

from PIL import Image
import torch
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

device = "cpu"

print("Loading model...")
model = build_sam3_image_model()
model = model.to(device)
model = model.float()
model.eval()

processor = Sam3Processor(model, device=device)

print("Loading image...")
image_path = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"

if not os.path.exists(image_path):
    raise FileNotFoundError(
        f"Image not found: {image_path}\n"
        "Put an image in the project folder and run:\n"
        "python test_sam3_hf.py your_image.jpg"
    )

image = Image.open(image_path).convert("RGB")
with torch.autocast(device_type="cpu", enabled=False):
    state = processor.set_image(image)

    print("Running prompt...")
    output = processor.set_text_prompt(
        prompt="a dog",
        state=state,
    )

print("Done!")
print(output.keys())