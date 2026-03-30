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
image = Image.open("test.jpg").convert("RGB")

with torch.autocast(device_type="cpu", enabled=False):
    state = processor.set_image(image)

    print("Running prompt...")
    output = processor.set_text_prompt(
        prompt="a dog",
        state=state,
    )

print("Done!")
print(output.keys())