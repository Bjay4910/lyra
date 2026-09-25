import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import urllib.request
import numpy as np
import argparse
import os
import sys

parser = argparse.ArgumentParser(
    description="Preprocess an image into input.bin and save PyTorch's reference output.")
parser.add_argument("image", nargs="?",
                    help="local image path or http(s):// URL (jpg/png); "
                         "defaults to the PyTorch Hub sample dog.jpg")
args = parser.parse_args()

DOWNLOAD_PATH = "test_image.jpg"
# Some hosts (e.g. Wikimedia) reject urllib's default "Python-urllib/x.y" agent.
USER_AGENT = "Mozilla/5.0 (compatible; lyra-prepare/1.0)"


def download(url, dest):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=30) as response, open(dest, "wb") as f:
            f.write(response.read())
    except Exception as e:
        sys.exit(f"Error: failed to download {url}: {e}")


if args.image is None:
    # Download a standard sample test image
    url = "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"
    download(url, DOWNLOAD_PATH)
    image_path = DOWNLOAD_PATH
    print(f"Using image: {image_path} (default sample dog.jpg)")
elif args.image.startswith(("http://", "https://")):
    download(args.image, DOWNLOAD_PATH)
    image_path = DOWNLOAD_PATH
    print(f"Using image from URL: {args.image}")
else:
    if not os.path.isfile(args.image):
        sys.exit(f"Error: image file not found: {args.image}")
    image_path = args.image
    print(f"Using image: {image_path}")

image = Image.open(image_path).convert("RGB")

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

input_tensor = preprocess(image)          # shape [3, 224, 224]
input_batch = input_tensor.unsqueeze(0)   # shape [1, 3, 224, 224]

# Save as raw float32 binary for Lyra (C++) to read
input_batch.numpy().astype(np.float32).tofile("input.bin")
print("Saved input.bin, shape:", input_batch.shape)

# Run the SAME input through real PyTorch, for comparison
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
model.eval()

with torch.no_grad():
    output = model(input_batch)

# Same label file Lyra reads; fall back to bare indices if it's missing.
labels = None
if os.path.isfile("imagenet_classes.txt"):
    with open("imagenet_classes.txt") as f:
        labels = [line.rstrip("\r\n") for line in f]

probs = torch.softmax(output[0], dim=0)
top5 = torch.topk(output[0], 5)
print("\nPyTorch top 5 predictions (class index: label - confidence, raw logit):")
for score, idx in zip(top5.values, top5.indices):
    i = idx.item()
    name = f"{labels[i]} - " if labels else ""
    print(f"  {i}: {name}{100 * probs[i].item():.1f}% (logit {score.item():.4f})")

output.numpy().astype(np.float32).tofile("pytorch_output.bin")
print("\nSaved pytorch_output.bin for comparison")
