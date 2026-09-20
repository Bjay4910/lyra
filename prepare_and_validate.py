import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import urllib.request
import numpy as np

# Download a standard sample test image
url = "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"
urllib.request.urlretrieve(url, "test_image.jpg")

image = Image.open("test_image.jpg").convert("RGB")

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

top5 = torch.topk(output[0], 5)
print("\nPyTorch top 5 predictions (class index : score):")
for score, idx in zip(top5.values, top5.indices):
    print(f"  class {idx.item()}: {score.item():.4f}")

output.numpy().astype(np.float32).tofile("pytorch_output.bin")
print("\nSaved pytorch_output.bin for comparison")
