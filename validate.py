import torch
import torch.nn as nn

class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Rebuild the model with the SAME random weights as before by loading
# the exact same seed-generated model isn't possible after the fact —
# so instead we reload the weights that got baked into the ONNX file itself.
import onnx
from onnx import numpy_helper
import numpy as np

onnx_model = onnx.load("tiny_mlp.onnx")

weights = {}
for initializer in onnx_model.graph.initializer:
    weights[initializer.name] = numpy_helper.to_array(initializer)

model = TinyMLP()
with torch.no_grad():
    model.fc1.weight.copy_(torch.tensor(weights["fc1.weight"]))
    model.fc1.bias.copy_(torch.tensor(weights["fc1.bias"]))
    model.fc2.weight.copy_(torch.tensor(weights["fc2.weight"]))
    model.fc2.bias.copy_(torch.tensor(weights["fc2.bias"]))

model.eval()

# Same input as our C++ test: [1, 2, 3, 4]
input_tensor = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

with torch.no_grad():
    output = model(input_tensor)

print("PyTorch output:", output.numpy())
