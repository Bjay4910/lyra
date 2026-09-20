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

model = TinyMLP()
model.eval()

dummy_input = torch.randn(1, 4)

torch.onnx.export(
    model,
    dummy_input,
    "tiny_mlp.onnx",
    input_names=["input"],
    output_names=["output"],
)

print("Exported tiny_mlp.onnx successfully.")
