import torch
import torchvision.models as models

model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
model.eval()

dummy_input = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model,
    dummy_input,
    "mobilenetv2.onnx",
    input_names=["input"],
    output_names=["output"],
)

print("Exported mobilenetv2.onnx successfully.")
