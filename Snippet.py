import quant_onnx
from quant_onnx.infer import ONNXModel

model = ONNXModel("my-onnx-optimized-model")
model.generate("Your prompt here.")

