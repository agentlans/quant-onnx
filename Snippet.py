import quant_onnx
from quant_onnx.infer import QuantizedModel

model = QuantizedModel("my-onnx-optimized-model")
model.generate("Fly me to the moon.")

