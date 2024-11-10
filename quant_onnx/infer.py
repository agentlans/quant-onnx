import torch
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForSeq2SeqLM
from onnxruntime import SessionOptions, GraphOptimizationLevel

class ONNXModel:
    def __init__(self, model_name_or_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Set up ONNX Runtime session options for optimization
        options = SessionOptions()
        options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = 1  # Reduce CPU usage

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        # Load pre-quantized ONNX model
        self.model = ORTModelForSeq2SeqLM.from_pretrained(
            model_name_or_path,
            provider="CUDAExecutionProvider" if self.device.type == "cuda" else "CPUExecutionProvider",
            session_options=options
        )

    def generate(self, input_text, max_length=50):
        """Generate output text from the input text."""
        if not isinstance(input_text, str):
            raise ValueError("Input must be a string.")
        
        with torch.no_grad():
            input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
            output_ids = self.model.generate(input_ids, max_length=max_length)
            output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # Clear CUDA cache if using GPU
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        return output_text

    def generate_batch(self, input_texts, max_length=50):
        """Generate output texts from a batch of input texts."""
        if not isinstance(input_texts, list) or not all(isinstance(text, str) for text in input_texts):
            raise ValueError("Input must be a list of strings.")
        
        with torch.no_grad():
            inputs = self.tokenizer(input_texts, padding=True, truncation=True, return_tensors="pt")
            input_ids = inputs.input_ids.to(self.device)
            attention_mask = inputs.attention_mask.to(self.device)
            
            output_ids = self.model.generate(input_ids, attention_mask=attention_mask, max_length=max_length)
            output_texts = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        # Clear CUDA cache if using GPU
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        return output_texts

    def __del__(self):
        """Destructor to ensure proper cleanup."""
        del self.model
        del self.tokenizer
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

# Example usage:
# inference_model = QuantizedModel("model_name_or_path")
# single_output = inference_model.generate("Your input text here.")
# batch_output = inference_model.generate_batch(["Input text 1", "Input text 2"])
# del inference_model  # Explicitly delete the model when done

