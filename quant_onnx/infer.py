import torch
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForSeq2SeqLM
from onnxruntime import SessionOptions, GraphOptimizationLevel

# Set up ONNX Runtime session options for optimization
options = SessionOptions()
options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
options.intra_op_num_threads = 1  # Reduce CPU usage

class QuantizedModel:
    def __init__(self, model_name_or_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        # Load pre-quantized ONNX model
        self.model = ORTModelForSeq2SeqLM.from_pretrained(
            model_name_or_path,
            provider="CUDAExecutionProvider" if self.device.type == "cuda" else "CPUExecutionProvider",
            session_options=options
        )

        # Move model to the appropriate device
        self.model.to(self.device)

    def generate(self, input_text, max_length=50):
        """Generate output text from the input text."""
        # Validate input type
        if not isinstance(input_text, str):
            raise ValueError("Input must be a string.")
        
        # Tokenize input
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)

        # Generate output
        output_ids = self.model.generate(input_ids, max_length=max_length)

        # Decode output
        output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        return output_text

    def generate_batch(self, input_texts, max_length=50):
        """Generate output texts from a batch of input texts."""
        # Validate input type
        if not isinstance(input_texts, list) or not all(isinstance(text, str) for text in input_texts):
            raise ValueError("Input must be a list of strings.")
        
        # Tokenize inputs in batch with padding and truncation
        input_ids = self.tokenizer(input_texts, padding=True, truncation=True, return_tensors="pt").input_ids.to(self.device)

        # Generate outputs for the batch
        output_ids = self.model.generate(input_ids, max_length=max_length)

        # Decode outputs for each generated sequence and return as a list
        output_texts = [self.tokenizer.decode(output_id, skip_special_tokens=True) for output_id in output_ids]

        return output_texts

# Example usage:
# inference_model = QuantizedModel("model_name_or_path")
# single_output = inference_model.generate("Your input text here.")
# batch_output = inference_model.generate_batch(["Input text 1", "Input text 2"])

