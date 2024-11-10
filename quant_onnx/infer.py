import torch
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForSeq2SeqLM
from onnxruntime import SessionOptions, GraphOptimizationLevel
import asyncio
from functools import lru_cache
import logging

class ONNXModel:
    def __init__(self, model_name_or_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        options = SessionOptions()
        options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = 1

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

        self.model = ORTModelForSeq2SeqLM.from_pretrained(
            model_name_or_path,
            provider="CUDAExecutionProvider" if self.device.type == "cuda" else "CPUExecutionProvider",
            session_options=options
        )

        self.logger = logging.getLogger(__name__)

    @lru_cache(maxsize=100)
    def generate(self, input_text, max_length=50):
        """Generate output text from the input text with caching."""
        return self._generate([input_text], max_length)[0]

    async def generate_async(self, input_text, max_length=50):
        """Asynchronous version of generate."""
        return await asyncio.to_thread(self.generate, input_text, max_length)

    def generate_batch(self, input_texts, max_length=50):
        """Generate output texts from a batch of input texts."""
        return self._generate(input_texts, max_length)

    def _generate(self, input_texts, max_length):
        if not all(isinstance(text, str) for text in input_texts):
            raise ValueError("All inputs must be strings.")
        
        try:
            with torch.no_grad():
                inputs = self.tokenizer(input_texts, padding=True, truncation=True, return_tensors="pt")
                input_ids = inputs.input_ids.to(self.device)
                attention_mask = inputs.attention_mask.to(self.device)
                
                output_ids = self.model.generate(input_ids, attention_mask=attention_mask, max_length=max_length)
                output_texts = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

            if self.device.type == "cuda":
                torch.cuda.empty_cache()

            return output_texts
        except Exception as e:
            self.logger.error(f"Error in generate: {str(e)}")
            raise

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__del__()

    def __del__(self):
        try:
            del self.model
            del self.tokenizer
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        except Exception as e:
            self.logger.error(f"Error in cleanup: {str(e)}")

# Example usage:
# async def main():
#     with ONNXModel("model_name_or_path") as inference_model:
#         single_output = await inference_model.generate_async("Your input text here.")
#         batch_output = inference_model.generate_batch(["Input text 1", "Input text 2"])
#     print(single_output, batch_output)

# asyncio.run(main())

