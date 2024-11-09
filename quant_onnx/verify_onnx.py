import sys
from optimum.onnxruntime import ORTModelForSeq2SeqLM
from transformers import AutoTokenizer

def verify_onnx_model(model_dir):
    print(f"Verifying ONNX model in directory: {model_dir}")

    try:
        # Load the ONNX model
        model = ORTModelForSeq2SeqLM.from_pretrained(model_dir)
        print("ONNX model loaded successfully.")

        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        print("Tokenizer loaded successfully.")

        # Prepare input
        input_text = "translate English to German: Hello, how are you?"
        inputs = tokenizer(input_text, return_tensors="pt")

        # Generate output
        outputs = model.generate(**inputs)

        # Decode the output
        decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

        print(f"\nInput: {input_text}")
        print(f"Output: {decoded_output}")
        print("\nVerification completed successfully!")

    except Exception as e:
        print(f"Error during verification: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python verify_onnx_model.py <model_directory>")
        sys.exit(1)

    model_dir = sys.argv[1]
    verify_onnx_model(model_dir)
