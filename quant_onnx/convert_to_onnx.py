import sys
import os
import logging
from optimum.onnxruntime import ORTModelForSeq2SeqLM
from transformers import AutoTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def convert_to_onnx(model_name: str, output_dir: str) -> None:
    """Converts a model to ONNX format and saves it along with the tokenizer."""
    logging.info(f"Starting conversion of {model_name} to ONNX format...")

    # Create the output directory if it doesn't exist
    try:
        os.makedirs(output_dir, exist_ok=True)
        logging.info(f"Output directory created at {output_dir}")
    except Exception as e:
        logging.error(f"Failed to create output directory: {e}")
        sys.exit(1)

    # Load and convert the model to ONNX
    try:
        ort_model = ORTModelForSeq2SeqLM.from_pretrained(model_name, export=True)
        logging.info(f"Model {model_name} loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load model {model_name}: {e}")
        sys.exit(1)

    # Load the tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        logging.info("Tokenizer loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load tokenizer for {model_name}: {e}")
        sys.exit(1)

    # Save the ONNX model
    try:
        ort_model.save_pretrained(output_dir)
        logging.info(f"ONNX model saved to {output_dir}")
    except Exception as e:
        logging.error(f"Failed to save ONNX model: {e}")
        sys.exit(1)

    # Save the tokenizer
    try:
        tokenizer.save_pretrained(output_dir)
        logging.info(f"Tokenizer saved to {output_dir}")
    except Exception as e:
        logging.error(f"Failed to save tokenizer: {e}")
        sys.exit(1)

def main():
    """Main function to handle command-line arguments and invoke conversion."""
    if len(sys.argv) != 3:
        print("Usage: python convert_to_onnx.py <model_name> <output_directory>")
        sys.exit(1)

    model_name = sys.argv[1]
    output_dir = sys.argv[2]

    convert_to_onnx(model_name, output_dir)
    logging.info("Conversion completed successfully!")

if __name__ == "__main__":
    main()

