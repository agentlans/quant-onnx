import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
from onnxsim import simplify
import argparse
import tempfile
import os
import subprocess
import logging
import shutil
from typing import Optional, Tuple

# Configure logging based on verbosity level
def configure_logging(log_level: int) -> None:
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')

# Map string log levels to logging module constants
def get_log_level(level_name: str) -> int:
    levels = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    return levels.get(level_name.upper(), logging.WARNING)

# Preprocess the ONNX model using ONNX Runtime command line
def preprocess_model(input_model_path: str) -> Optional[str]:
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.onnx') as temp_file:
            preprocessed_model_path = temp_file.name
            
            logging.info("Starting preprocessing of the model...")
            subprocess.run([
                'python', '-m', 'onnxruntime.quantization.preprocess',
                '--input', input_model_path,
                '--output', preprocessed_model_path
            ], check=True)
            
            logging.info("Model preprocessing completed successfully.")
            return preprocessed_model_path
    except subprocess.CalledProcessError as e:
        logging.error(f"Error during preprocessing {input_model_path}: {e}")
        return None  # Return None if preprocessing fails

# Load the ONNX model
def load_model(model_path: str) -> onnx.ModelProto:
    try:
        model = onnx.load(model_path)
        logging.info("Model loaded successfully.")
        return model
    except Exception as e:
        logging.error(f"Failed to load model from {model_path}: {e}")
        raise

# Quantize the model using dynamic quantization
def quantize_model(model: onnx.ModelProto) -> Tuple[onnx.ModelProto, str]:
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.onnx') as temp_file:
            quantized_model_path = temp_file.name
            
            logging.info("Starting model quantization...")
            quantize_dynamic(model, quantized_model_path, weight_type=QuantType.QUInt8)
            logging.info("Model quantization completed successfully.")
            return onnx.load(quantized_model_path), quantized_model_path
    except Exception as e:
        logging.error(f"Error during quantization: {e}")
        raise

# Simplify the quantized model
def simplify_model(model: onnx.ModelProto) -> onnx.ModelProto:
    try:
        logging.info("Starting model simplification...")
        simplified_model, check = simplify(model)
        
        if check:
            logging.info("Model simplification was successful.")
        else:
            logging.warning("Model simplification failed.")
        
        return simplified_model
    except Exception as e:
        logging.error(f"Error during simplification: {e}")
        raise

# Process all files in the input directory
def process_directory(input_dir: str, output_dir: str) -> None:
    for filename in os.listdir(input_dir):
        input_file_path = os.path.join(input_dir, filename)

        if filename.endswith('.onnx'):
            output_file_path = os.path.join(output_dir, filename)
            
            preprocessed_model_path = preprocess_model(input_file_path)

            # Try to use preprocessed model; if it fails, use original model instead.
            try:
                if preprocessed_model_path:
                    model = load_model(preprocessed_model_path)
                else:
                    logging.info("Using original input model due to preprocessing failure.")
                    model = load_model(input_file_path)

                quantized_model, temp_quantized_model_path = quantize_model(model)
                simplified_model = simplify_model(quantized_model)
                onnx.save(simplified_model, output_file_path)
                logging.info(f"Simplified model saved to {output_file_path}")

            except Exception as e:
                logging.error(f"An error occurred during processing of {filename}: {e}")

            finally:
                cleanup_temp_files(preprocessed_model_path, temp_quantized_model_path)

        else:
            copy_non_onnx_file(input_file_path, output_dir, filename)

def cleanup_temp_files(preprocessed_model_path: Optional[str], temp_quantized_model_path: Optional[str]) -> None:
    """Clean up temporary files created during processing."""
    if preprocessed_model_path and os.path.exists(preprocessed_model_path):
        os.remove(preprocessed_model_path)
        logging.info(f"Removed temporary file: {preprocessed_model_path}")

    if temp_quantized_model_path and os.path.exists(temp_quantized_model_path):
        os.remove(temp_quantized_model_path)
        logging.info(f"Removed temporary file: {temp_quantized_model_path}")

def copy_non_onnx_file(input_file_path: str, output_dir: str, filename: str) -> None:
    """Copy non-ONNX files to the output directory."""
    output_file_copy_path = os.path.join(output_dir, filename)
    shutil.copy2(input_file_path, output_file_copy_path)
    logging.info(f"Copied non-ONNX file to {output_file_copy_path}")

# Main execution flow
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process ONNX models for preprocessing, quantization, and simplification.")
    parser.add_argument("input_dir", type=str, help="Path to the input directory containing ONNX models.")
    parser.add_argument("output_dir", type=str, help="Path to save the processed models and copied files.")
    
    # Add verbosity flag for setting log level
    parser.add_argument('--log', type=str, default='WARNING', 
                        help="Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL). Default is WARNING.")

    args = parser.parse_args()

    # Configure logging based on user input
    log_level = get_log_level(args.log)
    configure_logging(log_level)

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Process all files in the specified directory
    process_directory(args.input_dir, args.output_dir)

