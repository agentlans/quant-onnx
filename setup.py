from setuptools import setup, find_packages

setup(
    name='quant-onnx',
    version='0.1.0',
    author='Alan Tseng',
    # author_email='your.email@example.com',
    description='A package for ONNX model conversion and quantization.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/agentlans/quant-onnx',  # Update with repository URL
    packages=find_packages(),  # Automatically find packages in the directory
    install_requires=[
        'onnx',  # Required for loading ONNX models
        'onnxruntime',  # Required for ONNX runtime operations
        'onnxruntime-tools',  # Required for quantization tools
        'onnxsim',  # Required for simplifying ONNX models
    ],
    entry_points={
        'console_scripts': [
            'convert_to_onnx = quant_onnx.convert_to_onnx:main',
            'quantize_onnx = quant_onnx.quantize_onnx:main',
            'verify_onnx = quant_onnx.verify_onnx:main',
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',  # Update if using a different license
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',  # Specify the minimum Python version required
)

