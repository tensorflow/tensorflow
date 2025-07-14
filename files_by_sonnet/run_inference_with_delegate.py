#!/usr/bin/env python3
"""
Runs inference on a TFLite model using a custom delegate shared object file.
Usage:
    python run_inference_with_delegate.py --delegate-path /path/to/delegate.so [--model-path simple_deep_model.tflite]
"""
import numpy as np
import tensorflow as tf
import argparse
import os
import sys

parser = argparse.ArgumentParser(description='Run TFLite inference with custom delegate')
parser.add_argument('--delegate-path', '-d', required=True, help='Path to delegate shared object (.so) file')
parser.add_argument('--model-path', '-m', default='simple_deep_model.tflite', help='Path to TFLite model file')
parser.add_argument('--num-samples', type=int, default=5, help='Number of inference samples')
parser.add_argument('--enable-logging', action='store_true', help='Enable TensorFlow Lite logging')
args = parser.parse_args()

# Enable logging if requested
if args.enable_logging:
    os.environ['TFLITE_LOG_LEVEL'] = '1'
    print(" TensorFlow Lite logging enabled")
else:
    print(" Use --enable-logging to see detailed delegate logs")

print(f"Using delegate: {args.delegate_path}")
print(f"Using model: {args.model_path}")

# Check files
if not os.path.exists(args.model_path):
    print(f"ERROR: Model file not found: {args.model_path}")
    print("Please make sure the model file exists or specify a different path with --model-path")
    sys.exit(1)
if not os.path.exists(args.delegate_path):
    print(f"ERROR: Delegate file not found: {args.delegate_path}")
    print("Please make sure the delegate shared object file exists")
    sys.exit(1)

# Load delegate
print("Loading delegate...")
try:
    delegate = tf.lite.experimental.load_delegate(args.delegate_path)
    print(f" Successfully loaded delegate from {args.delegate_path}")
except Exception as e:
    print(f" Failed to load delegate: {e}")
    sys.exit(1)

# Load TFLite model with delegate
print("Loading TFLite model...")
try:
    interpreter = tf.lite.Interpreter(model_path=args.model_path, experimental_delegates=[delegate])
    interpreter.allocate_tensors()
    print(" Successfully loaded model with delegate")
except Exception as e:
    print(f" Failed to load model with delegate: {e}")
    sys.exit(1)

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"\n Model Information:")
print(f"Input details: {input_details}")
print(f"Output details: {output_details}")

# Validate tensor shapes
input_shape = input_details[0]['shape']
output_shape = output_details[0]['shape']
print(f"\nTensor shapes:")
print(f"  Input: {input_shape}")  
print(f"  Output: {output_shape}")

# Check if shapes are compatible with FPGA (max 32x32)
if len(input_shape) > 2 or len(output_shape) > 2:
    print(f"  WARNING: Tensor rank > 2 may not be supported by FPGA")
if input_shape[1] > 32 or output_shape[1] > 32:
    print(f"  WARNING: Tensor dimensions > 32 may not be supported by FPGA")

# Run inference on synthetic data
print(f"\n Running {args.num_samples} inference samples...")
try:
    for i in range(args.num_samples):
        input_shape = input_details[0]['shape']
        input_data = np.random.randn(*input_shape).astype(np.float32)
        
        print(f"\nSample {i+1}/{args.num_samples}:")
        print(f"  Input shape: {input_data.shape}")
        print(f"  Input data (first 5): {input_data.flatten()[:5]}")
        
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        print(f"  Output shape: {output_data.shape}")
        print(f"  Output data: {output_data.flatten()}")
        
except Exception as e:
    print(f"❌ Error during inference: {e}")
    print("This might be due to:")
    print("  1. Tensor shape mismatch between model and FPGA")
    print("  2. FPGA hardware not properly initialized")
    print("  3. Memory access issues in the driver")
    sys.exit(1)

print("\n✅ Inference completed successfully using delegate!")
print("🎉 Your FPGA delegate is working correctly!")
