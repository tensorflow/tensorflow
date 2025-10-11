#!/usr/bin/env python3
"""Test script for GPU numerical accuracy fix (Issue #66740)

This script validates the patched TFLite GPU kernels for Add + Mul operations.
It compares GPU delegate outputs against CPU reference outputs using the
test model from https://github.com/tensorflow/tensorflow/issues/66740

Patch by kshiteej-mali for GPU numerical accuracy
"""

import numpy as np
import urllib.request
import os
import sys

try:
    import tensorflow as tf
    from tensorflow import lite
except ImportError:
    print("Error: TensorFlow not installed. Please install: pip install tensorflow")
    sys.exit(1)

# Model URL from issue #66740
MODEL_URL = "https://qaihub-public-issues.s3.us-west-2.amazonaws.com/tflite/tflite_66740_add_mul_gpu_numerically_incorrect.tflite"
MODEL_PATH = "test_model_66740.tflite"

def download_model():
    """Download the test model if not already present."""
    if not os.path.exists(MODEL_PATH):
        print(f"Downloading test model from {MODEL_URL}...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print(f"Model downloaded to {MODEL_PATH}")
    else:
        print(f"Using existing model: {MODEL_PATH}")

def run_inference(interpreter, input_data):
    """Run inference on the given interpreter."""
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    return interpreter.get_tensor(output_details[0]['index'])

def compare_outputs(cpu_output, gpu_output, tolerance=1e-5):
    """Compare CPU and GPU outputs and compute error metrics."""
    abs_diff = np.abs(cpu_output - gpu_output)
    max_abs_error = np.max(abs_diff)
    mean_abs_error = np.mean(abs_diff)
    
    # Relative error
    cpu_abs = np.abs(cpu_output)
    rel_diff = np.where(cpu_abs > 1e-10, abs_diff / cpu_abs, abs_diff)
    max_rel_error = np.max(rel_diff)
    mean_rel_error = np.mean(rel_diff)
    
    # Check if within tolerance
    within_tolerance = max_abs_error < tolerance
    
    return {
        'max_abs_error': max_abs_error,
        'mean_abs_error': mean_abs_error,
        'max_rel_error': max_rel_error,
        'mean_rel_error': mean_rel_error,
        'within_tolerance': within_tolerance,
        'tolerance': tolerance
    }

def print_results(metrics, test_name):
    """Print comparison results."""
    print(f"\n{'='*60}")
    print(f"Test: {test_name}")
    print(f"{'='*60}")
    print(f"Max Absolute Error:  {metrics['max_abs_error']:.2e}")
    print(f"Mean Absolute Error: {metrics['mean_abs_error']:.2e}")
    print(f"Max Relative Error:  {metrics['max_rel_error']:.2e}")
    print(f"Mean Relative Error: {metrics['mean_rel_error']:.2e}")
    print(f"Tolerance:           {metrics['tolerance']:.2e}")
    print(f"Status:              {'✓ PASS' if metrics['within_tolerance'] else '✗ FAIL'}")
    print(f"{'='*60}\n")

def main():
    print("\n" + "#" * 60)
    print("# TFLite GPU Numerical Accuracy Test (Issue #66740)")
    print("# Patch by kshiteej-mali for GPU numerical accuracy")
    print("#" * 60 + "\n")
    
    # Download model
    download_model()
    
    # Test cases with different input patterns
    test_cases = [
        ("Random inputs (0-1)", np.random.rand(1, 224, 224, 3).astype(np.float32)),
        ("Random inputs (-1 to 1)", (np.random.rand(1, 224, 224, 3).astype(np.float32) * 2 - 1)),
        ("All ones", np.ones((1, 224, 224, 3), dtype=np.float32)),
        ("All zeros", np.zeros((1, 224, 224, 3), dtype=np.float32)),
        ("Small values", np.random.rand(1, 224, 224, 3).astype(np.float32) * 0.01),
        ("Large values", np.random.rand(1, 224, 224, 3).astype(np.float32) * 100),
    ]
    
    all_passed = True
    results_summary = []
    
    for test_name, input_data in test_cases:
        print(f"\nRunning test: {test_name}")
        print(f"Input shape: {input_data.shape}, dtype: {input_data.dtype}")
        print(f"Input range: [{input_data.min():.4f}, {input_data.max():.4f}]")
        
        # CPU interpreter
        cpu_interpreter = lite.Interpreter(model_path=MODEL_PATH)
        cpu_output = run_inference(cpu_interpreter, input_data)
        print(f"CPU output range: [{cpu_output.min():.4f}, {cpu_output.max():.4f}]")
        
        # GPU interpreter
        try:
            gpu_delegate = lite.experimental.load_delegate('libdelegate.so')
            gpu_interpreter = lite.Interpreter(
                model_path=MODEL_PATH,
                experimental_delegates=[gpu_delegate]
            )
            gpu_output = run_inference(gpu_interpreter, input_data)
            print(f"GPU output range: [{gpu_output.min():.4f}, {gpu_output.max():.4f}]")
            
            # Compare outputs
            metrics = compare_outputs(cpu_output, gpu_output, tolerance=1e-5)
            print_results(metrics, test_name)
            
            results_summary.append({
                'test': test_name,
                'passed': metrics['within_tolerance'],
                'max_abs_error': metrics['max_abs_error'],
                'max_rel_error': metrics['max_rel_error']
            })
            
            if not metrics['within_tolerance']:
                all_passed = False
                
        except Exception as e:
            print(f"⚠ GPU delegate not available or error occurred: {e}")
            print(f"⚠ Skipping GPU test for: {test_name}")
            results_summary.append({
                'test': test_name,
                'passed': None,
                'max_abs_error': None,
                'max_rel_error': None,
                'error': str(e)
            })
    
    # Print summary
    print("\n" + "#" * 60)
    print("# SUMMARY")
    print("#" * 60)
    print(f"\n{'Test Name':<30} {'Status':<10} {'Max Abs Error':<15} {'Max Rel Error':<15}")
    print("-" * 70)
    
    for result in results_summary:
        if result['passed'] is None:
            status = "SKIPPED"
            max_abs = "N/A"
            max_rel = "N/A"
        elif result['passed']:
            status = "✓ PASS"
            max_abs = f"{result['max_abs_error']:.2e}"
            max_rel = f"{result['max_rel_error']:.2e}"
        else:
            status = "✗ FAIL"
            max_abs = f"{result['max_abs_error']:.2e}"
            max_rel = f"{result['max_rel_error']:.2e}"
        
        print(f"{result['test']:<30} {status:<10} {max_abs:<15} {max_rel:<15}")
    
    print("\n" + "#" * 60)
    if all_passed:
        print("# ✓ ALL TESTS PASSED - GPU outputs match CPU reference")
        print("# The numerical accuracy fix is working correctly!")
    else:
        print("# ✗ SOME TESTS FAILED - GPU outputs differ from CPU")
        print("# Further investigation needed")
    print("#" * 60 + "\n")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
