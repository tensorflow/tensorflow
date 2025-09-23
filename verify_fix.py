#!/usr/bin/env python3

"""
Simple verification script for the CropAndResize XLA fix.

This script checks if the basic functionality works without requiring
a full TensorFlow build. It focuses on the original error reproduction.
"""

import sys

def check_tensorflow_available():
    """Check if TensorFlow is available."""
    try:
        import tensorflow as tf
        print(f"✓ TensorFlow {tf.__version__} is available")
        return True, tf
    except ImportError:
        print("✗ TensorFlow is not installed or available")
        return False, None

def test_original_reproduction_case(tf):
    """Test the exact case from the original bug report."""
    print("\nTesting original reproduction case...")

    class SimpleModel(tf.keras.Model):
        def __init__(self):
            super(SimpleModel, self).__init__()

        @tf.function(jit_compile=True)  # This was the failing case
        def call(self, inputs):
            boxes = tf.constant([[0.1, 0.1, 0.5, 0.5]], dtype=tf.float32)
            box_indices = tf.constant([0], dtype=tf.int32)
            cropped = tf.image.crop_and_resize(inputs, boxes, box_indices, [32, 32])
            return cropped

    # Create test input (batch_size=1, height=64, width=64, channels=3)
    inputs = tf.random.uniform([1, 64, 64, 3], dtype=tf.float32)

    model = SimpleModel()

    try:
        # This was throwing: No registered 'CropAndResize' OpKernel for XLA_CPU_JIT
        result = model(inputs)
        print("✓ Original reproduction case PASSED!")
        print(f"✓ Input shape: {inputs.shape}, Output shape: {result.shape}")
        return True
    except Exception as e:
        error_msg = str(e)
        if "No registered 'CropAndResize' OpKernel for XLA" in error_msg:
            print("✗ Original error still occurs - XLA kernel not registered")
        else:
            print(f"✗ Different error occurred: {error_msg}")
        return False

def test_without_jit_compile(tf):
    """Test without JIT compilation to ensure basic functionality works."""
    print("\nTesting without JIT compilation (baseline)...")

    try:
        inputs = tf.random.uniform([1, 64, 64, 3], dtype=tf.float32)
        boxes = tf.constant([[0.1, 0.1, 0.5, 0.5]], dtype=tf.float32)
        box_indices = tf.constant([0], dtype=tf.int32)

        # This should always work (non-XLA path)
        result = tf.image.crop_and_resize(inputs, boxes, box_indices, [32, 32])
        print("✓ Non-JIT compilation works (baseline)")
        print(f"✓ Input shape: {inputs.shape}, Output shape: {result.shape}")
        return True
    except Exception as e:
        print(f"✗ Even baseline functionality failed: {e}")
        return False

def main():
    """Main function."""
    print("=" * 60)
    print("TensorFlow CropAndResize XLA Fix Verification")
    print("=" * 60)

    # Check TensorFlow availability
    tf_available, tf = check_tensorflow_available()
    if not tf_available:
        print("\nCannot proceed without TensorFlow. Please install TensorFlow first.")
        return 1

    # Test baseline functionality
    baseline_works = test_without_jit_compile(tf)
    if not baseline_works:
        print("\nBaseline functionality doesn't work. Check TensorFlow installation.")
        return 1

    # Test the fix
    fix_works = test_original_reproduction_case(tf)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if fix_works:
        print("🎉 SUCCESS: CropAndResize XLA JIT compilation fix is working!")
        print("\nThe original issue has been resolved:")
        print("- tf.image.crop_and_resize now works with @tf.function(jit_compile=True)")
        print("- XLA kernel for CropAndResize has been successfully implemented")
        return 0
    else:
        print("❌ FAILURE: The fix is not working yet.")
        print("\nPossible reasons:")
        print("- The TensorFlow build doesn't include the XLA kernel changes")
        print("- Additional compilation/build steps may be needed")
        print("- There might be compilation errors in the XLA implementation")
        return 1

if __name__ == "__main__":
    exit(main())