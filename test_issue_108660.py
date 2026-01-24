#!/usr/bin/env python3
"""
Test for TensorFlow issue #108660 - XLA JIT compilation with bicubic resize.

This test verifies that the ResizeBicubicOp XLA kernel registration fix works
by running tf.image.resize with method='bicubic' under XLA JIT compilation.

Prior to the fix, this would fail with:
  tensorflow.python.framework.errors_impl.InvalidArgumentError: No registered 
  'ResizeBicubic' OpKernel for XLA_CPU_JIT devices
  
After the fix, the operation should complete successfully.
"""

import tensorflow as tf
import numpy as np


def test_resize_bicubic_with_xla():
    """Test that bicubic resize works with XLA JIT compilation."""
    
    print("Testing TensorFlow issue #108660 fix...")
    print("=" * 70)
    
    # Create a simple test image (1, 4, 4, 3) - NHWC format
    test_image = np.array([[
        [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
        [[13, 14, 15], [16, 17, 18], [19, 20, 21], [22, 23, 24]],
        [[25, 26, 27], [28, 29, 30], [31, 32, 33], [34, 35, 36]],
        [[37, 38, 39], [40, 41, 42], [43, 44, 45], [46, 47, 48]]
    ]], dtype=np.float32)
    
    print(f"Input shape: {test_image.shape}")
    print(f"Target size: (8, 8)")
    print()
    
    # Test 1: Without XLA (should always work)
    print("Test 1: Resize without XLA JIT...")
    try:
        result_no_xla = tf.image.resize(
            test_image, 
            size=(8, 8), 
            method='bicubic'
        )
        print(f"✓ Success! Output shape: {result_no_xla.shape}")
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False
    
    print()
    
    # Test 2: With XLA JIT (this is what the fix enables)
    print("Test 2: Resize with XLA JIT compilation...")
    
    @tf.function(jit_compile=True)
    def resize_bicubic_xla(images):
        return tf.image.resize(images, size=(8, 8), method='bicubic')
    
    try:
        result_with_xla = resize_bicubic_xla(test_image)
        print(f"✓ Success! Output shape: {result_with_xla.shape}")
        print()
        print("=" * 70)
        print("✓ Issue #108660 is FIXED!")
        print("  ResizeBicubic now works with XLA JIT compilation")
        return True
    except tf.errors.InvalidArgumentError as e:
        if "No registered 'ResizeBicubic' OpKernel" in str(e):
            print(f"✗ Failed with expected error (not fixed yet):")
            print(f"  {e}")
            return False
        else:
            print(f"✗ Failed with unexpected error:")
            print(f"  {e}")
            return False
    except Exception as e:
        print(f"✗ Failed with unexpected error:")
        print(f"  {e}")
        return False


if __name__ == "__main__":
    success = test_resize_bicubic_with_xla()
    exit(0 if success else 1)
