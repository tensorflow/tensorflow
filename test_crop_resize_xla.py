#!/usr/bin/env python3

"""
Test script to verify CropAndResize XLA JIT compilation fix.

This script reproduces the original issue reported in TensorFlow #100521
and verifies that the XLA kernel implementation works correctly.
"""

import tensorflow as tf
import numpy as np


class SimpleModel(tf.keras.Model):
    def __init__(self):
        super(SimpleModel, self).__init__()

    @tf.function(jit_compile=True)  # This caused the original failure
    def call(self, inputs):
        boxes = tf.constant([[0.1, 0.1, 0.5, 0.5]], dtype=tf.float32)
        box_indices = tf.constant([0], dtype=tf.int32)
        cropped = tf.image.crop_and_resize(inputs, boxes, box_indices, [32, 32])
        return cropped


def test_basic_functionality():
    """Test basic CropAndResize functionality with XLA JIT compilation."""
    print("Testing basic CropAndResize functionality with XLA JIT...")

    # Create test input
    batch_size = 1
    height = 64
    width = 64
    channels = 3

    inputs = tf.random.uniform([batch_size, height, width, channels],
                               minval=0.0, maxval=1.0, dtype=tf.float32)

    # Create model and test
    model = SimpleModel()

    try:
        # This should work without errors now
        result = model(inputs)

        print(f"✓ Success! Input shape: {inputs.shape}")
        print(f"✓ Output shape: {result.shape}")
        print(f"✓ Expected output shape: (1, 32, 32, 3)")

        # Verify output shape
        expected_shape = (1, 32, 32, 3)
        if result.shape == expected_shape:
            print("✓ Output shape matches expected shape!")
            return True
        else:
            print(f"✗ Shape mismatch. Got {result.shape}, expected {expected_shape}")
            return False

    except Exception as e:
        print(f"✗ Error occurred: {e}")
        return False


def test_different_methods():
    """Test both bilinear and nearest neighbor methods."""
    print("\nTesting different interpolation methods...")

    inputs = tf.random.uniform([1, 100, 100, 3], dtype=tf.float32)
    boxes = tf.constant([[0.2, 0.2, 0.8, 0.8]], dtype=tf.float32)
    box_indices = tf.constant([0], dtype=tf.int32)
    crop_size = [50, 50]

    methods = ["bilinear", "nearest"]

    for method in methods:
        print(f"  Testing {method} method...")

        @tf.function(jit_compile=True)
        def crop_with_method(inputs, boxes, box_indices, crop_size):
            return tf.image.crop_and_resize(inputs, boxes, box_indices,
                                            crop_size, method=method)

        try:
            result = crop_with_method(inputs, boxes, box_indices, crop_size)
            print(f"    ✓ {method} method works! Output shape: {result.shape}")
        except Exception as e:
            print(f"    ✗ {method} method failed: {e}")
            return False

    return True


def test_multiple_boxes():
    """Test with multiple bounding boxes."""
    print("\nTesting with multiple boxes...")

    inputs = tf.random.uniform([2, 80, 80, 3], dtype=tf.float32)
    # Multiple boxes for different crops
    boxes = tf.constant([
        [0.1, 0.1, 0.4, 0.4],  # Top-left crop
        [0.6, 0.6, 0.9, 0.9],  # Bottom-right crop
        [0.0, 0.3, 1.0, 0.7],  # Horizontal strip
    ], dtype=tf.float32)
    box_indices = tf.constant([0, 1, 0], dtype=tf.int32)  # Use both batch items

    @tf.function(jit_compile=True)
    def multi_crop(inputs, boxes, box_indices):
        return tf.image.crop_and_resize(inputs, boxes, box_indices, [40, 40])

    try:
        result = multi_crop(inputs, boxes, box_indices)
        expected_shape = (3, 40, 40, 3)  # 3 boxes, 40x40 crop, 3 channels

        print(f"  ✓ Multiple boxes work! Output shape: {result.shape}")
        if result.shape == expected_shape:
            print(f"  ✓ Shape correct: {result.shape}")
            return True
        else:
            print(f"  ✗ Shape incorrect. Got {result.shape}, expected {expected_shape}")
            return False

    except Exception as e:
        print(f"  ✗ Multiple boxes failed: {e}")
        return False


def test_edge_cases():
    """Test edge cases like small crops and extrapolation."""
    print("\nTesting edge cases...")

    inputs = tf.random.uniform([1, 50, 50, 1], dtype=tf.float32)

    # Test small crop size
    boxes = tf.constant([[0.25, 0.25, 0.75, 0.75]], dtype=tf.float32)
    box_indices = tf.constant([0], dtype=tf.int32)

    @tf.function(jit_compile=True)
    def small_crop(inputs, boxes, box_indices):
        return tf.image.crop_and_resize(inputs, boxes, box_indices, [2, 2])

    try:
        result = small_crop(inputs, boxes, box_indices)
        print(f"  ✓ Small crop (2x2) works! Output shape: {result.shape}")
    except Exception as e:
        print(f"  ✗ Small crop failed: {e}")
        return False

    # Test with boxes outside image bounds (should use extrapolation_value)
    boxes_outside = tf.constant([[-0.1, -0.1, 0.3, 0.3]], dtype=tf.float32)

    @tf.function(jit_compile=True)
    def crop_outside_bounds(inputs, boxes, box_indices):
        return tf.image.crop_and_resize(inputs, boxes, box_indices, [20, 20],
                                        extrapolation_value=0.5)

    try:
        result = crop_outside_bounds(inputs, boxes_outside, box_indices)
        print(f"  ✓ Out-of-bounds crop works! Output shape: {result.shape}")
        return True
    except Exception as e:
        print(f"  ✗ Out-of-bounds crop failed: {e}")
        return False


def main():
    """Main test function."""
    print("=" * 60)
    print("TensorFlow CropAndResize XLA JIT Compilation Test")
    print("=" * 60)
    print(f"TensorFlow version: {tf.__version__}")
    print(f"XLA JIT enabled: {tf.config.optimizer.get_jit() is not None}")

    # Run all tests
    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("Different Methods", test_different_methods),
        ("Multiple Boxes", test_multiple_boxes),
        ("Edge Cases", test_edge_cases),
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n{'-' * 40}")
        print(f"Running: {test_name}")
        print(f"{'-' * 40}")
        success = test_func()
        results.append((test_name, success))

    # Summary
    print(f"\n{'=' * 60}")
    print("TEST SUMMARY")
    print(f"{'=' * 60}")

    passed = 0
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"{test_name:<20}: {status}")
        if success:
            passed += 1

    print(f"\nOverall: {passed}/{len(results)} tests passed")

    if passed == len(results):
        print("\n🎉 All tests passed! CropAndResize XLA JIT compilation is working!")
        return 0
    else:
        print(f"\n❌ {len(results) - passed} test(s) failed.")
        return 1


if __name__ == "__main__":
    exit(main())