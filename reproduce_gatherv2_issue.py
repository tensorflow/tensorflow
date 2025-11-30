#!/usr/bin/env python3
"""
Reproduce the GatherV2 CUDA initialization issue.
Issue #104876: CUDA device context initialization failure with invalid axis parameter.
"""

import tensorflow as tf
import sys

def test_gatherv2_invalid_axis():
    """Test GatherV2 with invalid axis parameter to reproduce the CUDA issue."""
    print("TensorFlow version:", tf.__version__)
    print("GPU Available:", tf.config.list_physical_devices('GPU'))
    
    # Create test data
    resource = tf.Variable([[1, 2, 3], [4, 5, 6]], dtype=tf.int32)
    indices = tf.constant([0, 1], dtype=tf.int32)
    dtype = tf.int32
    batch_dims = 0
    
    print("\nTesting GatherV2 with valid axis (axis=1):")
    try:
        # This should work fine
        output_valid = tf.raw_ops.GatherV2(
            params=resource, 
            indices=indices, 
            axis=1, 
            batch_dims=batch_dims
        )
        print("Valid axis test passed:", output_valid.numpy())
    except Exception as e:
        print(f"Unexpected error with valid axis: {e}")
    
    print("\nTesting GatherV2 with invalid axis (axis=9):")
    try:
        # This should fail gracefully but instead causes CUDA context issue
        output_invalid = tf.raw_ops.GatherV2(
            params=resource, 
            indices=indices, 
            axis=9,  # Invalid axis for 2D tensor
            batch_dims=batch_dims
        )
        print("Invalid axis test unexpectedly passed:", output_invalid.numpy())
    except Exception as e:
        print(f"Error with invalid axis: {type(e).__name__}: {e}")
        return str(e)

if __name__ == "__main__":
    error_msg = test_gatherv2_invalid_axis()
    if error_msg and "CUDA_ERROR_OUT_OF_MEMORY" in error_msg:
        print("\n✗ REPRODUCED: CUDA initialization failure due to invalid axis!")
        sys.exit(1)
    else:
        print("\n✓ Issue not reproduced - may be environment specific")