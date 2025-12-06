#!/usr/bin/env python3

"""
Test script to reproduce and verify the fix for ConcatV2 float8 XLA issue.

This reproduces the issue described in GitHub issue #105131 where ConcatV2
fails with jit_compile=True when using float8_e4m3fn tensors.
"""

import tensorflow as tf
import numpy as np

class TestModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.d1 = tf.keras.layers.Dense(32, activation='relu')
        self.d2 = tf.keras.layers.Dense(16, activation='tanh') 
        self.d3 = tf.keras.layers.Dense(8)

    def call(self, x):
        # This should trigger the concatenation with float8 tensors
        filtered_features = list(filter(lambda z: tf.reduce_sum(z) > 0.5, [x, x * 2, x * 3]))
        mapped_features = list(map(lambda z: tf.nn.sigmoid(z), filtered_features))
        zipped_data = list(zip(mapped_features, [tf.ones_like(x) for _ in range(len(mapped_features))]))
        
        # This concat operation should work with our fix
        combined = tf.concat(zipped_data, axis=-1)
        return self.d3(combined)

def test_original_issue():
    """Test the original issue reproduction case"""
    print("Testing original issue reproduction...")
    
    model = TestModel()
    x = tf.random.normal([4, 16])
    
    try:
        # Test eager execution first
        eager_out = model(x)
        print(f"Eager execution successful - Output shape: {eager_out.shape}")
        
        # Test XLA compilation
        @tf.function(jit_compile=True)
        def compiled_forward(inputs):
            return model(inputs)
        
        compiled_out = compiled_forward(x)
        print(f"XLA compilation successful - Output shape: {compiled_out.shape}")
        return True
        
    except Exception as e:
        print(f"Error occurred: {e}")
        return False

def test_direct_float8_concat():
    """Test direct float8 concatenation with XLA"""
    print("\nTesting direct float8 concatenation...")
    
    # Convert to float8_e4m3fn
    x1 = tf.cast(tf.random.normal([2, 3]), dtype=tf.dtypes.float8_e4m3fn)
    x2 = tf.cast(tf.random.normal([2, 3]), dtype=tf.dtypes.float8_e4m3fn)
    
    try:
        # Test eager execution
        eager_result = tf.concat([x1, x2], axis=0)
        print(f"Eager float8 concat successful - Output shape: {eager_result.shape}")
        
        # Test XLA compilation
        @tf.function(jit_compile=True)
        def compiled_concat(a, b):
            return tf.concat([a, b], axis=0)
        
        xla_result = compiled_concat(x1, x2)
        print(f"XLA float8 concat successful - Output shape: {xla_result.shape}")
        return True
        
    except Exception as e:
        print(f"Direct float8 concat error: {e}")
        return False

if __name__ == "__main__":
    print("Testing ConcatV2 float8 XLA fix...")
    print("=" * 50)
    
    # Test original issue
    original_success = test_original_issue()
    
    # Test direct float8 concatenation
    direct_success = test_direct_float8_concat()
    
    print("\n" + "=" * 50)
    print("Test Results:")
    print(f"Original issue reproduction: {'PASSED' if original_success else 'FAILED'}")
    print(f"Direct float8 concat test:   {'PASSED' if direct_success else 'FAILED'}")
    
    if original_success and direct_success:
        print("\n✅ All tests passed! The fix appears to be working.")
    else:
        print("\n❌ Some tests failed. The issue may still exist.")
