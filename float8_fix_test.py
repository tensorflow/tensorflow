#!/usr/bin/env python3
"""
Test to verify ConcatV2 Float8 XLA fix works correctly.
This should be run after building TensorFlow with the fix.
"""

import tensorflow as tf
import numpy as np

def test_simple_float8_concat():
    """Test simple float8 concatenation with XLA compilation"""
    print("Testing simple float8 concatenation...")
    
    # Create some test tensors and convert to float8_e4m3fn
    x = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32)
    y = tf.constant([[5.0, 6.0], [7.0, 8.0]], dtype=tf.float32)
    
    # Cast to float8_e4m3fn (the type mentioned in the original issue)
    x_f8 = tf.cast(x, tf.float8_e4m3fn)
    y_f8 = tf.cast(y, tf.float8_e4m3fn)
    
    @tf.function(jit_compile=True)
    def concat_with_xla(a, b):
        return tf.concat([a, b], axis=0)
    
    try:
        result = concat_with_xla(x_f8, y_f8)
        print(f"✅ Success! Result shape: {result.shape}, dtype: {result.dtype}")
        print(f"   Result values: {result}")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_original_reproduction_case():
    """Test the exact case from the GitHub issue"""
    print("\nTesting original reproduction case...")
    
    class TestModel(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.d1 = tf.keras.layers.Dense(32, activation='relu')
            self.d2 = tf.keras.layers.Dense(16, activation='tanh')
            self.d3 = tf.keras.layers.Dense(8)

        def call(self, x):
            filtered_features = list(filter(lambda z: tf.reduce_sum(z) > 0.5, [x, x * 2, x * 3]))
            mapped_features = list(map(lambda z: tf.nn.sigmoid(z), filtered_features))
            zipped_data = list(zip(mapped_features, [tf.ones_like(x) for _ in range(len(mapped_features))]))
            combined = tf.concat(zipped_data, axis=-1)
            return self.d3(combined)

    model = TestModel()
    x = tf.random.normal([4, 16])
    
    try:
        # Test eager first
        eager_result = model(x)
        print(f"✅ Eager execution successful: {eager_result.shape}")
        
        # Test XLA compilation
        @tf.function(jit_compile=True)
        def compiled_model(inputs):
            return model(inputs)
        
        xla_result = compiled_model(x)
        print(f"✅ XLA compilation successful: {xla_result.shape}")
        return True
        
    except Exception as e:
        print(f"❌ Error in reproduction case: {e}")
        return False

def main():
    print("ConcatV2 Float8 XLA Fix Test")
    print("=" * 40)
    print("This test verifies that the fix for GitHub issue #105131 works.")
    print("Make sure TensorFlow is built with the concat_op.cc changes.\n")
    
    # Run tests
    test1_passed = test_simple_float8_concat()
    test2_passed = test_original_reproduction_case()
    
    print("\n" + "=" * 40)
    print("Test Results:")
    print(f"Simple float8 concat:    {'PASS' if test1_passed else 'FAIL'}")
    print(f"Original reproduction:   {'PASS' if test2_passed else 'FAIL'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 All tests passed! The fix is working correctly.")
    else:
        print("\n⚠️  Some tests failed. The fix may need more work or TensorFlow needs to be rebuilt.")
        print("   Make sure you've built TensorFlow with the modified concat_op.cc file.")

if __name__ == "__main__":
    main()
