"""Test for dynamic shape preservation in concat operation with XLA compilation."""

import tensorflow as tf
import unittest


class ConcatDynamicShapeTest(unittest.TestCase):

  def test_dynamic_partition_concat_topk_matmul_xla(self):
    """Test that XLA preserves dynamic shapes through the full pipeline."""

    class TestModel(tf.keras.Model):
      def __init__(self):
        super().__init__()
        self.d1 = tf.keras.layers.Dense(64, activation='relu')
        self.d2 = tf.keras.layers.Dense(32)
        self.d3 = tf.keras.layers.Dense(16)

      def call(self, x, indices=None):
        x = self.d1(x)
        if indices is not None:
          (unique_vals, _) = tf.unique(indices)
          x = tf.nn.relu(tf.gather(x, unique_vals))
        else:
          x = tf.nn.relu(x)
        
        # This chain should preserve dynamic dimensions
        partitioned = tf.dynamic_partition(
            x, tf.cast(tf.reduce_sum(x, axis=1) > 0, tf.int32), 
            num_partitions=2)
        x = tf.concat(partitioned, axis=0)  # Critical: must preserve dynamic dim
        (top_k_values, _) = tf.nn.top_k(x, k=tf.shape(x)[0] // 2)
        x = tf.nn.relu(self.d2(top_k_values))
        return self.d3(x)

    model = TestModel()
    x = tf.random.normal([10, 64])
    indices = tf.random.uniform([10], maxval=5, dtype=tf.int32)
    
    # Test eager execution
    eager_out = model(x, indices)
    self.assertEqual(len(eager_out.shape), 2)
    self.assertEqual(eager_out.shape[1], 16)
    
    # Test XLA compilation - this should not fail with shape errors
    @tf.function(jit_compile=True)
    def compiled_forward(x_input, indices_input):
      return model(x_input, indices_input)
    
    # This should succeed without Matrix size-incompatible error
    compiled_out = compiled_forward(x, indices)
    
    # Verify output shapes match between eager and compiled
    self.assertEqual(eager_out.shape[1], compiled_out.shape[1])

  def test_concat_preserves_dynamic_dimensions(self):
    """Direct test of concat with dynamic partition outputs."""
    
    @tf.function(jit_compile=True)
    def test_concat_dynamic():
      x = tf.random.normal([8, 32])
      partitions = tf.cast(tf.reduce_sum(x, axis=1) > 0, tf.int32)
      
      # Create dynamic partition outputs
      partitioned = tf.dynamic_partition(x, partitions, num_partitions=2)
      
      # Concat should preserve dynamic dimension
      result = tf.concat(partitioned, axis=0)
      
      return result, tf.shape(result)
    
    result, shape = test_concat_dynamic()
    
    # Should succeed without errors and have reasonable output shape
    self.assertEqual(len(result.shape), 2)
    self.assertEqual(result.shape[1], 32)
    self.assertGreater(shape[0], 0)  # Dynamic first dimension


if __name__ == '__main__':
  # Run with eager execution to verify test logic
  tf.config.run_functions_eagerly(False)
  unittest.main()
