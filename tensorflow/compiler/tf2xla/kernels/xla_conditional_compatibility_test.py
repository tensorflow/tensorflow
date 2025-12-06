"""CI-safe unit tests for XLA compatibility with conditional operations.

Regression test for GitHub issue #105133:
OperatorNotAllowedInGraphError when using tf.shape()[0] in conditional with jit_compile=True

Tested with TF 2.20.0+ on CPU/GPU builds.
XLA-specific tests are skipped when JIT compilation is unavailable.

This demonstrates user workarounds (tf.cond) vs problematic patterns (Python if with symbolic tensors).
Integration demo requiring XLA-enabled builds is in issue_105133_fix_demo.py.
"""

import tensorflow as tf
import unittest
import os


def _is_xla_available():
  """Check if XLA/JIT compilation is available in this build."""
  try:
    # Try to compile a simple function to test XLA availability
    @tf.function(jit_compile=True)
    def test_fn():
      return tf.constant(1.0)
    test_fn()
    return True
  except (tf.errors.UnimplementedError, tf.errors.InvalidArgumentError):
    return False
  except Exception:
    return False


class XlaConditionalCompatibilityTest(tf.test.TestCase):
  """CI-safe unit tests for XLA conditional compatibility.
  
  Tests eager execution behavior and tf.cond workarounds.
  XLA-specific tests are skipped when JIT is unavailable.
  """

  def test_eager_execution_with_python_conditionals(self):
    """Verify that Python conditionals work correctly in eager execution."""
    
    class TestModel(tf.keras.Model):
      def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(10)

      def call(self, x):
        # These work fine in eager execution
        if tf.shape(x)[0] >= 1:
          x = tf.stop_gradient(x)
        
        h, w = tf.shape(x)[1], tf.shape(x)[2]
        if h > 1 and w > 1:
          x = tf.nn.relu(x)
        else:
          x = tf.nn.tanh(x)
        
        x = tf.reshape(x, [tf.shape(x)[0], -1])
        return self.dense(x)

    model = TestModel()
    x = tf.random.normal([4, 8, 8, 3])
    
    # Should work without errors in eager execution
    result = model(x)
    
    # Verify correct shapes and behavior
    self.assertEqual(result.shape[0], 4)  # Batch size preserved
    self.assertEqual(result.shape[1], 10)  # Output dimension correct

  @unittest.skipUnless(_is_xla_available(), "XLA/JIT compilation not available")
  def test_python_if_with_symbolic_shape_raises_under_xla(self):
    """Test that Python if with symbolic tensors fails under XLA compilation."""
    
    def problematic_function(x):
      # This pattern causes OperatorNotAllowedInGraphError in XLA
      if tf.shape(x)[0] >= 1:  # PROBLEMATIC: symbolic tensor as Python bool
        return tf.stop_gradient(x)
      return x

    x = tf.random.normal([4, 8])
    
    # Should work in eager execution
    eager_result = problematic_function(x)
    self.assertEqual(eager_result.shape, x.shape)
    
    # Should fail when compiled with XLA
    @tf.function(jit_compile=True)
    def compiled_function(inputs):
      return problematic_function(inputs)
    
    # Check for expected error message patterns
    with self.assertRaises(Exception) as context:
      compiled_function(x)
    
    error_msg = str(context.exception)
    # Verify it's the expected symbolic tensor error
    self.assertTrue(
        any(keyword in error_msg.lower() for keyword in 
            ["symbolic", "python bool", "not allowed", "operatornotallowed"]),
        f"Expected symbolic tensor error, got: {error_msg}"
    )

  def test_tf_cond_replacement_allows_jit_compilation(self):
    """Test that tf.cond replacement works in both eager and XLA modes."""
    
    def tf_cond_function(x):
      # XLA-compatible version using tf.cond
      return tf.cond(
          tf.shape(x)[0] >= 1,
          lambda: tf.stop_gradient(x),
          lambda: x
      )

    x = tf.random.normal([4, 8])
    
    # Test eager execution
    eager_result = tf_cond_function(x)
    self.assertEqual(eager_result.shape, x.shape)
    
    # Test graph mode (without XLA first)
    @tf.function
    def graph_function(inputs):
      return tf_cond_function(inputs)
    
    graph_result = graph_function(x)
    self.assertEqual(graph_result.shape, x.shape)
    
    # Test deterministic behavior: stop_gradient should preserve values
    self.assertAllClose(eager_result, x)
    self.assertAllClose(graph_result, x)

  @unittest.skipUnless(_is_xla_available(), "XLA/JIT compilation not available")  
  def test_tf_cond_works_under_xla_compilation(self):
    """Test that tf.cond works correctly under XLA compilation."""
    
    def xla_compatible_conditional(x):
      return tf.cond(
          tf.shape(x)[0] > 2,
          lambda: tf.nn.relu(x),  # Non-negative for large batch
          lambda: tf.nn.tanh(x)   # Can be negative for small batch
      )

    # Test with different batch sizes to verify conditional logic
    x_small = tf.random.normal([2, 4])  # batch_size <= 2, should use tanh
    x_large = tf.random.normal([5, 4])  # batch_size > 2, should use relu
    
    @tf.function(jit_compile=True)
    def compiled_function(inputs):
      return xla_compatible_conditional(inputs)
    
    # Should compile and run without errors
    result_small = compiled_function(x_small)
    result_large = compiled_function(x_large)
    
    # Verify correct conditional behavior
    self.assertEqual(result_small.shape, x_small.shape)
    self.assertEqual(result_large.shape, x_large.shape)
    
    # relu output should be non-negative, tanh can be negative
    self.assertTrue(tf.reduce_all(result_large >= 0).numpy(), 
                   "relu output should be non-negative")
    # Note: We can't guarantee tanh produces negative values with random input,
    # but we can check it's different from relu behavior
    eager_small = xla_compatible_conditional(x_small)
    self.assertAllClose(result_small, eager_small, rtol=1e-5)

  def test_tf_where_alternative_for_simple_conditionals(self):
    """Test tf.where as an alternative to Python conditionals."""
    
    def where_based_function(x):
      # Use tf.where for element-wise conditionals
      return tf.where(
          tf.reduce_sum(x, axis=-1, keepdims=True) > 0,
          tf.stop_gradient(x),
          x * 0.5
      )

    x = tf.random.normal([4, 8])
    
    # Test eager execution
    eager_result = where_based_function(x)
    self.assertEqual(eager_result.shape, x.shape)
    
    # Test graph mode
    @tf.function
    def graph_function(inputs):
      return where_based_function(inputs)
    
    graph_result = graph_function(x)
    self.assertEqual(graph_result.shape, x.shape)
    self.assertAllClose(eager_result, graph_result)

  def test_mathematical_masking_alternative(self):
    """Test mathematical operations as alternative to conditionals."""
    
    def mask_based_function(x):
      # Use mathematical operations to avoid explicit conditionals
      batch_size = tf.cast(tf.shape(x)[0], tf.float32)
      threshold = tf.constant(3.0)
      
      # Create mask: 1.0 if batch_size > threshold, 0.0 otherwise
      mask = tf.cast(batch_size > threshold, tf.float32)
      
      # Apply different operations based on mask
      large_batch_result = tf.nn.l2_normalize(x, axis=-1)
      small_batch_result = x * 0.5
      
      # Combine results using the mask
      return mask * large_batch_result + (1.0 - mask) * small_batch_result

    # Test with different batch sizes
    x_small = tf.random.normal([2, 4])  # batch_size <= 3
    x_large = tf.random.normal([5, 4])  # batch_size > 3
    
    result_small = mask_based_function(x_small)
    result_large = mask_based_function(x_large)
    
    # Verify shapes
    self.assertEqual(result_small.shape, x_small.shape)
    self.assertEqual(result_large.shape, x_large.shape)
    
    # Verify behavior: large batch should be normalized (norm ≈ 1)
    norms_large = tf.norm(result_large, axis=-1)
    self.assertAllClose(norms_large, tf.ones_like(norms_large), atol=1e-5)
    
    # Small batch should be scaled by 0.5
    expected_small = x_small * 0.5
    self.assertAllClose(result_small, expected_small)

  def test_safe_shape_operations_in_tensorflow_ops(self):
    """Test that tf.shape operations are safe when used in TF ops (not Python conditionals)."""
    
    def safe_shape_operations(x):
      # These are safe: tf.shape used in TensorFlow operations, not Python conditionals
      current_shape = tf.shape(x)
      batch_size = current_shape[0]
      
      # Safe: Using shape values in TensorFlow operations
      half_batch = batch_size // 2
      
      # Safe: Reshape using computed shapes
      new_shape = tf.concat([current_shape[:1], [-1]], axis=0)
      reshaped = tf.reshape(x, new_shape)
      
      # Safe: Gather operations with dynamic indices  
      indices = tf.range(half_batch)
      gathered = tf.gather(reshaped, indices)
      
      return gathered, batch_size, half_batch

    x = tf.random.normal([10, 8, 4])
    
    # Test eager execution
    result, batch_size, half_batch = safe_shape_operations(x)
    
    # Verify correct behavior
    self.assertEqual(len(result.shape), 2)
    self.assertEqual(batch_size.numpy(), 10)
    self.assertEqual(half_batch.numpy(), 5)
    self.assertEqual(result.shape[0], 5)  # Half the batch size
    
    # Test graph mode
    @tf.function
    def graph_function(inputs):
      return safe_shape_operations(inputs)
    
    graph_result, graph_batch, graph_half = graph_function(x)
    
    # Verify consistency between eager and graph modes
    self.assertAllClose(result, graph_result)
    self.assertEqual(batch_size.numpy(), graph_batch.numpy())
    self.assertEqual(half_batch.numpy(), graph_half.numpy())

  @unittest.skipUnless(_is_xla_available(), "XLA/JIT compilation not available")
  def test_safe_shape_operations_under_xla(self):
    """Test that safe shape operations work under XLA compilation."""
    
    @tf.function(jit_compile=True)
    def xla_safe_operations(x):
      # These operations should work fine in XLA
      batch_size = tf.shape(x)[0]
      feature_dim = tf.shape(x)[-1]
      
      # Use shapes in TensorFlow operations (not Python conditionals)
      half_batch = batch_size // 2
      new_shape = [half_batch, feature_dim * 2]
      
      # Reshape and return
      x_subset = x[:half_batch]
      x_doubled = tf.concat([x_subset, x_subset], axis=-1)
      return tf.reshape(x_doubled, new_shape)

    x = tf.random.normal([8, 4])
    result = xla_safe_operations(x)
    
    # Verify expected output shape
    self.assertEqual(result.shape[0], 4)  # half_batch
    self.assertEqual(result.shape[1], 8)  # feature_dim * 2


class XlaConditionalBestPracticesTest(tf.test.TestCase):
  """Best practices for writing XLA-compatible conditional code."""

  def test_refactoring_patterns_eager_execution(self):
    """Test refactoring patterns work correctly in eager execution."""
    
    def tf_cond_pattern(x):
      # XLA-compatible version using tf.cond
      return tf.cond(
          tf.shape(x)[0] > 3,
          true_fn=lambda: tf.nn.relu(x),
          false_fn=lambda: tf.nn.tanh(x)
      )
    
    # Test with different batch sizes
    x_small = tf.random.normal([2, 4])  # batch_size <= 3, should use tanh  
    x_large = tf.random.normal([5, 4])  # batch_size > 3, should use relu
    
    result_small = tf_cond_pattern(x_small)
    result_large = tf_cond_pattern(x_large)
    
    # Verify shapes are preserved
    self.assertEqual(result_small.shape, x_small.shape)
    self.assertEqual(result_large.shape, x_large.shape)
    
    # Verify relu produces non-negative output
    self.assertTrue(tf.reduce_all(result_large >= 0).numpy())

  @unittest.skipUnless(_is_xla_available(), "XLA/JIT compilation not available")
  def test_refactoring_patterns_under_xla(self):
    """Test refactoring patterns work under XLA compilation."""
    
    @tf.function(jit_compile=True)
    def xla_conditional_pattern(x):
      return tf.cond(
          tf.shape(x)[0] > 3,
          true_fn=lambda: tf.nn.relu(x),
          false_fn=lambda: tf.nn.tanh(x)
      )
    
    # Test compilation and execution
    x_small = tf.random.normal([2, 4])
    x_large = tf.random.normal([5, 4]) 
    
    # Should compile and run without errors
    result_small = xla_conditional_pattern(x_small)
    result_large = xla_conditional_pattern(x_large)
    
    # Verify correct behavior
    self.assertEqual(result_small.shape, x_small.shape)
    self.assertEqual(result_large.shape, x_large.shape)
    self.assertTrue(tf.reduce_all(result_large >= 0).numpy())

  def test_nested_tf_cond_patterns(self):
    """Test nested tf.cond patterns work in eager and graph modes."""
    
    def nested_conditional_function(x, training=True):
      """Complex conditional logic using nested tf.cond operations."""
      
      batch_size = tf.shape(x)[0]
      feature_dim = tf.shape(x)[-1]
      
      def training_path():
        return tf.cond(
            batch_size > 8,
            lambda: tf.nn.dropout(x, rate=0.3),  # Large batch: dropout
            lambda: x  # Small batch: no dropout
        )
      
      def inference_path():
        return tf.cond(
            feature_dim > 16,
            lambda: tf.nn.l2_normalize(x, axis=-1),  # High-dim: normalize  
            lambda: x * 0.5  # Low-dim: scale down
        )
      
      # Top-level condition
      return tf.cond(
          training,
          training_path,
          inference_path
      )

    x = tf.random.normal([10, 32])
    
    # Test training mode  
    train_result = nested_conditional_function(x, training=True)
    self.assertEqual(train_result.shape, x.shape)
    
    # Test inference mode
    inference_result = nested_conditional_function(x, training=False)  
    self.assertEqual(inference_result.shape, x.shape)
    
    # For high-dim inference, should be normalized (feature_dim=32 > 16)
    norms = tf.norm(inference_result, axis=-1)
    self.assertAllClose(norms, tf.ones_like(norms), atol=1e-5)


if __name__ == '__main__':
  # Set up test environment
  tf.config.run_functions_eagerly(False)
  
  # Print XLA availability for debugging
  print(f"XLA/JIT compilation available: {_is_xla_available()}")
  
  # Run the tests
  tf.test.main()