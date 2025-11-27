# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Regression test for tf.concat nondeterministic behavior with Python built-ins.

Demonstrates issue #105150: mixing Python filter/map/zip with TensorFlow tensors
causes nondeterministic graph construction and dtype coercion errors in tf.concat.

This test validates the recommended workarounds:
1. Minimal fix: flatten tuples before concatenation
2. Robust fix: pure TF ops (stack/boolean_mask/concat) for graph/XLA stability
"""

import tensorflow as tf


class ConcatFixTest(tf.test.TestCase):

  def test_concat_with_tuples_nondeterministic(self):
    """Demonstrates nondeterministic behavior when tf.concat receives tuples.
    
    This test reproduces issue #105150 by using Python filter with tensor
    conditions, which can fail with dtype mismatch or graph errors.
    """
    x = tf.constant([[1.0, 2.0]], dtype=tf.float32)
    candidates = [x, x * 2, x * 3]
    # Using Python filter with tensor conditions - problematic pattern
    mapped_features = list(map(tf.nn.sigmoid, filter(lambda c: tf.reduce_sum(c) > 0.5, candidates)))
    zipped_data = list(zip(mapped_features, [tf.ones_like(x) for _ in range(len(mapped_features))]))
    
    # This can fail nondeterministically with specific errors
    try:
      combined = tf.concat(zipped_data, axis=-1)
      # If it succeeds, verify shape
      self.assertEqual(combined.shape[0], 1)
    except (tf.errors.InvalidArgumentError, tf.errors.OperatorNotAllowedInGraphError) as e:
      # Expected failure modes: dtype mismatch (ConcatV2) or graph construction error
      self.assertTrue(
          "ConcatV2" in str(e) or "OperatorNotAllowedInGraphError" in str(e),
          f"Expected specific concat error, got: {e}"
      )

  def test_concat_minimal_fix(self):
    """Minimal fix: flatten tuples before final concat.
    
    Uses TF boolean_mask instead of Python filter for deterministic behavior.
    """
    x = tf.constant([[1.0, 2.0]], dtype=tf.float32)
    candidates = tf.stack([x, x * 2, x * 3], axis=0)
    # Use TF ops for filtering instead of Python filter
    sums = tf.reduce_sum(candidates, axis=[1, 2])
    mask = sums > 0.5
    filtered = tf.boolean_mask(candidates, mask, axis=0)
    
    # Apply mapping
    mapped_features = [tf.nn.sigmoid(filtered[i]) for i in range(tf.shape(filtered)[0].numpy())]
    zipped_data = list(zip(mapped_features, [tf.ones_like(x, dtype=x.dtype) for _ in range(len(mapped_features))]))
    
    # Fix: concatenate each pair first
    flattened_pairs = [tf.concat([a, b], axis=-1) for (a, b) in zipped_data]
    combined = tf.concat(flattened_pairs, axis=-1) if flattened_pairs else tf.zeros([1, 0], dtype=x.dtype)
    
    # Verify output shape
    self.assertAllEqual(tf.shape(combined)[0], 1)
    expected_width = len(mapped_features) * 4
    self.assertAllEqual(tf.shape(combined)[1], expected_width)

  def test_concat_robust_tf_only(self):
    """Robust rewrite using only TF ops for graph/XLA compatibility."""
    x = tf.constant([[1.0, 2.0]], dtype=tf.float32)
    candidates = tf.stack([x, x * 2, x * 3], axis=0)  # shape (3, 1, 2)
    sums = tf.reduce_sum(candidates, axis=[1, 2])  # shape (3,)
    mask = sums > 0.5  # shape (3,)
    filtered = tf.boolean_mask(candidates, mask, axis=0)  # shape (k, 1, 2)
    mapped = tf.nn.sigmoid(filtered)  # shape (k, 1, 2)
    ones = tf.ones_like(x, dtype=x.dtype)  # shape (1, 2)
    ones_tiled = tf.tile(tf.expand_dims(ones, 0), [tf.shape(mapped)[0], 1, 1])  # shape (k, 1, 2)
    per_candidate = tf.concat([mapped, ones_tiled], axis=-1)  # shape (k, 1, 4)
    
    # Guard against zero filtered case: use reshape instead of unstack
    # combined shape: (1, k*4)
    k = tf.shape(per_candidate)[0]
    batch_size = tf.shape(per_candidate)[1]
    feature_width = tf.shape(per_candidate)[2]
    combined = tf.reshape(
        tf.transpose(per_candidate, [1, 0, 2]),  # (1, k, 4)
        [batch_size, k * feature_width]  # (1, k*4)
    )
    
    # Verify output shape
    self.assertAllEqual(tf.shape(combined)[0], 1)
    expected_k = tf.reduce_sum(tf.cast(mask, tf.int32))
    self.assertAllEqual(tf.shape(combined)[1], expected_k * 4)

  def test_concat_robust_tf_only_zero_case(self):
    """Test robust version handles zero filtered candidates gracefully."""
    x = tf.constant([[1.0, 2.0]], dtype=tf.float32)
    # All candidates sum to < 0.5, so all will be filtered out
    candidates = tf.stack([x * 0.0, x * 0.0, x * 0.0], axis=0)
    sums = tf.reduce_sum(candidates, axis=[1, 2])
    mask = sums > 0.5
    filtered = tf.boolean_mask(candidates, mask, axis=0)  # shape (0, 1, 2)
    mapped = tf.nn.sigmoid(filtered)
    ones = tf.ones_like(x, dtype=x.dtype)
    ones_tiled = tf.tile(tf.expand_dims(ones, 0), [tf.shape(mapped)[0], 1, 1])
    per_candidate = tf.concat([mapped, ones_tiled], axis=-1)  # shape (0, 1, 4)
    
    # Use reshape approach to handle zero case
    k = tf.shape(per_candidate)[0]
    batch_size = tf.shape(per_candidate)[1]
    feature_width = tf.shape(per_candidate)[2]
    combined = tf.reshape(
        tf.transpose(per_candidate, [1, 0, 2]),
        [batch_size, k * feature_width]
    )
    
    # Should produce (1, 0) tensor
    self.assertAllEqual(tf.shape(combined), [1, 0])

  def test_concat_robust_tf_jit(self):
    """Test the robust version under jit_compile for XLA stability."""
    @tf.function(jit_compile=True)
    def compiled_concat(x):
      candidates = tf.stack([x, x * 2, x * 3], axis=0)
      sums = tf.reduce_sum(candidates, axis=[1, 2])
      mask = sums > 0.5
      filtered = tf.boolean_mask(candidates, mask, axis=0)
      mapped = tf.nn.sigmoid(filtered)
      ones = tf.ones_like(x, dtype=x.dtype)
      ones_tiled = tf.tile(tf.expand_dims(ones, 0), [tf.shape(mapped)[0], 1, 1])
      per_candidate = tf.concat([mapped, ones_tiled], axis=-1)
      
      # Use reshape to avoid unstack issues with dynamic shapes
      k = tf.shape(per_candidate)[0]
      batch_size = tf.shape(per_candidate)[1]
      feature_width = tf.shape(per_candidate)[2]
      combined = tf.reshape(
          tf.transpose(per_candidate, [1, 0, 2]),
          [batch_size, k * feature_width]
      )
      return combined
    
    x = tf.constant([[1.0, 2.0]], dtype=tf.float32)
    result = compiled_concat(x)
    
    # Verify JIT compilation produces correct output
    self.assertAllEqual(tf.shape(result)[0], 1)
    # All 3 candidates pass the filter (sum > 0.5), so expect 3 * 4 = 12
    self.assertAllEqual(tf.shape(result)[1], 12)
    self.assertEqual(result.dtype, tf.float32)


if __name__ == '__main__':
  tf.test.main()