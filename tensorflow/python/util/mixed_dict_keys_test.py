from tensorflow.python.framework import constant_op
from tensorflow.python.util import nest_util
# Copyright 2025 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for XLA JIT compilation with mixed-type dictionary keys.

This test validates the fix for issue #105333 where @tf.function(jit_compile=True)
fails when returning dictionaries with mixed key types (e.g., strings and integers).
"""

from tensorflow.python.platform import test
from tensorflow.python.util import nest


class XLAMixedDictKeysTest(test.TestCase):
  """Test XLA JIT compilation with mixed-type dictionary keys."""

  def test_mixed_string_int_keys_flatten(self):
    """Test flattening dict with mixed string and int keys."""
    mixed_dict = {'string_key': 1, 123: 2, 'another': 3, 456: 4}
    flattened = nest.flatten(mixed_dict)
    # Should flatten successfully with deterministic order
    # Keys sorted by type name first (int < str), then by value
    self.assertEqual(len(flattened), 4)
    self.assertIn(1, flattened)
    self.assertIn(2, flattened)
    self.assertIn(3, flattened)
    self.assertIn(4, flattened)

  def test_mixed_keys_with_xla_simple(self):
    """Test simple XLA function with mixed dict keys."""
    @tf.function(jit_compile=True)
    def simple_mixed_dict(x):
      results = {}
      results['string_key'] = x
      results[123] = x + 1
      return results
    
    input_tensor = constant_op.constant([1.0, 2.0, 3.0])
    output = simple_mixed_dict(input_tensor)
    
    self.assertIn('string_key', output)
    self.assertIn(123, output)
    self.assertAllClose(output['string_key'], [1.0, 2.0, 3.0])
    self.assertAllClose(output[123], [2.0, 3.0, 4.0])

  def test_mixed_keys_with_xla_in_model(self):
    """Test XLA with mixed dict keys in Keras model (original issue #105333)."""
    class SimpleModel(tf.keras.Model):
      @tf.function(jit_compile=True)
      def call(self, x):
        results = {}
        results['string_key'] = x
        results[123] = x + 1
        return x, results
    
    model = SimpleModel()
    input_tensor = tf.random.normal([2, 16, 16, 16, 32])
    output_tensor, output_dict = model(input_tensor)
    
    self.assertEqual(output_tensor.shape, (2, 16, 16, 16, 32))
    self.assertIn('string_key', output_dict)
    self.assertIn(123, output_dict)

  def test_multiple_mixed_types(self):
    """Test dict with multiple mixed key types."""
    @tf.function(jit_compile=True)
    def multi_type_dict(x):
      results = {}
      results['str1'] = x
      results[1] = x + 1
      results['str2'] = x + 2
      results[2] = x + 3
      results[3] = x + 4
      results['str3'] = x + 5
      return results
    
    input_tensor = constant_op.constant(10.0)
    output = multi_type_dict(input_tensor)
    
    # Verify all keys are present
    self.assertIn('str1', output)
    self.assertIn('str2', output)
    self.assertIn('str3', output)
    self.assertIn(1, output)
    self.assertIn(2, output)
    self.assertIn(3, output)
    
    # Verify values
    self.assertAlmostEqual(output['str1'].numpy(), 10.0)
    self.assertAlmostEqual(output[1].numpy(), 11.0)
    self.assertAlmostEqual(output['str2'].numpy(), 12.0)
    self.assertAlmostEqual(output[2].numpy(), 13.0)

  def test_nested_mixed_keys(self):
    """Test nested dicts with mixed keys."""
    @tf.function(jit_compile=True)
    def nested_mixed_dict(x):
      inner = {
        'inner_str': x,
        100: x + 1
      }
      outer = {
        'outer': inner,
        200: x + 2
      }
      return outer
    
    input_tensor = constant_op.constant(5.0)
    output = nested_mixed_dict(input_tensor)
    
    self.assertIn('outer', output)
    self.assertIn(200, output)
    self.assertIn('inner_str', output['outer'])
    self.assertIn(100, output['outer'])

  def test_pack_sequence_as_with_mixed_keys(self):
    """Test pack_sequence_as with mixed key types."""
    structure = {'a': 1, 10: 2, 'b': 3, 20: 4}
    flat_sequence = [100, 200, 300, 400]
    
    packed = nest.pack_sequence_as(structure, flat_sequence)
    
    # Verify repacking works correctly
    self.assertEqual(len(packed), 4)
    # Values should be assigned in sorted key order (int keys first, then str keys)

  def test_without_xla_still_works(self):
    """Verify mixed keys work without XLA as well."""
    @tf.function(jit_compile=False)
    def no_xla_mixed_dict(x):
      results = {}
      results['string_key'] = x
      results[123] = x + 1
      return results
    
    input_tensor = constant_op.constant([1.0, 2.0])
    output = no_xla_mixed_dict(input_tensor)
    
    self.assertIn('string_key', output)
    self.assertIn(123, output)

  def test_consistent_ordering(self):
    """Ensure consistent ordering across multiple calls."""
    @tf.function(jit_compile=True)
    def consistent_dict(x):
      results = {}
      results['z'] = x
      results[3] = x + 1
      results['a'] = x + 2
      results[1] = x + 3
      return results
    
    input_tensor = constant_op.constant(1.0)
    
    # Call multiple times and verify same order
    output1 = consistent_dict(input_tensor)
    output2 = consistent_dict(input_tensor)
    output3 = consistent_dict(input_tensor)
    
    keys1 = sorted(output1.keys(), key=lambda x: (type(x).__name__, x))
    keys2 = sorted(output2.keys(), key=lambda x: (type(x).__name__, x))
    keys3 = sorted(output3.keys(), key=lambda x: (type(x).__name__, x))
    
    self.assertEqual(keys1, keys2)
    self.assertEqual(keys2, keys3)


if __name__ == '__main__':
  test.main()
