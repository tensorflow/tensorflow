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
"""Tests for tensorflow.python.keras.engine.data_adapter."""

import tensorflow as tf

from tensorflow.python.keras.engine import data_adapter
from tensorflow.python.platform import test


class DataAdapterTest(test.TestCase):
  """Test class for data_adapter module functions."""

  def test_make_class_weight_map_fn_valid_class_weight(self):
    """Test valid integer and string keys in class_weight are processed correctly."""
    fn = data_adapter._make_class_weight_map_fn({'0': 0.5, 1: 1.5})
    self.assertTrue(callable(fn))  # Should return a function

  def test_make_class_weight_map_fn_invalid_key_type(self):
    """Test that float class_weight key raises ValueError."""
    with self.assertRaisesRegex(ValueError, "Invalid class_weight key.*0.5"):
      data_adapter._make_class_weight_map_fn({0.5: 1.0})

  def test_make_class_weight_map_fn_non_contiguous_keys(self):
    """Test that non-contiguous integer keys raise ValueError."""
    with self.assertRaisesRegex(ValueError,
                                "Expected `class_weight` to be a dict.*"):
      data_adapter._make_class_weight_map_fn({0: 0.3, 2: 0.7})

  def test_make_class_weight_map_fn_complex_weights_rejected(self):
    """Test that complex class weights are explicitly rejected."""
    with self.assertRaisesRegex(
        ValueError, 
        "Complex class weights are not supported"):
      data_adapter._make_class_weight_map_fn({
          0: complex(1.0, 0.0),
          1: complex(2.0, 0.0)
      })
    
    # Also test with TensorFlow complex types
    with self.assertRaisesRegex(
        ValueError,
        "Complex class weights are not supported"):
      data_adapter._make_class_weight_map_fn({
          0: tf.complex64(1.0 + 0j),
          1: tf.complex64(2.0 + 0j)
      })

  def test_make_class_weight_map_fn_valid_contiguous_keys(self):
    """Test that valid contiguous keys work correctly."""
    # Test with integer keys
    fn = data_adapter._make_class_weight_map_fn({0: 1.0, 1: 2.0, 2: 0.5})
    self.assertTrue(callable(fn))
    
    # Test with string keys that convert to integers
    fn = data_adapter._make_class_weight_map_fn({'0': 1.0, '1': 2.0})
    self.assertTrue(callable(fn))

  def test_make_class_weight_map_fn_invalid_string_keys(self):
    """Test that invalid string keys raise ValueError."""
    with self.assertRaisesRegex(ValueError, "Invalid class_weight key"):
      data_adapter._make_class_weight_map_fn({'invalid': 1.0, '1': 2.0})

  def test_make_class_weight_map_fn_mixed_valid_keys(self):
    """Test that mixing valid integer and string keys works."""
    fn = data_adapter._make_class_weight_map_fn({0: 1.0, '1': 2.0, 2: 0.5})
    self.assertTrue(callable(fn))

  def test_make_class_weight_map_fn_empty_dict(self):
    """Test that empty class_weight dict raises appropriate error."""
    with self.assertRaises(ValueError):
      data_adapter._make_class_weight_map_fn({})

  def test_make_class_weight_map_fn_single_class(self):
    """Test that single class weight works correctly."""
    fn = data_adapter._make_class_weight_map_fn({0: 1.5})
    self.assertTrue(callable(fn))

  def test_make_class_weight_map_fn_none_values(self):
    """Test that None values in class_weight raise appropriate error."""
    with self.assertRaises((ValueError, TypeError)):
      data_adapter._make_class_weight_map_fn({0: None, 1: 2.0})

  def test_make_class_weight_map_fn_negative_weights(self):
    """Test that negative weights are handled (should work for mathematical validity)."""
    fn = data_adapter._make_class_weight_map_fn({0: -1.0, 1: 2.0})
    self.assertTrue(callable(fn))

  def test_make_class_weight_map_fn_zero_weights(self):
    """Test that zero weights work correctly."""
    fn = data_adapter._make_class_weight_map_fn({0: 0.0, 1: 1.0})
    self.assertTrue(callable(fn))


if __name__ == '__main__':
  tf.test.main()
