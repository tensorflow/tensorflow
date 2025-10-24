# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for training_utils_v1."""

from tensorflow.python.keras.engine import training_utils_v1
from tensorflow.python.platform import test

class TrainingUtilsV1Test(test.TestCase):

  def test_invalid_string_class_weight_keys(self):
    """Test that invalid string keys in class_weight raise ValueError."""
    with self.assertRaisesRegex(ValueError,
                                "Invalid class_weight key.*invalid_key"):
      training_utils_v1.standardize_sample_or_class_weights(
          {'invalid_key': 1.0}, ['output'], 'class_weight')

  def test_invalid_float_class_weight_keys(self):
    """Test that float keys in class_weight raise ValueError."""
    with self.assertRaisesRegex(ValueError, "Invalid class_weight key.*0.5"):
      training_utils_v1.standardize_sample_or_class_weights(
          {0.5: 1.0}, ['output'], 'class_weight')

  def test_valid_integer_class_weight_keys(self):
    """Test that valid integer keys work correctly."""
    result = training_utils_v1.standardize_sample_or_class_weights(
        {0: 1.0}, ['output'], 'class_weight')
    self.assertIsNotNone(result)

  def test_valid_string_number_class_weight_keys(self):
    """Test that valid string number keys work correctly."""
    result = training_utils_v1.standardize_sample_or_class_weights(
        {'0': 1.0}, ['output'], 'class_weight')
    self.assertIsNotNone(result)

  def test_valid_output_name_class_weight_keys(self):
    """Test that valid output names work correctly."""
    result = training_utils_v1.standardize_sample_or_class_weights(
        {'output': 1.0}, ['output'], 'class_weight')
    self.assertEqual(result, [1.0])

  def test_sample_weight_not_affected(self):
    """Test that sample_weight validation is unaffected."""
    result = training_utils_v1.standardize_sample_or_class_weights(
        {'output': [1.0, 2.0]}, ['output'], 'sample_weight')
    self.assertEqual(result, [[1.0, 2.0]])

  def test_multi_output_invalid_class_weight_keys(self):
    """Test invalid keys in multi-output class_weight."""
    class_weight = {
        'output1': {'invalid_key': 1.0, 0: 2.0},
        'output2': {0: 1.0, 1: 2.0}
    }
    with self.assertRaisesRegex(ValueError,
                                "Invalid class_weight key.*invalid_key.*"
                                "output1"):
      training_utils_v1.standardize_sample_or_class_weights(
          class_weight, ['output1', 'output2'], 'class_weight')


if __name__ == '__main__':
  test.main()
