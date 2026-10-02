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
"""Tests for advanced activation layers."""

from tensorflow.python.keras.layers import advanced_activations
from tensorflow.python.platform import test


class LeakyReLUTest(test.TestCase):
  """Tests for the LeakyReLU layer."""

  def test_alpha_validation_none(self):
    """Test that alpha=None raises ValueError."""
    with self.assertRaisesRegex(
        ValueError, 'The alpha value of a Leaky ReLU layer cannot be None'
    ):
      advanced_activations.LeakyReLU(alpha=None)

  def test_alpha_validation_negative(self):
    """Test that negative alpha raises ValueError."""
    with self.assertRaisesRegex(
        ValueError, 'The alpha value of a Leaky ReLU layer should be >= 0'
    ):
      advanced_activations.LeakyReLU(alpha=-0.5)

  def test_alpha_validation_zero(self):
    """Test that alpha=0 is allowed (boundary case)."""
    layer = advanced_activations.LeakyReLU(alpha=0.0)
    self.assertEqual(float(layer.alpha), 0.0)

  def test_alpha_validation_positive(self):
    """Test that positive alpha works normally."""
    layer = advanced_activations.LeakyReLU(alpha=0.3)
    self.assertAlmostEqual(float(layer.alpha), 0.3)


if __name__ == '__main__':
  test.main()
