# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for Keras losses validation."""

import numpy as np

from tensorflow.python.keras import losses
from tensorflow.python.platform import test as test_lib


class BinaryCrossentropyValidationTest(test_lib.TestCase):

  def testInvalidLabelSmoothingClass(self):
    with self.assertRaisesRegex(ValueError, r'label_smoothing.*must be in'):
      losses.BinaryCrossentropy(label_smoothing=-0.1)
    with self.assertRaisesRegex(ValueError, r'label_smoothing.*must be in'):
      losses.BinaryCrossentropy(label_smoothing=1.1)

  def testValidLabelSmoothingClass(self):
    losses.BinaryCrossentropy(label_smoothing=0.0)
    losses.BinaryCrossentropy(label_smoothing=0.5)
    losses.BinaryCrossentropy(label_smoothing=1.0)

  def testInvalidLabelSmoothingFunctional(self):
    y_true = [[0, 1], [0, 0]]
    y_pred = [[0.6, 0.4], [0.4, 0.6]]
    with self.assertRaisesRegex(ValueError, r'label_smoothing.*must be in'):
      losses.binary_crossentropy(y_true, y_pred, label_smoothing=-0.1)
    with self.assertRaisesRegex(ValueError, r'label_smoothing.*must be in'):
      losses.binary_crossentropy(y_true, y_pred, label_smoothing=1.1)

  def testValidLabelSmoothingFunctional(self):
    y_true = [[0, 1], [0, 0]]
    y_pred = [[0.6, 0.4], [0.4, 0.6]]
    losses.binary_crossentropy(y_true, y_pred, label_smoothing=0.0)
    losses.binary_crossentropy(y_true, y_pred, label_smoothing=0.5)
    losses.binary_crossentropy(y_true, y_pred, label_smoothing=1.0)


class CategoricalCrossentropyAxisTest(test_lib.TestCase):
  """Tests for categorical_crossentropy with non-default axis."""

  def testLabelSmoothingWithAxis(self):
    """Verifies num_classes uses axis, not [-1], for label smoothing."""
    # Shape [num_classes, batch_size] = [3, 2] with axis=0
    y_true = np.array([[1.0, 0.0],
                       [0.0, 1.0],
                       [0.0, 0.0]], dtype=np.float32)
    y_pred = np.array([[0.8, 0.1],
                       [0.1, 0.8],
                       [0.1, 0.1]], dtype=np.float32)
    label_smoothing = 0.1

    # Expected smoothed y_true when num_classes=3 (correct, using axis=0):
    expected_y_true_correct = y_true * 0.9 + 0.1 / 3.0

    # Expected smoothed y_true when num_classes=2 (buggy, using [-1]):
    expected_y_true_buggy = y_true * 0.9 + 0.1 / 2.0

    # Loss with axis=0 and label_smoothing
    loss_axis_0 = losses.categorical_crossentropy(
        y_true, y_pred, axis=0, label_smoothing=label_smoothing)

    # Manual computation with correct smoothing (div by num_classes=3)
    correct_loss = losses.categorical_crossentropy(
        expected_y_true_correct, y_pred, axis=0, label_smoothing=0.0)

    # Manual computation with buggy smoothing (div by batch_size=2)
    buggy_loss = losses.categorical_crossentropy(
        expected_y_true_buggy, y_pred, axis=0, label_smoothing=0.0)

    # Verify the loss matches the correct (not buggy) computation
    self.assertAllClose(loss_axis_0, correct_loss)
    self.assertNotAllClose(loss_axis_0, buggy_loss)


if __name__ == '__main__':
  test_lib.main()
