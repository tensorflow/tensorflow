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

from tensorflow.python.framework import constant_op
from tensorflow.python.keras import losses
from tensorflow.python.platform import test as test_lib


class CategoricalCrossentropyValidationTest(test_lib.TestCase):

  def testValidLabelSmoothing(self):
    y_true = np.array([[0, 1, 0], [0, 0, 1]], dtype=np.float32)
    y_pred = np.array([[0.05, 0.95, 0], [0.1, 0.8, 0.1]], dtype=np.float32)

    # Valid values should not raise
    losses.categorical_crossentropy(y_true, y_pred, label_smoothing=0.0)
    losses.categorical_crossentropy(y_true, y_pred, label_smoothing=0.5)
    losses.categorical_crossentropy(y_true, y_pred, label_smoothing=1.0)
    losses.categorical_crossentropy(
        y_true, y_pred, label_smoothing=constant_op.constant(0.5))

  def testInvalidLabelSmoothingRaisesValueError(self):
    y_true = np.array([[0, 1, 0], [0, 0, 1]], dtype=np.float32)
    y_pred = np.array([[0.05, 0.95, 0], [0.1, 0.8, 0.1]], dtype=np.float32)

    with self.assertRaises(ValueError):
      losses.categorical_crossentropy(
          y_true, y_pred, label_smoothing=-0.1)

    with self.assertRaises(ValueError):
      losses.categorical_crossentropy(
          y_true, y_pred, label_smoothing=1.1)

  def testTensorLabelSmoothingSkipped(self):
    y_true = np.array([[0., 1., 0.], [0., 0., 1.]], dtype=np.float32)
    y_pred = np.array([[0.05, 0.95, 0.], [0.1, 0.8, 0.1]], dtype=np.float32)
    # Tensor label_smoothing bypasses static validation
    label_smoothing = constant_op.constant(1.5)
    loss = losses.categorical_crossentropy(
        y_true, y_pred, label_smoothing=label_smoothing)
    self.assertIsNotNone(loss)


class BinaryCrossentropyValidationTest(test_lib.TestCase):

  def testValidLabelSmoothing(self):
    y_true = np.array([[0, 1], [0, 0]], dtype=np.float32)
    y_pred = np.array([[0.6, 0.4], [0.4, 0.6]], dtype=np.float32)

    # Valid values should not raise
    losses.binary_crossentropy(y_true, y_pred, label_smoothing=0.0)
    losses.binary_crossentropy(y_true, y_pred, label_smoothing=0.5)
    losses.binary_crossentropy(y_true, y_pred, label_smoothing=1.0)
    losses.binary_crossentropy(
        y_true, y_pred, label_smoothing=constant_op.constant(0.5))

  def testInvalidLabelSmoothingRaisesValueError(self):
    y_true = np.array([[0, 1], [0, 0]], dtype=np.float32)
    y_pred = np.array([[0.6, 0.4], [0.4, 0.6]], dtype=np.float32)

    with self.assertRaises(ValueError):
      losses.binary_crossentropy(
          y_true, y_pred, label_smoothing=-0.1)

    with self.assertRaises(ValueError):
      losses.binary_crossentropy(
          y_true, y_pred, label_smoothing=1.1)

  def testTensorLabelSmoothingSkipped(self):
    y_true = np.array([[0., 1.], [0., 0.]], dtype=np.float32)
    y_pred = np.array([[0.6, 0.4], [0.4, 0.6]], dtype=np.float32)
    # Tensor label_smoothing bypasses static validation
    label_smoothing = constant_op.constant(1.5)
    loss = losses.binary_crossentropy(
        y_true, y_pred, label_smoothing=label_smoothing)
    self.assertIsNotNone(loss)


if __name__ == '__main__':
  test_lib.main()
