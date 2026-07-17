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


if __name__ == '__main__':
  test_lib.main()
