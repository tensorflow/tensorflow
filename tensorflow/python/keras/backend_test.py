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
"""Tests for Keras backend."""

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.keras import backend
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class BackendTest(test.TestCase):

  def test_categorical_crossentropy_zero_outputs(self):
    # Create a true label tensor and an all-zero prediction tensor
    y_true = constant_op.constant([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    y_pred = constant_op.constant([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

    # Calculate categorical crossentropy
    loss = backend.categorical_crossentropy(y_true, y_pred)

    # Verify that the loss is not NaN
    self.assertFalse(self.evaluate(math_ops.reduce_any(math_ops.is_nan(loss))))

    # Optionally verify the output matches expectation
    # With zero predictions, they sum to 0.
    # The code adds epsilon and clips to avoid division by zero and log(0).
    # The result should be a valid number, not NaN.
    loss_val = self.evaluate(loss)
    self.assertFalse(np.isnan(loss_val[0]))
    self.assertFalse(np.isnan(loss_val[1]))


if __name__ == '__main__':
  test.main()
