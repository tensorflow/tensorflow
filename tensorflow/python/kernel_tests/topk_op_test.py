# Copyright 2015 Google Inc. All Rights Reserved.
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

"""Tests for TopK op."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.python.platform

import numpy as np
import tensorflow as tf


class TopKTest(tf.test.TestCase):

  def _validateTopK(self, inputs, k, expected_values, expected_indices):
    np_values = np.array(expected_values)
    np_indices = np.array(expected_indices)
    with self.test_session():
      values_op, indices_op = tf.nn.top_k(inputs, k)
      values = values_op.eval()
      indices = indices_op.eval()
      self.assertAllClose(np_values, values)
      self.assertAllEqual(np_indices, indices)
      self.assertShapeEqual(np_values, values_op)
      self.assertShapeEqual(np_indices, indices_op)

  def testTop1(self):
    inputs = [[0.1, 0.3, 0.2, 0.4], [0.1, 0.3, 0.3, 0.2]]
    self._validateTopK(inputs, 1,
                       [[0.4], [0.3]],
                       [[3], [1]])

  def testTop2(self):
    inputs = [[0.1, 0.3, 0.2, 0.4], [0.1, 0.3, 0.3, 0.2]]
    self._validateTopK(inputs, 2,
                       [[0.4, 0.3], [0.3, 0.3]],
                       [[3, 1], [1, 2]])

  def testTopAll(self):
    inputs = [[0.1, 0.3, 0.2, 0.4], [0.1, 0.3, 0.3, 0.2]]
    self._validateTopK(inputs, 4,
                       [[0.4, 0.3, 0.2, 0.1], [0.3, 0.3, 0.2, 0.1]],
                       [[3, 1, 2, 0], [1, 2, 3, 0]])

  def testKNegative(self):
    inputs = [[0.1, 0.2], [0.3, 0.4]]
    with self.assertRaisesRegexp(ValueError, "less than minimum 1"):
      tf.nn.top_k(inputs, -1)

  def testKTooLarge(self):
    inputs = [[0.1, 0.2], [0.3, 0.4]]
    with self.assertRaisesRegexp(ValueError, "input must have at least k"):
      tf.nn.top_k(inputs, 4)


if __name__ == "__main__":
  tf.test.main()
