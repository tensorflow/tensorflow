# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

import numpy as np
import tensorflow as tf


class TopKTest(tf.test.TestCase):

  def _validateTopK(
      self, inputs, k, expected_values, expected_indices, sorted=True):
    np_values = np.array(expected_values)
    np_indices = np.array(expected_indices)
    with self.test_session():
      values_op, indices_op = tf.nn.top_k(inputs, k, sorted=sorted)
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

  def testTop3Unsorted(self):
    inputs = [[0.1, 0.3, 0.2, 0.4], [0.1, 0.3, 0.3, 0.2]]
    self._validateTopK(inputs, 3,
                       [[0.2, 0.3, 0.4], [0.2, 0.3, 0.3]],
                       [[2, 1, 3], [3, 1, 2]], sorted=False)

  def testTop3Vector(self):
    inputs = [3, 6, 15, 18, 6, 12, 1, 17, 3, 0, 4, 19, 1, 6]
    self._validateTopK(inputs, 3, [19, 18, 17], [11, 3, 7])

  def testTensorK(self):
    inputs = [3, 6, 15, 18, 6, 12, 1, 17, 3, 0, 4, 19, 1, 6]
    k = tf.constant(3)
    self._validateTopK(inputs, k, [19, 18, 17], [11, 3, 7])

  def testKNegative(self):
    inputs = [[0.1, 0.2], [0.3, 0.4]]
    with self.test_session():
      k = tf.placeholder(tf.int32)
      values, _ = tf.nn.top_k(inputs, k)
      with self.assertRaisesOpError("Need k >= 0, got -7"):
        values.eval(feed_dict={k: -7})

  def testKTooLarge(self):
    inputs = [[0.1, 0.2], [0.3, 0.4]]
    with self.assertRaisesRegexp(
        ValueError, r"input.shape \(2, 2\) must have last dimension >= k = 4"):
      tf.nn.top_k(inputs, 4)

  def testTopKGradients(self):
    with self.test_session() as sess:
      inputs = tf.placeholder(tf.int32, shape=[2, 5])
      values, _ = tf.nn.top_k(inputs, 3)
      grad = sess.run(
          tf.gradients(values,
                       inputs,
                       grad_ys=[[[1, 2, 3], [4, 5, 6]]]),
          feed_dict={inputs: [[2, -1, 1000, 3, 4], [1, 5, 2, 4, 3]]})[0]
    self.assertEqual(grad.tolist(), [[0, 0, 1, 3, 2], [0, 4, 0, 5, 6]])


if __name__ == "__main__":
  tf.test.main()
