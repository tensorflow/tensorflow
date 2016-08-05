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

"""Tests for tensorflow.ops.nn_ops.Pad."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


class PadOpTest(tf.test.TestCase):

  def _npPad(self, inp, paddings, mode):
    return np.pad(inp, paddings, mode=mode.lower())

  def testNpPad(self):
    self.assertAllEqual(
        np.array([[0, 0, 0, 0, 0, 0],
                  [0, 3, 3, 0, 0, 0],
                  [0, 4, 4, 0, 0, 0],
                  [0, 5, 5, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0]]),
        self._npPad(
            np.array([[3, 3], [4, 4], [5, 5]]),
            [[1, 2], [1, 3]],
            mode="constant"))

    self.assertAllEqual(
        np.array([[4, 3, 4, 9, 4, 3],
                  [1, 0, 1, 2, 1, 0],
                  [4, 3, 4, 9, 4, 3],
                  [1, 0, 1, 2, 1, 0]]),
        self._npPad(
            np.array([[0, 1, 2], [3, 4, 9]]),
            [[1, 1], [1, 2]],
            mode="reflect"))

    self.assertAllEqual(
        np.array([[0, 0, 1, 2, 2, 1],
                  [0, 0, 1, 2, 2, 1],
                  [3, 3, 4, 9, 9, 4],
                  [3, 3, 4, 9, 9, 4]]),
        self._npPad(
            np.array([[0, 1, 2], [3, 4, 9]]),
            [[1, 1], [1, 2]],
            mode="symmetric"))

  def _testPad(self, np_inputs, paddings, mode):
    np_val = self._npPad(np_inputs, paddings, mode=mode)
    with self.test_session():
      tf_val = tf.pad(np_inputs, paddings, mode=mode)
      out = tf_val.eval()
    self.assertAllEqual(np_val, out)
    self.assertShapeEqual(np_val, tf_val)

  def _testGradient(self, x, a, mode):
    with self.test_session():
      inx = tf.convert_to_tensor(x)
      xs = list(x.shape)
      ina = tf.convert_to_tensor(a)
      y = tf.pad(inx, ina, mode=mode)
      # Expected y's shape to be:
      ys = list(np.array(x.shape) + np.sum(np.array(a), axis=1))
      jacob_t, jacob_n = tf.test.compute_gradient(inx,
                                                  xs,
                                                  y,
                                                  ys,
                                                  x_init_value=x)
    self.assertAllClose(jacob_t, jacob_n, rtol=1e-5, atol=1e-5)

  def _testAll(self, np_inputs, paddings):
    for mode in ("CONSTANT", "REFLECT", "SYMMETRIC"):
      self._testPad(np_inputs, paddings, mode=mode)
      self._testPad(np_inputs, paddings, mode=mode)
      if np_inputs.dtype == np.float32:
        self._testGradient(np_inputs, paddings, mode=mode)

  def testInputDims(self):
    with self.test_session():
      with self.assertRaises(ValueError):
        tf.pad(
            tf.reshape([1, 2], shape=[1, 2, 1, 1, 1, 1]),
            tf.reshape([1, 2], shape=[1, 2]))

  def testPaddingsDim(self):
    with self.test_session():
      with self.assertRaises(ValueError):
        tf.pad(
            tf.reshape([1, 2], shape=[1, 2]),
            tf.reshape([1, 2], shape=[2]))

  def testPaddingsDim2(self):
    with self.test_session():
      with self.assertRaises(ValueError):
        tf.pad(
            tf.reshape([1, 2], shape=[1, 2]),
            tf.reshape([1, 2], shape=[2, 1]))

  def testPaddingsDim3(self):
    with self.test_session():
      with self.assertRaises(ValueError):
        tf.pad(
            tf.reshape([1, 2], shape=[1, 2]),
            tf.reshape([1, 2], shape=[1, 2]))

  def testPaddingsDim4(self):
    with self.test_session():
      with self.assertRaises(ValueError):
        tf.pad(
            tf.reshape([1, 2], shape=[1, 2]),
            tf.reshape([1, 2, 3, 4, 5, 6], shape=[3, 2]))

  def testPaddingsNonNegative(self):
    with self.test_session():
      with self.assertRaisesRegexp(ValueError, "must be non-negative"):
        tf.pad(
            tf.constant([1], shape=[1]),
            tf.constant([-1, 0], shape=[1, 2]))

  def testPaddingsNonNegative2(self):
    with self.test_session():
      with self.assertRaisesRegexp(ValueError, "must be non-negative"):
        tf.pad(
            tf.constant([1], shape=[1]),
            tf.constant([-1, 0], shape=[1, 2]))

  def testPaddingsMaximum(self):
    with self.test_session():
      with self.assertRaises(Exception):
        tf.pad(
            tf.constant([1], shape=[2]),
            tf.constant([2, 0], shape=[1, 2]),
            mode="REFLECT").eval()
      with self.assertRaises(Exception):
        tf.pad(
            tf.constant([1], shape=[2]),
            tf.constant([0, 3], shape=[1, 2]),
            mode="SYMMETRIC").eval()

  def testIntTypes(self):
    # TODO(touts): Figure out why the padding tests do not work on GPU
    # for int types and rank > 2.
    for t in [np.int32, np.int64]:
      self._testAll((np.random.rand(4, 4, 3) * 100).astype(t),
                    [[1, 0], [2, 3], [0, 2]])

  def testFloatTypes(self):
    for t in [np.float32, np.float64]:
      self._testAll(np.random.rand(2, 5).astype(t),
                    [[1, 0], [2, 0]])

  def testComplexTypes(self):
    for t in [np.complex64, np.complex128]:
      x = np.random.rand(2, 5).astype(t)
      self._testAll(x + 1j *x, [[1, 0], [2, 0]])

  def testShapeFunctionEdgeCases(self):
    # Unknown paddings shape.
    inp = tf.constant(0.0, shape=[4, 4, 4, 4])
    padded = tf.pad(inp, tf.placeholder(tf.int32))
    self.assertEqual([None, None, None, None], padded.get_shape().as_list())

    # Unknown input shape.
    inp = tf.placeholder(tf.float32)
    padded = tf.pad(inp, [[2, 2], [2, 2]])
    self.assertEqual([None, None], padded.get_shape().as_list())

    # Unknown input and paddings shape.
    inp = tf.placeholder(tf.float32)
    padded = tf.pad(inp, tf.placeholder(tf.int32))
    self.assertAllEqual(None, padded.get_shape().ndims)

  def testScalars(self):
    paddings = np.zeros((0, 2), dtype=np.int32)
    inp = np.asarray(7)
    with self.test_session():
      tf_val = tf.pad(inp, paddings)
      out = tf_val.eval()
    self.assertAllEqual(inp, out)
    self.assertShapeEqual(inp, tf_val)


if __name__ == "__main__":
  tf.test.main()
