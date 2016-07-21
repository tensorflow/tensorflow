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

"""Tests for SoftmaxOp and LogSoftmaxOp."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import numpy as np
import tensorflow as tf


class SoftmaxTest(tf.test.TestCase):

  def _npSoftmax(self, features, log=False):
    batch_dim = 0
    class_dim = 1
    batch_size = features.shape[batch_dim]
    e = np.exp(features -
               np.reshape(np.amax(features, axis=class_dim), [batch_size, 1]))
    softmax = e / np.reshape(np.sum(e, axis=class_dim), [batch_size, 1])
    if log:
      return np.log(softmax)
    else:
      return softmax

  def _testSoftmax(self, np_features, log=False, use_gpu=False):
    # A previous version of the code checked the op name rather than the op type
    # to distinguish between log and non-log.  Use an arbitrary name to catch
    # this bug in future.
    name = "arbitrary"
    np_softmax = self._npSoftmax(np_features, log=log)
    with self.test_session(use_gpu=use_gpu):
      if log:
        tf_softmax = tf.nn.log_softmax(np_features, name=name)
      else:
        tf_softmax = tf.nn.softmax(np_features, name=name)
      out = tf_softmax.eval()
    self.assertAllCloseAccordingToType(np_softmax, out)
    self.assertShapeEqual(np_softmax, tf_softmax)
    if not log:
      # Bonus check: the softmaxes should add to one in each
      # batch element.
      self.assertAllCloseAccordingToType(np.ones(out.shape[0]),
                                         np.sum(out, axis=1))

  def _testAll(self, features):
    self._testSoftmax(features, use_gpu=False)
    self._testSoftmax(features, log=True, use_gpu=False)
    self._testSoftmax(features, use_gpu=True)
    self._testSoftmax(features, log=True, use_gpu=True)
    self._testOverflow(use_gpu=True)


  def testNpSoftmax(self):
    features = [[1., 1., 1., 1.], [1., 2., 3., 4.]]
    # Batch 0: All exps are 1.  The expected result is
    # Softmaxes = [0.25, 0.25, 0.25, 0.25]
    # LogSoftmaxes = [-1.386294, -1.386294, -1.386294, -1.386294]
    #
    # Batch 1:
    # exps = [1., 2.718, 7.389, 20.085]
    # sum = 31.192
    # Softmaxes = exps / sum = [0.0320586, 0.08714432, 0.23688282, 0.64391426]
    # LogSoftmaxes = [-3.44019 , -2.44019 , -1.44019 , -0.44019]
    np_sm = self._npSoftmax(np.array(features))
    self.assertAllClose(
        np.array([[0.25, 0.25, 0.25, 0.25],
                  [0.0320586, 0.08714432, 0.23688282, 0.64391426]]),
        np_sm,
        rtol=1.e-5, atol=1.e-5)
    np_lsm = self._npSoftmax(np.array(features), log=True)
    self.assertAllClose(
        np.array([[-1.386294, -1.386294, -1.386294, -1.386294],
                  [-3.4401897, -2.4401897, -1.4401897, -0.4401897]]),
        np_lsm,
        rtol=1.e-5, atol=1.e-5)

  def testShapeMismatch(self):
    with self.assertRaises(ValueError):
      tf.nn.softmax([0., 1., 2., 3.])
    with self.assertRaises(ValueError):
      tf.nn.log_softmax([0., 1., 2., 3.])

  def _testOverflow(self, use_gpu=False):
    if use_gpu:
        type = np.float32
    else:
        type = np.float64
    max = np.finfo(type).max
    features = np.array(
        [[1., 1., 1., 1.],
         [max, 1., 2., 3.]]).astype(type)
    with self.test_session(use_gpu=use_gpu):
      tf_log_softmax = tf.nn.log_softmax(features)
      out = tf_log_softmax.eval()
    self.assertAllClose(
        np.array([[-1.386294, -1.386294, -1.386294, -1.386294],
                  [0, -max, -max, -max]]),
        out,
        rtol=1.e-5, atol=1.e-5)

  def testFloat(self):
    self._testAll(
        np.array([[1., 1., 1., 1.], [1., 2., 3., 4.]]).astype(np.float32))

  def testHalf(self):
    self._testAll(
        np.array([[1., 1., 1., 1.], [1., 2., 3., 4.]]).astype(np.float16))

  def testDouble(self):
    self._testSoftmax(
        np.array([[1., 1., 1., 1.], [1., 2., 3., 4.]]).astype(np.float64),
        use_gpu=False)
    self._testOverflow(use_gpu=False)


  def testEmpty(self):
    with self.test_session():
      x = tf.constant([[]], shape=[0, 3])
      self.assertEqual(0, tf.size(x).eval())
      expected_y = np.array([]).reshape(0, 3)
      np.testing.assert_array_equal(expected_y, tf.nn.softmax(x).eval())


if __name__ == "__main__":
  tf.test.main()
