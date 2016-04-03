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

"""Tests for tensorflow.ops.math_ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import test_util
from tensorflow.python.ops import constant_op
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import googletest

exp = np.exp
log = np.log


class LBetaTest(test_util.TensorFlowTestCase):

  def testOneDimensionalArg(self):
    # Should evaluate to 1 and 1/2.
    x_one = [1, 1.]
    x_one_half = [2, 1.]
    with self.test_session():
      self.assertAllClose(1, exp(math_ops.lbeta(x_one).eval()))
      self.assertAllClose(0.5, exp(math_ops.lbeta(x_one_half).eval()))

  def testTwoDimensionalArg(self):
    # Should evaluate to 1/2.
    x_one_half = [[2, 1.], [2, 1.]]
    with self.test_session():
      self.assertAllClose([0.5, 0.5], exp(math_ops.lbeta(x_one_half).eval()))

  def testLengthOneLastDimensionResultsInOne(self):
    # If there is only one coefficient, the formula still works, and we get one
    # as the answer, alwyas.
    x_a = [5.5]
    x_b = [0.1]
    with self.test_session():
      self.assertAllClose(1, exp(math_ops.lbeta(x_a).eval()))
      self.assertAllClose(1, exp(math_ops.lbeta(x_b).eval()))


class ReduceTest(test_util.TensorFlowTestCase):

  def testReduceAllDims(self):
    x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    with self.test_session():
      y_tf = math_ops.reduce_sum(x).eval()
      self.assertEqual(y_tf, 21)

class RoundTest(test_util.TensorFlowTestCase):

  def testRounding(self):
    x = [0.49, 0.7, -0.3, -0.8]
    for dtype in [np.float32, np.double]:
      x_np = np.array(x, dtype=dtype)
      for use_gpu in [True, False]:
        with self.test_session(use_gpu=use_gpu):
          x_tf = constant_op.constant(x_np, shape=x_np.shape)
          y_tf = math_ops.round(x_tf)
          y_tf_np = y_tf.eval()
          y_np = np.round(x_np)
          self.assertAllClose(y_tf_np, y_np, atol=1e-2)


class ModTest(test_util.TensorFlowTestCase):

  def testFloat(self):
    x = [0.5, 0.7, 0.3]
    for dtype in [np.float32, np.double]:
      # Test scalar and vector versions.
      for denom in [x[0], [x[0]] * 3]:
        x_np = np.array(x, dtype=dtype)
        with self.test_session():
          x_tf = constant_op.constant(x_np, shape=x_np.shape)
          y_tf = math_ops.mod(x_tf, denom)
          y_tf_np = y_tf.eval()
          y_np = np.fmod(x_np, denom)
        self.assertAllClose(y_tf_np, y_np, atol=1e-2)

  def testFixed(self):
    x = [5, 10, 23]
    for dtype in [np.int32, np.int64]:
      # Test scalar and vector versions.
      for denom in [x[0], x]:
        x_np = np.array(x, dtype=dtype)
        with self.test_session():
          x_tf = constant_op.constant(x_np, shape=x_np.shape)
          y_tf = math_ops.mod(x_tf, denom)
          y_tf_np = y_tf.eval()
          y_np = np.mod(x_np, denom)
        self.assertAllClose(y_tf_np, y_np)


class SquaredDifferenceTest(test_util.TensorFlowTestCase):

  def testSquaredDifference(self):
    x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    y = np.array([-3, -2, -1], dtype=np.int32)
    z = (x - y)*(x - y)
    with self.test_session():
      z_tf = math_ops.squared_difference(x, y).eval()
      self.assertAllClose(z, z_tf)

if __name__ == "__main__":
  googletest.main()
