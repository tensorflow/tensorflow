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
"""Tests for tensorflow.ops.argmax_op."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class GradientCorrectnessTest(test.TestCase):

  def testMultipleOutputChainedGradients(self):
    with self.test_session() as sess:
      x = constant_op.constant(1.0, dtype=dtypes.float32)
      yexp = math_ops.exp(x)
      yexplog = math_ops.log(yexp)
      grads = gradients_impl.gradients([yexp, yexplog], [x])
      grad_vals = sess.run(grads)
      exp1_plus_one = (1.0 + np.exp(1.0)).astype(np.float32)
      # [dexp(x)/dx + d(log(exp(x)))/dx] @ x=1 == exp(1) + 1
      self.assertAllClose(grad_vals[0], exp1_plus_one)

  def testIdentityGradient(self):
    x = constant_op.constant(3.)
    dx_dx, = gradients_impl.gradients(x, x)
    with self.test_session() as sess:
      self.assertAllClose(1., sess.run(dx_dx))

  def testIntegerIdentityGradient(self):
    x = constant_op.constant(3)
    dx_dx, = gradients_impl.gradients(x, x)
    with self.test_session() as sess:
      self.assertAllClose(1, sess.run(dx_dx))

  def testGradientWithIntegerPath(self):
    x = constant_op.constant([3.9, 4.1])
    k = math_ops.to_float(math_ops.to_int32(x))
    y = x * k
    dy_dx, = gradients_impl.gradients(y, x)
    with self.test_session() as sess:
      self.assertAllClose([3., 4.], sess.run(dy_dx))

  def testNoIntegerGradient1(self):
    x = constant_op.constant([3.9, 4.1])
    k = math_ops.to_float(math_ops.to_int32(x))
    y = k * k
    dy_dx, = gradients_impl.gradients(y, x)
    self.assertIsNone(dy_dx)

  def testNoIntegerGradient2(self):
    k = constant_op.constant([3, 4])
    x = math_ops.to_float(k)
    y = x * x
    dy_dk, = gradients_impl.gradients(y, k)
    self.assertIsNone(dy_dk)

  def testNoIntegerGradient3(self):
    k = constant_op.constant([3, 4])
    m = k * k
    dm_dk, = gradients_impl.gradients(m, k)
    self.assertIsNone(dm_dk)

  def testNoIntegerGradient4(self):
    k = constant_op.constant([3, 4])
    m = k * k * k
    dm_dk, = gradients_impl.gradients(m, k)
    self.assertIsNone(dm_dk)

  def testNoIntegerGradient5(self):
    k = constant_op.constant([3, 4])
    m = k * k
    n = m * m
    dn_dk, = gradients_impl.gradients(n, k)
    self.assertIsNone(dn_dk)

  def testNoIntegerGradient6(self):
    k = constant_op.constant(3)
    x = math_ops.to_float(k)
    grad_1, = gradients_impl.gradients(k * k, k)
    grad_2, = gradients_impl.gradients(x * x, k)
    grad_3, = gradients_impl.gradients(math_ops.square(k), k)
    grad_4, = gradients_impl.gradients(math_ops.square(x), k)
    self.assertIsNone(grad_1)
    self.assertIsNone(grad_2)
    self.assertIsNone(grad_3)
    self.assertIsNone(grad_4)


if __name__ == '__main__':
  test.main()
