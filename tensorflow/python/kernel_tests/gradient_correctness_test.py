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

from absl.testing import parameterized
import numpy as np

from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class GradientCorrectnessTest(test.TestCase, parameterized.TestCase):

  @parameterized.parameters(set((True, context.executing_eagerly())))
  def testMultipleOutputChainedGradients(self, use_tape):
    with test_util.AbstractGradientTape(use_tape=use_tape) as tape:
      x = constant_op.constant(1.0, dtype=dtypes.float32)
      tape.watch(x)

      yexp = math_ops.exp(x)
      yexplog = math_ops.log(yexp)
      grads = tape.gradient([yexp, yexplog], [x])
      grad_vals = self.evaluate(grads)
      exp1_plus_one = (1.0 + np.exp(1.0)).astype(np.float32)
      # [dexp(x)/dx + d(log(exp(x)))/dx] @ x=1 == exp(1) + 1
      self.assertAllClose(grad_vals[0], exp1_plus_one)

  @parameterized.parameters(set((True, context.executing_eagerly())))
  def testIdentityGradient(self, use_tape):
    x = constant_op.constant(3.)
    with test_util.AbstractGradientTape(use_tape=use_tape) as tape:
      tape.watch(x)
      dx_dx = tape.gradient(x, x)
    self.assertAllClose(1., self.evaluate(dx_dx))

  @parameterized.parameters(set((True, context.executing_eagerly())))
  def testIntegerIdentityGradient(self, use_tape):
    x = constant_op.constant(3)
    with test_util.AbstractGradientTape(use_tape=use_tape) as tape:
      tape.watch(x)
      dx_dx = tape.gradient(x, x)
    self.assertAllClose(1, self.evaluate(dx_dx))

  @parameterized.parameters(set((True, context.executing_eagerly())))
  def testGradientWithIntegerPath(self, use_tape):
    with test_util.AbstractGradientTape(use_tape=use_tape) as tape:
      x = constant_op.constant([3.9, 4.1])
      tape.watch(x)

      k = math_ops.cast(math_ops.cast(x, dtypes.int32), dtypes.float32)
      y = x * k
      dy_dx = tape.gradient(y, x)
      self.assertAllClose([3., 4.], self.evaluate(dy_dx))

  @parameterized.parameters(set((True, context.executing_eagerly())))
  def testNoIntegerGradient1(self, use_tape):
    with test_util.AbstractGradientTape(use_tape=use_tape) as tape:
      x = constant_op.constant([3.9, 4.1])
      tape.watch(x)

      k = math_ops.cast(math_ops.cast(x, dtypes.int32), dtypes.float32)
      y = k * k
      dy_dx = tape.gradient(y, x)
      self.assertIsNone(dy_dx)

  @parameterized.parameters(set((True, context.executing_eagerly())))
  def testNoIntegerGradient2(self, use_tape):
    with test_util.AbstractGradientTape(use_tape=use_tape) as tape:
      k = constant_op.constant([3, 4])
      x = math_ops.cast(k, dtypes.float32)
      tape.watch([k, x])

      y = x * x
      dy_dk = tape.gradient(y, k)
      self.assertIsNone(dy_dk)

  @parameterized.parameters(set((True, context.executing_eagerly())))
  def testNoIntegerGradient3(self, use_tape):
    with test_util.AbstractGradientTape(use_tape=use_tape) as tape:
      k = constant_op.constant([3, 4])
      tape.watch(k)

      m = k * k
      dm_dk = tape.gradient(m, k)
      self.assertIsNone(dm_dk)

  @parameterized.parameters(set((True, context.executing_eagerly())))
  def testNoIntegerGradient4(self, use_tape):
    with test_util.AbstractGradientTape(use_tape=use_tape) as tape:
      k = constant_op.constant([3, 4])
      tape.watch(k)

      m = k * k * k
      dm_dk = tape.gradient(m, k)
      self.assertIsNone(dm_dk)

  @parameterized.parameters(set((True, context.executing_eagerly())))
  def testNoIntegerGradient5(self, use_tape):
    with test_util.AbstractGradientTape(use_tape=use_tape) as tape:
      k = constant_op.constant([3, 4])
      tape.watch(k)

      m = k * k
      n = m * m
      dn_dk = tape.gradient(n, k)
      self.assertIsNone(dn_dk)

  @parameterized.parameters(set((True, context.executing_eagerly())))
  def testNoIntegerGradient6(self, use_tape):
    with test_util.AbstractGradientTape(
        use_tape=use_tape, persistent=True) as tape:
      k = constant_op.constant(3)
      tape.watch(k)

      x = math_ops.cast(k, dtypes.float32)
      grad_1 = tape.gradient(k * k, k)
      grad_2 = tape.gradient(x * x, k)
      grad_3 = tape.gradient(math_ops.square(k), k)
      grad_4 = tape.gradient(math_ops.square(x), k)
      self.assertIsNone(grad_1)
      self.assertIsNone(grad_2)
      self.assertIsNone(grad_3)
      self.assertIsNone(grad_4)


if __name__ == '__main__':
  test.main()
