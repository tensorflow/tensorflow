# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for Custom Gradient Ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.bayesflow.python.ops import custom_grad_impl
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


cg = custom_grad_impl


class CustomGradientTest(test.TestCase):

  def test_works_correctly(self):
    with self.test_session() as sess:
      f = lambda x: x**2 / 2
      g = lambda x: (x - 1)**3 / 3
      x_ = np.linspace(-100, 100, int(1e4)) + [0.]

      x = constant_op.constant(x_)
      fx = cg.custom_gradient(f(x), g(x), x)
      gx = gradients_impl.gradients(fx, x)[0]
      [fx_, gx_] = sess.run([fx, gx])

      self.assertAllClose(f(x_), fx_)
      self.assertAllClose(g(x_), gx_)

  def test_works_correctly_both_f_g_zero(self):
    with self.test_session() as sess:
      f = lambda x: x**2 / 2
      g = lambda x: x**3 / 3
      x_ = np.linspace(-100, 100, int(1e4)) + [0.]

      x = constant_op.constant(x_)
      fx = cg.custom_gradient(f(x), g(x), x)
      gx = gradients_impl.gradients(fx, x)[0]
      [fx_, gx_] = sess.run([fx, gx])

      self.assertAllClose(f(x_), fx_)
      self.assertAllClose(g(x_), gx_)

  def test_works_correctly_vector_of_vars(self):
    with self.test_session() as sess:
      x = variable_scope.get_variable(
          name="x",
          shape=[],
          dtype=dtypes.float32,
          initializer=init_ops.constant_initializer(2))
      y = variable_scope.get_variable(
          name="y",
          shape=[],
          dtype=dtypes.float32,
          initializer=init_ops.constant_initializer(3))
      sess.run([variables.global_variables_initializer()])

      f = lambda z: z[0] * z[1]
      g = lambda z: z[0]**2 * z[1]**2 / 2

      z = array_ops.stack([x, y])
      fz = cg.custom_gradient(f(z), g(z), z, axis=0)
      gz = gradients_impl.gradients(fz, variables.trainable_variables())
      [z_, fz_, gx_, gy_] = sess.run([z, fz, gz[0], gz[1]])

      self.assertEqual(f(z_), fz_)
      self.assertEqual(g(z_), gx_)
      self.assertEqual(g(z_), gy_)

  def test_works_correctly_side_vars(self):
    with self.test_session() as sess:
      x_ = np.float32(2.1)  # Adding extra tenth to force imprecision.
      y_ = np.float32(3.1)
      x = variable_scope.get_variable(
          name="x",
          shape=[],
          dtype=dtypes.float32,
          initializer=init_ops.constant_initializer(x_))
      y = variable_scope.get_variable(
          name="y",
          shape=[],
          dtype=dtypes.float32,
          initializer=init_ops.constant_initializer(y_))
      sess.run([variables.global_variables_initializer()])

      f = lambda x: x * y
      g = lambda z: math_ops.square(x) * y

      fx = cg.custom_gradient(f(x), g(x), x)
      gx = gradients_impl.gradients(fx, variables.trainable_variables())
      [x_, fx_, gx_] = sess.run([x, fx, gx[0]])
      gy_ = gx[1]

      self.assertEqual(x_ * y_, fx_)
      self.assertEqual(np.square(x_) * y_, gx_)
      self.assertEqual(None, gy_)

  def test_works_correctly_fx_gx_manually_stopped(self):
    with self.test_session() as sess:
      x_ = np.float32(2.1)  # Adding extra tenth to force imprecision.
      y_ = np.float32(3.1)
      x = variable_scope.get_variable(
          name="x",
          shape=[],
          dtype=dtypes.float32,
          initializer=init_ops.constant_initializer(x_))
      y = variable_scope.get_variable(
          name="y",
          shape=[],
          dtype=dtypes.float32,
          initializer=init_ops.constant_initializer(y_))
      sess.run([variables.global_variables_initializer()])

      stop = array_ops.stop_gradient  # For readability.

      # Basically we need to stop the `x` portion of `f`. And when we supply the
      # arg to `custom_gradient` we need to stop the complement, i.e., the `y`
      # part.
      f = lambda x: stop(x) * y
      g = lambda x: stop(math_ops.square(x)) * y
      fx = cg.custom_gradient(f(x), g(x), x + stop(y),
                              fx_gx_manually_stopped=True)

      gx = gradients_impl.gradients(fx, variables.trainable_variables())
      [x_, fx_, gx_, gy_] = sess.run([x, fx, gx[0], gx[1]])

      self.assertEqual(x_ * y_, fx_)
      self.assertEqual(np.square(x_) * y_, gx_)
      self.assertEqual(x_, gy_)


if __name__ == "__main__":
  test.main()
