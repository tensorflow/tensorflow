# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for LossScaleOptimizer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.mixed_precision.python import loss_scale_manager as lsm_lib
from tensorflow.contrib.mixed_precision.python import loss_scale_optimizer as lso
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import gradient_descent as gd


class LossScaleOptimizerTest(test.TestCase):

  def _build_graph(self, lr, init_val, loss_scale_opt_fn=None):
    x = variable_scope.get_variable(
        "x", initializer=init_val, dtype=dtypes.float32)
    c1 = constant_op.constant(1e4, dtype=dtypes.float16)
    c2 = constant_op.constant(1e-4, dtype=dtypes.float16)
    c3 = constant_op.constant(1e-4, dtype=dtypes.float16)
    if context.executing_eagerly():
      loss = lambda: math_ops.cast(x, dtypes.float16) * c1 * c2 * c3
    else:
      loss = math_ops.cast(x, dtypes.float16) * c1 * c2 * c3

    opt = gd.GradientDescentOptimizer(lr)
    if loss_scale_opt_fn:
      opt = loss_scale_opt_fn(opt)
    return x, loss, opt

  @test_util.run_in_graph_and_eager_modes()
  def test_float16_underflow_without_loss_scale(self):
    lr = 1
    init_val = 1.
    x, loss, opt = self._build_graph(lr, init_val)

    self.evaluate(variables.global_variables_initializer())
    self.evaluate(opt.minimize(loss, var_list=[x]))

    # Symbolic grad is c1 * c2 * c3 = 1e-4 and actual grad is 0, since in
    # backprop, c2 * c3 underflows in fp16 range. So variable isn't updated.
    expected_update = 0
    symbolic_update = 1e-4 * lr
    self.assertAllClose(
        init_val - expected_update,
        self.evaluate(x),
        rtol=0,
        atol=min(symbolic_update, 1e-6))

  @test_util.run_in_graph_and_eager_modes()
  def test_float16_with_loss_scale(self):
    lr = 1.
    init_val = 1.

    def loss_scale_opt_fn(opt):
      return lso.LossScaleOptimizer(opt, lsm_lib.FixedLossScaleManager(1e4))

    x, loss, opt = self._build_graph(lr, init_val, loss_scale_opt_fn)

    self.evaluate(variables.global_variables_initializer())
    self.evaluate(opt.minimize(loss, var_list=[x]))

    # Symbolic grad is c1 * c2 * c3 = 1e-4 and actual grad is the same, due to
    # up-scaled loss before backprop starts.
    expected_update = 1.e-4 * lr
    self.assertAllClose(
        init_val - expected_update,
        self.evaluate(x),
        rtol=0,
        atol=min(expected_update, 1e-6))

  @test_util.run_in_graph_and_eager_modes()
  def test_compute_gradients_with_loss_scale(self):
    lr = 1
    init_val = 1.

    def loss_scale_opt_fn(opt):
      return lso.LossScaleOptimizer(opt, lsm_lib.FixedLossScaleManager(1e4))

    x, loss, opt = self._build_graph(lr, init_val, loss_scale_opt_fn)
    grads_and_vars = opt.compute_gradients(loss, var_list=[x])

    self.assertEqual(len(grads_and_vars), 1)

    self.evaluate(variables.global_variables_initializer())
    g_v = self.evaluate(grads_and_vars[0][0])
    self.assertAllClose(g_v, 1e-4)
    self.assertIs(grads_and_vars[0][1], x)
    # Gradients aren't applied.
    self.assertAllClose(init_val, self.evaluate(x), rtol=0, atol=1e-6)

  @test_util.run_in_graph_and_eager_modes()
  def test_compute_gradients_without_loss_scale(self):
    lr = 1
    init_val = 1.
    x, loss, opt = self._build_graph(lr, init_val)
    grads_and_vars = opt.compute_gradients(loss, var_list=[x])

    self.assertEqual(len(grads_and_vars), 1)
    self.evaluate(variables.global_variables_initializer())
    g_v = self.evaluate(grads_and_vars[0][0])
    self.assertAllClose(g_v, 0)

  @test_util.run_in_graph_and_eager_modes()
  def test_apply_gradients(self):

    x = variable_scope.get_variable("x", initializer=1., dtype=dtypes.float32)
    dataset = dataset_ops.Dataset.from_tensor_slices([np.nan, np.inf, 0.1])
    itr = dataset.make_one_shot_iterator()

    lr = 1
    opt = gd.GradientDescentOptimizer(lr)
    lsm = lsm_lib.FixedLossScaleManager(1.e4)
    opt = lso.LossScaleOptimizer(opt, lsm)
    train_fn = lambda: opt.apply_gradients([(itr.get_next(), x)])
    if not context.executing_eagerly():
      train_op = train_fn()

    expected_output = [1, 1, 1 - 0.1]
    actual_output = []

    self.evaluate(variables.global_variables_initializer())
    for _ in range(3):
      # nan or inf is not applied.
      if context.executing_eagerly():
        train_fn()
      else:
        self.evaluate(train_op)
      actual_output.append(self.evaluate(x))
    self.assertAllClose(expected_output, actual_output)

  @test_util.run_in_graph_and_eager_modes()
  def test_apply_gradients_loss_scale_is_updated(self):

    class SimpleLossScaleManager(lsm_lib.LossScaleManager):
      """A simple loss scale manager for easier testing.

      It increments loss scale by 1 if grads are finite, and decreases loss
      scale by 1 if otherwise.
      """

      def __init__(self, loss_scale):
        self._loss_scale = variable_scope.variable(
            name="loss_scale",
            initial_value=loss_scale,
            dtype=dtypes.float32,
            trainable=False)

      def get_loss_scale(self):
        return self._loss_scale

      def update_loss_scale(self, if_finite_grads):
        return control_flow_ops.cond(
            if_finite_grads, lambda: state_ops.assign_add(self._loss_scale, 1),
            lambda: state_ops.assign_sub(self._loss_scale, 1))

    x = variable_scope.get_variable("x", initializer=1., dtype=dtypes.float32)
    dataset = dataset_ops.Dataset.from_tensor_slices([np.nan, np.inf, 0.1])
    itr = dataset.make_one_shot_iterator()

    lr = 1
    init_loss_scale = 8
    opt = gd.GradientDescentOptimizer(lr)
    lsm = SimpleLossScaleManager(init_loss_scale)
    opt = lso.LossScaleOptimizer(opt, lsm)
    train_fn = lambda: opt.apply_gradients([(itr.get_next(), x)])
    if not context.executing_eagerly():
      train_op = train_fn()

    self.evaluate(variables.global_variables_initializer())

    expected_loss_scale = [
        init_loss_scale - 1, init_loss_scale - 2, init_loss_scale - 2 + 1
    ]
    expected_output = [1, 1, 1 - 0.1]
    actual_output = []
    for i in range(3):
      # nan or inf is not applied.
      if context.executing_eagerly():
        train_fn()
      else:
        self.evaluate(train_op)
      actual_output.append(self.evaluate(x))
      self.assertAllClose(expected_loss_scale[i],
                          self.evaluate(lsm._loss_scale))
    self.assertAllClose(expected_output, actual_output)


if __name__ == "__main__":
  test.main()
