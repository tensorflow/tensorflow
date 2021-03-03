# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for AdagradDA optimizer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import adagrad_da


class AdagradDAOptimizerTest(xla_test.XLATestCase):

  def testAdagradDAWithoutRegularizationBasic1(self):
    for dtype in self.float_types:
      with self.session(), self.test_scope():
        global_step = resource_variable_ops.ResourceVariable(
            0, dtype=dtypes.int64)
        var0 = resource_variable_ops.ResourceVariable([0.0, 0.0], dtype=dtype)
        var1 = resource_variable_ops.ResourceVariable([0.0, 0.0], dtype=dtype)
        grads0 = constant_op.constant([0.1, 0.2], dtype=dtype)
        grads1 = constant_op.constant([0.01, 0.02], dtype=dtype)
        opt = adagrad_da.AdagradDAOptimizer(
            3.0,
            global_step,
            initial_gradient_squared_accumulator_value=0.1,
            l1_regularization_strength=0.0,
            l2_regularization_strength=0.0)
        update = opt.apply_gradients(
            zip([grads0, grads1], [var0, var1]), global_step=global_step)
        self.evaluate(variables.global_variables_initializer())

        self.assertAllClose([0.0, 0.0], self.evaluate(var0))
        self.assertAllClose([0.0, 0.0], self.evaluate(var1))

        # Run a step of AdagradDA
        update.run()

        # Let g be the gradient accumulator, gg be the gradient squared
        # accumulator, T be the global step, lr be the learning rate,
        # and k the initial gradient squared accumulator value.
        # w = \dfrac{sign(-g)*lr*|g - l1*T|_{+}}{l2*T*lr + \sqrt{k+gg})}
        # For -0.1*3.0*(0.1 - 0)/(0 + sqrt(0.1 + 0.1*0.1)) = -0.904534
        # similarly for others.
        self.assertAllCloseAccordingToType(
            np.array([-0.904534, -1.603567]), self.evaluate(var0))
        self.assertAllCloseAccordingToType(
            np.array([-0.094821, -0.189358]), self.evaluate(var1))

  def testAdagradDAwithoutRegularizationBasic2(self):
    for dtype in self.float_types:
      with self.session(), self.test_scope():
        global_step = resource_variable_ops.ResourceVariable(
            0, dtype=dtypes.int64)
        var0 = resource_variable_ops.ResourceVariable([1.0, 2.0], dtype=dtype)
        var1 = resource_variable_ops.ResourceVariable([4.0, 3.0], dtype=dtype)
        grads0 = constant_op.constant([0.1, 0.2], dtype=dtype)
        grads1 = constant_op.constant([0.01, 0.02], dtype=dtype)

        opt = adagrad_da.AdagradDAOptimizer(
            3.0,
            global_step,
            initial_gradient_squared_accumulator_value=0.1,
            l1_regularization_strength=0.0,
            l2_regularization_strength=0.0)
        update = opt.apply_gradients(
            zip([grads0, grads1], [var0, var1]), global_step=global_step)
        self.evaluate(variables.global_variables_initializer())

        self.assertAllCloseAccordingToType([1.0, 2.0], self.evaluate(var0))
        self.assertAllCloseAccordingToType([4.0, 3.0], self.evaluate(var1))

        # Run a step of AdagradDA
        update.run()

        self.assertAllCloseAccordingToType(
            np.array([-0.904534, -1.603567]), self.evaluate(var0))
        self.assertAllCloseAccordingToType(
            np.array([-0.094821, -0.189358]), self.evaluate(var1))

  def testAdagradDAWithL1(self):
    for dtype in self.float_types:
      with self.session(), self.test_scope():
        global_step = resource_variable_ops.ResourceVariable(
            0, dtype=dtypes.int64)
        var0 = resource_variable_ops.ResourceVariable([1.0, 2.0], dtype=dtype)
        var1 = resource_variable_ops.ResourceVariable([4.0, 3.0], dtype=dtype)
        grads0 = constant_op.constant([0.1, 0.2], dtype=dtype)
        grads1 = constant_op.constant([0.01, 0.02], dtype=dtype)

        opt = adagrad_da.AdagradDAOptimizer(
            3.0,
            global_step,
            initial_gradient_squared_accumulator_value=0.1,
            l1_regularization_strength=0.001,
            l2_regularization_strength=0.0)
        update = opt.apply_gradients(
            zip([grads0, grads1], [var0, var1]), global_step=global_step)
        self.evaluate(variables.global_variables_initializer())

        self.assertAllCloseAccordingToType([1.0, 2.0], self.evaluate(var0))
        self.assertAllCloseAccordingToType([4.0, 3.0], self.evaluate(var1))

        # Run a step of AdagradDA
        update.run()

        self.assertAllCloseAccordingToType(
            np.array([-0.895489, -1.59555]), self.evaluate(var0))
        self.assertAllCloseAccordingToType(
            np.array([-0.085339, -0.17989]), self.evaluate(var1))

  def testAdagradDAWithL1_L2(self):
    for dtype in self.float_types:
      with self.session(), self.test_scope():
        global_step = resource_variable_ops.ResourceVariable(
            0, dtype=dtypes.int64)
        var0 = resource_variable_ops.ResourceVariable([1.0, 2.0], dtype=dtype)
        var1 = resource_variable_ops.ResourceVariable([4.0, 3.0], dtype=dtype)
        grads0 = constant_op.constant([0.1, 0.2], dtype=dtype)
        grads1 = constant_op.constant([0.01, 0.02], dtype=dtype)

        opt = adagrad_da.AdagradDAOptimizer(
            3.0,
            global_step,
            initial_gradient_squared_accumulator_value=0.1,
            l1_regularization_strength=0.001,
            l2_regularization_strength=2.0)
        update = opt.apply_gradients(
            zip([grads0, grads1], [var0, var1]), global_step=global_step)
        self.evaluate(variables.global_variables_initializer())

        self.assertAllCloseAccordingToType([1.0, 2.0], self.evaluate(var0))
        self.assertAllCloseAccordingToType([4.0, 3.0], self.evaluate(var1))

        # Run a step of AdagradDA
        update.run()

        self.assertAllCloseAccordingToType(
            np.array([-0.046907, -0.093659]), self.evaluate(var0))
        self.assertAllCloseAccordingToType(
            np.array([-0.004275, -0.009023]), self.evaluate(var1))


if __name__ == "__main__":
  test.main()
