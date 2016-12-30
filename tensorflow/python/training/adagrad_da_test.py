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
"""Functional tests for AdagradDA operations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import adagrad_da


class AdagradDAOptimizerTest(test.TestCase):

  def testAdagradDAwithoutRegularizationBasic1(self):
    for dtype in [dtypes.float64, dtypes.float32]:
      with self.test_session() as sess:
        global_step = variables.Variable(0, dtype=dtypes.int64)
        var0 = variables.Variable([0.0, 0.0], dtype=dtype)
        var1 = variables.Variable([0.0, 0.0], dtype=dtype)
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
        variables.global_variables_initializer().run()

        v0_val, v1_val = sess.run([var0, var1])
        self.assertAllClose([0.0, 0.0], v0_val)
        self.assertAllClose([0.0, 0.0], v1_val)

        # Run a step of AdagradDA
        update.run()

        v0_val, v1_val = sess.run([var0, var1])
        # Let g to be gradient accumulator, gg to be gradient squared
        # accumulator, T be the global step, lr is the learning rate, and k the
        # initial gradient squared accumulator value.
        # w = \dfrac{sign(-g)*lr*|g - l1*T|_{+}}{l2*T*lr + \sqrt{k+gg})}
        # For -0.1*3.0*(0.1 - 0)/(0 + sqrt(0.1 + 0.1*0.1)) = -0.904534
        # similarly for others.
        self.assertAllCloseAccordingToType(
            np.array([-0.904534, -1.603567]), v0_val)
        self.assertAllCloseAccordingToType(
            np.array([-0.094821, -0.189358]), v1_val)

  def testAdagradDAwithoutRegularizationBasic2(self):
    for dtype in [dtypes.float64, dtypes.float32]:
      with self.test_session() as sess:
        global_step = variables.Variable(0, dtype=dtypes.int64)
        var0 = variables.Variable([1.0, 2.0], dtype=dtype)
        var1 = variables.Variable([4.0, 3.0], dtype=dtype)
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
        variables.global_variables_initializer().run()

        v0_val, v1_val = sess.run([var0, var1])
        self.assertAllCloseAccordingToType([1.0, 2.0], v0_val)
        self.assertAllCloseAccordingToType([4.0, 3.0], v1_val)

        # Run a step of AdagradDA
        update.run()

        v0_val, v1_val = sess.run([var0, var1])
        self.assertAllCloseAccordingToType(
            np.array([-0.904534, -1.603567]), v0_val)
        self.assertAllCloseAccordingToType(
            np.array([-0.094821, -0.189358]), v1_val)

  def testAdagradDAWithL1(self):
    for dtype in [dtypes.float64, dtypes.float32]:
      with self.test_session() as sess:
        global_step = variables.Variable(0, dtype=dtypes.int64)
        var0 = variables.Variable([1.0, 2.0], dtype=dtype)
        var1 = variables.Variable([4.0, 3.0], dtype=dtype)
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
        variables.global_variables_initializer().run()

        v0_val, v1_val = sess.run([var0, var1])
        self.assertAllCloseAccordingToType([1.0, 2.0], v0_val)
        self.assertAllCloseAccordingToType([4.0, 3.0], v1_val)

        # Run a step of AdagradDA
        update.run()

        v0_val, v1_val = sess.run([var0, var1])
        self.assertAllCloseAccordingToType(
            np.array([-0.895489, -1.59555]), v0_val)
        self.assertAllCloseAccordingToType(
            np.array([-0.085339, -0.17989]), v1_val)

  def testAdagradDAWithL1_L2(self):
    for dtype in [dtypes.float64, dtypes.float32]:
      with self.test_session() as sess:
        global_step = variables.Variable(0, dtype=dtypes.int64)
        var0 = variables.Variable([1.0, 2.0], dtype=dtype)
        var1 = variables.Variable([4.0, 3.0], dtype=dtype)
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
        variables.global_variables_initializer().run()

        v0_val, v1_val = sess.run([var0, var1])
        self.assertAllCloseAccordingToType([1.0, 2.0], v0_val)
        self.assertAllCloseAccordingToType([4.0, 3.0], v1_val)

        # Run a step of AdagradDA
        update.run()

        v0_val, v1_val = sess.run([var0, var1])
        self.assertAllCloseAccordingToType(
            np.array([-0.046907, -0.093659]), v0_val)
        self.assertAllCloseAccordingToType(
            np.array([-0.004275, -0.009023]), v1_val)


if __name__ == "__main__":
  test.main()
