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
"""Tests for Ftrl optimizer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.compiler.tests.xla_test import XLATestCase
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import adagrad
from tensorflow.python.training import ftrl
from tensorflow.python.training import gradient_descent


class FtrlOptimizerTest(XLATestCase):

  def initVariableAndGradient(self, dtype):
    var0 = resource_variable_ops.ResourceVariable([0.0, 0.0], dtype=dtype)
    var1 = resource_variable_ops.ResourceVariable([0.0, 0.0], dtype=dtype)
    grads0 = constant_op.constant([0.1, 0.2], dtype=dtype)
    grads1 = constant_op.constant([0.02, 0.04], dtype=dtype)

    return var0, var1, grads0, grads1

  def equivAdagradTest_FtrlPart(self, steps, dtype):
    var0, var1, grads0, grads1 = self.initVariableAndGradient(dtype)
    opt = ftrl.FtrlOptimizer(
        3.0,
        learning_rate_power=-0.5,  # using Adagrad learning rate
        initial_accumulator_value=0.1,
        l1_regularization_strength=0.0,
        l2_regularization_strength=0.0)
    ftrl_update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
    variables.global_variables_initializer().run()
    # Fetch params to validate initial values
    self.assertAllClose([0.0, 0.0], var0.eval())
    self.assertAllClose([0.0, 0.0], var1.eval())

    # Run Ftrl for a few steps
    for _ in range(steps):
      ftrl_update.run()

    return var0.eval(), var1.eval()

  def equivAdagradTest_AdagradPart(self, steps, dtype):
    var0, var1, grads0, grads1 = self.initVariableAndGradient(dtype)
    opt = adagrad.AdagradOptimizer(3.0, initial_accumulator_value=0.1)
    adagrad_update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
    variables.global_variables_initializer().run()
    # Fetch params to validate initial values
    self.assertAllClose([0.0, 0.0], var0.eval())
    self.assertAllClose([0.0, 0.0], var1.eval())

    # Run Adagrad for a few steps
    for _ in range(steps):
      adagrad_update.run()

    return var0.eval(), var1.eval()

  def equivGradientDescentTest_FtrlPart(self, steps, dtype):
    var0, var1, grads0, grads1 = self.initVariableAndGradient(dtype)
    opt = ftrl.FtrlOptimizer(
        3.0,
        learning_rate_power=-0.0,  # using Fixed learning rate
        initial_accumulator_value=0.1,
        l1_regularization_strength=0.0,
        l2_regularization_strength=0.0)
    ftrl_update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
    variables.global_variables_initializer().run()
    # Fetch params to validate initial values
    self.assertAllClose([0.0, 0.0], var0.eval())
    self.assertAllClose([0.0, 0.0], var1.eval())

    # Run Ftrl for a few steps
    for _ in range(steps):
      ftrl_update.run()

    return var0.eval(), var1.eval()

  def equivGradientDescentTest_GradientDescentPart(self, steps, dtype):
    var0, var1, grads0, grads1 = self.initVariableAndGradient(dtype)
    opt = gradient_descent.GradientDescentOptimizer(3.0, name="sgd")
    sgd_update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
    variables.global_variables_initializer().run()
    # Fetch params to validate initial values
    self.assertAllClose([0.0, 0.0], var0.eval())
    self.assertAllClose([0.0, 0.0], var1.eval())

    # Run GradientDescent for a few steps
    for _ in range(steps):
      sgd_update.run()

    return var0.eval(), var1.eval()

  def testFtrlwithoutRegularization(self):
    for dtype in self.float_types:
      with self.test_session(), self.test_scope():
        var0 = resource_variable_ops.ResourceVariable([0.0, 0.0], dtype=dtype)
        var1 = resource_variable_ops.ResourceVariable([0.0, 0.0], dtype=dtype)
        grads0 = constant_op.constant([0.1, 0.2], dtype=dtype)
        grads1 = constant_op.constant([0.01, 0.02], dtype=dtype)
        opt = ftrl.FtrlOptimizer(
            3.0,
            initial_accumulator_value=0.1,
            l1_regularization_strength=0.0,
            l2_regularization_strength=0.0)
        ftrl_update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
        variables.global_variables_initializer().run()
        # Fetch params to validate initial values
        self.assertAllClose([0.0, 0.0], var0.eval())
        self.assertAllClose([0.0, 0.0], var1.eval())

        # Run 3 steps FTRL
        for _ in range(3):
          ftrl_update.run()

        # Validate updated params
        self.assertAllCloseAccordingToType(
            np.array([-2.60260963, -4.29698515]), var0.eval())
        self.assertAllCloseAccordingToType(
            np.array([-0.28432083, -0.56694895]), var1.eval())

  def testFtrlwithoutRegularization2(self):
    for dtype in self.float_types:
      with self.test_session(), self.test_scope():
        var0 = resource_variable_ops.ResourceVariable([1.0, 2.0], dtype=dtype)
        var1 = resource_variable_ops.ResourceVariable([4.0, 3.0], dtype=dtype)
        grads0 = constant_op.constant([0.1, 0.2], dtype=dtype)
        grads1 = constant_op.constant([0.01, 0.02], dtype=dtype)
        opt = ftrl.FtrlOptimizer(
            3.0,
            initial_accumulator_value=0.1,
            l1_regularization_strength=0.0,
            l2_regularization_strength=0.0)
        ftrl_update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
        variables.global_variables_initializer().run()
        # Fetch params to validate initial values
        self.assertAllClose([1.0, 2.0], var0.eval())
        self.assertAllClose([4.0, 3.0], var1.eval())

        # Run 3 steps FTRL
        for _ in range(3):
          ftrl_update.run()

        # Validate updated params
        self.assertAllClose(
            np.array([-2.55607247, -3.98729396]), var0.eval(), 1e-5, 1e-5)
        self.assertAllClose(
            np.array([-0.28232238, -0.56096673]), var1.eval(), 1e-5, 1e-5)

  def testFtrlWithL1(self):
    for dtype in self.float_types:
      with self.test_session(), self.test_scope():
        var0 = resource_variable_ops.ResourceVariable([1.0, 2.0], dtype=dtype)
        var1 = resource_variable_ops.ResourceVariable([4.0, 3.0], dtype=dtype)
        grads0 = constant_op.constant([0.1, 0.2], dtype=dtype)
        grads1 = constant_op.constant([0.01, 0.02], dtype=dtype)
        opt = ftrl.FtrlOptimizer(
            3.0,
            initial_accumulator_value=0.1,
            l1_regularization_strength=0.001,
            l2_regularization_strength=0.0)
        ftrl_update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
        variables.global_variables_initializer().run()
        # Fetch params to validate initial values
        self.assertAllClose([1.0, 2.0], var0.eval())
        self.assertAllClose([4.0, 3.0], var1.eval())

        # Run 10 steps FTRL
        for _ in range(10):
          ftrl_update.run()

        # Validate updated params
        self.assertAllClose(np.array([-7.66718769, -10.91273689]), var0.eval())
        self.assertAllClose(np.array([-0.93460727, -1.86147261]), var1.eval())

  def testFtrlWithL1_L2(self):
    for dtype in self.float_types:
      with self.test_session(), self.test_scope():
        var0 = resource_variable_ops.ResourceVariable([1.0, 2.0], dtype=dtype)
        var1 = resource_variable_ops.ResourceVariable([4.0, 3.0], dtype=dtype)
        grads0 = constant_op.constant([0.1, 0.2], dtype=dtype)
        grads1 = constant_op.constant([0.01, 0.02], dtype=dtype)
        opt = ftrl.FtrlOptimizer(
            3.0,
            initial_accumulator_value=0.1,
            l1_regularization_strength=0.001,
            l2_regularization_strength=2.0)
        ftrl_update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
        variables.global_variables_initializer().run()
        # Fetch params to validate initial values
        self.assertAllClose([1.0, 2.0], var0.eval())
        self.assertAllClose([4.0, 3.0], var1.eval())

        # Run 10 steps FTRL
        for _ in range(10):
          ftrl_update.run()

        # Validate updated params
        self.assertAllClose(np.array([-0.24059935, -0.46829352]), var0.eval())
        self.assertAllClose(np.array([-0.02406147, -0.04830509]), var1.eval())

  # When variables are intialized with Zero, FTRL-Proximal has two properties:
  # 1. Without L1&L2 but with fixed learning rate, FTRL-Proximal is identical
  # with GradientDescent.
  # 2. Without L1&L2 but with adaptive learning rate, FTRL-Proximal is idential
  # with Adagrad.
  # So, basing on these two properties, we test if our implementation of
  # FTRL-Proximal performs same updates as Adagrad or GradientDescent.
  def testEquivAdagradwithoutRegularization(self):
    steps = 5
    for dtype in self.float_types:
      with self.test_session(), self.test_scope():
        val0, val1 = self.equivAdagradTest_FtrlPart(steps, dtype)
      with self.test_session(), self.test_scope():
        val2, val3 = self.equivAdagradTest_AdagradPart(steps, dtype)

    self.assertAllClose(val0, val2)
    self.assertAllClose(val1, val3)

  def testEquivGradientDescentwithoutRegularization(self):
    steps = 5
    for dtype in self.float_types:
      with self.test_session(), self.test_scope():
        val0, val1 = self.equivGradientDescentTest_FtrlPart(steps, dtype)
      with self.test_session(), self.test_scope():
        val2, val3 = self.equivGradientDescentTest_GradientDescentPart(
            steps, dtype)

    self.assertAllClose(val0, val2)
    self.assertAllClose(val1, val3)


if __name__ == "__main__":
  test.main()
