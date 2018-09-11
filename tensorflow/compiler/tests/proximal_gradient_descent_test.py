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
"""Tests for Proximal Gradient Descent optimizer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import gradient_descent
from tensorflow.python.training import proximal_gradient_descent


class ProximalGradientDescentOptimizerTest(xla_test.XLATestCase):

  def testResourceProximalGradientDescentwithoutRegularization(self):
    with self.cached_session(), self.test_scope():
      var0 = resource_variable_ops.ResourceVariable([0.0, 0.0])
      var1 = resource_variable_ops.ResourceVariable([0.0, 0.0])
      grads0 = constant_op.constant([0.1, 0.2])
      grads1 = constant_op.constant([0.01, 0.02])
      opt = proximal_gradient_descent.ProximalGradientDescentOptimizer(
          3.0, l1_regularization_strength=0.0, l2_regularization_strength=0.0)
      update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
      variables.global_variables_initializer().run()

      self.assertAllClose([0.0, 0.0], var0.eval())
      self.assertAllClose([0.0, 0.0], var1.eval())

      # Run 3 steps Proximal Gradient Descent.
      for _ in range(3):
        update.run()

      self.assertAllClose(np.array([-0.9, -1.8]), var0.eval())
      self.assertAllClose(np.array([-0.09, -0.18]), var1.eval())

  def testProximalGradientDescentwithoutRegularization2(self):
    with self.cached_session(), self.test_scope():
      var0 = resource_variable_ops.ResourceVariable([1.0, 2.0])
      var1 = resource_variable_ops.ResourceVariable([4.0, 3.0])
      grads0 = constant_op.constant([0.1, 0.2])
      grads1 = constant_op.constant([0.01, 0.02])

      opt = proximal_gradient_descent.ProximalGradientDescentOptimizer(
          3.0, l1_regularization_strength=0.0, l2_regularization_strength=0.0)
      update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
      variables.global_variables_initializer().run()

      self.assertAllClose([1.0, 2.0], var0.eval())
      self.assertAllClose([4.0, 3.0], var1.eval())

      # Run 3 steps Proximal Gradient Descent
      for _ in range(3):
        update.run()

      self.assertAllClose(np.array([0.1, 0.2]), var0.eval())
      self.assertAllClose(np.array([3.91, 2.82]), var1.eval())

  def testProximalGradientDescentWithL1(self):
    with self.cached_session(), self.test_scope():
      var0 = resource_variable_ops.ResourceVariable([1.0, 2.0])
      var1 = resource_variable_ops.ResourceVariable([4.0, 3.0])
      grads0 = constant_op.constant([0.1, 0.2])
      grads1 = constant_op.constant([0.01, 0.02])

      opt = proximal_gradient_descent.ProximalGradientDescentOptimizer(
          3.0, l1_regularization_strength=0.001, l2_regularization_strength=0.0)
      update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
      variables.global_variables_initializer().run()

      self.assertAllClose([1.0, 2.0], var0.eval())
      self.assertAllClose([4.0, 3.0], var1.eval())

      # Run 10 steps proximal gradient descent.
      for _ in range(10):
        update.run()

      self.assertAllClose(np.array([-1.988, -3.988001]), var0.eval())
      self.assertAllClose(np.array([3.67, 2.37]), var1.eval())

  def testProximalGradientDescentWithL1_L2(self):
    with self.cached_session(), self.test_scope():
      var0 = resource_variable_ops.ResourceVariable([1.0, 2.0])
      var1 = resource_variable_ops.ResourceVariable([4.0, 3.0])
      grads0 = constant_op.constant([0.1, 0.2])
      grads1 = constant_op.constant([0.01, 0.02])

      opt = proximal_gradient_descent.ProximalGradientDescentOptimizer(
          3.0, l1_regularization_strength=0.001, l2_regularization_strength=2.0)
      update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
      variables.global_variables_initializer().run()

      self.assertAllClose([1.0, 2.0], var0.eval())
      self.assertAllClose([4.0, 3.0], var1.eval())

      # Run 10 steps Proximal Gradient Descent
      for _ in range(10):
        update.run()

      self.assertAllClose(np.array([-0.0495, -0.0995]), var0.eval())
      self.assertAllClose(np.array([-0.0045, -0.0095]), var1.eval())

  def applyOptimizer(self, opt, steps=5):
    var0 = resource_variable_ops.ResourceVariable([1.0, 2.0])
    var1 = resource_variable_ops.ResourceVariable([3.0, 4.0])
    grads0 = constant_op.constant([0.1, 0.2])
    grads1 = constant_op.constant([0.01, 0.02])

    update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
    variables.global_variables_initializer().run()

    self.assertAllClose([1.0, 2.0], var0.eval())
    self.assertAllClose([3.0, 4.0], var1.eval())

    # Run ProximalAdagrad for a few steps
    for _ in range(steps):
      update.run()

    return var0.eval(), var1.eval()

  def testEquivGradientDescentwithoutRegularization(self):
    with self.cached_session(), self.test_scope():
      val0, val1 = self.applyOptimizer(
          proximal_gradient_descent.ProximalGradientDescentOptimizer(
              3.0,
              l1_regularization_strength=0.0,
              l2_regularization_strength=0.0))

    with self.cached_session(), self.test_scope():
      val2, val3 = self.applyOptimizer(
          gradient_descent.GradientDescentOptimizer(3.0))

    self.assertAllClose(val0, val2)
    self.assertAllClose(val1, val3)


if __name__ == "__main__":
  test.main()
