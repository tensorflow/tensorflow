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
"""Functional tests for Proximal Gradient Descent operations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import gradient_descent
from tensorflow.python.training import proximal_gradient_descent


class ProximalGradientDescentOptimizerTest(test.TestCase):

  def doTestProximalGradientDescentwithoutRegularization(
      self, use_resource=False):
    with self.test_session() as sess:
      if use_resource:
        var0 = resource_variable_ops.ResourceVariable([0.0, 0.0])
        var1 = resource_variable_ops.ResourceVariable([0.0, 0.0])
      else:
        var0 = variables.Variable([0.0, 0.0])
        var1 = variables.Variable([0.0, 0.0])
      grads0 = constant_op.constant([0.1, 0.2])
      grads1 = constant_op.constant([0.01, 0.02])
      opt = proximal_gradient_descent.ProximalGradientDescentOptimizer(
          3.0, l1_regularization_strength=0.0, l2_regularization_strength=0.0)
      update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
      variables.global_variables_initializer().run()

      v0_val, v1_val = sess.run([var0, var1])
      self.assertAllClose([0.0, 0.0], v0_val)
      self.assertAllClose([0.0, 0.0], v1_val)

      # Run 3 steps Proximal Gradient Descent.
      for _ in range(3):
        update.run()

      v0_val, v1_val = sess.run([var0, var1])
      self.assertAllClose(np.array([-0.9, -1.8]), v0_val)
      self.assertAllClose(np.array([-0.09, -0.18]), v1_val)

  def testProximalGradientDescentwithoutRegularization(self):
    self.doTestProximalGradientDescentwithoutRegularization(use_resource=False)

  def testResourceProximalGradientDescentwithoutRegularization(self):
    self.doTestProximalGradientDescentwithoutRegularization(use_resource=True)

  def testProximalGradientDescentwithoutRegularization2(self):
    with self.test_session() as sess:
      var0 = variables.Variable([1.0, 2.0])
      var1 = variables.Variable([4.0, 3.0])
      grads0 = constant_op.constant([0.1, 0.2])
      grads1 = constant_op.constant([0.01, 0.02])

      opt = proximal_gradient_descent.ProximalGradientDescentOptimizer(
          3.0, l1_regularization_strength=0.0, l2_regularization_strength=0.0)
      update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
      variables.global_variables_initializer().run()

      v0_val, v1_val = sess.run([var0, var1])
      self.assertAllClose([1.0, 2.0], v0_val)
      self.assertAllClose([4.0, 3.0], v1_val)

      # Run 3 steps Proximal Gradient Descent
      for _ in range(3):
        update.run()

      v0_val, v1_val = sess.run([var0, var1])
      self.assertAllClose(np.array([0.1, 0.2]), v0_val)
      self.assertAllClose(np.array([3.91, 2.82]), v1_val)

  def testMinimizeSparseResourceVariable(self):
    for dtype in [dtypes.float32, dtypes.float64]:
      with self.test_session():
        var0 = resource_variable_ops.ResourceVariable([[1.0, 2.0]], dtype=dtype)
        x = constant_op.constant([[4.0], [5.0]], dtype=dtype)
        pred = math_ops.matmul(embedding_ops.embedding_lookup([var0], [0]), x)
        loss = pred * pred
        sgd_op = proximal_gradient_descent.ProximalGradientDescentOptimizer(
            1.0).minimize(loss)
        variables.global_variables_initializer().run()
        # Fetch params to validate initial values
        self.assertAllCloseAccordingToType([[1.0, 2.0]], var0.eval())
        # Run 1 step of sgd
        sgd_op.run()
        # Validate updated params
        self.assertAllCloseAccordingToType(
            [[-111, -138]], var0.eval(), atol=0.01)

  def testProximalGradientDescentWithL1_L2(self):
    with self.test_session() as sess:
      var0 = variables.Variable([1.0, 2.0])
      var1 = variables.Variable([4.0, 3.0])
      grads0 = constant_op.constant([0.1, 0.2])
      grads1 = constant_op.constant([0.01, 0.02])

      opt = proximal_gradient_descent.ProximalGradientDescentOptimizer(
          3.0, l1_regularization_strength=0.001, l2_regularization_strength=2.0)
      update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
      variables.global_variables_initializer().run()

      v0_val, v1_val = sess.run([var0, var1])
      self.assertAllClose([1.0, 2.0], v0_val)
      self.assertAllClose([4.0, 3.0], v1_val)

      # Run 10 steps Proximal Gradient Descent
      for _ in range(10):
        update.run()

      v0_val, v1_val = sess.run([var0, var1])
      self.assertAllClose(np.array([0.037125, 0.074625]), v0_val)
      self.assertAllClose(np.array([0.003375, 0.007125]), v1_val)

  def applyOptimizer(self, opt, steps=5, is_sparse=False):
    if is_sparse:
      var0 = variables.Variable([[1.0], [2.0]])
      var1 = variables.Variable([[3.0], [4.0]])
      grads0 = ops.IndexedSlices(
          constant_op.constant(
              [0.1], shape=[1, 1]),
          constant_op.constant([0]),
          constant_op.constant([2, 1]))
      grads1 = ops.IndexedSlices(
          constant_op.constant(
              [0.02], shape=[1, 1]),
          constant_op.constant([1]),
          constant_op.constant([2, 1]))
    else:
      var0 = variables.Variable([1.0, 2.0])
      var1 = variables.Variable([3.0, 4.0])
      grads0 = constant_op.constant([0.1, 0.2])
      grads1 = constant_op.constant([0.01, 0.02])

    update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
    variables.global_variables_initializer().run()

    sess = ops.get_default_session()
    v0_val, v1_val = sess.run([var0, var1])
    if is_sparse:
      self.assertAllClose([[1.0], [2.0]], v0_val)
      self.assertAllClose([[3.0], [4.0]], v1_val)
    else:
      self.assertAllClose([1.0, 2.0], v0_val)
      self.assertAllClose([3.0, 4.0], v1_val)

    # Run ProximalAdagrad for a few steps
    for _ in range(steps):
      update.run()

    v0_val, v1_val = sess.run([var0, var1])
    return v0_val, v1_val

  def testEquivSparseGradientDescentwithoutRegularization(self):
    with self.test_session():
      val0, val1 = self.applyOptimizer(
          proximal_gradient_descent.ProximalGradientDescentOptimizer(
              3.0,
              l1_regularization_strength=0.0,
              l2_regularization_strength=0.0),
          is_sparse=True)

    with self.test_session():
      val2, val3 = self.applyOptimizer(
          gradient_descent.GradientDescentOptimizer(3.0), is_sparse=True)

    self.assertAllClose(val0, val2)
    self.assertAllClose(val1, val3)

  def testEquivGradientDescentwithoutRegularization(self):
    with self.test_session():
      val0, val1 = self.applyOptimizer(
          proximal_gradient_descent.ProximalGradientDescentOptimizer(
              3.0,
              l1_regularization_strength=0.0,
              l2_regularization_strength=0.0))

    with self.test_session():
      val2, val3 = self.applyOptimizer(
          gradient_descent.GradientDescentOptimizer(3.0))

    self.assertAllClose(val0, val2)
    self.assertAllClose(val1, val3)


if __name__ == "__main__":
  test.main()
