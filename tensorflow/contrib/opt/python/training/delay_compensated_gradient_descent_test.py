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
"""Tests for DelayCompensatedGradientDescentOptimizer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.opt.python.training import adamax
from tensorflow.contrib.opt.python.training import delay_compensated_gradient_descent
from tensorflow.python.client import session
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


def delay_compensated_gradient_descent_update_numpy(param,
                        g_t,
                        shadow,
                        alpha=0.01,
                        variance_parameter=2.0,
                        epsilon=1e-8):
  
  param_t = param - alpha * (g_t + variance_parameter * g_t * g_t * (param - shadow))
  shadow_t = param_t
  return param_t, shadow_t


def delay_compensated_gradient_descent_update_numpy_sparse(param,
                               indices,
                               g_t,
                               shadow,
                               alpha=0.01,
                               variance_parameter=2.0,
                               epsilon=1e-8):
  shadow_t, param_t = np.copy(shadow), np.copy(param)
  param_t_slice  = param[indices]
  shadow_t_slice = shadow[indices]

  param_t_slice  = param[indices] - alpha * (g_t + variance_parameter * g_t * g_t * (param_t_slice - shadow_t_slice))
  param_t[indices] = param_t_slice
  shadow_t[indices] = param_t_slice
  return param_t, shadow_t


class DelayCompensatedGradientDescentOptimizerTest(test.TestCase):

  def doTestSparse(self, use_resource=False):
    for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
      with self.cached_session():
        # Initialize variables for numpy implementation.
        zero_slots = lambda: np.zeros((3), dtype=dtype.as_numpy_dtype)
        shadow0, shadow1 = zero_slots(), zero_slots()
        var0_np = np.array([1.0, 2.0, 3.0], dtype=dtype.as_numpy_dtype)
        grads0_np = np.array([0.1, 0.1], dtype=dtype.as_numpy_dtype)
        var1_np = np.array([4.0, 5.0, 6.0], dtype=dtype.as_numpy_dtype)
        grads1_np = np.array([0.01, 0.01], dtype=dtype.as_numpy_dtype)

        if use_resource:
          var0 = resource_variable_ops.ResourceVariable(var0_np)
          var1 = resource_variable_ops.ResourceVariable(var1_np)
        else:
          var0 = variables.Variable(var0_np)
          var1 = variables.Variable(var1_np)
        grads0_np_indices = np.array([0, 1], dtype=np.int32)
        grads0 = ops.IndexedSlices(
            constant_op.constant(grads0_np),
            constant_op.constant(grads0_np_indices), constant_op.constant([2]))
        grads1_np_indices = np.array([2, 1], dtype=np.int32)
        grads1 = ops.IndexedSlices(
            constant_op.constant(grads1_np),
            constant_op.constant(grads1_np_indices), constant_op.constant([2]))
        opt = delay_compensated_gradient_descent.DelayCompensatedGradientDescentOptimizer(0.01)
        update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
        variables.global_variables_initializer().run()

        # Fetch params to validate initial values
        self.assertAllClose([1.0, 2.0, 3.0], var0.eval())
        self.assertAllClose([4.0, 5.0, 6.0], var1.eval())

        # Run 3 steps of DelayCompensatedGradientDescentOptimizer
        for t in range(1, 4):
          update.run()

          var0_np, shadow0 = delay_compensated_gradient_descent_update_numpy_sparse(
              var0_np, grads0_np_indices, grads0_np, shadow0)
          var1_np, shadow1 = delay_compensated_gradient_descent_update_numpy_sparse(
              var1_np, grads1_np_indices, grads1_np, shadow1)

          # Validate updated params
          self.assertAllCloseAccordingToType(var0_np, var0.eval())
          self.assertAllCloseAccordingToType(var1_np, var1.eval())

  def testSparse(self):
    self.doTestSparse(use_resource=False)

  def testResourceSparse(self):
    self.doTestSparse(use_resource=True)


  def testSparseRepeatedIndices(self):
    for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
      with self.cached_session():
        repeated_index_update_var = variables.Variable(
            [[1.0], [2.0]], dtype=dtype)
        aggregated_update_var = variables.Variable(
            [[1.0], [2.0]], dtype=dtype)
        grad_repeated_index = ops.IndexedSlices(
            constant_op.constant(
                [0.1, 0.1], shape=[2, 1], dtype=dtype),
            constant_op.constant([1, 1]),
            constant_op.constant([2, 1]))
        grad_aggregated = ops.IndexedSlices(
            constant_op.constant(
                [0.2], shape=[1, 1], dtype=dtype),
            constant_op.constant([1]),
            constant_op.constant([2, 1]))
        repeated_update = delay_compensated_gradient_descent.DelayCompensatedGradientDescentOptimizer(0.01).apply_gradients(
            [(grad_repeated_index, repeated_index_update_var)])
        aggregated_update = delay_compensated_gradient_descent.DelayCompensatedGradientDescentOptimizer(0.01).apply_gradients(
            [(grad_aggregated, aggregated_update_var)])
        variables.global_variables_initializer().run()
        self.assertAllClose(aggregated_update_var.eval(),
                            repeated_index_update_var.eval())
        for _ in range(3):
          repeated_update.run()
          aggregated_update.run()
          self.assertAllClose(aggregated_update_var.eval(),
                              repeated_index_update_var.eval())

  def doTestBasic(self, use_resource=False):
    for i, dtype in enumerate([dtypes.half, dtypes.float32, dtypes.float64]):
      with self.session(graph=ops.Graph()):
        # Initialize variables for numpy implementation.
        shadow0, shadow1 = 0.0, 0.0
        var0_np = np.array([1.0, 2.0], dtype=dtype.as_numpy_dtype)
        grads0_np = np.array([0.1, 0.1], dtype=dtype.as_numpy_dtype)
        var1_np = np.array([3.0, 4.0], dtype=dtype.as_numpy_dtype)
        grads1_np = np.array([0.01, 0.01], dtype=dtype.as_numpy_dtype)

        if use_resource:
          var0 = resource_variable_ops.ResourceVariable(
              var0_np, name="var0_%d" % i)
          var1 = resource_variable_ops.ResourceVariable(
              var1_np, name="var1_%d" % i)
        else:
          var0 = variables.Variable(var0_np)
          var1 = variables.Variable(var1_np)
        grads0 = constant_op.constant(grads0_np)
        grads1 = constant_op.constant(grads1_np)

        opt = delay_compensated_gradient_descent.DelayCompensatedGradientDescentOptimizer(0.01)
        update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))

        if not context.executing_eagerly():
          with ops.Graph().as_default():
            # Shouldn't return non-slot variables from other graphs.
            self.assertEqual(0, len(opt.variables()))

          self.evaluate(variables.global_variables_initializer())
          # Fetch params to validate initial values
          self.assertAllClose([1.0, 2.0], self.evaluate(var0))
          self.assertAllClose([3.0, 4.0], self.evaluate(var1))


        # Run 3 steps of DelayCompensatedGradientDescentOptimizer
        for t in range(1, 4):
          if not context.executing_eagerly():
            self.evaluate(update)
          elif t > 1:
            opt.apply_gradients(zip([grads0, grads1], [var0, var1]))


          var0_np, shadow0 = delay_compensated_gradient_descent_update_numpy(var0_np, grads0_np, shadow0)
          var1_np, shadow1 = delay_compensated_gradient_descent_update_numpy(var1_np, grads1_np, shadow1)

          # Validate updated params
          self.assertAllCloseAccordingToType(var0_np, self.evaluate(var0),
                                             rtol=1e-2)
          self.assertAllCloseAccordingToType(var1_np, self.evaluate(var1),
                                             rtol=1e-2)
          if use_resource:
            self.assertEqual("var0_%d/DelayCompensatedGradientDescentOptimizer:0" % (i,),
                             opt.get_slot(var=var0, name="shadow_0").name)

  def testBasic(self):
    with self.cached_session():
      self.doTestBasic(use_resource=False)

  @test_util.run_in_graph_and_eager_modes(reset_test=True)
  def testResourceBasic(self):
    self.doTestBasic(use_resource=True)

  def testTensorLearningRate(self):
    for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
      with self.cached_session():
        # Initialize variables for numpy implementation.
        shadow0, shadow1 = 0.0, 0.0
        var0_np = np.array([1.0, 2.0], dtype=dtype.as_numpy_dtype)
        grads0_np = np.array([0.1, 0.1], dtype=dtype.as_numpy_dtype)
        var1_np = np.array([3.0, 4.0], dtype=dtype.as_numpy_dtype)
        grads1_np = np.array([0.01, 0.01], dtype=dtype.as_numpy_dtype)

        var0 = variables.Variable(var0_np)
        var1 = variables.Variable(var1_np)
        grads0 = constant_op.constant(grads0_np)
        grads1 = constant_op.constant(grads1_np)
        opt = delay_compensated_gradient_descent.DelayCompensatedGradientDescentOptimizer(0.01)
        update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
        variables.global_variables_initializer().run()

        # Fetch params to validate initial values
        self.assertAllClose([1.0, 2.0], var0.eval())
        self.assertAllClose([3.0, 4.0], var1.eval())

        # Run 3 steps of DelayCompensatedGradientDescentOptimizer
        for t in range(1, 4):
          update.run()

          var0_np, shadow0 = delay_compensated_gradient_descent_update_numpy(var0_np, grads0_np, shadow0)
          var1_np, shadow1 = delay_compensated_gradient_descent_update_numpy(var1_np, grads1_np, shadow1)

          # Validate updated params
          self.assertAllCloseAccordingToType(var0_np, var0.eval())
          self.assertAllCloseAccordingToType(var1_np, var1.eval())

  def testSharing(self):
    for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
      with self.cached_session():
        # Initialize variables for numpy implementation.
        shadow0, shadow1 = 0.0, 0.0
        var0_np = np.array([1.0, 2.0], dtype=dtype.as_numpy_dtype)
        grads0_np = np.array([0.1, 0.1], dtype=dtype.as_numpy_dtype)
        var1_np = np.array([3.0, 4.0], dtype=dtype.as_numpy_dtype)
        grads1_np = np.array([0.01, 0.01], dtype=dtype.as_numpy_dtype)

        var0 = variables.Variable(var0_np)
        var1 = variables.Variable(var1_np)
        grads0 = constant_op.constant(grads0_np)
        grads1 = constant_op.constant(grads1_np)
        opt = delay_compensated_gradient_descent.DelayCompensatedGradientDescentOptimizer(0.01)
        update1 = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
        update2 = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
        variables.global_variables_initializer().run()
        # Fetch params to validate initial values
        self.assertAllClose([1.0, 2.0], var0.eval())
        self.assertAllClose([3.0, 4.0], var1.eval())

        # Run 3 steps of intertwined DelayCompensatedGradientDescentOptimizer1 and DelayCompensatedGradientDescentOptimizer2.
        for t in range(1, 4):
          if t % 2 == 0:
            update1.run()
          else:
            update2.run()

          var0_np, shadow0 = delay_compensated_gradient_descent_update_numpy(var0_np, grads0_np, shadow0)
          var1_np, shadow1 = delay_compensated_gradient_descent_update_numpy(var1_np, grads1_np, shadow1)

          # Validate updated params
          self.assertAllCloseAccordingToType(var0_np, var0.eval())
          self.assertAllCloseAccordingToType(var1_np, var1.eval())

  def testTwoSessions(self):
    optimizer = delay_compensated_gradient_descent.DelayCompensatedGradientDescentOptimizer(0.01)
    g = ops.Graph()
    with g.as_default():
      with session.Session():
        var0 = variables.Variable(np.array([1.0, 2.0]), name="v0")
        grads0 = constant_op.constant(np.array([0.1, 0.1]))
        optimizer.apply_gradients([(grads0, var0)])

    gg = ops.Graph()
    with gg.as_default():
      with session.Session():
        var0 = variables.Variable(np.array([1.0, 2.0]), name="v0")
        grads0 = constant_op.constant(np.array([0.1, 0.1]))

        # If the optimizer saves any state not keyed by graph the following line
        # fails.
        optimizer.apply_gradients([(grads0, var0)])

  def testSlotsUniqueEager(self):
    with context.eager_mode():
      v1 = resource_variable_ops.ResourceVariable(1.)
      v2 = resource_variable_ops.ResourceVariable(1.)
      opt = delay_compensated_gradient_descent.DelayCompensatedGradientDescentOptimizer(0.01)
      opt.minimize(lambda: v1 + v2)
      # There should be two unique slot variables
      # for v1 and v2 respectively.
      self.assertEqual(2, len(set(opt.variables())))


if __name__ == "__main__":
  test.main()
