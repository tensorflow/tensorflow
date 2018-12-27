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
"""Functional test for GradientDescent."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.keras.optimizer_v2 import gradient_descent
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


class GradientDescentOptimizerTest(test.TestCase):

  @test_util.run_in_graph_and_eager_modes
  def testBasic(self):
    for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
      with self.cached_session():
        var0 = resource_variable_ops.ResourceVariable([1.0, 2.0], dtype=dtype)
        var1 = resource_variable_ops.ResourceVariable([3.0, 4.0], dtype=dtype)
        grads0 = constant_op.constant([0.1, 0.1], dtype=dtype)
        grads1 = constant_op.constant([0.01, 0.01], dtype=dtype)
        sgd = gradient_descent.SGD(3.0)
        sgd_op = sgd.apply_gradients(zip([grads0, grads1], [var0, var1]))
        self.evaluate(variables.global_variables_initializer())
        # Run 1 step of sgd
        self.evaluate(sgd_op)
        # Validate updated params
        self.assertAllCloseAccordingToType([1.0 - 3.0 * 0.1, 2.0 - 3.0 * 0.1],
                                           self.evaluate(var0))
        self.assertAllCloseAccordingToType([3.0 - 3.0 * 0.01, 4.0 - 3.0 * 0.01],
                                           self.evaluate(var1))

  @test_util.run_in_graph_and_eager_modes
  def testBasicWithLearningRateDecay(self):
    for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
      with self.cached_session():
        var0 = resource_variable_ops.ResourceVariable([1.0, 2.0], dtype=dtype)
        var1 = resource_variable_ops.ResourceVariable([3.0, 4.0], dtype=dtype)
        grads0 = constant_op.constant([0.1, 0.1], dtype=dtype)
        grads1 = constant_op.constant([0.01, 0.01], dtype=dtype)
        learning_rate = 3.0
        decay = 0.5
        sgd = gradient_descent.SGD(learning_rate=learning_rate, decay=decay)
        if not context.executing_eagerly():
          sgd_op = sgd.apply_gradients(zip([grads0, grads1], [var0, var1]))
        self.evaluate(variables.global_variables_initializer())
        # Run 2 steps of sgd
        if not context.executing_eagerly():
          self.evaluate(sgd_op)
        else:
          sgd.apply_gradients(zip([grads0, grads1], [var0, var1]))
        # Validate updated params
        self.assertAllCloseAccordingToType([1.0 - 3.0 * 0.1, 2.0 - 3.0 * 0.1],
                                           self.evaluate(var0))
        self.assertAllCloseAccordingToType([3.0 - 3.0 * 0.01, 4.0 - 3.0 * 0.01],
                                           self.evaluate(var1))

        if not context.executing_eagerly():
          self.evaluate(sgd_op)
        else:
          sgd.apply_gradients(zip([grads0, grads1], [var0, var1]))
        # Validate updated params
        self.assertAllCloseAccordingToType(
            [1.0 - 3.0 * 0.1 - 2.0 * 0.1, 2.0 - 3.0 * 0.1 - 2.0 * 0.1],
            self.evaluate(var0))
        self.assertAllCloseAccordingToType(
            [3.0 - 3.0 * 0.01 - 2.0 * 0.01, 4.0 - 3.0 * 0.01 - 2.0 * 0.01],
            self.evaluate(var1))

  @test_util.run_in_graph_and_eager_modes
  def testBasicCallableParams(self):
    for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
      with self.cached_session():
        var0 = resource_variable_ops.ResourceVariable([1.0, 2.0], dtype=dtype)
        var1 = resource_variable_ops.ResourceVariable([3.0, 4.0], dtype=dtype)
        grads0 = constant_op.constant([0.1, 0.1], dtype=dtype)
        grads1 = constant_op.constant([0.01, 0.01], dtype=dtype)
        lr = lambda: 3.0
        sgd = gradient_descent.SGD(lr)
        sgd_op = sgd.apply_gradients(zip([grads0, grads1], [var0, var1]))
        self.evaluate(variables.global_variables_initializer())
        # Run 1 step of sgd
        self.evaluate(sgd_op)
        # Validate updated params
        self.assertAllCloseAccordingToType([1.0 - 3.0 * 0.1, 2.0 - 3.0 * 0.1],
                                           self.evaluate(var0))
        self.assertAllCloseAccordingToType([3.0 - 3.0 * 0.01, 4.0 - 3.0 * 0.01],
                                           self.evaluate(var1))

  @test_util.run_in_graph_and_eager_modes
  def testMinimizeResourceVariable(self):
    for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
      with self.cached_session():
        var0 = resource_variable_ops.ResourceVariable([[1.0, 2.0]], dtype=dtype)
        var1 = resource_variable_ops.ResourceVariable([3.0], dtype=dtype)
        x = constant_op.constant([[4.0], [5.0]], dtype=dtype)
        loss = lambda: math_ops.matmul(var0, x) + var1  # pylint: disable=cell-var-from-loop
        sgd = gradient_descent.SGD(1.0)
        sgd_op = sgd.minimize(loss, [var0, var1])
        self.evaluate(variables.global_variables_initializer())
        # Run 1 step of sgd
        self.evaluate(sgd_op)
        # Validate updated params
        self.assertAllCloseAccordingToType([[1.0 - 4.0, 2.0 - 5.0]],
                                           self.evaluate(var0))
        self.assertAllCloseAccordingToType([3.0 - 1.0], self.evaluate(var1))

  @test_util.run_deprecated_v1
  def testMinimizeSparseResourceVariable(self):
    for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
      with self.cached_session():
        var0 = resource_variable_ops.ResourceVariable([[1.0, 2.0]], dtype=dtype)
        var1 = resource_variable_ops.ResourceVariable([3.0], dtype=dtype)
        x = constant_op.constant([[4.0], [5.0]], dtype=dtype)

        def loss():
          pred = math_ops.matmul(embedding_ops.embedding_lookup([var0], [0]), x)  # pylint: disable=cell-var-from-loop
          pred += var1  # pylint: disable=cell-var-from-loop
          return pred * pred

        sgd_op = gradient_descent.SGD(1.0).minimize(loss, [var0, var1])
        self.evaluate(variables.global_variables_initializer())
        # Run 1 step of sgd
        self.evaluate(sgd_op)
        # Validate updated params
        np_pred = 1.0 * 4.0 + 2.0 * 5.0 + 3.0
        np_grad = 2 * np_pred
        self.assertAllCloseAccordingToType(
            [[1.0 - np_grad * 4.0, 2.0 - np_grad * 5.0]], self.evaluate(var0))
        self.assertAllCloseAccordingToType([3.0 - np_grad], self.evaluate(var1))

  def testTensorLearningRate(self):
    for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
      with self.cached_session():
        var0 = variables.Variable([1.0, 2.0], dtype=dtype)
        var1 = variables.Variable([3.0, 4.0], dtype=dtype)
        grads0 = constant_op.constant([0.1, 0.1], dtype=dtype)
        grads1 = constant_op.constant([0.01, 0.01], dtype=dtype)
        lrate = constant_op.constant(3.0)
        sgd_op = gradient_descent.SGD(lrate).apply_gradients(
            zip([grads0, grads1], [var0, var1]))
        self.evaluate(variables.global_variables_initializer())
        # Run 1 step of sgd
        self.evaluate(sgd_op)
        # Validate updated params
        self.assertAllCloseAccordingToType([1.0 - 3.0 * 0.1, 2.0 - 3.0 * 0.1],
                                           self.evaluate(var0))
        self.assertAllCloseAccordingToType([3.0 - 3.0 * 0.01, 4.0 - 3.0 * 0.01],
                                           self.evaluate(var1))

  @test_util.run_deprecated_v1
  def testGradWrtRef(self):
    for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
      with self.cached_session():
        opt = gradient_descent.SGD(3.0)
        values = [1.0, 3.0]
        vars_ = [variables.Variable([v], dtype=dtype) for v in values]
        loss = lambda: vars_[0] + vars_[1]  # pylint: disable=cell-var-from-loop
        grads_and_vars = opt._compute_gradients(loss, vars_)
        self.evaluate(variables.global_variables_initializer())
        for grad, _ in grads_and_vars:
          self.assertAllCloseAccordingToType([1.0], self.evaluate(grad))

  def testSparseBasic(self):
    for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
      with self.cached_session():
        var0 = variables.Variable([[1.0], [2.0]], dtype=dtype)
        var1 = variables.Variable([[3.0], [4.0]], dtype=dtype)
        grads0 = ops.IndexedSlices(
            constant_op.constant([0.1], shape=[1, 1], dtype=dtype),
            constant_op.constant([0]), constant_op.constant([2, 1]))
        grads1 = ops.IndexedSlices(
            constant_op.constant([0.01], shape=[1, 1], dtype=dtype),
            constant_op.constant([1]), constant_op.constant([2, 1]))
        sgd_op = gradient_descent.SGD(3.0).apply_gradients(
            zip([grads0, grads1], [var0, var1]))
        self.evaluate(variables.global_variables_initializer())
        # Run 1 step of sgd
        self.evaluate(sgd_op)
        # Validate updated params
        self.assertAllCloseAccordingToType([[1.0 - 3.0 * 0.1], [2.0]],
                                           self.evaluate(var0))
        self.assertAllCloseAccordingToType([[3.0], [4.0 - 3.0 * 0.01]],
                                           self.evaluate(var1))

  @test_util.run_deprecated_v1
  def testSparseBasicWithLearningRateDecay(self):
    for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
      with self.cached_session():
        var0 = variables.Variable([[1.0], [2.0]], dtype=dtype)
        var1 = variables.Variable([[3.0], [4.0]], dtype=dtype)
        grads0 = ops.IndexedSlices(
            constant_op.constant([0.1], shape=[1, 1], dtype=dtype),
            constant_op.constant([0]), constant_op.constant([2, 1]))
        grads1 = ops.IndexedSlices(
            constant_op.constant([0.01], shape=[1, 1], dtype=dtype),
            constant_op.constant([1]), constant_op.constant([2, 1]))
        sgd_op = gradient_descent.SGD(
            3.0, decay=0.5).apply_gradients(
                zip([grads0, grads1], [var0, var1]))
        self.evaluate(variables.global_variables_initializer())
        # Run 2 steps of sgd
        self.evaluate(sgd_op)
        # Validate updated params
        self.assertAllCloseAccordingToType([[1.0 - 3.0 * 0.1], [2.0]],
                                           self.evaluate(var0))
        self.assertAllCloseAccordingToType([[3.0], [4.0 - 3.0 * 0.01]],
                                           self.evaluate(var1))

        self.evaluate(sgd_op)
        # Validate updated params
        self.assertAllCloseAccordingToType(
            [[1.0 - 3.0 * 0.1 - 2.0 * 0.1], [2.0]], self.evaluate(var0))
        self.assertAllCloseAccordingToType(
            [[3.0], [4.0 - 3.0 * 0.01 - 2.0 * 0.01]], self.evaluate(var1))

  def testCapturingInDefunWhileExecutingEagerly(self):
    with context.eager_mode():
      optimizer = gradient_descent.SGD(1.0)

      def step():
        self.v = resource_variable_ops.ResourceVariable(1.0)
        with backprop.GradientTape() as tape:
          loss = self.v**2
        grad = tape.gradient(loss, self.v)
        optimizer.apply_gradients([(grad, self.v)])
        return self.v.read_value()

      compiled_step = function.defun(step)

      self.assertEqual(float(step()), -1.0)
      self.assertEqual(float(compiled_step()), -1.0)
      # This shouldn't fail; in particular, the learning rate tensor should
      # be an EagerTensor once again, not a graph Tensor.
      self.assertEqual(float(step()), -1.0)

  def testConstructSGDWithLR(self):
    opt = gradient_descent.SGD(lr=1.0)
    self.assertEqual(opt.lr, 1.0)
    opt_2 = gradient_descent.SGD(learning_rate=0.1, lr=1.0)
    self.assertEqual(opt_2.lr, 1.0)
    opt_3 = gradient_descent.SGD(learning_rate=0.1)
    self.assertEqual(opt_3.lr, 0.1)


class MomentumOptimizerTest(test.TestCase):

  def _update_nesterov_momentum_numpy(self, var, accum, g, lr, momentum):
    accum = accum * momentum - g * lr
    var += (accum * momentum - g * lr)
    return var, accum

  @test_util.run_in_graph_and_eager_modes
  def testBasic(self):
    for _, dtype in enumerate([dtypes.half, dtypes.float32, dtypes.float64]):
      with self.cached_session():
        var0 = resource_variable_ops.ResourceVariable([1.0, 2.0],
                                                      dtype=dtype,
                                                      name="var0")
        var1 = resource_variable_ops.ResourceVariable([3.0, 4.0],
                                                      dtype=dtype,
                                                      name="var1")
        grads0 = constant_op.constant([0.1, 0.1], dtype=dtype)
        grads1 = constant_op.constant([0.01, 0.01], dtype=dtype)
        learning_rate = 2.0
        momentum = 0.9
        mom_opt = gradient_descent.SGD(
            learning_rate=learning_rate, momentum=momentum)
        # self.assertFalse(mom_opt._initial_decay)
        mom_update = mom_opt.apply_gradients(
            zip([grads0, grads1], [var0, var1]))

        # Check we have slots
        slot0 = mom_opt.get_slot(var0, "momentum")
        self.assertEqual(slot0.get_shape(), var0.get_shape())
        slot1 = mom_opt.get_slot(var1, "momentum")
        self.assertEqual(slot1.get_shape(), var1.get_shape())

        # Step 1: the momentum accumulators where 0. So we should see a normal
        # update: v -= grad * learning_rate
        self.evaluate(variables.global_variables_initializer())
        self.evaluate(mom_update)
        # Check that the momentum accumulators have been updated.
        self.assertAllCloseAccordingToType(
            np.array([-0.2, -0.2]), self.evaluate(slot0))
        self.assertAllCloseAccordingToType(
            np.array([-0.02, -0.02]), self.evaluate(slot1))
        # Check that the parameters have been updated.
        self.assertAllCloseAccordingToType(
            np.array([1.0 - (0.1 * 2.0), 2.0 - (0.1 * 2.0)]),
            self.evaluate(var0))
        self.assertAllCloseAccordingToType(
            np.array([3.0 - (0.01 * 2.0), 4.0 - (0.01 * 2.0)]),
            self.evaluate(var1))
        # Step 2: the momentum accumulators contain the previous update.
        self.evaluate(mom_update)
        if context.executing_eagerly():
          mom_opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
        # Check that the momentum accumulators have been updated.
        self.assertAllCloseAccordingToType(
            np.array([(0.9 * (-0.2) - 2.0 * 0.1), (0.9 * (-0.2) - 2.0 * 0.1)]),
            self.evaluate(slot0))
        self.assertAllCloseAccordingToType(
            np.array([(0.9 * (-0.02) - 2.0 * 0.01),
                      (0.9 * (-0.02) - 2.0 * 0.01)]), self.evaluate(slot1))
        # Check that the parameters have been updated.
        self.assertAllCloseAccordingToType(
            np.array([
                1.0 - (0.1 * 2.0) - ((0.9 * 0.1 + 0.1) * 2.0),
                2.0 - (0.1 * 2.0) - ((0.9 * 0.1 + 0.1) * 2.0)
            ]), self.evaluate(var0))
        self.assertAllCloseAccordingToType(
            np.array([
                2.98 - ((0.9 * 0.01 + 0.01) * 2.0),
                3.98 - ((0.9 * 0.01 + 0.01) * 2.0)
            ]), self.evaluate(var1))

  @test_util.run_deprecated_v1
  def testNesterovMomentum(self):
    for dtype in [dtypes.float32, dtypes.float64]:
      with self.cached_session():
        var0 = resource_variable_ops.ResourceVariable([1.0, 2.0],
                                                      dtype=dtype,
                                                      name="var0")
        var1 = resource_variable_ops.ResourceVariable([3.0, 4.0],
                                                      dtype=dtype,
                                                      name="var1")
        var0_np = np.array([1.0, 2.0], dtype=dtype.as_numpy_dtype)
        var1_np = np.array([3.0, 4.0], dtype=dtype.as_numpy_dtype)
        accum0_np = np.array([0.0, 0.0], dtype=dtype.as_numpy_dtype)
        accum1_np = np.array([0.0, 0.0], dtype=dtype.as_numpy_dtype)
        loss = lambda: 5 * var0 * var0 + 3 * var1  # pylint: disable=cell-var-from-loop
        mom_op = gradient_descent.SGD(
            learning_rate=2.0, momentum=0.9, nesterov=True)
        opt_op = mom_op.minimize(loss, [var0, var1])
        variables.global_variables_initializer().run()
        for _ in range(1, 5):
          opt_op.run()
          var0_np, accum0_np = self._update_nesterov_momentum_numpy(
              var0_np, accum0_np, var0_np * 10, 2.0, 0.9)
          var1_np, accum1_np = self._update_nesterov_momentum_numpy(
              var1_np, accum1_np, 3, 2.0, 0.9)
          self.assertAllClose(var0_np, self.evaluate(var0))
          self.assertAllClose(var1_np, self.evaluate(var1))

  @test_util.run_deprecated_v1
  def testSparseNesterovMomentum(self):
    for dtype in [dtypes.float32, dtypes.float64]:
      with self.cached_session():
        var0_np = np.array([1.0, 2.0], dtype=dtype.as_numpy_dtype)
        var1_np = np.array([3.0, 4.0], dtype=dtype.as_numpy_dtype)
        accum0_np = np.array([0.0, 0.0], dtype=dtype.as_numpy_dtype)
        accum1_np = np.array([0.0, 0.0], dtype=dtype.as_numpy_dtype)
        grads = []
        for t in range(1, 5):
          grads.append(var0_np * 10)
          var0_np, accum0_np = self._update_nesterov_momentum_numpy(
              var0_np, accum0_np, var0_np * 10, 2.0, 0.9)
          var1_np, accum1_np = self._update_nesterov_momentum_numpy(
              var1_np, accum1_np, 3, 2.0, 0.9)
        var0_np = np.array([1.0, 2.0], dtype=dtype.as_numpy_dtype)
        var1_np = np.array([3.0, 4.0], dtype=dtype.as_numpy_dtype)
        accum0_np = np.array([0.0, 0.0], dtype=dtype.as_numpy_dtype)
        accum1_np = np.array([0.0, 0.0], dtype=dtype.as_numpy_dtype)
        var0 = resource_variable_ops.ResourceVariable(
            var0_np, dtype=dtype, name="var0")
        var1 = resource_variable_ops.ResourceVariable(
            var1_np, dtype=dtype, name="var1")
        mom_op = gradient_descent.SGD(
            learning_rate=2.0, momentum=0.9, nesterov=True)
        x_feed = array_ops.placeholder(dtype)
        y_feed = ops.IndexedSlices(x_feed, constant_op.constant([0, 1]),
                                   constant_op.constant([2]))
        grads_and_vars = [(y_feed, var0),
                          (constant_op.constant([3.0, 3.0], dtype=dtype), var1)]
        opt_update = mom_op.apply_gradients(grads_and_vars)
        variables.global_variables_initializer().run()
        for t in range(1, 5):
          opt_update.run(feed_dict={x_feed: grads[t - 1]})
          var0_np, accum0_np = self._update_nesterov_momentum_numpy(
              var0_np, accum0_np, var0_np * 10, 2.0, 0.9)
          var1_np, accum1_np = self._update_nesterov_momentum_numpy(
              var1_np, accum1_np, 3, 2.0, 0.9)
          self.assertAllClose(var0_np, self.evaluate(var0))
          self.assertAllClose(var1_np, self.evaluate(var1))

  @test_util.run_in_graph_and_eager_modes
  @test_util.run_deprecated_v1
  def testMinimizeSparseResourceVariable(self):
    for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
      # This test invokes the ResourceSparseApplyMomentum operation, which
      # did not have a registered GPU kernel as of April 2018. With graph
      # execution, the placement algorithm notices this and automatically
      # places the variable in CPU (host) memory. With eager execution,
      # the variable would be placed in GPU memory if available, which
      # would then conflict with the future invocation of the
      # ResourceSparseApplyMomentum operation.
      # To work around this discrepancy, for now we force the variable
      # to be placed on CPU.
      with ops.device("/cpu:0"):
        var0 = resource_variable_ops.ResourceVariable([[1.0, 2.0]], dtype=dtype)

      # pylint: disable=cell-var-from-loop
      def loss():
        x = constant_op.constant([[4.0], [5.0]], dtype=dtype)
        pred = math_ops.matmul(embedding_ops.embedding_lookup([var0], [0]), x)
        return pred * pred

      # pylint: enable=cell-var-from-loop

      opt = gradient_descent.SGD(learning_rate=1.0, momentum=0.0)
      sgd_op = opt.minimize(loss, [var0])
      self.evaluate(variables.global_variables_initializer())
      # Run 1 step of sgd
      self.evaluate(sgd_op)
      # Validate updated params
      self.assertAllCloseAccordingToType([[-111, -138]], self.evaluate(var0))

  @test_util.run_in_graph_and_eager_modes(reset_test=True)
  def testMinimizeWith2DIndicesForEmbeddingLookup(self):
    # This test invokes the ResourceSparseApplyMomentum operation, which
    # did not have a registered GPU kernel as of April 2018. With graph
    # execution, the placement algorithm notices this and automatically
    # places the variable in CPU (host) memory. With eager execution,
    # the variable would be placed in GPU memory if available, which
    # would then conflict with the future invocation of the
    # ResourceSparseApplyMomentum operation.
    # To work around this discrepancy, for now we force the variable
    # to be placed on CPU.
    with ops.device("/cpu:0"):
      var0 = resource_variable_ops.ResourceVariable(array_ops.ones([2, 2]))

    def loss():
      return math_ops.reduce_sum(embedding_ops.embedding_lookup(var0, [[1]]))

    opt = gradient_descent.SGD(learning_rate=1.0, momentum=0.0)
    sgd_op = opt.minimize(loss, [var0])
    self.evaluate(variables.global_variables_initializer())
    self.evaluate(sgd_op)
    self.assertAllCloseAccordingToType([[1, 1], [0, 0]], self.evaluate(var0))

  @test_util.run_deprecated_v1
  def testTensorLearningRateAndMomentum(self):
    for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
      with self.cached_session():
        var0 = variables.Variable([1.0, 2.0], dtype=dtype)
        var1 = variables.Variable([3.0, 4.0], dtype=dtype)
        grads0 = constant_op.constant([0.1, 0.1], dtype=dtype)
        grads1 = constant_op.constant([0.01, 0.01], dtype=dtype)
        mom_opt = gradient_descent.SGD(
            learning_rate=constant_op.constant(2.0),
            momentum=constant_op.constant(0.9))
        mom_update = mom_opt.apply_gradients(
            zip([grads0, grads1], [var0, var1]))
        variables.global_variables_initializer().run()
        # Check we have slots
        slot0 = mom_opt.get_slot(var0, "momentum")
        self.assertEqual(slot0.get_shape(), var0.get_shape())
        slot1 = mom_opt.get_slot(var1, "momentum")
        self.assertEqual(slot1.get_shape(), var1.get_shape())

        # Fetch params to validate initial values
        self.assertAllClose([1.0, 2.0], self.evaluate(var0))
        self.assertAllClose([3.0, 4.0], self.evaluate(var1))
        # Step 1: the momentum accumulators where 0. So we should see a normal
        # update: v -= grad * learning_rate
        mom_update.run()
        # Check that the momentum accumulators have been updated.
        self.assertAllCloseAccordingToType(
            np.array([-0.2, -0.2]), self.evaluate(slot0))
        self.assertAllCloseAccordingToType(
            np.array([-0.02, -0.02]), self.evaluate(slot1))
        # Check that the parameters have been updated.
        self.assertAllCloseAccordingToType(
            np.array([1.0 - (0.1 * 2.0), 2.0 - (0.1 * 2.0)]),
            self.evaluate(var0))
        self.assertAllCloseAccordingToType(
            np.array([3.0 - (0.01 * 2.0), 4.0 - (0.01 * 2.0)]),
            self.evaluate(var1))
        # Step 2: the momentum accumulators contain the previous update.
        mom_update.run()
        # Check that the momentum accumulators have been updated.
        self.assertAllCloseAccordingToType(
            np.array([(0.9 * (-0.2) - 2.0 * 0.1), (0.9 * (-0.2) - 2.0 * 0.1)]),
            self.evaluate(slot0))
        self.assertAllCloseAccordingToType(
            np.array([(0.9 * (-0.02) - 2.0 * 0.01),
                      (0.9 * (-0.02) - 2.0 * 0.01)]), self.evaluate(slot1))
        # Check that the parameters have been updated.
        self.assertAllCloseAccordingToType(
            np.array([
                1.0 - (0.1 * 2.0) - ((0.9 * 0.1 + 0.1) * 2.0),
                2.0 - (0.1 * 2.0) - ((0.9 * 0.1 + 0.1) * 2.0)
            ]), self.evaluate(var0))
        self.assertAllCloseAccordingToType(
            np.array([
                2.98 - ((0.9 * 0.01 + 0.01) * 2.0),
                3.98 - ((0.9 * 0.01 + 0.01) * 2.0)
            ]), self.evaluate(var1))

  @test_util.run_deprecated_v1
  def testSparse(self):
    for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
      with self.cached_session():
        var0 = variables.Variable(array_ops.zeros([4, 2], dtype=dtype))
        var1 = variables.Variable(constant_op.constant(1.0, dtype, [4, 2]))
        grads0 = ops.IndexedSlices(
            constant_op.constant([[.1, .1]], dtype=dtype),
            constant_op.constant([1]), constant_op.constant([4, 2]))
        grads1 = ops.IndexedSlices(
            constant_op.constant([[.01, .01], [.01, .01]], dtype=dtype),
            constant_op.constant([2, 3]), constant_op.constant([4, 2]))
        mom_opt = gradient_descent.SGD(learning_rate=2.0, momentum=0.9)
        mom_update = mom_opt.apply_gradients(
            zip([grads0, grads1], [var0, var1]))
        variables.global_variables_initializer().run()

        # Check we have slots
        slot0 = mom_opt.get_slot(var0, "momentum")
        self.assertEqual(slot0.get_shape(), var0.get_shape())
        slot1 = mom_opt.get_slot(var1, "momentum")
        self.assertEqual(slot1.get_shape(), var1.get_shape())

        # Fetch params to validate initial values
        self.assertAllClose([0, 0], self.evaluate(var0)[0])
        self.assertAllClose([0, 0], self.evaluate(var0)[1])
        self.assertAllClose([1, 1], self.evaluate(var1)[2])

        # Step 1: the momentum accumulators are 0. So we should see a normal
        # update: v -= grad * learning_rate
        mom_update.run()
        # Check that the momentum accumulators have been updated.
        self.assertAllCloseAccordingToType(
            np.array([0, 0]),
            self.evaluate(slot0)[0])
        self.assertAllCloseAccordingToType(
            np.array([-2.0 * .1, -2.0 * .1]),
            self.evaluate(slot0)[1])
        self.assertAllCloseAccordingToType(
            np.array([-2.0 * .01, -2.0 * .01]),
            self.evaluate(slot1)[2])
        # Check that the parameters have been updated.
        self.assertAllCloseAccordingToType(
            np.array([0, 0]),
            self.evaluate(var0)[0])
        self.assertAllCloseAccordingToType(
            np.array([-(0.1 * 2.0), -(0.1 * 2.0)]),
            self.evaluate(var0)[1])
        self.assertAllCloseAccordingToType(
            np.array([1.0 - (0.01 * 2.0), 1.0 - (0.01 * 2.0)]),
            self.evaluate(var1)[2])
        # Step 2: the momentum accumulators contain the previous update.
        mom_update.run()
        # Check that the momentum accumulators have been updated.
        self.assertAllClose(np.array([0, 0]), self.evaluate(slot0)[0])
        self.assertAllCloseAccordingToType(
            np.array([(0.9 * (-0.2) - 2.0 * 0.1), (0.9 * (-0.2) - 2.0 * 0.1)]),
            self.evaluate(slot0)[1])
        self.assertAllCloseAccordingToType(
            np.array([(0.9 * (-0.02) - 2.0 * 0.01),
                      (0.9 * (-0.02) - 2.0 * 0.01)]),
            self.evaluate(slot1)[2])
        # Check that the parameters have been updated.
        self.assertAllClose(np.array([0, 0]), self.evaluate(var0)[0])
        self.assertAllCloseAccordingToType(
            np.array([
                -(0.1 * 2.0) - ((0.9 * 0.1 + 0.1) * 2.0),
                -(0.1 * 2.0) - ((0.9 * 0.1 + 0.1) * 2.0)
            ]),
            self.evaluate(var0)[1])
        self.assertAllCloseAccordingToType(
            np.array([
                0.98 - ((0.9 * 0.01 + 0.01) * 2.0),
                0.98 - ((0.9 * 0.01 + 0.01) * 2.0)
            ]),
            self.evaluate(var1)[2])

  @test_util.run_deprecated_v1
  def testSharing(self):
    for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
      with self.cached_session():
        var0 = variables.Variable([1.0, 2.0], dtype=dtype)
        var1 = variables.Variable([3.0, 4.0], dtype=dtype)
        grads0 = constant_op.constant([0.1, 0.1], dtype=dtype)
        grads1 = constant_op.constant([0.01, 0.01], dtype=dtype)
        mom_opt = gradient_descent.SGD(learning_rate=2.0, momentum=0.9)
        mom_update1 = mom_opt.apply_gradients(
            zip([grads0, grads1], [var0, var1]))
        mom_update2 = mom_opt.apply_gradients(
            zip([grads0, grads1], [var0, var1]))
        variables.global_variables_initializer().run()

        slot0 = mom_opt.get_slot(var0, "momentum")
        self.assertEqual(slot0.get_shape(), var0.get_shape())
        slot1 = mom_opt.get_slot(var1, "momentum")
        self.assertEqual(slot1.get_shape(), var1.get_shape())

        # Fetch params to validate initial values
        self.assertAllClose([1.0, 2.0], self.evaluate(var0))
        self.assertAllClose([3.0, 4.0], self.evaluate(var1))
        # Step 1: the momentum accumulators where 0. So we should see a normal
        # update: v -= grad * learning_rate
        mom_update1.run()
        # Check that the momentum accumulators have been updated.
        self.assertAllCloseAccordingToType(
            np.array([-0.2, -0.2]), self.evaluate(slot0))
        self.assertAllCloseAccordingToType(
            np.array([-0.02, -0.02]), self.evaluate(slot1))
        # Check that the parameters have been updated.
        self.assertAllCloseAccordingToType(
            np.array([1.0 - (0.1 * 2.0), 2.0 - (0.1 * 2.0)]),
            self.evaluate(var0))
        self.assertAllCloseAccordingToType(
            np.array([3.0 - (0.01 * 2.0), 4.0 - (0.01 * 2.0)]),
            self.evaluate(var1))
        # Step 2: the second momentum accumulators contain the previous update.
        mom_update2.run()
        # Check that the momentum accumulators have been updated.
        self.assertAllCloseAccordingToType(
            np.array([(0.9 * (-0.2) - 2.0 * 0.1), (0.9 * (-0.2) - 2.0 * 0.1)]),
            self.evaluate(slot0))
        self.assertAllCloseAccordingToType(
            np.array([(0.9 * (-0.02) - 2.0 * 0.01),
                      (0.9 * (-0.02) - 2.0 * 0.01)]), self.evaluate(slot1))
        # Check that the parameters have been updated.
        self.assertAllCloseAccordingToType(
            np.array([
                1.0 - (0.1 * 2.0) - ((0.9 * 0.1 + 0.1) * 2.0),
                2.0 - (0.1 * 2.0) - ((0.9 * 0.1 + 0.1) * 2.0)
            ]), self.evaluate(var0))
        self.assertAllCloseAccordingToType(
            np.array([
                2.98 - ((0.9 * 0.01 + 0.01) * 2.0),
                3.98 - ((0.9 * 0.01 + 0.01) * 2.0)
            ]), self.evaluate(var1))

  @test_util.run_in_graph_and_eager_modes
  def testConfig(self):
    with self.cached_session():
      opt = gradient_descent.SGD(learning_rate=1.0, momentum=0.9, nesterov=True)
      config = opt.get_config()
      opt2 = gradient_descent.SGD.from_config(config)
      # assert both are equal float values.
      self.assertEqual(
          opt._get_hyper("learning_rate"), opt2._get_hyper("learning_rate"))
      self.assertEqual(opt._get_hyper("momentum"), opt2._get_hyper("momentum"))
      # self.assertEqual(opt._get_hyper("decay"), opt2._get_hyper("decay"))
      var0 = variables.Variable([[1.0], [2.0]], dtype=dtypes.float32)
      loss = lambda: 3 * var0
      # learning rate variable created when calling minimize.
      opt.minimize(loss, [var0])
      self.evaluate(variables.global_variables_initializer())
      config = opt.get_config()
      opt3 = gradient_descent.SGD.from_config(config)
      self.assertEqual(
          self.evaluate(opt._get_hyper("learning_rate")),
          opt3._get_hyper("learning_rate"))
      self.assertEqual(
          self.evaluate(opt._get_hyper("momentum")),
          opt3._get_hyper("momentum"))
      # self.assertEqual(
      #     self.evaluate(opt._get_hyper("decay")), opt3._get_hyper("decay"))
      self.assertTrue(opt3.nesterov)

  def testNesterovWithoutMomentum(self):
    with self.assertRaisesRegexp(ValueError, "must be between"):
      gradient_descent.SGD(learning_rate=1.0, momentum=2.0)

  def testConstructMomentumWithLR(self):
    opt = gradient_descent.SGD(lr=1.0, momentum=0.9)
    self.assertEqual(opt.lr, 1.0)
    opt_2 = gradient_descent.SGD(learning_rate=0.1, momentum=0.9, lr=1.0)
    self.assertEqual(opt_2.lr, 1.0)
    opt_3 = gradient_descent.SGD(learning_rate=0.1, momentum=0.9)
    self.assertEqual(opt_3.lr, 0.1)


if __name__ == "__main__":
  test.main()
