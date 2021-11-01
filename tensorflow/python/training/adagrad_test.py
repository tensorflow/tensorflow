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
"""Functional tests for aggregate operations."""

import numpy as np

from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import adagrad


class AdagradOptimizerTest(test.TestCase):

  def doTestBasic(self,
                  use_locking=False,
                  use_resource=False,
                  use_callable_params=False):
    for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
      if use_resource:
        var0 = resource_variable_ops.ResourceVariable([1.0, 2.0], dtype=dtype)
        var1 = resource_variable_ops.ResourceVariable([3.0, 4.0], dtype=dtype)
      else:
        var0 = variables.Variable([1.0, 2.0], dtype=dtype)
        var1 = variables.Variable([3.0, 4.0], dtype=dtype)
      grads0 = constant_op.constant([0.1, 0.1], dtype=dtype)
      grads1 = constant_op.constant([0.01, 0.01], dtype=dtype)

      learning_rate = lambda: 3.0
      if not use_callable_params:
        learning_rate = learning_rate()

      ada_opt = adagrad.AdagradOptimizer(
          learning_rate, initial_accumulator_value=0.1, use_locking=use_locking)

      if not context.executing_eagerly():
        ada_update = ada_opt.apply_gradients(
            zip([grads0, grads1], [var0, var1]))
        self.evaluate(variables.global_variables_initializer())

      # Fetch params to validate initial values
      v0_val, v1_val = self.evaluate([var0, var1])
      self.assertAllClose([1.0, 2.0], v0_val)
      self.assertAllClose([3.0, 4.0], v1_val)

      # Run 3 steps of adagrad
      for _ in range(3):
        if not context.executing_eagerly():
          self.evaluate(ada_update)
        else:
          ada_opt.apply_gradients(zip([grads0, grads1], [var0, var1]))

      # Validate updated params
      v0_val, v1_val = self.evaluate([var0, var1])
      self.assertAllCloseAccordingToType(
          np.array([-1.6026098728179932, -0.6026098728179932]), v0_val)
      self.assertAllCloseAccordingToType(
          np.array([2.715679168701172, 3.715679168701172]), v1_val)

  def testBasic(self):
    self.doTestBasic(use_locking=False)

  @test_util.run_in_graph_and_eager_modes
  def testBasicResource(self):
    self.doTestBasic(use_locking=False, use_resource=True)

  def testBasicCallableParams(self):
    with context.eager_mode():
      self.doTestBasic(
          use_locking=False, use_resource=True, use_callable_params=True)

  def testBasicLocked(self):
    self.doTestBasic(use_locking=True)

  def testMinimizeSparseResourceVariable(self):
    with ops.Graph().as_default():
      for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
        with self.cached_session():
          var0 = resource_variable_ops.ResourceVariable(
              [[1.0, 2.0], [3.0, 4.0]], dtype=dtype)
          x = constant_op.constant([[4.0], [5.0]], dtype=dtype)
          pred = math_ops.matmul(embedding_ops.embedding_lookup([var0], [0]), x)
          loss = pred * pred
          sgd_op = adagrad.AdagradOptimizer(1.0).minimize(loss)
          self.evaluate(variables.global_variables_initializer())
          # Fetch params to validate initial values
          self.assertAllCloseAccordingToType([[1.0, 2.0], [3.0, 4.0]],
                                             self.evaluate(var0))
          # Run 1 step of sgd
          sgd_op.run()
          # Validate updated params
          self.assertAllCloseAccordingToType([[0, 1], [3, 4]],
                                             self.evaluate(var0),
                                             atol=0.01)

  def testTensorLearningRate(self):
    with ops.Graph().as_default():
      for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
        with self.cached_session():
          var0 = variables.Variable([1.0, 2.0], dtype=dtype)
          var1 = variables.Variable([3.0, 4.0], dtype=dtype)
          grads0 = constant_op.constant([0.1, 0.1], dtype=dtype)
          grads1 = constant_op.constant([0.01, 0.01], dtype=dtype)
          ada_opt = adagrad.AdagradOptimizer(
              constant_op.constant(3.0), initial_accumulator_value=0.1)
          ada_update = ada_opt.apply_gradients(
              zip([grads0, grads1], [var0, var1]))
          self.evaluate(variables.global_variables_initializer())
          # Fetch params to validate initial values
          self.assertAllClose([1.0, 2.0], self.evaluate(var0))
          self.assertAllClose([3.0, 4.0], self.evaluate(var1))
          # Run 3 steps of adagrad
          for _ in range(3):
            ada_update.run()
          # Validate updated params
          self.assertAllCloseAccordingToType(
              np.array([-1.6026098728179932, -0.6026098728179932]),
              self.evaluate(var0))
          self.assertAllCloseAccordingToType(
              np.array([2.715679168701172, 3.715679168701172]),
              self.evaluate(var1))

  def testSparseBasic(self):
    with ops.Graph().as_default():
      for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
        with self.cached_session():
          var0 = variables.Variable([[1.0], [2.0]], dtype=dtype)
          var1 = variables.Variable([[3.0], [4.0]], dtype=dtype)
          grads0 = indexed_slices.IndexedSlices(
              constant_op.constant(
                  [0.1], shape=[1, 1], dtype=dtype),
              constant_op.constant([0]),
              constant_op.constant([2, 1]))
          grads1 = indexed_slices.IndexedSlices(
              constant_op.constant(
                  [0.01], shape=[1, 1], dtype=dtype),
              constant_op.constant([1]),
              constant_op.constant([2, 1]))
          ada_opt = adagrad.AdagradOptimizer(3.0, initial_accumulator_value=0.1)
          ada_update = ada_opt.apply_gradients(
              zip([grads0, grads1], [var0, var1]))
          self.evaluate(variables.global_variables_initializer())
          # Fetch params to validate initial values
          self.assertAllClose([[1.0], [2.0]], self.evaluate(var0))
          self.assertAllClose([[3.0], [4.0]], self.evaluate(var1))
          # Run 3 step of sgd
          for _ in range(3):
            ada_update.run()
          # Validate updated params
          self.assertAllCloseAccordingToType(
              np.array([[-1.6026098728179932], [2.0]]), self.evaluate(var0))
          self.assertAllCloseAccordingToType(
              np.array([[3.0], [3.715679168701172]]), self.evaluate(var1))

  def testSparseRepeatedIndices(self):
    with ops.Graph().as_default():
      for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
        with self.cached_session():
          repeated_index_update_var = variables.Variable(
              [[1.0], [2.0]], dtype=dtype)
          aggregated_update_var = variables.Variable(
              [[1.0], [2.0]], dtype=dtype)
          grad_repeated_index = indexed_slices.IndexedSlices(
              constant_op.constant(
                  [0.1, 0.1], shape=[2, 1], dtype=dtype),
              constant_op.constant([1, 1]),
              constant_op.constant([2, 1]))
          grad_aggregated = indexed_slices.IndexedSlices(
              constant_op.constant(
                  [0.2], shape=[1, 1], dtype=dtype),
              constant_op.constant([1]),
              constant_op.constant([2, 1]))
          repeated_update = adagrad.AdagradOptimizer(3.0).apply_gradients(
              [(grad_repeated_index, repeated_index_update_var)])
          aggregated_update = adagrad.AdagradOptimizer(3.0).apply_gradients(
              [(grad_aggregated, aggregated_update_var)])
          self.evaluate(variables.global_variables_initializer())
          self.assertAllClose(aggregated_update_var,
                              self.evaluate(repeated_index_update_var))
          for _ in range(3):
            repeated_update.run()
            aggregated_update.run()
            self.assertAllClose(aggregated_update_var,
                                self.evaluate(repeated_index_update_var))

  def testSparseRepeatedIndicesResourceVariable(self):
    with ops.Graph().as_default():
      for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
        with self.cached_session():
          var_repeated = resource_variable_ops.ResourceVariable(
              [1.0, 2.0], dtype=dtype)
          loss_repeated = math_ops.reduce_sum(
              embedding_ops.embedding_lookup(var_repeated, [0, 0]))
          var_aggregated = resource_variable_ops.ResourceVariable(
              [1.0, 2.0], dtype=dtype)
          loss_aggregated = 2 * math_ops.reduce_sum(
              embedding_ops.embedding_lookup(var_aggregated, [0]))
          update_op_repeated = adagrad.AdagradOptimizer(
              2.0).minimize(loss_repeated)
          update_op_aggregated = adagrad.AdagradOptimizer(
              2.0).minimize(loss_aggregated)
          self.evaluate(variables.global_variables_initializer())
          self.assertAllCloseAccordingToType(
              self.evaluate(var_repeated), self.evaluate(var_aggregated))
          for _ in range(3):
            update_op_repeated.run()
            update_op_aggregated.run()
            self.assertAllCloseAccordingToType(
                self.evaluate(var_repeated), self.evaluate(var_aggregated))

  def testSparseStability(self):
    with ops.Graph().as_default():
      for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
        with self.cached_session():
          shape = [1, 6]
          var0 = variables.Variable(
              [[
                  0.00872496, -0.106952, 0.110467, 0.226505, -0.0147257,
                  -0.0105945
              ]],
              dtype=dtype)
          grads0 = indexed_slices.IndexedSlices(
              constant_op.constant(
                  [[
                      -5.91278e-05, 5.31673e-05, -2.5779e-06, 4.29153e-05,
                      -8.4877e-05, -9.48906e-05
                  ]],
                  shape=shape,
                  dtype=dtype),
              constant_op.constant([0]),
              constant_op.constant(shape))
          ada_opt = adagrad.AdagradOptimizer(1.0, initial_accumulator_value=0.1)
          ada_update = ada_opt.apply_gradients(zip([grads0], [var0]))
          self.assertEqual(["accumulator"], ada_opt.get_slot_names())
          slot0 = ada_opt.get_slot(var0, "accumulator")
          init = variables.global_variables_initializer()
          for _ in range(100):
            init.run()
            ada_update.run()
            self.assertAllCloseAccordingToType(
                np.array([[0.1, 0.1, 0.1, 0.1, 0.1, 0.1]]),
                self.evaluate(slot0))
            self.assertAllCloseAccordingToType(
                np.array([[
                    0.00891194, -0.10712013, 0.11047515, 0.22636929, -0.0144573,
                    -0.01029443
                ]]), self.evaluate(var0))

  def testSharing(self):
    with ops.Graph().as_default():
      for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
        with self.cached_session():
          var0 = variables.Variable([1.0, 2.0], dtype=dtype)
          var1 = variables.Variable([3.0, 4.0], dtype=dtype)
          grads0 = constant_op.constant([0.1, 0.1], dtype=dtype)
          grads1 = constant_op.constant([0.01, 0.01], dtype=dtype)
          ada_opt = adagrad.AdagradOptimizer(3.0)
          # Apply the optimizer twice.  Both applications will use
          # the same accums.
          ada_update1 = ada_opt.apply_gradients(
              zip([grads0, grads1], [var0, var1]))
          ada_update2 = ada_opt.apply_gradients(
              zip([grads0, grads1], [var0, var1]))
          self.assertEqual(["accumulator"], ada_opt.get_slot_names())
          slot0 = ada_opt.get_slot(var0, "accumulator")
          self.assertEqual(slot0.get_shape(), var0.get_shape())
          slot1 = ada_opt.get_slot(var1, "accumulator")
          self.assertEqual(slot1.get_shape(), var1.get_shape())
          self.evaluate(variables.global_variables_initializer())

          # Fetch params to validate initial values.
          self.assertAllClose([1.0, 2.0], self.evaluate(var0))
          self.assertAllClose([3.0, 4.0], self.evaluate(var1))
          # Mix the first and the second adagrad for 3 steps.
          ada_update1.run()
          ada_update2.run()
          ada_update1.run()
          # Validate updated params (the same as with only 1 Adagrad).
          self.assertAllCloseAccordingToType(
              np.array([-1.6026098728179932, -0.6026098728179932]),
              self.evaluate(var0))
          self.assertAllCloseAccordingToType(
              np.array([2.715679168701172, 3.715679168701172]),
              self.evaluate(var1))

  def testDynamicShapeVariableWithCallableInit(self):
    with ops.Graph().as_default():
      var0 = variable_scope.get_variable("var0",
                                         initializer=constant_op.constant(1.),
                                         validate_shape=False)

      grads0 = constant_op.constant(0.1, dtype=dtypes.float32)
      learning_rate = lambda: 3.0

      ada_opt = adagrad.AdagradOptimizer(
          learning_rate, initial_accumulator_value=0.1, use_locking=True)

      if not context.executing_eagerly():
        ada_update = ada_opt.apply_gradients(
            zip([grads0], [var0]))
        self.evaluate(variables.global_variables_initializer())

      # Fetch params to validate initial values
      v0_val = self.evaluate([var0])
      self.assertAllClose([1.0], v0_val)

      # Run 3 steps of adagrad
      for _ in range(3):
        if not context.executing_eagerly():
          self.evaluate(ada_update)
        else:
          ada_opt.apply_gradients(zip([grads0], [var0]))

      # Validate updated params
      v0_val = self.evaluate([var0])
      self.assertAllCloseAccordingToType(
          np.array([-1.6026098728179932]), v0_val)


if __name__ == "__main__":
  test.main()
