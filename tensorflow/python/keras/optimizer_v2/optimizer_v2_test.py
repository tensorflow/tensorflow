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
"""Functional test for OptimizerV2."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.eager import context
from tensorflow.python.eager import function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.keras.optimizer_v2 import gradient_descent
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


class OptimizerTest(test.TestCase):

  @test_util.run_in_graph_and_eager_modes
  def testBasic(self):
    for _, dtype in enumerate([dtypes.half, dtypes.float32, dtypes.float64]):
      with self.cached_session():
        var0 = resource_variable_ops.ResourceVariable([1.0, 2.0], dtype=dtype)
        var1 = resource_variable_ops.ResourceVariable([3.0, 4.0], dtype=dtype)
        loss = lambda: 5 * var0 + 3 * var1  # pylint: disable=cell-var-from-loop
        if not context.executing_eagerly():
          loss = loss()
        sgd = gradient_descent.SGD(3.0)

        self.evaluate(variables.global_variables_initializer())
        # Fetch params to validate initial values
        self.assertAllClose([1.0, 2.0], self.evaluate(var0))
        self.assertAllClose([3.0, 4.0], self.evaluate(var1))
        # Run 1 step of sgd through optimizer
        opt_op = sgd.minimize(loss, var_list=[var0, var1])
        self.evaluate(variables.global_variables_initializer())
        self.evaluate(opt_op)
        # Validate updated params
        self.assertAllClose([-14., -13.], self.evaluate(var0))
        self.assertAllClose([-6., -5.], self.evaluate(var1))

  @test_util.run_in_graph_and_eager_modes
  def testAdaptiveLearningRate(self):
    for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
      var0 = resource_variable_ops.ResourceVariable([1.0, 2.0], dtype=dtype)
      var1 = resource_variable_ops.ResourceVariable([3.0, 4.0], dtype=dtype)

      def loss():
        return 5 * var0 + 3 * var1  # pylint: disable=cell-var-from-loop

      sgd = gradient_descent.SGD(1.0)

      self.evaluate(variables.global_variables_initializer())
      # Fetch params to validate initial values
      self.assertAllClose([1.0, 2.0], self.evaluate(var0))
      self.assertAllClose([3.0, 4.0], self.evaluate(var1))
      # Run 1 step of sgd through optimizer
      opt_op = sgd.minimize(loss, [var0, var1])
      self.evaluate(variables.global_variables_initializer())
      self.evaluate(opt_op)
      # Validate updated params
      # var0 = [1., 2.] - 1.0 * [5, 5]
      self.assertAllClose([-4., -3.], self.evaluate(var0))
      # var1 = [3., 4.] - 1.0 * [3, 3]
      self.assertAllClose([0., 1.], self.evaluate(var1))

      sgd.learning_rate = 0.5
      if context.executing_eagerly():
        sgd.minimize(loss, [var0, var1])
      else:
        self.evaluate(opt_op)
      # Validate updated params
      # var0 = [-4., -3.] - 0.5 * [5, 5]
      self.assertAllClose([-6.5, -5.5], self.evaluate(var0))
      # var1 = [0., 1.] - 0.5 * [3, 3]
      self.assertAllClose([-1.5, -0.5], self.evaluate(var1))

  @test_util.run_in_graph_and_eager_modes
  def testAggregationMethod(self):
    for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
      with self.cached_session():
        var0 = variables.Variable([1.0, 2.0], dtype=dtype)
        var1 = variables.Variable([3.0, 4.0], dtype=dtype)
        loss = lambda: 5 * var0 + 3 * var1  # pylint: disable=cell-var-from-loop
        if not context.executing_eagerly():
          loss = loss()
        sgd = gradient_descent.SGD(3.0)

        self.evaluate(variables.global_variables_initializer())
        # Fetch params to validate initial values
        self.assertAllClose([1.0, 2.0], self.evaluate(var0))
        self.assertAllClose([3.0, 4.0], self.evaluate(var1))
        # Run 1 step of sgd through optimizer
        opt_op = sgd.minimize(
            loss,
            var_list=[var0, var1],
            aggregation_method=gradients_impl.AggregationMethod
            .EXPERIMENTAL_ACCUMULATE_N)
        self.evaluate(variables.global_variables_initializer())
        self.evaluate(opt_op)
        # Validate updated params
        self.assertAllClose([-14., -13.], self.evaluate(var0))
        self.assertAllClose([-6., -5.], self.evaluate(var1))

  @test_util.run_in_graph_and_eager_modes
  def testPrecomputedGradient(self):
    for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
      with self.cached_session():
        var0 = variables.Variable([1.0, 2.0], dtype=dtype)
        var1 = variables.Variable([3.0, 4.0], dtype=dtype)
        loss = lambda: 5 * var0 + 3 * var1  # pylint: disable=cell-var-from-loop
        if not context.executing_eagerly():
          loss = loss()
        grad_loss = constant_op.constant([42, -42], dtype=dtype)
        sgd = gradient_descent.SGD(3.0)

        self.evaluate(variables.global_variables_initializer())
        # Fetch params to validate initial values
        self.assertAllClose([1.0, 2.0], self.evaluate(var0))
        self.assertAllClose([3.0, 4.0], self.evaluate(var1))
        # Run 1 step of sgd through optimizer
        opt_op = sgd.minimize(loss, var_list=[var0, var1], grad_loss=grad_loss)
        self.evaluate(variables.global_variables_initializer())
        self.evaluate(opt_op)
        # Validate updated params
        self.assertAllClose([1.0 - 3 * 5 * 42.0, 2.0 - 3 * 5 * (-42.0)],
                            self.evaluate(var0))
        self.assertAllClose([3.0 - 3 * 3 * 42.0, 4.0 - 3 * 3 * (-42.0)],
                            self.evaluate(var1))

  @test_util.run_in_graph_and_eager_modes
  def testNoGradients(self):
    for _, dtype in enumerate([dtypes.half, dtypes.float32, dtypes.float64]):
      with self.cached_session():
        var0 = resource_variable_ops.ResourceVariable([1.0, 2.0], dtype=dtype)
        var1 = resource_variable_ops.ResourceVariable([3.0, 4.0], dtype=dtype)
        loss = lambda: 5 * var0  # pylint: disable=cell-var-from-loop
        if not context.executing_eagerly():
          loss = loss()
        sgd_op = gradient_descent.SGD(3.0)
        with self.assertRaisesRegexp(ValueError, 'No gradients'):
          # var1 has no gradient
          sgd_op.minimize(loss, var_list=[var1])

  @test_util.run_in_graph_and_eager_modes
  def testNoGradientsForAnyVariables_Minimize(self):
    for _, dtype in enumerate([dtypes.half, dtypes.float32, dtypes.float64]):
      with self.cached_session():
        var0 = resource_variable_ops.ResourceVariable([1.0, 2.0], dtype=dtype)
        var1 = resource_variable_ops.ResourceVariable([3.0, 4.0], dtype=dtype)
        loss = lambda: constant_op.constant(5.0)
        if not context.executing_eagerly():
          loss = loss()

        sgd_op = gradient_descent.SGD(3.0)
        with self.assertRaisesRegexp(ValueError,
                                     'No gradients provided for any variable'):
          sgd_op.minimize(loss, var_list=[var0, var1])

  @test_util.run_in_graph_and_eager_modes
  def testNoGradientsForAnyVariables_ApplyGradients(self):
    for _, dtype in enumerate([dtypes.half, dtypes.float32, dtypes.float64]):
      with self.cached_session():
        var0 = resource_variable_ops.ResourceVariable([1.0, 2.0], dtype=dtype)
        var1 = resource_variable_ops.ResourceVariable([3.0, 4.0], dtype=dtype)
        sgd_op = gradient_descent.SGD(3.0)
        with self.assertRaisesRegexp(ValueError,
                                     'No gradients provided for any variable'):
          sgd_op.apply_gradients([(None, var0), (None, var1)])

  @test_util.run_in_graph_and_eager_modes
  def testGradientsAsVariables(self):
    for i, dtype in enumerate([dtypes.half, dtypes.float32, dtypes.float64]):
      with self.cached_session():
        var0 = resource_variable_ops.ResourceVariable([1.0, 2.0], dtype=dtype)
        var1 = resource_variable_ops.ResourceVariable([3.0, 4.0], dtype=dtype)
        loss = lambda: 5 * var0 + 3 * var1  # pylint: disable=cell-var-from-loop
        if not context.executing_eagerly():
          loss = loss()

        sgd = gradient_descent.SGD(3.0)
        grads_and_vars = sgd.compute_gradients(loss, [var0, var1])
        # Convert gradients to tf.Variables
        converted_grads = [
            resource_variable_ops.ResourceVariable(
                array_ops.zeros([2], dtype), name='c_%d_%d' % (i, j))
            for j, gv in enumerate(grads_and_vars)
        ]
        convert_ops = [
            state_ops.assign(converted_grads[j], gv[0])
            for j, gv in enumerate(grads_and_vars)
        ]

        # Run convert_ops to achieve the gradients converting
        self.evaluate(variables.global_variables_initializer())
        self.evaluate(convert_ops)
        # Fetch params to validate initial values
        self.assertAllClose([1.0, 2.0], self.evaluate(var0))
        self.assertAllClose([3.0, 4.0], self.evaluate(var1))

        # Run 1 step of sgd through optimizer
        converted_grads_and_vars = list(zip(converted_grads, [var0, var1]))
        opt_op = sgd.apply_gradients(converted_grads_and_vars)
        self.evaluate(variables.global_variables_initializer())
        self.evaluate(convert_ops)
        self.evaluate(opt_op)

        # Validate updated params
        self.assertAllClose([-14., -13.], self.evaluate(var0))
        self.assertAllClose([-6., -5.], self.evaluate(var1))

  @test_util.run_in_graph_and_eager_modes
  def testComputeGradientsWithTensors(self):
    with self.cached_session():
      x = ops.convert_to_tensor(1.0)

      def f():
        return x * x

      sgd = gradient_descent.SGD(3.0)
      grads_and_vars = sgd.compute_gradients(f, [x])
      self.assertEqual(1, len(grads_and_vars))
      grad, x_as_var = grads_and_vars[0]
      self.assertIs(x, x_as_var)
      self.assertEqual(2.0, self.evaluate(grad))

      with self.assertRaises(NotImplementedError):
        sgd.apply_gradients(grads_and_vars)

  @test_util.run_in_graph_and_eager_modes
  def testConstraint(self):
    constraint_01 = lambda x: clip_ops.clip_by_value(x, -0.1, 0.)
    constraint_0 = lambda x: clip_ops.clip_by_value(x, 0., 1.)
    with self.cached_session():
      var0 = variables.Variable([1.0, 2.0],
                                constraint=constraint_01)
      var1 = variables.Variable([3.0, 4.0],
                                constraint=constraint_0)
      loss = lambda: 5 * var0 + 3 * var1
      if not context.executing_eagerly():  # pylint: disable=cell-var-from-loop
        loss = loss()
      sgd = gradient_descent.SGD(3.0)

      self.evaluate(variables.global_variables_initializer())
      # Fetch params to validate initial values
      self.assertAllClose([1.0, 2.0], self.evaluate(var0))
      self.assertAllClose([3.0, 4.0], self.evaluate(var1))
      # Run 1 step of sgd through optimizer
      opt_op = sgd.minimize(loss, var_list=[var0, var1])
      self.evaluate(variables.global_variables_initializer())
      self.evaluate(opt_op)
      # Validate updated params
      self.assertAllClose([-0.1, -0.1], self.evaluate(var0))
      self.assertAllClose([0., 0.], self.evaluate(var1))

  @test_util.run_in_graph_and_eager_modes
  def testIterationWithoutMinimize(self):
    with self.cached_session():
      sgd = gradient_descent.SGD(3.0)
      self.evaluate(sgd.iteration.initializer)
      self.assertEqual(0, self.evaluate(sgd.iteration))

  @test_util.run_in_graph_and_eager_modes
  def testSerializationWithinDefun(self):
    with self.cached_session():
      sgd = gradient_descent.SGD(3.0)
      var0 = resource_variable_ops.ResourceVariable([1.0, 2.0],
                                                    dtype=dtypes.float32)
      loss = lambda: 3 * var0
      sgd.minimize(loss, [var0])

      def serialize():
        config = sgd.get_config()
        gradient_descent.SGD.from_config(config)

      compiled_serialize = function.defun(serialize)
      with self.assertRaisesRegexp(ValueError, 'inside defun'):
        compiled_serialize()


if __name__ == '__main__':
  test.main()
