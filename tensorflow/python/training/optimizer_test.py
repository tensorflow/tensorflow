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
"""Functional test for optimizer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.distribute import cross_device_ops
from tensorflow.python.distribute import distribute_utils
from tensorflow.python.distribute import mirrored_strategy
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import gradients_util
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import adam
from tensorflow.python.training import gradient_descent


class OptimizerTest(test.TestCase):

  @test_util.run_in_graph_and_eager_modes
  def testBasic(self):
    for i, dtype in enumerate([dtypes.half, dtypes.float32, dtypes.float64]):
      # Note that we name the variables uniquely here since the variables don't
      # seem to be getting deleted at the end of the loop.
      var0 = resource_variable_ops.ResourceVariable([1.0, 2.0], dtype=dtype,
                                                    name='a_%d' % i)
      var1 = resource_variable_ops.ResourceVariable([3.0, 4.0], dtype=dtype,
                                                    name='b_%d' % i)
      def loss():
        return 5 * var0 + 3 * var1  # pylint: disable=cell-var-from-loop
      # Note that for eager execution, minimize expects a function instead of a
      # Tensor.
      global_step = resource_variable_ops.ResourceVariable(
          array_ops.zeros([], dtypes.int64), name='global_step_%d' % i)
      sgd_op = gradient_descent.GradientDescentOptimizer(3.0)

      self.evaluate(variables.global_variables_initializer())
      # Fetch params to validate initial values
      self.assertAllClose([1.0, 2.0], self.evaluate(var0))
      self.assertAllClose([3.0, 4.0], self.evaluate(var1))
      # Run 1 step of sgd through optimizer
      opt_op = sgd_op.minimize(loss, global_step, [var0, var1])
      self.evaluate(opt_op)
      # Validate updated params
      self.assertAllClose([-14., -13.], self.evaluate(var0))
      self.assertAllClose([-6., -5.], self.evaluate(var1))

  @test_util.run_deprecated_v1
  def testAggregationMethod(self):
    for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
      with self.cached_session():
        var0 = variables.Variable([1.0, 2.0], dtype=dtype)
        var1 = variables.Variable([3.0, 4.0], dtype=dtype)
        cost = 5 * var0 + 3 * var1
        global_step = variables.Variable(
            array_ops.zeros([], dtypes.int64), name='global_step')
        sgd_op = gradient_descent.GradientDescentOptimizer(3.0)
        opt_op = sgd_op.minimize(
            cost,
            global_step, [var0, var1],
            aggregation_method=gradients_util.AggregationMethod.
            EXPERIMENTAL_ACCUMULATE_N)

        self.evaluate(variables.global_variables_initializer())
        # Fetch params to validate initial values
        self.assertAllClose([1.0, 2.0], self.evaluate(var0))
        self.assertAllClose([3.0, 4.0], self.evaluate(var1))
        # Run 1 step of sgd through optimizer
        opt_op.run()
        # Validate updated params
        self.assertAllClose([-14., -13.], self.evaluate(var0))
        self.assertAllClose([-6., -5.], self.evaluate(var1))

  @test_util.run_deprecated_v1
  def testPrecomputedGradient(self):
    for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
      with self.cached_session():
        var0 = variables.Variable([1.0, 2.0], dtype=dtype)
        var1 = variables.Variable([3.0, 4.0], dtype=dtype)
        cost = 5 * var0 + 3 * var1
        grad_loss = constant_op.constant([42, -42], dtype=dtype)
        global_step = variables.Variable(
            array_ops.zeros([], dtypes.int64), name='global_step')
        sgd_op = gradient_descent.GradientDescentOptimizer(3.0)
        opt_op = sgd_op.minimize(
            cost, global_step, [var0, var1], grad_loss=grad_loss)

        self.evaluate(variables.global_variables_initializer())
        # Fetch params to validate initial values
        self.assertAllClose([1.0, 2.0], self.evaluate(var0))
        self.assertAllClose([3.0, 4.0], self.evaluate(var1))
        # Run 1 step of sgd through optimizer
        opt_op.run()
        # Validate updated params
        self.assertAllClose([1.0 - 3 * 5 * 42.0, 2.0 - 3 * 5 * (-42.0)],
                            self.evaluate(var0))
        self.assertAllClose([3.0 - 3 * 3 * 42.0, 4.0 - 3 * 3 * (-42.0)],
                            self.evaluate(var1))

  @test_util.run_in_graph_and_eager_modes
  def testNoVariables(self):
    for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
      # pylint: disable=cell-var-from-loop
      def loss():
        var0 = resource_variable_ops.ResourceVariable(
            [1.0, 2.0], dtype=dtype, trainable=False, name='a')
        var1 = resource_variable_ops.ResourceVariable(
            [3.0, 4.0], dtype=dtype, trainable=False, name='b')
        return 5 * var0 + var1
      # pylint: enable=cell-var-from-loop
      sgd_op = gradient_descent.GradientDescentOptimizer(3.0)
      with self.assertRaisesRegex(ValueError, 'No.*variables'):
        sgd_op.minimize(loss)

  @test_util.run_in_graph_and_eager_modes
  def testNoGradients(self):
    for i, dtype in enumerate([dtypes.half, dtypes.float32, dtypes.float64]):
      # Note that we name the variables uniquely here since the variables don't
      # seem to be getting deleted at the end of the loop.
      var0 = resource_variable_ops.ResourceVariable([1.0, 2.0], dtype=dtype,
                                                    name='a%d' % i)
      var1 = resource_variable_ops.ResourceVariable([3.0, 4.0], dtype=dtype,
                                                    name='b%d' % i)
      # pylint: disable=cell-var-from-loop
      def loss():
        return 5 * var0
      # pylint: enable=cell-var-from-loop
      sgd_op = gradient_descent.GradientDescentOptimizer(3.0)
      with self.assertRaisesRegex(ValueError, 'No gradients'):
        # var1 has no gradient
        sgd_op.minimize(loss, var_list=[var1])

  @test_util.run_in_graph_and_eager_modes
  def testNoGradientsForAnyVariables_Minimize(self):
    for i, dtype in enumerate([dtypes.half, dtypes.float32, dtypes.float64]):
      # Note that we name the variables uniquely here since the variables don't
      # seem to be getting deleted at the end of the loop.
      var0 = resource_variable_ops.ResourceVariable([1.0, 2.0], dtype=dtype,
                                                    name='a_%d' % i)
      var1 = resource_variable_ops.ResourceVariable([3.0, 4.0], dtype=dtype,
                                                    name='b_%d' % i)
      def loss():
        return constant_op.constant(5.0)
      sgd_op = gradient_descent.GradientDescentOptimizer(3.0)
      with self.assertRaisesRegex(ValueError,
                                  'No gradients provided for any variable'):
        sgd_op.minimize(loss, var_list=[var0, var1])

  @test_util.run_in_graph_and_eager_modes
  def testNoGradientsForAnyVariables_ApplyGradients(self):
    for i, dtype in enumerate([dtypes.half, dtypes.float32, dtypes.float64]):
      # Note that we name the variables uniquely here since the variables don't
      # seem to be getting deleted at the end of the loop.
      var0 = resource_variable_ops.ResourceVariable([1.0, 2.0], dtype=dtype,
                                                    name='a_%d' % i)
      var1 = resource_variable_ops.ResourceVariable([3.0, 4.0], dtype=dtype,
                                                    name='b_%d' % i)
      sgd_op = gradient_descent.GradientDescentOptimizer(3.0)
      with self.assertRaisesRegex(ValueError,
                                  'No gradients provided for any variable'):
        sgd_op.apply_gradients([(None, var0), (None, var1)])

  @test_util.run_in_graph_and_eager_modes
  def testGradientsAsVariables(self):
    for i, dtype in enumerate([dtypes.half, dtypes.float32, dtypes.float64]):
      # Note that we name the variables uniquely here since the variables don't
      # seem to be getting deleted at the end of the loop.
      var0 = resource_variable_ops.ResourceVariable([1.0, 2.0], dtype=dtype,
                                                    name='a%d' % i)
      var1 = resource_variable_ops.ResourceVariable([3.0, 4.0], dtype=dtype,
                                                    name='b%d' % i)
      def loss():
        return 5 * var0 + 3 * var1  # pylint: disable=cell-var-from-loop
      sgd_op = gradient_descent.GradientDescentOptimizer(3.0)
      grads_and_vars = sgd_op.compute_gradients(loss, [var0, var1])
      # Convert gradients to tf.Variables
      converted_grads = [
          resource_variable_ops.ResourceVariable(array_ops.zeros([2], dtype),
                                                 name='c_%d_%d' % (i, j))
          for j, gv in enumerate(grads_and_vars)
      ]
      convert_ops = [
          state_ops.assign(converted_grads[j], gv[0])
          for j, gv in enumerate(grads_and_vars)
      ]

      self.evaluate(variables.global_variables_initializer())
      # Run convert_ops to achieve the gradients converting
      self.evaluate(convert_ops)
      # Fetch params to validate initial values
      self.assertAllClose([1.0, 2.0], self.evaluate(var0))
      self.assertAllClose([3.0, 4.0], self.evaluate(var1))

      # Run 1 step of sgd through optimizer
      converted_grads_and_vars = list(zip(converted_grads, [var0, var1]))
      opt_op = sgd_op.apply_gradients(converted_grads_and_vars)
      self.evaluate(opt_op)

      # Validate updated params
      self.assertAllClose([-14., -13.], self.evaluate(var0))
      self.assertAllClose([-6., -5.], self.evaluate(var1))

  @test_util.run_in_graph_and_eager_modes
  def testComputeGradientsWithTensors(self):
    x = ops.convert_to_tensor(1.0)
    def f():
      return x * x
    sgd_op = gradient_descent.GradientDescentOptimizer(3.0)
    grads_and_vars = sgd_op.compute_gradients(f, [x])
    self.assertEqual(1, len(grads_and_vars))
    grad, x_as_var = grads_and_vars[0]
    self.assertIs(x, x_as_var)
    self.assertEqual(2.0, self.evaluate(grad))

    with self.assertRaises(NotImplementedError):
      sgd_op.apply_gradients(grads_and_vars)

  @test_util.run_deprecated_v1
  def testTrainOp(self):
    with self.cached_session():
      var0 = variables.Variable([1.0, 2.0])
      var1 = variables.Variable([3.0, 4.0])
      cost = 5 * var0 + 3 * var1
      global_step = variables.Variable(
          array_ops.zeros([], dtypes.int64), name='global_step')
      sgd_op = gradient_descent.GradientDescentOptimizer(3.0)
      opt_op = sgd_op.minimize(cost, global_step, [var0, var1])
      self.assertTrue(opt_op in ops.get_collection(ops.GraphKeys.TRAIN_OP))

  @test_util.run_deprecated_v1
  def testConstraint(self):
    constraint_01 = lambda x: clip_ops.clip_by_value(x, -0.1, 0.)
    constraint_0 = lambda x: clip_ops.clip_by_value(x, 0., 1.)
    with self.cached_session():
      var0 = variables.Variable([1.0, 2.0],
                                constraint=constraint_01)
      var1 = variables.Variable([3.0, 4.0],
                                constraint=constraint_0)
      cost = 5 * var0 + 3 * var1
      global_step = variables.Variable(
          array_ops.zeros([], dtypes.int64), name='global_step')
      sgd_op = gradient_descent.GradientDescentOptimizer(3.0)
      opt_op = sgd_op.minimize(cost, global_step, [var0, var1])

      self.evaluate(variables.global_variables_initializer())
      # Fetch params to validate initial values
      self.assertAllClose([1.0, 2.0], self.evaluate(var0))
      self.assertAllClose([3.0, 4.0], self.evaluate(var1))
      # Run 1 step of sgd through optimizer
      opt_op.run()
      # Validate updated params
      self.assertAllClose([-0.1, -0.1], self.evaluate(var0))
      self.assertAllClose([0., 0.], self.evaluate(var1))

  @test_util.run_deprecated_v1
  def testGetSlotUnderDistributedStrategy(self):
    # Only run this test in graph mode so we don't need actual GPU.
    ds = mirrored_strategy.MirroredStrategy(
        ['CPU:0', 'GPU:0'],
        cross_device_ops=cross_device_ops.HierarchicalCopyAllReduce())
    # We need an optimizer that creates slots.
    optimizer = adam.AdamOptimizer()

    def f():
      v = variables.Variable([1.0])
      self.assertTrue(distribute_utils.is_distributed_variable(v))
      # Slot variables are created in the first call to apply_gradients.
      optimizer.apply_gradients([(ops.convert_to_tensor([1.0]), v)])
      self.assertTrue(optimizer.get_slot_names())
      for name in optimizer.get_slot_names():
        slot = optimizer.get_slot(v, name)
        self.assertIsNotNone(slot)
        self.assertTrue(distribute_utils.is_distributed_variable(slot))

    ds.run(f)


if __name__ == '__main__':
  test.main()
