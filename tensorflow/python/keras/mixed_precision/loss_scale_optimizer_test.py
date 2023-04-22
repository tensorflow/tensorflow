# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for LossScaleOptimizer."""

import os

from absl.testing import parameterized
import numpy as np

from tensorflow.python.distribute import central_storage_strategy
from tensorflow.python.distribute import distribution_strategy_context
from tensorflow.python.distribute import mirrored_strategy
from tensorflow.python.eager import context
from tensorflow.python.framework import config as tf_config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.keras import combinations
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.mixed_precision import loss_scale_optimizer
from tensorflow.python.keras.mixed_precision import test_util as mp_test_util
from tensorflow.python.keras.optimizer_v2 import adam
from tensorflow.python.keras.optimizer_v2 import gradient_descent
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_v2_toggles
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training.experimental import loss_scale as tf_loss_scale_module
from tensorflow.python.training.tracking import util as trackable_utils

# Disable not-callable lint error, as the linter is unable to detect that
# LossScale instances are callable.
# pylint: disable=not-callable


# If called outside any strategy.scope() calls, this will return the default
# strategy.
default_strategy_fn = distribution_strategy_context.get_strategy


def create_mirrored_strategy():
  if tf_config.list_logical_devices('GPU'):
    return mirrored_strategy.MirroredStrategy(['cpu:0', 'gpu:0'])
  else:
    return mirrored_strategy.MirroredStrategy(['cpu:0'])


TESTCASES = ({
    'testcase_name': 'Base',
    'strategy_fn': default_strategy_fn
}, {
    'testcase_name': 'Distribute',
    'strategy_fn': create_mirrored_strategy
})


@test_util.with_control_flow_v2
@combinations.generate(combinations.combine(mode=['graph', 'eager']))
class LossScaleOptimizerTest(test.TestCase, parameterized.TestCase):

  def _run_if_in_graph_mode(self, val):
    # Running only in graph mode is useful, because optimizers sometimes return
    # a value that, in Graph mode, is runnable with self.evaluate. But in Eager
    # mode, the optimizer already does the computations and the return value
    # cannot be run.
    if not context.executing_eagerly():
      self.evaluate(val)

  def _run_fn_with_grad_check(self, strategy, var, opt, expected_grad):
    grad_check_fn = mp_test_util.create_identity_with_grad_check_fn(
        expected_grad)
    loss = lambda: grad_check_fn(var) / strategy.num_replicas_in_sync
    return lambda: opt.minimize(loss, var_list=[var])

  @parameterized.named_parameters(*TESTCASES)
  def testFixedLossScaleAppliedToLossWithMinimize(self, strategy_fn):
    with strategy_fn().scope() as strategy:
      var = variables.Variable([5.0])
      opt = gradient_descent.SGD(2.0)
      loss_scale = 10.
      opt = loss_scale_optimizer.LossScaleOptimizer(opt, dynamic=False,
                                                    initial_scale=loss_scale)
      self.assertEqual(self.evaluate(opt.loss_scale), loss_scale)
      self.assertIsInstance(opt.loss_scale, ops.Tensor)
      # We need num_replicas_in_sync to divide loss_scale, otherwise loss_scale
      # / strategy.num_replicas_in_sync will not be exact, which could lead to
      # assertion failures due to rounding issues.
      self.assertEqual(loss_scale % strategy.num_replicas_in_sync, 0)
      run_fn = self._run_fn_with_grad_check(
          strategy, var, opt, loss_scale / strategy.num_replicas_in_sync)
      run_op = strategy.experimental_run(run_fn)
      self.evaluate(variables.global_variables_initializer())
      self._run_if_in_graph_mode(run_op)
      # The loss is the identity of the variable. Therefore the gradient is 1,
      # and so the variable will be init_val - grad * lr == 5 - 1 * 2 == 3
      self.assertAllClose([3.], self.evaluate(var))

  def testFixedLossScaleAppliedToLossWithGetGradients(self):
    with ops.Graph().as_default():
      var = variables.Variable([2.0])
      opt = gradient_descent.SGD(1.0)
      loss_scale = 10.
      opt = loss_scale_optimizer.LossScaleOptimizer(opt, dynamic=False,
                                                    initial_scale=loss_scale)
      grad_check_fn = mp_test_util.create_identity_with_grad_check_fn(
          loss_scale)
      loss = grad_check_fn(var)
      run_op = opt.get_gradients(loss, [var])
      self.evaluate(variables.global_variables_initializer())
      # This will cause an assertion to run, as
      # mp_test_util.create_identity_with_grad_check_fn added an assertion op.
      self.evaluate(run_op)

  def testDynamicAttrsWithFixedLossScale(self):
    opt = gradient_descent.SGD()
    opt = loss_scale_optimizer.LossScaleOptimizer(opt, dynamic=False,
                                                  initial_scale=2.)
    self.assertFalse(opt.dynamic)
    self.assertIsNone(opt.dynamic_counter)
    self.assertIsNone(opt.dynamic_growth_steps)

  def testGetScaledLoss(self):
    opt = gradient_descent.SGD(2.0)
    opt = loss_scale_optimizer.LossScaleOptimizer(opt, dynamic=False,
                                                  initial_scale=2.)
    loss = ops.convert_to_tensor_v2_with_dispatch(5.)
    self.assertEqual(10., self.evaluate(opt.get_scaled_loss(loss)))
    self.assertEqual(10., self.evaluate(opt.get_scaled_loss(lambda: loss)()))
    loss = ops.convert_to_tensor_v2_with_dispatch(5., dtype='float16')
    self.assertEqual(10., self.evaluate(opt.get_scaled_loss(loss)))
    self.assertEqual(10., self.evaluate(opt.get_scaled_loss(lambda: loss)()))

  def testGetUnscaledGradients(self):
    opt = gradient_descent.SGD(2.0)
    opt = loss_scale_optimizer.LossScaleOptimizer(opt, dynamic=False,
                                                  initial_scale=2)
    scaled_grads = [
        ops.convert_to_tensor_v2_with_dispatch(3.), None,
        ops.convert_to_tensor_v2_with_dispatch(-4., dtype='float16')
    ]
    grads = opt.get_unscaled_gradients(scaled_grads)
    grads = [self.evaluate(g) if g is not None else g for g in grads]
    self.assertEqual([1.5, None, -2.], grads)

  def testGetUnscaledSparseGradients(self):
    opt = gradient_descent.SGD(2.0)
    opt = loss_scale_optimizer.LossScaleOptimizer(opt, dynamic=False,
                                                  initial_scale=2)
    sparse_scaled_grad = ops.IndexedSlices(
        ops.convert_to_tensor_v2_with_dispatch([[4., 2.], [8., 5.]]),
        ops.convert_to_tensor_v2_with_dispatch([1, 3], dtype='int32'),
        dense_shape=ops.convert_to_tensor_v2_with_dispatch([5, 2],
                                                           dtype='int32'))
    sparse_grad = opt.get_unscaled_gradients([sparse_scaled_grad])[0]
    self.assertIsInstance(sparse_grad, ops.IndexedSlices)
    self.assertAllEqual([[2., 1.], [4., 2.5]],
                        self.evaluate(sparse_grad.values))

  @parameterized.named_parameters(*TESTCASES)
  def testDynamicLossScale(self, strategy_fn):
    strategy = strategy_fn()
    learning_rate = 2.
    expected_gradient = variables.Variable(learning_rate /
                                           strategy.num_replicas_in_sync)
    with strategy.scope():
      var = variables.Variable([5.0])
      opt = gradient_descent.SGD(learning_rate)
      opt = loss_scale_optimizer.LossScaleOptimizer(opt, initial_scale=2,
                                                    dynamic_growth_steps=1)
      self.assertEqual(opt.initial_scale, 2.)
      self.assertIsInstance(opt.initial_scale, float)
      self.assertEqual(opt.dynamic_growth_steps, 1)
      self.assertIsInstance(opt.dynamic_growth_steps, int)

      self.assertEqual(opt.initial_scale % strategy.num_replicas_in_sync, 0)
      run_fn = self._run_fn_with_grad_check(strategy, var, opt,
                                            expected_gradient)
      run_op = strategy.experimental_run(run_fn)
      self.evaluate(variables.global_variables_initializer())
      self._run_if_in_graph_mode(run_op)
      # The loss is the identity of the variable. Therefore the gradient is 1,
      # and so the variable will be init_val - grad * lr == 5 - 1 * 2 == 3
      self.assertAllClose([3.], self.evaluate(var))

      # Loss scale will be double, so the expected gradient is also doubled.
      self.evaluate(expected_gradient.assign(
          2 * learning_rate / strategy.num_replicas_in_sync))
      run_op = strategy.experimental_run(run_fn)
      self._run_if_in_graph_mode(run_op)
      # As before, the 2 is subtracted from the variable, making it's new value
      # 1.
      self.assertAllClose([1.], self.evaluate(var))

  def testDynamicLossScaleDefaultValues(self):
    opt = gradient_descent.SGD()
    opt = loss_scale_optimizer.LossScaleOptimizer(opt)
    self.assertEqual(opt.initial_scale, 2 ** 15)
    self.assertEqual(opt.dynamic_growth_steps, 2000)
    self.evaluate(variables.global_variables_initializer())
    self.assertEqual(self.evaluate(opt.loss_scale), 2 ** 15)

  # pylint: disable=cell-var-from-loop
  @parameterized.named_parameters(*TESTCASES)
  def testClipping(self, strategy_fn):
    strategy = strategy_fn()
    learning_rate = 2.
    for clip_type in ('clipnorm', 'global_clipnorm', 'clipvalue'):
      with strategy.scope(), self.subTest(clip_type=clip_type):
        var = variables.Variable([5.0])
        opt = gradient_descent.SGD(learning_rate, **{clip_type: 2.0})
        opt = loss_scale_optimizer.LossScaleOptimizer(opt, initial_scale=2,
                                                      dynamic_growth_steps=1)
        self.assertEqual(getattr(opt, clip_type), 2.0)
        self.assertEqual(opt.initial_scale % strategy.num_replicas_in_sync, 0)

        loss = lambda: var * 4 / strategy.num_replicas_in_sync
        run_fn = lambda: opt.minimize(loss, var_list=[var])

        # Test running with clipped gradients
        run_op = strategy.experimental_run(run_fn)
        self.evaluate(variables.global_variables_initializer())
        self._run_if_in_graph_mode(run_op)
        # The gradient is 4 but is clipped to 2, so the variable will be
        # init_val - clipped_grad * lr == 5 - 2 * 2 == 1
        self.assertAllClose([1.], self.evaluate(var))
        self.assertEqual(self.evaluate(opt.loss_scale), 4)

        # Test changing the clip amount and running again
        setattr(opt, clip_type, 3.0)
        run_op = strategy.experimental_run(run_fn)
        self._run_if_in_graph_mode(run_op)
        # The gradient is 4 but is clipped to 3, so the variable will be
        # prev_var - clipped_grad * lr == 1 - 3 * 2 == -5
        self.assertAllClose([-5.], self.evaluate(var))
        self.assertEqual(self.evaluate(opt.loss_scale), 8)

        # Test Inf gradients are still skipped instead of being clipped
        loss = lambda: var * float('Inf')
        run_fn = lambda: opt.minimize(loss, var_list=[var])
        run_op = strategy.experimental_run(run_fn)
        self._run_if_in_graph_mode(run_op)
        self.assertAllClose([-5.], self.evaluate(var))  # Var does not change
        self.assertEqual(self.evaluate(opt.loss_scale), 4)
  # pylint: enable=cell-var-from-loop

  @parameterized.named_parameters(*TESTCASES)
  def testDynamicUpdate(self, strategy_fn):
    with strategy_fn().scope() as strategy:
      var = variables.Variable([1.0, 2.0])
      opt = gradient_descent.SGD(1.0)
      opt = loss_scale_optimizer.LossScaleOptimizer(opt, initial_scale=2,
                                                    dynamic_growth_steps=1)

      # Test optimizer with finite gradients
      loss = lambda: var * 2.0 / strategy.num_replicas_in_sync
      run_fn = lambda: opt.minimize(loss, var_list=[var])
      run_op = strategy.experimental_run(run_fn)
      self.evaluate(variables.global_variables_initializer())
      self._run_if_in_graph_mode(run_op)
      # Gradient is 2, so variable will have 2 subtracted from it
      self.assertAllClose([-1.0, 0.0], self.evaluate(var))
      # Loss scale has doubled from 2 to 4
      self.assertEqual(4., self.evaluate(opt.loss_scale))

      # Test optimizer with NaN gradients
      loss = lambda: var * float('NaN')
      run_fn = lambda: opt.minimize(loss, var_list=[var])
      run_op = strategy.experimental_run(run_fn)
      self._run_if_in_graph_mode(run_op)
      # Variable should not change from before, due to NaN gradients.
      self.assertAllClose(self.evaluate(var), [-1.0, 0.0])
      # Loss scale should half due to NaN gradients.
      self.assertEqual(2., self.evaluate(opt.loss_scale))

  @parameterized.named_parameters(*TESTCASES)
  def testDynamicLossScaleWithFloat16Loss(self, strategy_fn):
    strategy = strategy_fn()
    learning_rate = 2.
    with strategy.scope():
      var = variables.Variable([5.0])
      opt = gradient_descent.SGD(learning_rate)
      opt = loss_scale_optimizer.LossScaleOptimizer(opt, initial_scale=2,
                                                    dynamic_growth_steps=1)

      def loss():
        return math_ops.cast(var / strategy.num_replicas_in_sync, 'float16')
      run_fn = lambda: opt.minimize(loss, var_list=[var])
      run_op = strategy.experimental_run(run_fn)
      self.evaluate(variables.global_variables_initializer())
      self._run_if_in_graph_mode(run_op)
      # The loss is the identity of the variable. Therefore the gradient is 1,
      # and so the variable will be init_val - grad * lr == 5 - 1 * 2 == 3
      self.assertAllClose([3.], self.evaluate(var))

  def testNanOnOneReplicaOnly(self):
    if not test_util.is_gpu_available():
      self.skipTest('Test requires GPU')
    if (not context.executing_eagerly() and
        not control_flow_v2_toggles.control_flow_v2_enabled()):
      self.skipTest('b/181283011: GradientTape does not work properly with '
                    'V1 control flow, and opt.minimize uses GradientTape')
    with create_mirrored_strategy().scope() as strategy:
      var = variables.Variable([1.0, 2.0])
      opt = gradient_descent.SGD(1.0)
      opt = loss_scale_optimizer.LossScaleOptimizer(opt, initial_scale=2,
                                                    dynamic_growth_steps=2)

      def loss():
        rep_id = (distribution_strategy_context.get_replica_context()
                  .replica_id_in_sync_group)
        # The last element of last replica's gradient is NaN.
        return control_flow_ops.cond(
            constant_op.constant(rep_id == 0), lambda: var * 2.,
            lambda: var * constant_op.constant([1., float('NaN')]))
      run_fn = lambda: opt.minimize(loss, var_list=[var])
      run_op = strategy.experimental_run(run_fn)
      self.evaluate(variables.global_variables_initializer())
      self._run_if_in_graph_mode(run_op)
      # Variable should not change from before, due to NaN gradients.
      self.assertAllClose(self.evaluate(var), [1.0, 2.0])
      # Loss scale should half due to NaN gradients.
      self.assertEqual(1., self.evaluate(opt.loss_scale))

  def testCustomAggregater(self):
    def gradient_aggregator(grads_and_vars):
      # Simulate an all-reduce where a replica has a NaN gradient by setting
      # the last gradient to NaN
      grads_and_vars = list(grads_and_vars)
      last_grad, last_var = grads_and_vars[-1]
      grads_and_vars[-1] = (last_grad * float('NaN'), last_var)
      return grads_and_vars

    var = variables.Variable([1.0, 2.0])
    opt = gradient_descent.SGD(1.0, gradient_aggregator=gradient_aggregator)
    opt = loss_scale_optimizer.LossScaleOptimizer(opt, initial_scale=2,
                                                  dynamic_growth_steps=2)

    loss = lambda: var * 2
    run_op = opt.minimize(loss, var_list=[var])
    self.evaluate(variables.global_variables_initializer())
    self._run_if_in_graph_mode(run_op)
    # Variable should not change from before, due to NaN gradients.
    self.assertAllClose(self.evaluate(var), [1.0, 2.0])
    # Loss scale should half due to NaN gradients.
    self.assertEqual(1., self.evaluate(opt.loss_scale))

  @parameterized.named_parameters(*TESTCASES)
  def testDynamicLossScaleWithSlots(self, strategy_fn):
    strategy_obj = strategy_fn()
    if (isinstance(strategy_obj, mirrored_strategy.MirroredStrategy) and
        control_flow_v2_toggles.control_flow_v2_enabled() and
        not context.executing_eagerly()):
      self.skipTest('b/138667997')
    with strategy_obj.scope() as strategy:
      var = variables.Variable([1.0, 2.0])
      # An SGD optimizer with momentum has slot variables.
      opt = gradient_descent.SGD(1.0, momentum=1.)
      initial_scale = 2.
      opt = loss_scale_optimizer.LossScaleOptimizer(
          opt, initial_scale=initial_scale, dynamic_growth_steps=1)
      loss = lambda: var / strategy.num_replicas_in_sync
      run_fn = lambda: opt.minimize(loss, var_list=[var])
      run_op = strategy.experimental_run(run_fn)
      self.evaluate(variables.global_variables_initializer())
      self._run_if_in_graph_mode(run_op)
      # The momentum accumulator starts at 0 and the gradient is 1. The
      # accumulator is incremented by the gradient, so it is now 1. Then the
      # variable is subtracted by the accumulator, so the variable is subtracted
      # by 1.
      self.assertAllClose([0.0, 1.0], self.evaluate(var))
      self.assertEqual(self.evaluate(opt.loss_scale), initial_scale * 2)

      run_op = strategy.experimental_run(run_fn)
      self._run_if_in_graph_mode(run_op)
      # The momentum accumulator was 1 before this step and the gradient is 1.
      # The accumulator is incremented by the gradient, so it is now 2. Then the
      # variable is subtracted by the accumulator, so the variable is subtracted
      # by 2.
      self.assertAllClose([-2., -1.], self.evaluate(var))
      self.assertEqual(self.evaluate(opt.loss_scale), initial_scale * 4)

      self.assertEqual(opt.get_slot_names(), ['momentum'])

  def testIterations(self):
    opt = gradient_descent.SGD(2.0)
    lso = loss_scale_optimizer.LossScaleOptimizer(opt, dynamic=False,
                                                  initial_scale=10.)
    lso.iterations = 7
    self.assertEqual(lso.iterations, 7)
    self.assertEqual(opt.iterations, 7)

  @parameterized.named_parameters(*TESTCASES)
  def testIterationsIncremented(self, strategy_fn):
    with strategy_fn().scope() as strategy:
      # Test iterations is incremented in opt.minimize.
      opt = gradient_descent.SGD(1.0)
      opt = loss_scale_optimizer.LossScaleOptimizer(opt)
      var = variables.Variable([5.0])
      loss = lambda: var * 2.0 / strategy.num_replicas_in_sync
      run_fn = lambda: opt.minimize(loss, [var])
      run_op = strategy.experimental_run(run_fn)
      self.evaluate(variables.global_variables_initializer())
      self._run_if_in_graph_mode(run_op)
      self.assertEqual(self.evaluate(var), 3.0)  # Grad is 2, so var is 5 - 2
      self.assertEqual(self.evaluate(opt.iterations), 1)

      # Test iterations is incremented in opt.minimize even if gradients aren't
      # applied to variables due to NaN gradients.
      loss = lambda: var * float('NaN')
      run_fn = lambda: opt.minimize(loss, [var])
      run_op = strategy.experimental_run(run_fn)
      self._run_if_in_graph_mode(run_op)
      self.assertEqual(self.evaluate(var), 3.0)
      self.assertEqual(self.evaluate(opt.iterations), 2)

  def testWeightMethods(self):
    with self.test_session():
      var = variables.Variable([1.0])
      opt = gradient_descent.SGD(1.0)
      opt = loss_scale_optimizer.LossScaleOptimizer(opt, initial_scale=2.,
                                                    dynamic_growth_steps=1)
      run_op = opt.minimize(lambda: var * 2, [var])
      self.evaluate(variables.global_variables_initializer())
      self._run_if_in_graph_mode(run_op)

      self.assertLen(opt.weights, 1)  # The 'iterations' weight
      self.assertEqual(self.evaluate(opt.weights[0]), 1)
      self.assertEqual(opt.get_weights()[0], 1)
      self.assertEqual(self.evaluate(opt.variables()[0]), 1)
      opt.set_weights([np.array(2.)])
      self.assertEqual(self.evaluate(opt.variables()[0]), 2)

  def testHyperParametersExposed(self):
    with self.cached_session():
      opt = adam.Adam(learning_rate=1.0, beta_1=0.5, beta_2=0.9)
      lso = loss_scale_optimizer.LossScaleOptimizer(opt)
      # Force hyperparameters to be created
      opt.lr  # pylint: disable=pointless-statement
      self.evaluate(variables.global_variables_initializer())

      self.assertEqual(self.evaluate(lso.beta_1), 0.5)
      self.assertIsInstance(lso.beta_1, variables.Variable)
      self.assertEqual(self.evaluate(lso.lr), 1.0)
      self.assertIs(lso.lr, opt.lr)
      self.assertIs(lso.lr, lso.learning_rate)

      lso.beta_1 = 0.25
      self.assertEqual(self.evaluate(lso.beta_1), 0.25)
      self.assertEqual(self.evaluate(opt.beta_1), 0.25)
      self.assertIs(lso.beta_1, opt.beta_1)
      opt.beta_1 = 0.75
      self.assertEqual(self.evaluate(lso.beta_1), 0.75)
      self.assertEqual(self.evaluate(opt.beta_1), 0.75)
      self.assertIs(lso.beta_1, opt.beta_1)
      lso.lr = 2.0
      self.assertEqual(self.evaluate(lso.lr), 2.0)
      self.assertEqual(self.evaluate(lso.learning_rate), 2.0)
      self.assertEqual(self.evaluate(opt.lr), 2.0)
      self.assertEqual(self.evaluate(opt.learning_rate), 2.0)
      self.assertIs(lso.lr, opt.lr)

      # Test setting attribute that is both attribute on LossScaleOptimizer and
      # hyperparameter on wrapped optimizer.
      class MyOpt(gradient_descent.SGD):

        def __init__(self):
          super().__init__()
          self._set_hyper('loss_scale', 123.)

      opt = MyOpt()
      lso = loss_scale_optimizer.LossScaleOptimizer(opt)
      with self.assertRaises(AttributeError):
        lso.loss_scale = 2.

  def testArbitraryAttributesNotExposed(self):
    opt = gradient_descent.SGD()
    lso = loss_scale_optimizer.LossScaleOptimizer(opt)
    self.assertFalse(opt.nesterov)
    with self.assertRaisesRegex(
        AttributeError,
        "'LossScaleOptimizer' object has no attribute 'nesterov'"):
      lso.nesterov  # pylint: disable=pointless-statement

    lso.nesterov = True
    self.assertTrue(lso.nesterov)
    self.assertFalse(opt.nesterov)

  def testDir(self):
    lso = loss_scale_optimizer.LossScaleOptimizer(gradient_descent.SGD())
    dir_result = dir(lso)
    self.assertIn('learning_rate', dir_result)  # Hyperparameter
    self.assertIn('lr', dir_result)  # Hyperparameter
    self.assertIn('minimize', dir_result)  # Attribute
    self.assertIn('loss_scale', dir_result)  # Attribute
    self.assertNotIn('nesterov', dir_result)  # Attribute on inner optimizer
    self.assertIn('nesterov', dir(lso.inner_optimizer))

  def testApplyGradientsGetsUnwrappedTensors(self):
    # Tests that gradients passed to apply_gradients are not wrapped in a
    # DistributionStrategy wrapper, such as PerReplica, but instead are raw
    # Tensors. Optimizer subclasses that override apply_gradients() expect raw
    # Tensors, even though the base Optimizer can handle PerReplica gradients.

    outer_self = self

    class MyOptimizer(gradient_descent.SGD):

      def apply_gradients(self,
                          grads_and_vars,
                          name=None,
                          experimental_aggregate_gradients=True):
        for grad, _ in grads_and_vars:
          outer_self.assertIsInstance(grad, ops.Tensor)
        return super(MyOptimizer,
                     self).apply_gradients(grads_and_vars, name,
                                           experimental_aggregate_gradients)

    with create_mirrored_strategy().scope() as strategy:
      var = variables.Variable([5.0])
      opt = MyOptimizer(learning_rate=1.0)
      opt = loss_scale_optimizer.LossScaleOptimizer(opt, dynamic=False,
                                                    initial_scale=1)
      loss = lambda: var * 2.0
      run_fn = lambda: opt.minimize(loss, [var])
      strategy.experimental_run(run_fn)

  @parameterized.named_parameters(*TESTCASES)
  def testV1Optimizer(self, strategy_fn):
    strategy = strategy_fn()
    learning_rate = 2.
    with strategy.scope():
      # Test FixedLossScale
      var = variables.Variable([5.0])
      opt = gradient_descent.SGD(learning_rate)
      opt = loss_scale_optimizer.LossScaleOptimizerV1(opt, loss_scale=2)
      self.assertIsInstance(opt.loss_scale, ops.Tensor)
      self.evaluate(variables.global_variables_initializer())
      self.assertEqual(self.evaluate(opt.loss_scale), 2)
      self.assertEqual(opt.initial_scale, 2)
      self.assertIsNone(opt.dynamic_growth_steps)
      run_fn = self._run_fn_with_grad_check(
          strategy, var, opt, 2 / strategy.num_replicas_in_sync)
      run_op = strategy.experimental_run(run_fn)
      self.evaluate(variables.global_variables_initializer())
      self._run_if_in_graph_mode(run_op)
      # The loss is the identity of the variable. Therefore the gradient is 1,
      # and so the variable will be init_val - grad * lr == 5 - 1 * 2 == 3
      self.assertAllClose([3.], self.evaluate(var))

      # Test DynamicLossScale
      var = variables.Variable([5.0])
      opt = gradient_descent.SGD(learning_rate)
      opt = loss_scale_optimizer.LossScaleOptimizerV1(opt, 'dynamic')
      self.assertEqual(opt.initial_scale, 2 ** 15)
      self.assertEqual(opt.dynamic_growth_steps, 2000)
      self.evaluate(variables.global_variables_initializer())
      self.assertEqual(self.evaluate(opt.loss_scale), 2 ** 15)
      for s in strategy.experimental_local_results(opt.dynamic_counter):
        self.assertEqual(self.evaluate(s), 0)

      loss = lambda: var * float('NaN')
      run_fn = lambda: opt.minimize(loss, var_list=[var])
      run_op = strategy.experimental_run(run_fn)
      self.evaluate(variables.global_variables_initializer())
      self._run_if_in_graph_mode(run_op)
      self.assertAllClose([5.], self.evaluate(var))
      self.assertEqual(self.evaluate(opt.loss_scale), 2 ** 14)
      for s in strategy.experimental_local_results(opt.dynamic_counter):
        self.assertEqual(self.evaluate(s), 0)

  @parameterized.named_parameters(*TESTCASES)
  def testPassingV1LossScale(self, strategy_fn):
    strategy = strategy_fn()
    learning_rate = 2.
    with strategy.scope():
      # Test FixedLossScale
      var = variables.Variable([5.0])
      opt = gradient_descent.SGD(learning_rate)
      loss_scale = tf_loss_scale_module.FixedLossScale(2.)
      opt = loss_scale_optimizer.LossScaleOptimizerV1(opt, loss_scale)
      self.assertIsInstance(opt.loss_scale, ops.Tensor)
      self.evaluate(variables.global_variables_initializer())
      self.assertEqual(self.evaluate(opt.loss_scale), 2)
      run_fn = self._run_fn_with_grad_check(
          strategy, var, opt, 2 / strategy.num_replicas_in_sync)
      run_op = strategy.experimental_run(run_fn)
      self.evaluate(variables.global_variables_initializer())
      self._run_if_in_graph_mode(run_op)
      # The loss is the identity of the variable. Therefore the gradient is 1,
      # and so the variable will be init_val - grad * lr == 5 - 1 * 2 == 3
      self.assertAllClose([3.], self.evaluate(var))

      # Test DynamicLossScale
      var = variables.Variable([5.0])
      opt = gradient_descent.SGD(learning_rate)
      loss_scale = tf_loss_scale_module.DynamicLossScale(
          initial_loss_scale=4, increment_period=1, multiplier=2)
      loss_scale._current_loss_scale.assign(2)
      opt = loss_scale_optimizer.LossScaleOptimizerV1(opt, loss_scale)
      self.assertEqual(opt.initial_scale, 4)
      self.assertEqual(opt.dynamic_growth_steps, 1)
      self.evaluate(variables.global_variables_initializer())
      # Current loss scale is not copied so loss scale is reinitialized to 4
      self.assertEqual(self.evaluate(opt.loss_scale), 4)
      for s in strategy.experimental_local_results(opt.dynamic_counter):
        self.assertEqual(self.evaluate(s), 0)

      run_fn = self._run_fn_with_grad_check(
          strategy, var, opt, 4 / strategy.num_replicas_in_sync)
      run_op = strategy.experimental_run(run_fn)
      self.evaluate(variables.global_variables_initializer())
      self._run_if_in_graph_mode(run_op)
      self.assertAllClose([3.], self.evaluate(var))

  def testPassingV1LossScaleErrors(self):
    opt = gradient_descent.SGD()
    loss_scale = tf_loss_scale_module.DynamicLossScale(multiplier=4)
    with self.assertRaisesRegex(
        ValueError, 'When passing a DynamicLossScale to "loss_scale", '
                    'DynamicLossScale.multiplier must be 2. Got: '
                    'DynamicLossScale'):
      loss_scale_optimizer.LossScaleOptimizerV1(opt, loss_scale)

    class MyLossScale(tf_loss_scale_module.LossScale):

      def __call__(self):
        return 1.

      def update(self, grads):
        return None, True

      def get_config(self):
        return {}

    with self.assertRaisesRegex(
        TypeError, 'Passing a LossScale that is not a FixedLossScale or a '
                   'DynamicLossScale is no longer supported. Got:'):
      loss_scale_optimizer.LossScaleOptimizerV1(opt, MyLossScale())

  def testLossScaleDelegationWithWrapper(self):
    # Test learning_rate is exposed when LossScaleOptimizer wraps another
    # wrapper.

    class MyOptimizer(optimizer_v2.OptimizerV2):

      def __init__(self):
        super().__init__('MyOptimizer')
        self.inner_optimizer = adam.Adam(learning_rate=1.0)

      @property
      def learning_rate(self):
        return self.inner_optimizer.learning_rate

      @learning_rate.setter
      def learning_rate(self, value):
        self.inner_optimizer.learning_rate = value

      def get_config(self):
        return {}

    with self.cached_session():
      opt = MyOptimizer()
      opt = loss_scale_optimizer.LossScaleOptimizer(opt)

      # Force hyperparameters to be created
      opt.learning_rate  # pylint: disable=pointless-statement
      self.evaluate(variables.global_variables_initializer())

      self.assertEqual(self.evaluate(opt.learning_rate), 1.0)
      self.assertEqual(
          self.evaluate(opt.inner_optimizer.inner_optimizer.learning_rate), 1.0)
      opt.learning_rate = 2.0
      self.assertEqual(self.evaluate(opt.learning_rate), 2.0)
      self.assertEqual(self.evaluate(
          opt.inner_optimizer.inner_optimizer.learning_rate), 2.0)

  @parameterized.named_parameters({
      'testcase_name': 'SaveAndRestoreBase',
      'strategy_fn': default_strategy_fn,
      'save_with_ls': True,
      'restore_with_ls': True,
  }, {
      'testcase_name': 'SaveAndRestoreDistribute',
      'strategy_fn': create_mirrored_strategy,
      'save_with_ls': True,
      'restore_with_ls': True,
  }, {
      'testcase_name': 'SaveBase',
      'strategy_fn': default_strategy_fn,
      'save_with_ls': True,
      'restore_with_ls': False,
  }, {
      'testcase_name': 'SaveDistribute',
      'strategy_fn': create_mirrored_strategy,
      'save_with_ls': True,
      'restore_with_ls': False,
  }, {
      'testcase_name': 'RestoreBase',
      'strategy_fn': default_strategy_fn,
      'save_with_ls': False,
      'restore_with_ls': True,
  }, {
      'testcase_name': 'RestoreDistribute',
      'strategy_fn': create_mirrored_strategy,
      'save_with_ls': False,
      'restore_with_ls': True,
  })
  def testCheckpoint(self, strategy_fn, save_with_ls, restore_with_ls):

    class MySGD(gradient_descent.SGD):
      """A custom optimizer that tracks an extra variable."""

      def __init__(self, *args, **kwargs):
        super(MySGD, self).__init__(*args, **kwargs)
        self.my_var = variables.Variable(0.)
        self._track_trackable(self.my_var, 'my_var')

    strategy = strategy_fn()
    replicas = strategy.num_replicas_in_sync
    if (isinstance(strategy, mirrored_strategy.MirroredStrategy) and
        not context.executing_eagerly()):
      # TODO(b/121381184): Enable running the test in this case.
      return

    with self.test_session(), strategy.scope():
      # Build and run a simple model.
      var = variables.Variable([2.0])
      opt = inner_opt = MySGD(1., momentum=1.)
      if save_with_ls:
        opt = loss_scale_optimizer.LossScaleOptimizer(opt, initial_scale=1.,
                                                      dynamic_growth_steps=2.)
      run_fn = lambda: opt.minimize(lambda: var / replicas + 1., var_list=[var])
      opt_op = strategy.experimental_run(run_fn)
      self.evaluate(variables.global_variables_initializer())
      self.evaluate(strategy.experimental_local_results(opt_op))

      # Assert values.
      self.assertEqual(self.evaluate(var), 1.)
      if save_with_ls:
        self.assertEqual(self.evaluate(opt.loss_scale), 1.)
        self.assertEqual(self.evaluate(opt.dynamic_counter), 1)
      slot_var = opt.get_slot(var, 'momentum')
      self.assertEqual(self.evaluate(slot_var).item(), -1)
      self.assertEqual(self.evaluate(opt.iterations), 1)

      # Set optimizer variable to check arbitrary optimizer attributes can be
      # saved/restored
      self.evaluate(inner_opt.my_var.assign(1.))

      # Save a checkpoint.
      checkpoint = trackable_utils.Checkpoint(optimizer=opt, var=var)
      prefix = os.path.join(self.get_temp_dir(), 'ckpt')
      save_path = checkpoint.save(prefix)

      # Create new model
      var = variables.Variable([2.0])
      opt = inner_opt = MySGD(1., momentum=1.)
      if restore_with_ls:
        opt = loss_scale_optimizer.LossScaleOptimizer(opt, initial_scale=1.,
                                                      dynamic_growth_steps=2.)

      # Restore new model.
      checkpoint = trackable_utils.Checkpoint(optimizer=opt, var=var)
      status = checkpoint.restore(save_path)
      if save_with_ls:
        status.assert_existing_objects_matched()
      else:
        status.assert_nontrivial_match()

      # Assert restored values. We can only assert in eager mode since the
      # variables are uninitialized in graph mode
      if context.executing_eagerly():
        self.assertEqual(self.evaluate(var), 1.)
        if save_with_ls and restore_with_ls:
          self.assertEqual(self.evaluate(opt.loss_scale), 1.)
          self.assertEqual(self.evaluate(opt.dynamic_counter), 1)
        elif restore_with_ls:
          self.assertEqual(self.evaluate(opt.loss_scale), 1.)
          self.assertEqual(self.evaluate(opt.dynamic_counter), 0)
        self.assertEqual(self.evaluate(opt.iterations), 1)

      # Run the model again.
      run_fn = lambda: opt.minimize(lambda: var / replicas + 1., var_list=[var])
      opt_op = strategy.experimental_run(run_fn)

      # Assert new values.
      self.evaluate(variables.global_variables_initializer())
      status.run_restore_ops()
      self.evaluate(strategy.experimental_local_results(opt_op))
      self.assertEqual(self.evaluate(var), -1)
      slot_var = opt.get_slot(var, 'momentum')
      self.assertEqual(self.evaluate(slot_var).item(), -2)
      self.assertEqual(self.evaluate(opt.iterations), 2)
      self.assertEqual(self.evaluate(inner_opt.my_var), 1)

      # Restore model again to test restoring after slots are created
      status = checkpoint.restore(save_path)
      if save_with_ls and restore_with_ls:
        status.assert_consumed()
      elif save_with_ls:
        status.assert_existing_objects_matched()
      elif restore_with_ls:
        status.assert_nontrivial_match()
      status.run_restore_ops()
      self.assertEqual(self.evaluate(var), 1)
      self.assertEqual(self.evaluate(slot_var).item(), -1)

  @combinations.generate(combinations.combine(
      get_config=['v1', 'v2', 'tf2_3'], from_config=['v1', 'v2']))
  def testGetConfigFixed(self, get_config, from_config):
    # Get a config from LossScaleOptimizerV1, LossScaleOptimizer, or the
    # LossScaleOptimizer from TF 2.3. Then restore the config into a
    # LossScaleOptimizerV1 or LossScaleOptimizer
    opt = gradient_descent.SGD(2., momentum=0.5)
    if get_config == 'v1':
      opt = loss_scale_optimizer.LossScaleOptimizerV1(opt, 2)
      config = opt.get_config()
    elif get_config == 'v2':
      opt = loss_scale_optimizer.LossScaleOptimizer(
          opt, dynamic=False, initial_scale=2)
      config = opt.get_config()
    else:
      self.assertEqual(get_config, 'tf2_3')
      config = {
          'optimizer': {
              'class_name': 'SGD',
              'config': {
                  'learning_rate': 2.0,
                  'momentum': 0.5,
                  'decay': 0.0,
                  'nesterov': False,
                  'name': 'SGD',
              }
          },
          'loss_scale': {
              'class_name': 'FixedLossScale',
              'config': {'loss_scale_value': 2.0}
          },
      }

    if from_config == 'v1':
      opt = loss_scale_optimizer.LossScaleOptimizerV1.from_config(config)
    else:
      self.assertEqual(from_config, 'v2')
      opt = loss_scale_optimizer.LossScaleOptimizer.from_config(config)

    # Force hyperparameters to be created
    opt.lr  # pylint: disable=pointless-statement
    self.evaluate(variables.global_variables_initializer())

    # Test attributes on the optimizer
    self.assertEqual(self.evaluate(opt.lr), 2.)
    self.assertEqual(self.evaluate(opt.inner_optimizer.lr), 2.)
    self.assertEqual(self.evaluate(opt.momentum), 0.5)
    self.assertEqual(self.evaluate(opt.loss_scale), 2.)
    self.assertEqual(opt.initial_scale, 2.)
    self.assertIsNone(opt.dynamic_growth_steps)
    self.assertIsNone(opt.dynamic_counter)
    self.assertFalse(opt.dynamic)

    # Ensure the optimizer can be used
    var = variables.Variable([5.0])
    run_op = self._run_fn_with_grad_check(
        distribution_strategy_context.get_strategy(), var, opt, 2)()
    self.evaluate(variables.global_variables_initializer())
    self._run_if_in_graph_mode(run_op)
    self.assertEqual(self.evaluate(var), [3.])

  @combinations.generate(combinations.combine(
      get_config=['v1', 'v2', 'tf2_3'], from_config=['v1', 'v2']))
  def testGetConfigDynamic(self, get_config, from_config):
    # Get a config from LossScaleOptimizerV1, LossScaleOptimizer, or the
    # LossScaleOptimizer from TF 2.3. Then restore the config into a
    # LossScaleOptimizerV1 or LossScaleOptimizer
    opt = gradient_descent.SGD(2., momentum=0.5)
    if get_config == 'v1':
      loss_scale = tf_loss_scale_module.DynamicLossScale(
          initial_loss_scale=2, increment_period=3)
      opt = loss_scale_optimizer.LossScaleOptimizerV1(opt, loss_scale)
      config = opt.get_config()
    elif get_config == 'v2':
      opt = loss_scale_optimizer.LossScaleOptimizer(opt, initial_scale=2,
                                                    dynamic_growth_steps=3)
      config = opt.get_config()
    else:
      self.assertEqual(get_config, 'tf2_3')
      config = {
          'optimizer': {
              'class_name': 'SGD',
              'config': {
                  'learning_rate': 2.0,
                  'momentum': 0.5,
                  'decay': 0.0,
                  'nesterov': False,
                  'name': 'SGD',
              }
          },
          'loss_scale': {
              'class_name': 'DynamicLossScale',
              'config': {
                  'initial_loss_scale': 2.0,
                  'increment_period': 3,
                  'multiplier': 2.0,
              }
          },
      }

    if from_config == 'v1':
      opt = loss_scale_optimizer.LossScaleOptimizerV1.from_config(config)
    else:
      self.assertEqual(from_config, 'v2')
      opt = loss_scale_optimizer.LossScaleOptimizer.from_config(config)

    # Force hyperparameters to be created
    opt.lr  # pylint: disable=pointless-statement
    self.evaluate(variables.global_variables_initializer())

    # Test attributes on the optimizer
    self.assertEqual(self.evaluate(opt.lr), 2.)
    self.assertEqual(self.evaluate(opt.inner_optimizer.lr), 2.)
    self.assertEqual(self.evaluate(opt.momentum), 0.5)
    self.assertEqual(self.evaluate(opt.loss_scale), 2.)
    self.assertEqual(opt.initial_scale, 2.)
    self.assertEqual(opt.dynamic_growth_steps, 3.)
    self.assertTrue(opt.dynamic)

    # Ensure the optimizer can be used
    var = variables.Variable([5.0])
    run_op = self._run_fn_with_grad_check(
        distribution_strategy_context.get_strategy(), var, opt, 2)()
    self.evaluate(variables.global_variables_initializer())
    self._run_if_in_graph_mode(run_op)
    self.assertEqual(self.evaluate(var), [3.])
    self.assertEqual(self.evaluate(opt.dynamic_counter), 1)

  def test_from_config_with_invalid_multiplier(self):
    config = {
        'optimizer': {
            'class_name': 'SGD',
            'config': {
                'learning_rate': 2.0,
                'momentum': 0.5,
                'decay': 0.0,
                'nesterov': False,
                'name': 'SGD',
            }
        },
        'loss_scale': {
            'class_name': 'DynamicLossScale',
            'config': {
                'initial_loss_scale': 2.0,
                'increment_period': 3,
                'multiplier': 4.0,
            }
        },
    }

    expected_error = ('Cannot deserialize LossScaleOptimizer with a '
                      'DynamicLossScale whose multiplier is not 2. Got '
                      'DynamicLossScale: DynamicLossScale\\(')
    with self.assertRaisesRegex(ValueError, expected_error):
      loss_scale_optimizer.LossScaleOptimizer.from_config(config)
    with self.assertRaisesRegex(ValueError, expected_error):
      loss_scale_optimizer.LossScaleOptimizerV1.from_config(config)

  @parameterized.named_parameters({
      'testcase_name': 'V2',
      'use_v1': False,
  }, {
      'testcase_name': 'V1',
      'use_v1': True,
  },)
  def testSerializationWithBuiltInOptimizer(self, use_v1):
    opt = gradient_descent.SGD(2., momentum=0.5)
    if use_v1:
      loss_scale = tf_loss_scale_module.DynamicLossScale(
          initial_loss_scale=2., increment_period=3.)
      opt = loss_scale_optimizer.LossScaleOptimizerV1(opt, loss_scale)
    else:
      opt = loss_scale_optimizer.LossScaleOptimizer(opt, initial_scale=2.,
                                                    dynamic_growth_steps=3.)
    config = optimizers.serialize(opt)
    opt = optimizers.deserialize(config)
    # Force hyperparameters to be created
    opt.lr  # pylint: disable=pointless-statement
    self.evaluate(variables.global_variables_initializer())

    self.assertEqual(self.evaluate(opt.lr), 2.)
    self.assertEqual(self.evaluate(opt.inner_optimizer.momentum), 0.5)
    self.assertEqual(self.evaluate(opt.loss_scale), 2.)
    self.assertEqual(opt.dynamic_growth_steps, 3.)
    self.assertTrue(opt.dynamic, 4.)
    # Deserializing a LossScaleOptimizer always always results in a V2
    # LossScaleOptimizer, even if serialized with a LossScaleOptimizerV1.
    self.assertAllEqual(type(opt), loss_scale_optimizer.LossScaleOptimizer)

    # Ensure the optimizer can be used
    var = variables.Variable([5.0])
    run_op = self._run_fn_with_grad_check(
        distribution_strategy_context.get_strategy(), var, opt, 2)()
    self.evaluate(variables.global_variables_initializer())
    self._run_if_in_graph_mode(run_op)
    self.assertEqual(self.evaluate(var), [3.])
    self.assertEqual(self.evaluate(opt.dynamic_counter), 1)

  def testSerializationWithCustomOptimizer(self):
    class MySGD(gradient_descent.SGD):

      def __init__(self, *args, **kwargs):
        super(MySGD, self).__init__(*args, **kwargs)
        self.my_attribute = 123

    opt = MySGD(2., momentum=0.5)
    opt = loss_scale_optimizer.LossScaleOptimizer(opt, initial_scale=2.,
                                                  dynamic_growth_steps=3.)
    config = optimizers.serialize(opt)
    custom_objects = {'MySGD': MySGD}
    opt = optimizers.deserialize(config, custom_objects=custom_objects)
    # Force hyperparameters to be created
    opt.lr  # pylint: disable=pointless-statement
    self.evaluate(variables.global_variables_initializer())

    self.assertEqual(self.evaluate(opt.lr), 2.)
    self.assertEqual(self.evaluate(opt.inner_optimizer.momentum), 0.5)
    self.assertEqual(self.evaluate(opt.loss_scale), 2.)
    self.assertEqual(opt.dynamic_growth_steps, 3.)
    self.assertEqual(opt.inner_optimizer.my_attribute, 123)

  def testUnsupportedStrategy(self):
    strategy = central_storage_strategy.CentralStorageStrategy()
    expected_error = (
        'Loss scaling is not supported with the tf.distribute.Strategy: '
        'CentralStorageStrategy. Try using a different Strategy, e.g. a '
        'MirroredStrategy')
    with strategy.scope(), self.assertRaisesRegex(ValueError, expected_error):
      loss_scale_optimizer.LossScaleOptimizer(gradient_descent.SGD())
    opt = loss_scale_optimizer.LossScaleOptimizer(gradient_descent.SGD())
    with strategy.scope():
      var = variables.Variable(1.0)
      loss = lambda: var * 2.0
      run_fn = lambda: opt.minimize(loss, [var])
      with self.assertRaisesRegex(ValueError, expected_error):
        strategy.experimental_run(run_fn)

  def testInvalidArgsWithFixedLossScale(self):
    opt = gradient_descent.SGD()
    with self.assertRaisesRegex(
        ValueError, '"initial_scale" must be specified if "dynamic" is False'):
      loss_scale_optimizer.LossScaleOptimizer(opt, dynamic=False)
    opt = gradient_descent.SGD()
    with self.assertRaisesRegex(
        ValueError, '"dynamic_growth_steps" must be None if "dynamic" is '
                    'False, but got: 2'):
      loss_scale_optimizer.LossScaleOptimizer(
          opt, dynamic=False, initial_scale=1, dynamic_growth_steps=2)

  def testDynamicMustBeBool(self):
    opt = gradient_descent.SGD()
    with self.assertRaisesRegex(
        TypeError, '"dynamic" argument to LossScaleOptimizer.__init__ must be '
                   "a bool, but got: 'dynamic'"):
      loss_scale_optimizer.LossScaleOptimizer(opt, 'dynamic')

  def testErrorWhenNesting(self):
    opt = gradient_descent.SGD()
    opt = loss_scale_optimizer.LossScaleOptimizer(opt)
    with self.assertRaisesRegex(
        TypeError, 'LossScaleOptimizer cannot wrap another LossScaleOptimizer'):
      loss_scale_optimizer.LossScaleOptimizer(opt)

  def testErrorWrappingSameOptimizerMultipleTimes(self):
    inner_opt = gradient_descent.SGD()
    loss_scale_optimizer.LossScaleOptimizer(inner_opt)
    with self.assertRaisesRegex(
        ValueError,
        '"inner_optimizer" is already wrapped by a LossScaleOptimizer.'):
      loss_scale_optimizer.LossScaleOptimizer(inner_opt)


if __name__ == '__main__':
  test.main()
