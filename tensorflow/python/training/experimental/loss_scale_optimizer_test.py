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
"""Tests for MixedPrecisionLossScaleOptimizer."""

import os

from absl.testing import parameterized
from tensorflow.python.checkpoint import checkpoint as trackable_utils
from tensorflow.python.distribute import distribution_strategy_context
from tensorflow.python.distribute import mirrored_strategy
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import custom_gradient
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import gradient_descent
from tensorflow.python.training import momentum
from tensorflow.python.training.experimental import loss_scale as loss_scale_module
from tensorflow.python.training.experimental import loss_scale_optimizer

# If called outside any strategy.scope() calls, this will return the default
# strategy.
default_strategy_fn = distribution_strategy_context.get_strategy


def create_mirrored_strategy():
  if context.num_gpus() >= 1:
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


def get_gradients(opt, loss, params):
  grads_and_vars = opt.compute_gradients(loss, params)
  grads, _ = zip(*grads_and_vars)
  return grads


def create_identity_with_grad_check_fn(expected_gradient, expected_dtype=None):
  """Returns a function that asserts it's gradient has a certain value.

  This serves as a hook to assert intermediate gradients have a certain value.
  This returns an identity function. The identity's gradient function is also
  the identity function, except it asserts that the gradient equals
  `expected_gradient` and has dtype `expected_dtype`.

  Args:
    expected_gradient: The gradient function asserts that the gradient is this
      value.
    expected_dtype: The gradient function asserts the gradient has this dtype.

  Returns:
    An identity function whose gradient function asserts the gradient has a
    certain value.
  """
  @custom_gradient.custom_gradient
  def _identity_with_grad_check(x):
    """Function that asserts it's gradient has a certain value."""
    x = array_ops.identity(x)
    def grad(dx):
      """Gradient function that asserts the gradient has a certain value."""
      if expected_dtype:
        assert dx.dtype == expected_dtype, (
            'dx.dtype should be %s but is: %s' % (expected_dtype, dx.dtype))
      expected_tensor = tensor_conversion.convert_to_tensor_v2(
          expected_gradient, dtype=dx.dtype, name='expected_gradient'
      )
      # Control dependency is to ensure input is available. It's possible the
      # dataset will throw a StopIteration to indicate there is no more data, in
      # which case we don't want to run the assertion.
      with ops.control_dependencies([x]):
        assert_op = check_ops.assert_equal(dx, expected_tensor)
      with ops.control_dependencies([assert_op]):
        dx = array_ops.identity(dx)
      return dx
    return x, grad
  # Keras sometimes has trouble serializing Lambda layers with a decorated
  # function. So we define and return a non-decorated function.
  def identity_with_grad_check(x):
    return _identity_with_grad_check(x)
  return identity_with_grad_check


class MixedPrecisionLossScaleOptimizerTest(test.TestCase,
                                           parameterized.TestCase):

  def _run_if_in_graph_mode(self, val):
    # Running only in graph mode is useful, because optimizers sometimes return
    # a value that, in Graph mode, is runnable with self.evaluate. But in Eager
    # mode, the optimizer already does the computations and the return value
    # cannot be run.
    if not context.executing_eagerly():
      self.evaluate(val)

  def _run_fn_with_grad_check(self, strategy, var, opt, expected_grad):
    grad_check_fn = create_identity_with_grad_check_fn(
        expected_grad)
    loss = lambda: grad_check_fn(var) / strategy.num_replicas_in_sync
    return lambda: opt.minimize(loss, var_list=[var])

  @parameterized.named_parameters(*TESTCASES)
  @test_util.run_in_graph_and_eager_modes
  def testFixedLossScaleAppliedToLossWithMinimize(self, strategy_fn):
    with strategy_fn().scope() as strategy:
      var = variables.Variable([5.0])
      opt = gradient_descent.GradientDescentOptimizer(2.0)
      loss_scale = 10.
      opt = loss_scale_optimizer.MixedPrecisionLossScaleOptimizer(
          opt, loss_scale)
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

  @test_util.deprecated_graph_mode_only
  def testFixedLossScaleAppliedToLossWithGetGradients(self):
    var = variables.Variable([2.0])
    opt = gradient_descent.GradientDescentOptimizer(1.0)
    loss_scale = 10.
    opt = loss_scale_optimizer.MixedPrecisionLossScaleOptimizer(opt, loss_scale)
    grad_check_fn = create_identity_with_grad_check_fn(loss_scale)
    loss = grad_check_fn(var)
    run_op = get_gradients(opt, loss, [var])
    self.evaluate(variables.global_variables_initializer())
    # This will cause an assertion to run, as
    # create_identity_with_grad_check_fn added an assertion op.
    self.evaluate(run_op)

  @parameterized.named_parameters(*TESTCASES)
  @test_util.run_in_graph_and_eager_modes
  def testDynamicLossScale(self, strategy_fn):
    strategy = strategy_fn()
    learning_rate = 2.
    expected_gradient = resource_variable_ops.ResourceVariable(
        learning_rate / strategy.num_replicas_in_sync)
    with strategy.scope():
      var = variables.Variable([5.0])
      opt = gradient_descent.GradientDescentOptimizer(learning_rate)
      loss_scale = loss_scale_module.DynamicLossScale(
          initial_loss_scale=2, increment_period=1, multiplier=2)
      opt = loss_scale_optimizer.MixedPrecisionLossScaleOptimizer(
          opt, loss_scale)
      self.assertEqual(
          loss_scale.initial_loss_scale % strategy.num_replicas_in_sync, 0)

      run_fn = self._run_fn_with_grad_check(strategy, var, opt,
                                            expected_gradient)
      run_op = strategy.experimental_run(run_fn)
      self.evaluate(variables.global_variables_initializer())
      self._run_if_in_graph_mode(run_op)
      # The loss is the identity of the variable. Therefore the gradient is 1,
      # and so the variable will be init_val - grad * lr == 5 - 1 * 2 == 3
      self.assertAllClose([3.], self.evaluate(var))

      # Loss scale will be double, so the expected gradient is also doubled.
      self.evaluate(
          expected_gradient.assign(2 * learning_rate /
                                   strategy.num_replicas_in_sync))
      run_op = strategy.experimental_run(run_fn)
      self._run_if_in_graph_mode(run_op)
      # As before, the 2 is subtracted from the variable, making it's new value
      # 1.
      self.assertAllClose([1.], self.evaluate(var))

  @parameterized.named_parameters(*TESTCASES)
  @test_util.run_in_graph_and_eager_modes
  def testDynamicUpdate(self, strategy_fn):
    with strategy_fn().scope() as strategy:
      var = variables.Variable([1.0, 2.0])
      opt = gradient_descent.GradientDescentOptimizer(1.0)
      loss_scale = loss_scale_module.DynamicLossScale(
          initial_loss_scale=2, increment_period=1, multiplier=2)
      opt = loss_scale_optimizer.MixedPrecisionLossScaleOptimizer(
          opt, loss_scale)

      # Test optimizer with finite gradients
      loss = lambda: var * 2.0 / strategy.num_replicas_in_sync
      run_fn = lambda: opt.minimize(loss, var_list=[var])
      run_op = strategy.experimental_run(run_fn)
      self.evaluate(variables.global_variables_initializer())
      self._run_if_in_graph_mode(run_op)
      # Gradient is 2, so variable will have 2 subtracted from it
      self.assertAllClose([-1.0, 0.0], self.evaluate(var))
      # Loss scale has doubled from 2 to 4
      self.assertEqual(4., self.evaluate(opt._loss_scale()))

      # Test optimizer with NaN gradients
      loss = lambda: var * float('NaN')
      run_fn = lambda: opt.minimize(loss, var_list=[var])
      run_op = strategy.experimental_run(run_fn)
      self._run_if_in_graph_mode(run_op)
      # Variable should not change from before, due to NaN gradients.
      self.assertAllClose(self.evaluate(var), [-1.0, 0.0])
      # Loss scale should half due to NaN gradients.
      self.assertEqual(2., self.evaluate(opt._loss_scale()))

  @parameterized.named_parameters(*TESTCASES)
  @test_util.run_in_graph_and_eager_modes
  def testDynamicLossScaleWithSlots(self, strategy_fn):
    with strategy_fn().scope() as strategy:
      var = variables.Variable([1.0, 2.0])
      # An SGD optimizer with momentum has slot variables.
      opt = momentum.MomentumOptimizer(1.0, momentum=1.)
      initial_loss_scale = 2.
      loss_scale = loss_scale_module.DynamicLossScale(
          initial_loss_scale=initial_loss_scale,
          increment_period=1,
          multiplier=4)
      opt = loss_scale_optimizer.MixedPrecisionLossScaleOptimizer(
          opt, loss_scale)
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
      self.assertEqual(self.evaluate(opt._loss_scale()), initial_loss_scale * 4)

      run_op = strategy.experimental_run(run_fn)
      self._run_if_in_graph_mode(run_op)
      # The momentum accumulator was 1 before this step and the gradient is 1.
      # The accumulator is incremented by the gradient, so it is now 2. Then the
      # variable is subtracted by the accumulator, so the variable is subtracted
      # by 2.
      self.assertAllClose([-2., -1.], self.evaluate(var))
      self.assertEqual(
          self.evaluate(opt._loss_scale()), initial_loss_scale * 16)

  @parameterized.named_parameters(*TESTCASES)
  @test_util.run_in_graph_and_eager_modes
  def testCheckpoint(self, strategy_fn):
    strategy = strategy_fn()
    if (isinstance(strategy, mirrored_strategy.MirroredStrategy) and
        not context.executing_eagerly()):
      # TODO(b/121381184): Enable running the test in this case.
      return

    with self.test_session(), strategy.scope():
      # Build and run a simple model.
      var = variables.Variable([2.0])
      loss_scale = loss_scale_module.DynamicLossScale(
          initial_loss_scale=1., increment_period=2., multiplier=2.)
      opt = momentum.MomentumOptimizer(1.0, momentum=1.)
      opt = loss_scale_optimizer.MixedPrecisionLossScaleOptimizer(
          opt, loss_scale)
      run_fn = lambda: opt.minimize(lambda: var + 1., var_list=[var])
      opt_op = strategy.experimental_run(run_fn)
      self.evaluate(variables.global_variables_initializer())
      self.evaluate(opt_op)
      self.assertEqual(self.evaluate(loss_scale()), 1.)
      self.assertEqual(self.evaluate(loss_scale._num_good_steps), 1)

      # Save a checkpoint.
      checkpoint = trackable_utils.Checkpoint(optimizer=opt)
      prefix = os.path.join(self.get_temp_dir(), 'ckpt')
      save_path = checkpoint.save(prefix)

      # Run model again.
      self.evaluate(strategy.experimental_run(run_fn))
      self.assertEqual(self.evaluate(loss_scale()), 2.)
      self.assertEqual(self.evaluate(loss_scale._num_good_steps), 0)

      # Load checkpoint and ensure loss scale is back to it's original value.
      status = checkpoint.restore(save_path)
      status.assert_consumed()
      status.run_restore_ops()
      self.assertEqual(self.evaluate(loss_scale()), 1.)
      self.assertEqual(self.evaluate(loss_scale._num_good_steps), 1)

  def testPassingNoneToLossScale(self):
    opt = gradient_descent.GradientDescentOptimizer(1.0)
    with self.assertRaisesRegex(ValueError, r'loss_scale cannot be None'):
      loss_scale_optimizer.MixedPrecisionLossScaleOptimizer(opt, None)


if __name__ == '__main__':
  test.main()
