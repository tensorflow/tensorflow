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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

from tensorflow.python.distribute import mirrored_strategy
from tensorflow.python.distribute import one_device_strategy
from tensorflow.python.eager import context
from tensorflow.python.framework import test_util
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.mixed_precision.experimental import loss_scale_optimizer
from tensorflow.python.keras.mixed_precision.experimental import test_util as mp_test_util
from tensorflow.python.keras.optimizer_v2 import gradient_descent
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


def create_one_device_strategy():
  return one_device_strategy.OneDeviceStrategy('cpu:0')


def create_mirrored_strategy():
  if context.num_gpus() >= 1:
    return mirrored_strategy.MirroredStrategy(['cpu:0', 'gpu:0'])
  else:
    return mirrored_strategy.MirroredStrategy(['cpu:0'])


TESTCASES = ({
    'testcase_name': 'Base',
    'strategy_fn': create_one_device_strategy
}, {
    'testcase_name': 'Distribute',
    'strategy_fn': create_mirrored_strategy
})


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
  @test_util.run_in_graph_and_eager_modes
  def testLossScaleAppliedToLossWithMinimize(self, strategy_fn):
    with strategy_fn().scope() as strategy:
      var = variables.Variable([5.0])
      opt = gradient_descent.SGD(2.0)
      loss_scale = 10.
      opt = loss_scale_optimizer.LossScaleOptimizer(opt, loss_scale)
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
  def testLossScaleAppliedToLossWithGetGradientsTest(self):
    var = variables.Variable([2.0])
    opt = gradient_descent.SGD(1.0)
    loss_scale = 10.
    opt = loss_scale_optimizer.LossScaleOptimizer(opt, loss_scale)
    grad_check_fn = mp_test_util.create_identity_with_grad_check_fn(loss_scale)
    loss = grad_check_fn(var)
    run_op = opt.get_gradients(loss, [var])
    self.evaluate(variables.global_variables_initializer())
    # This will cause an assertion to run, as
    # mp_test_util.create_identity_with_grad_check_fn added an assertion op.
    self.evaluate(run_op)

  def testInvalidConstructorArguments(self):
    with self.assertRaisesRegexp(ValueError,
                                 'must be an instance of OptimizerV2'):
      loss_scale_optimizer.LossScaleOptimizer(optimizers.SGD(), 10.)

    with self.assertRaisesRegexp(ValueError, 'does not support wrapping '
                                             'optimizers with a clipnorm'):
      loss_scale_optimizer.LossScaleOptimizer(
          gradient_descent.SGD(1.0, clipnorm=1.0), 10.)

    with self.assertRaisesRegexp(ValueError, 'does not support wrapping '
                                             'optimizers with a clipvalue'):
      loss_scale_optimizer.LossScaleOptimizer(
          gradient_descent.SGD(1.0, clipvalue=1.0), 10.)


if __name__ == '__main__':
  test.main()
