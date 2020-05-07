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
"""Tests for lsgt.LossScaleGradientTape."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
from tensorflow.python.compat import v2_compat
from tensorflow.python.distribute import distribution_strategy_context
from tensorflow.python.distribute import mirrored_strategy
from tensorflow.python.distribute import values
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import test_combinations
from tensorflow.python.keras.mixed_precision.experimental import autocast_variable
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training.experimental import loss_scale as loss_scale_module
from tensorflow.python.training.experimental import loss_scaling_gradient_tape as lsgt
from tensorflow.python.util import nest


# If called outside any strategy.scope() calls, this will return the default
# strategy.
default_strategy_fn = distribution_strategy_context.get_strategy


def create_mirrored_strategy():
  if context.num_gpus() >= 1:
    return mirrored_strategy.MirroredStrategy(['cpu:0', 'gpu:0'])
  else:
    return mirrored_strategy.MirroredStrategy(['cpu:0'])


class LossScaleGradientTapeTest(test.TestCase, parameterized.TestCase):

  def _run_with_strategy(self, run_fn, strategy, use_tf_function=False):
    """Runs `run_fn` under the DistributionStrategy `strategy`.

    Runs `run_fn` with `strategy.run`. Returns a list of the
    return values of `run_fn`, one per replica.

    Args:
      run_fn: The function to run.
      strategy: The DistributionStrategy to run `run_fn` with.
      use_tf_function: If True, call `run_fn` under a tf.function.

    Returns:
      A list of tensors, each being the return value of `run_fn` from one
      replica. If a nested structure is returned from `run_fn`, returns a
      nested structure, where each element is a list of tensors.
    """
    strategy_fn = lambda: strategy.run(run_fn)
    if use_tf_function:
      strategy_fn = def_function.function(strategy_fn)

    results = strategy_fn()

    def convert_tensor_to_list(tensor):
      if isinstance(tensor, values.DistributedValues):
        return strategy.experimental_local_results(tensor)
      else:
        return [tensor]
    return nest.map_structure(convert_tensor_to_list, results)

  @test_combinations.generate(test_combinations.combine(
      loss_scale=[loss_scale_module.FixedLossScale,
                  loss_scale_module.DynamicLossScale],
      strategy_fn=[default_strategy_fn, create_mirrored_strategy],
      use_tf_function=[True, False]
  ))
  def test_basic_tapes(self, loss_scale, strategy_fn, use_tf_function):
    loss_scale = loss_scale(32)
    strategy = strategy_fn()
    with strategy.scope():
      x = variables.Variable(3.0)
    def run_fn():
      with lsgt.LossScaleGradientTape(loss_scale) as g:
        y = x * x
      return g.gradient(y, x)
    dy_dx_list = self._run_with_strategy(run_fn, strategy, use_tf_function)
    self.assertEqual(loss_scale(), 32)
    for dy_dx in dy_dx_list:
      self.assertEqual(dy_dx, 6.0)

  @test_combinations.generate(test_combinations.combine(
      loss_scale=[loss_scale_module.FixedLossScale,
                  loss_scale_module.DynamicLossScale],
      strategy_fn=[default_strategy_fn, create_mirrored_strategy],
      use_tf_function=[True, False]
  ))
  def test_output_gradients(self, loss_scale, strategy_fn, use_tf_function):
    loss_scale = loss_scale(32)
    strategy = strategy_fn()
    with strategy.scope():
      x = variables.Variable(3.0)
    def run_fn():
      with lsgt.LossScaleGradientTape(loss_scale) as g:
        y = x * x
      return g.gradient(y, x, output_gradients=constant_op.constant(2.0))
    dy_dx_list = self._run_with_strategy(run_fn, strategy_fn(), use_tf_function)
    self.assertEqual(loss_scale(), 32)
    for dy_dx in dy_dx_list:
      self.assertEqual(dy_dx, 12.0)

  @test_combinations.generate(test_combinations.combine(
      loss_scale=[loss_scale_module.FixedLossScale,
                  loss_scale_module.DynamicLossScale],
      strategy_fn=[default_strategy_fn, create_mirrored_strategy],
      use_tf_function=[True, False]
  ))
  def test_multiple_source_types(self, loss_scale, strategy_fn,
                                 use_tf_function):
    loss_scale = loss_scale(32)
    strategy = strategy_fn()
    with strategy.scope():
      x1 = variables.Variable(1.0)  # Distributed variable
      x2 = variables.Variable([1.0, 2.0])  # Distributed non-scalar variable
      # Distributed AutoCastVariable
      x3 = autocast_variable.create_autocast_variable(variables.Variable(2.0))
    x4 = variables.Variable(2.0)  # Non-distributed variable
    x5 = constant_op.constant(2.0)  # Tensor
    def run_fn():
      with lsgt.LossScaleGradientTape(loss_scale) as g:
        g.watch(x5)
        y = x1 * x2 * x3 * x4 * x5
      return g.gradient(y, [x1, x2, x3, x4, x5])
    x1g, x2g, x3g, x4g, x5g = self._run_with_strategy(run_fn, strategy,
                                                      use_tf_function)
    self.assertEqual(loss_scale(), 32)
    for dy_dx1 in x1g:
      self.assertEqual(dy_dx1, 24.0)
    for dy_dx2 in x2g:
      self.assertAllEqual(dy_dx2, [8.0, 8.0])
    for dy_dx3 in x3g:
      self.assertEqual(dy_dx3, 12.0)
    for dy_dx4 in x4g:
      self.assertEqual(dy_dx4, 12.0)
    for dy_dx5 in x5g:
      self.assertEqual(dy_dx5, 12.0)

  @test_combinations.generate(test_combinations.combine(
      loss_scale=[loss_scale_module.FixedLossScale,
                  loss_scale_module.DynamicLossScale],
      strategy_fn=[default_strategy_fn, create_mirrored_strategy],
      use_tf_function=[True, False]
  ))
  def test_loss_scale_of_one(self, loss_scale, strategy_fn,
                             use_tf_function):
    loss_scale = loss_scale(1)
    strategy = strategy_fn()
    with strategy.scope():
      x = variables.Variable(3.0)
    def run_fn():
      with lsgt.LossScaleGradientTape(loss_scale) as g:
        y = x * x
      return g.gradient(y, x)
    dy_dx_list = self._run_with_strategy(run_fn, strategy, use_tf_function)
    self.assertEqual(loss_scale(), 1)
    for dy_dx in dy_dx_list:
      self.assertEqual(dy_dx, 6.0)

  @test_combinations.generate(test_combinations.combine(
      loss_scale=[loss_scale_module.FixedLossScale,
                  loss_scale_module.DynamicLossScale],
      strategy_fn=[default_strategy_fn],
      use_tf_function=[True, False],
      share_loss_scale=[True, False]
  ))
  def test_nested_tapes(self, loss_scale, strategy_fn, use_tf_function,
                        share_loss_scale):
    # TODO(reedwm): Support nested tapes with mirrored strategy. Currently this
    # does not work, as the set of active gradient tapes is a thread-local
    # variable. Mirrored strategy spawns new threads, making the outer gradient
    # tape non-active when using the inner gradient tape.
    outer_loss_scale = loss_scale(32)
    inner_loss_scale = outer_loss_scale if share_loss_scale else loss_scale(32)
    strategy = strategy_fn()
    with strategy.scope():
      x = variables.Variable(3.0)
    def run_fn():
      with lsgt.LossScaleGradientTape(outer_loss_scale) as g:
        with lsgt.LossScaleGradientTape(inner_loss_scale) as gg:
          y = x * x
        dy_dx = gg.gradient(y, x)
      d2y_dx2 = g.gradient(dy_dx, x)
      return dy_dx, d2y_dx2

    dy_dx_list, d2y_dx2_list = self._run_with_strategy(run_fn, strategy_fn(),
                                                       use_tf_function)
    self.assertEqual(outer_loss_scale(), 32)
    self.assertEqual(inner_loss_scale(), 32)
    for dy_dx in dy_dx_list:
      self.assertEqual(dy_dx, 6.0)
    for d2y_dx2 in d2y_dx2_list:
      self.assertEqual(d2y_dx2, 2.0)

  def test_non_persistent_tapes_error(self):
    x = variables.Variable(3.0)
    with lsgt.LossScaleGradientTape(loss_scale_module.FixedLossScale(32),
                                    persistent=False) as g:
      y = x * x
      z = y * y
    g.gradient(z, x)
    with self.assertRaisesRegexp(RuntimeError, 'persistent'):
      g.gradient(y, x)

  @test_combinations.generate(test_combinations.combine(
      loss_scale=[loss_scale_module.FixedLossScale,
                  loss_scale_module.DynamicLossScale],
      strategy_fn=[default_strategy_fn, create_mirrored_strategy],
      use_tf_function=[True, False]
  ))
  def test_persistent_tapes(self, loss_scale, strategy_fn, use_tf_function):
    ls = loss_scale(32)
    strategy = strategy_fn()
    with strategy.scope():
      x = variables.Variable(3.0)
    def run_fn():
      with lsgt.LossScaleGradientTape(ls, persistent=True) as g:
        y = x * x
        z = y * y
      dz_dx = g.gradient(z, x)
      dy_dx = g.gradient(y, x)
      return dz_dx, dy_dx

    dz_dx_list, dy_dx_list = self._run_with_strategy(run_fn, strategy_fn(),
                                                     use_tf_function)
    for dz_dx in dz_dx_list:
      self.assertEqual(dz_dx, 108.0)
    for dy_dx in dy_dx_list:
      self.assertEqual(dy_dx, 6.0)

  @test_combinations.generate(test_combinations.combine(
      loss_scale=[loss_scale_module.FixedLossScale,
                  loss_scale_module.DynamicLossScale],
  ))
  def test_nested_sources(self, loss_scale):
    x = (variables.Variable(19.0), (variables.Variable(8.),
                                    variables.Variable(9.)))
    with lsgt.LossScaleGradientTape(loss_scale(32)) as g:
      y = x * 13
    dy_dx = g.gradient(y, x)
    self.assertEqual(self.evaluate(dy_dx), (13., (13., 13.)))

  @test_combinations.generate(test_combinations.combine(
      loss_scale=[loss_scale_module.FixedLossScale,
                  loss_scale_module.DynamicLossScale],
  ))
  def test_nested_targets(self, loss_scale):
    w = variables.Variable(3.0)
    with lsgt.LossScaleGradientTape(loss_scale(32)) as g:
      x = w * 5
      y = w * 7
      z = w * 11
    grad = g.gradient([x, (y, z)], w)
    self.assertEqual(self.evaluate(grad), 23)

  @test_combinations.generate(test_combinations.combine(
      loss_scale=[loss_scale_module.FixedLossScale,
                  loss_scale_module.DynamicLossScale],
      strategy_fn=[default_strategy_fn, create_mirrored_strategy]
  ))
  def test_different_dtypes(self, loss_scale, strategy_fn):
    loss_scale = loss_scale(32)
    strategy = strategy_fn()
    with strategy.scope():
      x1 = variables.Variable(1.0, dtype='float16')
      x2 = variables.Variable(2.0, dtype='float32')
      x3 = variables.Variable(3.0, dtype='float64')
    def run_fn():
      with lsgt.LossScaleGradientTape(loss_scale) as g:
        y1 = x1 * math_ops.cast(x2, 'float16') * math_ops.cast(x3, 'float16')
        y2 = math_ops.cast(x1, 'float32') * x2 * math_ops.cast(x3, 'float32')
        y3 = math_ops.cast(x1, 'float64') * math_ops.cast(x2, 'float64') * x3
      return g.gradient([y1, y2, y3], [x1, x2, x3])
    dy_dx1_list, dy_dx2_list, dy_dx3_list = self._run_with_strategy(
        run_fn, strategy)
    self.assertEqual(loss_scale(), 32)
    for dy_dx1 in dy_dx1_list:
      self.assertEqual(dy_dx1, 18.0)
      self.assertEqual(dy_dx1.dtype, 'float16')
    for dy_dx2 in dy_dx2_list:
      self.assertEqual(dy_dx2, 9.0)
      self.assertEqual(dy_dx2.dtype, 'float32')
    for dy_dx3 in dy_dx3_list:
      self.assertEqual(dy_dx3, 6.0)
      self.assertEqual(dy_dx3.dtype, 'float64')

  @test_combinations.generate(test_combinations.combine(
      loss_scale=[loss_scale_module.FixedLossScale,
                  loss_scale_module.DynamicLossScale],
      strategy_fn=[default_strategy_fn, create_mirrored_strategy],
      use_tf_function=[True, False]
  ))
  def test_none_gradients(self, loss_scale, strategy_fn, use_tf_function):
    loss_scale = loss_scale(32)
    strategy = strategy_fn()
    with strategy.scope():
      x1 = variables.Variable(2.0)
      x2 = variables.Variable(2.0)
      x3 = variables.Variable(2.0)
      x4 = variables.Variable([2.0, 2.0])
      x5 = constant_op.constant(2.0)
      x6 = constant_op.constant(2.0)
    def run_fn():
      with lsgt.LossScaleGradientTape(loss_scale) as g:
        # x6 will have a None gradient because we do not watch it
        g.watch(x5)
        y = x1 * x3 * x5 * x6
      return g.gradient(y, [x1, x2, [x3, [x4], x5], x6])
    [x1g, x2g, [x3g, [x4g], x5g], x6g] = self._run_with_strategy(
        run_fn, strategy, use_tf_function)
    self.assertEqual(loss_scale(), 32)
    for dy_dx1 in x1g:
      self.assertEqual(dy_dx1, 8.0)
    self.assertEqual(x2g, [None])
    for dy_dx3 in x3g:
      self.assertEqual(dy_dx3, 8.0)
    self.assertEqual(x4g, [None])
    for dy_dx5 in x5g:
      self.assertEqual(dy_dx5, 8.0)
    self.assertEqual(x6g, [None])

  @test_combinations.generate(test_combinations.combine(
      loss_scale=[loss_scale_module.FixedLossScale,
                  loss_scale_module.DynamicLossScale],
      strategy_fn=[default_strategy_fn, create_mirrored_strategy],
      use_tf_function=[True, False]
  ))
  def test_zero_gradients(self, loss_scale, strategy_fn, use_tf_function):
    loss_scale = loss_scale(32)
    strategy = strategy_fn()
    with strategy.scope():
      x = variables.Variable(0.0)
    def run_fn():
      with lsgt.LossScaleGradientTape(loss_scale) as g:
        y = x * x
      return g.gradient(y, x)
    dy_dx_list = self._run_with_strategy(run_fn, strategy, use_tf_function)
    self.assertEqual(loss_scale(), 32)
    for dy_dx in dy_dx_list:
      # Assert zero gradients are not turned into Nones
      self.assertEqual(dy_dx, 0.0)

  @test_combinations.generate(test_combinations.combine(
      loss_scale=[loss_scale_module.FixedLossScale,
                  loss_scale_module.DynamicLossScale],
      strategy_fn=[default_strategy_fn, create_mirrored_strategy],
      non_finite_term=[np.inf, np.nan],
  ))
  def test_scaling_non_finite_gradient(self, loss_scale, strategy_fn,
                                       non_finite_term):
    loss_scale = loss_scale(32)
    x = variables.Variable(1.0)
    def run_fn():
      with lsgt.LossScaleGradientTape(loss_scale) as g:
        y = x * non_finite_term
      return g.gradient(y, x)

    dy_dx_list = self._run_with_strategy(run_fn, strategy_fn())
    check_fn = np.isposinf if non_finite_term == np.inf else np.isnan
    for dy_dx in dy_dx_list:
      self.assertTrue(check_fn(dy_dx))

  @test_combinations.generate(test_combinations.combine(
      strategy_fn=[default_strategy_fn, create_mirrored_strategy],
      non_finite_term=[np.inf, np.nan],
      use_tf_function=[True, False],
  ))
  def test_dynamic_scale_to_one_on_non_finite_gradient(
      self, strategy_fn, non_finite_term, use_tf_function):
    loss_scale = loss_scale_module.DynamicLossScale(initial_loss_scale=32)
    strategy = strategy_fn()
    with strategy.scope():
      x = variables.Variable(3.0)
    def run_fn():
      with lsgt.LossScaleGradientTape(loss_scale) as g:
        y = x * non_finite_term
      g.gradient(y, x)

    self._run_with_strategy(run_fn, strategy, use_tf_function)
    self.assertEqual(self.evaluate(loss_scale()), 1.0)

  @test_combinations.generate(test_combinations.combine(
      use_tf_function=[True, False],
  ))
  def test_dynamic_scale_to_one_on_non_finite_gradient_on_last_replica(
      self, use_tf_function):
    if context.num_gpus() < 1:
      # Requires the mirrored strategy to have two replicas: one on the CPU and
      # one on the GPU
      self.skipTest('Test requires at least 1 GPU')
    loss_scale = loss_scale_module.DynamicLossScale(initial_loss_scale=32)
    strategy = create_mirrored_strategy()
    with strategy.scope():
      x = variables.Variable(3.0)
    def run_fn():
      with lsgt.LossScaleGradientTape(loss_scale) as g:
        # The gradient will be finite on the first replica, and infinite on the
        # second
        rep_ctx = distribution_strategy_context.get_replica_context()
        if rep_ctx.replica_id_in_sync_group == rep_ctx.num_replicas_in_sync - 1:
          y = x * np.inf
        else:
          y = x * 2
      return g.gradient(y, x)

    replica0_grad, replica1_grad = self._run_with_strategy(
        run_fn, strategy, use_tf_function)
    self.assertEqual(self.evaluate(loss_scale()), 1.0)
    self.assertEqual(replica0_grad, 2.0)
    self.assertEqual(replica1_grad, np.inf)

  @test_combinations.generate(test_combinations.combine(
      strategy_fn=[default_strategy_fn, create_mirrored_strategy],
      non_finite_term=[np.inf, np.nan],
  ))
  def test_fixed_scaling_no_change_non_finite_gradient(self, strategy_fn,
                                                       non_finite_term):
    loss_scale = loss_scale_module.FixedLossScale(32)
    strategy = strategy_fn()
    with strategy.scope():
      x = variables.Variable(3.0)
    def run_fn():
      with lsgt.LossScaleGradientTape(loss_scale) as g:
        y = x * non_finite_term
      return g.gradient(y, x)

    dy_dx_list = self._run_with_strategy(run_fn, strategy)
    check_fn = np.isposinf if non_finite_term == np.inf else np.isnan
    for dy_dx in dy_dx_list:
      self.assertTrue(check_fn(self.evaluate(dy_dx)))
    self.assertEqual(self.evaluate(loss_scale()), 32.0)

  @test_combinations.generate(test_combinations.combine(
      strategy_fn=[default_strategy_fn, create_mirrored_strategy],
      use_tf_function=[True, False]
  ))
  def test_dynamic_loss_scaling_down_loop(self, strategy_fn, use_tf_function):
    loss_scale = loss_scale_module.DynamicLossScale(initial_loss_scale=32)
    strategy = strategy_fn()
    with strategy.scope():
      x = variables.Variable(3.0)
    def run_fn():
      with lsgt.LossScaleGradientTape(loss_scale) as g:
        y = x * (3.0 * (10**37))  # grad will be inf after scaling
      return g.gradient(y, x)

    dy_dx_list = self._run_with_strategy(run_fn, strategy, use_tf_function)
    self.assertEqual(self.evaluate(loss_scale()), 8.0)
    for dy_dx in dy_dx_list:
      self.assertAllClose(self.evaluate(dy_dx), (3.0 * (10**37)), atol=1e-06)

  @test_combinations.generate(test_combinations.combine(
      strategy_fn=[default_strategy_fn, create_mirrored_strategy],
      use_tf_function=[True, False]
  ))
  def test_dynamic_loss_scaling_inf_target_post_scale(self, strategy_fn,
                                                      use_tf_function):
    loss_scale = loss_scale_module.DynamicLossScale(initial_loss_scale=32.0)
    strategy = strategy_fn()
    with strategy.scope():
      x = variables.Variable(3.0 * (10**37))
    def run_fn():
      with lsgt.LossScaleGradientTape(loss_scale) as g:
        y = x * 3.0  # target will be inf after scaling
      return g.gradient(y, x)

    dy_dx_list = self._run_with_strategy(run_fn, strategy, use_tf_function)
    self.assertEqual(self.evaluate(loss_scale()), 32.0)
    for dy_dx in dy_dx_list:
      self.assertAllClose(self.evaluate(dy_dx), 3.0)

  @test_combinations.generate(
      test_combinations.combine(
          loss_scale=[
              loss_scale_module.FixedLossScale,
              loss_scale_module.DynamicLossScale
          ],
          strategy_fn=[default_strategy_fn, create_mirrored_strategy],
          use_tf_function=[True, False]))
  def test_transpose(self, loss_scale, strategy_fn, use_tf_function):
    # Calling tf.transpose insde a tf.function can cause static shape
    # information to be lost. This tests that LossScaleGradientTape can handle
    # this.
    loss_scale = loss_scale(32)
    strategy = strategy_fn()
    with strategy.scope():
      x = variables.Variable(array_ops.ones((2, 3)))

    def run_fn():
      with lsgt.LossScaleGradientTape(loss_scale) as g:
        y = array_ops.transpose(x) * 2.
      return g.gradient(y, x)

    dy_dx_list = self._run_with_strategy(run_fn, strategy, use_tf_function)
    self.assertEqual(loss_scale(), 32)
    for dy_dx in dy_dx_list:
      self.assertAllEqual(dy_dx, np.full((2, 3), 2.))

  def test_passing_non_loss_scale_raises_error(self):
    with self.assertRaisesRegexp(
        ValueError,
        '`loss_scale` must be an instance of LossScale, but got: 2.0'):
      lsgt.LossScaleGradientTape(2.0)

  def test_jacobian_raises_error(self):
    loss_scale = loss_scale_module.FixedLossScale(2.)
    x = variables.Variable([1.0, 2.0])
    with lsgt.LossScaleGradientTape(loss_scale) as g:
      y = x * 2
    with self.assertRaisesRegexp(
        NotImplementedError,
        'LossScaleGradientTape.jacobian is not yet implemented'):
      g.jacobian(y, x)

    x = variables.Variable([[1.0, 2.0], [3.0, 4.0]])
    with lsgt.LossScaleGradientTape(loss_scale) as g:
      y = x * 2
    with self.assertRaisesRegexp(
        NotImplementedError,
        'LossScaleGradientTape.batch_jacobian is not yet implemented'):
      g.batch_jacobian(y, x)


if __name__ == '__main__':
  v2_compat.enable_v2_behavior()
  test.main()
