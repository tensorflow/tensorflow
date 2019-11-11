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

    Runs `run_fn` with `strategy.experimental_run_v2`. Returns a list of the
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
    strategy_fn = lambda: strategy.experimental_run_v2(run_fn)
    if use_tf_function:
      strategy_fn = def_function.function(strategy_fn)

    results = strategy_fn()

    def convert_tensor_to_list(tensor):
      if isinstance(tensor, values.DistributedValues):
        return tensor.values
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
    def run_fn():
      x = constant_op.constant(3.0)
      with lsgt.LossScaleGradientTape(loss_scale) as g:
        g.watch(x)
        y = x * x
      return g.gradient(y, x)
    dy_dx_list = self._run_with_strategy(run_fn, strategy_fn(), use_tf_function)
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
    def run_fn():
      x = constant_op.constant(3.0)
      with lsgt.LossScaleGradientTape(loss_scale) as g:
        g.watch(x)
        y = x * x
      return g.gradient(y, x, output_gradients=constant_op.constant(2.0))
    dy_dx_list = self._run_with_strategy(run_fn, strategy_fn(), use_tf_function)
    self.assertEqual(loss_scale(), 32)
    for dy_dx in dy_dx_list:
      self.assertEqual(dy_dx, 12.0)

  @test_combinations.generate(test_combinations.combine(
      loss_scale=[loss_scale_module.FixedLossScale,
                  loss_scale_module.DynamicLossScale],
      strategy_fn=[default_strategy_fn],
      use_tf_function=[True, False]
  ))
  def test_nested_tapes(self, loss_scale, strategy_fn, use_tf_function):
    # TODO(reedwm): Support nested tapes with mirrored strategy. Currently this
    # does not work, as the set of active gradient tapes is a thread-local
    # variable. Mirrored strategy spawns new threads, making the outer gradient
    # tape non-active when using the inner gradient tape.
    outer_loss_scale = loss_scale(32)
    inner_loss_scale = loss_scale(32)
    def run_fn():
      x = constant_op.constant(3.0)
      with lsgt.LossScaleGradientTape(outer_loss_scale) as g:
        g.watch(x)
        with lsgt.LossScaleGradientTape(inner_loss_scale) as gg:
          gg.watch(x)
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
    x = constant_op.constant(3.0)
    with lsgt.LossScaleGradientTape(loss_scale_module.FixedLossScale(32),
                                    persistent=False) as g:
      g.watch(x)
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
    def run_fn():
      x = constant_op.constant(3.0)
      with lsgt.LossScaleGradientTape(ls, persistent=True) as g:
        g.watch(x)
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
    x = (constant_op.constant(19.0), (constant_op.constant(8.),
                                      constant_op.constant(9.)))
    with lsgt.LossScaleGradientTape(loss_scale(32)) as g:
      g.watch(x)
      y = x * 13
    dy_dx = g.gradient(y, x)
    self.assertEqual(self.evaluate(dy_dx), (13., (13., 13.)))

  @test_combinations.generate(test_combinations.combine(
      loss_scale=[loss_scale_module.FixedLossScale,
                  loss_scale_module.DynamicLossScale],
  ))
  def test_nested_targets(self, loss_scale):
    w = constant_op.constant(3.0)
    with lsgt.LossScaleGradientTape(loss_scale(32)) as g:
      g.watch(w)
      x = w * 5
      y = w * 7
      z = w * 11
    grad = g.gradient([x, (y, z)], w)
    self.assertEqual(self.evaluate(grad), 23)

  @test_combinations.generate(test_combinations.combine(
      loss_scale=[loss_scale_module.FixedLossScale,
                  loss_scale_module.DynamicLossScale],
      strategy_fn=[default_strategy_fn, create_mirrored_strategy],
      non_finite_term=[np.inf, np.nan],
  ))
  def test_scaling_non_finite_gradient(self, loss_scale, strategy_fn,
                                       non_finite_term):
    loss_scale = loss_scale(32)
    def run_fn():
      x = constant_op.constant(1.0)
      with lsgt.LossScaleGradientTape(loss_scale) as g:
        g.watch(x)
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
    def run_fn():
      x = constant_op.constant(1.0)
      with lsgt.LossScaleGradientTape(loss_scale) as g:
        g.watch(x)
        y = x * non_finite_term
      g.gradient(y, x)

    self._run_with_strategy(run_fn, strategy_fn(), use_tf_function)
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
    def run_fn():
      x = constant_op.constant(1.0)
      with lsgt.LossScaleGradientTape(loss_scale) as g:
        g.watch(x)
        # The gradient will be finite on the first replica, and infinite on the
        # second
        rep_ctx = distribution_strategy_context.get_replica_context()
        if rep_ctx.replica_id_in_sync_group == rep_ctx.num_replicas_in_sync - 1:
          y = x * np.inf
        else:
          y = x * 2
      return g.gradient(y, x)

    replica0_grad, replica1_grad = self._run_with_strategy(
        run_fn, create_mirrored_strategy(), use_tf_function)
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
    def run_fn():
      x = constant_op.constant(1.0)
      with lsgt.LossScaleGradientTape(loss_scale) as g:
        g.watch(x)
        y = x * non_finite_term
      return g.gradient(y, x)

    dy_dx_list = self._run_with_strategy(run_fn, strategy_fn())
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
    def run_fn():
      x = constant_op.constant(1.0)
      with lsgt.LossScaleGradientTape(loss_scale) as g:
        g.watch(x)
        y = x * (3.0 * (10**37))  # grad will be inf after scaling
      return g.gradient(y, x)

    dy_dx_list = self._run_with_strategy(run_fn, strategy_fn(), use_tf_function)
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
    def run_fn():
      x = constant_op.constant(3.0 * (10**37))
      with lsgt.LossScaleGradientTape(loss_scale) as g:
        g.watch(x)
        y = x * 3.0  # target will be inf after scaling
      return g.gradient(y, x)

    dy_dx_list = self._run_with_strategy(run_fn, strategy_fn(), use_tf_function)
    self.assertEqual(self.evaluate(loss_scale()), 32.0)
    for dy_dx in dy_dx_list:
      self.assertAllClose(self.evaluate(dy_dx), 3.0)


if __name__ == '__main__':
  v2_compat.enable_v2_behavior()
  test.main()
