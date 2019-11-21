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
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.platform import test
from tensorflow.python.training.experimental import loss_scale as loss_scale_module
from tensorflow.python.training.experimental import loss_scaling_gradient_tape as lsgt


class LossScaleGradientTapeTest(test.TestCase, parameterized.TestCase):

  @parameterized.parameters(loss_scale_module.FixedLossScale,
                            loss_scale_module.DynamicLossScale)
  def test_basic_tapes_eager_mode(self, loss_scale):
    x = constant_op.constant(3.0)
    with lsgt.LossScaleGradientTape(loss_scale(32)) as g:
      g.watch(x)
      y = x * x
    dy_dx = g.gradient(y, x)
    self.assertEqual(self.evaluate(dy_dx), 6.0)

  @parameterized.parameters(loss_scale_module.FixedLossScale,
                            loss_scale_module.DynamicLossScale)
  def test_basic_tapes_graph_mode(self, loss_scale):
    loss_scale = loss_scale(32)

    @def_function.function
    def _inner_test():
      x = constant_op.constant(3.0)
      with lsgt.LossScaleGradientTape(loss_scale) as g:
        g.watch(x)
        y = x * x
      return g.gradient(y, x)
    self.assertEqual(self.evaluate(_inner_test()), 6.0)

  @parameterized.parameters(loss_scale_module.FixedLossScale,
                            loss_scale_module.DynamicLossScale)
  def test_nested_tapes(self, loss_scale):
    x = constant_op.constant(3.0)
    with lsgt.LossScaleGradientTape(loss_scale(32)) as g:
      g.watch(x)
      with lsgt.LossScaleGradientTape(loss_scale(32)) as gg:
        gg.watch(x)
        y = x * x
      dy_dx = gg.gradient(y, x)
      self.assertEqual(self.evaluate(dy_dx), 6.0)
    d2y_dx2 = g.gradient(dy_dx, x)
    self.assertEqual(self.evaluate(d2y_dx2), 2.0)

  @parameterized.parameters(loss_scale_module.FixedLossScale,
                            loss_scale_module.DynamicLossScale)
  def test_non_persistent_tapes_error(self, loss_scale):
    x = constant_op.constant(3.0)
    with lsgt.LossScaleGradientTape(loss_scale(32), persistent=False) as g:
      g.watch(x)
      y = x * x
      z = y * y
    g.gradient(z, x)
    with self.assertRaisesRegexp(RuntimeError, 'persistent'):
      g.gradient(y, x)

  @parameterized.parameters(loss_scale_module.FixedLossScale,
                            loss_scale_module.DynamicLossScale)
  def test_persistent_tapes(self, loss_scale):
    x = constant_op.constant(3.0)
    with lsgt.LossScaleGradientTape(loss_scale(32), persistent=True) as g:
      g.watch(x)
      y = x * x
      z = y * y
    dz_dx = g.gradient(z, x)
    self.assertEqual(self.evaluate(dz_dx), 108.0)
    dy_dx = g.gradient(y, x)
    self.assertEqual(self.evaluate(dy_dx), 6.0)

  @parameterized.parameters(loss_scale_module.FixedLossScale,
                            loss_scale_module.DynamicLossScale)
  def test_nested_sources(self, loss_scale):
    x = (constant_op.constant(19.0), (constant_op.constant(8.),
                                      constant_op.constant(9.)))
    with lsgt.LossScaleGradientTape(loss_scale(32)) as g:
      g.watch(x)
      y = x * 13
    dy_dx = g.gradient(y, x)
    self.assertEqual(self.evaluate(dy_dx), (13., (13., 13.)))

  @parameterized.parameters(loss_scale_module.FixedLossScale,
                            loss_scale_module.DynamicLossScale)
  def test_nested_targets(self, loss_scale):
    w = constant_op.constant(3.0)
    with lsgt.LossScaleGradientTape(loss_scale(32)) as g:
      g.watch(w)
      x = w * 5
      y = w * 7
      z = w * 11
    grad = g.gradient([x, (y, z)], w)
    self.assertEqual(self.evaluate(grad), 23)

  @parameterized.parameters(loss_scale_module.FixedLossScale,
                            loss_scale_module.DynamicLossScale)
  def test_scaling_inf_gradient(self, loss_scale):
    x = constant_op.constant(1.0)
    with lsgt.LossScaleGradientTape(loss_scale(32)) as g:
      g.watch(x)
      y = x * np.inf
    dy_dx = g.gradient(y, x)
    self.assertEqual(self.evaluate(dy_dx), np.inf)

  @parameterized.parameters(loss_scale_module.FixedLossScale,
                            loss_scale_module.DynamicLossScale)
  def test_scaling_nan_gradient(self, loss_scale):
    x = constant_op.constant(1.0)
    with lsgt.LossScaleGradientTape(loss_scale(32)) as g:
      g.watch(x)
      y = x * np.nan
    dy_dx = g.gradient(y, x)
    self.assertTrue(np.isnan(self.evaluate(dy_dx)))

  @parameterized.parameters(np.inf, np.nan)
  def test_dynamic_scale_to_one_on_non_finite_gradient(self, non_finite_term):
    loss_scale = loss_scale_module.DynamicLossScale(initial_loss_scale=32)
    x = constant_op.constant(1.0)
    with lsgt.LossScaleGradientTape(loss_scale) as g:
      g.watch(x)
      y = x * non_finite_term
    g.gradient(y, x)
    self.assertEqual(self.evaluate(loss_scale()), 1.0)

  @parameterized.parameters([np.inf, np.isposinf], [np.nan, np.isnan])
  def test_fixed_scaling_no_change_non_finite_gradient(self, non_finite_term,
                                                       is_non_finite):
    loss_scale = loss_scale_module.FixedLossScale(32)
    x = constant_op.constant(1.0)
    with lsgt.LossScaleGradientTape(loss_scale) as g:
      g.watch(x)
      y = x * non_finite_term
    dy_dx = g.gradient(y, x)
    self.assertTrue(is_non_finite(self.evaluate(dy_dx)))
    self.assertEqual(self.evaluate(loss_scale()), 32.0)

  def test_dynamic_loss_scaling_down_loop(self):
    loss_scale = loss_scale_module.DynamicLossScale(initial_loss_scale=32)
    x = constant_op.constant(1.0)
    with lsgt.LossScaleGradientTape(loss_scale) as g:
      g.watch(x)
      y = x * (3.0 * (10**37))  # grad will be inf after scaling
    dy_dx = g.gradient(y, x)
    self.assertEqual(self.evaluate(loss_scale()), 8.0)
    self.assertAllClose(self.evaluate(dy_dx), (3.0 * (10**37)), atol=1e-06)

  def test_dynamic_loss_scaling_inf_target_post_scale(self):
    loss_scale = loss_scale_module.DynamicLossScale(initial_loss_scale=32.0)
    x = constant_op.constant(3.0 * (10**37))
    with lsgt.LossScaleGradientTape(loss_scale) as g:
      g.watch(x)
      y = x * 3.0  # target will be inf after scaling
    dy_dx = g.gradient(y, x)
    self.assertAllClose(self.evaluate(dy_dx), 3.0)
    self.assertEqual(self.evaluate(loss_scale()), 32.0)


if __name__ == '__main__':
  v2_compat.enable_v2_behavior()
  test.main()
