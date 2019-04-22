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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from absl.testing import parameterized

from tensorflow.python import tf2
from tensorflow.python.client import session
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import config
from tensorflow.python.framework import test_util
from tensorflow.python.keras.mixed_precision.experimental import loss_scale_optimizer as loss_scale_optimizer_v2
from tensorflow.python.keras.optimizer_v2 import gradient_descent as gradient_descent_v2
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import gradient_descent as gradient_descent_v1
from tensorflow.python.training.experimental import loss_scale_optimizer as loss_scale_optimizer_v1
from tensorflow.python.training.experimental import mixed_precision


if tf2.enabled():
  enable_mixed_precision_graph_rewrite = (
      mixed_precision.enable_mixed_precision_graph_rewrite)
else:
  enable_mixed_precision_graph_rewrite = (
      mixed_precision.enable_mixed_precision_graph_rewrite_v1)


class MixedPrecisionTest(test.TestCase, parameterized.TestCase):

  IGNORE_PERF_VAR = 'TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_IGNORE_PERFORMANCE'

  def setUp(self):
    super(MixedPrecisionTest, self).setUp()
    # Enable the tests to be run on pre-Volta GPUs by telling the grappler pass
    # to ignore performance and always transform the graph.
    self._original_ignore_perf_value = os.getenv(self.IGNORE_PERF_VAR)
    os.environ[self.IGNORE_PERF_VAR] = '1'

  def tearDown(self):
    # Set auto_mixed_precision back to it's default value.
    config.set_optimizer_experimental_options({'auto_mixed_precision': False})
    # Set the IGNORE_PERF_VAR variable back to it's original value.
    if self._original_ignore_perf_value is not None:
      os.environ[self.IGNORE_PERF_VAR] = self._original_ignore_perf_value
    else:
      del os.environ[self.IGNORE_PERF_VAR]
    super(MixedPrecisionTest, self).tearDown()

  @test_util.run_in_graph_and_eager_modes
  def test_wrap_optimizer(self):
    opt = gradient_descent_v1.GradientDescentOptimizer(1.0)
    opt = enable_mixed_precision_graph_rewrite(opt, 123.)
    self.assertIsInstance(
        opt, loss_scale_optimizer_v1.MixedPrecisionLossScaleOptimizer)
    self.assertEqual(self.evaluate(opt._loss_scale()), 123.)

    opt = gradient_descent_v2.SGD(1.0)
    opt = enable_mixed_precision_graph_rewrite(opt, 123.)
    self.assertIsInstance(
        opt, loss_scale_optimizer_v2.LossScaleOptimizer)
    self.assertEqual(self.evaluate(opt._loss_scale()), 123.)

  @test_util.run_in_graph_and_eager_modes
  def test_optimizer_errors(self):
    opt = 1
    if tf2.enabled():
      expected_regex = ('"opt" must be an instance of a '
                        'tf.keras.optimizers.Optimizer, but got')
    else:
      expected_regex = ('"opt" must be an instance of a tf.train.Optimizer or '
                        'a tf.keras.optimizers.Optimizer, but got')
    with self.assertRaisesRegexp(ValueError, expected_regex):
      enable_mixed_precision_graph_rewrite(opt)
    self.assertFalse(config.get_optimizer_experimental_options()
                     .get('auto_mixed_precision', False))

    opt = gradient_descent_v1.GradientDescentOptimizer(1.0)
    opt = loss_scale_optimizer_v1.MixedPrecisionLossScaleOptimizer(opt,
                                                                   'dynamic')
    with self.assertRaisesRegexp(ValueError,
                                 '"opt" must not already be an instance of a '
                                 'MixedPrecisionLossScaleOptimizer.'):
      enable_mixed_precision_graph_rewrite(opt)
    self.assertFalse(config.get_optimizer_experimental_options()
                     .get('auto_mixed_precision', False))

    opt = gradient_descent_v2.SGD(1.0)
    opt = loss_scale_optimizer_v2.LossScaleOptimizer(opt, 'dynamic')
    with self.assertRaisesRegexp(ValueError,
                                 '"opt" must not already be an instance of a '
                                 'LossScaleOptimizer.'):
      enable_mixed_precision_graph_rewrite(opt)
    self.assertFalse(config.get_optimizer_experimental_options()
                     .get('auto_mixed_precision', False))

  @test_util.run_gpu_only
  @test_util.run_in_graph_and_eager_modes
  def test_grappler_pass_enabled(self):
    opt = gradient_descent_v2.SGD(1.0)
    enable_mixed_precision_graph_rewrite(opt, 123.)

    var = variables.Variable([[1.0]])

    def overflow_in_float16():
      out = var * 2 ** 10
      out = math_ops.matmul(out, out)
      return array_ops.reshape(out, ())

    if context.executing_eagerly():
      f = def_function.function(overflow_in_float16)
      self.assertEqual(f().numpy(), float('Inf'))
      # Outside a def_function.function, the grappler pass will not be applied.
      self.assertAlmostEqual(overflow_in_float16().numpy(), 2 ** 20)
    else:
      with session.Session() as sess:
        out = overflow_in_float16()
        sess.run(var.initializer)
        self.assertEqual(sess.run(out), float('Inf'))


if __name__ == '__main__':
  test.main()
