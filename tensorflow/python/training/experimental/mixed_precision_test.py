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

from tensorflow.core.protobuf import config_pb2
from tensorflow.python import tf2
from tensorflow.python.client import session
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import config
from tensorflow.python.framework import test_util
from tensorflow.python.keras import testing_utils
from tensorflow.python.keras.mixed_precision.experimental import loss_scale_optimizer as loss_scale_optimizer_v2
from tensorflow.python.keras.mixed_precision.experimental import policy
from tensorflow.python.keras.optimizer_v2 import gradient_descent as gradient_descent_v2
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging
from tensorflow.python.training import gradient_descent as gradient_descent_v1
from tensorflow.python.training.experimental import loss_scale_optimizer as loss_scale_optimizer_v1
from tensorflow.python.training.experimental import mixed_precision
from tensorflow.python.training.experimental import mixed_precision_global_state


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
    # Set the IGNORE_PERF_VAR variable back to it's original value.
    if self._original_ignore_perf_value is not None:
      os.environ[self.IGNORE_PERF_VAR] = self._original_ignore_perf_value
    else:
      del os.environ[self.IGNORE_PERF_VAR]

    mixed_precision.disable_mixed_precision_graph_rewrite()
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

      # Test disabling mixed precision.
      mixed_precision.disable_mixed_precision_graph_rewrite()
      self.assertEqual(f().numpy(), 2 ** 20)
    else:
      with session.Session() as sess:
        out = overflow_in_float16()
        sess.run(var.initializer)
        self.assertEqual(sess.run(out), float('Inf'))

      # Test Session will enable the auto_mixed_precision grappler pass in a
      # ConfigProto passed by the user
      with session.Session(config=config_pb2.ConfigProto()) as sess:
        out = overflow_in_float16()
        sess.run(var.initializer)
        self.assertEqual(sess.run(out), float('Inf'))

      # Test disabling mixed precision.
      mixed_precision.disable_mixed_precision_graph_rewrite()
      with session.Session() as sess:
        out = overflow_in_float16()
        sess.run(var.initializer)
        self.assertAlmostEqual(sess.run(out), 2 ** 20)

  @test.mock.patch.object(tf_logging, 'warn')
  def test_warn_if_session_already_exists(self, mock_warn):
    # Set this to False, so Sessions created in previous tests do not trigger
    # the warning.
    mixed_precision_global_state.non_mixed_precision_session_created = False

    with session.Session():
      enable_mixed_precision_graph_rewrite(gradient_descent_v2.SGD(1.0))
      mock_warn.assert_any_call(
          'You already have existing Sessions that do not use mixed precision. '
          'enable_mixed_precision_graph_rewrite() will not affect these '
          'Sessions.')

  @test.mock.patch.object(tf_logging, 'warn')
  def test_do_not_warn_if_session_does_not_already_exist(self, mock_warn):
    # Set this to False, so Sessions created in previous tests do not trigger
    # the warning.
    mixed_precision_global_state.non_mixed_precision_session_created = False

    enable_mixed_precision_graph_rewrite(gradient_descent_v2.SGD(1.0))
    with session.Session():
      # Make sure the "You already have existing Sessions" warning was not
      # issued, since the Session was only created after
      # enable_mixed_precision_graph_rewrite.
      for call_arg in mock_warn.call_args_list:
        msg = call_arg[0][0]
        self.assertNotIn('You already have existing Sessions that do not use '
                         'mixed precision', msg)

  @testing_utils.enable_v2_dtype_behavior
  def test_error_if_policy_is_set(self):
    with policy.policy_scope('mixed_float16'):
      with self.assertRaisesRegexp(
          ValueError, 'a keras mixed precision Policy has been set'):
        enable_mixed_precision_graph_rewrite(gradient_descent_v2.SGD(1.0))
    # Test no error is thrown when the policy is current the default.
    enable_mixed_precision_graph_rewrite(gradient_descent_v2.SGD(1.0))


if __name__ == '__main__':
  test.main()
