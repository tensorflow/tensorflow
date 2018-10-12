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
"""Tests for class Monitor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy

from tensorflow.contrib.distribute.python import combinations
from tensorflow.contrib.distribute.python import monitor as monitor_lib
from tensorflow.contrib.distribute.python import one_device_strategy
from tensorflow.contrib.distribute.python.single_loss_example import single_loss_example
from tensorflow.python.client import session
from tensorflow.python.eager import context
from tensorflow.python.eager import test
from tensorflow.python.framework import ops
from tensorflow.python.training import gradient_descent


class MonitorTest(test.TestCase, parameterized.TestCase):

  @combinations.generate(
      combinations.times(
          combinations.distributions_and_v1_optimizers(),
          combinations.combine(mode=combinations.graph_and_eager_modes)))
  def testTrainNetwork(self, distribution, optimizer_fn):
    with distribution.scope():
      single_loss_step, layer = single_loss_example(optimizer_fn, distribution)

      if context.executing_eagerly():
        monitor = monitor_lib.Monitor(single_loss_step, None)
      else:
        with self.cached_session() as sess:
          monitor = monitor_lib.Monitor(single_loss_step, sess)

      monitor.run_steps(1)

      self.assertEqual(1, len(layer.trainable_variables))
      mirrored_weight_variable = layer.trainable_variables[0]
      start_error = self.evaluate(mirrored_weight_variable)
      start_error = abs(numpy.array(start_error) - 1)

      monitor.run_steps(9)
      end_error = self.evaluate(mirrored_weight_variable)
      end_error = abs(numpy.array(end_error) - 1)
      self.assertGreaterEqual(start_error, end_error)

  def testPassingASessionInEager(self):
    distribution = one_device_strategy.OneDeviceStrategy(
        "/device:CPU:0")
    step_function, _ = single_loss_example(
        lambda: gradient_descent.GradientDescentOptimizer(0.2), distribution)

    with session.Session() as sess, context.eager_mode():
      with self.assertRaisesRegexp(ValueError, "Should not provide"):
        _ = monitor_lib.Monitor(step_function, sess)

  def testNotPassingASessionInGraph(self):
    distribution = one_device_strategy.OneDeviceStrategy(
        "/device:CPU:0")
    step_function, _ = single_loss_example(
        lambda: gradient_descent.GradientDescentOptimizer(0.2), distribution)

    with context.graph_mode(), ops.Graph().as_default():
      with self.assertRaisesRegexp(ValueError, "Should provide"):
        _ = monitor_lib.Monitor(step_function, session=None)


if __name__ == "__main__":
  test.main()
