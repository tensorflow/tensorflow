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
"""Tests for running legacy optimizer code with DistributionStrategy."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy
from tensorflow.contrib.distribute.python import mirrored_strategy as mirrored_lib
from tensorflow.contrib.optimizer_v2 import adagrad as adagrad_v2
from tensorflow.contrib.optimizer_v2 import gradient_descent as gradient_descent_v2
from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import strategy_combinations
from tensorflow.python.distribute.single_loss_example import minimize_loss_example
from tensorflow.python.eager import context
from tensorflow.python.eager import test
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import variables


mirrored_strategy_with_gpu_and_cpu = combinations.NamedDistribution(
    "MirroredCPUAndGPU",
    lambda: mirrored_lib.MirroredStrategy(["/gpu:0", "/cpu:0"]),
    required_gpus=1)
mirrored_strategy_with_two_gpus = combinations.NamedDistribution(
    "Mirrored2GPUs",
    lambda: mirrored_lib.MirroredStrategy(["/gpu:0", "/gpu:1"]),
    required_gpus=2)

# pylint: disable=g-long-lambda
gradient_descent_optimizer_v2_fn = combinations.NamedObject(
    "GradientDescentV2", lambda: gradient_descent_v2.GradientDescentOptimizer(
        0.2))
adagrad_optimizer_v2_fn = combinations.NamedObject(
    "AdagradV2", lambda: adagrad_v2.AdagradOptimizer(0.001))

optimizers_v2 = [gradient_descent_optimizer_v2_fn, adagrad_optimizer_v2_fn]


def distributions_and_v2_optimizers():
  """DistributionStrategies and V2 Optimizers."""
  return combinations.combine(
      distribution=[
          strategy_combinations.one_device_strategy,
          strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
          strategy_combinations.mirrored_strategy_with_two_gpus,
      ],
      optimizer_fn=optimizers_v2)


class MinimizeLossOptimizerV2Test(test.TestCase, parameterized.TestCase):

  @combinations.generate(
      combinations.times(
          distributions_and_v2_optimizers(),
          combinations.combine(mode=["graph"], use_callable_loss=[True, False])
          + combinations.combine(mode=["eager"], use_callable_loss=[True])))
  def testTrainNetwork(self, distribution, optimizer_fn,
                       use_callable_loss=True):
    with distribution.scope():
      model_fn, dataset_fn, layer = minimize_loss_example(
          optimizer_fn, use_bias=True, use_callable_loss=use_callable_loss)
      iterator = distribution.make_input_fn_iterator(lambda _: dataset_fn())

      def run_step():
        return control_flow_ops.group(
            distribution.experimental_local_results(
                distribution.extended.call_for_each_replica(
                    model_fn, args=(iterator.get_next(),))))

      if not context.executing_eagerly():
        with self.cached_session() as sess:
          sess.run(iterator.initialize())
          run_step = sess.make_callable(run_step())
        self.evaluate(variables.global_variables_initializer())

      weights, biases = [], []
      for _ in range(10):
        run_step()

        weights.append(self.evaluate(layer.kernel))
        biases.append(self.evaluate(layer.bias))

      error = abs(numpy.add(numpy.squeeze(weights), numpy.squeeze(biases)) - 1)
      is_not_increasing = all(y <= x for x, y in zip(error, error[1:]))
      self.assertTrue(is_not_increasing)


if __name__ == "__main__":
  test.main()
