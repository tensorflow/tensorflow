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

from tensorflow.contrib.distribute.python import combinations
from tensorflow.contrib.distribute.python.single_loss_example import minimize_loss_example
from tensorflow.python.eager import context
from tensorflow.python.eager import test
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import variables


class MinimizeLossOptimizerV2Test(test.TestCase, parameterized.TestCase):

  @combinations.generate(
      combinations.times(
          combinations.distributions_and_v2_optimizers(),
          combinations.combine(mode=["graph"], use_callable_loss=[True, False])
          + combinations.combine(mode=["eager"], use_callable_loss=[True])))
  def testTrainNetwork(self, distribution, optimizer_fn,
                       use_callable_loss=True):
    with distribution.scope():
      model_fn, dataset_fn, layer = minimize_loss_example(
          optimizer_fn, use_bias=True, use_callable_loss=use_callable_loss)

      ds = distribution.distribute_dataset(dataset_fn)
      if context.executing_eagerly():
        iterator = ds.make_one_shot_iterator()
      else:
        iterator = ds.make_initializable_iterator()

      def run_step():
        return control_flow_ops.group(
            distribution.unwrap(
                distribution.extended.call_for_each_replica(
                    model_fn, args=(iterator.get_next(),))))

      if not context.executing_eagerly():
        with self.cached_session() as sess:
          sess.run(iterator.initializer)
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
