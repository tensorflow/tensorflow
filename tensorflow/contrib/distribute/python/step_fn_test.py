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
"""Tests for class Step."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy

from tensorflow.contrib.distribute.python import combinations
from tensorflow.contrib.distribute.python.single_loss_example import single_loss_example
from tensorflow.python.eager import context
from tensorflow.python.eager import test
from tensorflow.python.ops import variables


class SingleLossStepTest(test.TestCase, parameterized.TestCase):

  @combinations.generate(
      combinations.times(
          combinations.distributions_and_v1_optimizers(),
          combinations.combine(mode=combinations.graph_and_eager_modes),
          combinations.combine(is_tpu=[False])) +
      combinations.combine(
          distribution=[combinations.tpu_strategy],
          optimizer_fn=combinations.optimizers_v1,
          mode=["graph"],
          is_tpu=[True]))
  def testTrainNetwork(self, distribution, optimizer_fn, is_tpu):
    with distribution.scope():
      single_loss_step, layer = single_loss_example(
          optimizer_fn, distribution, use_bias=True, iterations_per_step=2)

      if context.executing_eagerly():
        single_loss_step.initialize()
        run_step = single_loss_step
      else:
        with self.cached_session() as sess:
          sess.run(single_loss_step.initialize())
          run_step = sess.make_callable(single_loss_step())
      self.evaluate(variables.global_variables_initializer())

      weights, biases = [], []
      for _ in range(5):
        run_step()
        weights.append(self.evaluate(layer.kernel))
        biases.append(self.evaluate(layer.bias))

      error = abs(numpy.add(numpy.squeeze(weights), numpy.squeeze(biases)) - 1)
      is_not_increasing = all(y <= x for x, y in zip(error, error[1:]))
      self.assertTrue(is_not_increasing)


if __name__ == "__main__":
  test.main()
