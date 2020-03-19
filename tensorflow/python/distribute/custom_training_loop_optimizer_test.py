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
"""Tests for custom training loops that involves advanced optimizer usage."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

from tensorflow.python import keras
from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import strategy_combinations
from tensorflow.python.distribute import values
from tensorflow.python.eager import def_function
from tensorflow.python.eager import test
from tensorflow.python.framework import ops
from tensorflow.python.ops import variables


class OptimizerTest(test.TestCase, parameterized.TestCase):

  @combinations.generate(
      combinations.times(
          combinations.combine(
              distribution=strategy_combinations.multidevice_strategies,
              mode=["eager"],
          ),
          combinations.concat(
              combinations.combine(
                  all_reduce_sum_gradients=True,
                  expected=[[[-0.3, -0.3], [-0.3, -0.3]]]),
              combinations.combine(
                  all_reduce_sum_gradients=False,
                  expected=[[[-0.1, -0.1], [-0.2, -0.2]]]),
          )))
  def test_custom_aggregation(self, distribution, all_reduce_sum_gradients,
                              expected):

    with distribution.scope():
      v = variables.Variable([0., 0.])
      optimizer = keras.optimizer_v2.gradient_descent.SGD(0.1)

    @def_function.function
    def optimize():
      grads = values.PerReplica([
          ops.convert_to_tensor([1., 1.]),
          ops.convert_to_tensor([2., 2.]),
      ])

      def step_fn(grads):
        optimizer.apply_gradients(
            [(grads, v)], all_reduce_sum_gradients=all_reduce_sum_gradients)
        return v.read_value()

      return distribution.experimental_local_results(
          distribution.run(step_fn, args=(grads,)))

    self.assertAllClose(optimize(), expected)

  @combinations.generate(
      combinations.combine(
          distribution=strategy_combinations.one_device_strategy,
          mode=["eager"],
          all_reduce_sum_gradients=[True, False]))
  def test_custom_aggregation_one_device(self, distribution,
                                         all_reduce_sum_gradients):

    with distribution.scope():
      v = variables.Variable([0., 0.])
      optimizer = keras.optimizer_v2.gradient_descent.SGD(0.1)

    @def_function.function
    def optimize():
      grads = ops.convert_to_tensor([1., 1.])

      def step_fn(grads):
        optimizer.apply_gradients(
            [(grads, v)], all_reduce_sum_gradients=all_reduce_sum_gradients)
        return v.read_value()

      return distribution.experimental_local_results(
          distribution.run(step_fn, args=(grads,)))

    self.assertAllClose(optimize(), [[-0.1, -0.1]])


if __name__ == "__main__":
  test.main()
