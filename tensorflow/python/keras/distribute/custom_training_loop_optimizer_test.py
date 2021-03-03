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
from tensorflow.python.distribute import combinations as ds_combinations
from tensorflow.python.distribute import strategy_combinations
from tensorflow.python.distribute import values
from tensorflow.python.eager import def_function
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_combinations as combinations
from tensorflow.python.keras.distribute import strategy_combinations as keras_strategy_combinations
from tensorflow.python.keras.optimizer_v2 import gradient_descent
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


class OptimizerTest(test.TestCase, parameterized.TestCase):

  @ds_combinations.generate(
      combinations.times(
          combinations.combine(
              distribution=keras_strategy_combinations.multidevice_strategies,
              mode=["eager"],
          ),
          combinations.combine(
              experimental_aggregate_gradients=True,
              expected=[[[-0.3, -0.3], [-0.3, -0.3]]]) +
          combinations.combine(
              experimental_aggregate_gradients=False,
              expected=[[[-0.1, -0.1], [-0.2, -0.2]]])
      ))
  def test_custom_aggregation(self, distribution,
                              experimental_aggregate_gradients, expected):

    with distribution.scope():
      v = variables.Variable([0., 0.])
      optimizer = gradient_descent.SGD(0.1)

    @def_function.function
    def optimize():
      grads = values.PerReplica([
          ops.convert_to_tensor_v2_with_dispatch([1., 1.]),
          ops.convert_to_tensor_v2_with_dispatch([2., 2.]),
      ])

      def step_fn(grads):
        optimizer.apply_gradients(
            [(grads, v)],
            experimental_aggregate_gradients=experimental_aggregate_gradients)
        return v.read_value()

      return distribution.experimental_local_results(
          distribution.run(step_fn, args=(grads,)))

    self.assertAllClose(optimize(), expected)

  @ds_combinations.generate(
      combinations.combine(
          distribution=strategy_combinations.one_device_strategy,
          mode=["eager"],
          experimental_aggregate_gradients=[True, False]))
  def test_custom_aggregation_one_device(self, distribution,
                                         experimental_aggregate_gradients):

    with distribution.scope():
      v = variables.Variable([0., 0.])
      optimizer = gradient_descent.SGD(0.1)

    @def_function.function
    def optimize():
      grads = ops.convert_to_tensor_v2_with_dispatch([1., 1.])

      def step_fn(grads):
        optimizer.apply_gradients(
            [(grads, v)],
            experimental_aggregate_gradients=experimental_aggregate_gradients)
        return v.read_value()

      return distribution.experimental_local_results(
          distribution.run(step_fn, args=(grads,)))

    self.assertAllClose(optimize(), [[-0.1, -0.1]])

  @ds_combinations.generate(
      combinations.combine(distribution=[
          strategy_combinations.central_storage_strategy_with_gpu_and_cpu
      ]))
  def test_custom_aggregation_central_storage(self, distribution):
    with distribution.scope():
      v = variables.Variable([0., 0.])
      optimizer = gradient_descent.SGD(0.1)

    grads = ops.convert_to_tensor_v2_with_dispatch([1., 1.])

    def step_fn(grads):
      with self.assertRaises(NotImplementedError):
        optimizer.apply_gradients([(grads, v)],
                                  experimental_aggregate_gradients=False)

    return distribution.run(step_fn, args=(grads,))


if __name__ == "__main__":
  test.main()
