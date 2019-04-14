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
"""Tests for custom training loops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
from tensorflow.python import tf2
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import strategy_combinations
from tensorflow.python.eager import def_function
from tensorflow.python.eager import test


class InputIterationTest(test.TestCase, parameterized.TestCase):

  @combinations.generate(
      combinations.combine(
          distribution=strategy_combinations.strategies_minus_tpu,
          mode=["eager"]
      ))
  def testFullEager(self, distribution):
    dataset = self._get_dataset()

    def train_step(data):
      return data

    dist_dataset = distribution.experimental_distribute_dataset(dataset)
    results = []
    for x in dist_dataset:
      output = distribution.experimental_local_results(
          distribution.experimental_run_v2(train_step, args=(x,)))
      results.append(output)
    self._validate_outputs(results)

  @combinations.generate(
      combinations.combine(
          distribution=strategy_combinations.strategies_minus_tpu,
          mode=["eager"]
      ))
  def testStepInFunction(self, distribution):
    dataset = self._get_dataset()

    @def_function.function
    def train_step(data):
      return data

    dist_dataset = distribution.experimental_distribute_dataset(dataset)
    results = []
    for x in dist_dataset:
      output = distribution.experimental_local_results(
          distribution.experimental_run_v2(train_step, args=(x,)))
      results.append(output)
    self._validate_outputs(results)

  @combinations.generate(
      combinations.combine(
          distribution=strategy_combinations.strategies_minus_tpu +
          [strategy_combinations.tpu_strategy_one_step],
          mode=["eager"]
      ))
  def testRunInFunction(self, distribution):
    dataset = self._get_dataset()

    def train_step(data):
      return data

    @def_function.function
    def f_train_step(input_data):
      return distribution.experimental_local_results(
          distribution.experimental_run_v2(train_step, args=(input_data,)))

    dist_dataset = distribution.experimental_distribute_dataset(dataset)
    results = []
    for x in dist_dataset:
      output = f_train_step(x)
      results.append(output)
    self._validate_outputs(results)

  @combinations.generate(
      combinations.combine(
          distribution=strategy_combinations.strategies_minus_tpu +
          [strategy_combinations.tpu_strategy_one_step],
          mode=["eager"]
      ))
  def testIterationInsideFunction(self, distribution):

    def step_fn(data):
      return data

    @def_function.function
    def train(dataset):
      results = []
      iterator = iter(dataset)
      # we iterate through the loop 5 times since we have 10 elements and a
      # global batch of 2.
      for _ in range(5):
        elem = next(iterator)
        output = distribution.experimental_local_results(
            distribution.experimental_run_v2(step_fn, args=(elem,)))
        results.append(output)
      return results

    dataset = self._get_dataset()
    dist_dataset = distribution.experimental_distribute_dataset(dataset)
    results = train(dist_dataset)
    self._validate_outputs(results)

  @combinations.generate(
      combinations.combine(
          distribution=strategy_combinations.strategies_minus_tpu +
          [strategy_combinations.tpu_strategy_one_step],
          mode=["eager"]
      ))
  def testIterationOutsideFunction(self, distribution):

    def train_step(data):
      return data

    @def_function.function
    def f_train_step(input_data):
      return distribution.experimental_local_results(
          distribution.experimental_run_v2(train_step, args=(input_data,)))

    dataset = self._get_dataset()
    dist_dataset = distribution.experimental_distribute_dataset(dataset)
    iterator = iter(dist_dataset)
    results = []
    # we iterate through the loop 5 times since we have 10 elements and a
    # global batch of 2.
    for _ in range(5):
      output = f_train_step(next(iterator))
      results.append(output)
    self._validate_outputs(results)

  def _get_dataset(self):
    if tf2.enabled():
      return dataset_ops.DatasetV2.range(10).batch(2)
    else:
      return dataset_ops.Dataset.range(10).batch(2)

  def _validate_outputs(self, actual_results):
    expected_results = [[i, i+1] for i in range(0, 10, 2)]
    self.assertEqual(len(expected_results), len(actual_results))

    for i, expected_result in enumerate(expected_results):
      final_result = []
      actual_result = actual_results[i]
      for val in actual_result:
        final_result.extend(val.numpy())
      self.assertAllEqual(expected_result, final_result)

if __name__ == "__main__":
  test.main()

