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
from tensorflow.python.eager import backprop
from tensorflow.python.eager import def_function
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.util import nest


class InputIterationTest(test.TestCase, parameterized.TestCase):

  @combinations.generate(
      combinations.combine(
          distribution=strategy_combinations.strategies_minus_tpu,
          mode=["eager"]))
  def testFullEager(self, distribution):
    dataset = self._get_dataset()

    def train_step(data):
      return math_ops.square(data)

    dist_dataset = distribution.experimental_distribute_dataset(dataset)
    results = []
    for x in dist_dataset:
      output = distribution.experimental_local_results(
          distribution.experimental_run_v2(train_step, args=(x,)))
      results.append(output)
    self._validate_outputs(results)

  @combinations.generate(
      combinations.combine(
          distribution=strategy_combinations.all_strategies,
          mode=["eager"]
      ))
  def testStepInFunction(self, distribution):
    dataset = self._get_dataset()

    @def_function.function
    def train_step(data):
      return math_ops.square(data)

    dist_dataset = distribution.experimental_distribute_dataset(dataset)
    results = []
    for x in dist_dataset:
      output = distribution.experimental_local_results(
          distribution.experimental_run_v2(train_step, args=(x,)))
      results.append(output)
    self._validate_outputs(results)

  @combinations.generate(
      combinations.combine(
          distribution=strategy_combinations.all_strategies,
          mode=["eager"]
      ))
  def testRunInFunction(self, distribution):
    dataset = self._get_dataset()

    def train_step(data):
      return math_ops.square(data)

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
          distribution=[
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
              strategy_combinations.tpu_strategy
          ],
          mode=["eager"]))
  def testNestedOutput(self, distribution):
    dataset = self._get_dataset()
    input_iterator = iter(distribution.experimental_distribute_dataset(dataset))

    @def_function.function
    def run(iterator):

      def computation(x):
        return [{
            "a": x - 1,
            "b": x + 1
        }]

      inputs = next(iterator)
      outputs = distribution.experimental_run_v2(computation, args=(inputs,))
      return nest.map_structure(distribution.experimental_local_results,
                                outputs)

    results = run(input_iterator)
    for replica in range(distribution.num_replicas_in_sync):
      # The input dataset is range(10), so the replica id is same as input.
      self.assertAllEqual(results[0]["a"][replica], [replica - 1])
      self.assertAllEqual(results[0]["b"][replica], [replica + 1])

  @combinations.generate(
      combinations.combine(
          distribution=strategy_combinations.all_strategies,
          mode=["eager"]
      ))
  def testRunInFunctionAutoGraphApplication(self, distribution):
    dataset = self._get_dataset()

    def train_step(data):
      return math_ops.square(data)

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
          distribution=strategy_combinations.all_strategies,
          mode=["eager"]
      ))
  def testDatasetIterationInFunction(self, distribution):
    with distribution.scope():
      a = variables.Variable(
          1.0, aggregation=variables.VariableAggregation.ONLY_FIRST_REPLICA)

    def train_step(_):
      a.assign_add(1.0)

    @def_function.function
    def f_train_step(dist_dataset):
      number_of_steps = constant_op.constant(0.0)
      product_of_means = constant_op.constant(2.0)
      for x in dist_dataset:  # loop with values modified each iteration
        number_of_steps += 1
        product_of_means *= math_ops.cast(
            distribution.reduce("MEAN", x, axis=0), product_of_means.dtype)

      for y in dist_dataset:  # loop with no intermediate state
        distribution.experimental_run_v2(train_step, args=(y,))

      return number_of_steps, product_of_means

    dataset = self._get_dataset()
    dist_dataset = distribution.experimental_distribute_dataset(dataset)

    number_of_steps, product_of_means = f_train_step(dist_dataset)
    self.assertEqual(5, number_of_steps.numpy())

    # 2.0 * (0+1)/2 * (2+3)/2 * (4+5)/2 * (6+7)/2 * (8+9)/2
    #  = (5 * 9 * 13 * 17) / 16
    self.assertNear((5 * 9 * 13 * 17) / 16, product_of_means.numpy(), 1e-3)

    # We set the initial value of `a` to 1 and iterate through the dataset 5
    # times(10/2 where 10 is the number of dataset elements and 2 is the batch
    # size). Hence the final result is 6.
    self.assertEqual(6.0, (a.numpy()))

  @combinations.generate(
      combinations.combine(
          distribution=strategy_combinations.all_strategies,
          mode=["eager"]
      ))
  def testDatasetAssertWithDynamicBatch(self, distribution):
    # Regression test for github issue 33517.
    def step_fn(data):
      assert_op = control_flow_ops.Assert(math_ops.less_equal(
          math_ops.reduce_max(data), 100.), [data])
      with ops.control_dependencies([assert_op]):
        return math_ops.square(data)

    @def_function.function
    def train(dataset):
      results = []
      iterator = iter(dataset)
      # we iterate through the loop 5 times since we have 3 elements and a
      # global batch of 2.
      for _ in range(2):
        elem = next(iterator)
        output = distribution.experimental_local_results(
            distribution.experimental_run_v2(step_fn, args=(elem,)))
        results.append(output)
      return results

    dataset = dataset_ops.DatasetV2.from_tensor_slices([5., 6., 7.,]).batch(2)
    # TODO(b/138326910): Remove Dataset V1 version once bug resolved.
    if not tf2.enabled():
      return dataset_ops.Dataset.from_tensor_slices([5., 6., 7.,]).batch(2)
    dist_dataset = distribution.experimental_distribute_dataset(dataset)
    results = train(dist_dataset)

    expected_results = [[25., 36.], [49.]]
    self.assertEqual(len(expected_results), len(results))

    # Need to expand results since output will be grouped differently depending
    # on the number of replicas.
    for i, expected_result in enumerate(expected_results):
      final_result = []
      actual_result = results[i]
      for val in actual_result:
        final_result.extend(val.numpy())
      self.assertAllEqual(expected_result, final_result)

  @combinations.generate(
      combinations.combine(
          distribution=strategy_combinations.all_strategies,
          mode=["eager"]
      ))
  def testIterationInsideFunction(self, distribution):

    def step_fn(data):
      return math_ops.square(data)

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
          distribution=strategy_combinations.all_strategies,
          mode=["eager"]
      ))
  def testIterationOutsideFunction(self, distribution):

    def train_step(data):
      return math_ops.square(data)

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
      return dataset_ops.DatasetV2.range(10).\
        map(lambda x: math_ops.cast(x, dtypes.int32)).batch(2)
    else:
      return dataset_ops.Dataset.range(10).\
        map(lambda x: math_ops.cast(x, dtypes.int32)).batch(2)

  def _validate_outputs(self, actual_results):
    expected_results = [[i**2, (i+1)**2] for i in range(0, 10, 2)]
    self.assertEqual(len(expected_results), len(actual_results))

    for i, expected_result in enumerate(expected_results):
      final_result = []
      actual_result = actual_results[i]
      for val in actual_result:
        final_result.extend(val.numpy())
      self.assertAllEqual(expected_result, final_result)


class GradientTapeTest(test.TestCase, parameterized.TestCase):

  @combinations.generate(
      combinations.combine(
          distribution=strategy_combinations.all_strategies,
          mode=["eager"],
          model_in_tf_function=[True, False]
      ))
  def testNestedFunction(self, distribution, model_in_tf_function):
    def model(x):
      return x * x

    if model_in_tf_function:
      model = def_function.function(model)

    with distribution.scope():
      x = variables.Variable(1.0)

      @def_function.function
      def train_step():
        def replica_step():
          with backprop.GradientTape() as tape:
            y = model(x)
          return tape.gradient(y, x)
        return distribution.experimental_run_v2(replica_step)

      grads = distribution.experimental_local_results(train_step())
      self.assertLen(grads, distribution.num_replicas_in_sync)
      self.assertTrue(all(g is not None for g in grads))


if __name__ == "__main__":
  test.main()
