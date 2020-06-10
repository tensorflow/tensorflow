# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for common methods in strategy classes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import multi_worker_test_base
from tensorflow.python.distribute import reduce_util
from tensorflow.python.distribute import strategy_combinations
from tensorflow.python.distribute import strategy_test_lib
from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class StrategyReduceTest(test.TestCase, parameterized.TestCase):

  @combinations.generate(
      combinations.combine(
          strategy=[strategy_combinations.multi_worker_mirrored_two_workers] +
          strategy_combinations.strategies_minus_tpu,
          mode=['eager']))
  def testSimpleReduce(self, strategy):

    def fn_eager():

      def replica_fn():
        return array_ops.ones((), dtypes.float32)

      per_replica_value = strategy.run(replica_fn)
      return strategy.reduce(
          reduce_util.ReduceOp.SUM, value=per_replica_value, axis=None)

    fn_graph = def_function.function(fn_eager)

    # Run reduce under the strategy scope to explicitly enter
    # strategy default_device scope.
    with strategy.scope():
      self.assertEqual(fn_eager().numpy(), 1.0 * strategy.num_replicas_in_sync)
      self.assertEqual(fn_graph().numpy(), 1.0 * strategy.num_replicas_in_sync)

    # Run reduce without a strategy scope to implicitly enter
    # strategy default_device scope.
    self.assertEqual(fn_eager().numpy(), 1.0 * strategy.num_replicas_in_sync)
    self.assertEqual(fn_graph().numpy(), 1.0 * strategy.num_replicas_in_sync)


class DistributedCollectiveAllReduceStrategyTest(
    strategy_test_lib.DistributionTestBase,
    parameterized.TestCase):

  @combinations.generate(
      combinations.combine(
          strategy=[strategy_combinations.multi_worker_mirrored_two_workers],
          mode=['eager']))
  def testDatasetFromFunction(self, strategy):
    def dataset_fn(input_context):
      global_batch_size = 10
      batch_size = input_context.get_per_replica_batch_size(global_batch_size)
      d = dataset_ops.DatasetV2.range(100).repeat().batch(batch_size)
      return d.shard(input_context.num_input_pipelines,
                     input_context.input_pipeline_id)

    expected_sum_on_workers = [10, 35]
    input_iterator = iter(
        strategy.experimental_distribute_datasets_from_function(dataset_fn))

    @def_function.function
    def run(iterator):
      return strategy.experimental_local_results(iterator.get_next())

    result = run(input_iterator)
    sum_value = math_ops.reduce_sum(result)
    self.assertEqual(
        sum_value.numpy(),
        expected_sum_on_workers[multi_worker_test_base.get_task_index()])


if __name__ == '__main__':
  combinations.main()
