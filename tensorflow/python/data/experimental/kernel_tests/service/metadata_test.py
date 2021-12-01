# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tf.data service metadata."""

import functools

from absl.testing import parameterized

from tensorflow.python.data.experimental.kernel_tests.service import test_base as data_service_test_base
from tensorflow.python.data.experimental.ops import data_service_ops
from tensorflow.python.data.experimental.ops import distribute
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import combinations
from tensorflow.python.platform import test


def _cardinality_test_combinations():
  """Generate test combinations for data service cardinality tests.

  We test only V2 combinations for the infinite and 0 cases because the `map`
  transformation for compression makes the cardinality unknown in TF1.

  Returns:
    test combinations.
  """

  def _reduce_cases_to_combinations(result, case):
    name, dataset_fn, sharding_policy, expected_result = case
    return result + combinations.combine(
        dataset_fn=combinations.NamedObject(name, dataset_fn),
        sharding_policy=sharding_policy,
        expected_result=expected_result)

  def _cases_to_combinations(cases):
    return functools.reduce(_reduce_cases_to_combinations, cases, [])

  def _infinite_dataset_with_hint_shard():
    return (dataset_ops.Dataset.range(10).shard(distribute.SHARD_HINT,
                                                distribute.SHARD_HINT).repeat())

  def _empty_dataset_with_hint_shard():
    return (dataset_ops.Dataset.range(0).shard(distribute.SHARD_HINT,
                                               distribute.SHARD_HINT))

  v2_only_cases = [
      ("NoShardingInfinite", lambda: dataset_ops.Dataset.range(10).repeat(),
       data_service_ops.ShardingPolicy.OFF, dataset_ops.INFINITE),
      ("DataShardingInfinite", lambda: dataset_ops.Dataset.range(10).repeat(),
       data_service_ops.ShardingPolicy.DATA, dataset_ops.INFINITE),
      ("NoShardingZero", lambda: dataset_ops.Dataset.range(0),
       data_service_ops.ShardingPolicy.OFF, 0),
      ("DynamicShardingZero", lambda: dataset_ops.Dataset.range(0),
       data_service_ops.ShardingPolicy.DYNAMIC, 0),
      ("DataShardingZero", lambda: dataset_ops.Dataset.range(0),
       data_service_ops.ShardingPolicy.DATA, 0),
      ("FileOrDataShardingZero", lambda: dataset_ops.Dataset.range(0),
       data_service_ops.ShardingPolicy.FILE_OR_DATA, 0),
      ("HintShardingZero", _empty_dataset_with_hint_shard,
       data_service_ops.ShardingPolicy.HINT, dataset_ops.UNKNOWN),
  ]
  v1_and_v2_cases = [
      ("Finite", lambda: dataset_ops.Dataset.range(10),
       data_service_ops.ShardingPolicy.OFF, dataset_ops.UNKNOWN),
      ("DynamicShardingUnknown", lambda: dataset_ops.Dataset.range(10).repeat(),
       data_service_ops.ShardingPolicy.DYNAMIC, dataset_ops.UNKNOWN),
      ("FileOrDataShardingUnknown",
       lambda: dataset_ops.Dataset.range(10).repeat(),
       data_service_ops.ShardingPolicy.FILE_OR_DATA, dataset_ops.UNKNOWN),
      ("HintShardingUnknown", _infinite_dataset_with_hint_shard,
       data_service_ops.ShardingPolicy.HINT, dataset_ops.UNKNOWN),
  ]

  v2_only_combinations = combinations.times(
      combinations.combine(tf_api_version=2, mode=["eager", "graph"]),
      _cases_to_combinations(v2_only_cases))
  v1_and_v2_combinations = combinations.times(
      combinations.combine(tf_api_version=[1, 2], mode=["eager", "graph"]),
      _cases_to_combinations(v1_and_v2_cases))
  return v2_only_combinations + v1_and_v2_combinations


class DataServiceCardinalityTest(data_service_test_base.TestBase,
                                 parameterized.TestCase):
  """Tests propagating cardinality through tf.data service."""

  @combinations.generate(_cardinality_test_combinations())
  def testCardinality(self, dataset_fn, sharding_policy, expected_result):
    cluster = data_service_test_base.TestCluster(num_workers=2)
    dataset = dataset_fn()
    dataset = self.make_distributed_dataset(
        dataset, cluster=cluster, processing_mode=sharding_policy)
    self.assertEqual(self.evaluate(dataset.cardinality()), expected_result)

  @combinations.generate(_cardinality_test_combinations())
  def testFromDatasetIdCardinality(self, dataset_fn, sharding_policy,
                                   expected_result):
    cluster = data_service_test_base.TestCluster(num_workers=2)
    dataset = dataset_fn()
    dataset_id = data_service_ops.register_dataset(
        cluster.dispatcher.target, dataset=dataset)
    dataset = data_service_ops.from_dataset_id(
        processing_mode=sharding_policy,
        service=cluster.dispatcher.target,
        dataset_id=dataset_id,
        element_spec=dataset.element_spec)
    self.assertEqual(self.evaluate(dataset.cardinality()), expected_result)


if __name__ == "__main__":
  test.main()
