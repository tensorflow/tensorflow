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
from unittest import mock

from absl.testing import parameterized

from tensorflow.python.compat import compat
from tensorflow.python.data.experimental.kernel_tests.service import test_base as data_service_test_base
from tensorflow.python.data.experimental.ops import data_service_ops
from tensorflow.python.data.experimental.ops import distribute
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import def_function
from tensorflow.python.framework import combinations
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import script_ops
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
      ("DynamicShardingInfinite", lambda: dataset_ops.Dataset.range(5).repeat(),
       data_service_ops.ShardingPolicy.DYNAMIC, dataset_ops.INFINITE),
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


class DataServiceMetadataTest(data_service_test_base.TestBase,
                              parameterized.TestCase):
  """Tests propagating data service metadata through tf.data service."""

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

  @combinations.generate(test_base.eager_only_combinations())
  def testFromDatasetIdDoesntRequireElementSpec(self):
    cluster = data_service_test_base.TestCluster(
        num_workers=1,
        work_dir=data_service_test_base.NO_WORK_DIR,
        fault_tolerant_mode=False,
        data_transfer_protocol="grpc")
    num_elements = 10
    dataset = dataset_ops.Dataset.range(num_elements)

    dataset_id = data_service_ops.register_dataset(cluster.dispatcher_address(),
                                                   dataset)
    dataset = data_service_ops.from_dataset_id(
        processing_mode=data_service_ops.ShardingPolicy.OFF,
        service=cluster.dispatcher_address(),
        dataset_id=dataset_id)
    self.assertDatasetProduces(dataset, list(range(num_elements)))

  @combinations.generate(test_base.graph_only_combinations())
  def testElementSpecGraphMode(self):
    cluster = data_service_test_base.TestCluster(
        num_workers=1,
        work_dir=data_service_test_base.NO_WORK_DIR,
        fault_tolerant_mode=False)
    num_elements = 10
    dataset = dataset_ops.Dataset.range(num_elements)
    dataset_id = data_service_ops.register_dataset(cluster.dispatcher_address(),
                                                   dataset)
    with self.assertRaisesRegex(
        ValueError, "In graph mode `element_spec` must be provided manually."):
      _ = data_service_ops.from_dataset_id(
          processing_mode=data_service_ops.ShardingPolicy.OFF,
          service=cluster.dispatcher_address(),
          dataset_id=dataset_id)

  @combinations.generate(test_base.eager_only_combinations())
  def testElementSpecMixedMode(self):
    cluster = data_service_test_base.TestCluster(
        num_workers=1,
        work_dir=data_service_test_base.NO_WORK_DIR,
        fault_tolerant_mode=False)
    num_elements = 10
    dataset = dataset_ops.Dataset.range(num_elements)

    @def_function.function
    def get_dataset_id():
      return data_service_ops.register_dataset(cluster.dispatcher_address(),
                                               dataset)

    dataset_id = get_dataset_id()
    dataset_id_val = tensor_util.constant_value(dataset_id)

    with self.assertRaisesRegex(
        ValueError,
        f"Failed to fetch element spec for dataset id {dataset_id_val} from "
        "tf.data service. If the dataset was registered in graph mode or "
        "inside a tf.function, the `element_spec` must be specified as an "
        "argument to `from_dataset_id`."):
      dataset = data_service_ops.from_dataset_id(
          processing_mode=data_service_ops.ShardingPolicy.OFF,
          service=cluster.dispatcher_address(),
          dataset_id=dataset_id)

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(compression=[None, "AUTO"])))
  def testFromDatasetIdOmitsCompression(self, compression):
    cluster = data_service_test_base.TestCluster(
        num_workers=1, data_transfer_protocol="grpc")
    dataset = dataset_ops.Dataset.from_tensor_slices(
        list("abcdefghijklmnopqrstuvwxyz"))
    def to_upper(x):
      return script_ops.numpy_function(
          func=lambda x: x.decode("utf-8").upper(), inp=[x], Tout=dtypes.string)
    dataset = dataset.map(to_upper, num_parallel_calls=dataset_ops.AUTOTUNE)
    with mock.patch.object(compat, "forward_compatible", return_value=True):
      dataset_id = data_service_ops.register_dataset(
          cluster.dispatcher.target, dataset=dataset, compression=compression)
      dataset = data_service_ops.from_dataset_id(
          processing_mode=data_service_ops.ShardingPolicy.OFF,
          service=cluster.dispatcher.target,
          dataset_id=dataset_id,
          element_spec=dataset.element_spec)
      self.assertDatasetProduces(dataset, list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))

  # Eager-only as querying `element_spec` is only supported in the eager mode.
  @combinations.generate(
      combinations.times(test_base.eager_only_combinations(),
                         combinations.combine(compression=[None, "AUTO"])))
  def testFromDatasetIdOmitsElementSpecAndCompression(self, compression):
    cluster = data_service_test_base.TestCluster(
        num_workers=1, data_transfer_protocol="grpc")
    dataset = dataset_ops.Dataset.from_tensor_slices(
        list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
    with mock.patch.object(compat, "forward_compatible", return_value=True):
      dataset_id = data_service_ops.register_dataset(
          cluster.dispatcher.target, dataset=dataset, compression=compression)
      dataset = data_service_ops.from_dataset_id(
          processing_mode=data_service_ops.ShardingPolicy.OFF,
          service=cluster.dispatcher.target,
          dataset_id=dataset_id)
      self.assertDatasetProduces(dataset, list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))

  def _testCompressionMismatch(self, dataset):
    cluster = data_service_test_base.TestCluster(
        num_workers=1, data_transfer_protocol="grpc")
    with mock.patch.object(compat, "forward_compatible", return_value=False):
      dataset_id = data_service_ops._register_dataset(
          cluster.dispatcher.target, dataset=dataset, compression=None)
      # `compression` is "AUTO" by default.
      dataset = data_service_ops._from_dataset_id(
          processing_mode=data_service_ops.ShardingPolicy.OFF,
          service=cluster.dispatcher.target,
          dataset_id=dataset_id,
          element_spec=dataset.element_spec)
      with self.assertRaises(errors.InvalidArgumentError):
        self.getDatasetOutput(dataset)

  @combinations.generate(
      combinations.times(test_base.default_test_combinations()))
  def testCompressionDtypeMismatch(self):
    dataset = dataset_ops.Dataset.from_tensor_slices(
        list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
    self._testCompressionMismatch(dataset)

  @combinations.generate(
      combinations.times(test_base.default_test_combinations()))
  def testCompressionShapeMismatch(self):
    dataset = dataset_ops.Dataset.from_tensor_slices([[1, 2], [3, 4]])
    self._testCompressionMismatch(dataset)

  # Only test eager mode since nested datasets are not allowed in graph mode.
  @combinations.generate(
      combinations.times(test_base.eager_only_combinations()))
  def testCompressionVariantMismatch(self):
    # Use a nested dataset as an example of a variant.
    dataset = dataset_ops.Dataset.from_tensors(dataset_ops.Dataset.range(10))
    self._testCompressionMismatch(dataset)


if __name__ == "__main__":
  test.main()
