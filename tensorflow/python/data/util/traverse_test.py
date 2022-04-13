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
"""Tests for utilities for traversing the dataset construction graph."""

from absl.testing import parameterized

from tensorflow.python.compat import compat
from tensorflow.python.data.experimental.ops import data_service_ops
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import traverse
from tensorflow.python.framework import combinations
from tensorflow.python.ops import gen_dataset_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class _TestDataset(dataset_ops.UnaryUnchangedStructureDataset):

  def __init__(self, input_dataset):
    self._input_dataset = input_dataset
    temp_variant_tensor = gen_dataset_ops.prefetch_dataset(
        input_dataset._variant_tensor,
        buffer_size=1,
        **self._flat_structure)
    variant_tensor = gen_dataset_ops.model_dataset(
        temp_variant_tensor, **self._flat_structure)
    super(_TestDataset, self).__init__(input_dataset, variant_tensor)


class TraverseTest(test_base.DatasetTestBase, parameterized.TestCase):

  @combinations.generate(test_base.graph_only_combinations())
  def testOnlySource(self):
    ds = dataset_ops.Dataset.range(10)
    variant_tensor_ops = traverse.obtain_all_variant_tensor_ops(ds)
    self.assertAllEqual(["RangeDataset"], [x.name for x in variant_tensor_ops])

  @combinations.generate(test_base.graph_only_combinations())
  def testSimplePipeline(self):
    ds = dataset_ops.Dataset.range(10).map(math_ops.square)
    variant_tensor_ops = traverse.obtain_all_variant_tensor_ops(ds)
    self.assertSetEqual(
        set(["MapDataset", "RangeDataset"]),
        set(x.name for x in variant_tensor_ops))

  @combinations.generate(test_base.graph_only_combinations())
  def testConcat(self):
    ds1 = dataset_ops.Dataset.range(10)
    ds2 = dataset_ops.Dataset.range(10)
    ds = ds1.concatenate(ds2)
    variant_tensor_ops = traverse.obtain_all_variant_tensor_ops(ds)
    self.assertSetEqual(
        set(["ConcatenateDataset", "RangeDataset", "RangeDataset_1"]),
        set(x.name for x in variant_tensor_ops))

  @combinations.generate(test_base.graph_only_combinations())
  def testZip(self):
    ds1 = dataset_ops.Dataset.range(10)
    ds2 = dataset_ops.Dataset.range(10)
    ds = dataset_ops.Dataset.zip((ds1, ds2))
    variant_tensor_ops = traverse.obtain_all_variant_tensor_ops(ds)
    self.assertSetEqual(
        set(["ZipDataset", "RangeDataset", "RangeDataset_1"]),
        set(x.name for x in variant_tensor_ops))

  @combinations.generate(test_base.graph_only_combinations())
  def testMultipleVariantTensors(self):
    ds = dataset_ops.Dataset.range(10)
    ds = _TestDataset(ds)
    variant_tensor_ops = traverse.obtain_all_variant_tensor_ops(ds)
    self.assertSetEqual(
        set(["RangeDataset", "ModelDataset", "PrefetchDataset"]),
        set(x.name for x in variant_tensor_ops))

  @combinations.generate(test_base.graph_only_combinations())
  def testFlatMap(self):
    ds1 = dataset_ops.Dataset.range(10).repeat(10)

    def map_fn(ds):

      def _map(x):
        return ds.batch(x)

      return _map

    ds2 = dataset_ops.Dataset.range(20).prefetch(1)
    ds2 = ds2.flat_map(map_fn(ds1))
    variant_tensor_ops = traverse.obtain_all_variant_tensor_ops(ds2)
    self.assertSetEqual(
        set([
            "FlatMapDataset", "PrefetchDataset", "RepeatDataset",
            "RangeDataset", "RangeDataset_1"
        ]), set(x.name for x in variant_tensor_ops))

  @combinations.generate(test_base.graph_only_combinations())
  def testTfDataService(self):
    ds = dataset_ops.Dataset.range(10)
    ds = ds.apply(
        data_service_ops.distribute("parallel_epochs", "grpc://foo:0"))
    ops = traverse.obtain_capture_by_value_ops(ds)
    data_service_dataset_op = ("DataServiceDatasetV3"
                               if compat.forward_compatible(2021, 12, 10) else
                               "DataServiceDatasetV2")
    self.assertContainsSubset(
        ["RangeDataset", data_service_dataset_op, "DummyIterationCounter"],
        set(x.name for x in ops))


if __name__ == "__main__":
  test.main()
