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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import traverse
from tensorflow.python.framework import test_util
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


class TraverseTest(test.TestCase):

  @test_util.run_deprecated_v1
  def testOnlySource(self):
    ds = dataset_ops.Dataset.range(10)
    variant_tensor_ops = traverse.obtain_all_variant_tensor_ops(ds)
    self.assertAllEqual(["RangeDataset"], [x.name for x in variant_tensor_ops])

  @test_util.run_deprecated_v1
  def testSimplePipeline(self):
    ds = dataset_ops.Dataset.range(10).map(math_ops.square)
    variant_tensor_ops = traverse.obtain_all_variant_tensor_ops(ds)
    self.assertSetEqual(
        set(["MapDataset", "RangeDataset"]),
        set(x.name for x in variant_tensor_ops))

  @test_util.run_deprecated_v1
  def testConcat(self):
    ds1 = dataset_ops.Dataset.range(10)
    ds2 = dataset_ops.Dataset.range(10)
    ds = ds1.concatenate(ds2)
    variant_tensor_ops = traverse.obtain_all_variant_tensor_ops(ds)
    self.assertSetEqual(
        set(["ConcatenateDataset", "RangeDataset", "RangeDataset_1"]),
        set(x.name for x in variant_tensor_ops))

  @test_util.run_deprecated_v1
  def testZip(self):
    ds1 = dataset_ops.Dataset.range(10)
    ds2 = dataset_ops.Dataset.range(10)
    ds = dataset_ops.Dataset.zip((ds1, ds2))
    variant_tensor_ops = traverse.obtain_all_variant_tensor_ops(ds)
    self.assertSetEqual(
        set(["ZipDataset", "RangeDataset", "RangeDataset_1"]),
        set(x.name for x in variant_tensor_ops))

  @test_util.run_deprecated_v1
  def testMultipleVariantTensors(self):
    ds = dataset_ops.Dataset.range(10)
    ds = _TestDataset(ds)
    variant_tensor_ops = traverse.obtain_all_variant_tensor_ops(ds)
    self.assertSetEqual(
        set(["RangeDataset", "ModelDataset", "PrefetchDataset"]),
        set(x.name for x in variant_tensor_ops))

  @test_util.run_deprecated_v1
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


if __name__ == "__main__":
  test.main()
