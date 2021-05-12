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
"""Tests for distributed_epoch processing mode."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

from tensorflow.python.data.experimental.kernel_tests.service import test_base as data_service_test_base
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import combinations
from tensorflow.python.framework import errors
from tensorflow.python.platform import test


class DistributedEpochTest(data_service_test_base.TestBase,
                           parameterized.TestCase):

  @combinations.generate(test_base.default_test_combinations())
  def testBasic(self):
    cluster = data_service_test_base.TestCluster(num_workers=2)
    num_elements = 100
    ds = dataset_ops.Dataset.range(num_elements)
    ds = self.make_distributed_dataset(
        ds, cluster, processing_mode="distributed_epoch")
    self.assertDatasetProduces(
        ds, list(range(num_elements)), assert_items_equal=True)

  @combinations.generate(test_base.default_test_combinations())
  def testDistributeTensorSlices(self):
    cluster = data_service_test_base.TestCluster(num_workers=2)
    vals = [5, 1, 2, 4]
    ds = dataset_ops.Dataset.from_tensor_slices(vals)
    ds = self.make_distributed_dataset(
        ds, cluster, processing_mode="distributed_epoch")
    self.assertDatasetProduces(ds, vals, assert_items_equal=True)

  @combinations.generate(test_base.default_test_combinations())
  def testDistributeInterleave(self):
    cluster = data_service_test_base.TestCluster(num_workers=2)
    elements = [1, 5, 0]
    ds = dataset_ops.Dataset.from_tensor_slices(elements)
    ds = ds.interleave(lambda x: dataset_ops.Dataset.from_tensor_slices([x]))
    ds = self.make_distributed_dataset(
        ds, cluster, processing_mode="distributed_epoch")
    self.assertDatasetProduces(ds, elements, assert_items_equal=True)

  @combinations.generate(test_base.default_test_combinations())
  def testDistributeParallelInterleave(self):
    cluster = data_service_test_base.TestCluster(num_workers=2)
    elements = [1, 5, 0]
    ds = dataset_ops.Dataset.from_tensor_slices(elements)
    ds = ds.interleave(
        lambda x: dataset_ops.Dataset.from_tensor_slices([x]),
        num_parallel_calls=dataset_ops.AUTOTUNE)
    ds = self.make_distributed_dataset(
        ds, cluster, processing_mode="distributed_epoch")
    self.assertDatasetProduces(ds, elements, assert_items_equal=True)

  @combinations.generate(test_base.default_test_combinations())
  def testDistributeFlatMap(self):
    cluster = data_service_test_base.TestCluster(num_workers=2)
    elements = [1, 5, 0]
    ds = dataset_ops.Dataset.from_tensor_slices(elements)
    ds = ds.flat_map(lambda x: dataset_ops.Dataset.from_tensor_slices([x]))
    ds = self.make_distributed_dataset(
        ds, cluster, processing_mode="distributed_epoch")
    self.assertDatasetProduces(ds, elements, assert_items_equal=True)

  @combinations.generate(test_base.default_test_combinations())
  def testDistributeRepeat(self):
    cluster = data_service_test_base.TestCluster(num_workers=2)
    num_repeats = 5
    num_elements = 20
    ds = dataset_ops.Dataset.range(num_elements).repeat(num_repeats)
    ds = self.make_distributed_dataset(
        ds, cluster, processing_mode="distributed_epoch")
    self.assertDatasetProduces(
        ds, num_repeats * list(range(num_elements)), assert_items_equal=True)

  @combinations.generate(test_base.default_test_combinations())
  def testDistributeForeverRepeat(self):
    cluster = data_service_test_base.TestCluster(num_workers=2)
    num_elements = 20
    elements_to_read = 1000
    ds = dataset_ops.Dataset.range(num_elements).repeat()
    ds = self.make_distributed_dataset(
        ds, cluster, processing_mode="distributed_epoch")
    get_next = self.getNext(ds)
    results = {}
    for _ in range(elements_to_read):
      val = self.evaluate(get_next())
      if val not in results:
        results[val] = 0
      results[val] += 1
    for i in range(num_elements):
      self.assertGreater(results[i], elements_to_read / num_elements / 2)

  @combinations.generate(test_base.default_test_combinations())
  def testDistributeForeverRepeatFewElements(self):
    num_workers = 5
    cluster = data_service_test_base.TestCluster(num_workers=num_workers)
    # Less than the number of workers, so that some workers get zero elements on
    # the first repetition.
    num_elements = 1
    ds = dataset_ops.Dataset.range(num_elements).repeat()
    ds = self.make_distributed_dataset(
        ds, cluster, processing_mode="distributed_epoch")
    get_next = self.getNext(ds)
    for _ in range(20):
      self.assertEqual(self.evaluate(get_next()), 0)

    # Stop all but one worker and check that we can still read.
    for i in range(num_workers - 1):
      cluster.workers[i].stop()
    for _ in range(20):
      self.assertEqual(self.evaluate(get_next()), 0)

  @combinations.generate(test_base.default_test_combinations())
  def testDistributeShuffleAndRepeat(self):
    cluster = data_service_test_base.TestCluster(num_workers=2)
    num_repeats = 5
    num_elements = 20
    ds = dataset_ops.Dataset.range(num_elements).shuffle(num_elements).repeat(
        num_repeats)
    ds = self.make_distributed_dataset(
        ds, cluster, processing_mode="distributed_epoch")
    self.assertDatasetProduces(
        ds, num_repeats * list(range(num_elements)), assert_items_equal=True)

  @combinations.generate(test_base.default_test_combinations())
  def testOnZippedDataset(self):
    ds_1 = dataset_ops.Dataset.range(10)
    ds_2 = dataset_ops.Dataset.range(10)
    cluster = data_service_test_base.TestCluster(num_workers=1)

    ds_3 = dataset_ops.Dataset.zip((ds_1, ds_2))
    ds_3 = self.make_distributed_dataset(
        ds_3, cluster, processing_mode="distributed_epoch")

    error_regex = ("Cannot create a split provider for dataset " +
                   "of type ZipDataset")
    with self.assertRaisesRegex(errors.UnimplementedError, error_regex):
      self.getDatasetOutput(ds_3)

  @combinations.generate(test_base.default_test_combinations())
  def testOnDistributedDataset(self):
    cluster_1 = data_service_test_base.TestCluster(num_workers=1)
    cluster_2 = data_service_test_base.TestCluster(num_workers=1)
    num_sizes = 10
    size_repeats = 5
    numbers = [1 * i for i in range(num_sizes)] * size_repeats
    ds = dataset_ops.Dataset.from_tensor_slices(numbers)
    ds = self.make_distributed_dataset(
        ds, cluster_1, processing_mode="parallel_epochs")
    ds = ds.map(lambda x: x + 1)
    ds = self.make_distributed_dataset(
        ds, cluster_2, processing_mode="distributed_epoch")

    error_regex = ("Cannot create a split provider for dataset " +
                   "of type DataServiceDataset")
    with self.assertRaisesRegex(errors.UnimplementedError, error_regex):
      self.getDatasetOutput(ds)


if __name__ == "__main__":
  test.main()
