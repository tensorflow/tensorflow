# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for sharing datasets across jobs."""

from absl.testing import parameterized

from tensorflow.python.data.experimental.kernel_tests.service import test_base as data_service_test_base
from tensorflow.python.data.experimental.ops import data_service_ops
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import combinations
from tensorflow.python.framework import errors
from tensorflow.python.platform import test


class SharedDatasetTest(data_service_test_base.TestBase,
                        parameterized.TestCase):

  @combinations.generate(
      combinations.times(test_base.default_test_combinations()))
  def testSharedDataset(self):
    cluster = data_service_test_base.TestCluster(num_workers=1)
    dataset1 = self._make_distributed_range_dataset(
        10, cluster, job_name="vizier_job", dataset_name="vizier_dataset")
    self.assertDatasetProduces(dataset1, list(range(10)))

    # The second dataset is empty since the dataset is shared, and the first
    # client has read all elements from the dataset.
    dataset2 = self._make_distributed_range_dataset(
        10, cluster, job_name="vizier_job", dataset_name="vizier_dataset")
    self.assertDatasetProduces(dataset2, list())

  @combinations.generate(
      combinations.times(test_base.default_test_combinations()))
  def testRepeatedDataset(self):
    cluster = data_service_test_base.TestCluster(num_workers=1)
    dataset1 = dataset_ops.Dataset.range(10).repeat(5)
    dataset1 = dataset1.apply(
        data_service_ops.distribute(
            data_service_ops.ShardingPolicy.OFF,
            cluster.dispatcher_address(),
            job_name="vizier_job",
            dataset_name="vizier_dataset"))

    dataset2 = dataset_ops.Dataset.range(10).repeat(5)
    dataset2 = dataset2.apply(
        data_service_ops.distribute(
            data_service_ops.ShardingPolicy.OFF,
            cluster.dispatcher_address(),
            job_name="vizier_job",
            dataset_name="vizier_dataset"))

    # The second dataset is empty since the dataset is shared, and the first
    # client has read all elements from the dataset.
    self.assertDatasetProduces(dataset1, list(range(10)) * 5)
    self.assertDatasetProduces(dataset2, list())

  @combinations.generate(
      combinations.times(test_base.default_test_combinations()))
  def testInfiniteDataset(self):
    cluster = data_service_test_base.TestCluster(num_workers=1)
    dataset1 = dataset_ops.Dataset.range(1000000).repeat()
    dataset1 = dataset1.apply(
        data_service_ops.distribute(
            data_service_ops.ShardingPolicy.OFF,
            cluster.dispatcher_address(),
            job_name="vizier_job",
            dataset_name="vizier_dataset"))
    get_next1 = self.getNext(dataset1)

    dataset2 = dataset_ops.Dataset.range(1000000).repeat()
    dataset2 = dataset2.apply(
        data_service_ops.distribute(
            data_service_ops.ShardingPolicy.OFF,
            cluster.dispatcher_address(),
            job_name="vizier_job",
            dataset_name="vizier_dataset"))
    get_next2 = self.getNext(dataset2)

    result1 = {self.evaluate(get_next1()) for _ in range(10)}
    result2 = {self.evaluate(get_next2()) for _ in range(10)}
    self.assertLen(result1, 10)
    self.assertLen(result2, 10)
    # Since the clients read from a shared dataset, the results should not
    # overlap.
    self.assertEmpty(result1 & result2)

  @combinations.generate(
      combinations.times(test_base.default_test_combinations()))
  def testDifferentDatasetNames(self):
    # The dataset is not shared. Both clients will read the complete dataset.
    cluster = data_service_test_base.TestCluster(num_workers=1)
    dataset1 = self._make_distributed_range_dataset(
        10, cluster, job_name="vizier_job", dataset_name="vizier_dataset1")
    self.assertDatasetProduces(dataset1, list(range(10)))

    dataset2 = self._make_distributed_range_dataset(
        10, cluster, job_name="vizier_job", dataset_name="vizier_dataset2")
    self.assertDatasetProduces(dataset2, list(range(10)))

  @combinations.generate(
      combinations.times(test_base.default_test_combinations()))
  def testDifferentJobNames(self):
    # The dataset is not shared. Both clients will read the complete dataset.
    cluster = data_service_test_base.TestCluster(num_workers=1)
    dataset1 = self._make_distributed_range_dataset(
        10, cluster, job_name="vizier_job1", dataset_name="vizier_dataset")
    self.assertDatasetProduces(dataset1, list(range(10)))

    dataset2 = self._make_distributed_range_dataset(
        10, cluster, job_name="vizier_job2", dataset_name="vizier_dataset")
    self.assertDatasetProduces(dataset2, list(range(10)))

  @combinations.generate(
      combinations.times(test_base.default_test_combinations()))
  def testNamedAndUnnamedDatasets(self):
    # The dataset is not shared. Both clients will read the complete dataset.
    cluster = data_service_test_base.TestCluster(num_workers=1)
    dataset1 = self._make_distributed_range_dataset(
        10, cluster, job_name="vizier_job")
    self.assertDatasetProduces(dataset1, list(range(10)))

    dataset2 = self._make_distributed_range_dataset(
        10, cluster, job_name="vizier_job", dataset_name="vizier_dataset")
    self.assertDatasetProduces(dataset2, list(range(10)))

  @combinations.generate(
      combinations.times(test_base.default_test_combinations()))
  def testInvalidNamedDataset(self):
    cluster = data_service_test_base.TestCluster(num_workers=1)
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        "Datasets with the same name should have the same structure"):
      dataset1 = self._make_distributed_range_dataset(
          10, cluster, job_name="vizier_job", dataset_name="vizier_dataset")
      dataset2 = self._make_distributed_range_dataset(
          20, cluster, job_name="vizier_job", dataset_name="vizier_dataset")
      _ = self.getDatasetOutput(dataset1)
      _ = self.getDatasetOutput(dataset2)

  def _make_distributed_range_dataset(self,
                                      num_elements,
                                      cluster,
                                      job_name=None,
                                      dataset_name=None):
    dataset = dataset_ops.Dataset.range(num_elements)
    dataset_id = data_service_ops.register_dataset(
        cluster.dispatcher.target, dataset, dataset_name=dataset_name)
    return data_service_ops.from_dataset_id(
        dataset_id=dataset_id,
        processing_mode=data_service_ops.ShardingPolicy.OFF,
        service=cluster.dispatcher.target,
        element_spec=dataset.element_spec,
        job_name=job_name)


if __name__ == "__main__":
  test.main()
