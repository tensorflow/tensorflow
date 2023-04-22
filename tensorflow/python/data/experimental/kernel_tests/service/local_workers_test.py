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
"""Tests tf.data service with local and remote workers."""

from absl.testing import parameterized

from tensorflow.python.data.experimental.kernel_tests.service import multi_process_cluster
from tensorflow.python.data.experimental.kernel_tests.service import test_base as data_service_test_base
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import combinations
from tensorflow.python.framework import errors


class LocalWorkersTest(data_service_test_base.TestBase, parameterized.TestCase):
  """Tests reading from local workers if `target_workers` is `local`."""

  @combinations.generate(test_base.default_test_combinations())
  def testOneLocalWorker(self):
    cluster = multi_process_cluster.MultiProcessCluster(
        num_local_workers=1, num_remote_workers=5)
    num_elements = 10
    ds = self.make_distributed_range_dataset(
        num_elements, cluster, target_workers="local")
    self.assertDatasetProduces(ds, list(range(num_elements)))

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(
              num_local_workers=[1, 3], num_remote_workers=[0, 3])))
  def testLocalWorkers(self, num_local_workers, num_remote_workers):
    cluster = multi_process_cluster.MultiProcessCluster(
        num_local_workers=num_local_workers,
        num_remote_workers=num_remote_workers)
    num_elements = 10
    ds = self.make_distributed_range_dataset(
        num_elements, cluster, target_workers="LOCAL")
    self.assertDatasetProduces(
        ds,
        num_local_workers * list(range(num_elements)),
        assert_items_equal=True)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(
              num_local_workers=[1, 3], num_remote_workers=[0, 3])))
  def testRepeatedDataset(self, num_local_workers, num_remote_workers):
    cluster = multi_process_cluster.MultiProcessCluster(
        num_local_workers=num_local_workers,
        num_remote_workers=num_remote_workers)
    num_elements = 10
    num_repetitions = 5
    ds = self.make_distributed_range_dataset(
        num_elements, cluster, target_workers="LOCAL")
    ds = ds.repeat(num_repetitions)
    self.assertDatasetProduces(
        ds,
        expected_output=num_local_workers * num_repetitions *
        list(range(num_elements)),
        assert_items_equal=True)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(
              num_local_workers=[1, 3], num_remote_workers=[0, 3])))
  def testPrefetchingDataset(self, num_local_workers, num_remote_workers):
    cluster = multi_process_cluster.MultiProcessCluster(
        num_local_workers=num_local_workers,
        num_remote_workers=num_remote_workers)
    num_elements = 10
    ds = self.make_distributed_range_dataset(
        num_elements, cluster, target_workers="LOCAL")
    ds = ds.prefetch(10)
    self.assertDatasetProduces(
        ds,
        expected_output=num_local_workers * list(range(num_elements)),
        assert_items_equal=True)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(
              num_local_workers=[1, 3], num_remote_workers=[0, 3])))
  def testMultipleEpochs(self, num_local_workers, num_remote_workers):
    cluster = multi_process_cluster.MultiProcessCluster(
        num_local_workers=num_local_workers,
        num_remote_workers=num_remote_workers)
    num_elements = 10
    ds = self.make_distributed_range_dataset(
        num_elements, cluster, target_workers="LOCAL")
    for _ in range(10):
      self.assertDatasetProduces(
          ds,
          num_local_workers * list(range(num_elements)),
          assert_items_equal=True)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(
              num_local_workers=[1, 3], num_remote_workers=[0, 3])))
  def testDistributedEpoch(self, num_local_workers, num_remote_workers):
    cluster = multi_process_cluster.MultiProcessCluster(
        num_local_workers=num_local_workers,
        num_remote_workers=num_remote_workers)
    num_elements = 100
    ds = self.make_distributed_range_dataset(
        num_elements,
        cluster,
        processing_mode="distributed_epoch",
        target_workers="LOCAL")
    self.assertDatasetProduces(
        ds, list(range(num_elements)), assert_items_equal=True)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(
              num_local_workers=[1, 3], num_remote_workers=[0, 3])))
  def testEmptyDataset(self, num_local_workers, num_remote_workers):
    cluster = multi_process_cluster.MultiProcessCluster(
        num_local_workers=num_local_workers,
        num_remote_workers=num_remote_workers)
    num_elements = 0
    ds = self.make_distributed_range_dataset(
        num_elements, cluster, target_workers="LOCAL")
    self.assertDatasetProduces(ds, [])

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(
              num_local_workers=[1, 3], num_remote_workers=[0, 3])))
  def testNonLocalRead(self, num_local_workers, num_remote_workers):
    """This test ensures the remote workers are running and producing data."""

    cluster = multi_process_cluster.MultiProcessCluster(
        num_local_workers=num_local_workers,
        num_remote_workers=num_remote_workers)
    num_elements = 10
    ds = self.make_distributed_range_dataset(
        num_elements, cluster, target_workers="any")
    num_workers = num_local_workers + num_remote_workers
    self.assertDatasetProduces(
        ds, num_workers * list(range(num_elements)), assert_items_equal=True)

  @combinations.generate(test_base.default_test_combinations())
  def testNoLocalWorker(self):
    cluster = multi_process_cluster.MultiProcessCluster(
        num_local_workers=0, num_remote_workers=3)
    num_elements = 10
    ds = self.make_distributed_range_dataset(
        num_elements, cluster, target_workers="LOCAL")
    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                "no local worker is found"):
      get_next = self.getNext(ds)
      self.evaluate(get_next())

  @combinations.generate(test_base.default_test_combinations())
  def testCoordinatedRead(self):
    cluster = multi_process_cluster.MultiProcessCluster(
        num_local_workers=3, num_remote_workers=3)
    ds = dataset_ops.Dataset.range(10).repeat()
    ds = self.make_distributed_dataset(
        ds,
        cluster,
        job_name="test",
        consumer_index=0,
        num_consumers=3,
        target_workers="LOCAL")
    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                "Coordinated reads require non-local workers"):
      get_next = self.getNext(ds)
      self.evaluate(get_next())


if __name__ == "__main__":
  multi_process_cluster.test_main()
