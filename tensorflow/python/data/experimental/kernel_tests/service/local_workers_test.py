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
from tensorflow.python.data.experimental.ops import data_service_ops
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import combinations
from tensorflow.python.framework import errors

import multiprocessing

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
  def testDynamicSharding(self, num_local_workers, num_remote_workers):
    cluster = multi_process_cluster.MultiProcessCluster(
        num_local_workers=num_local_workers,
        num_remote_workers=num_remote_workers)
    num_elements = 100
    ds = self.make_distributed_range_dataset(
        num_elements,
        cluster,
        processing_mode=data_service_ops.ShardingPolicy.DYNAMIC,
        target_workers="LOCAL")
    self.assertDatasetProduces(
        ds, list(range(num_elements)), assert_items_equal=True)

  @combinations.generate(test_base.default_test_combinations())
  def testMultipleConsumers(self):
    num_local_workers, num_remote_workers = 1, 3
    cluster = multi_process_cluster.MultiProcessCluster(
        num_local_workers=num_local_workers,
        num_remote_workers=num_remote_workers)
    # Because the elements in datasets are prefetched one per
    # CPU core, a static number here may be excessively large
    # for small numbers of CPU cores, or too small for high
    # CPU core count machines, or probably both.
    # In this case the below formula should satisfy both needs.
    num_elements = 50 + (multiprocessing.cpu_count() * 2)
    num_consumers = 8
    iterators = []
    for _ in range(num_consumers):
      dataset = self.make_distributed_range_dataset(
          num_elements, cluster, job_name="shared_job")
      iterators.append(self.getNext(dataset))

    results = []
    for _ in range(10):
      for it in iterators:
        results.append(self.evaluate(it()))
    for it in iterators:
      results.extend(self.getIteratorOutput(it))

    self.assertCountEqual(results, (num_local_workers + num_remote_workers) *
                          list(range(num_elements)))

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
              num_local_workers=[0, 3], num_remote_workers=[1, 3])))
  def testNonLocalRead(self, num_local_workers, num_remote_workers):
    """This test ensures the remote workers are running and producing data."""

    cluster = multi_process_cluster.MultiProcessCluster(
        num_local_workers=num_local_workers,
        num_remote_workers=num_remote_workers)
    num_elements = 10
    ds = self.make_distributed_range_dataset(num_elements, cluster)
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

    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        "Local reads require local tf.data workers, but no local worker is "
        "found."):
      self.getDatasetOutput(ds)

  @combinations.generate(test_base.default_test_combinations())
  def testInconsistentTargetWorkers(self):
    cluster = multi_process_cluster.MultiProcessCluster(
        num_local_workers=3, num_remote_workers=3)
    ds = dataset_ops.Dataset.range(10)
    datasets = [
        self.make_distributed_dataset(
            ds, cluster, job_name="test_job", target_workers=target_workers)
        for target_workers in ["AUTO", "ANY", "LOCAL"]
    ]

    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        "but found an existing job with different parameters: "
        "Existing target workers: <AUTO>"):
      for dataset in datasets:
        self.getDatasetOutput(dataset)

  @combinations.generate(test_base.default_test_combinations())
  def testAnonymousJobWithDifferentTargetWorkers(self):
    num_local_workers, num_remote_workers = (3, 3)
    cluster = multi_process_cluster.MultiProcessCluster(num_local_workers,
                                                        num_remote_workers)
    num_elements = 10
    ds = dataset_ops.Dataset.range(num_elements)
    datasets = {
        target_workers: self.make_distributed_dataset(
            ds, cluster, target_workers=target_workers)
        for target_workers in ["AUTO", "ANY", "LOCAL"]
    }

    num_workers = num_local_workers + num_remote_workers
    self.assertDatasetProduces(
        datasets["AUTO"],
        num_workers * list(range(num_elements)),
        assert_items_equal=True)
    self.assertDatasetProduces(
        datasets["ANY"],
        num_workers * list(range(num_elements)),
        assert_items_equal=True)
    self.assertDatasetProduces(
        datasets["LOCAL"],
        num_local_workers * list(range(num_elements)),
        assert_items_equal=True)

  @combinations.generate(test_base.default_test_combinations())
  def testCoordinatedRead(self):
    cluster = multi_process_cluster.MultiProcessCluster(
        num_local_workers=3, num_remote_workers=3)
    ds = dataset_ops.Dataset.range(10).repeat()
    ds = self.make_distributed_dataset(
        ds,
        cluster,
        job_name="test_job",
        consumer_index=0,
        num_consumers=3,
        target_workers="LOCAL")
    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                "Coordinated reads require non-local workers"):
      self.getDatasetOutput(ds)


class LocalTaskGarbageCollectTest(data_service_test_base.TestBase,
                                  parameterized.TestCase):
  """Tests garbage collecting unused local worker tasks.

  The user typically creates an iterator in each epoch. This should delete the
  previous iterator and releases the resources of it.
  """

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(num_remote_workers=[0, 3])))
  def testMultipleEpochs(self, num_remote_workers):
    num_local_workers = 1
    cluster = multi_process_cluster.MultiProcessCluster(
        num_local_workers=num_local_workers,
        num_remote_workers=num_remote_workers)

    num_epochs, num_steps = 5, 5
    dataset = self._make_distributed_infinite_range_dataset(cluster)
    for _ in range(num_epochs):
      # For each iteration, the previous iterator is garbage collected.
      get_next = self.getNext(dataset)
      for i in range(num_steps):
        self.assertEqual(self.evaluate(get_next()), i)

  @combinations.generate(
      combinations.times(test_base.eager_only_combinations(),
                         combinations.combine(num_remote_workers=[0, 3])))
  def testMultipleEpochsSharedJob(self, num_remote_workers):
    num_local_workers = 1
    cluster = multi_process_cluster.MultiProcessCluster(
        num_local_workers=num_local_workers,
        num_remote_workers=num_remote_workers)

    num_epochs, num_steps = 5, 5
    dataset = self._make_distributed_infinite_range_dataset(
        cluster, job_name="shared_job_name")
    for _ in range(num_epochs):
      # For each iteration, the previous iterator is garbage collected.
      get_next = self.getNext(dataset)
      for i in range(num_steps):
        self.assertEqual(self.evaluate(get_next()), i)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(
              num_remote_workers=[0, 3], job_name=[None, "shared_job_name"])))
  def testRepeatDistributedDataset(self, num_remote_workers, job_name):
    num_local_workers = 1
    cluster = multi_process_cluster.MultiProcessCluster(
        num_local_workers=num_local_workers,
        num_remote_workers=num_remote_workers)
    dataset = self.make_distributed_range_dataset(
        10, cluster, job_name=job_name, target_workers="LOCAL")
    dataset = dataset.repeat(3)
    self.assertDatasetProduces(dataset, list(range(10)) * 3)

  @combinations.generate(
      combinations.times(test_base.eager_only_combinations(),
                         combinations.combine(num_remote_workers=[0, 3])))
  def testReadFromDeletedTask(self, num_remote_workers):
    num_local_workers = 1
    cluster = multi_process_cluster.MultiProcessCluster(
        num_local_workers=num_local_workers,
        num_remote_workers=num_remote_workers)

    num_steps = 10
    dataset = self._make_distributed_infinite_range_dataset(
        cluster, job_name="shared_job_name")
    get_next = self.getNext(dataset)
    for i in range(num_steps):
      self.assertEqual(self.evaluate(get_next()), i)

    # Re-creating the dataset resets the iterator index, so the second iterator
    # reads from the same task as the first, which has been deleted.
    dataset = self._make_distributed_infinite_range_dataset(
        cluster, job_name="shared_job_name")
    get_next = self.getNext(dataset)
    with self.assertRaisesRegex(errors.FailedPreconditionError,
                                "which has been deleted."):
      _ = self.evaluate(get_next())

  @combinations.generate(
      combinations.times(test_base.graph_only_combinations(),
                         combinations.combine(num_remote_workers=[0, 3])))
  def testReadFromDeletedTask_GraphMode(self, num_remote_workers):
    num_local_workers = 1
    cluster = multi_process_cluster.MultiProcessCluster(
        num_local_workers=num_local_workers,
        num_remote_workers=num_remote_workers)

    num_steps = 10
    dataset = self._make_distributed_infinite_range_dataset(
        cluster, job_name="shared_job_name")
    with self.session() as sess:
      get_next = self.getNext(dataset)
      for i in range(num_steps):
        self.assertEqual(sess.run(get_next()), i)

    # Re-creating the dataset resets the iterator index, so the second iterator
    # reads from the same task as the first, which has been deleted.
    dataset = self._make_distributed_infinite_range_dataset(
        cluster, job_name="shared_job_name")
    with self.assertRaisesRegex(errors.FailedPreconditionError,
                                "which has been deleted."):
      with self.session() as sess:
        get_next = self.getNext(dataset)
        sess.run(get_next())

  @combinations.generate(
      combinations.times(test_base.eager_only_combinations(),
                         combinations.combine(num_remote_workers=[0, 3])))
  def testMultipleEpochs_WorkerRestart(self, num_remote_workers):
    num_local_workers = 1
    cluster = multi_process_cluster.MultiProcessCluster(
        num_local_workers=num_local_workers,
        num_remote_workers=num_remote_workers)

    num_steps = 10
    dataset = self._make_distributed_infinite_range_dataset(
        cluster, job_name="shared_job_name")

    get_next = self.getNext(dataset)
    for i in range(num_steps):
      self.assertEqual(self.evaluate(get_next()), i)

    # Verifies the worker re-creates the task after the iterator is deleted and
    # the worker restarts.
    del get_next
    cluster.restart_local_workers()

    get_next = self.getNext(dataset)
    for i in range(num_steps):
      self.assertEqual(self.evaluate(get_next()), i)

  @combinations.generate(
      combinations.times(test_base.eager_only_combinations(),
                         combinations.combine(num_remote_workers=[0, 3])))
  def testMultipleEpochs_DispatcherRestart(self, num_remote_workers):
    num_local_workers = 1
    cluster = multi_process_cluster.MultiProcessCluster(
        num_local_workers=num_local_workers,
        num_remote_workers=num_remote_workers)

    num_steps = 10
    dataset = self._make_distributed_infinite_range_dataset(
        cluster, job_name="shared_job_name")
    get_next = self.getNext(dataset)
    for i in range(num_steps):
      self.assertEqual(self.evaluate(get_next()), i)

    # Verifies the worker re-creates the task after the iterator is deleted and
    # the dispatcher restarts.
    del get_next
    cluster.restart_dispatcher()

    get_next = self.getNext(dataset)
    for i in range(num_steps):
      self.assertEqual(self.evaluate(get_next()), i)

  def _make_distributed_infinite_range_dataset(self, cluster, job_name=None):
    dataset = dataset_ops.Dataset.range(1000000).repeat()
    return self.make_distributed_dataset(
        dataset,
        cluster=cluster,
        job_name=job_name,
        processing_mode=data_service_ops.ShardingPolicy.OFF,
        target_workers="LOCAL")


if __name__ == "__main__":
  multi_process_cluster.test_main()
