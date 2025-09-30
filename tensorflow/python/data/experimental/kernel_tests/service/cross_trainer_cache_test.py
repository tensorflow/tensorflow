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
"""Tests for sharing datasets across training jobs."""

import multiprocessing

from absl.testing import parameterized
from tensorflow.python.data.experimental.kernel_tests.service import test_base as data_service_test_base
from tensorflow.python.data.experimental.ops import data_service_ops
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import combinations
from tensorflow.python.framework import errors
from tensorflow.python.platform import test


class CrossTrainerCacheTest(data_service_test_base.TestBase,
                            parameterized.TestCase):
  """Tests for sharing datasets across jobs using a cross-trainer cache."""

  @combinations.generate(test_base.default_test_combinations())
  def testEnableCrossTrainerCache(self):
    """Tests cross-trainer cache with `distribute`."""
    cluster = self._create_cluster(num_workers=1)
    dataset = dataset_ops.Dataset.range(10000000).repeat()
    dataset1 = self.make_distributed_dataset(
        dataset,
        cluster,
        job_name="job",
        cross_trainer_cache=data_service_ops.CrossTrainerCache(
            trainer_id="Trainer 1"))
    self.assertDatasetProduces(dataset1.take(10), list(range(10)))

    # The second client reads the same data from the cross-trainer cache.
    dataset2 = self.make_distributed_dataset(
        dataset,
        cluster,
        job_name="job",
        cross_trainer_cache=data_service_ops.CrossTrainerCache(
            trainer_id="Trainer 2"))
    self.assertDatasetProduces(dataset2.take(10), list(range(10)))

  @combinations.generate(test_base.default_test_combinations())
  def testFromDatasetId(self):
    """Tests cross-trainer cache with `register_dataset`/`from_dataset_id`."""
    cluster = self._create_cluster(num_workers=1)
    dataset = dataset_ops.Dataset.range(10000000).repeat()
    dataset_id1 = data_service_ops.register_dataset(
        cluster.dispatcher.target, dataset, dataset_id="dataset_id")
    dataset1 = data_service_ops.from_dataset_id(
        processing_mode=data_service_ops.ShardingPolicy.OFF,
        service=cluster.dispatcher.target,
        dataset_id=dataset_id1,
        element_spec=dataset.element_spec,
        job_name="job",
        cross_trainer_cache=data_service_ops.CrossTrainerCache(
            trainer_id="Trainer 1"))
    self.assertDatasetProduces(dataset1.take(10), list(range(10)))

    dataset_id2 = data_service_ops.register_dataset(
        cluster.dispatcher.target, dataset, dataset_id="dataset_id")
    dataset2 = data_service_ops.from_dataset_id(
        processing_mode=data_service_ops.ShardingPolicy.OFF,
        service=cluster.dispatcher.target,
        dataset_id=dataset_id2,
        element_spec=dataset.element_spec,
        job_name="job",
        cross_trainer_cache=data_service_ops.CrossTrainerCache(
            trainer_id="Trainer 2"))
    self.assertDatasetProduces(dataset2.take(10), list(range(10)))

  @combinations.generate(test_base.default_test_combinations())
  def testDisableCrossTrainerCacheByDefault(self):
    cluster = self._create_cluster(num_workers=1)
    dataset = dataset_ops.Dataset.range(10000000).repeat()
    dataset1 = self.make_distributed_dataset(dataset, cluster, job_name="job")
    self.assertDatasetProduces(dataset1.take(10), list(range(10)))

    # The two clients use the same job. The second client can't read the data
    # already read by the first client.
    dataset2 = self.make_distributed_dataset(dataset, cluster, job_name="job")
    output = self.getDatasetOutput(dataset2.take(10))
    self.assertGreaterEqual(output[0], 10)

  @combinations.generate(test_base.default_test_combinations())
  def testConcurrentReaders(self):
    # Fetching an element from the dataset will trigger prefetches of more
    # elements, one per CPU core which will be placed in the cache.
    # However if the number of prefetches exceeds the space available in
    # the cache then the sliding window will be moved forward away from
    # the element just read thus negating the use of the cache as other
    # trainers will not get the correct element.
    # Hence the need to calculate the size of the cache based on the
    # number of CPU cores and the element size of 423. The extra 8
    # entries are simply a bit of margin.
    num_cpus = multiprocessing.cpu_count()
    cluster = self._create_cluster(
        num_workers=1, cross_trainer_cache_size_bytes=(num_cpus + 8) * 423)
    num_readers = 20
    num_elements = 50
    dataset = dataset_ops.Dataset.range(10000000).repeat()

    datasets = []
    iterators = []
    for i in range(num_readers):
      distributed_dataset = self.make_distributed_dataset(
          dataset,
          cluster,
          job_name="job",
          cross_trainer_cache=data_service_ops.CrossTrainerCache(
              trainer_id=f"Trainer {i}"),
          max_outstanding_requests=1)
      iterator = self.getNext(distributed_dataset)
      datasets.append(distributed_dataset)
      iterators.append(iterator)

    for i in range(num_elements):
      # All the readers read the same element in one step.
      for j in range(num_readers):
        self.assertEqual(self.evaluate(iterators[j]()), i)

  @combinations.generate(test_base.default_test_combinations())
  def testSlowClientSkipsData(self):
    cluster = self._create_cluster(
        num_workers=1, cross_trainer_cache_size_bytes=500)
    dataset = dataset_ops.Dataset.range(10000000).repeat()
    dataset1 = self.make_distributed_dataset(
        dataset,
        cluster,
        job_name="job",
        cross_trainer_cache=data_service_ops.CrossTrainerCache(
            trainer_id="Trainer 1"))
    self.assertDatasetProduces(dataset1.take(200), list(range(200)))

    dataset2 = self.make_distributed_dataset(
        dataset,
        cluster,
        job_name="job",
        cross_trainer_cache=data_service_ops.CrossTrainerCache(
            trainer_id="Trainer 2"))
    dataset2 = dataset2.take(200)
    output = self.getDatasetOutput(dataset2)
    # When the cache is small, the second trainer couldn't read the beginning of
    # the dataset. It can still read 200 elements from the dataset, because the
    # dataset is infinite.
    self.assertGreater(output[0], 0)
    self.assertLen(output, 200)

  @combinations.generate(test_base.default_test_combinations())
  def testSmallCache(self):
    cluster = self._create_cluster(
        num_workers=1, cross_trainer_cache_size_bytes=500)
    dataset = dataset_ops.Dataset.range(10000000).repeat()
    num_readers = 20

    for i in range(num_readers):
      # Even if the cache is small and may discard old data, each trainer can
      # still read the required number of elements because the input dataset is
      # infinite.
      distributed_dataset = self.make_distributed_dataset(
          dataset,
          cluster,
          job_name="job",
          cross_trainer_cache=data_service_ops.CrossTrainerCache(
              trainer_id=f"Trainer {i}"))
      output = self.getDatasetOutput(distributed_dataset.take(200))
      self.assertLen(output, 200)

  @combinations.generate(test_base.default_test_combinations())
  def testShuffleDataset(self):
    cluster = self._create_cluster(num_workers=1)
    dataset = dataset_ops.Dataset.range(10000000).repeat().shuffle(
        buffer_size=100)
    dataset1 = self.make_distributed_dataset(
        dataset,
        cluster,
        job_name="job",
        cross_trainer_cache=data_service_ops.CrossTrainerCache(
            trainer_id="Trainer 1"))
    output1 = self.getDatasetOutput(dataset1.take(10))

    dataset2 = self.make_distributed_dataset(
        dataset,
        cluster,
        job_name="job",
        cross_trainer_cache=data_service_ops.CrossTrainerCache(
            trainer_id="Trainer 2"))
    output2 = self.getDatasetOutput(dataset2.take(10))
    self.assertEqual(output1, output2)

  @combinations.generate(test_base.default_test_combinations())
  def testSameTrainerID(self):
    # Jobs from the same training cluster do not reuse data from the cache.
    cluster = self._create_cluster(num_workers=1)
    dataset = dataset_ops.Dataset.range(10000000).repeat()
    dataset1 = self.make_distributed_dataset(
        dataset,
        cluster,
        job_name="job",
        cross_trainer_cache=data_service_ops.CrossTrainerCache(
            trainer_id="Trainer ID"))
    self.assertDatasetProduces(dataset1.take(10), list(range(10)))

    dataset2 = self.make_distributed_dataset(
        dataset,
        cluster,
        job_name="job",
        cross_trainer_cache=data_service_ops.CrossTrainerCache(
            trainer_id="Trainer ID"))
    output = self.getDatasetOutput(dataset2.take(10))
    self.assertGreaterEqual(output[0], 10)

  @combinations.generate(test_base.default_test_combinations())
  def testDifferentJobNames(self):
    cluster = self._create_cluster(num_workers=1)
    dataset = dataset_ops.Dataset.range(10000000).repeat()
    dataset1 = self.make_distributed_dataset(
        dataset,
        cluster,
        job_name="job1",
        cross_trainer_cache=data_service_ops.CrossTrainerCache(
            trainer_id="Trainer 1"))
    self.assertDatasetProduces(dataset1.take(10), list(range(10)))

    dataset2 = self.make_distributed_dataset(
        dataset,
        cluster,
        job_name="job2",
        cross_trainer_cache=data_service_ops.CrossTrainerCache(
            trainer_id="Trainer 2"))
    self.assertDatasetProduces(dataset2.take(10), list(range(10)))

  @combinations.generate(test_base.default_test_combinations())
  def testDynamicSharding(self):
    cluster = self._create_cluster(num_workers=2)
    dataset = dataset_ops.Dataset.range(10000000).repeat()
    dataset1 = self.make_distributed_dataset(
        dataset,
        cluster,
        processing_mode=data_service_ops.ShardingPolicy.DYNAMIC,
        job_name="job",
        cross_trainer_cache=data_service_ops.CrossTrainerCache(
            trainer_id="Trainer 1"))
    output1 = self.getDatasetOutput(dataset1.take(100))

    # The second client reads the same data from the cross-trainer cache.
    dataset2 = self.make_distributed_dataset(
        dataset,
        cluster,
        processing_mode=data_service_ops.ShardingPolicy.DYNAMIC,
        job_name="job",
        cross_trainer_cache=data_service_ops.CrossTrainerCache(
            trainer_id="Trainer 2"))
    output2 = self.getDatasetOutput(dataset2.take(100))
    # Verifies the intersection is non-empty.
    self.assertTrue(set(output1) & set(output2))

  @combinations.generate(test_base.default_test_combinations())
  def testNoCompression(self):
    cluster = self._create_cluster(num_workers=1)
    dataset = dataset_ops.Dataset.range(10000000).repeat()
    dataset1 = self.make_distributed_dataset(
        dataset,
        cluster,
        job_name="job",
        compression=None,
        cross_trainer_cache=data_service_ops.CrossTrainerCache(
            trainer_id="Trainer 1"))
    self.assertDatasetProduces(dataset1.take(10), list(range(10)))

    dataset2 = self.make_distributed_dataset(
        dataset,
        cluster,
        job_name="job",
        compression=None,
        cross_trainer_cache=data_service_ops.CrossTrainerCache(
            trainer_id="Trainer 2"))
    self.assertDatasetProduces(dataset2.take(10), list(range(10)))

  @combinations.generate(test_base.default_test_combinations())
  def testCompressionMismatch(self):
    cluster = self._create_cluster(num_workers=1)
    dataset = dataset_ops.Dataset.range(10000000).repeat()
    dataset1 = self.make_distributed_dataset(
        dataset,
        cluster,
        job_name="job",
        cross_trainer_cache=data_service_ops.CrossTrainerCache(
            trainer_id="Trainer 1"))
    self.assertDatasetProduces(dataset1.take(10), list(range(10)))

    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                "Data type mismatch"):
      dataset2 = self.make_distributed_dataset(
          dataset,
          cluster,
          job_name="job",
          compression=None,
          cross_trainer_cache=data_service_ops.CrossTrainerCache(
              trainer_id="Trainer 1"))
      self.getDatasetOutput(dataset2)

  @combinations.generate(test_base.default_test_combinations())
  def testRequiresJobName(self):
    cluster = self._create_cluster(num_workers=1)
    dataset = dataset_ops.Dataset.range(10000000).repeat()
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        "Cross-trainer caching requires named jobs. Got empty `job_name`."):
      dataset = self.make_distributed_dataset(
          dataset,
          cluster,
          job_name=None,
          cross_trainer_cache=data_service_ops.CrossTrainerCache(
              trainer_id="Trainer 1"))
      self.getDatasetOutput(dataset)

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(range_=[0, 10])))
  def testRequiresInfiniteDataset(self, range_):
    cluster = self._create_cluster(num_workers=1)
    dataset = dataset_ops.Dataset.range(range_).map(lambda x: x + 1)
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        "Cross-trainer caching requires the input dataset to be infinite."):
      dataset = dataset.apply(
          data_service_ops.distribute(
              processing_mode=data_service_ops.ShardingPolicy.OFF,
              service=cluster.dispatcher.target,
              job_name="job_name",
              cross_trainer_cache=data_service_ops.CrossTrainerCache(
                  trainer_id="Trainer ID")))
      self.getDatasetOutput(dataset)

  @combinations.generate(test_base.eager_only_combinations())
  def testMultipleIterationsForOneDatasetEagerMode(self):
    cluster = self._create_cluster(num_workers=1)
    dataset = dataset_ops.Dataset.range(10000000).repeat()
    dataset1 = self.make_distributed_dataset(
        dataset,
        cluster,
        job_name="job",
        cross_trainer_cache=data_service_ops.CrossTrainerCache(
            trainer_id="Trainer 1"))
    # In the eager mode, each iteration creates a new data service job and does
    # not reuse cached data. We disallow this use case.
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        "Cross-trainer caching requires infinite datasets and disallows "
        "multiple repetitions of the same dataset."):
      self.getDatasetOutput(dataset1.take(10))
      self.getDatasetOutput(dataset1.take(10))
      self.getDatasetOutput(dataset1.take(10))

  @combinations.generate(test_base.graph_only_combinations())
  def testMultipleIterationsForOneDatasetGraphMode(self):
    cluster = self._create_cluster(num_workers=1)
    dataset = dataset_ops.Dataset.range(10000000).repeat()
    dataset1 = self.make_distributed_dataset(
        dataset,
        cluster,
        job_name="job",
        cross_trainer_cache=data_service_ops.CrossTrainerCache(
            trainer_id="Trainer 1"))
    # These clients are assumed to be from the same training cluster. Thus, they
    # do not reuse data from the cross-trainer cache.
    output1 = self.getDatasetOutput(dataset1.take(10))
    output1 += self.getDatasetOutput(dataset1.take(10))
    output1 += self.getDatasetOutput(dataset1.take(10))
    self.assertLen(set(output1), 30)

    dataset2 = self.make_distributed_dataset(
        dataset,
        cluster,
        job_name="job",
        cross_trainer_cache=data_service_ops.CrossTrainerCache(
            trainer_id="Trainer 2"))
    # These clients reuse some data from the previous clients (not exactly the
    # same data due to client-side buffering).
    output2 = self.getDatasetOutput(dataset2.take(10))
    output2 += self.getDatasetOutput(dataset2.take(10))
    output2 += self.getDatasetOutput(dataset2.take(10))
    self.assertTrue(set(output1) & set(output2))

  @combinations.generate(test_base.default_test_combinations())
  def testDisallowCoordinatedRead(self):
    cluster = self._create_cluster(num_workers=1)
    dataset = dataset_ops.Dataset.range(10000000).repeat()
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        "Cross-trainer caching does not support coordinated reads."):
      dataset = self.make_distributed_dataset(
          dataset,
          cluster,
          job_name="job",
          num_consumers=1,
          consumer_index=0,
          cross_trainer_cache=data_service_ops.CrossTrainerCache(
              trainer_id="Trainer 1"))
      self.getDatasetOutput(dataset)

  @combinations.generate(test_base.default_test_combinations())
  def testNamedJobMismatch(self):
    cluster = self._create_cluster(num_workers=1)
    dataset = dataset_ops.Dataset.range(10000000).repeat()
    dataset1 = self.make_distributed_dataset(
        dataset,
        cluster,
        job_name="job",
        cross_trainer_cache=data_service_ops.CrossTrainerCache(
            trainer_id="Trainer 1"))
    self.assertDatasetProduces(dataset1.take(10), list(range(10)))

    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        "Existing cross-trainer cache: <enabled>; got <disabled>"):
      dataset2 = self.make_distributed_dataset(
          dataset, cluster, job_name="job", cross_trainer_cache=None)
      self.getDatasetOutput(dataset2)

  @combinations.generate(test_base.default_test_combinations())
  def testRequiresNonEmptyTrainerID(self):
    cluster = self._create_cluster(num_workers=2)
    dataset = dataset_ops.Dataset.range(10000000).repeat()

    with self.assertRaisesRegex(
        ValueError,
        "tf.data service cross-trainer cache requires a non-empty trainer ID."):
      self.make_distributed_dataset(
          dataset,
          cluster,
          job_name="job",
          cross_trainer_cache=data_service_ops.CrossTrainerCache(
              trainer_id=None))

  def _create_cluster(self,
                      num_workers,
                      cross_trainer_cache_size_bytes=10 * (2**30)):
    cluster = data_service_test_base.TestCluster(num_workers=0)
    for _ in range(num_workers):
      worker = data_service_test_base.TestWorker(
          dispatcher_address=cluster.dispatcher_address(),
          shutdown_quiet_period_ms=0,
          cross_trainer_cache_size_bytes=cross_trainer_cache_size_bytes)
      worker.start()
      cluster.workers.append(worker)
    return cluster


if __name__ == "__main__":
  test.main()
