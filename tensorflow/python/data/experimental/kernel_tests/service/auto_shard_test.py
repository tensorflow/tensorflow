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
"""Tests auto-sharding datasets with tf.data service."""

from absl.testing import parameterized

from tensorflow.core.protobuf import data_service_pb2
from tensorflow.python.data.experimental.kernel_tests.service import multi_process_cluster
from tensorflow.python.data.experimental.kernel_tests.service import test_base as data_service_test_base
from tensorflow.python.data.experimental.kernel_tests.service.multi_process_cluster import MultiProcessCluster
from tensorflow.python.data.experimental.ops import data_service_ops
from tensorflow.python.data.experimental.ops import distribute
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.kernel_tests import tf_record_test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import readers
from tensorflow.python.framework import combinations
from tensorflow.python.framework import errors


def _make_service_cluster(
    num_workers,
    local_shard_index,
    worker_addresses=None,
    deployment_mode=data_service_pb2.DEPLOYMENT_MODE_COLOCATED):
  if worker_addresses is None:
    worker_addresses = ["localhost" for _ in range(num_workers)]

  cluster = MultiProcessCluster(
      num_local_workers=0,
      num_remote_workers=0,
      worker_addresses=worker_addresses,
      deployment_mode=deployment_mode)
  for _ in range(local_shard_index):
    cluster.start_remote_worker()
  cluster.start_local_worker()
  for _ in range(num_workers - local_shard_index - 1):
    cluster.start_remote_worker()
  return cluster


# pylint:disable=g-complex-comprehension
class AutoShardTest(data_service_test_base.TestBase,
                    tf_record_test_base.TFRecordTestBase,
                    parameterized.TestCase):
  """Tests auto-sharding datasets with tf.data service."""

  def setUp(self):
    super(AutoShardTest, self).setUp()
    self._num_files = 10
    self._num_records = 10
    self._filenames = self._createFiles()

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(sharding_policy=[
              data_service_ops.ShardingPolicy.DATA,
              data_service_ops.ShardingPolicy.FILE_OR_DATA
          ])))
  def testRangeDataset_AutoShard(self, sharding_policy):
    cluster = _make_service_cluster(num_workers=5, local_shard_index=1)
    dataset = dataset_ops.Dataset.range(20)
    dataset = self.make_distributed_dataset(
        dataset, cluster=cluster, processing_mode=sharding_policy)
    self.assertDatasetProduces(dataset, [1, 6, 11, 16])

  @combinations.generate(test_base.default_test_combinations())
  def testRangeDataset_FileShard(self):
    cluster = _make_service_cluster(num_workers=5, local_shard_index=1)
    dataset = dataset_ops.Dataset.range(20)
    dataset = self.make_distributed_dataset(
        dataset,
        cluster=cluster,
        processing_mode=data_service_ops.ShardingPolicy.FILE)
    with self.assertRaisesRegex(errors.NotFoundError,
                                "Found an unshardable source dataset"):
      self.getDatasetOutput(dataset)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(worker_index=[distribute.SHARD_HINT, 0, 5])))
  def testRangeDataset_ShardHint(self, worker_index):
    cluster = _make_service_cluster(num_workers=5, local_shard_index=1)
    dataset = dataset_ops.Dataset.range(20)
    # With HINT sharding, `num_shards` should be `SHARD_HINT`; `index` can be
    # any value.
    dataset = dataset.shard(
        num_shards=distribute.SHARD_HINT, index=worker_index)
    dataset = self.make_distributed_dataset(
        dataset,
        cluster=cluster,
        processing_mode=data_service_ops.ShardingPolicy.HINT)
    self.assertDatasetProduces(dataset, [1, 6, 11, 16])

  @combinations.generate(test_base.default_test_combinations())
  def testRangeDataset_InvalidWorkerIndexUsingShardHint(self):
    cluster = _make_service_cluster(num_workers=5, local_shard_index=1)
    dataset = dataset_ops.Dataset.range(20)
    # With HINT sharding, `SHARD_HINT` should be passed to `num_shards`, not
    # `index`.
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        r"Index must be between 0 and 4 \(currently index = -1\)."):
      dataset = dataset.shard(num_shards=5, index=distribute.SHARD_HINT)
      dataset = self.make_distributed_dataset(
          dataset,
          cluster=cluster,
          processing_mode=data_service_ops.ShardingPolicy.HINT)
      self.getDatasetOutput(dataset)

  @combinations.generate(test_base.default_test_combinations())
  def testRangeDataset_NoShardHint(self):
    cluster = _make_service_cluster(num_workers=5, local_shard_index=1)
    dataset = dataset_ops.Dataset.range(20)
    # No SHARD_HINT is provided. The given sharding arguments will be used.
    dataset = dataset.shard(num_shards=1, index=0)
    dataset = self.make_distributed_dataset(
        dataset,
        cluster=cluster,
        processing_mode=data_service_ops.ShardingPolicy.HINT)
    self.assertDatasetProduces(dataset, list(range(20)))

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(sharding_policy=[
              data_service_ops.ShardingPolicy.OFF,
              data_service_ops.ShardingPolicy.FILE_OR_DATA
          ])))
  def testRangeDataset_ShardHintUsedInWrongShardingPolicy(
      self, sharding_policy):
    cluster = _make_service_cluster(num_workers=5, local_shard_index=1)
    dataset = dataset_ops.Dataset.range(20)
    dataset = dataset.shard(distribute.SHARD_HINT, distribute.SHARD_HINT)
    dataset = self.make_distributed_dataset(
        dataset, cluster=cluster, processing_mode=sharding_policy)
    with self.assertRaisesRegex(
        errors.FailedPreconditionError, "tf.data service with "
        "`tf.data.experimental.service.ShardingPolicy.HINT` processing mode."):
      self.getDatasetOutput(dataset)

  @combinations.generate(test_base.default_test_combinations())
  def testRangeDataset_NoShard(self):
    cluster = _make_service_cluster(num_workers=5, local_shard_index=1)
    dataset = dataset_ops.Dataset.range(20)
    dataset = self.make_distributed_dataset(
        dataset,
        cluster=cluster,
        processing_mode=data_service_ops.ShardingPolicy.OFF,
        target_workers="LOCAL")
    self.assertDatasetProduces(dataset, list(range(20)))

  @combinations.generate(test_base.default_test_combinations())
  def testRangeDataset_OneWorker(self):
    """Makes sure shards from all workers form the complete dataset."""
    cluster = _make_service_cluster(num_workers=1, local_shard_index=0)
    dataset = dataset_ops.Dataset.range(20)
    dataset = self.make_distributed_dataset(
        dataset,
        cluster=cluster,
        processing_mode=data_service_ops.ShardingPolicy.FILE_OR_DATA)
    self.assertDatasetProduces(dataset, list(range(20)))

  @combinations.generate(test_base.default_test_combinations())
  def testRangeDataset_ReadFromAnyWorker(self):
    # When deployment mode is unspecified, the client will read from any worker.
    cluster = _make_service_cluster(
        num_workers=5, local_shard_index=1, deployment_mode=None)
    dataset = dataset_ops.Dataset.range(20)
    dataset = self.make_distributed_dataset(
        dataset,
        cluster=cluster,
        processing_mode=data_service_ops.ShardingPolicy.FILE_OR_DATA)
    self.assertDatasetProduces(
        dataset, list(range(20)), assert_items_equal=True)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(sharding_policy=[
              data_service_ops.ShardingPolicy.FILE_OR_DATA,
              data_service_ops.ShardingPolicy.FILE
          ])))
  def testTFRecordDataset_AutoShard(self, sharding_policy):
    cluster = _make_service_cluster(num_workers=5, local_shard_index=3)
    dataset = dataset_ops.Dataset.list_files(self._filenames, shuffle=False)
    dataset = dataset.flat_map(readers.TFRecordDataset)
    dataset = self.make_distributed_dataset(
        dataset,
        cluster=cluster,
        processing_mode=sharding_policy,
        target_workers="LOCAL")

    expected = [
        b"Record %d of file %d" % (record, file)
        for file in (3, 8)
        for record in range(0, 10)
    ]
    self.assertDatasetProduces(dataset, expected)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(sharding_policy=[
              data_service_ops.ShardingPolicy.FILE_OR_DATA,
              data_service_ops.ShardingPolicy.FILE
          ])))
  def testTFRecordDataset_ShuffleFileList(self, sharding_policy):
    cluster = _make_service_cluster(num_workers=5, local_shard_index=3)
    dataset = dataset_ops.Dataset.list_files(self._filenames, shuffle=True)
    dataset = dataset.flat_map(readers.TFRecordDataset)
    dataset = self.make_distributed_dataset(
        dataset, cluster=cluster, processing_mode=sharding_policy)

    expected = [
        b"Record %d of file %d" % (record, file)
        for file in (3, 8)
        for record in range(0, 10)
    ]
    self.assertDatasetProduces(dataset, expected, assert_items_equal=True)

  @combinations.generate(test_base.default_test_combinations())
  def testTFRecordDataset_DataShard(self):
    cluster = _make_service_cluster(num_workers=5, local_shard_index=3)
    dataset = dataset_ops.Dataset.list_files(self._filenames, shuffle=False)
    dataset = dataset.flat_map(readers.TFRecordDataset)
    dataset = self.make_distributed_dataset(
        dataset,
        cluster=cluster,
        processing_mode=data_service_ops.ShardingPolicy.DATA)

    expected = [
        b"Record %d of file %d" % (record, file)
        for file in range(0, 10)
        for record in (3, 8)
    ]
    self.assertDatasetProduces(dataset, expected)

  @combinations.generate(test_base.default_test_combinations())
  def testTFRecordDataset_HintDataShard(self):
    cluster = _make_service_cluster(num_workers=5, local_shard_index=3)
    dataset = dataset_ops.Dataset.list_files(self._filenames, shuffle=False)
    dataset = dataset.flat_map(readers.TFRecordDataset)
    dataset = dataset.shard(distribute.SHARD_HINT, distribute.SHARD_HINT)
    dataset = self.make_distributed_dataset(
        dataset,
        cluster=cluster,
        processing_mode=data_service_ops.ShardingPolicy.HINT)

    expected = [
        b"Record %d of file %d" % (record, file)
        for file in range(0, 10)
        for record in (3, 8)
    ]
    self.assertDatasetProduces(dataset, expected)

  @combinations.generate(test_base.default_test_combinations())
  def testTFRecordDataset_HintFileShard(self):
    cluster = _make_service_cluster(num_workers=5, local_shard_index=3)
    dataset = dataset_ops.Dataset.list_files(self._filenames, shuffle=False)
    dataset = dataset.shard(distribute.SHARD_HINT, distribute.SHARD_HINT)
    dataset = dataset.flat_map(readers.TFRecordDataset)
    dataset = self.make_distributed_dataset(
        dataset,
        cluster=cluster,
        processing_mode=data_service_ops.ShardingPolicy.HINT)

    expected = [
        b"Record %d of file %d" % (record, file)
        for file in (3, 8)
        for record in range(0, 10)
    ]
    self.assertDatasetProduces(dataset, expected)

  @combinations.generate(test_base.default_test_combinations())
  def testTFRecordDataset_NoShard(self):
    cluster = _make_service_cluster(num_workers=5, local_shard_index=3)
    dataset = dataset_ops.Dataset.list_files(self._filenames, shuffle=False)
    dataset = dataset.flat_map(readers.TFRecordDataset)
    dataset = self.make_distributed_dataset(
        dataset,
        cluster=cluster,
        processing_mode=data_service_ops.ShardingPolicy.OFF,
        target_workers="LOCAL")

    expected = [
        b"Record %d of file %d" % (record, file)
        for file in range(0, 10)
        for record in range(0, 10)
    ]
    self.assertDatasetProduces(dataset, expected)

  @combinations.generate(test_base.default_test_combinations())
  def testTFRecordDataset_ReadFromAnyWorker(self):
    # When deployment mode is unspecified, the client will read from any worker.
    cluster = _make_service_cluster(
        num_workers=5, local_shard_index=3, deployment_mode=None)
    dataset = dataset_ops.Dataset.list_files(self._filenames, shuffle=False)
    dataset = dataset.flat_map(readers.TFRecordDataset)
    dataset = self.make_distributed_dataset(
        dataset,
        cluster=cluster,
        processing_mode=data_service_ops.ShardingPolicy.FILE_OR_DATA)

    expected = [
        b"Record %d of file %d" % (record, file)
        for file in range(0, 10)
        for record in range(0, 10)
    ]
    self.assertDatasetProduces(dataset, expected, assert_items_equal=True)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(sharding_policy=[
              data_service_ops.ShardingPolicy.FILE_OR_DATA,
              data_service_ops.ShardingPolicy.FILE
          ])))
  def testTFRecordDataset_FewerFilesThanWorkers(self, sharding_policy):
    cluster = _make_service_cluster(num_workers=5, local_shard_index=3)
    dataset = dataset_ops.Dataset.list_files(self._filenames[:4], shuffle=False)
    dataset = dataset.flat_map(readers.TFRecordDataset)
    dataset = self.make_distributed_dataset(
        dataset, cluster=cluster, processing_mode=sharding_policy)

    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        "not enough for the required 5 shards/workers."):
      self.getDatasetOutput(dataset)

  @combinations.generate(test_base.default_test_combinations())
  def testTFRecordDataset_FewerFilesThanWorkers_HintShard(self):
    cluster = _make_service_cluster(num_workers=5, local_shard_index=3)
    dataset = dataset_ops.Dataset.list_files(self._filenames[:4], shuffle=False)
    dataset = dataset.shard(distribute.SHARD_HINT, distribute.SHARD_HINT)
    dataset = dataset.flat_map(readers.TFRecordDataset)
    dataset = self.make_distributed_dataset(
        dataset,
        cluster=cluster,
        processing_mode=data_service_ops.ShardingPolicy.HINT)

    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        "not enough for the required 5 shards/workers."):
      self.getDatasetOutput(dataset)

  @combinations.generate(test_base.default_test_combinations())
  def testTFRecordDataset_FewerFilesThanWorkers_DataShard(self):
    cluster = _make_service_cluster(num_workers=5, local_shard_index=3)
    dataset = dataset_ops.Dataset.list_files(self._filenames[:4], shuffle=False)
    dataset = dataset.flat_map(readers.TFRecordDataset)
    dataset = self.make_distributed_dataset(
        dataset,
        cluster=cluster,
        processing_mode=data_service_ops.ShardingPolicy.DATA)

    expected = [
        b"Record %d of file %d" % (record, file)
        for file in range(0, 4)
        for record in (3, 8)
    ]
    self.assertDatasetProduces(dataset, expected, assert_items_equal=True)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(sharding_policy=[
              data_service_ops.ShardingPolicy.FILE_OR_DATA,
              data_service_ops.ShardingPolicy.DATA
          ])))
  def testBatchDataset(self, sharding_policy):
    cluster = _make_service_cluster(num_workers=5, local_shard_index=1)
    dataset = dataset_ops.Dataset.range(20)
    dataset = dataset.batch(batch_size=3, drop_remainder=False)
    dataset = self.make_distributed_dataset(
        dataset, cluster=cluster, processing_mode=sharding_policy)
    self.assertDatasetProduces(dataset, [[3, 4, 5], [18, 19]])

  @combinations.generate(test_base.default_test_combinations())
  def testInterleaveDataset(self):
    cluster = _make_service_cluster(num_workers=5, local_shard_index=3)
    dataset = dataset_ops.Dataset.list_files(self._filenames, shuffle=False)
    dataset = dataset.interleave(
        readers.TFRecordDataset,
        cycle_length=10,
        num_parallel_calls=dataset_ops.AUTOTUNE)
    dataset = dataset.prefetch(buffer_size=dataset_ops.AUTOTUNE)
    dataset = self.make_distributed_dataset(
        dataset,
        cluster=cluster,
        processing_mode=data_service_ops.ShardingPolicy.FILE_OR_DATA)
    dataset = dataset.prefetch(buffer_size=dataset_ops.AUTOTUNE)

    expected = [
        b"Record %d of file %d" % (record, file)
        for record in range(0, 10)
        for file in (3, 8)
    ]
    self.assertDatasetProduces(dataset, expected)

  @combinations.generate(test_base.default_test_combinations())
  def testZipDataset(self):
    cluster = _make_service_cluster(num_workers=5, local_shard_index=3)
    dataset1 = dataset_ops.Dataset.list_files(self._filenames, shuffle=False)
    dataset1 = dataset1.interleave(
        readers.TFRecordDataset,
        cycle_length=10,
        num_parallel_calls=dataset_ops.AUTOTUNE)
    dataset2 = dataset_ops.Dataset.list_files(self._filenames, shuffle=False)
    dataset2 = dataset2.interleave(
        readers.TFRecordDataset,
        cycle_length=10,
        num_parallel_calls=dataset_ops.AUTOTUNE)
    dataset = dataset_ops.Dataset.zip((dataset1, dataset2))
    dataset = dataset.prefetch(buffer_size=dataset_ops.AUTOTUNE)
    dataset = self.make_distributed_dataset(
        dataset,
        cluster=cluster,
        processing_mode=data_service_ops.ShardingPolicy.FILE_OR_DATA)

    expected = [(b"Record %d of file %d" % (record, file),
                 b"Record %d of file %d" % (record, file))
                for record in range(0, 10)
                for file in (3, 8)]
    self.assertDatasetProduces(dataset, expected)

  @combinations.generate(test_base.default_test_combinations())
  def testConcatenateDataset(self):
    cluster = _make_service_cluster(num_workers=5, local_shard_index=3)
    dataset1 = dataset_ops.Dataset.list_files(self._filenames, shuffle=False)
    dataset1 = dataset1.interleave(
        readers.TFRecordDataset,
        cycle_length=10,
        num_parallel_calls=dataset_ops.AUTOTUNE)
    dataset2 = dataset_ops.Dataset.list_files(self._filenames, shuffle=False)
    dataset2 = dataset2.interleave(
        readers.TFRecordDataset,
        cycle_length=10,
        num_parallel_calls=dataset_ops.AUTOTUNE)
    dataset = dataset1.concatenate(dataset2)
    dataset = dataset.prefetch(buffer_size=dataset_ops.AUTOTUNE)
    dataset = self.make_distributed_dataset(
        dataset,
        cluster=cluster,
        processing_mode=data_service_ops.ShardingPolicy.FILE_OR_DATA)

    expected = [
        b"Record %d of file %d" % (record, file)
        for record in range(0, 10)
        for file in (3, 8)
    ]
    expected += expected
    self.assertDatasetProduces(dataset, expected)

  @combinations.generate(test_base.default_test_combinations())
  def testEmptyDataset(self):
    cluster = _make_service_cluster(num_workers=5, local_shard_index=3)
    dataset = dataset_ops.Dataset.range(0)
    dataset = self.make_distributed_dataset(
        dataset,
        cluster=cluster,
        processing_mode=data_service_ops.ShardingPolicy.FILE_OR_DATA)
    self.assertDatasetProduces(dataset, [])

  @combinations.generate(test_base.default_test_combinations())
  def testAnonymousPorts(self):
    cluster = _make_service_cluster(
        num_workers=5,
        local_shard_index=3,
        worker_addresses=["localhost:%port%" for _ in range(5)])
    dataset = dataset_ops.Dataset.range(20)
    dataset = self.make_distributed_dataset(
        dataset,
        cluster=cluster,
        processing_mode=data_service_ops.ShardingPolicy.FILE_OR_DATA)
    self.assertDatasetProduces(dataset, [3, 8, 13, 18])

  @combinations.generate(test_base.default_test_combinations())
  def testNamedPorts(self):
    cluster = _make_service_cluster(
        num_workers=5,
        local_shard_index=3,
        worker_addresses=["localhost:%port_worker%" for _ in range(5)])
    dataset = dataset_ops.Dataset.range(20)
    dataset = self.make_distributed_dataset(
        dataset,
        cluster=cluster,
        processing_mode=data_service_ops.ShardingPolicy.FILE_OR_DATA)
    self.assertDatasetProduces(dataset, [3, 8, 13, 18])

  @combinations.generate(test_base.default_test_combinations())
  def testInvalidPorts(self):
    with self.assertRaisesRegex(RuntimeError,
                                "The worker's address is not configured"):
      _ = _make_service_cluster(
          num_workers=5,
          local_shard_index=0,
          worker_addresses=["localhost:worker" for _ in range(5)])

  @combinations.generate(test_base.default_test_combinations())
  def testEmptyWorkerList(self):
    cluster = _make_service_cluster(
        num_workers=5, local_shard_index=1, worker_addresses=[])
    dataset = dataset_ops.Dataset.range(20)
    dataset = self.make_distributed_dataset(
        dataset,
        cluster=cluster,
        processing_mode=data_service_ops.ShardingPolicy.FILE_OR_DATA)
    with self.assertRaisesRegex(errors.NotFoundError,
                                "Worker .* is not in the workers list."):
      self.getDatasetOutput(dataset)

  @combinations.generate(test_base.default_test_combinations())
  def testWorkerNotFound(self):
    worker_addresses = [f"fake_worker_{i}" for i in range(5)]
    with self.assertRaisesRegex(RuntimeError,
                                "The worker's address is not configured"):
      _ = _make_service_cluster(
          num_workers=5, local_shard_index=0, worker_addresses=worker_addresses)

  @combinations.generate(test_base.default_test_combinations())
  def testMoreWorkersThanConfigured(self):
    worker_addresses = ["localhost:%port%"]
    with self.assertRaisesRegex(
        RuntimeError,
        "other workers are already running at the configured host"):
      _ = _make_service_cluster(
          num_workers=5, local_shard_index=1, worker_addresses=worker_addresses)

  @combinations.generate(test_base.default_test_combinations())
  def testNoLocalWorkers(self):
    cluster = multi_process_cluster.MultiProcessCluster(
        num_local_workers=0, num_remote_workers=3)
    dataset = dataset_ops.Dataset.list_files(self._filenames, shuffle=False)
    dataset = dataset.flat_map(readers.TFRecordDataset)
    dataset = self.make_distributed_dataset(
        dataset,
        cluster=cluster,
        processing_mode=data_service_ops.ShardingPolicy.FILE_OR_DATA)
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        "Static sharding policy <FILE_OR_DATA> requires local tf.data workers"):
      self.getDatasetOutput(dataset)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(
              sharding_policy=list(data_service_ops.ShardingPolicy))))
  def testEnumerateShardingPolicies(self, sharding_policy):
    """Verifies tf.data service handles every sharding policy with no errors."""
    cluster = _make_service_cluster(num_workers=5, local_shard_index=3)
    dataset = dataset_ops.Dataset.list_files(self._filenames, shuffle=False)
    dataset = dataset.flat_map(readers.TFRecordDataset)
    dataset = self.make_distributed_dataset(
        dataset, cluster=cluster, processing_mode=sharding_policy)
    self.getDatasetOutput(dataset)


if __name__ == "__main__":
  multi_process_cluster.test_main()
