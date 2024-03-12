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
"""Tests for tf.data.experimental.distributed_save."""

import os
import time

from absl.testing import parameterized

import numpy as np
from tensorflow.python.data.experimental.kernel_tests.service import test_base as data_service_test_base
from tensorflow.python.data.experimental.ops import data_service_ops
from tensorflow.python.data.experimental.ops import distributed_save_op
from tensorflow.python.data.experimental.service import _pywrap_snapshot_utils
from tensorflow.python.data.kernel_tests import checkpoint_test_base
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import combinations
from tensorflow.python.framework import errors
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


class DistributedSaveTest(
    data_service_test_base.TestBase,
    parameterized.TestCase):

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(num_workers=[1, 3], num_elements=[0, 10, 1000])))
  def testSaveLoad(self, num_workers, num_elements):
    cluster = data_service_test_base.TestCluster(num_workers=num_workers)
    snapshot_dir = data_service_test_base.TempDir()
    dataset = dataset_ops.Dataset.range(num_elements)
    self.evaluate(distributed_save_op.distributed_save(
        dataset, snapshot_dir.full_path, cluster.dispatcher_address()))
    _wait_for_snapshot(snapshot_dir.full_path)

    dataset = dataset_ops.Dataset.load(snapshot_dir.full_path)
    self.assertDatasetProduces(
        dataset, list(range(num_elements)), assert_items_equal=True)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(compression=[None, "AUTO", "GZIP"])))
  def testCompression(self, compression):
    cluster = data_service_test_base.TestCluster(num_workers=1)
    snapshot_dir = data_service_test_base.TempDir()
    dataset = dataset_ops.Dataset.range(10)
    self.evaluate(distributed_save_op.distributed_save(
        dataset,
        snapshot_dir.full_path,
        cluster.dispatcher_address(),
        compression=compression))
    _wait_for_snapshot(snapshot_dir.full_path)

    dataset = dataset_ops.Dataset.load(snapshot_dir.full_path)
    self.assertDatasetProduces(
        dataset, list(range(10)), assert_items_equal=True)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(num_workers=[1, 3], num_repetitions=[1, 5])))
  def testRepeatedDataset(self, num_workers, num_repetitions):
    cluster = data_service_test_base.TestCluster(num_workers=num_workers)
    snapshot_dir = data_service_test_base.TempDir()
    dataset = dataset_ops.Dataset.range(1000)
    dataset = dataset.repeat(num_repetitions)
    self.evaluate(distributed_save_op.distributed_save(
        dataset,
        snapshot_dir.full_path,
        cluster.dispatcher_address()))
    _wait_for_snapshot(snapshot_dir.full_path)

    dataset = dataset_ops.Dataset.load(snapshot_dir.full_path)
    self.assertDatasetProduces(
        dataset, list(range(1000)) * num_repetitions, assert_items_equal=True)

  @combinations.generate(test_base.default_test_combinations())
  def testChooseFromDatasets(self):
    cluster = data_service_test_base.TestCluster(num_workers=1)
    snapshot_dir = data_service_test_base.TempDir()
    datasets = [
        dataset_ops.Dataset.from_tensor_slices(["a", "a", "a", "a", "a"]),
        dataset_ops.Dataset.from_tensor_slices(["b", "b", "b", "b", "b"]),
        dataset_ops.Dataset.from_tensor_slices(["c", "c", "c", "c", "c"])]
    choice_dataset = dataset_ops.Dataset.range(3).repeat()
    dataset = dataset_ops.Dataset.choose_from_datasets(datasets, choice_dataset)
    self.evaluate(distributed_save_op.distributed_save(
        dataset, snapshot_dir.full_path, cluster.dispatcher_address()))

    dataset = dataset_ops.Dataset.load(snapshot_dir.full_path)
    self.assertDatasetProduces(
        dataset, [b"a", b"b", b"c"] * 5, assert_items_equal=True)

  @combinations.generate(test_base.default_test_combinations())
  def testChooseFromRepeatedDatasets(self):
    cluster = data_service_test_base.TestCluster(num_workers=1)
    snapshot_dir = data_service_test_base.TempDir()
    datasets = [
        dataset_ops.Dataset.from_tensors("a").repeat(5),
        dataset_ops.Dataset.from_tensors("b").repeat(5),
        dataset_ops.Dataset.from_tensors("c").repeat(10)]
    choice_dataset = dataset_ops.Dataset.range(3).repeat()
    dataset = dataset_ops.Dataset.choose_from_datasets(
        datasets, choice_dataset, stop_on_empty_dataset=False)
    self.evaluate(distributed_save_op.distributed_save(
        dataset, snapshot_dir.full_path, cluster.dispatcher_address()))
    _wait_for_snapshot(snapshot_dir.full_path)

    dataset = dataset_ops.Dataset.load(snapshot_dir.full_path)
    self.assertDatasetProduces(
        dataset, [b"a", b"b", b"c"] * 5 + [b"c"] * 5, assert_items_equal=True)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(num_workers=[1, 3])))
  def testWriteMultipleDatasets(self, num_workers):
    cluster = data_service_test_base.TestCluster(num_workers=num_workers)
    snapshot_dir = data_service_test_base.TempDir()
    dataset1 = dataset_ops.Dataset.range(100)
    datasets = [
        dataset_ops.Dataset.from_tensors("a").repeat(5),
        dataset_ops.Dataset.from_tensors("b").repeat(5),
        dataset_ops.Dataset.from_tensors("c").repeat(5)]
    choice_dataset = dataset_ops.Dataset.range(3).repeat()
    dataset2 = dataset_ops.Dataset.choose_from_datasets(
        datasets, choice_dataset)

    snapshot_path1 = os.path.join(snapshot_dir.full_path, "snapshot1")
    snapshot_path2 = os.path.join(snapshot_dir.full_path, "snapshot2")
    self.evaluate(
        distributed_save_op.distributed_save(
            dataset1, snapshot_path1, cluster.dispatcher_address()))
    self.evaluate(
        distributed_save_op.distributed_save(
            dataset2, snapshot_path2, cluster.dispatcher_address()))
    _wait_for_snapshot(snapshot_path1)
    _wait_for_snapshot(snapshot_path2)

    dataset1 = dataset_ops.Dataset.load(snapshot_path1)
    self.assertDatasetProduces(
        dataset1, list(range(100)), assert_items_equal=True)
    self.assertDatasetProduces(
        dataset2, [b"a", b"b", b"c"] * 5, assert_items_equal=True)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(
              num_workers=[1, 3], snapshot_max_chunk_size_bytes=[1, 100])))
  def testLoadWithCustomReaderFunc(
      self, num_workers, snapshot_max_chunk_size_bytes):
    cluster = data_service_test_base.TestCluster(
        num_workers=num_workers,
        snapshot_max_chunk_size_bytes=snapshot_max_chunk_size_bytes)
    snapshot_dir = data_service_test_base.TempDir()
    dataset = dataset_ops.Dataset.range(10)
    self.evaluate(
        distributed_save_op.distributed_save(
            dataset, snapshot_dir.full_path, cluster.dispatcher_address()))
    _wait_for_snapshot(snapshot_dir.full_path)

    def custom_reader_func(datasets):
      datasets = datasets.shuffle(3)
      return datasets.interleave(
          lambda x: x, num_parallel_calls=dataset_ops.AUTOTUNE)

    dataset = dataset_ops.Dataset.load(
        snapshot_dir.full_path, reader_func=custom_reader_func)
    self.assertDatasetProduces(
        dataset, list(range(10)), assert_items_equal=True)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(
              num_workers=[1, 3],
              repeated_load=[1, 5],
              sharding_policy=[
                  data_service_ops.ShardingPolicy.OFF,
                  data_service_ops.ShardingPolicy.DYNAMIC])))
  def testDistributedLoad(self, num_workers, repeated_load, sharding_policy):
    cluster = data_service_test_base.TestCluster(num_workers=num_workers)
    snapshot_dir = data_service_test_base.TempDir()
    dataset = dataset_ops.Dataset.range(10)
    self.evaluate(
        distributed_save_op.distributed_save(
            dataset, snapshot_dir.full_path, cluster.dispatcher_address()))
    _wait_for_snapshot(snapshot_dir.full_path)

    dataset = dataset_ops.Dataset.load(snapshot_dir.full_path)
    if repeated_load > 1:
      dataset = dataset.repeat(repeated_load)
    dataset = dataset.apply(
        data_service_ops.distribute(
            processing_mode=sharding_policy,
            service=cluster.dispatcher_address()))

    expected = list(range(10)) * repeated_load
    if sharding_policy == data_service_ops.ShardingPolicy.OFF:
      expected *= num_workers
    self.assertDatasetProduces(dataset, expected, assert_items_equal=True)

  @combinations.generate(test_base.default_test_combinations())
  def testImbalancedZipAndRepeat(self):
    smaller_num_elements = 200
    larger_num_elements = 1000
    repetitions = 3

    cluster = data_service_test_base.TestCluster(num_workers=1)
    snapshot_dir = data_service_test_base.TempDir()
    dataset1 = dataset_ops.Dataset.range(smaller_num_elements)
    dataset2 = dataset_ops.Dataset.range(larger_num_elements)
    dataset = dataset_ops.Dataset.zip((dataset1, dataset2))
    dataset = dataset.repeat(repetitions)
    self.evaluate(
        distributed_save_op.distributed_save(
            dataset, snapshot_dir.full_path, cluster.dispatcher_address()))
    _wait_for_snapshot(snapshot_dir.full_path)

    dataset = dataset_ops.Dataset.load(snapshot_dir.full_path)
    expected = repetitions * (
        list(zip(range(smaller_num_elements), range(smaller_num_elements))))
    self.assertDatasetProduces(dataset, expected, assert_items_equal=True)

  @combinations.generate(test_base.default_test_combinations())
  def testSnapshotDoesNotExist(self):
    cluster = data_service_test_base.TestCluster(num_workers=1)
    snapshot_dir = data_service_test_base.TempDir()
    with self.assertRaises(errors.NotFoundError):
      dataset = dataset_ops.Dataset.load(snapshot_dir.full_path)
      dataset = dataset.apply(
          data_service_ops.distribute(
              data_service_ops.ShardingPolicy.OFF,
              cluster.dispatcher_address()))
      self.getDatasetOutput(dataset)

  @combinations.generate(test_base.default_test_combinations())
  def testDuplicateSnapshot(self):
    cluster = data_service_test_base.TestCluster(num_workers=1)
    snapshot_dir = data_service_test_base.TempDir()
    dataset = dataset_ops.Dataset.range(10)
    with self.assertRaisesRegex(
        errors.AlreadyExistsError, "already started or completed"):
      self.evaluate(
          distributed_save_op.distributed_save(
              dataset, snapshot_dir.full_path, cluster.dispatcher_address()))
      self.evaluate(
          distributed_save_op.distributed_save(
              dataset, snapshot_dir.full_path, cluster.dispatcher_address()))

  @combinations.generate(test_base.default_test_combinations())
  def testWorkerFailure(self):
    cluster = data_service_test_base.TestCluster(num_workers=1)
    snapshot_dir = data_service_test_base.TempDir()
    components = np.array([1.0, 2.0, 3.0, np.nan, 5.0]).astype(np.float32)
    dataset = dataset_ops.Dataset.from_tensor_slices(components)
    dataset = dataset.map(lambda x: array_ops.check_numerics(x, "message"))
    self.evaluate(
        distributed_save_op.distributed_save(
            dataset, snapshot_dir.full_path, cluster.dispatcher_address()))
    _wait_for_error(snapshot_dir.full_path)

    with self.assertRaisesRegex(
        ValueError, "The save job failed to write it."):
      dataset = dataset_ops.Dataset.load(snapshot_dir.full_path)
      self.getDatasetOutput(dataset)

  @combinations.generate(test_base.default_test_combinations())
  def testBadDispatcherAddress(self):
    dataset = dataset_ops.Dataset.range(10)
    with self.assertRaisesRegex(ValueError, "must be a string"):
      self.evaluate(distributed_save_op.distributed_save(dataset, "", 1))
    with self.assertRaisesRegex(ValueError, "must not be empty"):
      self.evaluate(distributed_save_op.distributed_save(dataset, "", ""))

  @combinations.generate(test_base.default_test_combinations())
  def testBadCardinality(self):
    cluster = data_service_test_base.TestCluster(num_workers=1)
    snapshot_dir = data_service_test_base.TempDir()
    dataset = dataset_ops.Dataset.range(10).repeat()
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        "Saving an infinite dataset is not allowed"):
      self.evaluate(distributed_save_op.distributed_save(
          dataset, snapshot_dir.full_path, cluster.dispatcher_address()))

  @combinations.generate(test_base.default_test_combinations())
  def testBadElementSpec(self):
    cluster = data_service_test_base.TestCluster(num_workers=1)
    snapshot_dir = data_service_test_base.TempDir()
    dataset = dataset_ops.Dataset.range(10)
    self.evaluate(distributed_save_op.distributed_save(
        dataset, snapshot_dir.full_path,
        cluster.dispatcher_address(),
        compression="AUTO"))
    _wait_for_snapshot(snapshot_dir.full_path)

    with self.assertRaisesRegex(
        ValueError,
        "User specified element_spec bad_element_spec, but the actual "
        "element_spec is TensorSpec"):
      _ = dataset_ops.Dataset.load(snapshot_dir.full_path,
                                   element_spec="bad_element_spec")

  @combinations.generate(test_base.default_test_combinations())
  def testBadCompression(self):
    cluster = data_service_test_base.TestCluster(num_workers=1)
    snapshot_dir = data_service_test_base.TempDir()
    dataset = dataset_ops.Dataset.range(10)
    self.evaluate(distributed_save_op.distributed_save(
        dataset, snapshot_dir.full_path,
        cluster.dispatcher_address(),
        compression="AUTO"))
    _wait_for_snapshot(snapshot_dir.full_path)

    with self.assertRaisesRegex(
        ValueError,
        "User specified compression ZLIB, but the actual compression is "
        "SNAPPY."):
      _ = dataset_ops.Dataset.load(snapshot_dir.full_path, compression="ZLIB")

  @combinations.generate(test_base.default_test_combinations())
  def testRequiresFaultTolerantMode(self):
    cluster = data_service_test_base.TestCluster(
        num_workers=1, fault_tolerant_mode=False)
    snapshot_dir = data_service_test_base.TempDir()
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        "tf.data distributed snapshot requires running tf.data service in the "
        "fault tolerant mode."):
      self.evaluate(distributed_save_op.distributed_save(
          dataset_ops.Dataset.range(10), snapshot_dir.full_path,
          cluster.dispatcher_address(),
          compression="AUTO"))


class LoadCheckpointTest(
    data_service_test_base.TestBase,
    checkpoint_test_base.CheckpointTestBase,
    parameterized.TestCase):

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          checkpoint_test_base.default_test_combinations(),
      )
  )
  def testLoadCheckpoint(self, verify_fn):
    cluster = data_service_test_base.TestCluster(num_workers=1)
    snapshot_dir = data_service_test_base.TempDir()
    dataset = dataset_ops.Dataset.range(10)
    self.evaluate(
        distributed_save_op.distributed_save(
            dataset, snapshot_dir.full_path, cluster.dispatcher_address()))
    _wait_for_snapshot(snapshot_dir.full_path)

    def _build_ds():
      return dataset_ops.Dataset.load(snapshot_dir.full_path)

    verify_fn(self, _build_ds, num_outputs=10, assert_items_equal=True)


def _wait_for_snapshot(snapshot_path):
  while not os.path.exists(
      _pywrap_snapshot_utils.TF_DATA_SnapshotDoneFilePath(snapshot_path)):
    time.sleep(0.1)


def _wait_for_error(snapshot_path):
  while not os.path.exists(
      _pywrap_snapshot_utils.TF_DATA_SnapshotErrorFilePath(snapshot_path)):
    time.sleep(0.1)


if __name__ == "__main__":
  test.main()
