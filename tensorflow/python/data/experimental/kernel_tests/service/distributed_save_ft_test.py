# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
"""Fault tolerance tests for tf.data service snapshots."""

import collections
import os
import pathlib
import time

from absl.testing import parameterized

from tensorflow.python.data.experimental.kernel_tests.service import test_base as data_service_test_base
from tensorflow.python.data.experimental.ops import distributed_save_op
from tensorflow.python.data.experimental.service import _pywrap_snapshot_utils
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import combinations
from tensorflow.python.framework import errors
from tensorflow.python.platform import test


def write_file(path):
  os.makedirs(os.path.dirname(path), exist_ok=True)
  with open(path, "w") as _:
    pass


def get_stream_assignment(
    cluster,
    worker_idx,
    path,
    block=True,
    active_only=False):
  while True:
    for progress in cluster.workers[worker_idx].snapshot_task_progresses():
      if (progress.snapshot_task_base_path.decode() == path
          and not (active_only and progress.completed)):
        return progress.snapshot_task_stream_index
    if not block:
      break
    time.sleep(0.1)


def get_stream_assignments(
    cluster,
    num_workers,
    paths,
    block=True,
    active_only=False):
  assignments = collections.defaultdict(dict)
  for worker_idx in range(num_workers):
    for path in paths:
      assignment = get_stream_assignment(
          cluster, worker_idx, path, block, active_only)
      if assignment is not None:
        assignments[worker_idx][path] = assignment
  return assignments


def snapshot_is_done(path):
  return os.path.exists(
      _pywrap_snapshot_utils.TF_DATA_SnapshotDoneFilePath(path))


def snapshot_has_error(path):
  return os.path.exists(
      _pywrap_snapshot_utils.TF_DATA_SnapshotErrorFilePath(path))


def snapshots_are_done(paths):
  return all([snapshot_is_done(path) for path in paths])


def wait_for_snapshot(paths, f=lambda: None):
  if isinstance(paths, str):
    paths = [paths]
  while not all([snapshot_is_done(path) or snapshot_has_error(path)
                 for path in paths]):
    f()
    time.sleep(0.1)


class SnapshotFtTest(data_service_test_base.TestBase, parameterized.TestCase):

  maxDiff = None

  @combinations.generate(test_base.default_test_combinations())
  def testSnapshotRecoverySucceeds(self):
    cluster = data_service_test_base.TestCluster(num_workers=1)
    snapshot_dir = data_service_test_base.TempDir()
    dataset = self._get_dataset()
    self.evaluate(distributed_save_op.distributed_save(
        dataset, snapshot_dir.full_path, cluster.dispatcher_address()))
    cluster.restart_dispatcher()

  @combinations.generate(test_base.default_test_combinations())
  def testSnapshotRecoveryBlocksOverwrite(self):
    cluster = data_service_test_base.TestCluster(num_workers=1)
    snapshot_dir = data_service_test_base.TempDir()
    dataset = self._get_dataset()
    self.evaluate(distributed_save_op.distributed_save(
        dataset, snapshot_dir.full_path, cluster.dispatcher_address()))

    cluster.restart_dispatcher()
    with self.assertRaisesRegex(
        errors.AlreadyExistsError, "is already started or completed"):
      self.evaluate(distributed_save_op.distributed_save(
          dataset, snapshot_dir.full_path, cluster.dispatcher_address()))

  # TODO(b/250921378): Figure out why tsan times out when there is a worker.
  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(
              bad_stream_dir_name=["stream_", "stream_x", "stream_-1"])))
  def testSnapshotRecoveryFailsWithBadStreamName(self, bad_stream_dir_name):
    cluster = data_service_test_base.TestCluster(num_workers=0)
    snapshot_dir = data_service_test_base.TempDir()
    self.evaluate(distributed_save_op.distributed_save(
        self._get_dataset(),
        snapshot_dir.full_path,
        cluster.dispatcher_address()))

    self._make_stream_dir(snapshot_dir.full_path, bad_stream_dir_name)
    with self.assertRaisesRegex(RuntimeError, "Can't parse"):
      cluster.restart_dispatcher()

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(
              bad_source_dir_name=["source_", "source_x", "source_-1"])))
  def testSnapshotRecoveryFailsWithBadSourceName(self, bad_source_dir_name):
    cluster = data_service_test_base.TestCluster(num_workers=0)
    snapshot_dir = data_service_test_base.TempDir()
    self.evaluate(distributed_save_op.distributed_save(
        self._get_dataset(),
        snapshot_dir.full_path,
        cluster.dispatcher_address()))

    os.makedirs(os.path.join(self._splits_dir(snapshot_dir.full_path),
                             bad_source_dir_name))
    with self.assertRaisesRegex(RuntimeError, "Can't parse"):
      cluster.restart_dispatcher()

  @combinations.generate(test_base.default_test_combinations())
  def testSnapshotRecoveryFailsWithOutOfBoundsSourceName(self):
    cluster = data_service_test_base.TestCluster(num_workers=0)
    snapshot_dir = data_service_test_base.TempDir()
    self.evaluate(distributed_save_op.distributed_save(
        self._get_dataset(),
        snapshot_dir.full_path,
        cluster.dispatcher_address()))

    os.makedirs(os.path.join(self._splits_dir(snapshot_dir.full_path),
                             "source_1"))
    with self.assertRaisesRegex(RuntimeError, "Found conflict"):
      cluster.restart_dispatcher()

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(
              bad_split_filename=[
                  "split_",
                  "split_x_0",
                  "split_-1_0",
                  "split_0_x",
                  "split_0_-1"])))
  def testSnapshotRecoveryFailsWithBadSplitNames(self, bad_split_filename):
    cluster = data_service_test_base.TestCluster(num_workers=0)
    snapshot_dir = data_service_test_base.TempDir()
    self.evaluate(distributed_save_op.distributed_save(
        self._get_dataset(),
        snapshot_dir.full_path,
        cluster.dispatcher_address()))

    write_file(os.path.join(self._source_dir(snapshot_dir.full_path),
                            bad_split_filename))
    with self.assertRaisesRegex(
        ValueError,
        "Expected split_<local_split_index>_<global_split_index>"):
      cluster.restart_dispatcher()

  @combinations.generate(test_base.default_test_combinations())
  def testSnapshotRecoveryFailsWithOutOfOrderSplitName(self):
    cluster = data_service_test_base.TestCluster(num_workers=0)
    snapshot_dir = data_service_test_base.TempDir()
    self.evaluate(distributed_save_op.distributed_save(
        self._get_dataset(),
        snapshot_dir.full_path,
        cluster.dispatcher_address()))

    write_file(os.path.join(self._source_dir(snapshot_dir.full_path),
                            "split_1_0"))
    with self.assertRaisesRegex(
        ValueError,
        "The local split index 1 exceeds the global split index 0"):
      cluster.restart_dispatcher()

  @combinations.generate(test_base.default_test_combinations())
  def testSnapshotRecoveryFailsWithMissingGlobalIndexInSplitNames(self):
    cluster = data_service_test_base.TestCluster(num_workers=0)
    snapshot_dir = data_service_test_base.TempDir()
    self.evaluate(distributed_save_op.distributed_save(
        self._get_dataset(),
        snapshot_dir.full_path,
        cluster.dispatcher_address()))

    write_file(os.path.join(self._source_dir(snapshot_dir.full_path),
                            "split_0_1"))
    with self.assertRaisesRegex(RuntimeError, "Found missing global"):
      cluster.restart_dispatcher()

  @combinations.generate(test_base.default_test_combinations())
  def testSnapshotRecoveryFailsWithDuplicateGlobalIndexInSplitName(self):
    cluster = data_service_test_base.TestCluster(num_workers=0)
    snapshot_dir = data_service_test_base.TempDir()
    self.evaluate(distributed_save_op.distributed_save(
        self._get_dataset(),
        snapshot_dir.full_path,
        cluster.dispatcher_address()))

    write_file(os.path.join(self._source_dir(
        snapshot_dir.full_path, stream_idx=0), "split_0_1"))
    write_file(os.path.join(self._source_dir(
        snapshot_dir.full_path, stream_idx=1, worker=1), "split_0_1"))
    with self.assertRaisesRegex(RuntimeError, "Found duplicate global"):
      cluster.restart_dispatcher()

  @combinations.generate(test_base.default_test_combinations())
  def testSnapshotRecoveryFailsWithDuplicateWorkerAssignment(self):
    cluster = data_service_test_base.TestCluster(num_workers=0)
    snapshot_dir = data_service_test_base.TempDir()
    self.evaluate(distributed_save_op.distributed_save(
        self._get_dataset(),
        snapshot_dir.full_path,
        cluster.dispatcher_address()))

    write_file(os.path.join(
        self._source_dir(snapshot_dir.full_path, stream_idx=0), "split_0_1"))
    write_file(os.path.join(
        self._source_dir(snapshot_dir.full_path, stream_idx=1), "split_0_1"))
    with self.assertRaisesRegex(RuntimeError, "worker is already assigned"):
      cluster.restart_dispatcher()

  @combinations.generate(test_base.default_test_combinations())
  def testStreamsReassignedAfterDispatcherRestart(self):
    n = 5
    cluster = data_service_test_base.TestCluster(num_workers=n)
    snapshot_dir = data_service_test_base.TempDir()
    dataset = self._get_dataset(dataset_range=10000)
    self.evaluate(distributed_save_op.distributed_save(
        dataset, snapshot_dir.full_path, cluster.dispatcher_address()))

    get_streams = lambda: cluster.snapshot_streams(snapshot_dir.full_path)
    while len(get_streams()) != n:
      time.sleep(0.1)
    cluster.restart_dispatcher()
    streams = get_streams()
    while len(streams) != n:
      time.sleep(0.1)
      streams = get_streams()
    self.assertCountEqual([stream.index for stream in streams], range(n))

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(
              worker_max_concurrent_snapshots=[1, 2])))
  def testWorkersDontExceedMaxStreamAssignments(
      self, worker_max_concurrent_snapshots):
    num_workers = 2
    num_snapshots = 10
    cluster = data_service_test_base.TestCluster(
        num_workers=num_workers,
        worker_max_concurrent_snapshots=worker_max_concurrent_snapshots)
    snapshot_dir = data_service_test_base.TempDir()
    paths = []
    for i in range(num_snapshots):
      paths.append(f"{snapshot_dir.full_path}_{i}")
      self.evaluate(
          distributed_save_op.distributed_save(
              dataset_ops.Dataset.range(5000),
              paths[i],
              cluster.dispatcher_address()))

    # A mapping of worker idx to max active assignments observed at any time.
    max_assignments = collections.defaultdict(int)

    def get_assignments_and_update_max_assignments():
      assignments = get_stream_assignments(
          cluster, num_workers, paths, block=False, active_only=True)
      for worker_idx, worker_assignments in assignments.items():
        max_assignments[worker_idx] = max(max_assignments[worker_idx],
                                          len(worker_assignments))
      return assignments

    # Blocks until each worker has at least the max expected active assignments.
    while True:
      assignments = get_assignments_and_update_max_assignments()
      all_workers_have_assignments = len(assignments) == num_workers
      each_worker_has_enough_assignments = all([
          len(per_worker_assignments) >= worker_max_concurrent_snapshots
          for per_worker_assignments in assignments.values()])
      if all_workers_have_assignments and each_worker_has_enough_assignments:
        break
      time.sleep(0.1)

    cluster.restart_dispatcher()
    wait_for_snapshot(paths, get_assignments_and_update_max_assignments)
    self.assertValuesEqual(list(max_assignments.values()),
                           [worker_max_concurrent_snapshots] * num_workers)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(num_workers=[1, 3])))
  def testDatasetRecoversAndCompletes(self, num_workers):
    cluster = data_service_test_base.TestCluster(num_workers=num_workers)
    snapshot_dir = data_service_test_base.TempDir()
    dataset = dataset_ops.Dataset.range(1000)
    self.evaluate(
        distributed_save_op.distributed_save(
            dataset,
            snapshot_dir.full_path,
            cluster.dispatcher_address(),
            compression=None))

    for i in range(num_workers):
      cluster.stop_worker(i)
      cluster.restart_dispatcher()
      cluster.restart_worker(i)
    wait_for_snapshot(snapshot_dir.full_path)
    self.assertTrue(snapshot_is_done(snapshot_dir.full_path))

    dataset = dataset_ops.Dataset.load(snapshot_dir.full_path)
    self.assertDatasetProduces(dataset, range(1000), assert_items_equal=True)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(num_workers=[1, 3], num_sources=[1, 3])))
  def testLargeMultiSourceSnapshotRecoversAndCompletes(
      self, num_workers, num_sources):
    cluster = data_service_test_base.TestCluster(num_workers=num_workers)
    snapshot_dir = data_service_test_base.TempDir()
    dataset = self._get_dataset(dataset_range=1000, num_sources=num_sources)
    self.evaluate(distributed_save_op.distributed_save(
        dataset, snapshot_dir.full_path, cluster.dispatcher_address()))

    for i in range(num_workers):
      cluster.stop_worker(i)
      cluster.restart_dispatcher()
      cluster.restart_worker(i)
    wait_for_snapshot(snapshot_dir.full_path)
    self.assertTrue(snapshot_is_done(snapshot_dir.full_path))

    if num_workers > 1:
      # Can't verify the elements with more than 1 worker since the splits
      # receive by each worker for each source is non-deterministic.
      # For example, with dynamic sharding, worker 1 may receive 999 splits for
      # source 0 and 0 splis for source 1; worker 2 may receive 0 splits for
      # source 0 and 999 splits for source 1. In this case, the snapshot will
      # contain no elements since `zip` drops the elements when the zipped
      # datasets have different cardinalities.
      return

    dataset = dataset_ops.Dataset.load(snapshot_dir.full_path)
    expected = (
        list(range(1000))
        if num_sources == 1
        else list(zip(*([range(1000)] * num_sources))))
    self.assertDatasetProduces(dataset, expected, assert_items_equal=True)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(
              num_elements=[1, 2, 1000],
              num_workers=[1, 3],
              num_repetitions=[1, 10])))
  def testRepeatedDatasetRecoversAndCompletes(
      self, num_elements, num_workers, num_repetitions):
    cluster = data_service_test_base.TestCluster(num_workers=num_workers)
    snapshot_dir = data_service_test_base.TempDir()
    ds = dataset_ops.Dataset.range(num_elements)
    ds = ds.repeat(num_repetitions)
    self.evaluate(distributed_save_op.distributed_save(
        ds, snapshot_dir.full_path, cluster.dispatcher_address()))

    for i in range(num_workers):
      cluster.stop_worker(i)
      cluster.restart_dispatcher()
      cluster.restart_worker(i)
    wait_for_snapshot(snapshot_dir.full_path)
    self.assertTrue(snapshot_is_done(snapshot_dir.full_path))

    dataset = dataset_ops.Dataset.load(snapshot_dir.full_path)
    self.assertDatasetProduces(
        dataset,
        list(range(num_elements)) * num_repetitions,
        assert_items_equal=True)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(num_workers=[1, 5], num_sources=[1, 3])))
  def testNonrepeatedDatasetDoesntProduceSecondRepetitionDir(
      self, num_workers, num_sources):
    cluster = data_service_test_base.TestCluster(num_workers=num_workers)
    snapshot_dir = data_service_test_base.TempDir()
    dataset = self._get_dataset(dataset_range=1000, num_sources=num_sources)
    self.evaluate(distributed_save_op.distributed_save(
        dataset, snapshot_dir.full_path, cluster.dispatcher_address()))

    for i in range(num_workers):
      cluster.stop_worker(i)
      cluster.restart_dispatcher()
      cluster.restart_worker(i)
    wait_for_snapshot(snapshot_dir.full_path)
    self.assertTrue(snapshot_is_done(snapshot_dir.full_path))
    for stream_idx in range(num_workers):
      for source_idx in range(num_sources):
        self.assertFalse(
            os.path.exists(
                os.path.join(
                    snapshot_dir.full_path,
                    "streams",
                    f"stream_{stream_idx}",
                    "splits",
                    f"source_{source_idx}",
                    "repetition_1")))

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(num_workers=[1, 3])))
  def testMultipleDatasetRecoversAndCompletes(self, num_workers):
    cluster = data_service_test_base.TestCluster(num_workers=num_workers)
    snapshot_dir = data_service_test_base.TempDir()
    dataset1 = dataset_ops.Dataset.range(1000)
    datasets = [
        dataset_ops.Dataset.from_tensors("a").repeat(50),
        dataset_ops.Dataset.from_tensors("b").repeat(50),
        dataset_ops.Dataset.from_tensors("c").repeat(50)]
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

    for i in range(num_workers):
      cluster.stop_worker(i)
      cluster.restart_dispatcher()
      cluster.restart_worker(i)
    wait_for_snapshot(snapshot_path1)
    wait_for_snapshot(snapshot_path2)

    dataset = dataset_ops.Dataset.load(snapshot_path1)
    self.assertDatasetProduces(
        dataset, list(range(1000)), assert_items_equal=True)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(num_workers=[1, 3])))
  def testNestedDataset(self, num_workers):
    cluster = data_service_test_base.TestCluster(num_workers=num_workers)
    snapshot_dir = data_service_test_base.TempDir()
    dataset = dataset_ops.Dataset.from_tensor_slices(range(100))
    def interleave_fn(x):
      ds = dataset_ops.Dataset.from_tensor_slices(range(x))
      def flat_map_fn(y):
        return dataset_ops.Dataset.from_tensor_slices([y])
      return ds.flat_map(flat_map_fn)
    dataset = dataset.interleave(
        interleave_fn, cycle_length=2, num_parallel_calls=2)
    self.evaluate(
        distributed_save_op.distributed_save(
            dataset, snapshot_dir.full_path, cluster.dispatcher_address()))

    for i in range(num_workers):
      cluster.stop_worker(i)
      cluster.restart_dispatcher()
      cluster.restart_worker(i)
    wait_for_snapshot(snapshot_dir.full_path)
    self.assertTrue(snapshot_is_done(snapshot_dir.full_path))

  def _get_dataset(self, dataset_range=10, num_sources=1):
    dataset = dataset_ops.Dataset.range(dataset_range)
    if num_sources > 1:
      dataset = dataset_ops.Dataset.zip((dataset,) * num_sources)
    return dataset

  def _splits_dir(self, snapshot_path, stream_idx=0, worker=0):
    stream_name = f"stream_{stream_idx}"
    self._make_stream_dir(snapshot_path, stream_name, worker=worker)
    return os.path.join(snapshot_path, "streams", stream_name, "splits")

  def _source_dir(self, snapshot_path, stream_idx=0, source_idx=0, worker=0):
    return os.path.join(
        self._splits_dir(snapshot_path, stream_idx, worker=worker),
        f"source_{source_idx}",
        "repetition_0")

  def _make_stream_dir(self, snapshot_path, stream_name, worker=0):
    stream_dir = os.path.join(snapshot_path, "streams", stream_name)
    os.makedirs(stream_dir)
    pathlib.Path(os.path.join(stream_dir, "owner_worker")).write_text(
        f"{worker}")


if __name__ == "__main__":
  test.main()
