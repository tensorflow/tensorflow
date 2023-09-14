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
import tempfile
import time

from absl.testing import parameterized

from tensorflow.python.data.experimental.kernel_tests.service import test_base as data_service_test_base
from tensorflow.python.data.experimental.ops import distributed_save_op
from tensorflow.python.data.experimental.service import _pywrap_snapshot_utils
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import test_mode
from tensorflow.python.framework import combinations
from tensorflow.python.framework import errors
from tensorflow.python.platform import test

# Enum value for `SnapshotStreamInfo` states.
_ORPHAN = 2
_DONE = 4


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


def wait_for_snapshots(paths, f=lambda: None):
  while not all([snapshot_is_done(path) or snapshot_has_error(path)
                 for path in paths]):
    f()
    time.sleep(0.1)


class SnapshotFtTest(data_service_test_base.TestBase, parameterized.TestCase):

  maxDiff = None

  def setUp(self):
    super().setUp()
    self._path = os.path.join(
        tempfile.mkdtemp(dir=self.get_temp_dir()),
        "snapshot_ft_test",
    )
    # TODO(b/268586560): Enable `warm_start` for `snapshot_ft_test`.
    test_mode.toggle_test_mode(False)

  # This "manual" setup function is needed due to some bad interaction between
  # `setUp` and `combinations` that causes the dataset to be out-of-scope.
  # It additionally can't take in a `Dataset` as input.
  def setup(self, num_workers=1, ds_size=10, num_sources=1):
    ds = dataset_ops.Dataset.range(ds_size)
    if num_sources > 1:
      ds = dataset_ops.Dataset.zip((ds,) * num_sources)
    cluster = data_service_test_base.TestCluster(num_workers=num_workers)
    self.evaluate(distributed_save_op.distributed_save(
        ds, self._path, cluster.dispatcher_address()
    ))
    return cluster, ds

  def splits_dir(self, stream_idx=0, worker=0):
    stream_name = f"stream_{stream_idx}"
    self._make_stream_dir(stream_name, worker=worker)
    return os.path.join(
        self._path,
        "streams",
        stream_name,
        "splits",
    )

  def source_dir(self, stream_idx=0, source_idx=0, worker=0):
    return os.path.join(
        self.splits_dir(stream_idx, worker=worker),
        f"source_{source_idx}",
        "repetition_0",
    )

  def _make_stream_dir(self, stream_name, worker=0):
    stream_dir = os.path.join(self._path, "streams", stream_name)
    os.makedirs(stream_dir)
    pathlib.Path(os.path.join(stream_dir, "owner_worker")).write_text(
        f"{worker}"
    )

  @combinations.generate(test_base.default_test_combinations())
  def testSnapshotRecoverySucceeds(self):
    cluster, _ = self.setup()
    cluster.restart_dispatcher()

  @combinations.generate(test_base.default_test_combinations())
  def testSnapshotRecoveryBlocksOverwrite(self):
    cluster, ds = self.setup()
    cluster.restart_dispatcher()
    with self.assertRaisesRegex(
        errors.AlreadyExistsError, "is already started or completed"
    ):
      self.evaluate(distributed_save_op.distributed_save(
          ds, self._path, cluster.dispatcher_address()
      ))

  # TODO(b/250921378): Figure out why tsan times out when there is a worker.
  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(
              bad_stream_dir_name=["stream_", "stream_x", "stream_-1"]
          ),
      )
  )
  def testSnapshotRecoveryFailsWithBadStreamName(self, bad_stream_dir_name):
    cluster, _ = self.setup(num_workers=0)
    self._make_stream_dir(bad_stream_dir_name)
    with self.assertRaisesRegex(ValueError, "Can't parse"):
      cluster.restart_dispatcher()

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(
              bad_source_dir_name=["source_", "source_x", "source_-1"]
          ),
      )
  )
  def testSnapshotRecoveryFailsWithBadSourceName(self, bad_source_dir_name):
    cluster, _ = self.setup(num_workers=0)
    os.makedirs(os.path.join(self.splits_dir(), bad_source_dir_name))
    with self.assertRaisesRegex(ValueError, "Can't parse"):
      cluster.restart_dispatcher()

  @combinations.generate(test_base.default_test_combinations())
  def testSnapshotRecoveryFailsWithOutOfBoundsSourceName(self):
    cluster, _ = self.setup(num_workers=0)
    os.makedirs(os.path.join(self.splits_dir(), "source_1"))
    with self.assertRaisesRegex(ValueError, "Found conflict"):
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
                  "split_0_-1",
              ]
          ),
      )
  )
  def testSnapshotRecoveryFailsWithBadSplitNames(self, bad_split_filename):
    cluster, _ = self.setup(num_workers=0)
    write_file(os.path.join(self.source_dir(), bad_split_filename))
    with self.assertRaisesRegex(
        ValueError, "Expected split_<local_split_index>_<global_split_index>"):
      cluster.restart_dispatcher()

  @combinations.generate(test_base.default_test_combinations())
  def testSnapshotRecoveryFailsWithOutOfOrderSplitName(self):
    cluster, _ = self.setup(num_workers=0)
    write_file(os.path.join(self.source_dir(), "split_1_0"))
    with self.assertRaisesRegex(
        ValueError, "The local split index 1 exceeds the global split index 0"):
      cluster.restart_dispatcher()

  @combinations.generate(test_base.default_test_combinations())
  def testSnapshotRecoveryFailsWithMissingGlobalIndexInSplitNames(self):
    cluster, _ = self.setup(num_workers=0)
    write_file(os.path.join(self.source_dir(), "split_0_1"))
    with self.assertRaisesRegex(ValueError, "Found missing global"):
      cluster.restart_dispatcher()

  @combinations.generate(test_base.default_test_combinations())
  def testSnapshotRecoveryFailsWithDuplicateGlobalIndexInSplitName(self):
    cluster, _ = self.setup(num_workers=0)
    write_file(os.path.join(self.source_dir(stream_idx=0), "split_0_1"))
    write_file(
        os.path.join(self.source_dir(stream_idx=1, worker=1), "split_0_1")
    )
    with self.assertRaisesRegex(ValueError, "Found duplicate global"):
      cluster.restart_dispatcher()

  @combinations.generate(test_base.default_test_combinations())
  def testSnapshotRecoveryFailsWithDuplicateWorkerAssignment(self):
    cluster, _ = self.setup(num_workers=0)
    write_file(os.path.join(self.source_dir(stream_idx=0), "split_0_1"))
    write_file(os.path.join(self.source_dir(stream_idx=1), "split_0_1"))
    with self.assertRaisesRegex(ValueError, "worker is already assigned"):
      cluster.restart_dispatcher()

  @combinations.generate(test_base.default_test_combinations())
  def testStreamsReassignedAfterDispatcherRestart(self):
    n = 5
    cluster, _ = self.setup(num_workers=n, ds_size=10000)
    get_streams = lambda: cluster.snapshot_streams(self._path)
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

    paths = []
    for i in range(num_snapshots):
      paths.append(f"{self._path}_{i}")
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
    wait_for_snapshots(paths, get_assignments_and_update_max_assignments)
    self.assertValuesEqual(list(max_assignments.values()),
                           [worker_max_concurrent_snapshots] * num_workers)

  @combinations.generate(test_base.default_test_combinations())
  def testDatasetRecoversAndCompletes(self):
    cluster = data_service_test_base.TestCluster(num_workers=3)
    ds = dataset_ops.Dataset.range(1000)
    self.evaluate(distributed_save_op.distributed_save(
        ds, self._path, cluster.dispatcher_address(), compression=None))

    # Blocks until all workers have streams.
    get_stream_assignments(cluster, 3, [self._path])
    cluster.stop_worker(0)
    cluster.restart_dispatcher()
    cluster.restart_worker(0)
    self._wait_for_snapshot()
    self.assertTrue(self._snapshot_is_done())

    dataset = dataset_ops.Dataset.load(self._path)
    self.assertDatasetProduces(dataset, range(1000), assert_items_equal=True)

  @combinations.generate(test_base.default_test_combinations())
  def testLargeMultiSourceSnapshotRecoversAndCompletes(self):
    n = 5
    cluster, _ = self.setup(num_workers=n, ds_size=1000, num_sources=3)
    # Blocks until all workers have streams.
    get_stream_assignments(cluster, n, [self._path])
    cluster.stop_worker(0)
    self.assertTrue(
        os.path.exists(
            os.path.join(self._path, "streams", "stream_0", "checkpoints")))

    cluster.restart_dispatcher()
    cluster.restart_worker(0)
    self._wait_for_snapshot()
    self.assertTrue(self._snapshot_is_done())
    # TODO(b/250921378): Verify the number of elements.

  @combinations.generate(test_base.default_test_combinations())
  def testRepeatedDatasetRecoversAndCompletes(self):
    cluster = data_service_test_base.TestCluster(num_workers=3)
    ds = dataset_ops.Dataset.range(100)
    ds = ds.repeat(10)
    self.evaluate(distributed_save_op.distributed_save(
        ds, self._path, cluster.dispatcher_address()))

    # Blocks until all workers have streams.
    get_stream_assignments(cluster, 3, [self._path])
    cluster.stop_worker(0)
    cluster.restart_dispatcher()
    cluster.restart_worker(0)
    self._wait_for_snapshot()
    self.assertTrue(self._snapshot_is_done())

    dataset = dataset_ops.Dataset.load(self._path)
    self.assertDatasetProduces(
        dataset, list(range(100)) * 10, assert_items_equal=True)

  @combinations.generate(test_base.default_test_combinations())
  def testNonrepeatedDatasetDoesntProduceSecondRepetitionDir(self):
    num_workers = 5
    num_sources = 3
    cluster, _ = self.setup(
        num_workers=num_workers,
        ds_size=1000,
        num_sources=num_sources,
    )
    # Blocks until all workers have streams.
    get_stream_assignments(cluster, num_workers, [self._path])
    cluster.stop_worker(0)
    cluster.restart_worker(0)
    self._wait_for_snapshot()
    self.assertTrue(self._snapshot_is_done())
    for stream_idx in range(num_workers):
      for source_idx in range(num_sources):
        self.assertFalse(
            os.path.exists(
                os.path.join(
                    self._path,
                    "streams",
                    f"stream_{stream_idx}",
                    "splits",
                    f"source_{source_idx}",
                    "repetition_1",
                )
            )
        )

  @combinations.generate(test_base.default_test_combinations())
  def testMultipleDatasetRecoversAndCompletes(self):
    cluster = data_service_test_base.TestCluster(num_workers=3)
    dataset1 = dataset_ops.Dataset.range(1000)
    datasets = [
        dataset_ops.Dataset.from_tensors("a").repeat(50),
        dataset_ops.Dataset.from_tensors("b").repeat(50),
        dataset_ops.Dataset.from_tensors("c").repeat(50),
    ]
    choice_dataset = dataset_ops.Dataset.range(3).repeat()
    dataset2 = dataset_ops.Dataset.choose_from_datasets(
        datasets, choice_dataset
    )

    snapshot_path1 = os.path.join(self._path, "snapshot1")
    snapshot_path2 = os.path.join(self._path, "snapshot2")
    self.evaluate(
        distributed_save_op.distributed_save(
            dataset1, snapshot_path1, cluster.dispatcher_address()
        )
    )
    self.evaluate(
        distributed_save_op.distributed_save(
            dataset2, snapshot_path2, cluster.dispatcher_address()
        )
    )

    # Blocks until all workers have streams.
    get_stream_assignments(cluster, 3, [snapshot_path1, snapshot_path2])
    cluster.stop_worker(0)
    cluster.restart_dispatcher()
    cluster.restart_worker(0)
    while not os.path.exists(os.path.join(snapshot_path1, "DONE")):
      time.sleep(0.1)
    while not os.path.exists(os.path.join(snapshot_path2, "DONE")):
      time.sleep(0.1)
    # TODO(b/250921378): Verify the number of elements.

  @combinations.generate(test_base.default_test_combinations())
  def testNestedDataset(self):
    cluster = data_service_test_base.TestCluster(num_workers=1)
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
            dataset, self._path, cluster.dispatcher_address()))
    # Blocks until all workers have streams.
    get_stream_assignments(cluster, 1, [self._path])
    time.sleep(1)
    cluster.stop_worker(0)
    cluster.restart_dispatcher()
    cluster.restart_worker(0)
    self._wait_for_snapshot()
    self.assertTrue(self._snapshot_is_done())

  def _snapshot_is_done(self):
    return snapshot_is_done(self._path)

  def _snapshot_has_error(self):
    return snapshot_has_error(self._path)

  def _wait_for_snapshot(self):
    return wait_for_snapshots([self._path])


if __name__ == "__main__":
  test.main()
