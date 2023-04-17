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
import os
import pathlib
import tempfile
import time

from absl.testing import parameterized

from tensorflow.python.data.experimental.kernel_tests.service import test_base as data_service_test_base
from tensorflow.python.data.experimental.ops import distributed_save_op
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import test_mode
from tensorflow.python.framework import combinations
from tensorflow.python.platform import test

# Enum value for `SnapshotStreamInfo` states.
_ORPHAN = 2
_DONE = 4


def write_file(path):
  os.makedirs(os.path.dirname(path), exist_ok=True)
  with open(path, "w") as _:
    pass


def get_stream_assignment(cluster, worker_idx, snapshot_idx=0):
  while not cluster.workers[worker_idx].snapshot_task_progresses():
    time.sleep(0.1)
  return (
      cluster.workers[worker_idx]
      .snapshot_task_progresses()[snapshot_idx]
      .snapshot_task_stream_index
  )


def get_stream_assignments(cluster, n, snapshot_idx=0):
  assignments = {}
  for i in range(n):
    assignments[i] = get_stream_assignment(cluster, i, snapshot_idx)
  return assignments


class SnapshotFtTest(data_service_test_base.TestBase, parameterized.TestCase):

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
    distributed_save_op.distributed_save(
        ds, self._path, cluster.dispatcher_address()
    )
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
    )

  def _make_stream_dir(self, stream_name, worker=0):
    stream_dir = os.path.join(self._path, "streams", stream_name)
    os.makedirs(stream_dir)
    pathlib.Path(os.path.join(stream_dir, "owner_worker")).write_text(
        f"{worker}"
    )

  @combinations.generate(test_base.eager_only_combinations())
  def testSnapshotRecoverySucceeds(self):
    cluster, _ = self.setup()
    cluster.restart_dispatcher()

  @combinations.generate(test_base.eager_only_combinations())
  def testSnapshotRecoveryBlocksOverwrite(self):
    cluster, ds = self.setup()
    cluster.restart_dispatcher()
    with self.assertRaisesOpError("is already started or completed"):
      distributed_save_op.distributed_save(
          ds, self._path, cluster.dispatcher_address()
      )

  # TODO(b/250921378): Figure out why tsan times out when there is a worker.
  @combinations.generate(
      combinations.times(
          test_base.eager_only_combinations(),
          combinations.combine(
              bad_stream_dir_name=["stream_", "stream_x", "stream_-1"]
          ),
      )
  )
  def testSnapshotRecoveryFailsWithBadStreamName(self, bad_stream_dir_name):
    cluster, _ = self.setup(num_workers=0)
    self._make_stream_dir(bad_stream_dir_name)
    with self.assertRaisesRegex(ValueError, "can't parse"):
      cluster.restart_dispatcher()

  @combinations.generate(
      combinations.times(
          test_base.eager_only_combinations(),
          combinations.combine(
              bad_source_dir_name=["source_", "source_x", "source_-1"]
          ),
      )
  )
  def testSnapshotRecoveryFailsWithBadSourceName(self, bad_source_dir_name):
    cluster, _ = self.setup(num_workers=0)
    os.makedirs(os.path.join(self.splits_dir(), bad_source_dir_name))
    with self.assertRaisesRegex(ValueError, "can't parse"):
      cluster.restart_dispatcher()

  @combinations.generate(test_base.eager_only_combinations())
  def testSnapshotRecoveryFailsWithOutOfBoundsSourceName(self):
    cluster, _ = self.setup(num_workers=0)
    os.makedirs(os.path.join(self.splits_dir(), "source_1"))
    with self.assertRaisesRegex(ValueError, "found conflict"):
      cluster.restart_dispatcher()

  @combinations.generate(
      combinations.times(
          test_base.eager_only_combinations(),
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

  @combinations.generate(test_base.eager_only_combinations())
  def testSnapshotRecoveryFailsWithOutOfOrderSplitName(self):
    cluster, _ = self.setup(num_workers=0)
    write_file(os.path.join(self.source_dir(), "split_1_0"))
    with self.assertRaisesRegex(
        ValueError, "The local split index 1 exceeds the global split index 0"):
      cluster.restart_dispatcher()

  @combinations.generate(test_base.eager_only_combinations())
  def testSnapshotRecoveryFailsWithOutOfBoundsSplitName(self):
    cluster, _ = self.setup(num_workers=0)
    write_file(os.path.join(self.source_dir(), "split_1_1"))
    with self.assertRaisesRegex(ValueError, "found conflict"):
      cluster.restart_dispatcher()

  @combinations.generate(test_base.eager_only_combinations())
  def testSnapshotRecoveryFailsWithMissingGlobalIndexInSplitNames(self):
    cluster, _ = self.setup(num_workers=0)
    write_file(os.path.join(self.source_dir(), "split_0_1"))
    with self.assertRaisesRegex(ValueError, "found missing global"):
      cluster.restart_dispatcher()

  @combinations.generate(test_base.eager_only_combinations())
  def testSnapshotRecoveryFailsWithDuplicateGlobalIndexInSplitName(self):
    cluster, _ = self.setup(num_workers=0)
    write_file(os.path.join(self.source_dir(stream_idx=0), "split_0_1"))
    write_file(
        os.path.join(self.source_dir(stream_idx=1, worker=1), "split_0_1")
    )
    with self.assertRaisesRegex(ValueError, "found duplicate global"):
      cluster.restart_dispatcher()

  @combinations.generate(test_base.eager_only_combinations())
  def testSnapshotRecoveryFailsWithDuplicateWorkerAssignment(self):
    cluster, _ = self.setup(num_workers=0)
    write_file(os.path.join(self.source_dir(stream_idx=0), "split_0_1"))
    write_file(os.path.join(self.source_dir(stream_idx=1), "split_0_1"))
    with self.assertRaisesRegex(ValueError, "worker is already assigned"):
      cluster.restart_dispatcher()

  @combinations.generate(test_base.eager_only_combinations())
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

  @combinations.generate(test_base.eager_only_combinations())
  def testLargeMultiSourceSnapshotRecoversAndCompletes(self):
    n = 5
    cluster, _ = self.setup(num_workers=n, ds_size=1000, num_sources=3)
    get_stream_assignments(cluster, n)  # Block until all workers have streams.
    cluster.stop_worker(0)
    cluster.restart_dispatcher()
    cluster.restart_worker(0)
    self._wait_for_snapshot()
    self.assertTrue(self._snapshot_is_done())

  def _snapshot_is_done(self):
    return os.path.exists(os.path.join(self._path, "DONE"))

  def _snapshot_has_error(self):
    return os.path.exists(os.path.join(self._path, "ERROR"))

  def _wait_for_snapshot(self):
    while not (self._snapshot_is_done() or self._snapshot_has_error()):
      time.sleep(0.1)


if __name__ == "__main__":
  test.main()
