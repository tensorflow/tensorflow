# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for the MapDataset serialization."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl.testing import parameterized

from tensorflow.python.data.experimental.kernel_tests.serialization import dataset_serialization_test_base
from tensorflow.python.data.experimental.ops import snapshot
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import combinations
from tensorflow.python.framework import ops
from tensorflow.python.platform import test


class SnapshotDatasetSerializationTest(
    dataset_serialization_test_base.DatasetSerializationTestBase,
    parameterized.TestCase):

  def _build_snapshot_dataset(self,
                              num_threads=1,
                              repeat=False,
                              pending_snapshot_expiry_seconds=-1,
                              shard_size_bytes=None):

    def ds_fn():
      self.snapshot_dir = os.path.join(self.get_temp_dir(), "snapshot")
      if not os.path.exists(self.snapshot_dir):
        os.mkdir(self.snapshot_dir)
      dataset = dataset_ops.Dataset.range(1000)
      dataset = dataset.apply(
          snapshot.legacy_snapshot(
              self.snapshot_dir,
              num_writer_threads=num_threads,
              writer_buffer_size=2 * num_threads,
              num_reader_threads=num_threads,
              reader_buffer_size=2 * num_threads,
              pending_snapshot_expiry_seconds=pending_snapshot_expiry_seconds,
              shard_size_bytes=shard_size_bytes))
      if repeat:
        dataset = dataset.repeat(2)
      return dataset

    return ds_fn

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(pending_snapshot_expiry_seconds=[None, 1])))
  def testSnapshotBeforeEpochEnd(self, pending_snapshot_expiry_seconds):
    ds_fn = self._build_snapshot_dataset(
        pending_snapshot_expiry_seconds=pending_snapshot_expiry_seconds)
    outputs = self.gen_outputs(ds_fn, [], 100, verify_exhausted=False)
    self.assertSequenceEqual(outputs, range(100))
    outputs.extend(
        self.gen_outputs(
            ds_fn, [], 900, ckpt_saved=True, verify_exhausted=False))
    self.assertSequenceEqual(outputs, range(1000))

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(pending_snapshot_expiry_seconds=[None, 1])))
  def testCheckpointBeforeOneEpochThenRunFewStepsSmallShardMultiThread(
      self, pending_snapshot_expiry_seconds):
    ds_fn = self._build_snapshot_dataset(
        pending_snapshot_expiry_seconds=pending_snapshot_expiry_seconds,
        shard_size_bytes=100)

    outputs = []
    with ops.Graph().as_default() as g:
      init_op, get_next_op, saver = self._build_graph(ds_fn)
      with self.session(graph=g) as sess:
        self._initialize(init_op, sess)
        start = 0
        end = 100
        num_iters = end - start
        for _ in range(num_iters):
          outputs.append(sess.run(get_next_op))
        self._save(sess, saver)
        start = 100
        end = 400
        num_iters = end - start
        for _ in range(num_iters):
          outputs.append(sess.run(get_next_op))
    self.assertSequenceEqual(outputs, range(400))

    outputs = outputs[:100]
    outputs.extend(
        self.gen_outputs(
            ds_fn, [], 900, ckpt_saved=True, verify_exhausted=False))
    self.assertSequenceEqual(outputs, range(1000))
    fp_dir_list = os.listdir(self.snapshot_dir)
    self.assertLen(list(fp_dir_list), 2)
    for d in fp_dir_list:
      if not d.endswith("-graph.pbtxt"):
        fp_dir = os.path.join(self.snapshot_dir, d)
        run_dir_list = os.listdir(fp_dir)
        self.assertLen(list(run_dir_list), 2)
        for e in run_dir_list:
          if e != "snapshot.metadata":
            run_dir = os.path.join(fp_dir, e)
            self.assertLen(list(os.listdir(run_dir)), 258)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(pending_snapshot_expiry_seconds=[None, 1])))
  def testCheckpointBeforeOneEpochThenRunFewSteps(
      self, pending_snapshot_expiry_seconds):
    ds_fn = self._build_snapshot_dataset(
        pending_snapshot_expiry_seconds=pending_snapshot_expiry_seconds)

    # Generate 200 entries from iterator but save checkpoint after producing
    # 100.
    outputs = self.gen_outputs(
        ds_fn, [100], 200, verify_exhausted=False, save_checkpoint_at_end=False)
    self.assertSequenceEqual(outputs, range(200))

    outputs = outputs[:100]
    outputs.extend(
        self.gen_outputs(
            ds_fn, [], 900, ckpt_saved=True, verify_exhausted=False))
    self.assertSequenceEqual(outputs, range(1000))

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(pending_snapshot_expiry_seconds=[None, 1])))
  def testCheckpointBeforeOneEpochThenRunFewStepsMultipleThreads(
      self, pending_snapshot_expiry_seconds):
    ds_fn = self._build_snapshot_dataset(
        num_threads=2,
        pending_snapshot_expiry_seconds=pending_snapshot_expiry_seconds)

    # Generate 200 entries from iterator but save checkpoint after producing
    # 100.
    outputs = self.gen_outputs(
        ds_fn, [100], 200, verify_exhausted=False, save_checkpoint_at_end=False)
    self.assertSequenceEqual(outputs, range(200))

    outputs = outputs[:100]
    outputs.extend(
        self.gen_outputs(
            ds_fn, [], 900, ckpt_saved=True, verify_exhausted=False))
    self.assertSequenceEqual(outputs, range(1000))

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(pending_snapshot_expiry_seconds=[None, 1])))
  def testCheckpointAfterOneEpoch(self, pending_snapshot_expiry_seconds):
    ds_fn = self._build_snapshot_dataset(
        repeat=True,
        pending_snapshot_expiry_seconds=pending_snapshot_expiry_seconds)

    # Generate 1100 entries from iterator and save checkpoint.
    outputs = self.gen_outputs(ds_fn, [], 1100, verify_exhausted=False)
    self.assertSequenceEqual(outputs, list(range(1000)) + list(range(100)))

    # Restore from checkpoint and produce the rest of the elements from the
    # iterator.
    t = self.gen_outputs(
        ds_fn, [], 900, ckpt_saved=True, verify_exhausted=False)
    outputs.extend(t)
    self.assertSequenceEqual(
        outputs,
        list(range(1000)) + list(range(100)) + list(range(900)))

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(pending_snapshot_expiry_seconds=[None, 1])))
  def testCheckpointAfterOneEpochThenRunFewSteps(
      self, pending_snapshot_expiry_seconds):
    ds_fn = self._build_snapshot_dataset(
        repeat=True,
        pending_snapshot_expiry_seconds=pending_snapshot_expiry_seconds)

    # Generate 200 entries from iterator but save checkpoint after producing
    # 100.
    outputs = self.gen_outputs(
        ds_fn, [1100],
        1200,
        verify_exhausted=False,
        save_checkpoint_at_end=False)
    self.assertSequenceEqual(
        outputs,
        list(range(1000)) + list(range(100)) + list(range(100)))

    outputs = outputs[:1100]
    t = self.gen_outputs(
        ds_fn, [], 900, ckpt_saved=True, verify_exhausted=False)
    outputs.extend(t)
    self.assertSequenceEqual(
        outputs, (list(range(1000)) + list(range(100)) + list(range(900))))


if __name__ == "__main__":
  test.main()
