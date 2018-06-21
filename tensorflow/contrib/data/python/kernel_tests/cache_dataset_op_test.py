# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for the experimental features of CacheDataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensorflow.contrib.data.python.kernel_tests import dataset_serialization_test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import errors
from tensorflow.python.platform import test


class CacheToFileDatasetSerializationTest(
    dataset_serialization_test_base.DatasetSerializationTestBase):

  def setUp(self):
    self.range_size = 10
    self.num_repeats = 3
    self.num_outputs = self.range_size * self.num_repeats
    self.cache_file_prefix = 'test'

  def ds_fn(self):
    return dataset_ops.Dataset.range(self.range_size).cache(
        os.path.join(self.get_temp_dir(),
                     self.cache_file_prefix)).repeat(self.num_repeats)

  def expected_outputs(self):
    return list(range(self.range_size)) * self.num_repeats

  def testCheckpointBeforeOneEpoch(self):
    # Generate 5 entries from iterator and save checkpoint.
    outputs = self.gen_outputs(self.ds_fn, [], 5, verify_exhausted=False)
    self.assertSequenceEqual(outputs, range(5))

    # Restore from checkpoint and produce the rest of the elements from the
    # iterator.
    outputs.extend(
        self.gen_outputs(
            self.ds_fn, [],
            self.num_outputs - 5,
            ckpt_saved=True,
            verify_exhausted=False))
    self.assertSequenceEqual(outputs, self.expected_outputs())

  def testCheckpointBeforeOneEpochThenRunFewSteps(self):
    # Generate 8 entries from iterator but save checkpoint after producing
    # 5.
    outputs = self.gen_outputs(
        self.ds_fn, [5],
        8,
        verify_exhausted=False,
        save_checkpoint_at_end=False)
    self.assertSequenceEqual(outputs, range(8))

    # Restoring from checkpoint and running GetNext should return a
    # `AlreadExistsError` now because the lockfile already exists.
    with self.assertRaises(errors.AlreadyExistsError):
      self.gen_outputs(
          self.ds_fn, [],
          self.num_outputs - 5,
          ckpt_saved=True,
          verify_exhausted=False)

  def testCheckpointAfterOneEpoch(self):
    # Generate 15 entries from iterator and save checkpoint.
    outputs = self.gen_outputs(self.ds_fn, [], 15, verify_exhausted=False)
    self.assertSequenceEqual(outputs, list(range(10)) + list(range(5)))

    # Restore from checkpoint and produce the rest of the elements from the
    # iterator.
    outputs.extend(
        self.gen_outputs(
            self.ds_fn, [],
            self.num_outputs - 15,
            ckpt_saved=True,
            verify_exhausted=False))
    self.assertSequenceEqual(outputs, self.expected_outputs())

  def testCheckpointAfterOneEpochThenRunFewSteps(self):
    # Generate 18 entries from iterator but save checkpoint after producing
    # 15.
    outputs = self.gen_outputs(
        self.ds_fn, [15],
        18,
        verify_exhausted=False,
        save_checkpoint_at_end=False)
    self.assertSequenceEqual(outputs, list(range(10)) + list(range(8)))

    outputs = list(range(10)) + list(range(5)) + self.gen_outputs(
        self.ds_fn, [],
        self.num_outputs - 15,
        ckpt_saved=True,
        verify_exhausted=False)
    self.assertSequenceEqual(outputs, list(range(10)) * 3)

  def testCheckpointBeforeOneEpochButRunCompleteEpoch(self):
    # Generate 13 entries from iterator but save checkpoint after producing
    # 5.
    outputs = self.gen_outputs(
        self.ds_fn, [5],
        13,
        verify_exhausted=False,
        save_checkpoint_at_end=False)
    self.assertSequenceEqual(outputs, list(range(10)) + list(range(3)))

    # Since we ran for more than one epoch, the cache was completely written.
    # The ckpt was saved when the iterator was in cache-write mode. Test that
    # the iterator falls back to read mode after restoring if the cache has
    # been completely written.

    outputs = list(range(5)) + self.gen_outputs(
        self.ds_fn, [],
        self.num_outputs - 5,
        ckpt_saved=True,
        verify_exhausted=False)
    self.assertSequenceEqual(outputs, list(range(10)) * 3)

  def testCheckpointUnusedWriterIterator(self):
    # Checkpoint before get_next is called even once.
    outputs = self.gen_outputs(self.ds_fn, [], 0, verify_exhausted=False)
    self.assertSequenceEqual(outputs, [])

    outputs = self.gen_outputs(
        self.ds_fn, [],
        self.num_outputs,
        ckpt_saved=True,
        verify_exhausted=False)
    self.assertSequenceEqual(outputs, list(range(10)) * 3)

  def testCheckpointUnusedMidwayWriterIterator(self):
    # Produce 5 elements and checkpoint.
    outputs = self.gen_outputs(self.ds_fn, [], 5, verify_exhausted=False)
    self.assertSequenceEqual(outputs, range(5))

    # Restore from checkpoint, then produce no elements and checkpoint.
    outputs.extend(
        self.gen_outputs(
            self.ds_fn, [], 0, ckpt_saved=True, verify_exhausted=False))
    self.assertSequenceEqual(outputs, range(5))

    # Restore from checkpoint and produce rest of the elements.
    outputs.extend(
        self.gen_outputs(
            self.ds_fn, [],
            self.num_outputs - 5,
            ckpt_saved=True,
            verify_exhausted=False))
    self.assertSequenceEqual(outputs, list(range(10)) * 3)

  def testUnusedCheckpointError(self):
    # Produce 5 elements and save ckpt.
    outputs = self.gen_outputs(self.ds_fn, [], 5, verify_exhausted=False)
    self.assertSequenceEqual(outputs, range(5))

    # Since the complete cache has not been written, a new iterator which does
    # not restore the checkpoint will throw an error since there is a partial
    # cache shard.
    with self.assertRaises(errors.AlreadyExistsError):
      outputs = self.gen_outputs(
          self.ds_fn, [], self.num_outputs, verify_exhausted=False)

  def testIgnoreCheckpointIfCacheWritten(self):
    # Produce 15 elements and save ckpt. This will write the complete cache.
    outputs = self.gen_outputs(self.ds_fn, [], 15, verify_exhausted=False)
    self.assertSequenceEqual(outputs, list(range(10)) + list(range(5)))

    # Build the iterator again but do not restore from ckpt. Since the cache
    # has already been written we should be able to use it.
    outputs = self.gen_outputs(
        self.ds_fn, [], self.num_outputs, verify_exhausted=False)
    self.assertSequenceEqual(outputs, list(range(10)) * 3)


if __name__ == '__main__':
  test.main()
