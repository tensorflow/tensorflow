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
"""Tests for the `SnapshotDataset` transformation."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import time

from absl.testing import parameterized

from tensorflow.python.data.experimental.kernel_tests import reader_dataset_ops_test_base
from tensorflow.python.data.experimental.ops import snapshot
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import readers as core_readers
from tensorflow.python.framework import combinations
from tensorflow.python.framework import errors
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.platform import test


class SnapshotDatasetTest(reader_dataset_ops_test_base.TFRecordDatasetTestBase,
                          parameterized.TestCase):

  def setUp(self):
    super(SnapshotDatasetTest, self).setUp()
    self.removeTFRecords()

  def removeTFRecords(self):
    for filename in self.test_filenames:
      os.remove(filename)
    self.test_filenames = []

  def setUpTFRecord(self, num_files=10, num_records=10):
    self._num_files = num_files
    self._num_records = num_records
    self.test_filenames = self._createFiles()

  def makeSnapshotDirectory(self):
    tmpdir = self.get_temp_dir()
    tmpdir = os.path.join(tmpdir, "snapshot")
    os.mkdir(tmpdir)
    return tmpdir

  def assertSnapshotDirectoryContains(
      self, directory, num_fingerprints, num_runs_per_fp, num_snapshot_files):
    dirlist_raw = os.listdir(directory)
    dirlist = []

    # Ignore the graphdef pbtxts we write for debugging purposes.
    for i in range(len(dirlist_raw)):
      if not dirlist_raw[i].endswith("-graph.pbtxt"):
        dirlist.append(dirlist_raw[i])

    self.assertLen(dirlist, num_fingerprints)

    for i in range(num_fingerprints):
      fingerprint_dir = os.path.join(directory, dirlist[i])
      fingerprint_dir_list = sorted(os.listdir(fingerprint_dir))
      self.assertLen(fingerprint_dir_list, num_runs_per_fp + 1)
      self.assertEqual(fingerprint_dir_list[num_runs_per_fp],
                       "snapshot.metadata")

      for j in range(num_runs_per_fp):
        run_dir = os.path.join(fingerprint_dir, fingerprint_dir_list[j])
        run_dirlist = sorted(os.listdir(run_dir))
        self.assertLen(run_dirlist, num_snapshot_files)

        file_counter = 0
        for filename in run_dirlist:
          self.assertEqual(filename, "%08d.snapshot" % file_counter)
          file_counter += 1

  @combinations.generate(test_base.default_test_combinations())
  def testWriteDifferentPipelinesInOneDirectory(self):
    tmpdir = self.makeSnapshotDirectory()

    dataset = dataset_ops.Dataset.range(1000)
    dataset = dataset.apply(snapshot.snapshot(tmpdir))
    self.assertDatasetProduces(dataset, list(range(1000)))

    dataset = dataset_ops.Dataset.range(1001)
    dataset = dataset.apply(snapshot.snapshot(tmpdir))
    self.assertDatasetProduces(dataset, list(range(1001)))

    self.assertSnapshotDirectoryContains(tmpdir, 2, 1, 1)

  @combinations.generate(test_base.default_test_combinations())
  def testWriteSnapshotMultipleSimultaneous(self):
    tmpdir = self.makeSnapshotDirectory()

    dataset1 = dataset_ops.Dataset.range(1000)
    dataset1 = dataset1.apply(snapshot.snapshot(tmpdir))
    next1 = self.getNext(dataset1)

    dataset2 = dataset_ops.Dataset.range(1000)
    dataset2 = dataset2.apply(snapshot.snapshot(tmpdir))
    next2 = self.getNext(dataset2)

    for i in range(0, 1000):
      self.assertEqual(i, self.evaluate(next1()))
      self.assertEqual(i, self.evaluate(next2()))

    # we check that only one copy of the metadata has been written, and the
    # one that lost the race would be in passthrough mode.
    self.assertSnapshotDirectoryContains(tmpdir, 1, 1, 1)

  @combinations.generate(test_base.default_test_combinations())
  def testGetNextCreatesDir(self):
    tmpdir = self.makeSnapshotDirectory()

    # We create two iterators but call getNext on only one.
    dataset1 = dataset_ops.Dataset.range(1000)
    dataset1 = dataset1.apply(snapshot.snapshot(tmpdir))
    next1 = self.getNext(dataset1)

    dataset2 = dataset_ops.Dataset.range(1001)
    dataset2 = dataset2.apply(snapshot.snapshot(tmpdir))
    _ = self.getNext(dataset2)

    for _ in range(1000):
      self.evaluate(next1())

    # We check that only one directory is created.
    self.assertSnapshotDirectoryContains(tmpdir, 1, 1, 1)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(compression=[
              snapshot.COMPRESSION_NONE, snapshot.COMPRESSION_GZIP,
              snapshot.COMPRESSION_SNAPPY
          ])))
  def testWriteSnapshotSimpleSuccessful(self, compression):
    tmpdir = self.makeSnapshotDirectory()

    dataset = dataset_ops.Dataset.range(1000)
    dataset = dataset.apply(snapshot.snapshot(tmpdir, compression=compression))
    self.assertDatasetProduces(dataset, list(range(1000)))

    self.assertSnapshotDirectoryContains(tmpdir, 1, 1, 1)

  @combinations.generate(test_base.default_test_combinations())
  def testWriteSnapshotRepeatAfterwards(self):
    tmpdir = self.makeSnapshotDirectory()

    dataset = dataset_ops.Dataset.range(10)
    dataset = dataset.apply(snapshot.snapshot(tmpdir))
    dataset = dataset.repeat(10)
    self.assertDatasetProduces(dataset, list(range(10)) * 10)

    self.assertSnapshotDirectoryContains(tmpdir, 1, 1, 1)

  @combinations.generate(test_base.default_test_combinations())
  def testSpecifySnapshotNameWriteAndRead(self):
    tmpdir = self.makeSnapshotDirectory()

    dataset = dataset_ops.Dataset.range(10)
    dataset = dataset.apply(
        snapshot.snapshot(tmpdir, snapshot_name="my_custom_snapshot"))
    dataset = dataset.repeat(10)
    self.assertDatasetProduces(dataset, list(range(10)) * 10)

    self.assertSnapshotDirectoryContains(tmpdir, 1, 1, 1)
    self.assertTrue(
        os.path.exists(os.path.join(tmpdir, "custom-my_custom_snapshot")))
    self.assertTrue(
        os.path.exists(
            os.path.join(tmpdir, "custom-my_custom_snapshot", "custom")))

  @combinations.generate(test_base.default_test_combinations())
  def testForcePassthroughMode(self):
    tmpdir = self.makeSnapshotDirectory()

    dataset = dataset_ops.Dataset.range(10)
    dataset = dataset.apply(snapshot.snapshot(tmpdir, mode="passthrough"))
    dataset = dataset.repeat(10)
    self.assertDatasetProduces(dataset, list(range(10)) * 10)

    self.assertSnapshotDirectoryContains(tmpdir, 0, 0, 0)

  @combinations.generate(test_base.default_test_combinations())
  def testForceWriteMode(self):
    tmpdir = self.makeSnapshotDirectory()

    dataset = dataset_ops.Dataset.range(10)
    dataset = dataset.apply(snapshot.snapshot(tmpdir, mode="write"))
    dataset = dataset.repeat(10)
    self.assertDatasetProduces(dataset, list(range(10)) * 10)

    # We will end up writing 10 different runs.
    self.assertSnapshotDirectoryContains(tmpdir, 1, 10, 1)

  @combinations.generate(test_base.default_test_combinations())
  def testForceReadMode(self):
    tmpdir = self.makeSnapshotDirectory()

    # We write a copy of the snapshot first.
    dataset = dataset_ops.Dataset.range(10)
    dataset = dataset.apply(
        snapshot.snapshot(
            tmpdir, mode="write", snapshot_name="my_custom_snapshot"))
    self.assertDatasetProduces(dataset, list(range(10)))

    # We move the run to a new name.
    shutil.move(
        os.path.join(tmpdir, "custom-my_custom_snapshot"),
        os.path.join(tmpdir, "custom-my_custom_snapshot_2"))

    # Even though the snapshot.metadata is pointing to the old run that no
    # longer exists after we moved, we force it to read from the run we specify.
    dataset = dataset_ops.Dataset.range(10)
    dataset = dataset.apply(
        snapshot.snapshot(
            tmpdir, mode="read", snapshot_name="my_custom_snapshot_2"))
    self.assertDatasetProduces(dataset, list(range(10)))

    # We should still have one snapshot and one run.
    self.assertSnapshotDirectoryContains(tmpdir, 1, 1, 1)

  @combinations.generate(test_base.default_test_combinations())
  def testForceReadNonexistentSnapshot(self):
    tmpdir = self.makeSnapshotDirectory()
    dataset = dataset_ops.Dataset.range(10)
    with self.assertRaises(errors.NotFoundError):
      dataset = dataset.apply(snapshot.snapshot(tmpdir, mode="read"))
      get_next = self.getNext(dataset)
      self.evaluate(get_next())

  @combinations.generate(test_base.default_test_combinations())
  def testForceReadNonexistentNamedSnapshot(self):
    tmpdir = self.makeSnapshotDirectory()
    dataset = dataset_ops.Dataset.range(10)
    with self.assertRaises(errors.NotFoundError):
      dataset = dataset.apply(
          snapshot.snapshot(
              tmpdir, mode="read", snapshot_name="my_nonexistent_snapshot"))
      get_next = self.getNext(dataset)
      self.evaluate(get_next())

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(compression=[
              snapshot.COMPRESSION_NONE, snapshot.COMPRESSION_GZIP,
              snapshot.COMPRESSION_SNAPPY
          ])))
  def testReadSnapshotBackAfterWrite(self, compression):
    self.setUpTFRecord()
    filenames = self.test_filenames

    expected = [
        b"Record %d of file %d" % (r, f)  # pylint:disable=g-complex-comprehension
        for f in range(0, 10)
        for r in range(0, 10)
    ]

    tmpdir = self.makeSnapshotDirectory()
    dataset = core_readers._TFRecordDataset(filenames)
    dataset = dataset.apply(snapshot.snapshot(tmpdir, compression=compression))
    self.assertDatasetProduces(dataset, expected)

    # remove the original files and try to read the data back only from snapshot
    self.removeTFRecords()

    dataset2 = core_readers._TFRecordDataset(filenames)
    dataset2 = dataset2.apply(snapshot.snapshot(
        tmpdir, compression=compression))
    self.assertDatasetProduces(dataset2, expected)

  @combinations.generate(test_base.default_test_combinations())
  def testReadShuffledSnapshotAfterWrite(self):
    self.setUpTFRecord(num_files=10, num_records=50)
    filenames = self.test_filenames

    expected = [
        b"Record %d of file %d" % (r, f)  # pylint:disable=g-complex-comprehension
        for f in range(0, 10)
        for r in range(0, 50)
    ]

    tmpdir = self.makeSnapshotDirectory()
    dataset = core_readers._TFRecordDataset(filenames)
    dataset = dataset.apply(snapshot.snapshot(tmpdir, shard_size_bytes=10))
    self.assertDatasetProduces(dataset, expected)

    # remove the original files and try to read the data back only from snapshot
    self.removeTFRecords()

    dataset2 = core_readers._TFRecordDataset(filenames)
    dataset2 = dataset2.apply(snapshot.snapshot(tmpdir, shuffle_on_read=True))
    next2 = self.getNext(dataset2)

    res1 = self.evaluate(next2())
    res2 = self.evaluate(next2())
    res3 = self.evaluate(next2())
    res4 = self.evaluate(next2())
    res5 = self.evaluate(next2())

    # make sure that we don't read the file back in the same order.
    self.assertNotEqual([res1, res2, res3, res4, res5], expected[0:5])

    # make sure all the elements are still there
    dataset3 = core_readers._TFRecordDataset(filenames)
    dataset3 = dataset3.apply(snapshot.snapshot(tmpdir, shuffle_on_read=True))
    self.assertDatasetProduces(dataset3, expected, assert_items_equal=True)

  @combinations.generate(test_base.default_test_combinations())
  def testReadShuffledSnapshotWithSeedAfterWrite(self):
    self.setUpTFRecord(num_files=10, num_records=50)
    filenames = self.test_filenames

    expected = [
        b"Record %d of file %d" % (r, f)  # pylint:disable=g-complex-comprehension
        for f in range(0, 10)
        for r in range(0, 50)
    ]

    tmpdir = self.makeSnapshotDirectory()
    dataset = core_readers._TFRecordDataset(filenames)
    dataset = dataset.apply(snapshot.snapshot(tmpdir, shard_size_bytes=10))
    self.assertDatasetProduces(dataset, expected)

    # remove the original files and try to read the data back only from snapshot
    self.removeTFRecords()

    dataset2 = core_readers._TFRecordDataset(filenames)
    dataset2 = dataset2.apply(
        snapshot.snapshot(tmpdir, shuffle_on_read=True, shuffle_seed=123456))
    next2 = self.getNext(dataset2)

    dataset3 = core_readers._TFRecordDataset(filenames)
    dataset3 = dataset3.apply(
        snapshot.snapshot(tmpdir, shuffle_on_read=True, shuffle_seed=123456))
    next3 = self.getNext(dataset3)

    # make sure that the items are read back in the same order for both datasets
    for _ in range(500):
      res2 = self.evaluate(next2())
      res3 = self.evaluate(next3())
      self.assertEqual(res2, res3)

  @combinations.generate(test_base.default_test_combinations())
  def testReadSnapshotParallelAfterWrite(self):
    self.setUpTFRecord(10, 4000)
    filenames = self.test_filenames

    expected = [
        b"Record %d of file %d" % (r, f)  # pylint:disable=g-complex-comprehension
        for f in range(0, 10)
        for r in range(0, 4000)
    ]

    tmpdir = self.makeSnapshotDirectory()
    dataset = core_readers._TFRecordDataset(filenames)
    dataset = dataset.apply(
        snapshot.snapshot(
            tmpdir,
            shard_size_bytes=1024 * 1024,
            num_reader_threads=2,
            reader_buffer_size=10))
    self.assertDatasetProduces(dataset, expected, assert_items_equal=True)

    # remove the original files and try to read the data back only from
    # snapshot.
    self.removeTFRecords()

    dataset2 = core_readers._TFRecordDataset(filenames)
    dataset2 = dataset2.apply(
        snapshot.snapshot(
            tmpdir,
            shard_size_bytes=1024 * 1024,
            num_reader_threads=2,
            reader_buffer_size=10))
    self.assertDatasetProduces(dataset2, expected, assert_items_equal=True)

  # Not testing Snappy here because Snappy reads currently require a lot of
  # memory.
  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.times(
              combinations.combine(compression=[
                  snapshot.COMPRESSION_NONE, snapshot.COMPRESSION_GZIP
              ]),
              combinations.combine(threads=2, size=[1, 2]) +
              combinations.combine(threads=8, size=[1, 4, 8]))))
  def testReadSnapshotBackAfterMultiThreadedWrite(
      self, compression, threads, size):
    self.setUpTFRecord()
    filenames = self.test_filenames

    expected = [
        b"Record %d of file %d" % (r, f)  # pylint:disable=g-complex-comprehension
        for f in range(0, 10)
        for r in range(0, 10)
    ]

    tmpdir = self.makeSnapshotDirectory()
    dataset = core_readers._TFRecordDataset(filenames)
    dataset = dataset.apply(
        snapshot.snapshot(
            tmpdir,
            compression=compression,
            num_writer_threads=threads,
            writer_buffer_size=size))
    self.assertDatasetProduces(dataset, expected)

    # remove the original files and try to read the data back only from
    # snapshot
    self.removeTFRecords()

    dataset2 = core_readers._TFRecordDataset(filenames)
    dataset2 = dataset2.apply(
        snapshot.snapshot(tmpdir, compression=compression))
    self.assertDatasetProduces(dataset2, expected, assert_items_equal=True)

  @combinations.generate(test_base.default_test_combinations())
  def testSameFingerprintWithDifferentInitializationOrder(self):
    tmpdir = self.makeSnapshotDirectory()

    dataset1 = dataset_ops.Dataset.range(0, 100)
    dataset2 = dataset_ops.Dataset.range(100, 200)
    dataset3 = dataset_ops.Dataset.range(200, 300)

    dataset = dataset1.concatenate(dataset2).concatenate(dataset3)
    dataset = dataset.apply(snapshot.snapshot(tmpdir))
    self.assertDatasetProduces(dataset, list(range(300)))

    dataset4 = dataset_ops.Dataset.range(200, 300)
    dataset5 = dataset_ops.Dataset.range(100, 200)
    dataset6 = dataset_ops.Dataset.range(0, 100)

    dataset = dataset6.concatenate(dataset5).concatenate(dataset4)
    dataset = dataset.apply(snapshot.snapshot(tmpdir))
    self.assertDatasetProduces(dataset, list(range(300)))

    self.assertSnapshotDirectoryContains(tmpdir, 1, 1, 1)

  @combinations.generate(test_base.default_test_combinations())
  def testExpiredSnapshotRewrite(self):
    tmpdir = self.makeSnapshotDirectory()

    dataset1 = dataset_ops.Dataset.range(1000)
    dataset1 = dataset1.apply(
        snapshot.snapshot(tmpdir, pending_snapshot_expiry_seconds=1))
    next1 = self.getNext(dataset1)

    # Don't finish reading dataset1, so it is never finalized
    for _ in range(500):
      self.evaluate(next1())
    self.assertSnapshotDirectoryContains(tmpdir, 1, 1, 1)

    time.sleep(2)

    # Creating dataset2 after we run through dataset1 due to eager mode, where
    # the snapshot state is determined immediately upon dataset creation. We
    # only want to determine the snapshot state for dataset2 after the first
    # snapshot has expired.
    dataset2 = dataset_ops.Dataset.range(1000)
    dataset2 = dataset2.apply(
        snapshot.snapshot(tmpdir, pending_snapshot_expiry_seconds=1))
    next2 = self.getNext(dataset2)

    for _ in range(500):
      self.evaluate(next2())
    self.assertSnapshotDirectoryContains(tmpdir, 1, 2, 1)

  @combinations.generate(test_base.default_test_combinations())
  def testSpecifyShardSize(self):
    tmpdir = self.makeSnapshotDirectory()

    dataset = dataset_ops.Dataset.from_tensor_slices([1.0])
    dataset = dataset.map(lambda x: gen_array_ops.broadcast_to(x, [1024, 1024]))
    dataset = dataset.repeat(10)
    dataset = dataset.apply(
        snapshot.snapshot(tmpdir, shard_size_bytes=10 * 1024 * 1024))
    next_fn = self.getNext(dataset)

    for _ in range(10):
      self.evaluate(next_fn())

    self.assertSnapshotDirectoryContains(tmpdir, 1, 1, 4)

  @combinations.generate(test_base.default_test_combinations())
  def testAdditionalOperationsAfterReadBack(self):
    self.setUpTFRecord()
    filenames = self.test_filenames

    expected = [
        b"Record %d of file %d" % (r, f)  # pylint:disable=g-complex-comprehension
        for f in range(0, 10)
        for r in range(0, 10)
    ]

    tmpdir = self.makeSnapshotDirectory()
    dataset = core_readers._TFRecordDataset(filenames)
    dataset = dataset.apply(snapshot.snapshot(tmpdir))
    self.assertDatasetProduces(dataset, expected)

    # remove the original files and try to read the data back only from snapshot
    self.removeTFRecords()

    dataset2 = core_readers._TFRecordDataset(filenames)
    dataset2 = dataset2.apply(snapshot.snapshot(tmpdir))
    self.assertDatasetProduces(dataset2, expected)

    expected_after = [
        b"cord %d of file %d" % (r, f)  # pylint:disable=g-complex-comprehension
        for f in range(0, 10)
        for r in range(0, 10)
    ]

    dataset3 = core_readers._TFRecordDataset(filenames)
    dataset3 = dataset3.apply(snapshot.snapshot(tmpdir))
    dataset3 = dataset3.map(lambda x: string_ops.substr_v2(x, 2, 1000))
    self.assertDatasetProduces(dataset3, expected_after)


if __name__ == "__main__":
  test.main()
