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
from absl.testing import parameterized

from tensorflow.python.data.experimental.kernel_tests import reader_dataset_ops_test_base
from tensorflow.python.data.experimental.ops import snapshot
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import readers as core_readers
from tensorflow.python.framework import test_util
from tensorflow.python.ops import string_ops
from tensorflow.python.platform import test


@test_util.run_all_in_graph_and_eager_modes
class SnapshotDatasetTest(reader_dataset_ops_test_base.TFRecordDatasetTestBase,
                          parameterized.TestCase):

  def setUp(self):
    super(SnapshotDatasetTest, self).setUp()
    self.removeTFRecords()

  def removeTFRecords(self):
    for filename in self.test_filenames:
      os.remove(filename)
    self.test_filenames = []

  def setUpTFRecord(self):
    self._num_files = 10
    self._num_records = 10
    self.test_filenames = self._createFiles()

  def makeSnapshotDirectory(self):
    tmpdir = self.get_temp_dir()
    tmpdir = os.path.join(tmpdir, "snapshot")
    os.mkdir(tmpdir)
    return tmpdir

  def assertSnapshotDirectoryContains(
      self, directory, num_fingerprints, num_runs_per_fp, num_snapshot_files):
    dirlist = os.listdir(directory)
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

  def testWriteDifferentPipelinesInOneDirectory(self):
    tmpdir = self.makeSnapshotDirectory()

    dataset = dataset_ops.Dataset.range(1000)
    dataset = dataset.apply(snapshot.snapshot(tmpdir))
    self.assertDatasetProduces(dataset, list(range(1000)))

    dataset = dataset_ops.Dataset.range(1001)
    dataset = dataset.apply(snapshot.snapshot(tmpdir))
    self.assertDatasetProduces(dataset, list(range(1001)))

    self.assertSnapshotDirectoryContains(tmpdir, 2, 1, 1)

  def testWriteSnapshotMultipleSimultaneous(self):
    tmpdir = self.makeSnapshotDirectory()

    dataset1 = dataset_ops.Dataset.range(1000)
    dataset1 = dataset1.apply(snapshot.snapshot(tmpdir))
    next1 = self.getNext(dataset1)

    dataset2 = dataset_ops.Dataset.range(1000)
    dataset2 = dataset2.apply(snapshot.snapshot(tmpdir))
    next2 = self.getNext(dataset2)

    for _ in range(1000):
      self.evaluate(next1())
      self.evaluate(next2())

    # we check that only one copy of the metadata has been written, and the
    # one that lost the race would be in passthrough mode.
    self.assertSnapshotDirectoryContains(tmpdir, 1, 1, 1)

  @parameterized.parameters(snapshot.COMPRESSION_NONE,
                            snapshot.COMPRESSION_GZIP)
  def testWriteSnapshotSimpleSuccessful(self, compression):
    tmpdir = self.makeSnapshotDirectory()

    dataset = dataset_ops.Dataset.range(1000)
    dataset = dataset.apply(snapshot.snapshot(tmpdir, compression=compression))
    self.assertDatasetProduces(dataset, list(range(1000)))

    self.assertSnapshotDirectoryContains(tmpdir, 1, 1, 1)

  def testWriteSnapshotRepeatAfterwards(self):
    tmpdir = self.makeSnapshotDirectory()

    dataset = dataset_ops.Dataset.range(10)
    dataset = dataset.apply(snapshot.snapshot(tmpdir))
    dataset = dataset.repeat(10)
    self.assertDatasetProduces(dataset, list(range(10)) * 10)

    self.assertSnapshotDirectoryContains(tmpdir, 1, 1, 1)

  @parameterized.parameters(snapshot.COMPRESSION_NONE,
                            snapshot.COMPRESSION_GZIP)
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
