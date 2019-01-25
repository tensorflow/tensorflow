# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for `tf.data.FixedLengthRecordDataset`."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import zlib

from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import readers
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test
from tensorflow.python.util import compat


@test_util.run_all_in_graph_and_eager_modes
class FixedLengthRecordDatasetTest(test_base.DatasetTestBase):

  def setUp(self):
    super(FixedLengthRecordDatasetTest, self).setUp()
    self._num_files = 2
    self._num_records = 7
    self._header_bytes = 5
    self._record_bytes = 3
    self._footer_bytes = 2

  def _record(self, f, r):
    return compat.as_bytes(str(f * 2 + r) * self._record_bytes)

  def _createFiles(self, compression_type=None):
    filenames = []
    for i in range(self._num_files):
      fn = os.path.join(self.get_temp_dir(), "fixed_length_record.%d.txt" % i)
      filenames.append(fn)

      contents = []
      contents.append(b"H" * self._header_bytes)
      for j in range(self._num_records):
        contents.append(self._record(i, j))
      contents.append(b"F" * self._footer_bytes)
      contents = b"".join(contents)

      if not compression_type:
        with open(fn, "wb") as f:
          f.write(contents)
      elif compression_type == "GZIP":
        with gzip.GzipFile(fn, "wb") as f:
          f.write(contents)
      elif compression_type == "ZLIB":
        contents = zlib.compress(contents)
        with open(fn, "wb") as f:
          f.write(contents)
      else:
        raise ValueError("Unsupported compression_type", compression_type)

    return filenames

  def _testFixedLengthRecordDataset(self, compression_type=None):
    test_filenames = self._createFiles(compression_type=compression_type)

    def dataset_fn(filenames, num_epochs, batch_size=None):
      repeat_dataset = readers.FixedLengthRecordDataset(
          filenames,
          self._record_bytes,
          self._header_bytes,
          self._footer_bytes,
          compression_type=compression_type).repeat(num_epochs)
      if batch_size:
        return repeat_dataset.batch(batch_size)
      return repeat_dataset

    # Basic test: read from file 0.
    self.assertDatasetProduces(
        dataset_fn([test_filenames[0]], 1),
        expected_output=[
            self._record(0, i) for i in range(self._num_records)
        ])

    # Basic test: read from file 1.
    self.assertDatasetProduces(
        dataset_fn([test_filenames[1]], 1),
        expected_output=[
            self._record(1, i) for i in range(self._num_records)
        ])

    # Basic test: read from both files.
    expected_output = []
    for j in range(self._num_files):
      expected_output.extend(
          [self._record(j, i) for i in range(self._num_records)])
    self.assertDatasetProduces(
        dataset_fn(test_filenames, 1), expected_output=expected_output)

    # Test repeated iteration through both files.
    get_next = self.getNext(dataset_fn(test_filenames, 10))
    for _ in range(10):
      for j in range(self._num_files):
        for i in range(self._num_records):
          self.assertEqual(self._record(j, i), self.evaluate(get_next()))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())

    # Test batched and repeated iteration through both files.
    get_next = self.getNext(dataset_fn(test_filenames, 10, self._num_records))
    for _ in range(10):
      for j in range(self._num_files):
        self.assertAllEqual(
            [self._record(j, i) for i in range(self._num_records)],
            self.evaluate(get_next()))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())

  def testFixedLengthRecordDatasetNoCompression(self):
    self._testFixedLengthRecordDataset()

  def testFixedLengthRecordDatasetGzipCompression(self):
    self._testFixedLengthRecordDataset(compression_type="GZIP")

  def testFixedLengthRecordDatasetZlibCompression(self):
    self._testFixedLengthRecordDataset(compression_type="ZLIB")

  def testFixedLengthRecordDatasetBuffering(self):
    test_filenames = self._createFiles()
    dataset = readers.FixedLengthRecordDataset(
        test_filenames,
        self._record_bytes,
        self._header_bytes,
        self._footer_bytes,
        buffer_size=10)
    expected_output = []
    for j in range(self._num_files):
      expected_output.extend(
          [self._record(j, i) for i in range(self._num_records)])
    self.assertDatasetProduces(dataset, expected_output=expected_output)

  def testFixedLengthRecordDatasetWrongSize(self):
    test_filenames = self._createFiles()
    dataset = readers.FixedLengthRecordDataset(
        test_filenames,
        self._record_bytes + 1,  # Incorrect record length.
        self._header_bytes,
        self._footer_bytes,
        buffer_size=10)
    self.assertDatasetProduces(
        dataset,
        expected_error=(
            errors.InvalidArgumentError,
            r"Excluding the header \(5 bytes\) and footer \(2 bytes\), input "
            r"file \".*fixed_length_record.0.txt\" has body length 21 bytes, "
            r"which is not an exact multiple of the record length \(4 bytes\).")
        )


if __name__ == "__main__":
  test.main()
