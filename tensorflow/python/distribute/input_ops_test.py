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
"""Tests for input pipeline modifications for distribution strategies."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import readers
from tensorflow.python.distribute import input_ops
from tensorflow.python.framework import errors
from tensorflow.python.lib.io import python_io
from tensorflow.python.platform import test
from tensorflow.python.util import compat


class AutoShardDatasetTest(test.TestCase):

  def setUp(self):
    super(AutoShardDatasetTest, self).setUp()
    self._num_files = 10
    self._num_records = 4
    self._num_shards = 2
    self._shard_index = 0
    self._record_bytes = 10

  def _record(self, r, f):
    return compat.as_bytes("Record %d of file %d" % (r, f))

  def _text_line(self, r, f):
    return compat.as_bytes("Text line %d of file %d" % (r, f))

  def _fixed_length_record(self, r, f):
    return compat.as_bytes(str((r * f) % 10) * self._record_bytes)

  def _createTFRecordFiles(self):
    filenames = []
    for i in range(self._num_files):
      fn = os.path.join(self.get_temp_dir(), "tf_record.%d.txt" % i)
      filenames.append(fn)
      writer = python_io.TFRecordWriter(fn)
      for j in range(self._num_records):
        record = self._record(j, i)
        writer.write(record)
      writer.close()
    return filenames

  def _createTextFiles(self):
    filenames = []
    for i in range(self._num_files):
      fn = os.path.join(self.get_temp_dir(), "text_line.%d.txt" % i)
      filenames.append(fn)
      contents = []
      for j in range(self._num_records):
        contents.append(self._text_line(j, i))
        if j + 1 != self._num_records or i == 0:
          contents.append(b"\r\n")
      contents = b"".join(contents)

      with open(fn, "wb") as f:
        f.write(contents)
    return filenames

  def _createFixedLengthRecordFiles(self):
    filenames = []
    for i in range(self._num_files):
      fn = os.path.join(self.get_temp_dir(), "fixed_length_record.%d.txt" % i)
      filenames.append(fn)
      with open(fn, "wb") as f:
        for j in range(self._num_records):
          f.write(self._fixed_length_record(j, i))
    return filenames

  def _verifySimpleShardingOutput(self, dataset, record_fn):
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
    with self.cached_session() as sess:
      for f in range(self._shard_index, self._num_files, self._num_shards):
        for r in range(self._num_records):
          self.assertAllEqual(record_fn(r, f), self.evaluate(next_element))
      with self.assertRaises(errors.OutOfRangeError):
        self.evaluate(next_element)

  def testTFRecordDataset(self):
    dataset = readers.TFRecordDataset(self._createTFRecordFiles())
    dataset = input_ops.auto_shard_dataset(
        dataset, self._num_shards, self._shard_index)

    self._verifySimpleShardingOutput(dataset, self._record)

  def testFlatMap(self):
    dataset = dataset_ops.Dataset.from_tensor_slices(
        self._createTFRecordFiles())
    dataset = dataset.flat_map(readers.TFRecordDataset)
    dataset = input_ops.auto_shard_dataset(
        dataset, self._num_shards, self._shard_index)

    self._verifySimpleShardingOutput(dataset, self._record)

  def testInterleave(self):
    dataset = dataset_ops.Dataset.from_tensor_slices(
        self._createTFRecordFiles())
    dataset = dataset.interleave(
        readers.TFRecordDataset, cycle_length=4, block_length=self._num_records)
    dataset = input_ops.auto_shard_dataset(
        dataset, self._num_shards, self._shard_index)

    # Since block_length == num records in each file, the output will still
    # contain records in order of files.
    self._verifySimpleShardingOutput(dataset, self._record)

  def testListfiles(self):
    filenames = self._createTFRecordFiles()
    file_pattern = filenames[0].rsplit(os.sep, 1)[0] + "/tf_record.*.txt"
    dataset = dataset_ops.Dataset.list_files(file_pattern, shuffle=False)
    dataset = dataset.flat_map(readers.TFRecordDataset)
    dataset = input_ops.auto_shard_dataset(
        dataset, self._num_shards, self._shard_index)

    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
    with self.cached_session() as sess:
      actual, expected = [], []
      for f in range(self._shard_index, self._num_files, self._num_shards):
        for r in range(self._num_records):
          actual.append(self.evaluate(next_element))
          expected.append(self._record(r, f))
      with self.assertRaises(errors.OutOfRangeError):
        self.evaluate(next_element)
      self.assertAllEqual(expected, actual)

  def testComplexPipeline(self):
    # Setup a complex input pipeline.
    batch_size = 2
    num_epochs = 5
    dataset = dataset_ops.Dataset.from_tensor_slices(
        self._createTFRecordFiles())
    dataset = dataset.shuffle(buffer_size=self._num_files)
    dataset = dataset.flat_map(readers.TFRecordDataset)
    dataset = dataset.prefetch(buffer_size=batch_size)
    dataset = dataset.shuffle(2 * self._num_files * self._num_records)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.map(lambda x: x)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=None)

    # Auto shard.
    dataset = input_ops.auto_shard_dataset(
        dataset, self._num_shards, self._shard_index)

    # Verify output.
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
    with self.cached_session() as sess:
      actual = []
      num_iterations = (self._num_files * self._num_records * num_epochs) // (
          self._num_shards * batch_size)
      for _ in range(num_iterations):
        actual.extend(self.evaluate(next_element))
      with self.assertRaises(errors.OutOfRangeError):
        self.evaluate(next_element)

      expected = []
      for f in range(0, self._num_files, self._num_shards):
        for r in range(self._num_records):
          expected.append(self._record(r, f))
      expected *= num_epochs

      self.assertAllEqual(sorted(expected), sorted(actual))

  def testZip(self):
    dataset1 = readers.TFRecordDataset(self._createTFRecordFiles())
    dataset2 = readers.TextLineDataset(self._createTextFiles())
    dataset = dataset_ops.Dataset.zip((dataset1, dataset2))
    dataset = input_ops.auto_shard_dataset(
        dataset, self._num_shards, self._shard_index)

    record_fn = lambda r, f: (self._record(r, f), self._text_line(r, f))
    self._verifySimpleShardingOutput(dataset, record_fn)

  def testConcat(self):
    dataset1 = readers.TFRecordDataset(self._createTFRecordFiles())
    dataset2 = readers.TextLineDataset(self._createTextFiles())
    dataset = dataset1.concatenate(dataset2)
    dataset = input_ops.auto_shard_dataset(
        dataset, self._num_shards, self._shard_index)

    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
    with self.cached_session() as sess:
      for f in range(self._shard_index, self._num_files, self._num_shards):
        for r in range(self._num_records):
          self.assertAllEqual(self._record(r, f), self.evaluate(next_element))
      for f in range(self._shard_index, self._num_files, self._num_shards):
        for r in range(self._num_records):
          self.assertAllEqual(
              self._text_line(r, f), self.evaluate(next_element))
      with self.assertRaises(errors.OutOfRangeError):
        self.evaluate(next_element)

  def testTextLineReader(self):
    dataset = readers.TextLineDataset(self._createTextFiles())
    dataset = input_ops.auto_shard_dataset(
        dataset, self._num_shards, self._shard_index)

    self._verifySimpleShardingOutput(dataset, self._text_line)

  def testTextLineReaderWithFlatMap(self):
    dataset = dataset_ops.Dataset.from_tensor_slices(self._createTextFiles())
    dataset = dataset.flat_map(readers.TextLineDataset)
    dataset = input_ops.auto_shard_dataset(
        dataset, self._num_shards, self._shard_index)

    self._verifySimpleShardingOutput(dataset, self._text_line)

  def testFixedLengthReader(self):
    dataset = readers.FixedLengthRecordDataset(
        self._createFixedLengthRecordFiles(), self._record_bytes)
    dataset = input_ops.auto_shard_dataset(
        dataset, self._num_shards, self._shard_index)

    self._verifySimpleShardingOutput(dataset, self._fixed_length_record)

  def testFixedLengthReaderWithFlatMap(self):
    dataset = dataset_ops.Dataset.from_tensor_slices(
        self._createFixedLengthRecordFiles())
    dataset = dataset.flat_map(
        lambda f: readers.FixedLengthRecordDataset(f, self._record_bytes))
    dataset = input_ops.auto_shard_dataset(
        dataset, self._num_shards, self._shard_index)

    self._verifySimpleShardingOutput(dataset, self._fixed_length_record)

if __name__ == "__main__":
  test.main()
