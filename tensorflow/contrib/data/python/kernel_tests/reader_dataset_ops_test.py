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
"""Tests for the experimental input pipeline ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import zlib

import numpy as np

from tensorflow.contrib.data.python.kernel_tests import dataset_serialization_test_base
from tensorflow.contrib.data.python.ops import readers
from tensorflow.core.example import example_pb2
from tensorflow.core.example import feature_pb2
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.data.ops import readers as core_readers
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.lib.io import python_io
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.platform import test
from tensorflow.python.util import compat


class TextLineDatasetTestBase(test.TestCase):

  def _lineText(self, f, l):
    return compat.as_bytes("%d: %d" % (f, l))

  def _createFiles(self,
                   num_files,
                   num_lines,
                   crlf=False,
                   compression_type=None):
    filenames = []
    for i in range(num_files):
      fn = os.path.join(self.get_temp_dir(), "text_line.%d.txt" % i)
      filenames.append(fn)
      contents = []
      for j in range(num_lines):
        contents.append(self._lineText(i, j))
        # Always include a newline after the record unless it is
        # at the end of the file, in which case we include it
        if j + 1 != num_lines or i == 0:
          contents.append(b"\r\n" if crlf else b"\n")
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


class TextLineDatasetSerializationTest(
    TextLineDatasetTestBase,
    dataset_serialization_test_base.DatasetSerializationTestBase):

  def _build_iterator_graph(self, test_filenames, compression_type=None):
    return core_readers.TextLineDataset(
        test_filenames, compression_type=compression_type, buffer_size=10)

  def testTextLineCore(self):
    compression_types = [None, "GZIP", "ZLIB"]
    num_files = 5
    lines_per_file = 5
    num_outputs = num_files * lines_per_file
    for compression_type in compression_types:
      test_filenames = self._createFiles(
          num_files,
          lines_per_file,
          crlf=True,
          compression_type=compression_type)
      # pylint: disable=cell-var-from-loop
      self.run_core_tests(
          lambda: self._build_iterator_graph(test_filenames, compression_type),
          lambda: self._build_iterator_graph(test_filenames), num_outputs)
      # pylint: enable=cell-var-from-loop


class FixedLengthRecordReaderTestBase(test.TestCase):

  def setUp(self):
    super(FixedLengthRecordReaderTestBase, self).setUp()
    self._num_files = 2
    self._num_records = 7
    self._header_bytes = 5
    self._record_bytes = 3
    self._footer_bytes = 2

  def _record(self, f, r):
    return compat.as_bytes(str(f * 2 + r) * self._record_bytes)

  def _createFiles(self):
    filenames = []
    for i in range(self._num_files):
      fn = os.path.join(self.get_temp_dir(), "fixed_length_record.%d.txt" % i)
      filenames.append(fn)
      with open(fn, "wb") as f:
        f.write(b"H" * self._header_bytes)
        for j in range(self._num_records):
          f.write(self._record(i, j))
        f.write(b"F" * self._footer_bytes)
    return filenames


class FixedLengthRecordDatasetSerializationTest(
    FixedLengthRecordReaderTestBase,
    dataset_serialization_test_base.DatasetSerializationTestBase):

  def _build_iterator_graph(self, num_epochs, compression_type=None):
    filenames = self._createFiles()
    return core_readers.FixedLengthRecordDataset(
        filenames, self._record_bytes, self._header_bytes,
        self._footer_bytes).repeat(num_epochs)

  def testFixedLengthRecordCore(self):
    num_epochs = 5
    num_outputs = num_epochs * self._num_files * self._num_records
    self.run_core_tests(lambda: self._build_iterator_graph(num_epochs),
                        lambda: self._build_iterator_graph(num_epochs * 2),
                        num_outputs)


class TFRecordDatasetTestBase(test.TestCase):

  def setUp(self):
    super(TFRecordDatasetTestBase, self).setUp()
    self._num_files = 2
    self._num_records = 7

    self.test_filenames = self._createFiles()

    self.filenames = array_ops.placeholder(dtypes.string, shape=[None])
    self.num_epochs = array_ops.placeholder_with_default(
        constant_op.constant(1, dtypes.int64), shape=[])
    self.compression_type = array_ops.placeholder_with_default("", shape=[])
    self.batch_size = array_ops.placeholder(dtypes.int64, shape=[])

    repeat_dataset = core_readers.TFRecordDataset(
        self.filenames, self.compression_type).repeat(self.num_epochs)
    batch_dataset = repeat_dataset.batch(self.batch_size)

    iterator = iterator_ops.Iterator.from_structure(batch_dataset.output_types)
    self.init_op = iterator.make_initializer(repeat_dataset)
    self.init_batch_op = iterator.make_initializer(batch_dataset)
    self.get_next = iterator.get_next()

  def _record(self, f, r):
    return compat.as_bytes("Record %d of file %d" % (r, f))

  def _createFiles(self):
    filenames = []
    for i in range(self._num_files):
      fn = os.path.join(self.get_temp_dir(), "tf_record.%d.txt" % i)
      filenames.append(fn)
      writer = python_io.TFRecordWriter(fn)
      for j in range(self._num_records):
        writer.write(self._record(i, j))
      writer.close()
    return filenames


class TFRecordDatasetSerializationTest(
    TFRecordDatasetTestBase,
    dataset_serialization_test_base.DatasetSerializationTestBase):

  def _build_iterator_graph(self,
                            num_epochs,
                            batch_size=1,
                            compression_type=None,
                            buffer_size=None):
    filenames = self._createFiles()
    if compression_type is "ZLIB":
      zlib_files = []
      for i, fn in enumerate(filenames):
        with open(fn, "rb") as f:
          cdata = zlib.compress(f.read())
          zfn = os.path.join(self.get_temp_dir(), "tfrecord_%s.z" % i)
          with open(zfn, "wb") as f:
            f.write(cdata)
          zlib_files.append(zfn)
      filenames = zlib_files

    elif compression_type is "GZIP":
      gzip_files = []
      for i, fn in enumerate(self.test_filenames):
        with open(fn, "rb") as f:
          gzfn = os.path.join(self.get_temp_dir(), "tfrecord_%s.gz" % i)
          with gzip.GzipFile(gzfn, "wb") as gzf:
            gzf.write(f.read())
          gzip_files.append(gzfn)
      filenames = gzip_files

    return core_readers.TFRecordDataset(
        filenames, compression_type,
        buffer_size=buffer_size).repeat(num_epochs).batch(batch_size)

  def testTFRecordWithoutBufferCore(self):
    num_epochs = 5
    batch_size = num_epochs
    num_outputs = num_epochs * self._num_files * self._num_records // batch_size
    # pylint: disable=g-long-lambda
    self.run_core_tests(
        lambda: self._build_iterator_graph(num_epochs, batch_size,
                                           buffer_size=0),
        lambda: self._build_iterator_graph(num_epochs * 2, batch_size),
        num_outputs)
    self.run_core_tests(
        lambda: self._build_iterator_graph(num_epochs, buffer_size=0), None,
        num_outputs * batch_size)
    # pylint: enable=g-long-lambda

  def testTFRecordWithBufferCore(self):
    num_epochs = 5
    num_outputs = num_epochs * self._num_files * self._num_records
    self.run_core_tests(lambda: self._build_iterator_graph(num_epochs),
                        lambda: self._build_iterator_graph(num_epochs * 2),
                        num_outputs)

  def testTFRecordWithCompressionCore(self):
    num_epochs = 5
    num_outputs = num_epochs * self._num_files * self._num_records
    self.run_core_tests(
        lambda: self._build_iterator_graph(num_epochs, compression_type="ZLIB"),
        lambda: self._build_iterator_graph(num_epochs * 2), num_outputs)
    self.run_core_tests(
        lambda: self._build_iterator_graph(num_epochs, compression_type="GZIP"),
        lambda: self._build_iterator_graph(num_epochs * 2), num_outputs)


class ReadBatchFeaturesTest(test.TestCase):

  def setUp(self):
    super(ReadBatchFeaturesTest, self).setUp()
    self._num_files = 2
    self._num_records = 7
    self.test_filenames = self._createFiles()

  def _read_batch_features(self,
                           filenames,
                           num_epochs,
                           batch_size,
                           reader_num_threads=1,
                           parser_num_threads=1,
                           shuffle=False,
                           shuffle_seed=None):
    self.filenames = filenames
    self.num_epochs = num_epochs
    self.batch_size = batch_size

    return readers.make_batched_features_dataset(
        file_pattern=self.filenames,
        batch_size=self.batch_size,
        features={
            "file": parsing_ops.FixedLenFeature([], dtypes.int64),
            "record": parsing_ops.FixedLenFeature([], dtypes.int64),
            "keywords": parsing_ops.VarLenFeature(dtypes.string)
        },
        reader=core_readers.TFRecordDataset,
        num_epochs=self.num_epochs,
        shuffle=shuffle,
        shuffle_seed=shuffle_seed,
        reader_num_threads=reader_num_threads,
        parser_num_threads=parser_num_threads).make_one_shot_iterator(
        ).get_next()

  def _record(self, f, r):
    example = example_pb2.Example(features=feature_pb2.Features(
        feature={
            "file":
                feature_pb2.Feature(int64_list=feature_pb2.Int64List(
                    value=[f])),
            "record":
                feature_pb2.Feature(int64_list=feature_pb2.Int64List(
                    value=[r])),
            "keywords":
                feature_pb2.Feature(bytes_list=feature_pb2.BytesList(
                    value=self._get_keywords(f, r)))
        }))
    return example.SerializeToString()

  def _get_keywords(self, f, r):
    num_keywords = 1 + (f + r) % 2
    keywords = []
    for index in range(num_keywords):
      keywords.append(compat.as_bytes("keyword%d" % index))
    return keywords

  def _createFiles(self):
    filenames = []
    for i in range(self._num_files):
      fn = os.path.join(self.get_temp_dir(), "tf_record.%d.txt" % i)
      filenames.append(fn)
      writer = python_io.TFRecordWriter(fn)
      for j in range(self._num_records):
        writer.write(self._record(i, j))
      writer.close()
    return filenames

  def _run_actual_batch(self, outputs, sess):
    file_op = outputs["file"]
    keywords_indices_op = outputs["keywords"].indices
    keywords_values_op = outputs["keywords"].values
    keywords_dense_shape_op = outputs["keywords"].dense_shape
    record_op = outputs["record"]
    return sess.run([
        file_op, keywords_indices_op, keywords_values_op,
        keywords_dense_shape_op, record_op
    ])

  def _next_actual_batch(self, sess):
    return self._run_actual_batch(self.outputs, sess)

  def _next_expected_batch(self,
                           file_indices,
                           batch_size,
                           num_epochs,
                           cycle_length=1):

    def _next_record(file_indices):
      for j in file_indices:
        for i in range(self._num_records):
          yield j, i

    def _next_record_interleaved(file_indices, cycle_length):
      return self._interleave([_next_record([i]) for i in file_indices],
                              cycle_length)

    file_batch = []
    keywords_batch_indices = []
    keywords_batch_values = []
    keywords_batch_max_len = 0
    record_batch = []
    batch_index = 0
    for _ in range(num_epochs):
      if cycle_length == 1:
        next_records = _next_record(file_indices)
      else:
        next_records = _next_record_interleaved(file_indices, cycle_length)
      for record in next_records:
        f = record[0]
        r = record[1]
        file_batch.append(f)
        record_batch.append(r)
        keywords = self._get_keywords(f, r)
        keywords_batch_values.extend(keywords)
        keywords_batch_indices.extend([[batch_index, i]
                                       for i in range(len(keywords))])
        batch_index += 1
        keywords_batch_max_len = max(keywords_batch_max_len, len(keywords))
        if len(file_batch) == batch_size:
          yield [
              file_batch, keywords_batch_indices, keywords_batch_values,
              [batch_size, keywords_batch_max_len], record_batch
          ]
          file_batch = []
          keywords_batch_indices = []
          keywords_batch_values = []
          keywords_batch_max_len = 0
          record_batch = []
          batch_index = 0
    if file_batch:
      yield [
          file_batch, keywords_batch_indices, keywords_batch_values,
          [len(file_batch), keywords_batch_max_len], record_batch
      ]

  def _interleave(self, iterators, cycle_length):
    pending_iterators = iterators
    open_iterators = []
    num_open = 0
    for i in range(cycle_length):
      if pending_iterators:
        open_iterators.append(pending_iterators.pop(0))
        num_open += 1

    while num_open:
      for i in range(min(cycle_length, len(open_iterators))):
        if open_iterators[i] is None:
          continue
        try:
          yield next(open_iterators[i])
        except StopIteration:
          if pending_iterators:
            open_iterators[i] = pending_iterators.pop(0)
          else:
            open_iterators[i] = None
            num_open -= 1

  def _verify_records(self,
                      sess,
                      batch_size,
                      file_index=None,
                      num_epochs=1,
                      interleave_cycle_length=1):
    if file_index is not None:
      file_indices = [file_index]
    else:
      file_indices = range(self._num_files)

    for expected_batch in self._next_expected_batch(
        file_indices, batch_size, num_epochs, interleave_cycle_length):
      actual_batch = self._next_actual_batch(sess)
      for i in range(len(expected_batch)):
        self.assertAllEqual(expected_batch[i], actual_batch[i])

  def testRead(self):
    for batch_size in [1, 2]:
      for num_epochs in [1, 10]:
        with ops.Graph().as_default() as g:
          with self.test_session(graph=g) as sess:
            # Basic test: read from file 0.
            self.outputs = self._read_batch_features(
                filenames=self.test_filenames[0],
                num_epochs=num_epochs,
                batch_size=batch_size)
            self._verify_records(sess, batch_size, 0, num_epochs=num_epochs)
            with self.assertRaises(errors.OutOfRangeError):
              self._next_actual_batch(sess)

        with ops.Graph().as_default() as g:
          with self.test_session(graph=g) as sess:
            # Basic test: read from file 1.
            self.outputs = self._read_batch_features(
                filenames=self.test_filenames[1],
                num_epochs=num_epochs,
                batch_size=batch_size)
            self._verify_records(sess, batch_size, 1, num_epochs=num_epochs)
            with self.assertRaises(errors.OutOfRangeError):
              self._next_actual_batch(sess)

        with ops.Graph().as_default() as g:
          with self.test_session(graph=g) as sess:
            # Basic test: read from both files.
            self.outputs = self._read_batch_features(
                filenames=self.test_filenames,
                num_epochs=num_epochs,
                batch_size=batch_size)
            self._verify_records(sess, batch_size, num_epochs=num_epochs)
            with self.assertRaises(errors.OutOfRangeError):
              self._next_actual_batch(sess)

  def testReadWithEquivalentDataset(self):
    features = {
        "file": parsing_ops.FixedLenFeature([], dtypes.int64),
        "record": parsing_ops.FixedLenFeature([], dtypes.int64),
    }
    dataset = (core_readers.TFRecordDataset(self.test_filenames)
               .map(lambda x: parsing_ops.parse_single_example(x, features))
               .repeat(10).batch(2))
    iterator = dataset.make_initializable_iterator()
    init_op = iterator.initializer
    next_element = iterator.get_next()

    with self.test_session() as sess:
      sess.run(init_op)
      for file_batch, _, _, _, record_batch in self._next_expected_batch(
          range(self._num_files), 2, 10):
        actual_batch = sess.run(next_element)
        self.assertAllEqual(file_batch, actual_batch["file"])
        self.assertAllEqual(record_batch, actual_batch["record"])
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(next_element)

  def testReadWithFusedShuffleRepeatDataset(self):
    num_epochs = 5
    total_records = num_epochs * self._num_records
    for batch_size in [1, 2]:
      # Test that shuffling with same seed produces the same result.
      with ops.Graph().as_default() as g:
        with self.test_session(graph=g) as sess:
          outputs1 = self._read_batch_features(
              filenames=self.test_filenames[0],
              num_epochs=num_epochs,
              batch_size=batch_size,
              shuffle=True,
              shuffle_seed=5)
          outputs2 = self._read_batch_features(
              filenames=self.test_filenames[0],
              num_epochs=num_epochs,
              batch_size=batch_size,
              shuffle=True,
              shuffle_seed=5)
          for _ in range(total_records // batch_size):
            batch1 = self._run_actual_batch(outputs1, sess)
            batch2 = self._run_actual_batch(outputs2, sess)
            for i in range(len(batch1)):
              self.assertAllEqual(batch1[i], batch2[i])

      # Test that shuffling with different seeds produces a different order.
      with ops.Graph().as_default() as g:
        with self.test_session(graph=g) as sess:
          outputs1 = self._read_batch_features(
              filenames=self.test_filenames[0],
              num_epochs=num_epochs,
              batch_size=batch_size,
              shuffle=True,
              shuffle_seed=5)
          outputs2 = self._read_batch_features(
              filenames=self.test_filenames[0],
              num_epochs=num_epochs,
              batch_size=batch_size,
              shuffle=True,
              shuffle_seed=15)
          all_equal = True
          for _ in range(total_records // batch_size):
            batch1 = self._run_actual_batch(outputs1, sess)
            batch2 = self._run_actual_batch(outputs2, sess)
            for i in range(len(batch1)):
              all_equal = all_equal and np.array_equal(batch1[i], batch2[i])
          self.assertFalse(all_equal)

  def testParallelReadersAndParsers(self):
    num_epochs = 5
    for batch_size in [1, 2]:
      for reader_num_threads in [2, 4]:
        for parser_num_threads in [2, 4]:
          with ops.Graph().as_default() as g:
            with self.test_session(graph=g) as sess:
              self.outputs = self._read_batch_features(
                  filenames=self.test_filenames,
                  num_epochs=num_epochs,
                  batch_size=batch_size,
                  reader_num_threads=reader_num_threads,
                  parser_num_threads=parser_num_threads)
              self._verify_records(
                  sess,
                  batch_size,
                  num_epochs=num_epochs,
                  interleave_cycle_length=reader_num_threads)
              with self.assertRaises(errors.OutOfRangeError):
                self._next_actual_batch(sess)


if __name__ == "__main__":
  test.main()
