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
                           shuffle_seed=None,
                           drop_final_batch=False):
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
        parser_num_threads=parser_num_threads,
        drop_final_batch=drop_final_batch).make_one_shot_iterator(
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

  def testDropFinalBatch(self):
    for batch_size in [1, 2]:
      for num_epochs in [1, 10]:
        with ops.Graph().as_default():
          # Basic test: read from file 0.
          self.outputs = self._read_batch_features(
              filenames=self.test_filenames[0],
              num_epochs=num_epochs,
              batch_size=batch_size,
              drop_final_batch=True)
          for _, tensor in self.outputs.items():
            if isinstance(tensor, ops.Tensor):  # Guard against SparseTensor.
              self.assertEqual(tensor.shape[0], batch_size)


class MakeCsvDatasetTest(test.TestCase):

  COLUMN_TYPES = [
      dtypes.int32, dtypes.int64, dtypes.float32, dtypes.float64, dtypes.string
  ]
  COLUMNS = ["col%d" % i for i in range(len(COLUMN_TYPES))]
  DEFAULT_VALS = [[], [], [], [], ["NULL"]]
  DEFAULTS = [
      constant_op.constant([], dtype=dtypes.int32),
      constant_op.constant([], dtype=dtypes.int64),
      constant_op.constant([], dtype=dtypes.float32),
      constant_op.constant([], dtype=dtypes.float64),
      constant_op.constant(["NULL"], dtype=dtypes.string)
  ]
  LABEL = COLUMNS[0]

  def setUp(self):
    super(MakeCsvDatasetTest, self).setUp()
    self._num_files = 2
    self._num_records = 11
    self._test_filenames = self._create_files()

  def _csv_values(self, fileno, recordno):
    return [
        fileno,
        recordno,
        fileno * recordno * 0.5,
        fileno * recordno + 0.5,
        "record %d" % recordno if recordno % 2 == 1 else "",
    ]

  def _csv_record(self, fileno, recordno):
    return ",".join(str(v) for v in self._csv_values(fileno, recordno))

  def _create_file(self, fileno, header=True, comment=True):
    fn = os.path.join(self.get_temp_dir(), "csv_file%d.csv" % fileno)
    f = open(fn, "w")
    if header:
      f.write(",".join(self.COLUMNS) + "\n")
    for recno in range(self._num_records):
      f.write(self._csv_record(fileno, recno) + "\n")
      if comment:
        f.write("# Some comment goes here. Should be ignored!\n")
    f.close()
    return fn

  def _create_files(self):
    filenames = []
    for i in range(self._num_files):
      filenames.append(self._create_file(i))
    return filenames

  def _make_csv_dataset(
      self,
      filenames,
      defaults,
      column_names=COLUMNS,
      label_name=LABEL,
      batch_size=1,
      num_epochs=1,
      shuffle=False,
      shuffle_seed=None,
      header=True,
      comment="#",
      na_value="",
      default_float_type=dtypes.float32,
  ):
    return readers.make_csv_dataset(
        filenames,
        batch_size=batch_size,
        column_names=column_names,
        column_defaults=defaults,
        label_name=label_name,
        num_epochs=num_epochs,
        shuffle=shuffle,
        shuffle_seed=shuffle_seed,
        header=header,
        comment=comment,
        na_value=na_value,
        default_float_type=default_float_type,
    )

  def _next_actual_batch(self, file_indices, batch_size, num_epochs, defaults):
    features = {col: list() for col in self.COLUMNS}
    for _ in range(num_epochs):
      for i in file_indices:
        for j in range(self._num_records):
          values = self._csv_values(i, j)
          for n, v in enumerate(values):
            if v == "":  # pylint: disable=g-explicit-bool-comparison
              values[n] = defaults[n][0]
          values[-1] = values[-1].encode("utf-8")

          # Regroup lists by column instead of row
          for n, col in enumerate(self.COLUMNS):
            features[col].append(values[n])
          if len(list(features.values())[0]) == batch_size:
            yield features
            features = {col: list() for col in self.COLUMNS}

  def _run_actual_batch(self, outputs, sess):
    features, labels = sess.run(outputs)
    batch = [features[k] for k in self.COLUMNS if k != self.LABEL]
    batch.append(labels)
    return batch

  def _verify_records(
      self,
      sess,
      dataset,
      file_indices,
      defaults=tuple(DEFAULT_VALS),
      label_name=LABEL,
      batch_size=1,
      num_epochs=1,
  ):
    iterator = dataset.make_one_shot_iterator()
    get_next = iterator.get_next()

    for expected_features in self._next_actual_batch(file_indices, batch_size,
                                                     num_epochs, defaults):
      actual_features = sess.run(get_next)

      if label_name is not None:
        expected_labels = expected_features.pop(label_name)
        # Compare labels
        self.assertAllEqual(expected_labels, actual_features[1])
        actual_features = actual_features[0]  # Extract features dict from tuple

      for k in expected_features.keys():
        # Compare features
        self.assertAllEqual(expected_features[k], actual_features[k])

    with self.assertRaises(errors.OutOfRangeError):
      sess.run(get_next)

  def test_make_csv_dataset(self):
    defaults = self.DEFAULTS

    with ops.Graph().as_default() as g:
      with self.test_session(graph=g) as sess:
        # Basic test: read from file 0.
        dataset = self._make_csv_dataset(self._test_filenames[0], defaults)
        self._verify_records(sess, dataset, [0])
    with ops.Graph().as_default() as g:
      with self.test_session(graph=g) as sess:
        # Basic test: read from file 1.
        dataset = self._make_csv_dataset(self._test_filenames[1], defaults)
        self._verify_records(sess, dataset, [1])
    with ops.Graph().as_default() as g:
      with self.test_session(graph=g) as sess:
        # Read from both files.
        dataset = self._make_csv_dataset(self._test_filenames, defaults)
        self._verify_records(sess, dataset, range(self._num_files))
    with ops.Graph().as_default() as g:
      with self.test_session(graph=g) as sess:
        # Read from both files. Exercise the `batch` and `num_epochs` parameters
        # of make_csv_dataset and make sure they work.
        dataset = self._make_csv_dataset(
            self._test_filenames, defaults, batch_size=2, num_epochs=10)
        self._verify_records(
            sess, dataset, range(self._num_files), batch_size=2, num_epochs=10)

  def test_make_csv_dataset_with_bad_columns(self):
    """Tests that exception is raised when input is malformed.
    """
    dupe_columns = self.COLUMNS[:-1] + self.COLUMNS[:1]
    defaults = self.DEFAULTS

    # Duplicate column names
    with self.assertRaises(ValueError):
      self._make_csv_dataset(
          self._test_filenames, defaults, column_names=dupe_columns)

    # Label key not one of column names
    with self.assertRaises(ValueError):
      self._make_csv_dataset(
          self._test_filenames, defaults, label_name="not_a_real_label")

  def test_make_csv_dataset_with_no_label(self):
    """Tests that CSV datasets can be created when no label is specified.
    """
    defaults = self.DEFAULTS
    with ops.Graph().as_default() as g:
      with self.test_session(graph=g) as sess:
        # Read from both files. Make sure this works with no label key supplied.
        dataset = self._make_csv_dataset(
            self._test_filenames,
            defaults,
            batch_size=2,
            num_epochs=10,
            label_name=None)
        self._verify_records(
            sess,
            dataset,
            range(self._num_files),
            batch_size=2,
            num_epochs=10,
            label_name=None)

  def test_make_csv_dataset_with_no_comments(self):
    """Tests that datasets can be created from CSV files with no header line.
    """
    defaults = self.DEFAULTS
    file_without_header = self._create_file(
        len(self._test_filenames), comment=False)
    with ops.Graph().as_default() as g:
      with self.test_session(graph=g) as sess:
        dataset = self._make_csv_dataset(
            file_without_header,
            defaults,
            batch_size=2,
            num_epochs=10,
            comment=None,
        )
        self._verify_records(
            sess,
            dataset,
            [len(self._test_filenames)],
            batch_size=2,
            num_epochs=10,
        )

  def test_make_csv_dataset_with_no_header(self):
    """Tests that datasets can be created from CSV files with no header line.
    """
    defaults = self.DEFAULTS
    file_without_header = self._create_file(
        len(self._test_filenames), header=False)
    with ops.Graph().as_default() as g:
      with self.test_session(graph=g) as sess:
        dataset = self._make_csv_dataset(
            file_without_header,
            defaults,
            batch_size=2,
            num_epochs=10,
            header=False,
        )
        self._verify_records(
            sess,
            dataset,
            [len(self._test_filenames)],
            batch_size=2,
            num_epochs=10,
        )

  def test_make_csv_dataset_with_types(self):
    """Tests that defaults can be a dtype instead of a Tensor for required vals.
    """
    defaults = [d for d in self.COLUMN_TYPES[:-1]]
    defaults.append(constant_op.constant(["NULL"], dtype=dtypes.string))
    with ops.Graph().as_default() as g:
      with self.test_session(graph=g) as sess:
        dataset = self._make_csv_dataset(self._test_filenames, defaults)
        self._verify_records(sess, dataset, range(self._num_files))

  def test_make_csv_dataset_with_no_col_names(self):
    """Tests that datasets can be created when column names are not specified.

    In that case, we should infer the column names from the header lines.
    """
    defaults = self.DEFAULTS
    with ops.Graph().as_default() as g:
      with self.test_session(graph=g) as sess:
        # Read from both files. Exercise the `batch` and `num_epochs` parameters
        # of make_csv_dataset and make sure they work.
        dataset = self._make_csv_dataset(
            self._test_filenames,
            defaults,
            column_names=None,
            batch_size=2,
            num_epochs=10)
        self._verify_records(
            sess, dataset, range(self._num_files), batch_size=2, num_epochs=10)

  def test_make_csv_dataset_type_inference(self):
    """Tests that datasets can be created when no defaults are specified.

    In that case, we should infer the types from the first N records.
    """
    # Test that it works with standard test files (with comments, header, etc)
    with ops.Graph().as_default() as g:
      with self.test_session(graph=g) as sess:
        dataset = self._make_csv_dataset(
            self._test_filenames, defaults=None, batch_size=2, num_epochs=10)
        self._verify_records(
            sess,
            dataset,
            range(self._num_files),
            batch_size=2,
            num_epochs=10,
            defaults=[[], [], [], [], [""]])

    # Test on a deliberately tricky file
    fn = os.path.join(self.get_temp_dir(), "file.csv")
    expected_dtypes = [
        dtypes.int32, dtypes.int64, dtypes.float32, dtypes.float32,
        dtypes.string, dtypes.string
    ]
    rows = [[0, 0, 0, "NAN", "", "a"], [1, 2**31 + 1, 2**64, 123, "NAN", ""],
            ['"123"', 2, 2**64, 123.4, "NAN", '"cd,efg"']]
    expected = [[0, 0, 0, 0, "", "a"], [1, 2**31 + 1, 2**64, 123, "", ""],
                [123, 2, 2**64, 123.4, "", "cd,efg"]]
    for row in expected:
      row[-1] = row[-1].encode("utf-8")  # py3 expects byte strings
      row[-2] = row[-2].encode("utf-8")  # py3 expects byte strings
    col_names = ["col%d" % i for i in range(len(expected_dtypes))]
    with open(fn, "w") as f:
      f.write(",".join(col_names))
      f.write("\n")
      for row in rows:
        f.write(",".join([str(v) if v else "" for v in row]) + "\n")

    with ops.Graph().as_default() as g:
      with self.test_session(graph=g) as sess:
        dataset = self._make_csv_dataset(
            fn,
            defaults=None,
            column_names=None,
            batch_size=1,
            num_epochs=1,
            label_name=None,
            na_value="NAN",
            default_float_type=dtypes.float32,
        )
        features = dataset.make_one_shot_iterator().get_next()
        # Check that types match
        for i in range(len(expected_dtypes)):
          assert features["col%d" % i].dtype == expected_dtypes[i]
        for i in range(len(rows)):
          assert sess.run(features) == dict(zip(col_names, expected[i]))

    # With float64 as default type for floats
    expected_dtypes = [
        dtypes.int32, dtypes.int64, dtypes.float64, dtypes.float64,
        dtypes.string, dtypes.string
    ]
    with ops.Graph().as_default() as g:
      with self.test_session(graph=g) as sess:
        dataset = self._make_csv_dataset(
            fn,
            defaults=None,
            column_names=None,
            batch_size=1,
            num_epochs=1,
            label_name=None,
            na_value="NAN",
            default_float_type=dtypes.float64,
        )
        features = dataset.make_one_shot_iterator().get_next()
        # Check that types match
        for i in range(len(expected_dtypes)):
          assert features["col%d" % i].dtype == expected_dtypes[i]
        for i in range(len(rows)):
          assert sess.run(features) == dict(zip(col_names, expected[i]))

  def test_make_csv_dataset_with_shuffle(self):
    total_records = self._num_files * self._num_records
    defaults = self.DEFAULTS
    for batch_size in [1, 2]:
      with ops.Graph().as_default() as g:
        with self.test_session(graph=g) as sess:
          # Test that shuffling with the same seed produces the same result
          dataset1 = self._make_csv_dataset(
              self._test_filenames,
              defaults,
              batch_size=batch_size,
              shuffle=True,
              shuffle_seed=5)
          dataset2 = self._make_csv_dataset(
              self._test_filenames,
              defaults,
              batch_size=batch_size,
              shuffle=True,
              shuffle_seed=5)
          outputs1 = dataset1.make_one_shot_iterator().get_next()
          outputs2 = dataset2.make_one_shot_iterator().get_next()
          for _ in range(total_records // batch_size):
            batch1 = self._run_actual_batch(outputs1, sess)
            batch2 = self._run_actual_batch(outputs2, sess)
            for i in range(len(batch1)):
              self.assertAllEqual(batch1[i], batch2[i])

      with ops.Graph().as_default() as g:
        with self.test_session(graph=g) as sess:
          # Test that shuffling with a different seed produces different results
          dataset1 = self._make_csv_dataset(
              self._test_filenames,
              defaults,
              batch_size=batch_size,
              shuffle=True,
              shuffle_seed=5)
          dataset2 = self._make_csv_dataset(
              self._test_filenames,
              defaults,
              batch_size=batch_size,
              shuffle=True,
              shuffle_seed=6)
          outputs1 = dataset1.make_one_shot_iterator().get_next()
          outputs2 = dataset2.make_one_shot_iterator().get_next()
          all_equal = False
          for _ in range(total_records // batch_size):
            batch1 = self._run_actual_batch(outputs1, sess)
            batch2 = self._run_actual_batch(outputs2, sess)
            for i in range(len(batch1)):
              all_equal = all_equal and np.array_equal(batch1[i], batch2[i])
          self.assertFalse(all_equal)


if __name__ == "__main__":
  test.main()
