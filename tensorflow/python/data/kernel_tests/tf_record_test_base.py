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
"""Base class for testing reader datasets."""

import os

from tensorflow.core.example import example_pb2
from tensorflow.core.example import feature_pb2
from tensorflow.python.data.experimental.ops import readers
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import readers as core_readers
from tensorflow.python.framework import dtypes
from tensorflow.python.lib.io import python_io
from tensorflow.python.ops import parsing_ops
from tensorflow.python.util import compat


class FeaturesTestBase(test_base.DatasetTestBase):
  """Base class for testing TFRecord-based features."""

  def setUp(self):
    super(FeaturesTestBase, self).setUp()
    self._num_files = 2
    self._num_records = 7
    self._filenames = self._createFiles()

  def make_batch_feature(self,
                         filenames,
                         num_epochs,
                         batch_size,
                         label_key=None,
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
            "keywords": parsing_ops.VarLenFeature(dtypes.string),
            "label": parsing_ops.FixedLenFeature([], dtypes.string),
        },
        label_key=label_key,
        reader=core_readers.TFRecordDataset,
        num_epochs=self.num_epochs,
        shuffle=shuffle,
        shuffle_seed=shuffle_seed,
        reader_num_threads=reader_num_threads,
        parser_num_threads=parser_num_threads,
        drop_final_batch=drop_final_batch)

  def _record(self, f, r, l):
    example = example_pb2.Example(
        features=feature_pb2.Features(
            feature={
                "file":
                    feature_pb2.Feature(
                        int64_list=feature_pb2.Int64List(value=[f])),
                "record":
                    feature_pb2.Feature(
                        int64_list=feature_pb2.Int64List(value=[r])),
                "keywords":
                    feature_pb2.Feature(
                        bytes_list=feature_pb2.BytesList(
                            value=self._get_keywords(f, r))),
                "label":
                    feature_pb2.Feature(
                        bytes_list=feature_pb2.BytesList(
                            value=[compat.as_bytes(l)]))
            }))
    return example.SerializeToString()

  def _get_keywords(self, f, r):
    num_keywords = 1 + (f + r) % 2
    keywords = []
    for index in range(num_keywords):
      keywords.append(compat.as_bytes("keyword%d" % index))
    return keywords

  def _sum_keywords(self, num_files):
    sum_keywords = 0
    for i in range(num_files):
      for j in range(self._num_records):
        sum_keywords += 1 + (i + j) % 2
    return sum_keywords

  def _createFiles(self):
    filenames = []
    for i in range(self._num_files):
      fn = os.path.join(self.get_temp_dir(), "tf_record.%d.txt" % i)
      filenames.append(fn)
      writer = python_io.TFRecordWriter(fn)
      for j in range(self._num_records):
        writer.write(self._record(i, j, "fake-label"))
      writer.close()
    return filenames

  def _run_actual_batch(self, outputs, label_key_provided=False):
    if label_key_provided:
      # outputs would be a tuple of (feature dict, label)
      features, label = self.evaluate(outputs())
    else:
      features = self.evaluate(outputs())
      label = features["label"]
    file_out = features["file"]
    keywords_indices = features["keywords"].indices
    keywords_values = features["keywords"].values
    keywords_dense_shape = features["keywords"].dense_shape
    record = features["record"]
    return ([
        file_out, keywords_indices, keywords_values, keywords_dense_shape,
        record, label
    ])

  def _next_actual_batch(self, label_key_provided=False):
    return self._run_actual_batch(self.outputs, label_key_provided)

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

  def _next_expected_batch(self,
                           file_indices,
                           batch_size,
                           num_epochs,
                           cycle_length=1):

    def _next_record(file_indices):
      for j in file_indices:
        for i in range(self._num_records):
          yield j, i, compat.as_bytes("fake-label")

    def _next_record_interleaved(file_indices, cycle_length):
      return self._interleave([_next_record([i]) for i in file_indices],
                              cycle_length)

    file_batch = []
    keywords_batch_indices = []
    keywords_batch_values = []
    keywords_batch_max_len = 0
    record_batch = []
    batch_index = 0
    label_batch = []
    for _ in range(num_epochs):
      if cycle_length == 1:
        next_records = _next_record(file_indices)
      else:
        next_records = _next_record_interleaved(file_indices, cycle_length)
      for record in next_records:
        f = record[0]
        r = record[1]
        label_batch.append(record[2])
        file_batch.append(f)
        record_batch.append(r)
        keywords = self._get_keywords(f, r)
        keywords_batch_values.extend(keywords)
        keywords_batch_indices.extend(
            [[batch_index, i] for i in range(len(keywords))])
        batch_index += 1
        keywords_batch_max_len = max(keywords_batch_max_len, len(keywords))
        if len(file_batch) == batch_size:
          yield [
              file_batch, keywords_batch_indices, keywords_batch_values,
              [batch_size, keywords_batch_max_len], record_batch, label_batch
          ]
          file_batch = []
          keywords_batch_indices = []
          keywords_batch_values = []
          keywords_batch_max_len = 0
          record_batch = []
          batch_index = 0
          label_batch = []
    if file_batch:
      yield [
          file_batch, keywords_batch_indices, keywords_batch_values,
          [len(file_batch), keywords_batch_max_len], record_batch, label_batch
      ]

  def _verify_records(self,
                      batch_size,
                      file_index=None,
                      num_epochs=1,
                      label_key_provided=False,
                      interleave_cycle_length=1):
    if file_index is not None:
      file_indices = [file_index]
    else:
      file_indices = range(self._num_files)

    for expected_batch in self._next_expected_batch(
        file_indices,
        batch_size,
        num_epochs,
        cycle_length=interleave_cycle_length):
      actual_batch = self._next_actual_batch(
          label_key_provided=label_key_provided)
      for i in range(len(expected_batch)):
        self.assertAllEqual(expected_batch[i], actual_batch[i])


class TFRecordTestBase(test_base.DatasetTestBase):
  """Base class for TFRecord-based tests."""

  def setUp(self):
    super(TFRecordTestBase, self).setUp()
    self._num_files = 2
    self._num_records = 7
    self._filenames = self._createFiles()

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

  def _next_expected_batch(self, file_indices, batch_size, num_epochs,
                           cycle_length, drop_final_batch, use_parser_fn):

    def _next_record(file_indices):
      for j in file_indices:
        for i in range(self._num_records):
          yield j, i

    def _next_record_interleaved(file_indices, cycle_length):
      return self._interleave([_next_record([i]) for i in file_indices],
                              cycle_length)

    record_batch = []
    batch_index = 0
    for _ in range(num_epochs):
      if cycle_length == 1:
        next_records = _next_record(file_indices)
      else:
        next_records = _next_record_interleaved(file_indices, cycle_length)
      for f, r in next_records:
        record = self._record(f, r)
        if use_parser_fn:
          record = record[1:]
        record_batch.append(record)
        batch_index += 1
        if len(record_batch) == batch_size:
          yield record_batch
          record_batch = []
          batch_index = 0
    if record_batch and not drop_final_batch:
      yield record_batch

  def _verify_records(self, outputs, batch_size, file_index, num_epochs,
                      interleave_cycle_length, drop_final_batch, use_parser_fn):
    if file_index is not None:
      if isinstance(file_index, list):
        file_indices = file_index
      else:
        file_indices = [file_index]
    else:
      file_indices = range(self._num_files)

    for expected_batch in self._next_expected_batch(
        file_indices, batch_size, num_epochs, interleave_cycle_length,
        drop_final_batch, use_parser_fn):
      actual_batch = self.evaluate(outputs())
      self.assertAllEqual(expected_batch, actual_batch)

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
