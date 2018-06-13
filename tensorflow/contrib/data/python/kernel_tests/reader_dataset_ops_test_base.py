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

import os

from tensorflow.contrib.data.python.ops import readers
from tensorflow.core.example import example_pb2
from tensorflow.core.example import feature_pb2
from tensorflow.python.data.ops import readers as core_readers
from tensorflow.python.framework import dtypes
from tensorflow.python.lib.io import python_io
from tensorflow.python.ops import parsing_ops
from tensorflow.python.platform import test
from tensorflow.python.util import compat


class ReadBatchFeaturesTestBase(test.TestCase):
  """Base class for setting up and testing `make_batched_feature_dataset`."""

  def setUp(self):
    super(ReadBatchFeaturesTestBase, self).setUp()
    self._num_files = 2
    self._num_records = 7
    self.test_filenames = self._createFiles()

  def make_batch_feature(self,
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
        drop_final_batch=drop_final_batch)

  def _record(self, f, r):
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
                            value=self._get_keywords(f, r)))
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
        keywords_batch_indices.extend(
            [[batch_index, i] for i in range(len(keywords))])
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

  def verify_records(self,
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
