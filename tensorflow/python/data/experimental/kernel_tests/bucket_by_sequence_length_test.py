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
"""Tests for `tf.data.experimental.bucket_by_sequence_length()."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

from tensorflow.python.data.experimental.ops import grouping
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


def _element_length_fn(x, y=None):
  del y
  return array_ops.shape(x)[0]


def _to_sparse_tensor(record):
  return sparse_tensor.SparseTensor(**record)


def _format_record(array, sparse):
  if sparse:
    return {
        "values": array,
        "indices": [[i] for i in range(len(array))],
        "dense_shape": (len(array),)
    }
  return array


def _get_record_type(sparse):
  if sparse:
    return {
        "values": dtypes.int64,
        "indices": dtypes.int64,
        "dense_shape": dtypes.int64
    }
  return dtypes.int32


def _get_record_shape(sparse):
  if sparse:
    return {
        "values": tensor_shape.TensorShape([None,]),
        "indices": tensor_shape.TensorShape([None, 1]),
        "dense_shape": tensor_shape.TensorShape([1,])
    }
  return tensor_shape.TensorShape([None])


class BucketBySequenceLengthTest(test_base.DatasetTestBase):

  def testBucket(self):

    boundaries = [10, 20, 30]
    batch_sizes = [10, 8, 4, 2]
    lengths = [8, 13, 25, 35]

    def build_dataset(sparse):
      def _generator():
        # Produce 1 batch for each bucket
        elements = []
        for batch_size, length in zip(batch_sizes, lengths):
          record_len = length - 1
          for _ in range(batch_size):
            elements.append([1] * record_len)
            record_len = length
        random.shuffle(elements)
        for el in elements:
          yield (_format_record(el, sparse),)
      dataset = dataset_ops.Dataset.from_generator(
          _generator,
          (_get_record_type(sparse),),
          (_get_record_shape(sparse),))
      if sparse:
        dataset = dataset.map(lambda x: (_to_sparse_tensor(x),))
      return dataset

    def _test_bucket_by_padding(no_padding):
      dataset = build_dataset(sparse=no_padding)
      dataset = dataset.apply(
          grouping.bucket_by_sequence_length(
              _element_length_fn,
              boundaries,
              batch_sizes,
              no_padding=no_padding))
      batch, = dataset_ops.make_one_shot_iterator(dataset).get_next()

      with self.cached_session() as sess:
        batches = []
        for _ in range(4):
          batches.append(self.evaluate(batch))
        with self.assertRaises(errors.OutOfRangeError):
          self.evaluate(batch)
      batch_sizes_val = []
      lengths_val = []
      for batch in batches:
        shape = batch.dense_shape if no_padding else batch.shape
        batch_size = shape[0]
        length = shape[1]
        batch_sizes_val.append(batch_size)
        lengths_val.append(length)
        sum_check = batch.values.sum() if no_padding else batch.sum()
        self.assertEqual(sum_check, batch_size * length - 1)
      self.assertEqual(sum(batch_sizes_val), sum(batch_sizes))
      self.assertEqual(sorted(batch_sizes), sorted(batch_sizes_val))
      self.assertEqual(sorted(lengths), sorted(lengths_val))

    for no_padding in (True, False):
      _test_bucket_by_padding(no_padding)

  def testPadToBoundary(self):

    boundaries = [10, 20, 30]
    batch_sizes = [10, 8, 4, 2]
    lengths = [8, 13, 25]

    def element_gen():
      # Produce 1 batch for each bucket
      elements = []
      for batch_size, length in zip(batch_sizes[:-1], lengths):
        for _ in range(batch_size):
          elements.append([1] * length)
      random.shuffle(elements)
      for el in elements:
        yield (el,)
      for _ in range(batch_sizes[-1]):
        el = [1] * (boundaries[-1] + 5)
        yield (el,)

    element_len = lambda el: array_ops.shape(el)[0]
    dataset = dataset_ops.Dataset.from_generator(
        element_gen, (dtypes.int64,), ([None],)).apply(
            grouping.bucket_by_sequence_length(
                element_len, boundaries, batch_sizes,
                pad_to_bucket_boundary=True))
    batch, = dataset_ops.make_one_shot_iterator(dataset).get_next()

    with self.cached_session() as sess:
      batches = []
      for _ in range(3):
        batches.append(self.evaluate(batch))
      with self.assertRaisesOpError("bucket_boundaries"):
        self.evaluate(batch)
    batch_sizes_val = []
    lengths_val = []
    for batch in batches:
      batch_size = batch.shape[0]
      length = batch.shape[1]
      batch_sizes_val.append(batch_size)
      lengths_val.append(length)
    batch_sizes = batch_sizes[:-1]
    self.assertEqual(sum(batch_sizes_val), sum(batch_sizes))
    self.assertEqual(sorted(batch_sizes), sorted(batch_sizes_val))
    self.assertEqual([boundary - 1 for boundary in sorted(boundaries)],
                     sorted(lengths_val))

  def testPadToBoundaryNoExtraneousPadding(self):

    boundaries = [3, 7, 11]
    batch_sizes = [2, 2, 2, 2]
    lengths = range(1, 11)

    def element_gen():
      for length in lengths:
        yield ([1] * length,)

    element_len = lambda element: array_ops.shape(element)[0]
    dataset = dataset_ops.Dataset.from_generator(
        element_gen, (dtypes.int64,), ([None],)).apply(
            grouping.bucket_by_sequence_length(
                element_len, boundaries, batch_sizes,
                pad_to_bucket_boundary=True))
    batch, = dataset_ops.make_one_shot_iterator(dataset).get_next()

    with self.cached_session() as sess:
      batches = []
      for _ in range(5):
        batches.append(self.evaluate(batch))
      with self.assertRaises(errors.OutOfRangeError):
        self.evaluate(batch)

    self.assertAllEqual(batches[0], [[1, 0],
                                     [1, 1]])
    self.assertAllEqual(batches[1], [[1, 1, 1, 0, 0, 0],
                                     [1, 1, 1, 1, 0, 0]])
    self.assertAllEqual(batches[2], [[1, 1, 1, 1, 1, 0],
                                     [1, 1, 1, 1, 1, 1]])
    self.assertAllEqual(batches[3], [[1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                                     [1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
    self.assertAllEqual(batches[4], [[1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                                     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

  def testTupleElements(self):

    def build_dataset(sparse):
      def _generator():
        text = [[1, 2, 3], [3, 4, 5, 6, 7], [1, 2], [8, 9, 0, 2, 3]]
        label = [1, 2, 1, 2]
        for x, y in zip(text, label):
          yield (_format_record(x, sparse), y)
      dataset = dataset_ops.Dataset.from_generator(
          generator=_generator,
          output_types=(_get_record_type(sparse), dtypes.int32),
          output_shapes=(_get_record_shape(sparse),
                         tensor_shape.TensorShape([])))
      if sparse:
        dataset = dataset.map(lambda x, y: (_to_sparse_tensor(x), y))
      return dataset

    def _test_tuple_elements_by_padding(no_padding):
      dataset = build_dataset(sparse=no_padding)
      dataset = dataset.apply(grouping.bucket_by_sequence_length(
          element_length_func=_element_length_fn,
          bucket_batch_sizes=[2, 2, 2],
          bucket_boundaries=[0, 8],
          no_padding=no_padding))
      shapes = dataset.output_shapes
      self.assertEqual([None, None], shapes[0].as_list())
      self.assertEqual([None], shapes[1].as_list())

    for no_padding in (True, False):
      _test_tuple_elements_by_padding(no_padding)

  def testBucketSparse(self):
    """Tests bucketing of sparse tensors (case where `no_padding` == True).

    Test runs on following dataset:
      [
        [0],
        [0, 1],
        [0, 1, 2]
        ...
        [0, ..., max_len - 1]
      ]
    Sequences are bucketed by length and batched with
      `batch_size` < `bucket_size`.
    """

    min_len = 0
    max_len = 100
    batch_size = 7
    bucket_size = 10

    def _build_dataset():
      input_data = [range(i+1) for i in range(min_len, max_len)]
      def generator_fn():
        for record in input_data:
          yield _format_record(record, sparse=True)
      dataset = dataset_ops.Dataset.from_generator(
          generator=generator_fn,
          output_types=_get_record_type(sparse=True))
      dataset = dataset.map(_to_sparse_tensor)
      return dataset

    def _compute_expected_batches():
      """Computes expected batch outputs and stores in a set."""
      all_expected_sparse_tensors = set()
      for bucket_start_len in range(min_len, max_len, bucket_size):
        for batch_offset in range(0, bucket_size, batch_size):
          batch_start_len = bucket_start_len + batch_offset
          batch_end_len = min(batch_start_len + batch_size,
                              bucket_start_len + bucket_size)
          expected_indices = []
          expected_values = []
          for length in range(batch_start_len, batch_end_len):
            for val in range(length + 1):
              expected_indices.append((length - batch_start_len, val))
              expected_values.append(val)
          expected_sprs_tensor = (tuple(expected_indices),
                                  tuple(expected_values))
          all_expected_sparse_tensors.add(expected_sprs_tensor)
      return all_expected_sparse_tensors

    def _compute_batches(dataset):
      """Computes actual batch outputs of dataset and stores in a set."""
      batch = dataset_ops.make_one_shot_iterator(dataset).get_next()
      all_sparse_tensors = set()
      with self.cached_session() as sess:
        with self.assertRaises(errors.OutOfRangeError):
          while True:
            output = self.evaluate(batch)
            sprs_tensor = (tuple([tuple(idx) for idx in output.indices]),
                           tuple(output.values))
            all_sparse_tensors.add(sprs_tensor)
      return all_sparse_tensors

    dataset = _build_dataset()
    boundaries = range(min_len + bucket_size + 1, max_len, bucket_size)
    dataset = dataset.apply(grouping.bucket_by_sequence_length(
        _element_length_fn,
        boundaries,
        [batch_size] * (len(boundaries) + 1),
        no_padding=True))
    batches = _compute_batches(dataset)
    expected_batches = _compute_expected_batches()
    self.assertEqual(batches, expected_batches)


if __name__ == "__main__":
  test.main()
