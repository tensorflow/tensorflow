# -*- coding: utf-8 -*-
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

import time

from absl.testing import parameterized
import numpy as np

from tensorflow.python.client import session
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.platform import test
from tensorflow.python.util import compat


class BatchDatasetTest(test_base.DatasetTestBase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('even', 28, 14, False),
      ('uneven_with_remainder', 28, 15, False),
      ('uneven_without_remainder', 28, 15, True),
      ('empty', 0, 14, False),
  )
  def testBatchDataset(self, count, batch_size, drop_remainder):
    """Tests the batch dataset logic for various input configurations.

    Args:
      count: the number of input elements
      batch_size: the batch size
      drop_remainder: whether a smaller batch size should be produced if batch
        size does not divide number of inputs evenly
    """

    # The pipeline is TensorSliceDataset -> MapDataset(square_3) ->
    # RepeatDataset(count) -> BatchDataset(batch_size).
    components = (np.arange(7),
                  np.array([[1, 2, 3]]) * np.arange(7)[:, np.newaxis],
                  np.array(37.0) * np.arange(7))

    count_t = array_ops.placeholder(dtypes.int64, shape=[])
    batch_size_t = array_ops.placeholder(dtypes.int64, shape=[])
    drop_remainder_t = array_ops.placeholder(dtypes.bool, shape=[])

    def _map_fn(x, y, z):
      return math_ops.square(x), math_ops.square(y), math_ops.square(z)

    iterator = (
        dataset_ops.Dataset.from_tensor_slices(components).map(_map_fn)
        .repeat(count).batch(batch_size,
                             drop_remainder).make_initializable_iterator())
    init_op = iterator.initializer
    get_next = iterator.get_next()

    if drop_remainder:
      dim0 = batch_size
    else:
      dim0 = None
    self.assertEqual([[dim0] + list(c.shape[1:]) for c in components],
                     [t.shape.as_list() for t in get_next])

    with self.cached_session() as sess:
      sess.run(
          init_op,
          feed_dict={
              count_t: count,
              batch_size_t: batch_size,
              drop_remainder_t: drop_remainder
          })
      num_full_batches = (count * 7) // batch_size
      for i in range(num_full_batches):
        result = self.evaluate(get_next)
        for component, result_component in zip(components, result):
          for j in range(batch_size):
            self.assertAllEqual(component[(i * batch_size + j) % 7]**2,
                                result_component[j])
      if not drop_remainder and (count * 7) % batch_size > 0:
        result = self.evaluate(get_next)
        for component, result_component in zip(components, result):
          for j in range((count * 7) % batch_size):
            self.assertAllEqual(
                component[(num_full_batches * batch_size + j) % 7]**2,
                result_component[j])
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  def testBatchDatasetInvalidBatchSize(self):
    iterator = (dataset_ops.Dataset.range(10).batch(0).make_one_shot_iterator())
    get_next = iterator.get_next()

    with self.cached_session() as sess:
      with self.assertRaises(errors.InvalidArgumentError):
        sess.run(get_next)

  def testBatchSparse(self):

    def _sparse(i):
      return sparse_tensor.SparseTensorValue(
          indices=[[0]], values=(i * [1]), dense_shape=[1])

    iterator = dataset_ops.Dataset.range(10).map(_sparse).batch(
        5).make_initializable_iterator()
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.cached_session() as sess:
      self.evaluate(init_op)
      for i in range(2):
        actual = self.evaluate(get_next)
        expected = sparse_tensor.SparseTensorValue(
            indices=[[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]],
            values=[i * 5, i * 5 + 1, i * 5 + 2, i * 5 + 3, i * 5 + 4],
            dense_shape=[5, 1])
        self.assertTrue(sparse_tensor.is_sparse(actual))
        self.assertSparseValuesEqual(actual, expected)
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  def testBatchSparseWithDifferentDenseShapes(self):

    def _sparse(i):
      return sparse_tensor.SparseTensorValue(
          indices=array_ops.expand_dims(
              math_ops.range(i, dtype=dtypes.int64), 1),
          values=array_ops.fill([math_ops.to_int32(i)], i),
          dense_shape=[i])

    iterator = dataset_ops.Dataset.range(10).map(_sparse).batch(
        5).make_initializable_iterator()
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.cached_session() as sess:
      self.evaluate(init_op)
      for i in range(2):
        actual = self.evaluate(get_next)
        expected_indices = []
        expected_values = []
        for j in range(5):
          for k in range(i * 5 + j):
            expected_indices.append([j, k])
            expected_values.append(i * 5 + j)
        expected = sparse_tensor.SparseTensorValue(
            indices=expected_indices,
            values=expected_values,
            dense_shape=[5, (i + 1) * 5 - 1])
        self.assertTrue(sparse_tensor.is_sparse(actual))
        self.assertSparseValuesEqual(actual, expected)
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  def testNestedBatchSparse(self):

    def _sparse(i):
      return sparse_tensor.SparseTensorValue(
          indices=[[0]], values=(i * [1]), dense_shape=[1])

    iterator = dataset_ops.Dataset.range(10).map(_sparse).batch(5).batch(
        2).make_initializable_iterator()
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.cached_session() as sess:
      self.evaluate(init_op)
      actual = self.evaluate(get_next)
      expected = sparse_tensor.SparseTensorValue(
          indices=[[0, 0, 0], [0, 1, 0], [0, 2, 0], [0, 3, 0], [0, 4, 0],
                   [1, 0, 0], [1, 1, 0], [1, 2, 0], [1, 3, 0], [1, 4, 0]],
          values=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
          dense_shape=[2, 5, 1])
      self.assertTrue(sparse_tensor.is_sparse(actual))
      self.assertSparseValuesEqual(actual, expected)
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  def testBatchShapeError(self):

    def generator():
      yield [1.0, 2.0, 3.0]
      yield [4.0, 5.0, 6.0]
      yield [7.0, 8.0, 9.0, 10.0]

    iterator = (
        dataset_ops.Dataset.from_generator(
            generator, dtypes.float32, output_shapes=[None]).batch(3)
        .make_initializable_iterator())
    next_element = iterator.get_next()

    with self.cached_session() as sess:
      self.evaluate(iterator.initializer)
      with self.assertRaisesRegexp(
          errors.InvalidArgumentError,
          r'Cannot batch tensors with different shapes in component 0. '
          r'First element had shape \[3\] and element 2 had shape \[4\].'):
        sess.run(next_element)


def _random_seq_lens(count):
  return np.random.randint(20, size=(count,)).astype(np.int32)


class PaddedBatchDatasetTest(test_base.DatasetTestBase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('default_padding', _random_seq_lens(32), 4, [-1], False),
      ('constant_padding', _random_seq_lens(32), 4, [25], False),
      ('uneven_with_remainder', _random_seq_lens(34), 4, [-1], False),
      ('uneven_without_remainder', _random_seq_lens(34), 4, [-1], True),
  )
  def testPaddedBatchDataset(self, seq_lens, batch_size, padded_shapes,
                             drop_remainder):
    """Tests the padded batch dataset logic for various input configurations.

    Args:
      seq_lens: the input sequence lengths
      batch_size: the batch size
      padded_shapes: the padded shapes to use
      drop_remainder: whether a smaller batch size should be produced if batch
        size does not divide number of inputs evenly
    """

    seq_lens_t = array_ops.placeholder(dtypes.int32, shape=[None])
    batch_size_t = array_ops.placeholder(dtypes.int64, shape=[])
    padded_shapes_t = array_ops.placeholder(dtypes.int64, shape=[1])
    drop_remainder_t = array_ops.placeholder(dtypes.bool, shape=[])

    iterator = (
        dataset_ops.Dataset.from_tensor_slices(seq_lens_t)
        .map(lambda x: array_ops.fill([x], x)).padded_batch(
            batch_size=batch_size_t,
            drop_remainder=drop_remainder_t,
            padded_shapes=padded_shapes_t).make_initializable_iterator())

    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.cached_session() as sess:
      sess.run(
          init_op,
          feed_dict={
              seq_lens_t: seq_lens,
              batch_size_t: batch_size,
              padded_shapes_t: padded_shapes,
              drop_remainder_t: drop_remainder,
          })

      num_full_batches = len(seq_lens) // batch_size

      for i in range(num_full_batches):
        result = self.evaluate(get_next)
        padded_len = padded_shapes[0]
        if padded_len is None or padded_len == -1:
          padded_len = np.max(result) if result.size > 0 else 0
        self.assertEqual((batch_size, padded_len), result.shape)
        for j in range(batch_size):
          seq_len = seq_lens[(i * batch_size) + j]
          self.assertAllEqual(result[j, :seq_len], [seq_len] * seq_len)
          self.assertAllEqual(result[j, seq_len:],
                              [0] * (padded_len - seq_len))

      if not drop_remainder and len(seq_lens) % batch_size > 0:
        result = self.evaluate(get_next)
        padded_len = np.max(result) if result.size > 0 else 0
        self.assertEqual((len(seq_lens) % batch_size, padded_len),
                         result.shape)
        for j in range(len(seq_lens) % batch_size):
          seq_len = seq_lens[num_full_batches * batch_size + j]
          self.assertAllEqual(result[j, :seq_len], [seq_len] * seq_len)
          self.assertAllEqual(result[j, seq_len:],
                              [0] * (padded_len - seq_len))

      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  def testPaddedBatchShortPadding(self):
    iterator = (
        dataset_ops.Dataset.from_tensor_slices([6, 5, 5, 5, 5])
        .map(lambda x: array_ops.fill([x], x)).padded_batch(
            batch_size=4, padded_shapes=[5]).make_one_shot_iterator())
    get_next = iterator.get_next()

    with self.cached_session() as sess:
      with self.assertRaises(errors.DataLossError):
        sess.run(get_next)

  def testPaddedBatchEmptyTensors(self):
    iterator = (
        dataset_ops.Dataset.from_tensor_slices([0, 0, 0, 0])
        .map(lambda x: array_ops.fill([x], x)).padded_batch(
            batch_size=4, padded_shapes=[-1]).make_one_shot_iterator())
    get_next = iterator.get_next()

    with self.cached_session() as sess:
      result = self.evaluate(get_next)
      self.assertAllEqual([[], [], [], []], result)
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  def testPaddedBatchDatasetNonDefaultPadding(self):
    seq_lens = array_ops.placeholder(dtypes.int32, shape=[None])
    padded_shape = array_ops.placeholder(dtypes.int64, shape=[1])

    def fill_tuple(x):
      filled = array_ops.fill([x], x)
      return (filled, string_ops.as_string(filled))

    iterator = (
        dataset_ops.Dataset.from_tensor_slices(seq_lens).map(fill_tuple)
        .padded_batch(
            4,
            padded_shapes=(padded_shape, padded_shape),
            padding_values=(-1, '<end>')).make_initializable_iterator())

    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.cached_session() as sess:
      # Test with random sequence lengths, and max padding.
      random_seq_lens = np.random.randint(20, size=(32,)).astype(np.int32)
      sess.run(
          init_op, feed_dict={
              padded_shape: [-1],
              seq_lens: random_seq_lens
          })
      for i in range(8):
        result = self.evaluate(get_next)
        padded_len = np.max(result[0])
        self.assertEqual((4, padded_len), result[0].shape)
        self.assertEqual((4, padded_len), result[1].shape)
        for j in range(4):
          seq_len = random_seq_lens[(i * 4) + j]
          self.assertAllEqual(result[0][j, :seq_len], [seq_len] * seq_len)
          self.assertAllEqual(result[0][j, seq_len:],
                              [-1] * (padded_len - seq_len))
          self.assertAllEqual(result[1][j, :seq_len],
                              [compat.as_bytes(str(seq_len))] * seq_len)
          self.assertAllEqual(result[1][j, seq_len:],
                              [b'<end>'] * (padded_len - seq_len))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  def testPaddedBatchDatasetUnicode(self):
    # See GitHub issue 16149
    def generator():
      data = [[u'Простой', u'тест', u'юникода'],
              [u'никогда', u'не', u'бывает', u'простым']]

      for seq in data:
        yield seq, [0, 1, 2, 3]

    dataset = dataset_ops.Dataset.from_generator(
        generator, (dtypes.string, dtypes.int32),
        (tensor_shape.TensorShape([None]), tensor_shape.TensorShape([None])))
    padded_dataset = dataset.padded_batch(
        2, padded_shapes=([None], [None]), padding_values=('', 0))
    with self.cached_session() as sess:
      next_element = padded_dataset.make_one_shot_iterator().get_next()
      sess.run(next_element)

  def testPaddedBatchDatasetShapeSpecifications(self):
    int_placeholder = array_ops.placeholder(dtypes.int32)
    float_placeholder = array_ops.placeholder(dtypes.float32)
    string_placeholder = array_ops.placeholder(dtypes.string)
    input_dataset = dataset_ops.Dataset.from_tensors(
        (int_placeholder, float_placeholder, string_placeholder))

    # Test different ways of specifying the `padded_shapes` argument.
    dynamic_padding_from_tensor_shapes = input_dataset.padded_batch(
        32,
        padded_shapes=(tensor_shape.TensorShape([None]),
                       tensor_shape.TensorShape([None, None]),
                       tensor_shape.TensorShape([37])))
    dynamic_padding_from_lists = input_dataset.padded_batch(
        32, padded_shapes=([None], [None, None], [37]))
    dynamic_padding_from_lists_with_minus_one = input_dataset.padded_batch(
        32, padded_shapes=([-1], [-1, -1], [37]))
    dynamic_padding_from_tensors = input_dataset.padded_batch(
        32,
        padded_shapes=(constant_op.constant([-1], dtype=dtypes.int64),
                       constant_op.constant([-1, -1], dtype=dtypes.int64),
                       constant_op.constant([37], dtype=dtypes.int64)))

    for dataset in [
        dynamic_padding_from_tensor_shapes, dynamic_padding_from_lists,
        dynamic_padding_from_lists_with_minus_one, dynamic_padding_from_tensors
    ]:
      self.assertEqual([None, None], dataset.output_shapes[0].as_list())
      self.assertEqual([None, None, None], dataset.output_shapes[1].as_list())
      self.assertEqual([None, 37], dataset.output_shapes[2].as_list())

  def testPaddedBatchSparseError(self):

    def _map_fn(i):
      return sparse_tensor.SparseTensorValue(
          indices=[[0, 0]], values=(i * [1]), dense_shape=[1, 1]), i

    with self.assertRaises(TypeError):
      _ = dataset_ops.Dataset.range(10).map(_map_fn).padded_batch(10)

  def testPaddedBatchShapeError(self):
    with self.assertRaisesRegexp(
        ValueError, r'The padded shape \(1,\) is not compatible with the '
        r'corresponding input component shape \(\).'):
      _ = dataset_ops.Dataset.range(10).padded_batch(5, padded_shapes=[1])

    with self.assertRaisesRegexp(
        ValueError, r'The padded shape \(1,\) is not compatible with the '
        r'corresponding input component shape \(3,\).'):
      _ = dataset_ops.Dataset.from_tensors([1, 2, 3]).padded_batch(
          5, padded_shapes=[1])

    with self.assertRaisesRegexp(
        ValueError, r'Padded shape .* must be a 1-D tensor '
        r'of tf.int64 values, but its shape was \(2, 2\).'):
      _ = dataset_ops.Dataset.from_tensors([1, 2, 3]).padded_batch(
          5, padded_shapes=[[1, 1], [1, 1]])

    with self.assertRaisesRegexp(
        TypeError, r'Padded shape .* must be a 1-D tensor '
        r'of tf.int64 values, but its element type was float32.'):
      _ = dataset_ops.Dataset.from_tensors([1, 2, 3]).padded_batch(
          5, padded_shapes=constant_op.constant([1., 2., 3.]))

    with self.assertRaisesRegexp(
        ValueError, r'The padded shape \(1,\) is not compatible with the '
        r'corresponding input component shape \(\).'):
      shape_as_tensor = constant_op.constant([1], dtype=dtypes.int64)
      _ = dataset_ops.Dataset.range(10).padded_batch(
          5, padded_shapes=shape_as_tensor)

    with self.assertRaisesRegexp(
        ValueError,
        r'The padded shape \((\?|None), (\?|None)\) is not compatible with the '
        r'corresponding input component shape \(\).'):
      shape_as_tensor = array_ops.placeholder(dtypes.int64, shape=[2])
      _ = dataset_ops.Dataset.range(10).padded_batch(
          5, padded_shapes=shape_as_tensor)


class BatchDatasetBenchmark(test.Benchmark):

  def benchmarkBatchSparse(self):
    non_zeros_per_row_values = [0, 1, 5, 10, 100]
    batch_size_values = [1, 32, 64, 128, 1024]

    sparse_placeholder = array_ops.sparse_placeholder(dtype=dtypes.int64)
    batch_size_placeholder = array_ops.placeholder(dtype=dtypes.int64, shape=[])

    dataset = dataset_ops.Dataset.from_tensors(sparse_placeholder).repeat(
        ).batch(batch_size_placeholder)
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()

    for non_zeros_per_row in non_zeros_per_row_values:

      sparse_value = sparse_tensor.SparseTensorValue(
          indices=np.arange(non_zeros_per_row, dtype=np.int64)[:, np.newaxis],
          values=np.arange(non_zeros_per_row, dtype=np.int64),
          dense_shape=[1000])

      for batch_size in batch_size_values:

        with session.Session() as sess:
          sess.run(iterator.initializer, feed_dict={
              sparse_placeholder: sparse_value,
              batch_size_placeholder: batch_size})
          # Run five steps to warm up the session caches before taking the
          # first measurement.
          for _ in range(5):
            sess.run(next_element.indices.op)
          deltas = []
          for _ in range(100):
            start = time.time()
            for _ in range(100):
              sess.run(next_element.indices.op)
            end = time.time()
            deltas.append(end - start)

        median_wall_time = np.median(deltas) / 100.0

        print('Batch sparse dataset non-zeros per row: %d batch_size: %d '
              'wall time: %f'
              % (non_zeros_per_row, batch_size, median_wall_time))
        self.report_benchmark(
            iters=10000, wall_time=median_wall_time,
            name='benchmark_batch_sparse_dataset_nnz_%d_batch_size_%d' % (
                non_zeros_per_row, batch_size))


if __name__ == '__main__':
  test.main()
