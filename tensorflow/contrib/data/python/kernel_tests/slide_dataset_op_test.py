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
"""Tests for the experimental input pipeline ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np

from tensorflow.contrib.data.python.ops import sliding
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


@test_util.run_v1_only("deprecated API, no eager or V2 test coverage")
class SlideDatasetTest(test_base.DatasetTestBase, parameterized.TestCase):

  @parameterized.named_parameters(
      ("1", 20, 14, 7, 1),
      ("2", 20, 17, 9, 1),
      ("3", 20, 14, 14, 1),
      ("4", 20, 10, 14, 1),
      ("5", 20, 14, 19, 1),
      ("6", 20, 4, 1, 2),
      ("7", 20, 2, 1, 6),
      ("8", 20, 4, 7, 2),
      ("9", 20, 2, 7, 6),
      ("10", 1, 10, 4, 1),
      ("11", 0, 10, 4, 1),
  )
  def testSlideDataset(self, count, window_size, window_shift, window_stride):
    """Tests a dataset that slides a window its input elements."""
    components = (np.arange(7),
                  np.array([[1, 2, 3]]) * np.arange(7)[:, np.newaxis],
                  np.array(37.0) * np.arange(7))

    count_t = array_ops.placeholder(dtypes.int64, shape=[])
    window_size_t = array_ops.placeholder(dtypes.int64, shape=[])
    window_shift_t = array_ops.placeholder(dtypes.int64, shape=[])
    window_stride_t = array_ops.placeholder(dtypes.int64, shape=[])

    def _map_fn(x, y, z):
      return math_ops.square(x), math_ops.square(y), math_ops.square(z)

    # The pipeline is TensorSliceDataset -> MapDataset(square_3) ->
    # RepeatDataset(count) ->
    # _SlideDataset(window_size, window_shift, window_stride).
    iterator = dataset_ops.make_initializable_iterator(
        dataset_ops.Dataset.from_tensor_slices(components).map(_map_fn)
        .repeat(count).apply(
            sliding.sliding_window_batch(
                window_size=window_size_t,
                window_shift=window_shift_t,
                window_stride=window_stride_t)))
    init_op = iterator.initializer
    get_next = iterator.get_next()

    self.assertEqual([[None] + list(c.shape[1:]) for c in components],
                     [t.shape.as_list() for t in get_next])

    with self.cached_session() as sess:
      sess.run(
          init_op,
          feed_dict={
              count_t: count,
              window_size_t: window_size,
              window_shift_t: window_shift,
              window_stride_t: window_stride
          })
      num_batches = (count * 7 - (
          (window_size - 1) * window_stride + 1)) // window_shift + 1
      for i in range(num_batches):
        result = sess.run(get_next)
        for component, result_component in zip(components, result):
          for j in range(window_size):
            self.assertAllEqual(
                component[(i * window_shift + j * window_stride) % 7]**2,
                result_component[j])
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  @parameterized.named_parameters(
      ("1", 20, 14, 7, 1),
      ("2", 20, 17, 9, 1),
      ("3", 20, 14, 14, 1),
      ("4", 20, 10, 14, 1),
      ("5", 20, 14, 19, 1),
      ("6", 20, 4, 1, 2),
      ("7", 20, 2, 1, 6),
      ("8", 20, 4, 7, 2),
      ("9", 20, 2, 7, 6),
      ("10", 1, 10, 4, 1),
      ("11", 0, 10, 4, 1),
  )
  def testSlideDatasetDeprecated(self, count, window_size, stride,
                                 window_stride):
    """Tests a dataset that slides a window its input elements."""
    components = (np.arange(7),
                  np.array([[1, 2, 3]]) * np.arange(7)[:, np.newaxis],
                  np.array(37.0) * np.arange(7))

    count_t = array_ops.placeholder(dtypes.int64, shape=[])
    window_size_t = array_ops.placeholder(dtypes.int64, shape=[])
    stride_t = array_ops.placeholder(dtypes.int64, shape=[])
    window_stride_t = array_ops.placeholder(dtypes.int64, shape=[])

    def _map_fn(x, y, z):
      return math_ops.square(x), math_ops.square(y), math_ops.square(z)

    # The pipeline is TensorSliceDataset -> MapDataset(square_3) ->
    # RepeatDataset(count) -> _SlideDataset(window_size, stride, window_stride).
    iterator = dataset_ops.make_initializable_iterator(
        dataset_ops.Dataset.from_tensor_slices(components).map(_map_fn)
        .repeat(count).apply(
            sliding.sliding_window_batch(
                window_size=window_size_t,
                stride=stride_t,
                window_stride=window_stride_t)))
    init_op = iterator.initializer
    get_next = iterator.get_next()

    self.assertEqual([[None] + list(c.shape[1:]) for c in components],
                     [t.shape.as_list() for t in get_next])

    with self.cached_session() as sess:
      sess.run(
          init_op,
          feed_dict={
              count_t: count,
              window_size_t: window_size,
              stride_t: stride,
              window_stride_t: window_stride
          })
      num_batches = (count * 7 - (
          (window_size - 1) * window_stride + 1)) // stride + 1
      for i in range(num_batches):
        result = sess.run(get_next)
        for component, result_component in zip(components, result):
          for j in range(window_size):
            self.assertAllEqual(
                component[(i * stride + j * window_stride) % 7]**2,
                result_component[j])
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  @parameterized.named_parameters(
      ("1", 14, 0, 3, 1),
      ("2", 14, 3, 0, 1),
      ("3", 14, 3, 3, 0),
  )
  def testSlideDatasetInvalid(self, count, window_size, window_shift,
                              window_stride):
    count_t = array_ops.placeholder(dtypes.int64, shape=[])
    window_size_t = array_ops.placeholder(dtypes.int64, shape=[])
    window_shift_t = array_ops.placeholder(dtypes.int64, shape=[])
    window_stride_t = array_ops.placeholder(dtypes.int64, shape=[])

    iterator = dataset_ops.make_initializable_iterator(
        dataset_ops.Dataset.range(10).map(lambda x: x).repeat(count_t).apply(
            sliding.sliding_window_batch(
                window_size=window_size_t,
                window_shift=window_shift_t,
                window_stride=window_stride_t)))
    init_op = iterator.initializer

    with self.cached_session() as sess:
      with self.assertRaises(errors.InvalidArgumentError):
        sess.run(
            init_op,
            feed_dict={
                count_t: count,
                window_size_t: window_size,
                window_shift_t: window_shift,
                window_stride_t: window_stride
            })

  def testSlideDatasetValueError(self):
    with self.assertRaises(ValueError):
      dataset_ops.Dataset.range(10).map(lambda x: x).apply(
          sliding.sliding_window_batch(
              window_size=1, stride=1, window_shift=1, window_stride=1))

  def testSlideSparse(self):

    def _sparse(i):
      return sparse_tensor.SparseTensorValue(
          indices=[[0]], values=(i * [1]), dense_shape=[1])

    iterator = dataset_ops.make_initializable_iterator(
        dataset_ops.Dataset.range(10).map(_sparse).apply(
            sliding.sliding_window_batch(window_size=5, window_shift=3)))
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.cached_session() as sess:
      sess.run(init_op)
      num_batches = (10 - 5) // 3 + 1
      for i in range(num_batches):
        actual = sess.run(get_next)
        expected = sparse_tensor.SparseTensorValue(
            indices=[[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]],
            values=[i * 3, i * 3 + 1, i * 3 + 2, i * 3 + 3, i * 3 + 4],
            dense_shape=[5, 1])
        self.assertTrue(sparse_tensor.is_sparse(actual))
        self.assertSparseValuesEqual(actual, expected)
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  def testSlideSparseWithDifferentDenseShapes(self):

    def _sparse(i):
      return sparse_tensor.SparseTensorValue(
          indices=array_ops.expand_dims(
              math_ops.range(i, dtype=dtypes.int64), 1),
          values=array_ops.fill([math_ops.to_int32(i)], i),
          dense_shape=[i])

    iterator = dataset_ops.make_initializable_iterator(
        dataset_ops.Dataset.range(10).map(_sparse).apply(
            sliding.sliding_window_batch(window_size=5, window_shift=3)))
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.cached_session() as sess:
      sess.run(init_op)
      num_batches = (10 - 5) // 3 + 1
      for i in range(num_batches):
        actual = sess.run(get_next)
        expected_indices = []
        expected_values = []
        for j in range(5):
          for k in range(i * 3 + j):
            expected_indices.append([j, k])
            expected_values.append(i * 3 + j)
        expected = sparse_tensor.SparseTensorValue(
            indices=expected_indices,
            values=expected_values,
            dense_shape=[5, i * 3 + 5 - 1])
        self.assertTrue(sparse_tensor.is_sparse(actual))
        self.assertSparseValuesEqual(actual, expected)
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  def testNestedSlideSparse(self):

    def _sparse(i):
      return sparse_tensor.SparseTensorValue(
          indices=[[0]], values=(i * [1]), dense_shape=[1])

    iterator = dataset_ops.make_initializable_iterator(
        dataset_ops.Dataset.range(10).map(_sparse).apply(
            sliding.sliding_window_batch(window_size=4, window_shift=2)).apply(
                sliding.sliding_window_batch(window_size=3, window_shift=1)))
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.cached_session() as sess:
      sess.run(init_op)
      # Slide: 1st batch.
      actual = sess.run(get_next)
      expected = sparse_tensor.SparseTensorValue(
          indices=[[0, 0, 0], [0, 1, 0], [0, 2, 0], [0, 3, 0], [1, 0, 0],
                   [1, 1, 0], [1, 2, 0], [1, 3, 0], [2, 0, 0], [2, 1, 0],
                   [2, 2, 0], [2, 3, 0]],
          values=[0, 1, 2, 3, 2, 3, 4, 5, 4, 5, 6, 7],
          dense_shape=[3, 4, 1])
      self.assertTrue(sparse_tensor.is_sparse(actual))
      self.assertSparseValuesEqual(actual, expected)
      # Slide: 2nd batch.
      actual = sess.run(get_next)
      expected = sparse_tensor.SparseTensorValue(
          indices=[[0, 0, 0], [0, 1, 0], [0, 2, 0], [0, 3, 0], [1, 0, 0],
                   [1, 1, 0], [1, 2, 0], [1, 3, 0], [2, 0, 0], [2, 1, 0],
                   [2, 2, 0], [2, 3, 0]],
          values=[2, 3, 4, 5, 4, 5, 6, 7, 6, 7, 8, 9],
          dense_shape=[3, 4, 1])
      self.assertTrue(sparse_tensor.is_sparse(actual))
      self.assertSparseValuesEqual(actual, expected)
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  def testSlideShapeError(self):

    def generator():
      yield [1.0, 2.0, 3.0]
      yield [4.0, 5.0, 6.0]
      yield [7.0, 8.0, 9.0, 10.0]

    iterator = dataset_ops.make_initializable_iterator(
        dataset_ops.Dataset.from_generator(
            generator, dtypes.float32, output_shapes=[None]).apply(
                sliding.sliding_window_batch(window_size=3, window_shift=1)))
    next_element = iterator.get_next()

    with self.cached_session() as sess:
      sess.run(iterator.initializer)
      with self.assertRaisesRegexp(
          errors.InvalidArgumentError,
          r"Cannot batch tensors with different shapes in component 0. "
          r"First element had shape \[3\] and element 2 had shape \[4\]."):
        sess.run(next_element)


if __name__ == "__main__":
  test.main()
