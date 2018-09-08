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

from absl.testing import parameterized
import numpy as np

from tensorflow.contrib.data.python.ops import batching
from tensorflow.contrib.data.python.ops import grouping
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.platform import test


class WindowDatasetTest(test.TestCase, parameterized.TestCase):

  def _structuredDataset(self, structure, shape, dtype):
    if structure is None:
      return dataset_ops.Dataset.from_tensors(
          array_ops.zeros(shape, dtype=dtype))
    else:
      return dataset_ops.Dataset.zip(
          tuple([
              self._structuredDataset(substructure, shape, dtype)
              for substructure in structure
          ]))

  def _structuredElement(self, structure, shape, dtype):
    if structure is None:
      return array_ops.zeros(shape, dtype=dtype)
    else:
      return tuple([
          self._structuredElement(substructure, shape, dtype)
          for substructure in structure
      ])

  def _assertEqual(self, xs, ys):
    self.assertEqual(type(xs), type(ys))
    if isinstance(xs, tuple) and isinstance(ys, tuple):
      self.assertEqual(len(xs), len(ys))
      for x, y in zip(xs, ys):
        self._assertEqual(x, y)
    elif isinstance(xs, np.ndarray) and isinstance(ys, np.ndarray):
      self.assertAllEqual(xs, ys)
    else:
      self.assertEqual(xs, ys)

  @parameterized.named_parameters(
      ("1", None, np.int32([]), dtypes.bool),
      ("2", None, np.int32([]), dtypes.int32),
      ("3", None, np.int32([]), dtypes.float32),
      ("4", None, np.int32([]), dtypes.string),
      ("5", None, np.int32([2]), dtypes.int32),
      ("6", None, np.int32([2, 2]), dtypes.int32),
      ("7", (None, None, None), np.int32([]), dtypes.int32),
      ("8", (None, (None, None)), np.int32([]), dtypes.int32),
  )
  def testWindowDatasetFlatMap(self, structure, shape, dtype):
    """Tests windowing by chaining it with flat map.

    Args:
      structure: the input structure
      shape: the input shape
      dtype: the input data type
    """

    def fn(*args):
      if len(args) == 1 and not isinstance(args[0], tuple):
        return args[0]
      return dataset_ops.Dataset.zip(
          tuple([fn(*arg) if isinstance(arg, tuple) else arg for arg in args]))

    dataset = self._structuredDataset(structure, shape, dtype).apply(
        grouping.window_dataset(5)).flat_map(fn)
    get_next = dataset.make_one_shot_iterator().get_next()
    with self.test_session() as sess:
      expected = sess.run(self._structuredElement(structure, shape, dtype))
      actual = sess.run(get_next)
      self._assertEqual(expected, actual)

  @parameterized.named_parameters(
      ("1", None, np.int32([]), dtypes.bool),
      ("2", None, np.int32([]), dtypes.int32),
      ("3", None, np.int32([]), dtypes.float32),
      ("4", None, np.int32([]), dtypes.string),
      ("5", None, np.int32([2]), dtypes.int32),
      ("6", None, np.int32([2, 2]), dtypes.int32),
      ("7", (None, None, None), np.int32([]), dtypes.int32),
      ("8", (None, (None, None)), np.int32([]), dtypes.int32),
  )
  def testWindowDatasetBatchDense(self, structure, shape, dtype):
    """Tests batching of dense tensor windows.

    Args:
      structure: the input structure
      shape: the input shape
      dtype: the input data type
    """

    def fn(*args):
      if len(args) == 1 and not isinstance(args[0], tuple):
        return batching.batch_window(args[0])

      return tuple([
          fn(*arg) if isinstance(arg, tuple) else batching.batch_window(arg)
          for arg in args
      ])

    dataset = self._structuredDataset(structure, shape, dtype).repeat(5).apply(
        grouping.window_dataset(5)).apply(grouping._map_x_dataset(fn))
    get_next = dataset.make_one_shot_iterator().get_next()
    with self.test_session() as sess:
      expected = sess.run(
          self._structuredElement(structure, np.concatenate(
              ([5], shape), axis=0), dtype))
      actual = sess.run(get_next)
      self._assertEqual(expected, actual)

  @parameterized.named_parameters(
      ("1", np.int32([])),
      ("2", np.int32([1])),
      ("3", np.int32([1, 2, 3])),
  )
  def testWindowDatasetBatchDenseDynamicShape(self, shape):
    """Tests batching of dynamically shaped dense tensor windows.

    Args:
      shape: the input shape
    """

    shape_t = array_ops.placeholder(dtypes.int32)
    dataset = dataset_ops.Dataset.from_tensors(
        array_ops.zeros(shape_t)).repeat(5).apply(
            grouping.window_dataset(5)).apply(
                grouping._map_x_dataset(batching.batch_window))
    iterator = dataset.make_initializable_iterator()
    init_op = iterator.initializer
    get_next = iterator.get_next()
    with self.test_session() as sess:
      sess.run(init_op, {shape_t: shape})
      expected = sess.run(
          self._structuredElement(None, np.concatenate(([5], shape), axis=0),
                                  dtypes.int32))
      actual = sess.run(get_next)
      self._assertEqual(expected, actual)

  def _make_dense_to_sparse_fn(self, is_scalar):

    def dense_to_sparse_scalar(tensor):
      indices = [[]]
      values = array_ops.expand_dims(tensor, 0)
      shape = []
      return sparse_tensor.SparseTensorValue(indices, values, shape)

    def dense_to_sparse_non_scalar(tensor):
      indices = array_ops.where(array_ops.ones_like(tensor, dtype=dtypes.bool))
      values = array_ops.gather_nd(tensor, indices)
      shape = array_ops.shape(tensor, out_type=dtypes.int64)
      return sparse_tensor.SparseTensorValue(indices, values, shape)

    if is_scalar:
      return dense_to_sparse_scalar
    return dense_to_sparse_non_scalar

  def _structuredSparseDataset(self, structure, shape, dtype):
    dense_to_sparse = self._make_dense_to_sparse_fn(len(shape) == 0)  # pylint: disable=g-explicit-length-test
    if structure is None:
      return dataset_ops.Dataset.from_tensors(
          dense_to_sparse(array_ops.zeros(shape, dtype=dtype)))
    else:
      return dataset_ops.Dataset.zip(
          tuple([
              self._structuredSparseDataset(substructure, shape, dtype)
              for substructure in structure
          ]))

  def _structuredSparseElement(self, structure, shape, dtype):
    dense_to_sparse = self._make_dense_to_sparse_fn(len(shape) == 0)  # pylint: disable=g-explicit-length-test
    if structure is None:
      return dense_to_sparse(array_ops.zeros(shape, dtype=dtype))
    else:
      return tuple([
          self._structuredSparseElement(substructure, shape, dtype)
          for substructure in structure
      ])

  @parameterized.named_parameters(
      ("1", None, np.int32([]), dtypes.bool),
      ("2", None, np.int32([]), dtypes.int32),
      ("3", None, np.int32([]), dtypes.float32),
      ("4", None, np.int32([]), dtypes.string),
      ("5", None, np.int32([2]), dtypes.int32),
      ("6", None, np.int32([2, 2]), dtypes.int32),
      ("7", (None, None, None), np.int32([]), dtypes.int32),
      ("8", (None, (None, None)), np.int32([]), dtypes.int32),
  )
  def testWindowDatasetBatchSparse(self, structure, shape, dtype):
    """Tests batching of sparse tensor windows.

    Args:
      structure: the input structure
      shape: the input shape
      dtype: the input data type
    """

    def fn(*args):
      if len(args) == 1 and not isinstance(args[0], tuple):
        return batching.batch_window(args[0])

      return tuple([
          fn(*arg) if isinstance(arg, tuple) else batching.batch_window(arg)
          for arg in args
      ])

    dataset = self._structuredSparseDataset(
        structure, shape, dtype).repeat(5).apply(
            grouping.window_dataset(5)).apply(grouping._map_x_dataset(fn))
    get_next = dataset.make_one_shot_iterator().get_next()
    with self.test_session() as sess:
      expected = sess.run(
          self._structuredSparseElement(structure,
                                        np.concatenate(([5], shape), axis=0),
                                        dtype))
      actual = sess.run(get_next)
      self._assertEqual(expected, actual)

  @parameterized.named_parameters(
      ("1", np.int32([])),
      ("2", np.int32([1])),
      ("3", np.int32([1, 2, 3])),
  )
  def testWindowDatasetBatchSparseDynamicShape(self, shape):
    """Tests batching of dynamically shaped sparse tensor windows.

    Args:
      shape: the input shape
    """

    shape_t = array_ops.placeholder(dtypes.int32)
    dataset = dataset_ops.Dataset.from_tensors(array_ops.zeros(shape_t)).map(
        self._make_dense_to_sparse_fn(len(shape) == 0)).repeat(5).apply(  # pylint: disable=g-explicit-length-test
            grouping.window_dataset(5)).apply(
                grouping._map_x_dataset(batching.batch_window))
    iterator = dataset.make_initializable_iterator()
    init_op = iterator.initializer
    get_next = iterator.get_next()
    with self.test_session() as sess:
      sess.run(init_op, {shape_t: shape})
      expected = sess.run(
          self._structuredSparseElement(None,
                                        np.concatenate(([5], shape), axis=0),
                                        dtypes.int32))
      actual = sess.run(get_next)
      self._assertEqual(expected, actual)

  def _structuredRaggedDataset(self, structure, shapes, dtype):

    if structure is None:
      return dataset_ops.Dataset.from_tensor_slices(shapes).map(
          lambda shape: array_ops.zeros(shape, dtype=dtype))
    else:
      return dataset_ops.Dataset.zip(
          tuple([
              self._structuredRaggedDataset(substructure, shapes, dtype)
              for substructure in structure
          ]))

  @parameterized.named_parameters(
      ("1", None, np.int32([[1], [2], [3]]), dtypes.bool, [-1]),
      ("2", None, np.int32([[1], [2], [3]]), dtypes.int32, [-1]),
      ("3", None, np.int32([[1], [2], [3]]), dtypes.float32, [-1]),
      ("4", None, np.int32([[1], [2], [3]]), dtypes.string, [-1]),
      ("5", None, np.int32([[1, 3], [2, 2], [3, 1]]), dtypes.int32, [-1, -1]),
      ("6", None, np.int32([[3, 1, 3], [1, 3, 1]]), dtypes.int32, [-1, -1, -1]),
      ("7", (None, None, None), np.int32([[1], [2], [3]]), dtypes.int32, [-1]),
      ("8", (None,
             (None, None)), np.int32([[1], [2], [3]]), dtypes.int32, [-1]),
      ("9", None, np.int32([[1], [2], [3]]), dtypes.int32, [-1]),
      ("10", None, np.int32([[1], [2], [3]]), dtypes.int32, np.int32([10])),
  )
  def testWindowDatasetPaddedBatchDense(self, structure, shapes, dtype,
                                        padded_shape):
    """Tests padded batching of dense tensor windows.

    Args:
      structure: the input structure
      shapes: the input shapes
      dtype: the input data type
      padded_shape: the shape to pad the output to
    """

    def fn(*args):
      if len(args) == 1 and not isinstance(args[0], tuple):
        return batching.padded_batch_window(args[0], padded_shape)

      return tuple([
          fn(*arg) if isinstance(arg, tuple) else batching.padded_batch_window(
              arg, padded_shape) for arg in args
      ])

    dataset = self._structuredRaggedDataset(structure, shapes, dtype).apply(
        grouping.window_dataset(len(shapes))).apply(
            grouping._map_x_dataset(fn))
    get_next = dataset.make_one_shot_iterator().get_next()
    with self.test_session() as sess:
      expected_shape = np.maximum(np.amax(shapes, axis=0), padded_shape)
      expected = sess.run(
          self._structuredElement(
              structure,
              np.concatenate((np.int32([len(shapes)]), expected_shape)), dtype))
      actual = sess.run(get_next)
      self._assertEqual(expected, actual)

  @parameterized.named_parameters(
      ("1", np.int32([[1], [2], [3]]), [-1]),
      ("2", np.int32([[1, 3], [2, 2], [3, 1]]), [-1, -1]),
      ("3", np.int32([[3, 1, 3], [1, 3, 1]]), [-1, -1, -1]),
  )
  def testWindowDatasetPaddedBatchDenseDynamicShape(self, shapes, padded_shape):
    """Tests padded batching of dynamically shaped dense tensor windows.

    Args:
      shapes: the input shapes
      padded_shape: the shape to pad the output to
    """

    shapes_t = array_ops.placeholder(dtypes.int32)
    dataset = dataset_ops.Dataset.from_tensor_slices(shapes_t).map(
        lambda shape: array_ops.zeros(shape, dtype=dtypes.int32)).apply(
            grouping.window_dataset(len(shapes))).apply(
                grouping._map_x_dataset(
                    lambda x: batching.padded_batch_window(x, padded_shape)))
    iterator = dataset.make_initializable_iterator()
    init_op = iterator.initializer
    get_next = iterator.get_next()
    with self.test_session() as sess:
      sess.run(init_op, {shapes_t: shapes})
      expected_shape = np.maximum(np.amax(shapes, axis=0), padded_shape)
      expected = sess.run(
          self._structuredElement(
              None, np.concatenate((np.int32([len(shapes)]), expected_shape)),
              dtypes.int32))
      actual = sess.run(get_next)
      self._assertEqual(expected, actual)

  @parameterized.named_parameters(
      ("1", np.int32([[1]]), np.int32([0])),
      ("2", np.int32([[10], [20]]), np.int32([15])),
  )
  def testWindowDatasetPaddedBatchDenseInvalid(self, shapes, padded_shape):
    """Tests invalid padded batching of dense tensor windows.

    Args:
      shapes: the input shapes
      padded_shape: the shape to pad the output to
    """

    dataset = dataset_ops.Dataset.from_tensor_slices(shapes).map(
        lambda shape: array_ops.zeros(shape, dtype=dtypes.int32)).apply(
            grouping.window_dataset(len(shapes))).apply(
                grouping._map_x_dataset(
                    lambda x: batching.padded_batch_window(x, padded_shape)))
    get_next = dataset.make_one_shot_iterator().get_next()
    with self.test_session() as sess:
      with self.assertRaises(errors.InvalidArgumentError):
        sess.run(get_next)

  def _structuredRaggedSparseDataset(self, structure, shapes, dtype):

    def map_fn(shape):
      dense_to_sparse = self._make_dense_to_sparse_fn(False)
      return dense_to_sparse(array_ops.zeros(shape, dtype=dtype))

    if structure is None:
      return dataset_ops.Dataset.from_tensor_slices(shapes).map(map_fn)
    else:
      return dataset_ops.Dataset.zip(
          tuple([
              self._structuredRaggedSparseDataset(substructure, shapes, dtype)
              for substructure in structure
          ]))

  def _structuredRaggedSparseElement(self, structure, shapes, dtype,
                                     padded_shape):
    if structure is None:
      dense_shape = np.maximum(np.amax(shapes, axis=0), padded_shape)
      values = []
      for shape in shapes:
        dense_to_sparse = self._make_dense_to_sparse_fn(len(shape) == 0)  # pylint: disable=g-explicit-length-test
        sparse = dense_to_sparse(array_ops.zeros(shape, dtype=dtype))
        padded_sparse = sparse_tensor.SparseTensor(sparse.indices,
                                                   sparse.values, dense_shape)
        reshaped_sparse = sparse_ops.sparse_reshape(
            padded_sparse,
            array_ops.concat([np.array([1], dtype=np.int64), dense_shape], 0))
        values.append(reshaped_sparse)
      return sparse_ops.sparse_concat(0, values)
    else:
      return tuple([
          self._structuredRaggedSparseElement(substructure, shapes, dtype,
                                              padded_shape)
          for substructure in structure
      ])

  @parameterized.named_parameters(
      ("1", None, np.int64([[1], [2], [3]]), dtypes.bool, [-1]),
      ("2", None, np.int64([[1], [2], [3]]), dtypes.int32, [-1]),
      ("3", None, np.int64([[1], [2], [3]]), dtypes.float32, [-1]),
      ("4", None, np.int64([[1], [2], [3]]), dtypes.string, [-1]),
      ("5", None, np.int64([[1, 3], [2, 2], [3, 1]]), dtypes.int32, [-1, -1]),
      ("6", None, np.int64([[1, 3, 1], [3, 1, 3]]), dtypes.int32, [-1, -1, -1]),
      ("7", (None, None, None), np.int64([[1], [2], [3]]), dtypes.int32, [-1]),
      ("8", (None,
             (None, None)), np.int64([[1], [2], [3]]), dtypes.int32, [-1]),
      ("9", None, np.int64([[1], [2], [3]]), dtypes.int32, [-1]),
      ("10", None, np.int64([[1], [2], [3]]), dtypes.int32, np.int64([10])),
  )
  def testWindowDatasetPaddedBatchSparse(self, structure, shapes, dtype,
                                         padded_shape):
    """Tests padded batching of sparse tensor windows.

    Args:
      structure: the input structure
      shapes: the input shapes
      dtype: the input data type
      padded_shape: the shape to pad the output to
    """

    def fn(*args):
      if len(args) == 1 and not isinstance(args[0], tuple):
        return batching.padded_batch_window(args[0], padded_shape)

      return tuple([
          fn(*arg) if isinstance(arg, tuple) else batching.padded_batch_window(
              arg, padded_shape) for arg in args
      ])

    dataset = self._structuredRaggedSparseDataset(
        structure, shapes, dtype).apply(grouping.window_dataset(
            len(shapes))).apply(grouping._map_x_dataset(fn))
    get_next = dataset.make_one_shot_iterator().get_next()
    with self.test_session() as sess:
      expected = sess.run(
          self._structuredRaggedSparseElement(structure, shapes, dtype,
                                              padded_shape))
      actual = sess.run(get_next)
      self._assertEqual(expected, actual)

  @parameterized.named_parameters(
      ("1", np.int64([[1], [2], [3]]), [-1]),
      ("2", np.int64([[1, 3], [2, 2], [3, 1]]), [-1, -1]),
      ("3", np.int64([[3, 1, 3], [1, 3, 1]]), [-1, -1, -1]),
  )
  def testWindowDatasetPaddedBatchSparseDynamicShape(self, shapes,
                                                     padded_shape):
    """Tests padded batching of dynamically shaped sparse tensor windows.

    Args:
      shapes: the input shapes
      padded_shape: the shape to pad the output to
    """

    shapes_t = array_ops.placeholder(dtypes.int32)
    dataset = dataset_ops.Dataset.from_tensor_slices(shapes_t).map(
        lambda shape: array_ops.zeros(shape, dtype=dtypes.int32)).map(
            self._make_dense_to_sparse_fn(False)
        ).apply(grouping.window_dataset(len(shapes))).apply(
            grouping._map_x_dataset(
                lambda x: batching.padded_batch_window(x, padded_shape)))
    iterator = dataset.make_initializable_iterator()
    init_op = iterator.initializer
    get_next = iterator.get_next()
    with self.test_session() as sess:
      sess.run(init_op, {shapes_t: shapes})
      expected = sess.run(
          self._structuredRaggedSparseElement(None, shapes, dtypes.int32,
                                              padded_shape))
      actual = sess.run(get_next)
      self._assertEqual(expected, actual)

  @parameterized.named_parameters(
      ("1", np.int64([[1]]), [0]),
      ("2", np.int64([[10], [20]]), [15]),
  )
  def testWindowDatasetPaddedBatchSparseInvalid(self, shapes, padded_shape):
    """Tests invalid padded batching of sparse tensor windows.

    Args:
      shapes: the input shapes
      padded_shape: the shape to pad the output to
    """

    dataset = dataset_ops.Dataset.from_tensor_slices(shapes).map(
        lambda shape: array_ops.zeros(shape, dtype=dtypes.int32)).map(
            self._make_dense_to_sparse_fn(False)
        ).apply(grouping.window_dataset(len(shapes))).apply(
            grouping._map_x_dataset(
                lambda x: batching.padded_batch_window(x, padded_shape)))
    get_next = dataset.make_one_shot_iterator().get_next()
    with self.test_session() as sess:
      with self.assertRaises(errors.InvalidArgumentError):
        sess.run(get_next)


if __name__ == "__main__":
  test.main()
