# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for `tf.data.experimental.dense_to_ragged_batch`."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np

from tensorflow.python.data.experimental.ops import batching
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import nest
from tensorflow.python.framework import combinations
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import ragged_concat_ops
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.platform import test


def _make_scalar_ds(nrows):
  """Create a test dataset with scalar elements."""
  return dataset_ops.Dataset.from_tensor_slices(np.arange(nrows))


def _make_vector_ds(nrows):
  """Create a test dataset with vector elements (of varying size)."""
  return _make_scalar_ds(nrows).map(lambda x: array_ops.fill([x], x))


def _make_matrix_ds1(nrows):
  """Create a test dataset with matrix elements (of varying size)."""
  return _make_scalar_ds(nrows).map(lambda x: array_ops.fill([x, 2], x))


def _make_matrix_ds2(nrows):
  """Create a test dataset with matrix elements (of varying size)."""
  return _make_scalar_ds(nrows).map(lambda x: array_ops.fill([2, x], x))


def _make_matrix_ds_fully_defined(nrows):
  """Create a test dataset with matrix elements (of varying size)."""
  return _make_scalar_ds(nrows).map(lambda x: array_ops.fill([2, 3], x))


def _make_5dtensor_ds(nrows):
  """Create a test dataset with matrix elements (of varying size)."""
  return _make_scalar_ds(nrows).map(
      lambda x: array_ops.fill([2, x, 3, 2*x, 4], x))


def _make_ragged_ds(nrows):
  """Create a test dataset with RaggedTensor elements (of varying size)."""
  values = [[[i] * (i % 3) for i in range(j)] * (j % 3) for j in range(nrows)]
  rt = ragged_factory_ops.constant(values)
  return dataset_ops.Dataset.from_tensor_slices(rt)


def _make_dict_ds(nrows):
  """Create a test set with various element shapes."""
  def transform(x):
    return {
        'shape=[]': ops.convert_to_tensor(x),
        'shape=[x]': math_ops.range(x),
        'shape=[x, 2]': array_ops.fill([x, 2], x),
        'shape=[2, x]': array_ops.fill([2, x], x),
        'shape=[2, x, 3, 2x, 4]': array_ops.fill([2, x, 3, 2*x, 4], x)
    }
  return _make_scalar_ds(nrows).map(transform)


def _make_tuple_ds(nrows):
  """Create a test set with various element shapes."""
  def transform(x):
    return (ops.convert_to_tensor(x),
            math_ops.range(x),
            array_ops.fill([x, 2], x))
  return _make_scalar_ds(nrows).map(transform)


def _to_list(v):
  return v.to_list() if hasattr(v, 'to_list') else v.tolist()


class RaggedBatchTest(test_base.DatasetTestBase, parameterized.TestCase):

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(
              make_dataset=[
                  _make_scalar_ds, _make_vector_ds, _make_matrix_ds1,
                  _make_matrix_ds2, _make_ragged_ds, _make_5dtensor_ds,
                  _make_dict_ds, _make_tuple_ds, _make_matrix_ds_fully_defined,
              ],
              nrows=[0, 20, 23],
              batch_size=[4],
              drop_remainder=[True, False])))
  def testRaggedBatchDataset(self, make_dataset, nrows, batch_size,
                             drop_remainder):
    dataset = make_dataset(nrows)

    # Get the unbatched rows (so we can check expected values).
    get_next = self.getNext(dataset)
    rows = [nest.map_structure(_to_list, self.evaluate(get_next()))
            for _ in range(nrows)]

    # Batch the dataset, and check that batches match slices from `rows`.
    batched_dataset = dataset.apply(
        batching.dense_to_ragged_batch(batch_size, drop_remainder))
    get_next = self.getNext(batched_dataset)
    for start_row in range(0, nrows, batch_size):
      end_row = start_row + batch_size
      if end_row > nrows and drop_remainder:
        break
      end_row = min(end_row, nrows)
      result = self.evaluate(get_next())

      # Use nest for potentially nested datasets.
      nest.map_structure_up_to(
          result, lambda a, *b: self.assertAllEqual(a, list(b)),
          result, *rows[start_row:end_row])

    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())

  def testRaggedBatchDatasetWithStructuredElements(self):
    nrows = 20
    batch_size = 4

    def make_structure(x):
      return {
          'dense':
              array_ops.fill([x], x),
          'ragged':
              ragged_concat_ops.stack(
                  [array_ops.stack([x]),
                   array_ops.stack([x, x])]),
          'sparse':
              sparse_tensor.SparseTensor([[x]], [x], [100])
      }

    dataset = dataset_ops.Dataset.from_tensor_slices(np.arange(nrows))
    dataset = dataset.map(make_structure)
    dataset = dataset.apply(batching.dense_to_ragged_batch(batch_size))
    get_next = self.getNext(dataset)

    for i in range(0, nrows, batch_size):
      result = self.evaluate(get_next())
      rows = range(i, i + batch_size)
      self.assertAllEqual(result['dense'], [[r] * r for r in rows])
      self.assertAllEqual(result['ragged'], [[[r], [r, r]] for r in rows])
      self.assertAllEqual(result['sparse'].indices, list(enumerate(rows)))
      self.assertAllEqual(result['sparse'].values, rows)
      self.assertAllEqual(result['sparse'].dense_shape, [4, 100])


if __name__ == '__main__':
  test.main()
