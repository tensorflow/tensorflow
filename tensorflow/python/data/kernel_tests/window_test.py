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
"""Tests for `tf.data.Dataset.window()`."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np

from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import nest
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


@test_util.run_all_in_graph_and_eager_modes
class WindowTest(test_base.DatasetTestBase, parameterized.TestCase):

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
      ("12", 20, 14, 7, 1, False),
      ("13", 20, 17, 9, 1, False),
      ("14", 20, 14, 14, 1, False),
      ("15", 20, 10, 14, 1, False),
      ("16", 20, 14, 19, 1, False),
      ("17", 20, 4, 1, 2, False),
      ("18", 20, 2, 1, 6, False),
      ("19", 20, 4, 7, 2, False),
      ("20", 20, 2, 7, 6, False),
      ("21", 1, 10, 4, 1, False),
      ("22", 0, 10, 4, 1, False),
  )
  def testWindowDataset(self, count, size, shift, stride, drop_remainder=True):
    """Tests a dataset that slides a window its input elements."""
    components = (np.arange(7),
                  np.array([[1, 2, 3]]) * np.arange(7)[:, np.newaxis],
                  np.array(37.0) * np.arange(7))

    def _map_fn(x, y, z):
      return math_ops.square(x), math_ops.square(y), math_ops.square(z)

    def _flat_map_fn(x, y, z):
      return dataset_ops.Dataset.zip((x.batch(batch_size=size),
                                      y.batch(batch_size=size),
                                      z.batch(batch_size=size)))

    dataset = dataset_ops.Dataset.from_tensor_slices(components).map(
        _map_fn).repeat(count).window(
            size=size,
            shift=shift,
            stride=stride,
            drop_remainder=drop_remainder).flat_map(_flat_map_fn)
    get_next = self.getNext(dataset)

    self.assertEqual([[None] + list(c.shape[1:]) for c in components],
                     [ts.as_list() for ts in nest.flatten(
                         dataset_ops.get_legacy_output_shapes(dataset))])

    num_full_batches = max(0,
                           (count * 7 - ((size - 1) * stride + 1)) // shift + 1)
    for i in range(num_full_batches):
      result = self.evaluate(get_next())
      for component, result_component in zip(components, result):
        for j in range(size):
          self.assertAllEqual(component[(i * shift + j * stride) % 7]**2,
                              result_component[j])
    if not drop_remainder:
      num_partial_batches = (count * 7) // shift + (
          (count * 7) % shift > 0) - num_full_batches
      for i in range(num_partial_batches):
        result = self.evaluate(get_next())
        for component, result_component in zip(components, result):
          remaining = (count * 7) - ((num_full_batches + i) * shift)
          num_elements = remaining // stride + ((remaining % stride) > 0)
          for j in range(num_elements):
            self.assertAllEqual(
                component[((num_full_batches + i) * shift + j * stride) % 7]**2,
                result_component[j])
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())

  @parameterized.named_parameters(
      ("1", 14, 0, 3, 1),
      ("2", 14, 3, 0, 1),
      ("3", 14, 3, 3, 0),
  )
  def testWindowDatasetInvalid(self, count, size, shift, stride):
    with self.assertRaises(errors.InvalidArgumentError):
      ds = dataset_ops.Dataset.range(10).map(lambda x: x).repeat(count).window(
          size=size, shift=shift,
          stride=stride).flat_map(lambda x: x.batch(batch_size=size))
      self.evaluate(ds._variant_tensor)

  def testWindowDifferentNestedStructures(self):
    ds = dataset_ops.Dataset.from_tensor_slices(([1, 2], [3, 4])).window(2)
    self.getNext(ds)
    ds = dataset_ops.Dataset.from_tensor_slices({"a": [1, 2]}).window(2)
    self.getNext(ds)

  def testWindowSparse(self):

    def _sparse(i):
      return sparse_tensor.SparseTensorValue(
          indices=[[0]], values=(i * [1]), dense_shape=[1])

    dataset = dataset_ops.Dataset.range(10).map(_sparse).window(
        size=5, shift=3,
        drop_remainder=True).flat_map(lambda x: x.batch(batch_size=5))

    num_batches = (10 - 5) // 3 + 1
    expected_output = [
        sparse_tensor.SparseTensorValue(
            indices=[[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]],
            values=[i * 3, i * 3 + 1, i * 3 + 2, i * 3 + 3, i * 3 + 4],
            dense_shape=[5, 1]) for i in range(num_batches)
    ]
    self.assertDatasetProduces(dataset, expected_output=expected_output)

  def testWindowSparseWithDifferentDenseShapes(self):

    def _sparse(i):
      return sparse_tensor.SparseTensorValue(
          indices=array_ops.expand_dims(
              math_ops.range(i, dtype=dtypes.int64), 1),
          values=array_ops.fill([math_ops.cast(i, dtypes.int32)], i),
          dense_shape=[i])

    dataset = dataset_ops.Dataset.range(10).map(_sparse).window(
        size=5, shift=3,
        drop_remainder=True).flat_map(lambda x: x.batch(batch_size=5))

    expected_output = []
    num_batches = (10 - 5) // 3 + 1
    for i in range(num_batches):
      expected_indices = []
      expected_values = []
      for j in range(5):
        for k in range(i * 3 + j):
          expected_indices.append([j, k])
          expected_values.append(i * 3 + j)
      expected_output.append(
          sparse_tensor.SparseTensorValue(
              indices=expected_indices,
              values=expected_values,
              dense_shape=[5, i * 3 + 5 - 1]))
    self.assertDatasetProduces(dataset, expected_output=expected_output)

  def testNestedWindowSparse(self):

    def _sparse(i):
      return sparse_tensor.SparseTensorValue(
          indices=[[0]], values=(i * [1]), dense_shape=[1])

    dataset = dataset_ops.Dataset.range(10).map(_sparse).window(
        size=4, shift=2,
        drop_remainder=True).flat_map(lambda x: x.batch(batch_size=4)).window(
            size=3, shift=1,
            drop_remainder=True).flat_map(lambda x: x.batch(batch_size=3))

    expected_output = [
        sparse_tensor.SparseTensorValue(
            indices=[[0, 0, 0], [0, 1, 0], [0, 2, 0], [0, 3, 0], [1, 0, 0],
                     [1, 1, 0], [1, 2, 0], [1, 3, 0], [2, 0, 0], [2, 1, 0],
                     [2, 2, 0], [2, 3, 0]],
            values=[0, 1, 2, 3, 2, 3, 4, 5, 4, 5, 6, 7],
            dense_shape=[3, 4, 1]),
        sparse_tensor.SparseTensorValue(
            indices=[[0, 0, 0], [0, 1, 0], [0, 2, 0], [0, 3, 0], [1, 0, 0],
                     [1, 1, 0], [1, 2, 0], [1, 3, 0], [2, 0, 0], [2, 1, 0],
                     [2, 2, 0], [2, 3, 0]],
            values=[2, 3, 4, 5, 4, 5, 6, 7, 6, 7, 8, 9],
            dense_shape=[3, 4, 1])
    ]
    self.assertDatasetProduces(dataset, expected_output=expected_output)

  def testWindowShapeError(self):

    def generator():
      yield [1.0, 2.0, 3.0]
      yield [4.0, 5.0, 6.0]
      yield [7.0, 8.0, 9.0, 10.0]

    dataset = dataset_ops.Dataset.from_generator(
        generator, dtypes.float32, output_shapes=[None]).window(
            size=3, shift=1).flat_map(lambda x: x.batch(batch_size=3))
    self.assertDatasetProduces(
        dataset,
        expected_error=(
            errors.InvalidArgumentError,
            r"Cannot batch tensors with different shapes in component 0. "
            r"First element had shape \[3\] and element 2 had shape \[4\]."))

  def testWindowIgnoreErrors(self):
    input_values = np.float32([1., np.nan, 2., np.nan, 3.])
    dataset = dataset_ops.Dataset.from_tensor_slices(input_values).map(
        lambda x: array_ops.check_numerics(x, "message")).window(
            size=2, shift=2, stride=2,
            drop_remainder=True).flat_map(lambda x: x.batch(batch_size=2))
    self.assertDatasetProduces(
        dataset, expected_output=[np.float32([1., 2.]),
                                  np.float32([2., 3.])])

  def testNestedOutput(self):
    if not context.executing_eagerly():
      self.skipTest("self.evaluate() does not work with a dataset")
    dataset = dataset_ops.Dataset.range(100)
    dataset = dataset_ops.Dataset.zip((dataset, dataset)).window(10)
    for i, nested_dataset in enumerate(dataset):
      x, y = nested_dataset
      self.assertDatasetProduces(x, range(i*10, (i+1)*10))
      self.assertDatasetProduces(y, range(i*10, (i+1)*10))


if __name__ == "__main__":
  test.main()
