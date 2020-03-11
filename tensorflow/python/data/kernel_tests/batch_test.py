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
"""Tests for `tf.data.Dataset.batch()`."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np

from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import nest
from tensorflow.python.framework import combinations
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import ragged_concat_ops
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_math_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import test


class BatchTest(test_base.DatasetTestBase, parameterized.TestCase):

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(
              count=[0, 28], batch_size=[14, 15], drop_remainder=[True,
                                                                  False])))
  def testBasic(self, count, batch_size, drop_remainder):
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

    def _map_fn(x, y, z):
      return math_ops.square(x), math_ops.square(y), math_ops.square(z)

    dataset = dataset_ops.Dataset.from_tensor_slices(components).map(
        _map_fn).repeat(count).batch(batch_size, drop_remainder)
    get_next = self.getNext(dataset)

    if drop_remainder:
      dim0 = batch_size
    else:
      dim0 = None
    self.assertEqual(
        [ts.as_list() for ts in nest.flatten(
            dataset_ops.get_legacy_output_shapes(dataset))],
        [[dim0] + list(c.shape[1:]) for c in components])

    num_full_batches = (count * 7) // batch_size
    for i in range(num_full_batches):
      result = self.evaluate(get_next())
      for component, result_component in zip(components, result):
        for j in range(batch_size):
          self.assertAllEqual(component[(i * batch_size + j) % 7]**2,
                              result_component[j])
    if not drop_remainder and (count * 7) % batch_size > 0:
      result = self.evaluate(get_next())
      for component, result_component in zip(components, result):
        for j in range((count * 7) % batch_size):
          self.assertAllEqual(
              component[(num_full_batches * batch_size + j) % 7]**2,
              result_component[j])
    with self.assertRaises(errors.OutOfRangeError):
      result = self.evaluate(get_next())

  @combinations.generate(test_base.default_test_combinations())
  def testInvalidBatchSize(self):
    with self.assertRaises(errors.InvalidArgumentError):
      dataset = (dataset_ops.Dataset.range(10).batch(0))
      self.evaluate(dataset._variant_tensor)

  @combinations.generate(test_base.default_test_combinations())
  def testDataset(self):

    def map_fn(i):
      return dataset_ops.Dataset.from_tensors(i)

    dataset = dataset_ops.Dataset.range(10).map(map_fn).batch(5)
    dataset = dataset.map(lambda x: x)
    dataset = dataset.unbatch().flat_map(lambda x: x)
    self.assertDatasetProduces(dataset, expected_output=range(10))

  def testSparse(self):

    def _sparse(i):
      return sparse_tensor.SparseTensorValue(
          indices=[[0]], values=(i * [1]), dense_shape=[1])

    dataset = dataset_ops.Dataset.range(10).map(_sparse).batch(5)
    expected_output = [
        sparse_tensor.SparseTensorValue(
            indices=[[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]],
            values=[i * 5, i * 5 + 1, i * 5 + 2, i * 5 + 3, i * 5 + 4],
            dense_shape=[5, 1]) for i in range(2)
    ]
    self.assertDatasetProduces(dataset, expected_output=expected_output)

  @combinations.generate(test_base.default_test_combinations())
  def testSparseWithDifferentDenseShapes(self):

    def _sparse(i):
      return sparse_tensor.SparseTensorValue(
          indices=array_ops.expand_dims(
              math_ops.range(i, dtype=dtypes.int64), 1),
          values=array_ops.fill([math_ops.cast(i, dtypes.int32)], i),
          dense_shape=[i])

    dataset = dataset_ops.Dataset.range(10).map(_sparse).batch(5)
    expected_output = []
    for i in range(2):
      expected_indices = []
      expected_outputs = []
      for j in range(5):
        for k in range(i * 5 + j):
          expected_indices.append([j, k])
          expected_outputs.append(i * 5 + j)
      expected_output.append(
          sparse_tensor.SparseTensorValue(
              indices=expected_indices,
              values=expected_outputs,
              dense_shape=[5, (i + 1) * 5 - 1]))
    self.assertDatasetProduces(dataset, expected_output=expected_output)

  @combinations.generate(test_base.default_test_combinations())
  def testSparseNested(self):

    def _sparse(i):
      return sparse_tensor.SparseTensorValue(
          indices=[[0]], values=(i * [1]), dense_shape=[1])

    dataset = dataset_ops.Dataset.range(10).map(_sparse).batch(5).batch(2)
    expected_output = [
        sparse_tensor.SparseTensorValue(
            indices=[[0, 0, 0], [0, 1, 0], [0, 2, 0], [0, 3, 0], [0, 4, 0],
                     [1, 0, 0], [1, 1, 0], [1, 2, 0], [1, 3, 0], [1, 4, 0]],
            values=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            dense_shape=[2, 5, 1])
    ]
    self.assertDatasetProduces(dataset, expected_output=expected_output)

  @combinations.generate(test_base.default_test_combinations())
  def testShapeError(self):

    def generator():
      yield [1.0, 2.0, 3.0]
      yield [4.0, 5.0, 6.0]
      yield [7.0, 8.0, 9.0, 10.0]

    dataset = (
        dataset_ops.Dataset.from_generator(
            generator, dtypes.float32, output_shapes=[None]).batch(3))
    self.assertDatasetProduces(
        dataset,
        expected_error=(
            errors.InvalidArgumentError,
            r'Cannot batch tensors with different shapes in component 0. First '
            r'element had shape \[3\] and element 2 had shape \[4\].'))

  @combinations.generate(test_base.default_test_combinations())
  def testRagged(self):

    def _ragged(i):
      return ragged_tensor.RaggedTensor.from_tensor(i * [[1]])

    dataset = dataset_ops.Dataset.range(10).map(_ragged).batch(5)
    expected_output = [
        ragged_factory_ops.constant([[[0]], [[1]], [[2]], [[3]], [[4]]]),
        ragged_factory_ops.constant([[[5]], [[6]], [[7]], [[8]], [[9]]])
    ]
    self.assertDatasetProduces(dataset, expected_output=expected_output)

  @combinations.generate(test_base.default_test_combinations())
  def testRaggedWithDifferentShapes(self):
    dataset = dataset_ops.Dataset.range(10).map(ragged_math_ops.range).batch(5)
    expected_output = [
        ragged_concat_ops.stack([ragged_math_ops.range(i) for i in range(5)]),
        ragged_concat_ops.stack(
            [ragged_math_ops.range(i) for i in range(5, 10)])
    ]
    self.assertDatasetProduces(dataset, expected_output=expected_output)

  @combinations.generate(test_base.default_test_combinations())
  def testRaggedNested(self):

    def _ragged(i):
      return ragged_tensor.RaggedTensor.from_tensor(i * [[1]])

    dataset = dataset_ops.Dataset.range(10).map(_ragged).batch(5).batch(2)
    expected_output = [
        ragged_factory_ops.constant([[[[0]], [[1]], [[2]], [[3]], [[4]]],
                                     [[[5]], [[6]], [[7]], [[8]], [[9]]]])
    ]
    self.assertDatasetProduces(dataset, expected_output=expected_output)

  @combinations.generate(test_base.default_test_combinations())
  def testNoneComponent(self):
    dataset = dataset_ops.Dataset.range(10).map(lambda x: (x, None)).batch(
        10).map(lambda x, y: x)
    self.assertDatasetProduces(dataset, expected_output=[list(range(10))])


if __name__ == '__main__':
  test.main()
