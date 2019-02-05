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
"""Tests for `tf.data.Dataset.from_tensor_slices()."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test


@test_util.run_all_in_graph_and_eager_modes
class FromTensorSlicesTest(test_base.DatasetTestBase):

  def testFromTensorSlices(self):
    """Test a dataset that represents the slices from a tuple of tensors."""
    components = (
        np.tile(np.array([[1], [2], [3], [4]]), 20), np.tile(
            np.array([[12], [13], [14], [15]]), 22),
        np.array([37.0, 38.0, 39.0, 40.0])
    )

    dataset = dataset_ops.Dataset.from_tensor_slices(components)
    get_next = self.getNext(dataset)

    self.assertEqual([c.shape[1:] for c in components],
                     [shape for shape in dataset.output_shapes])

    for i in range(4):
      results = self.evaluate(get_next())
      for component, result_component in zip(components, results):
        self.assertAllEqual(component[i], result_component)
    with self.assertRaises(errors.OutOfRangeError):
      results = self.evaluate(get_next())

  def testFromTensorSlicesSparse(self):
    """Test a dataset that represents the slices from a tuple of tensors."""
    components = (sparse_tensor.SparseTensorValue(
        indices=np.array([[0, 0], [1, 0], [2, 0]]),
        values=np.array([0, 0, 0]),
        dense_shape=np.array([3, 1])),
                  sparse_tensor.SparseTensorValue(
                      indices=np.array([[0, 0], [1, 1], [2, 2]]),
                      values=np.array([1, 2, 3]),
                      dense_shape=np.array([3, 3])))

    dataset = dataset_ops.Dataset.from_tensor_slices(components)

    self.assertEqual(
        [tensor_shape.TensorShape(c.dense_shape[1:]) for c in components],
        [shape for shape in dataset.output_shapes])

    expected = [
        (sparse_tensor.SparseTensorValue(
            indices=np.array([[0]]),
            values=np.array([0]),
            dense_shape=np.array([1])),
         sparse_tensor.SparseTensorValue(
             indices=np.array([[0]]),
             values=np.array([1]),
             dense_shape=np.array([3]))),
        (sparse_tensor.SparseTensorValue(
            indices=np.array([[0]]),
            values=np.array([0]),
            dense_shape=np.array([1])),
         sparse_tensor.SparseTensorValue(
             indices=np.array([[1]]),
             values=np.array([2]),
             dense_shape=np.array([3]))),
        (sparse_tensor.SparseTensorValue(
            indices=np.array([[0]]),
            values=np.array([0]),
            dense_shape=np.array([1])),
         sparse_tensor.SparseTensorValue(
             indices=np.array([[2]]),
             values=np.array([3]),
             dense_shape=np.array([3]))),
    ]
    self.assertDatasetProduces(dataset, expected_output=expected)

  def testFromTensorSlicesMixed(self):
    """Test a dataset that represents the slices from a tuple of tensors."""
    components = (np.tile(np.array([[1], [2], [3]]), 20),
                  np.tile(np.array([[12], [13], [14]]), 22),
                  np.array([37.0, 38.0, 39.0]),
                  sparse_tensor.SparseTensorValue(
                      indices=np.array([[0, 0], [1, 0], [2, 0]]),
                      values=np.array([0, 0, 0]),
                      dense_shape=np.array([3, 1])),
                  sparse_tensor.SparseTensorValue(
                      indices=np.array([[0, 0], [1, 1], [2, 2]]),
                      values=np.array([1, 2, 3]),
                      dense_shape=np.array([3, 3])))

    dataset = dataset_ops.Dataset.from_tensor_slices(components)
    get_next = self.getNext(dataset)
    self.assertEqual([
        tensor_shape.TensorShape(c.dense_shape[1:])
        if sparse_tensor.is_sparse(c) else c.shape[1:] for c in components
    ], [shape for shape in dataset.output_shapes])

    expected = [
        (sparse_tensor.SparseTensorValue(
            indices=np.array([[0]]),
            values=np.array([0]),
            dense_shape=np.array([1])),
         sparse_tensor.SparseTensorValue(
             indices=np.array([[0]]),
             values=np.array([1]),
             dense_shape=np.array([3]))),
        (sparse_tensor.SparseTensorValue(
            indices=np.array([[0]]),
            values=np.array([0]),
            dense_shape=np.array([1])),
         sparse_tensor.SparseTensorValue(
             indices=np.array([[1]]),
             values=np.array([2]),
             dense_shape=np.array([3]))),
        (sparse_tensor.SparseTensorValue(
            indices=np.array([[0]]),
            values=np.array([0]),
            dense_shape=np.array([1])),
         sparse_tensor.SparseTensorValue(
             indices=np.array([[2]]),
             values=np.array([3]),
             dense_shape=np.array([3]))),
    ]
    for i in range(3):
      results = self.evaluate(get_next())
      for component, result_component in zip(
          (list(zip(*components[:3]))[i] + expected[i]), results):
        if sparse_tensor.is_sparse(component):
          self.assertSparseValuesEqual(component, result_component)
        else:
          self.assertAllEqual(component, result_component)
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())

  def testFromTensorSlicesWithDict(self):
    components = {"foo": [1, 2, 3], "bar": [[4.0], [5.0], [6.0]]}
    dataset = dataset_ops.Dataset.from_tensor_slices(components)
    get_next = self.getNext(dataset)

    self.assertEqual(dtypes.int32, dataset.output_types["foo"])
    self.assertEqual(dtypes.float32, dataset.output_types["bar"])
    self.assertEqual((), dataset.output_shapes["foo"])
    self.assertEqual((1,), dataset.output_shapes["bar"])

    for i in range(3):
      results = self.evaluate(get_next())
      self.assertEqual(components["foo"][i], results["foo"])
      self.assertEqual(components["bar"][i], results["bar"])
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())

if __name__ == "__main__":
  test.main()
