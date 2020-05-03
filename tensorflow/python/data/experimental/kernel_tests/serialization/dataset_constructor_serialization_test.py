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
"""Tests for the dataset constructors serialization."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np

from tensorflow.python.data.experimental.kernel_tests.serialization import dataset_serialization_test_base
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import combinations
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.platform import test


class FromTensorsSerializationTest(
    dataset_serialization_test_base.DatasetSerializationTestBase,
    parameterized.TestCase):

  def _build_tensor_dataset(self, variable_array):
    components = (variable_array, np.array([1, 2, 3]), np.array(37.0))

    return dataset_ops.Dataset.from_tensors(components)

  @combinations.generate(test_base.default_test_combinations())
  def testFromTensorsCore(self):
    # Equal length components
    arr = np.array(1)
    num_outputs = 1
    self.run_core_tests(lambda: self._build_tensor_dataset(arr),
                        num_outputs)


class FromTensorSlicesSerializationTest(
    dataset_serialization_test_base.DatasetSerializationTestBase,
    parameterized.TestCase):

  def _build_tensor_slices_dataset(self, components):
    return dataset_ops.Dataset.from_tensor_slices(components)

  @combinations.generate(test_base.default_test_combinations())
  def testFromTensorSlicesCore(self):
    # Equal length components
    components = (np.tile(np.array([[1], [2], [3], [4]]), 20),
                  np.tile(np.array([[12], [13], [14], [15]]), 22),
                  np.array([37.0, 38.0, 39.0, 40.0]))

    dict_components = {"foo": [1, 2, 3], "bar": [[4.0], [5.0], [6.0]]}

    self.run_core_tests(lambda: self._build_tensor_slices_dataset(components),
                        4)
    self.run_core_tests(
        lambda: self._build_tensor_slices_dataset(dict_components), 3)


class FromSparseTensorSlicesSerializationTest(
    dataset_serialization_test_base.DatasetSerializationTestBase,
    parameterized.TestCase):

  def _build_sparse_tensor_slice_dataset(self, slices):
    indices = np.array(
        [[i, j] for i in range(len(slices)) for j in range(len(slices[i]))],
        dtype=np.int64)
    values = np.array([val for s in slices for val in s], dtype=np.float64)
    dense_shape = np.array(
        [len(slices), max(len(s) for s in slices) + 1], dtype=np.int64)
    sparse_components = sparse_tensor.SparseTensor(indices, values, dense_shape)
    return dataset_ops.Dataset.from_sparse_tensor_slices(sparse_components)

  @combinations.generate(
      combinations.combine(
          tf_api_version=1,
          mode=["graph", "eager"]))
  def testFromSparseTensorSlicesCore(self):
    slices = [[1., 2., 3.], [1.], [1.], [1., 2.], [], [1., 2.], [], [], []]

    self.run_core_tests(
        lambda: self._build_sparse_tensor_slice_dataset(slices),
        9,
        sparse_tensors=True)


if __name__ == "__main__":
  test.main()
