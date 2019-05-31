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
"""Tests for the BatchDataset serialization."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.data.experimental.kernel_tests.serialization import dataset_serialization_test_base
from tensorflow.python.data.experimental.ops import batching
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


class BatchDatasetSerializationTest(
    dataset_serialization_test_base.DatasetSerializationTestBase):

  def build_dataset(self, multiplier=15.0, tensor_slice_len=2, batch_size=2):
    components = (
        np.arange(tensor_slice_len),
        np.array([[1, 2, 3]]) * np.arange(tensor_slice_len)[:, np.newaxis],
        np.array(multiplier) * np.arange(tensor_slice_len))

    return dataset_ops.Dataset.from_tensor_slices(components).batch(batch_size)

  def testCore(self):
    tensor_slice_len = 8
    batch_size = 2
    num_outputs = tensor_slice_len // batch_size
    self.run_core_tests(
        lambda: self.build_dataset(15.0, tensor_slice_len, batch_size),
        lambda: self.build_dataset(20.0, tensor_slice_len, batch_size),
        num_outputs)

  def _build_dataset_dense_to_sparse(self, components):
    return dataset_ops.Dataset.from_tensor_slices(components).map(
        lambda x: array_ops.fill([x], x)).apply(
            batching.dense_to_sparse_batch(4, [12]))

  def testDenseToSparseBatchDatasetCore(self):
    components = np.random.randint(5, size=(40,)).astype(np.int32)
    diff_comp = np.random.randint(2, size=(100,)).astype(np.int32)

    num_outputs = len(components) // 4
    self.run_core_tests(lambda: self._build_dataset_dense_to_sparse(components),
                        lambda: self._build_dataset_dense_to_sparse(diff_comp),
                        num_outputs)

  def _sparse(self, i):
    return sparse_tensor.SparseTensorValue(
        indices=[[0]], values=(i * [1]), dense_shape=[1])

  def _build_dataset_sparse(self, batch_size=5):
    return dataset_ops.Dataset.range(10).map(self._sparse).batch(batch_size)

  def testSparseCore(self):
    self.run_core_tests(self._build_dataset_sparse,
                        lambda: self._build_dataset_sparse(2), 2)

  def _build_dataset_nested_sparse(self):
    return dataset_ops.Dataset.range(10).map(self._sparse).batch(5).batch(2)

  def testNestedSparseCore(self):
    self.run_core_tests(self._build_dataset_nested_sparse, None, 1)


if __name__ == "__main__":
  test.main()
