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
"""Tests for tensorflow.python.framework.composite_tensor_utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import composite_tensor_utils
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged import ragged_tensor_value
from tensorflow.python.platform import googletest


class CompositeTensorTest(test_util.TensorFlowTestCase):

  def test_is_composite(self):
    # Validate that all composite tensor and value types return true.
    self.assertTrue(
        composite_tensor_utils.is_composite_or_composite_value(
            sparse_tensor.SparseTensor([[0, 0]], [1], [1, 1])))
    self.assertTrue(
        composite_tensor_utils.is_composite_or_composite_value(
            sparse_tensor.SparseTensorValue([[0, 0]], [1], [1, 1])))
    self.assertTrue(
        composite_tensor_utils.is_composite_or_composite_value(
            ragged_tensor.RaggedTensor.from_row_splits([0, 1, 2], [0, 1, 3])))
    self.assertTrue(
        composite_tensor_utils.is_composite_or_composite_value(
            ragged_tensor_value.RaggedTensorValue(
                np.array([0, 1, 2]), np.array([0, 1, 3]))))

    # Test that numpy arrays and tensors return false.
    self.assertFalse(
        composite_tensor_utils.is_composite_or_composite_value(
            np.ndarray([0, 1])))
    self.assertFalse(
        composite_tensor_utils.is_composite_or_composite_value(
            ops.convert_to_tensor([3, 1])))

  def test_sparse_concatenation(self):
    tensor_1 = sparse_tensor.SparseTensor([[0, 0]], [1], [1, 1])
    tensor_2 = sparse_tensor.SparseTensor([[0, 0]], [2], [1, 1])
    concatenated_tensor = composite_tensor_utils.append_composite_tensor(
        tensor_1, tensor_2)
    evaluated_tensor = self.evaluate(concatenated_tensor)
    self.assertAllEqual(evaluated_tensor.indices, [[0, 0], [1, 0]])
    self.assertAllEqual(evaluated_tensor.values, [1, 2])
    self.assertAllEqual(evaluated_tensor.dense_shape, [2, 1])

  def test_sparse_value_concatenation(self):
    tensor_1 = sparse_tensor.SparseTensorValue([[0, 0]], [1], [1, 1])
    tensor_2 = sparse_tensor.SparseTensorValue([[0, 0]], [2], [1, 1])
    concatenated_tensor = composite_tensor_utils.append_composite_tensor(
        tensor_1, tensor_2)
    self.assertAllEqual(concatenated_tensor.indices, [[0, 0], [1, 0]])
    self.assertAllEqual(concatenated_tensor.values, [1, 2])
    self.assertAllEqual(concatenated_tensor.dense_shape, [2, 1])

  def test_ragged_concatenation(self):
    tensor_1 = ragged_tensor.RaggedTensor.from_row_splits(
        np.array([0, 1, 2]), np.array([0, 1, 3]))
    tensor_2 = ragged_tensor.RaggedTensor.from_row_splits(
        np.array([3, 4, 5]), np.array([0, 2, 3]))
    concatenated_tensor = composite_tensor_utils.append_composite_tensor(
        tensor_1, tensor_2)
    evaluated_tensor = self.evaluate(concatenated_tensor)

    self.assertAllEqual(evaluated_tensor.values, [0, 1, 2, 3, 4, 5])
    self.assertAllEqual(evaluated_tensor.row_splits, [0, 1, 3, 5, 6])

  def test_ragged_value_concatenation(self):
    tensor_1 = ragged_tensor_value.RaggedTensorValue(
        np.array([0, 1, 2]), np.array([0, 1, 3]))
    tensor_2 = ragged_tensor_value.RaggedTensorValue(
        np.array([3, 4, 5]), np.array([0, 2, 3]))
    concatenated_tensor = composite_tensor_utils.append_composite_tensor(
        tensor_1, tensor_2)

    self.assertAllEqual(concatenated_tensor.values, [0, 1, 2, 3, 4, 5])
    self.assertAllEqual(concatenated_tensor.row_splits, [0, 1, 3, 5, 6])


if __name__ == '__main__':
  googletest.main()
