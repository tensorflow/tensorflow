# Copyright 2024 The TensorFlow Authors.
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
"""Tests for SparseConcat overflow handling."""

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.platform import test


class SparseConcatOverflowTest(test.TestCase):

  def testSparseConcatLargeShapeOverflow(self):
    indices1 = constant_op.constant(
        [[0, 0, 0], [1, 1, 0], [2, 0, 1]], dtype=dtypes.int64)
    values1 = constant_op.constant(["tensor", "flow", "test"])
    shape1 = constant_op.constant([5, 2, 2147483647], dtype=dtypes.int64)

    indices2 = constant_op.constant(
        [[0, 1, 0], [1, 0, 0], [2, 1, 1], [3, 0, 1]], dtype=dtypes.int64)
    values2 = constant_op.constant(["a", "b", "c", "d"])
    shape2 = constant_op.constant(
        [5, 1879048192, 536870912], dtype=dtypes.int64)

    sp1 = sparse_tensor.SparseTensor(indices1, values1, shape1)
    sp2 = sparse_tensor.SparseTensor(indices2, values2, shape2)

    with self.assertRaises(errors.InvalidArgumentError):
      sparse_tensor.sparse_concat(
          axis=1,
          sp_inputs=[sp1, sp2])


if __name__ == "__main__":
  test.main()
