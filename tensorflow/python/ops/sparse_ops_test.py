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
"""Tests for sparse ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.platform import googletest


@test_util.run_all_in_graph_and_eager_modes
class SparseOpsTest(test_util.TensorFlowTestCase, parameterized.TestCase):

  def testSparseEye(self):
    def test_one(n, m, as_tensors):
      expected = np.eye(n, m)
      if as_tensors:
        m = constant_op.constant(m)
        n = constant_op.constant(n)
      s = sparse_ops.sparse_eye(n, m)
      d = sparse_ops.sparse_to_dense(s.indices, s.dense_shape, s.values)
      self.assertAllEqual(self.evaluate(d), expected)

    for n in range(2, 10, 2):
      for m in range(2, 10, 2):
        # Test with n and m as both constants and tensors.
        test_one(n, m, True)
        test_one(n, m, False)

  def testSparseExpandDims(self):
    for rank in range(1, 4):
      # Create a dummy input. When rank=3, shape=[2, 4, 6].
      shape = np.arange(1, rank + 1) * 2
      before = np.arange(np.prod(shape)).reshape(shape)

      # Make entries sparse.
      before *= np.random.binomial(1, .2, before.shape)
      dense_shape = before.shape
      indices = np.array(np.where(before)).T
      values = before[before != 0]

      # Try every possible valid value of axis.
      for axis in range(-rank - 1, rank):
        expected_after = np.expand_dims(before, axis)

        for axis_as_tensor in [False, True]:
          dense_shape_t = constant_op.constant(dense_shape, dtype=dtypes.int64)
          indices_t = constant_op.constant(indices)
          values_t = constant_op.constant(values)
          before_t = sparse_tensor.SparseTensor(
              indices=indices_t, values=values_t, dense_shape=dense_shape_t)

          if axis_as_tensor:
            axis = constant_op.constant(axis)

          s = sparse_ops.sparse_expand_dims(before_t, axis)
          d = sparse_ops.sparse_to_dense(s.indices, s.dense_shape, s.values)
          self.assertAllEqual(self.evaluate(d), expected_after)

  @parameterized.parameters([
      (math_ops.abs, [1.0, -1.0, 3.0, -4.0], [1.0, 1.0, 3.0, 4.0]),
      (math_ops.negative, [1.0, -1.0, 3.0, -4.0], [-1.0, 1.0, -3.0, 4.0]),
      (math_ops.sign, [3.0, -2.0, 0.0, -4.0], [1.0, -1.0, 0.0, -1.0]),
      (math_ops.square, [1.0, -1.0, 3.0, -4.0], [1.0, 1.0, 9.0, 16.0]),
  ])
  def testUnarySparseDispatch(self, op, values, expected):
    st = sparse_tensor.SparseTensor(
        indices=[[0, 0], [0, 1], [2, 0], [2, 4]],
        values=values,
        dense_shape=[3, 6])
    result = op(st)
    result_value = self.evaluate(result)
    self.assertAllEqual(result_value.indices, st.indices)
    self.assertAllEqual(result_value.values, expected)
    self.assertAllEqual(result_value.dense_shape, st.dense_shape)


if __name__ == '__main__':
  googletest.main()
