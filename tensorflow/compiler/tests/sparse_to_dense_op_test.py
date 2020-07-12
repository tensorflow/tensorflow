# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tensorflow.kernels.sparse_op."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.platform import test


def _SparseToDense(sparse_indices,
                   output_size,
                   sparse_values,
                   default_value,
                   validate_indices=True):
  feed_sparse_indices = array_ops.placeholder(dtypes.int32)
  feed_dict = {feed_sparse_indices: sparse_indices}
  return sparse_ops.sparse_to_dense(
      feed_sparse_indices,
      output_size,
      sparse_values,
      default_value=default_value,
      validate_indices=validate_indices).eval(feed_dict=feed_dict)


class SparseToDenseTest(xla_test.XLATestCase):

  def testInt(self):
    with self.session(), self.test_scope():
      tf_ans = _SparseToDense([1, 3], [5], 1, 0)
    np_ans = np.array([0, 1, 0, 1, 0]).astype(np.int32)
    self.assertAllClose(np_ans, tf_ans)

  def testFloat(self):
    with self.session(), self.test_scope():
      tf_ans = _SparseToDense([1, 3], [5], 1.0, 0.0)
    np_ans = np.array([0, 1, 0, 1, 0]).astype(np.float32)
    self.assertAllClose(np_ans, tf_ans)

  def testSetValue(self):
    with self.session(), self.test_scope():
      tf_ans = _SparseToDense([1, 3], [5], [1, 2], -1)
    np_ans = np.array([-1, 1, -1, 2, -1]).astype(np.int32)
    self.assertAllClose(np_ans, tf_ans)

  def testSetSingleValue(self):
    with self.session(), self.test_scope():
      tf_ans = _SparseToDense([1, 3], [5], 1, -1)
    np_ans = np.array([-1, 1, -1, 1, -1]).astype(np.int32)
    self.assertAllClose(np_ans, tf_ans)

  def test2d(self):
    # pylint: disable=bad-whitespace
    with self.session(), self.test_scope():
      tf_ans = _SparseToDense([[1, 3], [2, 0]], [3, 4], 1, -1)
    np_ans = np.array([[-1, -1, -1, -1],
                       [-1, -1, -1,  1],
                       [ 1, -1, -1, -1]]).astype(np.int32)
    self.assertAllClose(np_ans, tf_ans)

  def testZeroDefault(self):
    with self.session():
      x = sparse_ops.sparse_to_dense(2, [4], 7).eval()
      self.assertAllEqual(x, [0, 0, 7, 0])

  def test3d(self):
    with self.session(), self.test_scope():
      tf_ans = _SparseToDense([[1, 3, 0], [2, 0, 1]], [3, 4, 2], 1, -1)
    np_ans = np.ones((3, 4, 2), dtype=np.int32) * -1
    np_ans[1, 3, 0] = 1
    np_ans[2, 0, 1] = 1
    self.assertAllClose(np_ans, tf_ans)

  def testDegenerateIndexMatrix(self):
    with self.session(), self.test_scope():
      tf_ans = _SparseToDense([[2], [3], [4], [5], [6], [7], [8], [9]], [10],
                              [1, 2, 3, 4, 5, 6, 7, 8], -1)
    self.assertAllClose([-1, -1, 1, 2, 3, 4, 5, 6, 7, 8], tf_ans)

  def testBadShape(self):
    with self.session(), self.test_scope():
      with self.assertRaisesWithPredicateMatch(ValueError, "must be rank 1"):
        _SparseToDense([1, 3], [[5], [3]], 1, -1)

  @test_util.disable_mlir_bridge("Error handling")
  def testBadValue(self):
    with self.session(), self.test_scope():
      with self.assertRaisesOpError(
          r"sparse_values has incorrect shape \[2,1\], "
          r"should be \[\] or \[2\]"):
        _SparseToDense([1, 3], [5], [[5], [3]], -1)

  @test_util.disable_mlir_bridge("Error handling")
  def testBadNumValues(self):
    with self.session(), self.test_scope():
      with self.assertRaisesOpError(
          r"sparse_values has incorrect shape \[3\], should be \[\] or \[2\]"):
        _SparseToDense([1, 3], [5], [1, 2, 3], -1)

  @test_util.disable_mlir_bridge("Error handling")
  def testBadDefault(self):
    with self.session(), self.test_scope():
      with self.assertRaisesOpError("default_value should be a scalar"):
        _SparseToDense([1, 3], [5], [1, 2], [0])


if __name__ == "__main__":
  test.main()
