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
  return sparse_ops.sparse_to_dense(
      sparse_indices,
      output_size,
      sparse_values,
      default_value=default_value,
      validate_indices=validate_indices)


class SparseToDenseTest(test.TestCase):

  @test_util.run_deprecated_v1
  def testInt(self):
    with self.session(use_gpu=False):
      tf_ans = _SparseToDense([1, 3], [5], 1, 0).eval()
    np_ans = np.array([0, 1, 0, 1, 0]).astype(np.int32)
    self.assertAllClose(np_ans, tf_ans)

  @test_util.run_deprecated_v1
  def testFloat(self):
    with self.session(use_gpu=False):
      tf_ans = _SparseToDense([1, 3], [5], 1.0, 0.0).eval()
    np_ans = np.array([0, 1, 0, 1, 0]).astype(np.float32)
    self.assertAllClose(np_ans, tf_ans)

  @test_util.run_deprecated_v1
  def testString(self):
    with self.session(use_gpu=False):
      tf_ans = _SparseToDense([1, 3], [5], "a", "b").eval()
    np_ans = np.array(["b", "a", "b", "a", "b"]).astype(np.string_)
    self.assertAllEqual(np_ans, tf_ans)

  @test_util.run_deprecated_v1
  def testSetValue(self):
    with self.session(use_gpu=False):
      tf_ans = _SparseToDense([1, 3], [5], [1, 2], -1).eval()
    np_ans = np.array([-1, 1, -1, 2, -1]).astype(np.int32)
    self.assertAllClose(np_ans, tf_ans)

  @test_util.run_deprecated_v1
  def testSetSingleValue(self):
    with self.session(use_gpu=False):
      tf_ans = _SparseToDense([1, 3], [5], 1, -1).eval()
    np_ans = np.array([-1, 1, -1, 1, -1]).astype(np.int32)
    self.assertAllClose(np_ans, tf_ans)

  @test_util.run_deprecated_v1
  def test2d(self):
    # pylint: disable=bad-whitespace
    with self.session(use_gpu=False):
      tf_ans = _SparseToDense([[1, 3], [2, 0]], [3, 4], 1, -1).eval()
    np_ans = np.array([[-1, -1, -1, -1],
                       [-1, -1, -1,  1],
                       [ 1, -1, -1, -1]]).astype(np.int32)
    self.assertAllClose(np_ans, tf_ans)

  @test_util.run_deprecated_v1
  def testZeroDefault(self):
    with self.cached_session():
      x = sparse_ops.sparse_to_dense(2, [4], 7).eval()
      self.assertAllEqual(x, [0, 0, 7, 0])

  @test_util.run_deprecated_v1
  def test3d(self):
    with self.session(use_gpu=False):
      tf_ans = _SparseToDense([[1, 3, 0], [2, 0, 1]], [3, 4, 2], 1, -1).eval()
    np_ans = np.ones((3, 4, 2), dtype=np.int32) * -1
    np_ans[1, 3, 0] = 1
    np_ans[2, 0, 1] = 1
    self.assertAllClose(np_ans, tf_ans)

  @test_util.run_deprecated_v1
  def testBadShape(self):
    with self.cached_session():
      with self.assertRaisesWithPredicateMatch(ValueError, "must be rank 1"):
        _SparseToDense([1, 3], [[5], [3]], 1, -1)

  @test_util.run_deprecated_v1
  def testBadValue(self):
    with self.cached_session():
      dense = _SparseToDense([1, 3], [5], [[5], [3]], -1)
      with self.assertRaisesOpError(
          r"sparse_values has incorrect shape \[2,1\], "
          r"should be \[\] or \[2\]"):
        self.evaluate(dense)

  @test_util.run_deprecated_v1
  def testBadNumValues(self):
    with self.cached_session():
      dense = _SparseToDense([1, 3], [5], [1, 2, 3], -1)
      with self.assertRaisesOpError(
          r"sparse_values has incorrect shape \[3\], should be \[\] or \[2\]"):
        self.evaluate(dense)

  @test_util.run_deprecated_v1
  def testBadDefault(self):
    with self.cached_session():
      dense = _SparseToDense([1, 3], [5], [1, 2], [0])
      with self.assertRaisesOpError("default_value should be a scalar"):
        self.evaluate(dense)

  @test_util.run_deprecated_v1
  def testOutOfBoundsIndicesWithWithoutValidation(self):
    with self.cached_session():
      dense = _SparseToDense(
          sparse_indices=[[1], [10]],
          output_size=[5],
          sparse_values=[-1.0, 1.0],
          default_value=0.0)
      with self.assertRaisesOpError(
          r"indices\[1\] = \[10\] is out of bounds: need 0 <= index < \[5\]"):
        self.evaluate(dense)
      # Disable checks, the allocation should still fail.
      with self.assertRaisesOpError("out of bounds"):
        dense_without_validation = _SparseToDense(
            sparse_indices=[[1], [10]],
            output_size=[5],
            sparse_values=[-1.0, 1.0],
            default_value=0.0,
            validate_indices=False)
        self.evaluate(dense_without_validation)

  @test_util.run_deprecated_v1
  def testRepeatingIndicesWithWithoutValidation(self):
    with self.cached_session():
      dense = _SparseToDense(
          sparse_indices=[[1], [1]],
          output_size=[5],
          sparse_values=[-1.0, 1.0],
          default_value=0.0)
      with self.assertRaisesOpError(r"indices\[1\] = \[1\] is repeated"):
        self.evaluate(dense)
      # Disable checks
      dense_without_validation = _SparseToDense(
          sparse_indices=[[1], [1]],
          output_size=[5],
          sparse_values=[-1.0, 1.0],
          default_value=0.0,
          validate_indices=False)
      self.evaluate(dense_without_validation)

  @test_util.run_deprecated_v1
  def testUnsortedIndicesWithWithoutValidation(self):
    with self.cached_session():
      dense = _SparseToDense(
          sparse_indices=[[2], [1]],
          output_size=[5],
          sparse_values=[-1.0, 1.0],
          default_value=0.0)
      with self.assertRaisesOpError(r"indices\[1\] = \[1\] is out of order"):
        self.evaluate(dense)
      # Disable checks
      dense_without_validation = _SparseToDense(
          sparse_indices=[[2], [1]],
          output_size=[5],
          sparse_values=[-1.0, 1.0],
          default_value=0.0,
          validate_indices=False)
      self.evaluate(dense_without_validation)

  @test_util.run_deprecated_v1
  def testShapeInferenceKnownShape(self):
    with self.session(use_gpu=False):
      indices = array_ops.placeholder(dtypes.int64)

      shape = [4, 5, 6]
      output = sparse_ops.sparse_to_dense(indices, shape, 1, 0)
      self.assertEqual(output.get_shape(), [4, 5, 6])

      shape = array_ops.placeholder(dtypes.int64, shape=(3,))
      output = sparse_ops.sparse_to_dense(indices, shape, 1, 0)
      self.assertEqual(output.get_shape().as_list(), [None, None, None])

  @test_util.run_deprecated_v1
  def testShapeInferenceUnknownShape(self):
    with self.session(use_gpu=False):
      indices = array_ops.placeholder(dtypes.int64)
      shape = array_ops.placeholder(dtypes.int64)
      output = sparse_ops.sparse_to_dense(indices, shape, 1, 0)
      self.assertEqual(output.get_shape().ndims, None)


if __name__ == "__main__":
  test.main()
