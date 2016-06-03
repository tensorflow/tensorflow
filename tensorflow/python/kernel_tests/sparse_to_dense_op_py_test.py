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
import tensorflow as tf


def _SparseToDense(sparse_indices, output_size, sparse_values,
                   default_value, validate_indices=True):
  return tf.sparse_to_dense(sparse_indices, output_size,
                            sparse_values,
                            default_value=default_value,
                            validate_indices=validate_indices)


class SparseToDenseTest(tf.test.TestCase):

  def testInt(self):
    with self.test_session(use_gpu=False):
      tf_ans = _SparseToDense([1, 3], [5], 1, 0).eval()
    np_ans = np.array([0, 1, 0, 1, 0]).astype(np.int32)
    self.assertAllClose(np_ans, tf_ans)

  def testFloat(self):
    with self.test_session(use_gpu=False):
      tf_ans = _SparseToDense([1, 3], [5], 1.0, 0.0).eval()
    np_ans = np.array([0, 1, 0, 1, 0]).astype(np.float32)
    self.assertAllClose(np_ans, tf_ans)

  def testString(self):
    with self.test_session(use_gpu=False):
      tf_ans = _SparseToDense([1, 3], [5], "a", "b").eval()
    np_ans = np.array(["b", "a", "b", "a", "b"]).astype(np.string_)
    self.assertAllEqual(np_ans, tf_ans)

  def testSetValue(self):
    with self.test_session(use_gpu=False):
      tf_ans = _SparseToDense([1, 3], [5], [1, 2], -1).eval()
    np_ans = np.array([-1, 1, -1, 2, -1]).astype(np.int32)
    self.assertAllClose(np_ans, tf_ans)

  def testSetSingleValue(self):
    with self.test_session(use_gpu=False):
      tf_ans = _SparseToDense([1, 3], [5], 1, -1).eval()
    np_ans = np.array([-1, 1, -1, 1, -1]).astype(np.int32)
    self.assertAllClose(np_ans, tf_ans)

  def test2d(self):
    # pylint: disable=bad-whitespace
    with self.test_session(use_gpu=False):
      tf_ans = _SparseToDense([[1, 3], [2, 0]], [3, 4], 1, -1).eval()
    np_ans = np.array([[-1, -1, -1, -1],
                       [-1, -1, -1,  1],
                       [ 1, -1, -1, -1]]).astype(np.int32)
    self.assertAllClose(np_ans, tf_ans)

  def testZeroDefault(self):
    with self.test_session():
      x = tf.sparse_to_dense(2, [4], 7).eval()
      self.assertAllEqual(x, [0, 0, 7, 0])

  def test3d(self):
    with self.test_session(use_gpu=False):
      tf_ans = _SparseToDense([[1, 3, 0], [2, 0, 1]], [3, 4, 2], 1, -1).eval()
    np_ans = np.ones((3, 4, 2), dtype=np.int32) * -1
    np_ans[1, 3, 0] = 1
    np_ans[2, 0, 1] = 1
    self.assertAllClose(np_ans, tf_ans)

  def testBadShape(self):
    with self.test_session():
      with self.assertRaisesWithPredicateMatch(
          ValueError, lambda e: ("Input shape should be a vector" == str(e))):
        _SparseToDense([1, 3], [[5], [3]], 1, -1)

  def testBadValue(self):
    with self.test_session():
      dense = _SparseToDense([1, 3], [5], [[5], [3]], -1)
      with self.assertRaisesOpError(
          r"sparse_values has incorrect shape \[2,1\], "
          r"should be \[\] or \[2\]"):
        dense.eval()

  def testBadNumValues(self):
    with self.test_session():
      dense = _SparseToDense([1, 3], [5], [1, 2, 3], -1)
      with self.assertRaisesOpError(
          r"sparse_values has incorrect shape \[3\], should be \[\] or \[2\]"):
        dense.eval()

  def testBadDefault(self):
    with self.test_session():
      dense = _SparseToDense([1, 3], [5], [1, 2], [0])
      with self.assertRaisesOpError("default_value should be a scalar"):
        dense.eval()

  def testOutOfBoundsIndicesWithWithoutValidation(self):
    with self.test_session():
      dense = _SparseToDense(
          sparse_indices=[[1], [10]], output_size=[5],
          sparse_values=[-1.0, 1.0], default_value=0.0)
      with self.assertRaisesOpError(
          r"indices\[1\] = \[10\] is out of bounds: need 0 <= index < \[5\]"):
        dense.eval()
      # Disable checks, the allocation should still fail.
      with self.assertRaisesOpError("out of bounds"):
        dense_without_validation = _SparseToDense(
            sparse_indices=[[1], [10]], output_size=[5],
            sparse_values=[-1.0, 1.0], default_value=0.0,
            validate_indices=False)
        dense_without_validation.eval()

  def testRepeatingIndicesWithWithoutValidation(self):
    with self.test_session():
      dense = _SparseToDense(
          sparse_indices=[[1], [1]], output_size=[5],
          sparse_values=[-1.0, 1.0], default_value=0.0)
      with self.assertRaisesOpError(r"indices\[1\] = \[1\] is repeated"):
        dense.eval()
      # Disable checks
      dense_without_validation = _SparseToDense(
          sparse_indices=[[1], [1]], output_size=[5],
          sparse_values=[-1.0, 1.0], default_value=0.0, validate_indices=False)
      dense_without_validation.eval()

  def testUnsortedIndicesWithWithoutValidation(self):
    with self.test_session():
      dense = _SparseToDense(
          sparse_indices=[[2], [1]], output_size=[5],
          sparse_values=[-1.0, 1.0], default_value=0.0)
      with self.assertRaisesOpError(r"indices\[1\] = \[1\] is out of order"):
        dense.eval()
      # Disable checks
      dense_without_validation = _SparseToDense(
          sparse_indices=[[2], [1]], output_size=[5],
          sparse_values=[-1.0, 1.0], default_value=0.0, validate_indices=False)
      dense_without_validation.eval()

  def testShapeInferenceKnownShape(self):
    with self.test_session(use_gpu=False):
      indices = tf.placeholder(tf.int64)

      shape = [4, 5, 6]
      output = tf.sparse_to_dense(indices, shape, 1, 0)
      self.assertEqual(output.get_shape(), [4, 5, 6])

      shape = tf.placeholder(tf.int64, shape=(3,))
      output = tf.sparse_to_dense(indices, shape, 1, 0)
      self.assertEqual(output.get_shape().as_list(), [None, None, None])

  def testShapeInferenceUnknownShape(self):
    with self.test_session(use_gpu=False):
      indices = tf.placeholder(tf.int64)
      shape = tf.placeholder(tf.int64)
      output = tf.sparse_to_dense(indices, shape, 1, 0)
      self.assertEqual(output.get_shape().ndims, None)


if __name__ == "__main__":
  tf.test.main()
