# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for SparseReshape."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


class SparseReshapeTest(tf.test.TestCase):

  def _SparseTensorPlaceholder(self):
    return tf.SparseTensor(
        tf.placeholder(tf.int64), tf.placeholder(tf.float64),
        tf.placeholder(tf.int64))

  def _SparseTensorValue_5x6(self):
    ind = np.array([
        [0, 0], [1, 0], [1, 3], [1, 4], [3, 2], [3, 3]
    ]).astype(np.int64)
    val = np.array([0, 10, 13, 14, 32, 33]).astype(np.float64)

    shape = np.array([5, 6]).astype(np.int64)
    return tf.SparseTensorValue(ind, val, shape)

  def _SparseTensorValue_2x3x4(self):
    ind = np.array([
        [0, 0, 1], [0, 1, 0], [0, 1, 2], [1, 0, 3], [1, 1, 1], [1, 1, 3],
        [1, 2, 2]
    ])
    val = np.array([1, 10, 12, 103, 111, 113, 122])
    shape = np.array([2, 3, 4])
    return tf.SparseTensorValue(ind, val, shape)

  def testSameShape(self):
    with self.test_session(use_gpu=False) as sess:
      sp_input = self._SparseTensorPlaceholder()
      input_val = self._SparseTensorValue_5x6()
      sp_output = tf.sparse_reshape(sp_input, [5, 6])

      output_val = sess.run(sp_output, {sp_input: input_val})
      self.assertAllEqual(output_val.indices, input_val.indices)
      self.assertAllEqual(output_val.values, input_val.values)
      self.assertAllEqual(output_val.shape, input_val.shape)

  def testSameShapeWithInferredDim(self):
    with self.test_session(use_gpu=False) as sess:
      sp_input = self._SparseTensorPlaceholder()
      input_val = self._SparseTensorValue_5x6()
      sp_output = tf.sparse_reshape(sp_input, [-1, 6])

      output_val = sess.run(sp_output, {sp_input: input_val})
      self.assertAllEqual(output_val.indices, input_val.indices)
      self.assertAllEqual(output_val.values, input_val.values)
      self.assertAllEqual(output_val.shape, input_val.shape)

  def testNewShapeSameRank(self):
    with self.test_session(use_gpu=False) as sess:
      sp_input = self._SparseTensorPlaceholder()
      input_val = self._SparseTensorValue_5x6()
      sp_output = tf.sparse_reshape(sp_input, [3, 10])

      output_val = sess.run(sp_output, {sp_input: input_val})
      self.assertAllEqual(output_val.indices, np.array([
          [0, 0], [0, 6], [0, 9], [1, 0], [2, 0], [2, 1]
      ]))
      self.assertAllEqual(output_val.values, input_val.values)
      self.assertAllEqual(output_val.shape, [3, 10])

  def testNewShapeSameRankWithInferredDim(self):
    with self.test_session(use_gpu=False) as sess:
      sp_input = self._SparseTensorPlaceholder()
      input_val = self._SparseTensorValue_5x6()
      sp_output = tf.sparse_reshape(sp_input, [3, -1])

      output_val = sess.run(sp_output, {sp_input: input_val})
      self.assertAllEqual(output_val.indices, np.array([
          [0, 0], [0, 6], [0, 9], [1, 0], [2, 0], [2, 1]
      ]))
      self.assertAllEqual(output_val.values, input_val.values)
      self.assertAllEqual(output_val.shape, [3, 10])

  def testUpRank(self):
    with self.test_session(use_gpu=False) as sess:
      sp_input = self._SparseTensorPlaceholder()
      input_val = self._SparseTensorValue_5x6()
      sp_output = tf.sparse_reshape(sp_input, [2, 3, 5])

      output_val = sess.run(sp_output, {sp_input: input_val})
      self.assertAllEqual(output_val.indices, np.array([
          [0, 0, 0], [0, 1, 1], [0, 1, 4], [0, 2, 0], [1, 1, 0], [1, 1, 1]
      ]))
      self.assertAllEqual(output_val.values, input_val.values)
      self.assertAllEqual(output_val.shape, [2, 3, 5])

  def testUpRankWithInferredDim(self):
    with self.test_session(use_gpu=False) as sess:
      sp_input = self._SparseTensorPlaceholder()
      input_val = self._SparseTensorValue_5x6()
      sp_output = tf.sparse_reshape(sp_input, [2, -1, 5])

      output_val = sess.run(sp_output, {sp_input: input_val})
      self.assertAllEqual(output_val.indices, np.array([
          [0, 0, 0], [0, 1, 1], [0, 1, 4], [0, 2, 0], [1, 1, 0], [1, 1, 1]
      ]))
      self.assertAllEqual(output_val.values, input_val.values)
      self.assertAllEqual(output_val.shape, [2, 3, 5])

  def testDownRank(self):
    with self.test_session(use_gpu=False) as sess:
      sp_input = self._SparseTensorPlaceholder()
      input_val = self._SparseTensorValue_2x3x4()
      sp_output = tf.sparse_reshape(sp_input, [6, 4])

      output_val = sess.run(sp_output, {sp_input: input_val})
      self.assertAllEqual(output_val.indices, np.array([
          [0, 1], [1, 0], [1, 2], [3, 3], [4, 1], [4, 3], [5, 2]
      ]))
      self.assertAllEqual(output_val.values, input_val.values)
      self.assertAllEqual(output_val.shape, [6, 4])

  def testDownRankWithInferredDim(self):
    with self.test_session(use_gpu=False) as sess:
      sp_input = self._SparseTensorPlaceholder()
      input_val = self._SparseTensorValue_2x3x4()
      sp_output = tf.sparse_reshape(sp_input, [6, -1])

      output_val = sess.run(sp_output, {sp_input: input_val})
      self.assertAllEqual(output_val.indices, np.array([
          [0, 1], [1, 0], [1, 2], [3, 3], [4, 1], [4, 3], [5, 2]
      ]))
      self.assertAllEqual(output_val.values, input_val.values)
      self.assertAllEqual(output_val.shape, [6, 4])

  def testMultipleInferredDims(self):
    with self.test_session(use_gpu=False) as sess:
      sp_input = self._SparseTensorPlaceholder()
      input_val = self._SparseTensorValue_5x6()
      sp_output = tf.sparse_reshape(sp_input, [4, -1, -1])
      with self.assertRaisesOpError("only one output shape size may be -1"):
        sess.run(sp_output, {sp_input: input_val})

  def testMismatchedSizes(self):
    with self.test_session(use_gpu=False) as sess:
      sp_input = self._SparseTensorPlaceholder()
      input_val = self._SparseTensorValue_5x6()
      sp_output = tf.sparse_reshape(sp_input, [4, 7])
      with self.assertRaisesOpError(
          "Input to reshape is a tensor with 30 dense values"):
        sess.run(sp_output, {sp_input: input_val})

  def testMismatchedSizesWithInferredDim(self):
    with self.test_session(use_gpu=False) as sess:
      sp_input = self._SparseTensorPlaceholder()
      input_val = self._SparseTensorValue_5x6()
      sp_output = tf.sparse_reshape(sp_input, [4, -1])
      with self.assertRaisesOpError("requested shape requires a multiple"):
        sess.run(sp_output, {sp_input: input_val})

  def testPartialShapes(self):
    with self.test_session(use_gpu=False):
      # Incorporate new rank into shape information if known
      sp_input = self._SparseTensorPlaceholder()
      sp_output = tf.sparse_reshape(sp_input, [2, 3, 5])
      self.assertListEqual(sp_output.indices.get_shape().as_list(), [None, 3])
      self.assertListEqual(sp_output.shape.get_shape().as_list(), [3])

      # Incorporate known shape information about input indices in output
      # indices
      sp_input = self._SparseTensorPlaceholder()
      sp_input.indices.set_shape([5, None])
      sp_output = tf.sparse_reshape(sp_input, [2, 3, 5])
      self.assertListEqual(sp_output.indices.get_shape().as_list(), [5, 3])
      self.assertListEqual(sp_output.shape.get_shape().as_list(), [3])

      # Even if new_shape has no shape information, we know the ranks of
      # output indices and shape
      sp_input = self._SparseTensorPlaceholder()
      sp_input.indices.set_shape([5, None])
      new_shape = tf.placeholder(tf.int64)
      sp_output = tf.sparse_reshape(sp_input, new_shape)
      self.assertListEqual(sp_output.indices.get_shape().as_list(), [5, None])
      self.assertListEqual(sp_output.shape.get_shape().as_list(), [None])

  def testDenseReshapeSemantics(self):
    with self.test_session(use_gpu=False) as sess:
      # Compute a random rank-5 initial shape and new shape, randomly sparsify
      # it, and check that the output of SparseReshape has the same semantics
      # as a dense reshape.
      factors = np.array([2] * 4 + [3] * 4 + [5] * 4)  # 810k total elements
      orig_rank = np.random.randint(2, 7)
      orig_map = np.random.randint(orig_rank, size=factors.shape)
      orig_shape = [np.prod(factors[orig_map == d]) for d in range(orig_rank)]
      new_rank = np.random.randint(2, 7)
      new_map = np.random.randint(new_rank, size=factors.shape)
      new_shape = [np.prod(factors[new_map == d]) for d in range(new_rank)]

      orig_dense = np.random.uniform(size=orig_shape)
      orig_indices = np.transpose(np.nonzero(orig_dense < 0.5))
      orig_values = orig_dense[orig_dense < 0.5]

      new_dense = np.reshape(orig_dense, new_shape)
      new_indices = np.transpose(np.nonzero(new_dense < 0.5))
      new_values = new_dense[new_dense < 0.5]

      sp_input = self._SparseTensorPlaceholder()
      input_val = tf.SparseTensorValue(orig_indices, orig_values, orig_shape)
      sp_output = tf.sparse_reshape(sp_input, new_shape)

      output_val = sess.run(sp_output, {sp_input: input_val})
      self.assertAllEqual(output_val.indices, new_indices)
      self.assertAllEqual(output_val.values, new_values)
      self.assertAllEqual(output_val.shape, new_shape)


if __name__ == "__main__":
  tf.test.main()
