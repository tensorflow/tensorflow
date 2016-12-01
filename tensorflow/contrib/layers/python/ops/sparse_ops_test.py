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
"""Tests for tensorflow.contrib.layers.python.ops.sparse_ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow.contrib.layers.python.ops import sparse_ops


class SparseOpsTest(tf.test.TestCase):

  def test_dense_to_sparse_tensor_1d(self):
    with self.test_session() as sess:
      st = sparse_ops.dense_to_sparse_tensor([1, 0, 2, 0])
      result = sess.run(st)
    self.assertEqual(result.indices.dtype, np.int64)
    self.assertEqual(result.values.dtype, np.int32)
    self.assertEqual(result.shape.dtype, np.int64)
    self.assertAllEqual([[0], [2]], result.indices)
    self.assertAllEqual([1, 2], result.values)
    self.assertAllEqual([4], result.shape)

  def test_dense_to_sparse_tensor_1d_float(self):
    with self.test_session() as sess:
      st = sparse_ops.dense_to_sparse_tensor([1.5, 0.0, 2.3, 0.0])
      result = sess.run(st)
    self.assertEqual(result.indices.dtype, np.int64)
    self.assertEqual(result.values.dtype, np.float32)
    self.assertEqual(result.shape.dtype, np.int64)
    self.assertAllEqual([[0], [2]], result.indices)
    self.assertAllClose([1.5, 2.3], result.values)
    self.assertAllEqual([4], result.shape)

  def test_dense_to_sparse_tensor_1d_bool(self):
    with self.test_session() as sess:
      st = sparse_ops.dense_to_sparse_tensor([True, False, True, False])
      result = sess.run(st)
    self.assertEqual(result.indices.dtype, np.int64)
    self.assertEqual(result.values.dtype, np.bool)
    self.assertEqual(result.shape.dtype, np.int64)
    self.assertAllEqual([[0], [2]], result.indices)
    self.assertAllEqual([True, True], result.values)
    self.assertAllEqual([4], result.shape)

  def test_dense_to_sparse_tensor_1d_str(self):
    with self.test_session() as sess:
      st = sparse_ops.dense_to_sparse_tensor([b'qwe', b'', b'ewq', b''])
      result = sess.run(st)
    self.assertEqual(result.indices.dtype, np.int64)
    self.assertEqual(result.values.dtype, np.object)
    self.assertEqual(result.shape.dtype, np.int64)
    self.assertAllEqual([[0], [2]], result.indices)
    self.assertAllEqual([b'qwe', b'ewq'], result.values)
    self.assertAllEqual([4], result.shape)

  def test_dense_to_sparse_tensor_1d_str_special_ignore(self):
    with self.test_session() as sess:
      st = sparse_ops.dense_to_sparse_tensor(
          [b'qwe', b'', b'ewq', b''], ignore_value=b'qwe')
      result = sess.run(st)
    self.assertEqual(result.indices.dtype, np.int64)
    self.assertEqual(result.values.dtype, np.object)
    self.assertEqual(result.shape.dtype, np.int64)
    self.assertAllEqual([[1], [2], [3]], result.indices)
    self.assertAllEqual([b'', b'ewq', b''], result.values)
    self.assertAllEqual([4], result.shape)

  def test_dense_to_sparse_tensor_2d(self):
    with self.test_session() as sess:
      st = sparse_ops.dense_to_sparse_tensor([[1, 2, 0, 0], [3, 4, 5, 0]])
      result = sess.run(st)
    self.assertAllEqual([[0, 0], [0, 1], [1, 0], [1, 1], [1, 2]],
                        result.indices)
    self.assertAllEqual([1, 2, 3, 4, 5], result.values)
    self.assertAllEqual([2, 4], result.shape)

  def test_dense_to_sparse_tensor_3d(self):
    with self.test_session() as sess:
      st = sparse_ops.dense_to_sparse_tensor([[[1, 2, 0, 0], [3, 4, 5, 0]],
                                              [[7, 8, 0, 0], [9, 0, 0, 0]]])
      result = sess.run(st)
    self.assertAllEqual([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [0, 1, 2],
                         [1, 0, 0], [1, 0, 1], [1, 1, 0]], result.indices)
    self.assertAllEqual([1, 2, 3, 4, 5, 7, 8, 9], result.values)
    self.assertAllEqual([2, 2, 4], result.shape)

  def test_dense_to_sparse_tensor_1d_no_shape(self):
    with self.test_session() as sess:
      tensor = tf.placeholder(shape=[None], dtype=tf.int32)
      st = sparse_ops.dense_to_sparse_tensor(tensor)
      result = sess.run(st, feed_dict={tensor: [0, 100, 0, 3]})
    self.assertAllEqual([[1], [3]], result.indices)
    self.assertAllEqual([100, 3], result.values)
    self.assertAllEqual([4], result.shape)

  def test_dense_to_sparse_tensor_3d_no_shape(self):
    with self.test_session() as sess:
      tensor = tf.placeholder(shape=[None, None, None], dtype=tf.int32)
      st = sparse_ops.dense_to_sparse_tensor(tensor)
      result = sess.run(st,
                        feed_dict={
                            tensor: [[[1, 2, 0, 0], [3, 4, 5, 0]],
                                     [[7, 8, 0, 0], [9, 0, 0, 0]]]
                        })
    self.assertAllEqual([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [0, 1, 2],
                         [1, 0, 0], [1, 0, 1], [1, 1, 0]], result.indices)
    self.assertAllEqual([1, 2, 3, 4, 5, 7, 8, 9], result.values)
    self.assertAllEqual([2, 2, 4], result.shape)

  def test_convert_to_sparse_undef_shape(self):
    with self.test_session():
      with self.assertRaises(ValueError):
        tensor = tf.placeholder(dtype=tf.int32)
        sparse_ops.dense_to_sparse_tensor(tensor)


if __name__ == '__main__':
  tf.test.main()
