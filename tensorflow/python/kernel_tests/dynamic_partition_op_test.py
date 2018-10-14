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
"""Tests for the DynamicPartition op."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import gradients_impl
import tensorflow.python.ops.data_flow_grad  # pylint: disable=unused-import
from tensorflow.python.platform import test


class DynamicPartitionTest(test.TestCase):

  def testSimpleOneDimensional(self):
    with self.test_session(use_gpu=True) as sess:
      data = constant_op.constant([0, 13, 2, 39, 4, 17], dtype=dtypes.float32)
      indices = constant_op.constant([0, 0, 2, 3, 2, 1])
      partitions = data_flow_ops.dynamic_partition(
          data, indices, num_partitions=4)
      partition_vals = sess.run(partitions)

    self.assertEqual(4, len(partition_vals))
    self.assertAllEqual([0, 13], partition_vals[0])
    self.assertAllEqual([17], partition_vals[1])
    self.assertAllEqual([2, 4], partition_vals[2])
    self.assertAllEqual([39], partition_vals[3])
    # Vector data input to DynamicPartition results in
    # `num_partitions` vectors of unknown length.
    self.assertEqual([None], partitions[0].get_shape().as_list())
    self.assertEqual([None], partitions[1].get_shape().as_list())
    self.assertEqual([None], partitions[2].get_shape().as_list())
    self.assertEqual([None], partitions[3].get_shape().as_list())

  def testSimpleTwoDimensional(self):
    with self.test_session(use_gpu=True) as sess:
      data = constant_op.constant([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11],
                                   [12, 13, 14], [15, 16, 17]],
                                  dtype=dtypes.float32)
      indices = constant_op.constant([0, 0, 2, 3, 2, 1])
      partitions = data_flow_ops.dynamic_partition(
          data, indices, num_partitions=4)
      partition_vals = sess.run(partitions)

    self.assertEqual(4, len(partition_vals))
    self.assertAllEqual([[0, 1, 2], [3, 4, 5]], partition_vals[0])
    self.assertAllEqual([[15, 16, 17]], partition_vals[1])
    self.assertAllEqual([[6, 7, 8], [12, 13, 14]], partition_vals[2])
    self.assertAllEqual([[9, 10, 11]], partition_vals[3])
    # Vector data input to DynamicPartition results in
    # `num_partitions` matrices with an unknown number of rows, and 3 columns.
    self.assertEqual([None, 3], partitions[0].get_shape().as_list())
    self.assertEqual([None, 3], partitions[1].get_shape().as_list())
    self.assertEqual([None, 3], partitions[2].get_shape().as_list())
    self.assertEqual([None, 3], partitions[3].get_shape().as_list())

  def testLargeOneDimensional(self):
    num = 100000
    data_list = [x for x in range(num)]
    indices_list = [x % 2 for x in range(num)]
    part1 = [x for x in range(num) if x % 2 == 0]
    part2 = [x for x in range(num) if x % 2 == 1]
    with self.test_session(use_gpu=True) as sess:
      data = constant_op.constant(data_list, dtype=dtypes.float32)
      indices = constant_op.constant(indices_list, dtype=dtypes.int32)
      partitions = data_flow_ops.dynamic_partition(
          data, indices, num_partitions=2)
      partition_vals = sess.run(partitions)

    self.assertEqual(2, len(partition_vals))
    self.assertAllEqual(part1, partition_vals[0])
    self.assertAllEqual(part2, partition_vals[1])

  def testLargeTwoDimensional(self):
    rows = 100000
    cols = 100
    data_list = [None] * rows
    for i in range(rows):
      data_list[i] = [i for _ in range(cols)]
    num_partitions = 97
    indices_list = [(i ** 2) % num_partitions for i in range(rows)]
    parts = [[] for _ in range(num_partitions)]
    for i in range(rows):
      parts[(i ** 2) % num_partitions].append(data_list[i])
    with self.test_session(use_gpu=True) as sess:
      data = constant_op.constant(data_list, dtype=dtypes.float32)
      indices = constant_op.constant(indices_list, dtype=dtypes.int32)
      partitions = data_flow_ops.dynamic_partition(
          data, indices, num_partitions=num_partitions)
      partition_vals = sess.run(partitions)

    self.assertEqual(num_partitions, len(partition_vals))
    for i in range(num_partitions):
      # reshape because of empty parts
      parts_np = np.array(parts[i], dtype=np.float).reshape(-1, cols)
      self.assertAllEqual(parts_np, partition_vals[i])

  def testSimpleComplex(self):
    data_list = [1 + 2j, 3 + 4j, 5 + 6j, 7 + 8j]
    indices_list = [1, 0, 1, 0]
    with self.test_session(use_gpu=True) as sess:
      data = constant_op.constant(data_list, dtype=dtypes.complex64)
      indices = constant_op.constant(indices_list, dtype=dtypes.int32)
      partitions = data_flow_ops.dynamic_partition(
          data, indices, num_partitions=2)
      partition_vals = sess.run(partitions)

    self.assertEqual(2, len(partition_vals))
    self.assertAllEqual([3 + 4j, 7 + 8j], partition_vals[0])
    self.assertAllEqual([1 + 2j, 5 + 6j], partition_vals[1])

  def testScalarPartitions(self):
    data_list = [10, 13, 12, 11]
    with self.test_session(use_gpu=True) as sess:
      data = constant_op.constant(data_list, dtype=dtypes.float64)
      indices = 3
      partitions = data_flow_ops.dynamic_partition(
          data, indices, num_partitions=4)
      partition_vals = sess.run(partitions)

    self.assertEqual(4, len(partition_vals))
    self.assertAllEqual(np.array([], dtype=np.float64).reshape(-1, 4),
                        partition_vals[0])
    self.assertAllEqual(np.array([], dtype=np.float64).reshape(-1, 4),
                        partition_vals[1])
    self.assertAllEqual(np.array([], dtype=np.float64).reshape(-1, 4),
                        partition_vals[2])
    self.assertAllEqual(np.array([10, 13, 12, 11],
                                 dtype=np.float64).reshape(-1, 4),
                        partition_vals[3])

  def testHigherRank(self):
    np.random.seed(7)
    with self.test_session(use_gpu=True) as sess:
      for n in 2, 3:
        for shape in (4,), (4, 5), (4, 5, 2):
          partitions = np.random.randint(n, size=np.prod(shape)).reshape(shape)
          for extra_shape in (), (6,), (6, 7):
            data = np.random.randn(*(shape + extra_shape))
            partitions_t = constant_op.constant(partitions, dtype=dtypes.int32)
            data_t = constant_op.constant(data)
            outputs = data_flow_ops.dynamic_partition(
                data_t, partitions_t, num_partitions=n)
            self.assertEqual(n, len(outputs))
            outputs_val = sess.run(outputs)
            for i, output in enumerate(outputs_val):
              self.assertAllEqual(output, data[partitions == i])

            # Test gradients
            outputs_grad = [7 * output for output in outputs_val]
            grads = gradients_impl.gradients(outputs, [data_t, partitions_t],
                                             outputs_grad)
            self.assertEqual(grads[1], None)  # Partitions has no gradients
            self.assertAllEqual(7 * data, sess.run(grads[0]))

  def testEmptyParts(self):
    data_list = [1, 2, 3, 4]
    indices_list = [1, 3, 1, 3]
    with self.test_session(use_gpu=True) as sess:
      data = constant_op.constant(data_list, dtype=dtypes.float32)
      indices = constant_op.constant(indices_list, dtype=dtypes.int32)
      partitions = data_flow_ops.dynamic_partition(
          data, indices, num_partitions=4)
      partition_vals = sess.run(partitions)

    self.assertEqual(4, len(partition_vals))
    self.assertAllEqual([], partition_vals[0])
    self.assertAllEqual([1, 3], partition_vals[1])
    self.assertAllEqual([], partition_vals[2])
    self.assertAllEqual([2, 4], partition_vals[3])

  def testEmptyDataTwoDimensional(self):
    data_list = [[], []]
    indices_list = [0, 1]
    with self.test_session(use_gpu=True) as sess:
      data = constant_op.constant(data_list, dtype=dtypes.float32)
      indices = constant_op.constant(indices_list, dtype=dtypes.int32)
      partitions = data_flow_ops.dynamic_partition(
          data, indices, num_partitions=3)
      partition_vals = sess.run(partitions)

    self.assertEqual(3, len(partition_vals))
    self.assertAllEqual([[]], partition_vals[0])
    self.assertAllEqual([[]], partition_vals[1])
    self.assertAllEqual(np.array([], dtype=np.float).reshape(0, 0),
                        partition_vals[2])

  def testEmptyPartitions(self):
    data_list = []
    indices_list = []
    with self.test_session(use_gpu=True) as sess:
      data = constant_op.constant(data_list, dtype=dtypes.float32)
      indices = constant_op.constant(indices_list, dtype=dtypes.int32)
      partitions = data_flow_ops.dynamic_partition(
          data, indices, num_partitions=2)
      partition_vals = sess.run(partitions)

    self.assertEqual(2, len(partition_vals))
    self.assertAllEqual([], partition_vals[0])
    self.assertAllEqual([], partition_vals[1])

  @unittest.skip("Fails on windows.")
  def testGPUTooManyParts(self):
    # This test only makes sense on the GPU. There we do not check
    # for errors. In this case, we should discard all but the first
    # num_partitions indices.
    if not test.is_gpu_available():
      return

    data_list = [1, 2, 3, 4, 5, 6]
    indices_list = [6, 5, 4, 3, 1, 0]
    with self.test_session(use_gpu=True) as sess:
      data = constant_op.constant(data_list, dtype=dtypes.float32)
      indices = constant_op.constant(indices_list, dtype=dtypes.int32)
      partitions = data_flow_ops.dynamic_partition(
          data, indices, num_partitions=2)
      partition_vals = sess.run(partitions)

    self.assertEqual(2, len(partition_vals))
    self.assertAllEqual([6], partition_vals[0])
    self.assertAllEqual([5], partition_vals[1])

  @unittest.skip("Fails on windows.")
  def testGPUPartsTooLarge(self):
    # This test only makes sense on the GPU. There we do not check
    # for errors. In this case, we should discard all the values
    # larger than num_partitions.
    if not test.is_gpu_available():
      return

    data_list = [1, 2, 3, 4, 5, 6]
    indices_list = [10, 11, 2, 12, 0, 1000]
    with self.test_session(use_gpu=True) as sess:
      data = constant_op.constant(data_list, dtype=dtypes.float32)
      indices = constant_op.constant(indices_list, dtype=dtypes.int32)
      partitions = data_flow_ops.dynamic_partition(
          data, indices, num_partitions=5)
      partition_vals = sess.run(partitions)

    self.assertEqual(5, len(partition_vals))
    self.assertAllEqual([5], partition_vals[0])
    self.assertAllEqual([], partition_vals[1])
    self.assertAllEqual([3], partition_vals[2])
    self.assertAllEqual([], partition_vals[3])
    self.assertAllEqual([], partition_vals[4])

  @unittest.skip("Fails on windows.")
  def testGPUAllIndicesBig(self):
    # This test only makes sense on the GPU. There we do not check
    # for errors. In this case, we should discard all the values
    # and have an empty output.
    if not test.is_gpu_available():
      return

    data_list = [1.1, 2.1, 3.1, 4.1, 5.1, 6.1]
    indices_list = [90, 70, 60, 100, 110, 40]
    with self.test_session(use_gpu=True) as sess:
      data = constant_op.constant(data_list, dtype=dtypes.float32)
      indices = constant_op.constant(indices_list, dtype=dtypes.int32)
      partitions = data_flow_ops.dynamic_partition(
          data, indices, num_partitions=40)
      partition_vals = sess.run(partitions)

    self.assertEqual(40, len(partition_vals))
    for i in range(40):
      self.assertAllEqual([], partition_vals[i])

  def testErrorIndexOutOfRange(self):
    with self.cached_session() as sess:
      data = constant_op.constant([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11],
                                   [12, 13, 14]])
      indices = constant_op.constant([0, 2, 99, 2, 2])
      partitions = data_flow_ops.dynamic_partition(
          data, indices, num_partitions=4)
      with self.assertRaisesOpError(r"partitions\[2\] = 99 is not in \[0, 4\)"):
        sess.run(partitions)

  def testScalarIndexOutOfRange(self):
    with self.cached_session() as sess:
      bad = 17
      data = np.zeros(5)
      partitions = data_flow_ops.dynamic_partition(data, bad, num_partitions=7)
      with self.assertRaisesOpError(r"partitions = 17 is not in \[0, 7\)"):
        sess.run(partitions)

  def testHigherRankIndexOutOfRange(self):
    with self.cached_session() as sess:
      shape = (2, 3)
      indices = array_ops.placeholder(shape=shape, dtype=np.int32)
      data = np.zeros(shape + (5,))
      partitions = data_flow_ops.dynamic_partition(
          data, indices, num_partitions=7)
      for i in xrange(2):
        for j in xrange(3):
          bad = np.zeros(shape, dtype=np.int32)
          bad[i, j] = 17
          with self.assertRaisesOpError(
              r"partitions\[%d,%d\] = 17 is not in \[0, 7\)" % (i, j)):
            sess.run(partitions, feed_dict={indices: bad})

  def testErrorWrongDimsIndices(self):
    data = constant_op.constant([[0], [1], [2]])
    indices = constant_op.constant([[0], [0]])
    with self.assertRaises(ValueError):
      data_flow_ops.dynamic_partition(data, indices, num_partitions=4)

  #  see https://github.com/tensorflow/tensorflow/issues/17106
  def testCUBBug(self):
    x = constant_op.constant(np.random.randn(3072))
    inds = [0]*189 + [1]*184 + [2]*184 + [3]*191 + [4]*192 + [5]*195 + [6]*195
    inds += [7]*195 + [8]*188 + [9]*195 + [10]*188 + [11]*202 + [12]*194
    inds += [13]*194 + [14]*194 + [15]*192
    self.assertEqual(len(inds), x.shape[0])
    partitioned = data_flow_ops.dynamic_partition(x, inds, 16)
    with self.cached_session() as sess:
      res = sess.run(partitioned)
    self.assertEqual(res[-1].shape[0], 192)


if __name__ == "__main__":
  test.main()
