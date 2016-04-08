# Copyright 2015 Google Inc. All Rights Reserved.
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

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf


class DynamicPartitionTest(tf.test.TestCase):

  def testSimpleOneDimensional(self):
    with self.test_session() as sess:
      data = tf.constant([0, 13, 2, 39, 4, 17])
      indices = tf.constant([0, 0, 2, 3, 2, 1])
      partitions = tf.dynamic_partition(data, indices, num_partitions=4)
      partition_vals = sess.run(partitions)

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
    with self.test_session() as sess:
      data = tf.constant([[0, 1, 2], [3, 4, 5], [6, 7, 8],
                                   [9, 10, 11], [12, 13, 14], [15, 16, 17]])
      indices = tf.constant([0, 0, 2, 3, 2, 1])
      partitions = tf.dynamic_partition(data, indices, num_partitions=4)
      partition_vals = sess.run(partitions)

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

  def testHigherRank(self):
    np.random.seed(7)
    with self.test_session() as sess:
      for n in 2, 3:
        for shape in (4,), (4, 5), (4, 5, 2):
          partitions = np.random.randint(n, size=np.prod(shape)).reshape(shape)
          for extra_shape in (), (6,), (6, 7):
            data = np.random.randn(*(shape + extra_shape))
            partitions_t = tf.constant(partitions, dtype=tf.int32)
            data_t = tf.constant(data)
            outputs = tf.dynamic_partition(
                data_t, partitions_t, num_partitions=n)
            self.assertEqual(n, len(outputs))
            outputs_val = sess.run(outputs)
            for i, output in enumerate(outputs_val):
              self.assertAllEqual(output, data[partitions == i])

            # Test gradients
            outputs_grad = [7 * output for output in outputs_val]
            grads = tf.gradients(outputs, [data_t, partitions_t], outputs_grad)
            self.assertEqual(grads[1], None)  # Partitions has no gradients
            self.assertAllEqual(7 * data, sess.run(grads[0]))

  def testErrorIndexOutOfRange(self):
    with self.test_session() as sess:
      data = tf.constant([[0, 1, 2], [3, 4, 5], [6, 7, 8],
                                   [9, 10, 11], [12, 13, 14]])
      indices = tf.constant([0, 2, 99, 2, 2])
      partitions = tf.dynamic_partition(data, indices, num_partitions=4)
      with self.assertRaisesOpError(r"partitions\[2\] = 99 is not in \[0, 4\)"):
        sess.run(partitions)

  def testScalarIndexOutOfRange(self):
    with self.test_session() as sess:
      bad = 17
      data = np.zeros(5)
      partitions = tf.dynamic_partition(data, bad, num_partitions=7)
      with self.assertRaisesOpError(r"partitions = 17 is not in \[0, 7\)"):
        sess.run(partitions)

  def testHigherRankIndexOutOfRange(self):
    with self.test_session() as sess:
      shape = (2, 3)
      indices = tf.placeholder(shape=shape, dtype=np.int32)
      data = np.zeros(shape + (5,))
      partitions = tf.dynamic_partition(data, indices, num_partitions=7)
      for i in xrange(2):
        for j in xrange(3):
          bad = np.zeros(shape, dtype=np.int32)
          bad[i, j] = 17
          with self.assertRaisesOpError(
              r"partitions\[%d,%d\] = 17 is not in \[0, 7\)" % (i, j)):
            sess.run(partitions, feed_dict={indices: bad})

  def testErrorWrongDimsIndices(self):
    data = tf.constant([[0], [1], [2]])
    indices = tf.constant([[0], [0]])
    with self.assertRaises(ValueError):
      tf.dynamic_partition(data, indices, num_partitions=4)


if __name__ == "__main__":
  tf.test.main()
