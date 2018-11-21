# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for `tf.data.experimental.dense_to_sparse_batch()."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.data.experimental.ops import batching
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


class DenseToSparseBatchTest(test_base.DatasetTestBase):

  def testDenseToSparseBatchDataset(self):
    components = np.random.randint(12, size=(100,)).astype(np.int32)
    iterator = (
        dataset_ops.Dataset.from_tensor_slices(components)
        .map(lambda x: array_ops.fill([x], x)).apply(
            batching.dense_to_sparse_batch(4, [12]))
        .make_initializable_iterator())
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.cached_session() as sess:
      self.evaluate(init_op)

      for start in range(0, len(components), 4):
        results = self.evaluate(get_next)
        self.assertAllEqual([[i, j]
                             for i, c in enumerate(components[start:start + 4])
                             for j in range(c)], results.indices)
        self.assertAllEqual(
            [c for c in components[start:start + 4] for _ in range(c)],
            results.values)
        self.assertAllEqual([min(4,
                                 len(components) - start), 12],
                            results.dense_shape)

      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  def testDenseToSparseBatchDatasetWithUnknownShape(self):
    components = np.random.randint(5, size=(40,)).astype(np.int32)
    iterator = (
        dataset_ops.Dataset.from_tensor_slices(components)
        .map(lambda x: array_ops.fill([x, x], x)).apply(
            batching.dense_to_sparse_batch(
                4, [5, None])).make_initializable_iterator())
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.cached_session() as sess:
      self.evaluate(init_op)

      for start in range(0, len(components), 4):
        results = self.evaluate(get_next)
        self.assertAllEqual([[i, j, z]
                             for i, c in enumerate(components[start:start + 4])
                             for j in range(c)
                             for z in range(c)], results.indices)
        self.assertAllEqual([
            c
            for c in components[start:start + 4] for _ in range(c)
            for _ in range(c)
        ], results.values)
        self.assertAllEqual([
            min(4,
                len(components) - start), 5,
            np.max(components[start:start + 4])
        ], results.dense_shape)

      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  def testDenseToSparseBatchDatasetWithInvalidShape(self):
    input_tensor = array_ops.constant([[1]])
    with self.assertRaisesRegexp(ValueError, "Dimension -2 must be >= 0"):
      dataset_ops.Dataset.from_tensors(input_tensor).apply(
          batching.dense_to_sparse_batch(4, [-2])).make_initializable_iterator()

  def testDenseToSparseBatchDatasetShapeErrors(self):
    input_tensor = array_ops.placeholder(dtypes.int32)
    iterator = (
        dataset_ops.Dataset.from_tensors(input_tensor).apply(
            batching.dense_to_sparse_batch(4, [12]))
        .make_initializable_iterator())
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.cached_session() as sess:
      # Initialize with an input tensor of incompatible rank.
      sess.run(init_op, feed_dict={input_tensor: [[1]]})
      with self.assertRaisesRegexp(errors.InvalidArgumentError,
                                   "incompatible with the row shape"):
        sess.run(get_next)

      # Initialize with an input tensor that is larger than `row_shape`.
      sess.run(init_op, feed_dict={input_tensor: range(13)})
      with self.assertRaisesRegexp(errors.DataLossError,
                                   "larger than the row shape"):
        sess.run(get_next)


if __name__ == "__main__":
  test.main()
