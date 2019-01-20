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
"""Tests for `tf.data.Dataset.reduce()`."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np

from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


@test_util.run_all_in_graph_and_eager_modes
class ReduceTest(test_base.DatasetTestBase, parameterized.TestCase):

  def testSum(self):
    for i in range(10):
      ds = dataset_ops.Dataset.range(1, i + 1)
      result = ds.reduce(
          constant_op.constant(0, dtype=dtypes.int64), lambda x, y: x + y)
      self.assertEqual(((i + 1) * i) // 2, self.evaluate(result))

  def testSumTuple(self):

    def reduce_fn(state, value):
      v1, v2 = value
      return state + v1 + v2

    for i in range(10):
      ds = dataset_ops.Dataset.range(1, i + 1)
      ds = dataset_ops.Dataset.zip((ds, ds))
      result = ds.reduce(constant_op.constant(0, dtype=dtypes.int64), reduce_fn)
      self.assertEqual(((i + 1) * i), self.evaluate(result))

  def testSumAndCount(self):

    def reduce_fn(state, value):
      s, c = state
      return s + value, c + 1

    for i in range(10):
      ds = dataset_ops.Dataset.range(1, i + 1)
      result = ds.reduce((constant_op.constant(0, dtype=dtypes.int64),
                          constant_op.constant(0, dtype=dtypes.int64)),
                         reduce_fn)
      s, c = self.evaluate(result)
      self.assertEqual(((i + 1) * i) // 2, s)
      self.assertEqual(i, c)

  # NOTE: This test is specific to graph mode and is skipped in eager mode.
  @test_util.run_deprecated_v1
  def testSkipEagerSquareUsingPlaceholder(self):
    delta = array_ops.placeholder(dtype=dtypes.int64)

    def reduce_fn(state, _):
      return state + delta

    for i in range(10):
      ds = dataset_ops.Dataset.range(1, i + 1)
      result = ds.reduce(np.int64(0), reduce_fn)
      with self.cached_session() as sess:
        square = sess.run(result, feed_dict={delta: i})
        self.assertEqual(i * i, square)

  def testSparse(self):

    def reduce_fn(_, value):
      return value

    def make_sparse_fn(i):
      return sparse_tensor.SparseTensorValue(
          indices=np.array([[0, 0]]),
          values=(i * np.array([1])),
          dense_shape=np.array([1, 1]))

    for i in range(10):
      ds = dataset_ops.Dataset.from_tensors(make_sparse_fn(i+1))
      result = ds.reduce(make_sparse_fn(0), reduce_fn)
      self.assertSparseValuesEqual(make_sparse_fn(i + 1), self.evaluate(result))

  def testNested(self):

    def reduce_fn(state, value):
      state["dense"] += value["dense"]
      state["sparse"] = value["sparse"]
      return state

    def make_sparse_fn(i):
      return sparse_tensor.SparseTensorValue(
          indices=np.array([[0, 0]]),
          values=(i * np.array([1])),
          dense_shape=np.array([1, 1]))

    def map_fn(i):
      return {"dense": math_ops.cast(i, dtype=dtypes.int64),
              "sparse": make_sparse_fn(math_ops.cast(i, dtype=dtypes.int64))}

    for i in range(10):
      ds = dataset_ops.Dataset.range(1, i + 1).map(map_fn)
      result = ds.reduce(map_fn(0), reduce_fn)
      result = self.evaluate(result)
      self.assertEqual(((i + 1) * i) // 2, result["dense"])
      self.assertSparseValuesEqual(make_sparse_fn(i), result["sparse"])


if __name__ == "__main__":
  test.main()
