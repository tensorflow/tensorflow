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

import time

from absl.testing import parameterized
import numpy as np

from tensorflow.python.data.experimental.ops import testing
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import function
from tensorflow.python.framework import combinations
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


class ReduceTest(test_base.DatasetTestBase, parameterized.TestCase):

  @combinations.generate(test_base.default_test_combinations())
  def testSum(self):
    for i in range(10):
      ds = dataset_ops.Dataset.range(1, i + 1)
      result = ds.reduce(np.int64(0), lambda x, y: x + y)
      self.assertEqual(((i + 1) * i) // 2, self.evaluate(result))

  @combinations.generate(test_base.default_test_combinations())
  def testSumTuple(self):

    def reduce_fn(state, value):
      v1, v2 = value
      return state + v1 + v2

    for i in range(10):
      ds = dataset_ops.Dataset.range(1, i + 1)
      ds = dataset_ops.Dataset.zip((ds, ds))
      result = ds.reduce(constant_op.constant(0, dtype=dtypes.int64), reduce_fn)
      self.assertEqual(((i + 1) * i), self.evaluate(result))

  @combinations.generate(test_base.default_test_combinations())
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

  @combinations.generate(combinations.combine(tf_api_version=1, mode="graph"))
  def testSquareUsingPlaceholder(self):
    delta = array_ops.placeholder(dtype=dtypes.int64)

    def reduce_fn(state, _):
      return state + delta

    for i in range(10):
      ds = dataset_ops.Dataset.range(1, i + 1)
      result = ds.reduce(np.int64(0), reduce_fn)
      with self.cached_session() as sess:
        square = sess.run(result, feed_dict={delta: i})
        self.assertEqual(i * i, square)

  @combinations.generate(test_base.default_test_combinations())
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
      self.assertValuesEqual(make_sparse_fn(i + 1), self.evaluate(result))

  @combinations.generate(test_base.default_test_combinations())
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
      self.assertValuesEqual(make_sparse_fn(i), result["sparse"])

  @combinations.generate(test_base.default_test_combinations())
  def testDatasetSideEffect(self):
    counter_var = variables.Variable(0)

    def increment_fn(x):
      counter_var.assign_add(1)
      return x

    def dataset_fn():
      return dataset_ops.Dataset.range(10).map(increment_fn)

    def reduce_fn(state, value):
      return state + value

    @function.defun
    def fn():
      _ = dataset_fn().reduce(np.int64(0), reduce_fn)
      return "hello"

    self.evaluate(counter_var.initializer)
    self.assertEqual(self.evaluate(fn()), b"hello")
    self.assertEqual(self.evaluate(counter_var), 10)

  @combinations.generate(test_base.default_test_combinations())
  def testSideEffect(self):
    counter_var = variables.Variable(0)

    def dataset_fn():
      return dataset_ops.Dataset.range(10)

    def reduce_fn(state, value):
      counter_var.assign_add(1)
      return state + value

    @function.defun
    def fn():
      _ = dataset_fn().reduce(np.int64(0), reduce_fn)
      return "hello"

    self.evaluate(counter_var.initializer)
    self.assertEqual(self.evaluate(fn()), b"hello")
    self.assertEqual(self.evaluate(counter_var), 10)

  @combinations.generate(test_base.default_test_combinations())
  def testAutomaticControlDependencies(self):
    counter_var = variables.Variable(1)

    def dataset_fn():
      return dataset_ops.Dataset.range(1)

    def reduce1_fn(state, value):
      counter_var.assign(counter_var + 1)
      return state + value

    def reduce2_fn(state, value):
      counter_var.assign(counter_var * 2)
      return state + value

    @function.defun
    def fn():
      _ = dataset_fn().reduce(np.int64(0), reduce1_fn)
      _ = dataset_fn().reduce(np.int64(0), reduce2_fn)
      return "hello"

    self.evaluate(counter_var.initializer)
    self.assertEqual(self.evaluate(fn()), b"hello")
    self.assertEqual(self.evaluate(counter_var), 4)

  @combinations.generate(test_base.default_test_combinations())
  def testStateOnGPU(self):
    if not test_util.is_gpu_available():
      self.skipTest("No GPUs available.")

    state = constant_op.constant(0, dtype=dtypes.int64)

    def reduce_fn(state, value):
      with ops.device("/gpu:0"):
        return state + value

    for i in range(10):
      ds = dataset_ops.Dataset.range(1, i + 1)
      result = ds.reduce(state, reduce_fn)
      self.assertEqual(((i + 1) * i) // 2, self.evaluate(result))

  @combinations.generate(combinations.combine(tf_api_version=1, mode="graph"))
  def testCancellation(self):
    ds = dataset_ops.Dataset.from_tensors(1).repeat()
    result = ds.reduce(0, lambda x, y: x + y)
    with self.cached_session() as sess:
      # The `result` op is guaranteed to not complete before cancelled because
      # the dataset that is being reduced is infinite.
      thread = self.checkedThread(self.assert_op_cancelled, args=(result,))
      thread.start()
      time.sleep(0.2)
      sess.close()
      thread.join()

  @combinations.generate(test_base.default_test_combinations())
  def testInvalidFunction(self):
    ds = dataset_ops.Dataset.range(5)
    with self.assertRaises(errors.InvalidArgumentError):
      self.evaluate(ds.reduce(0, lambda _, __: ()))

  @combinations.generate(test_base.default_test_combinations())
  def testOptions(self):
    dataset = dataset_ops.Dataset.range(5)
    dataset = dataset.apply(testing.assert_next(["MapAndBatch"]))
    dataset = dataset.map(lambda x: x).batch(5)
    self.evaluate(dataset.reduce(0, lambda state, value: state))


if __name__ == "__main__":
  test.main()
