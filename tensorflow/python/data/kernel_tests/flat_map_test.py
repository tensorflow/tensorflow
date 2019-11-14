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
"""Tests for `tf.data.Dataset.flat_map()`."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

from absl.testing import parameterized
import numpy as np

from tensorflow.python.client import session
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import combinations
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops.ragged import ragged_conversion_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import test
from tensorflow.python.training import server_lib


class FlatMapTest(test_base.DatasetTestBase, parameterized.TestCase):

  # pylint: disable=g-long-lambda
  @combinations.generate(test_base.default_test_combinations())
  def testFlatMapDataset(self):
    repeats = [1, 2, 3, 4, 5, 0, 1]
    components = np.array(repeats, dtype=np.int64)
    dataset = dataset_ops.Dataset.from_tensor_slices(components).flat_map(
        lambda x: dataset_ops.Dataset.from_tensors([x]).repeat(x))
    expected_output = []
    for i in repeats:
      expected_output.extend([[i]] * i)
    self.assertDatasetProduces(dataset, expected_output=expected_output)

  @combinations.generate(test_base.default_test_combinations())
  def testNestedFlatMapDataset(self):
    repeats = [[1, 2], [3, 4], [5, 0], [1, 7]]
    components = np.array(repeats, dtype=np.int64)
    dataset = dataset_ops.Dataset.from_tensor_slices(components).flat_map(
        lambda x: dataset_ops.Dataset.from_tensor_slices(x).flat_map(
            lambda y: dataset_ops.Dataset.from_tensors(y).repeat(y))
    )
    expected_output = []
    for row in repeats:
      for i in row:
        expected_output.extend([i] * i)
    self.assertDatasetProduces(dataset, expected_output=expected_output)

  # Note: no eager mode coverage, session specific test.
  @combinations.generate(
      combinations.combine(tf_api_version=[1, 2], mode=["graph"]))
  def testSkipEagerSharedResourceNestedFlatMapDataset(self):
    repeats = [[1, 2], [3, 4], [5, 0], [1, 7]]
    components = np.array(repeats, dtype=np.int64)
    iterator = (
        dataset_ops.make_initializable_iterator(
            dataset_ops.Dataset.from_tensor_slices(components).flat_map(
                lambda x: dataset_ops.Dataset.from_tensor_slices(x).flat_map(
                    lambda y: dataset_ops.Dataset.from_tensors(y).repeat(y))),
            shared_name="shared_flat_map_iterator"))
    init_op = iterator.initializer
    get_next = iterator.get_next()

    # Create two concurrent sessions that share the same iterator
    # resource on the same server, and verify that a random
    # interleaving of `Session.run(get_next)` calls on the two
    # sessions yields the expected result.
    server = server_lib.Server.create_local_server()
    with session.Session(server.target) as sess1:
      with session.Session(server.target) as sess2:
        for _ in range(3):
          sess = random.choice([sess1, sess2])
          sess.run(init_op)
          for row in repeats:
            for i in row:
              for _ in range(i):
                sess = random.choice([sess1, sess2])
                self.assertEqual(i, sess.run(get_next))

        with self.assertRaises(errors.OutOfRangeError):
          sess = random.choice([sess1, sess2])
          sess.run(get_next)

  @combinations.generate(test_base.default_test_combinations())
  def testMapDict(self):
    dataset = dataset_ops.Dataset.range(10).map(
        lambda x: {"foo": x * 2, "bar": x ** 2}).flat_map(
            lambda d: dataset_ops.Dataset.from_tensors(
                d["foo"]).repeat(d["bar"]))
    get_next = self.getNext(dataset)
    for i in range(10):
      for _ in range(i**2):
        self.assertEqual(i * 2, self.evaluate(get_next()))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())

  @combinations.generate(test_base.default_test_combinations())
  def testSparse(self):
    def _map_fn(i):
      return sparse_tensor.SparseTensorValue(
          indices=[[0, 0], [1, 1]], values=(i * [1, -1]), dense_shape=[2, 2])

    def _flat_map_fn(x):
      return dataset_ops.Dataset.from_tensor_slices(
          sparse_ops.sparse_to_dense(x.indices, x.dense_shape, x.values))

    dataset = dataset_ops.Dataset.range(10).map(_map_fn).flat_map(_flat_map_fn)
    expected_output = []
    for i in range(10):
      for j in range(2):
        expected_output.append([i, 0] if j % 2 == 0 else [0, -i])
    self.assertDatasetProduces(dataset, expected_output=expected_output)

  @combinations.generate(test_base.default_test_combinations())
  def testTensorArray(self):
    def _map_fn(i):
      i = math_ops.cast(i, dtypes.int32)
      return (
          tensor_array_ops.TensorArray(
              dtype=dtypes.int32, element_shape=(), size=i)
          .unstack(math_ops.range(i)))

    def _flat_map_fn(x):
      self.assertIsInstance(x, tensor_array_ops.TensorArray)
      return dataset_ops.Dataset.from_tensor_slices(x.stack())

    dataset = dataset_ops.Dataset.range(10).map(_map_fn).flat_map(_flat_map_fn)

    expected_output = []
    for i in range(10):
      for j in range(i):
        expected_output.append(j)

    self.assertDatasetProduces(dataset, expected_output=expected_output)

  @combinations.generate(test_base.default_test_combinations())
  def testRagged(self):

    def _map_fn(i):
      return ragged_tensor.RaggedTensor.from_tensor(i * [[1], [-1]])

    def _flat_map_fn(x):
      return dataset_ops.Dataset.from_tensor_slices(
          ragged_conversion_ops.to_tensor(x))

    dataset = dataset_ops.Dataset.range(10).map(_map_fn).flat_map(_flat_map_fn)
    expected_output = []
    for i in range(10):
      expected_output.append([i])
      expected_output.append([-i])
    self.assertDatasetProduces(dataset, expected_output=expected_output)

if __name__ == "__main__":
  test.main()
