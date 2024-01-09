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
"""Tests for `tf.data.Dataset.sparse_batch`."""
from absl.testing import parameterized
import numpy as np

from tensorflow.python.data.kernel_tests import checkpoint_test_base
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import combinations
from tensorflow.python.framework import errors
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


class DenseToSparseBatchTest(test_base.DatasetTestBase, parameterized.TestCase):

  @combinations.generate(test_base.default_test_combinations())
  def testBasic(self):
    components = np.random.randint(12, size=(100,)).astype(np.int32)
    dataset = dataset_ops.Dataset.from_tensor_slices(components).map(
        lambda x: array_ops.fill([x], x)).sparse_batch(4, [12])
    get_next = self.getNext(dataset)

    for start in range(0, len(components), 4):
      results = self.evaluate(get_next())
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
      self.evaluate(get_next())

  @combinations.generate(test_base.default_test_combinations())
  def testWithUnknownShape(self):
    components = np.random.randint(5, size=(40,)).astype(np.int32)
    dataset = dataset_ops.Dataset.from_tensor_slices(components).map(
        lambda x: array_ops.fill([x, x], x)).sparse_batch(4, [5, None])

    get_next = self.getNext(dataset)

    for start in range(0, len(components), 4):
      results = self.evaluate(get_next())
      self.assertAllEqual([[i, j, z]
                           for i, c in enumerate(components[start:start + 4])
                           for j in range(c)
                           for z in range(c)], results.indices)
      self.assertAllEqual([
          c for c in components[start:start + 4] for _ in range(c)
          for _ in range(c)
      ], results.values)
      self.assertAllEqual([
          min(4,
              len(components) - start), 5,
          np.max(components[start:start + 4])
      ], results.dense_shape)

    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())

  @combinations.generate(test_base.default_test_combinations())
  def testWithInvalidShape(self):
    input_tensor = array_ops.constant([[1]])
    with self.assertRaisesRegex(ValueError, "Dimension -2 must be >= 0"):
      dataset_ops.Dataset.from_tensors(input_tensor).sparse_batch(4, [-2])

  @combinations.generate(test_base.default_test_combinations())
  def testShapeErrors(self):

    def dataset_fn(input_tensor):
      return dataset_ops.Dataset.from_tensors(input_tensor).sparse_batch(
          4, [12])

    # Initialize with an input tensor of incompatible rank.
    get_next = self.getNext(dataset_fn([[1]]))
    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                "incompatible with the row shape"):
      self.evaluate(get_next())

    # Initialize with an input tensor that is larger than `row_shape`.
    get_next = self.getNext(dataset_fn(np.int32(range(13))))
    with self.assertRaisesRegex(errors.DataLossError,
                                "larger than the row shape"):
      self.evaluate(get_next())


class DenseToSparseBatchCheckpointTest(checkpoint_test_base.CheckpointTestBase,
                                       parameterized.TestCase):

  def _build_dataset(self, components):
    return dataset_ops.Dataset.from_tensor_slices(components).map(
        lambda x: array_ops.fill([x], x)).sparse_batch(4, [12])

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         checkpoint_test_base.default_test_combinations()))
  def test(self, verify_fn):
    components = np.random.randint(5, size=(40,)).astype(np.int32)

    num_outputs = len(components) // 4
    verify_fn(self, lambda: self._build_dataset(components), num_outputs)


if __name__ == "__main__":
  test.main()
