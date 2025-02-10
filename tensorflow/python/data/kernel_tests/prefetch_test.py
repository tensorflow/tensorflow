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
"""Tests for `tf.data.Dataset.prefetch()`."""
import time

from absl.testing import parameterized
import numpy as np

from tensorflow.python.data.experimental.ops import random_access
from tensorflow.python.data.kernel_tests import checkpoint_test_base
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.data.ops import prefetch_op
from tensorflow.python.framework import combinations
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.ops import script_ops
from tensorflow.python.platform import test


class PrefetchTest(test_base.DatasetTestBase, parameterized.TestCase):

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(buffer_size=[-1, None, 0, 42])))
  def testBufferSize(self, buffer_size):
    dataset = dataset_ops.Dataset.range(10).prefetch(buffer_size=buffer_size)
    self.assertDatasetProduces(dataset, expected_output=range(10))

  @combinations.generate(
      combinations.times(test_base.eager_only_combinations(),
                         combinations.combine(buffer_size=[0, 1, 2, 42])))
  def testPrefetching(self, buffer_size):
    dataset = dataset_ops.Dataset.range(1000)

    calls = 0

    @script_ops.eager_py_func(Tout=[dtypes.int64])
    def map_fn(x):
      nonlocal calls
      calls += 1
      return x

    dataset = dataset.map(map_fn)
    dataset = dataset.prefetch(buffer_size=buffer_size)
    it = iter(dataset)
    for _ in range(10):
      next(it)

    # Wait for the prefetch buffer to fill up.
    while calls != 10+buffer_size:
      time.sleep(0.1)
    # Wait some extra time to make sure the prefetch buffer isn't fetching more
    # elements than it should.
    time.sleep(0.5)
    self.assertEqual(calls, 10+buffer_size)

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(buffer_size=[-2, -42])))
  def testInvalidBufferSize(self, buffer_size):
    with self.assertRaises(errors.InvalidArgumentError):
      dataset = dataset_ops.Dataset.range(10).prefetch(buffer_size=buffer_size)
      self.evaluate(dataset._variant_tensor)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(
              buffer_size=[-1, None, 0, 42], slack_period=[1, 8])))
  def testPrefetchWithSlack(self, buffer_size, slack_period):
    dataset = dataset_ops.Dataset.range(100)
    dataset = prefetch_op._PrefetchDataset(  # pylint: disable=protected-access
        dataset, buffer_size, slack_period=slack_period)
    self.assertDatasetProduces(dataset, expected_output=range(100))

  @combinations.generate(combinations.combine(tf_api_version=1, mode="graph"))
  def testPrefetchCancellation(self):

    def map_py_fn(x):
      while x > -1:
        x = x * 1
      return x

    dataset = dataset_ops.Dataset.range(10).map(map_py_fn).prefetch(3)
    get_next = self.getNext(dataset)

    with self.cached_session() as sess:
      thread = self.checkedThread(self.assert_op_cancelled, args=(get_next(),))
      thread.start()
      time.sleep(2)
      sess.close()
      thread.join()

  @combinations.generate(test_base.default_test_combinations())
  def testName(self):
    dataset = dataset_ops.Dataset.from_tensors(42).prefetch(1, name="prefetch")
    self.assertDatasetProduces(dataset, [42])


class PrefetchCheckpointTest(checkpoint_test_base.CheckpointTestBase,
                             parameterized.TestCase):

  def build_dataset(self, options=None):
    dataset = dataset_ops.Dataset.range(100).prefetch(10)
    if options:
      dataset = dataset.with_options(options)
    return dataset

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          checkpoint_test_base.default_test_combinations(),
          combinations.combine(symbolic_checkpoint=[False, True])))
  def test(self, verify_fn, symbolic_checkpoint):
    options = options_lib.Options()
    options.experimental_symbolic_checkpoint = symbolic_checkpoint
    verify_fn(self, lambda: self.build_dataset(options), num_outputs=100)


class PrefetchRandomAccessTest(test_base.DatasetTestBase,
                               parameterized.TestCase):

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(index=[-1, 10, 11])))
  def testInvalidIndex(self, index):
    dataset = dataset_ops.Dataset.range(10).prefetch(buffer_size=5)
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(random_access.at(dataset, index=index))

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(index=[-2, 0, 1])))
  def testEmptyDataset(self, index):
    dataset = dataset_ops.Dataset.from_tensor_slices([]).prefetch(buffer_size=5)
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(random_access.at(dataset, index=index))

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(elements=[10, 50, 100], buffer_size=[0, 5, 10])))
  def testMultipleCombinations(self, elements, buffer_size):
    dataset = dataset_ops.Dataset.range(elements).prefetch(
        buffer_size=buffer_size)
    len_dataset = self.evaluate(dataset.cardinality())
    expected_output = np.arange(elements)
    for i in range(len_dataset):
      self.assertEqual(
          self.evaluate(random_access.at(dataset, index=i)), expected_output[i])
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(random_access.at(dataset, index=len_dataset))


if __name__ == "__main__":
  test.main()
