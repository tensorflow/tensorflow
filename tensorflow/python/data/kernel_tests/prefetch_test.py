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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

from absl.testing import parameterized

from tensorflow.python.data.kernel_tests import checkpoint_test_base
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import combinations
from tensorflow.python.framework import errors
from tensorflow.python.platform import test


class PrefetchTest(test_base.DatasetTestBase, parameterized.TestCase):

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(buffer_size=[-1, None, 0, 42])))
  def testBufferSize(self, buffer_size):
    dataset = dataset_ops.Dataset.range(10).prefetch(buffer_size=buffer_size)
    self.assertDatasetProduces(dataset, expected_output=range(10))

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
    dataset = dataset_ops.PrefetchDataset(
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
      time.sleep(0.5)
      sess.close()
      thread.join()


class PrefetchCheckpointTest(checkpoint_test_base.CheckpointTestBase,
                             parameterized.TestCase):

  def build_dataset(self, seed):
    return dataset_ops.Dataset.range(100).prefetch(10).shuffle(
        buffer_size=10, seed=seed, reshuffle_each_iteration=False)

  @combinations.generate(test_base.default_test_combinations())
  def testCore(self):
    num_outputs = 100
    self.run_core_tests(lambda: self.build_dataset(10), num_outputs)


if __name__ == "__main__":
  test.main()
