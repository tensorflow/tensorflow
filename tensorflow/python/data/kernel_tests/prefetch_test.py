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

from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test


@test_util.run_all_in_graph_and_eager_modes
class PrefetchTest(test_base.DatasetTestBase, parameterized.TestCase):

  @parameterized.parameters((-1), (0), (5))
  def testBufferSize(self, buffer_size):
    dataset = dataset_ops.Dataset.range(10).prefetch(buffer_size=buffer_size)
    self.assertDatasetProduces(dataset, expected_output=range(10))

  @parameterized.parameters((-2), (-42))
  def testInvalidBufferSize(self, buffer_size):
    with self.assertRaises(errors.InvalidArgumentError):
      dataset = dataset_ops.Dataset.range(10).prefetch(buffer_size=buffer_size)
      self.evaluate(dataset._variant_tensor)

  @parameterized.parameters(*[(buffer_size, slack_period)
                              for buffer_size in (-1, None, 0, 5)
                              for slack_period in (1, 8)])
  def testPrefetchWithSlack(self, buffer_size, slack_period):
    dataset = dataset_ops.Dataset.range(100)
    dataset = dataset_ops.PrefetchDataset(
        dataset, buffer_size, slack_period=slack_period)
    self.assertDatasetProduces(dataset, expected_output=range(100))

  @test_util.run_v1_only("graph-mode specific test")
  def testSkipEagerPrefetchCancellation(self):

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


if __name__ == "__main__":
  test.main()
