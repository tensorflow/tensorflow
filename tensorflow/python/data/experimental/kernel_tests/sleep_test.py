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
"""Tests for `tf.data.experimental.sleep()`."""
import time

from absl.testing import parameterized

from tensorflow.python.data.experimental.ops import testing
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import combinations
from tensorflow.python.framework import errors
from tensorflow.python.platform import test


class SleepTest(test_base.DatasetTestBase, parameterized.TestCase):

  @combinations.generate(test_base.default_test_combinations())
  def testSleep(self):
    self.skipTest("b/123597912")
    sleep_microseconds = 100
    dataset = dataset_ops.Dataset.range(10).apply(
        testing.sleep(sleep_microseconds))
    next_element = self.getNext(dataset)
    start_time = time.time()
    for i in range(10):
      self.assertEqual(i, self.evaluate(next_element()))
    end_time = time.time()
    self.assertGreater(end_time - start_time, (10 * sleep_microseconds) / 1e6)
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(next_element())

  @combinations.generate(combinations.combine(tf_api_version=1, mode="graph"))
  def testSleepCancellation(self):
    sleep_microseconds = int(1e6) * 1000
    ds = dataset_ops.Dataset.range(1)
    ds = ds.apply(testing.sleep(sleep_microseconds))
    ds = ds.prefetch(1)
    get_next = self.getNext(ds, requires_initialization=True)

    with self.cached_session() as sess:
      thread = self.checkedThread(self.assert_op_cancelled, args=(get_next(),))
      thread.start()
      time.sleep(0.2)
      sess.close()
      thread.join()

  @combinations.generate(combinations.combine(tf_api_version=1, mode="graph"))
  def testSleepBackgroundCancellation(self):
    ds = dataset_ops.Dataset.range(1)

    sleep_microseconds = int(1e6) * 1000
    ds_sleep = dataset_ops.Dataset.range(1)
    ds_sleep = ds.apply(testing.sleep(sleep_microseconds))

    ds = ds.concatenate(ds_sleep)
    ds = ds.prefetch(1)

    get_next = self.getNext(ds, requires_initialization=True)

    with self.cached_session():
      self.assertEqual(self.evaluate(get_next()), 0)


if __name__ == "__main__":
  test.main()
