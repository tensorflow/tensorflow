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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

from tensorflow.python.data.experimental.ops import sleep
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test

_NUMPY_RANDOM_SEED = 42


@test_util.run_all_in_graph_and_eager_modes
class SleepTest(test_base.DatasetTestBase):

  def testSleep(self):
    self.skipTest("b/123597912")
    sleep_microseconds = 100
    dataset = dataset_ops.Dataset.range(10).apply(
        sleep.sleep(sleep_microseconds))
    next_element = self.getNext(dataset)
    start_time = time.time()
    for i in range(10):
      self.assertEqual(i, self.evaluate(next_element()))
    end_time = time.time()
    self.assertGreater(end_time - start_time, (10 * sleep_microseconds) / 1e6)
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(next_element())


if __name__ == "__main__":
  test.main()
