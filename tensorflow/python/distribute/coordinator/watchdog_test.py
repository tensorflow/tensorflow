# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for watchdog.py."""

import os
import time

from absl.testing import parameterized

from tensorflow.python.distribute.coordinator import watchdog
from tensorflow.python.eager import test


class WatchDogTest(test.TestCase, parameterized.TestCase):

  @parameterized.parameters(True, False)
  def testWatchDogTimeout(self, use_env_var):
    tmp_file = self.create_tempfile()
    f = open(tmp_file, "w+")

    triggerred_count = [0]

    def on_triggered_fn():
      triggerred_count[0] += 1

    timeout = 3
    if use_env_var:
      os.environ["TF_CLUSTER_COORDINATOR_WATCH_DOG_TIMEOUT"] = str(timeout)
      wd = watchdog.WatchDog(traceback_file=f, on_triggered=on_triggered_fn)
    else:
      wd = watchdog.WatchDog(
          timeout=timeout, traceback_file=f, on_triggered=on_triggered_fn)
    time.sleep(6)

    self.assertGreaterEqual(triggerred_count[0], 1)
    wd.report_closure_done()
    time.sleep(1)
    self.assertGreaterEqual(triggerred_count[0], 1)
    time.sleep(5)
    self.assertGreaterEqual(triggerred_count[0], 2)

    wd.stop()
    time.sleep(5)
    last_triggered_count = triggerred_count[0]
    time.sleep(10)
    self.assertEqual(last_triggered_count, triggerred_count[0])

    f.close()
    with open(tmp_file) as f:
      self.assertIn("Current thread", f.read())


if __name__ == "__main__":
  test.main()
