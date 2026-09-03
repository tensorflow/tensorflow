# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
"""Tests verifying AutoStrategy hardware detection and strategy resolution."""

import json
import os
from unittest import mock

from tensorflow.python.distribute import auto_strategy
from tensorflow.python.distribute import one_device_strategy
from tensorflow.python.distribute.experimental import (
    multi_worker_mirrored_strategy)
from tensorflow.python.framework import config
from tensorflow.python.platform import test


class AutoStrategyTest(test.TestCase):

  def setUp(self):
    super(AutoStrategyTest, self).setUp()
    self.original_tf_config = os.environ.get("TF_CONFIG")

  def tearDown(self):
    if self.original_tf_config is not None:
      os.environ["TF_CONFIG"] = self.original_tf_config
    elif "TF_CONFIG" in os.environ:
      del os.environ["TF_CONFIG"]
    super(AutoStrategyTest, self).tearDown()

  @mock.patch.object(config, "list_physical_devices")
  def testFallbackToCPU(self, mock_list_physical_devices):
    mock_list_physical_devices.return_value = []
    strategy = auto_strategy.AutoStrategy()
    self.assertIsInstance(strategy, one_device_strategy.OneDeviceStrategy)
    # device_util.resolve() returns a fully-qualified device string; the
    # format varies by environment (e.g. "/device:CPU:0" locally vs.
    # "/job:localhost/replica:0/task:0/device:CPU:0" on CI runners).
    # Assert only on the meaningful suffix that is always present.
    self.assertIn("CPU:0", strategy.extended._device)

  @mock.patch.object(
      multi_worker_mirrored_strategy, "MultiWorkerMirroredStrategy")
  def testMultiWorkerDetection(self, mock_mwms_cls):
    # Prevent the real constructor from starting a live gRPC
    # CoordinationService and blocking indefinitely waiting for a second
    # worker (localhost:23456) that never connects, causing a 300s+ timeout.
    os.environ["TF_CONFIG"] = json.dumps({
        "cluster": {
            "worker": ["localhost:12345", "localhost:23456"]
        },
        "task": {"type": "worker", "index": 0}
    })
    strategy = auto_strategy.AutoStrategy()
    # Verify AutoStrategy dispatched to MultiWorkerMirroredStrategy.
    mock_mwms_cls.assert_called_once()
    self.assertIs(strategy, mock_mwms_cls.return_value)


if __name__ == "__main__":
  test.main()
