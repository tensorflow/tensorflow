# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for the SWIG-wrapped device lib."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import device_lib
from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest
from tensorflow.python.platform import test


class DeviceLibTest(test_util.TensorFlowTestCase):

  def testListLocalDevices(self):
    devices = device_lib.list_local_devices()
    self.assertGreater(len(devices), 0)
    self.assertEqual(devices[0].device_type, "CPU")

    devices = device_lib.list_local_devices(config_pb2.ConfigProto())
    self.assertGreater(len(devices), 0)
    self.assertEqual(devices[0].device_type, "CPU")

    # GPU test
    if test.is_gpu_available():
      self.assertGreater(len(devices), 1)
      self.assertIn("GPU", [d.device_type for d in devices])


if __name__ == "__main__":
  googletest.main()
