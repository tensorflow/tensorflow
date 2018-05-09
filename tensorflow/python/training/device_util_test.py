# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for device utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.platform import test
from tensorflow.python.training import device_util


class DeviceUtilTest(test.TestCase):

  def testCurrentDeviceWithGlobalGraph(self):
    with ops.device("/cpu:0"):
      self.assertEqual(device_util.current(), "/device:CPU:0")

    with ops.device("/job:worker"):
      with ops.device("/cpu:0"):
        self.assertEqual(device_util.current(), "/job:worker/device:CPU:0")

    with ops.device("/cpu:0"):
      with ops.device("/gpu:0"):
        self.assertEqual(device_util.current(), "/device:GPU:0")

  def testCurrentDeviceWithNonGlobalGraph(self):
    with ops.Graph().as_default():
      with ops.device("/cpu:0"):
        self.assertEqual(device_util.current(), "/device:CPU:0")

  def testCurrentDeviceWithEager(self):
    with context.eager_mode():
      with ops.device("/cpu:0"):
        self.assertEqual(device_util.current(),
                         "/job:localhost/replica:0/task:0/device:CPU:0")

  def testCanonicalizeWithoutDefaultDevice(self):
    self.assertEqual(
        device_util.canonicalize("/cpu:0"),
        "/job:localhost/replica:0/task:0/device:CPU:0")
    self.assertEqual(
        device_util.canonicalize("/job:worker/cpu:0"),
        "/job:worker/replica:0/task:0/device:CPU:0")
    self.assertEqual(
        device_util.canonicalize("/job:worker/task:1/cpu:0"),
        "/job:worker/replica:0/task:1/device:CPU:0")

  def testCanonicalizeWithDefaultDevice(self):
    self.assertEqual(
        device_util.canonicalize("/job:worker/task:1/cpu:0", default="/gpu:0"),
        "/job:worker/replica:0/task:1/device:CPU:0")
    self.assertEqual(
        device_util.canonicalize("/job:worker/task:1", default="/gpu:0"),
        "/job:worker/replica:0/task:1/device:GPU:0")
    self.assertEqual(
        device_util.canonicalize("/cpu:0", default="/job:worker"),
        "/job:worker/replica:0/task:0/device:CPU:0")

  def testResolveWithDeviceScope(self):
    with ops.device("/gpu:0"):
      self.assertEqual(
          device_util.resolve("/job:worker/task:1/cpu:0"),
          "/job:worker/replica:0/task:1/device:CPU:0")
      self.assertEqual(
          device_util.resolve("/job:worker/task:1"),
          "/job:worker/replica:0/task:1/device:GPU:0")
    with ops.device("/job:worker"):
      self.assertEqual(
          device_util.resolve("/cpu:0"),
          "/job:worker/replica:0/task:0/device:CPU:0")


if __name__ == "__main__":
  test.main()
