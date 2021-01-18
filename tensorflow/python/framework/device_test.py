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

"""Tests for tensorflow.python.framework.device."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

from tensorflow.python.eager import context
from tensorflow.python.framework import device
from tensorflow.python.framework import device_spec
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest


TEST_V1_AND_V2 = (("v1", device_spec.DeviceSpecV1),
                  ("v2", device_spec.DeviceSpecV2))


class DeviceTest(test_util.TensorFlowTestCase, parameterized.TestCase):

  @parameterized.named_parameters(*TEST_V1_AND_V2)
  def testMerge(self, DeviceSpec):  # pylint: disable=invalid-name
    d = DeviceSpec.from_string("/job:muu/task:1/device:MyFunnyDevice:2")
    self.assertEqual("/job:muu/task:1/device:MyFunnyDevice:2", d.to_string())

    if not context.executing_eagerly():
      with ops.device(device.merge_device("/device:GPU:0")):
        var1 = variables.Variable(1.0)
        self.assertEqual("/device:GPU:0", var1.device)
        with ops.device(device.merge_device("/job:worker")):
          var2 = variables.Variable(1.0)
          self.assertEqual("/job:worker/device:GPU:0", var2.device)
          with ops.device(device.merge_device("/device:CPU:0")):
            var3 = variables.Variable(1.0)
            self.assertEqual("/job:worker/device:CPU:0", var3.device)
            with ops.device(device.merge_device("/job:ps")):
              var4 = variables.Variable(1.0)
              self.assertEqual("/job:ps/device:CPU:0", var4.device)

  def testCanonicalName(self):
    self.assertEqual("/job:foo/replica:0",
                     device.canonical_name("/job:foo/replica:0"))
    self.assertEqual("/job:foo/replica:0",
                     device.canonical_name("/replica:0/job:foo"))

    self.assertEqual("/job:foo/replica:0/task:0",
                     device.canonical_name("/job:foo/replica:0/task:0"))
    self.assertEqual("/job:foo/replica:0/task:0",
                     device.canonical_name("/job:foo/task:0/replica:0"))

    self.assertEqual("/device:CPU:0",
                     device.canonical_name("/device:CPU:0"))
    self.assertEqual("/device:GPU:2",
                     device.canonical_name("/device:GPU:2"))

    self.assertEqual("/job:foo/replica:0/task:0/device:GPU:0",
                     device.canonical_name(
                         "/job:foo/replica:0/task:0/device:GPU:0"))
    self.assertEqual("/job:foo/replica:0/task:0/device:GPU:0",
                     device.canonical_name(
                         "/device:GPU:0/task:0/replica:0/job:foo"))

  def testCheckValid(self):
    device.check_valid("/job:foo/replica:0")

    with self.assertRaisesRegex(ValueError, "invalid literal for int"):
      device.check_valid("/job:j/replica:foo")

    with self.assertRaisesRegex(ValueError, "invalid literal for int"):
      device.check_valid("/job:j/task:bar")

    with self.assertRaisesRegex(ValueError, "Unknown attribute: 'bar'"):
      device.check_valid("/bar:muu/baz:2")

    with self.assertRaisesRegex(ValueError, "Cannot specify multiple device"):
      device.check_valid("/cpu:0/device:GPU:2")


if __name__ == "__main__":
  googletest.main()
