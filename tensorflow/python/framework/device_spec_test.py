# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tensorflow.python.framework.device_spec."""

from absl.testing import parameterized

from tensorflow.python.framework import device_spec
from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest


TEST_V1_AND_V2 = (("v1", device_spec.DeviceSpecV1),
                  ("v2", device_spec.DeviceSpecV2))


class DeviceSpecTest(test_util.TensorFlowTestCase, parameterized.TestCase):

  @parameterized.named_parameters(*TEST_V1_AND_V2)
  def test_empty(self, device_spec_type):
    d = device_spec_type()
    self.assertEqual("", d.to_string())
    d.parse_from_string("")
    self.assertEqual("", d.to_string())

  @parameterized.named_parameters(*TEST_V1_AND_V2)
  def test_constructor(self, device_spec_type):
    d = device_spec_type(job="j", replica=0, task=1,
                         device_type="CPU", device_index=2)
    self.assertEqual("j", d.job)
    self.assertEqual(0, d.replica)
    self.assertEqual(1, d.task)
    self.assertEqual("CPU", d.device_type)
    self.assertEqual(2, d.device_index)
    self.assertEqual("/job:j/replica:0/task:1/device:CPU:2", d.to_string())

    d = device_spec_type(device_type="GPU", device_index=0)
    self.assertEqual("/device:GPU:0", d.to_string())

  def testto_string_legacy(self):
    """DeviceSpecV1 allows direct mutation."""
    d = device_spec.DeviceSpecV1()
    d.job = "foo"
    self.assertEqual("/job:foo", d.to_string())
    d.task = 3
    self.assertEqual("/job:foo/task:3", d.to_string())
    d.device_type = "CPU"
    d.device_index = 0
    self.assertEqual("/job:foo/task:3/device:CPU:0", d.to_string())
    d.task = None
    d.replica = 12
    self.assertEqual("/job:foo/replica:12/device:CPU:0", d.to_string())
    d.device_type = "GPU"
    d.device_index = 2
    self.assertEqual("/job:foo/replica:12/device:GPU:2", d.to_string())
    d.device_type = "CPU"
    d.device_index = 1
    self.assertEqual("/job:foo/replica:12/device:CPU:1", d.to_string())
    d.device_type = None
    d.device_index = None
    self.assertEqual("/job:foo/replica:12", d.to_string())

    # Test wildcard
    d = device_spec.DeviceSpecV1(job="foo", replica=12, task=3,
                                 device_type="GPU")
    self.assertEqual("/job:foo/replica:12/task:3/device:GPU:*", d.to_string())

  @parameterized.named_parameters(*TEST_V1_AND_V2)
  def test_replace(self, device_spec_type):
    d = device_spec_type()
    d = d.replace(job="foo")
    self.assertEqual("/job:foo", d.to_string())

    d = d.replace(task=3)
    self.assertEqual("/job:foo/task:3", d.to_string())

    d = d.replace(device_type="CPU", device_index=0)
    self.assertEqual("/job:foo/task:3/device:CPU:0", d.to_string())

    d = d.replace(task=None, replica=12)
    self.assertEqual("/job:foo/replica:12/device:CPU:0", d.to_string())

    d = d.replace(device_type="GPU", device_index=2)
    self.assertEqual("/job:foo/replica:12/device:GPU:2", d.to_string())

    d = d.replace(device_type="CPU", device_index=1)
    self.assertEqual("/job:foo/replica:12/device:CPU:1", d.to_string())

    d = d.replace(device_type=None, device_index=None)
    self.assertEqual("/job:foo/replica:12", d.to_string())

    # Test wildcard
    d = device_spec.DeviceSpecV1(job="foo", replica=12, task=3,
                                 device_type="GPU")
    self.assertEqual("/job:foo/replica:12/task:3/device:GPU:*", d.to_string())

  @parameterized.named_parameters(*TEST_V1_AND_V2)
  def testto_string(self, device_spec_type):
    d = device_spec_type(job="foo")
    self.assertEqual("/job:foo", d.to_string())

    d = device_spec_type(job="foo", task=3)
    self.assertEqual("/job:foo/task:3", d.to_string())

    d = device_spec_type(job="foo", task=3, device_type="cpu", device_index=0)
    self.assertEqual("/job:foo/task:3/device:CPU:0", d.to_string())

    d = device_spec_type(job="foo", replica=12, device_type="cpu",
                         device_index=0)
    self.assertEqual("/job:foo/replica:12/device:CPU:0", d.to_string())

    d = device_spec_type(job="foo", replica=12, device_type="gpu",
                         device_index=2)
    self.assertEqual("/job:foo/replica:12/device:GPU:2", d.to_string())

    d = device_spec_type(job="foo", replica=12)
    self.assertEqual("/job:foo/replica:12", d.to_string())

    # Test wildcard
    d = device_spec_type(job="foo", replica=12, task=3, device_type="GPU")
    self.assertEqual("/job:foo/replica:12/task:3/device:GPU:*", d.to_string())

  def test_parse_legacy(self):
    d = device_spec.DeviceSpecV1()
    d.parse_from_string("/job:foo/replica:0")
    self.assertEqual("/job:foo/replica:0", d.to_string())

    d.parse_from_string("/replica:1/task:0/cpu:0")
    self.assertEqual("/replica:1/task:0/device:CPU:0", d.to_string())

    d.parse_from_string("/replica:1/task:0/device:CPU:0")
    self.assertEqual("/replica:1/task:0/device:CPU:0", d.to_string())

    d.parse_from_string("/job:muu/device:GPU:2")
    self.assertEqual("/job:muu/device:GPU:2", d.to_string())

    with self.assertRaisesRegex(ValueError,
                                "Multiple device types are not allowed"):
      d.parse_from_string("/job:muu/device:GPU:2/cpu:0")

  @parameterized.named_parameters(*TEST_V1_AND_V2)
  def test_to_from_string(self, device_spec_type):
    d = device_spec_type.from_string("/job:foo/replica:0")
    self.assertEqual("/job:foo/replica:0", d.to_string())
    self.assertEqual(0, d.replica)

    d = device_spec_type.from_string("/replica:1/task:0/cpu:0")
    self.assertEqual("/replica:1/task:0/device:CPU:0", d.to_string())
    self.assertAllEqual([1, 0, "CPU", 0],
                        [d.replica, d.task, d.device_type, d.device_index])

    d = device_spec_type.from_string("/replica:1/task:0/device:CPU:0")
    self.assertEqual("/replica:1/task:0/device:CPU:0", d.to_string())
    self.assertAllEqual([1, 0, "CPU", 0],
                        [d.replica, d.task, d.device_type, d.device_index])

    d = device_spec_type.from_string("/job:muu/device:GPU:2")
    self.assertEqual("/job:muu/device:GPU:2", d.to_string())
    self.assertAllEqual(["muu", "GPU", 2],
                        [d.job, d.device_type, d.device_index])

    with self.assertRaisesRegex(ValueError,
                                "Multiple device types are not allowed"):
      d.parse_from_string("/job:muu/device:GPU:2/cpu:0")

  def test_merge_legacy(self):
    d = device_spec.DeviceSpecV1.from_string("/job:foo/replica:0")
    self.assertEqual("/job:foo/replica:0", d.to_string())

    d.merge_from(device_spec.DeviceSpecV1.from_string("/task:1/device:GPU:2"))
    self.assertEqual("/job:foo/replica:0/task:1/device:GPU:2", d.to_string())

    d = device_spec.DeviceSpecV1()
    d.merge_from(device_spec.DeviceSpecV1.from_string("/task:1/cpu:0"))
    self.assertEqual("/task:1/device:CPU:0", d.to_string())

    d.merge_from(device_spec.DeviceSpecV1.from_string("/job:boo/device:GPU:0"))
    self.assertEqual("/job:boo/task:1/device:GPU:0", d.to_string())

    d.merge_from(device_spec.DeviceSpecV1.from_string("/job:muu/cpu:2"))
    self.assertEqual("/job:muu/task:1/device:CPU:2", d.to_string())
    d.merge_from(device_spec.DeviceSpecV1.from_string(
        "/job:muu/device:MyFunnyDevice:2"))
    self.assertEqual("/job:muu/task:1/device:MyFunnyDevice:2", d.to_string())

  def test_merge_removed(self):
    with self.assertRaises(AttributeError):
      d = device_spec.DeviceSpecV2()
      d.merge_from(device_spec.DeviceSpecV2.from_string("/task:1/cpu:0"))

  @parameterized.named_parameters(*TEST_V1_AND_V2)
  def test_combine(self, device_spec_type):
    d = device_spec_type.from_string("/job:foo/replica:0")
    self.assertEqual("/job:foo/replica:0", d.to_string())

    d = d.make_merged_spec(
        device_spec_type.from_string("/task:1/device:GPU:2"))
    self.assertEqual("/job:foo/replica:0/task:1/device:GPU:2", d.to_string())

    d = device_spec_type()
    d = d.make_merged_spec(device_spec_type.from_string("/task:1/cpu:0"))
    self.assertEqual("/task:1/device:CPU:0", d.to_string())

    d = d.make_merged_spec(
        device_spec_type.from_string("/job:boo/device:GPU:0"))
    self.assertEqual("/job:boo/task:1/device:GPU:0", d.to_string())

    d = d.make_merged_spec(device_spec_type.from_string("/job:muu/cpu:2"))
    self.assertEqual("/job:muu/task:1/device:CPU:2", d.to_string())
    d = d.make_merged_spec(device_spec_type.from_string(
        "/job:muu/device:MyFunnyDevice:2"))
    self.assertEqual("/job:muu/task:1/device:MyFunnyDevice:2", d.to_string())


if __name__ == "__main__":
  googletest.main()
