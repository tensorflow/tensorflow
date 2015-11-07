"""Tests for tensorflow.python.framework.device."""
import tensorflow.python.platform

from tensorflow.python.framework import device
from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest


class DeviceTest(test_util.TensorFlowTestCase):

  def testEmpty(self):
    d = device.Device()
    self.assertEquals("", d.ToString())
    d.parse_from_string("")
    self.assertEquals("", d.ToString())

  def testConstructor(self):
    d = device.Device(job="j", replica=0, task=1,
                      device_type="CPU", device_index=2)
    self.assertEquals("j", d.job)
    self.assertEquals(0, d.replica)
    self.assertEquals(1, d.task)
    self.assertEquals("CPU", d.device_type)
    self.assertEquals(2, d.device_index)
    self.assertEquals("/job:j/replica:0/task:1/device:CPU:2", d.to_string())

    d = device.Device(device_type="GPU", device_index=0)
    self.assertEquals("/device:GPU:0", d.to_string())

  def testto_string(self):
    d = device.Device()
    d.job = "foo"
    self.assertEquals("/job:foo", d.to_string())
    d.task = 3
    self.assertEquals("/job:foo/task:3", d.to_string())
    d.device_type = "CPU"
    d.device_index = 0
    self.assertEquals("/job:foo/task:3/device:CPU:0", d.to_string())
    d.task = None
    d.replica = 12
    self.assertEquals("/job:foo/replica:12/device:CPU:0", d.to_string())
    d.device_type = "GPU"
    d.device_index = 2
    self.assertEquals("/job:foo/replica:12/device:GPU:2", d.to_string())
    d.device_type = "CPU"
    d.device_index = 1
    self.assertEquals("/job:foo/replica:12/device:CPU:1", d.to_string())
    d.device_type = None
    d.device_index = None
    d.cpu = None
    self.assertEquals("/job:foo/replica:12", d.to_string())

    # Test wildcard
    d = device.Device(job="foo", replica=12, task=3, device_type="GPU")
    self.assertEquals("/job:foo/replica:12/task:3/device:GPU:*", d.to_string())

  def testParse(self):
    d = device.Device()
    d.parse_from_string("/job:foo/replica:0")
    self.assertEquals("/job:foo/replica:0", d.to_string())
    d.parse_from_string("/replica:1/task:0/cpu:0")
    self.assertEquals("/replica:1/task:0/device:CPU:0", d.to_string())
    d.parse_from_string("/replica:1/task:0/device:CPU:0")
    self.assertEquals("/replica:1/task:0/device:CPU:0", d.to_string())
    d.parse_from_string("/job:muu/gpu:2")
    self.assertEquals("/job:muu/device:GPU:2", d.to_string())
    with self.assertRaises(Exception) as e:
      d.parse_from_string("/job:muu/gpu:2/cpu:0")
    self.assertTrue("Cannot specify multiple device" in e.exception.message)

  def testFromString(self):
    d = device.from_string("/job:foo/replica:0")
    self.assertEquals("/job:foo/replica:0", d.to_string())
    with self.assertRaises(Exception) as e:
      d = device.from_string("/job:muu/gpu:2/cpu:0")
    self.assertTrue("Cannot specify multiple device" in e.exception.message)

    d = device.from_string("/job:foo/replica:0/task:3/cpu:*")
    self.assertEquals(None, d.device_index)
    d = device.from_string("/job:foo/replica:0/task:3/gpu:7")
    self.assertEquals(7, d.device_index)
    d = device.from_string("/job:foo/replica:0/task:3/device:GPU:7")
    self.assertEquals(7, d.device_index)

  def testMerge(self):
    d = device.from_string("/job:foo/replica:0")
    self.assertEquals("/job:foo/replica:0", d.to_string())
    d.merge_from(device.from_string("/task:1/gpu:2"))
    self.assertEquals("/job:foo/replica:0/task:1/device:GPU:2", d.to_string())

    d = device.Device()
    d.merge_from(device.from_string("/task:1/cpu:0"))
    self.assertEquals("/task:1/device:CPU:0", d.to_string())
    d.merge_from(device.from_string("/job:boo/gpu:0"))
    self.assertEquals("/job:boo/task:1/device:GPU:0", d.to_string())
    d.merge_from(device.from_string("/job:muu/cpu:2"))
    self.assertEquals("/job:muu/task:1/device:CPU:2", d.to_string())
    d.merge_from(device.from_string("/job:muu/device:MyFunnyDevice:2"))
    self.assertEquals("/job:muu/task:1/device:MyFunnyDevice:2", d.to_string())

  def testCheckValid(self):
    device.CheckValid("/job:foo/replica:0")

    with self.assertRaises(Exception) as e:
      device.CheckValid("/job:j/replica:foo")
    self.assertTrue("invalid literal for int" in e.exception.message)

    with self.assertRaises(Exception) as e:
      device.CheckValid("/job:j/task:bar")
    self.assertTrue("invalid literal for int" in e.exception.message)

    with self.assertRaises(Exception) as e:
      device.CheckValid("/bar:muu/baz:2")
    self.assertTrue("Unknown attribute: 'bar'" in e.exception.message)

    with self.assertRaises(Exception) as e:
      device.CheckValid("/cpu:0/gpu:2")
    self.assertTrue("Cannot specify multiple device" in e.exception.message)


if __name__ == "__main__":
  googletest.main()
