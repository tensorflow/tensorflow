# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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

from tensorflow.python.eager import context
from tensorflow.python.eager import custom_device_testutil
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.platform import test


class CustomDeviceTest(test.TestCase):

  def testRegisterCustomDevice(self):
    device_name = '/job:localhost/replica:0/task:0/device:CUSTOM:0'
    device, device_info, arrived_flag, executed_flag = (
        custom_device_testutil.GetLoggingDeviceCapsules(device_name))
    context.register_custom_device(device, device_name, device_info)
    self.assertFalse(custom_device_testutil.FlagValue(arrived_flag))
    self.assertFalse(custom_device_testutil.FlagValue(executed_flag))
    with ops.device(device_name):
      x = constant_op.constant(1.)
      y = x * constant_op.constant(2.)
    self.assertTrue(custom_device_testutil.FlagValue(executed_flag))
    # There was no copy onto the device. Actually I'm not sure how to trigger
    # that from Python.
    self.assertFalse(custom_device_testutil.FlagValue(arrived_flag))
    with self.assertRaisesRegex(errors.InternalError, 'Trying to copy'):
      y.numpy()


if __name__ == '__main__':
  ops.enable_eager_execution()
  test.main()
