# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for the open source DTensor Python API."""

import os

# pylint: disable=g-direct-tensorflow-import
from tensorflow.dtensor.python import config
from tensorflow.dtensor.python.tests import test_util
from tensorflow.python.eager import context
from tensorflow.python.framework import device as tf_device
from tensorflow.python.platform import test as tf_test
# pylint: enable=g-direct-tensorflow-import


class ConfigTest(tf_test.TestCase):

  def setUp(self):
    super().setUp()
    test_util.reset_logical_devices('CPU', 2)
    if test_util.is_gpu_present():
      test_util.reset_logical_devices('GPU', 2)

  def tearDown(self):
    os.environ.pop(config._DT_JOBS, [])
    super().tearDown()

  def test_env_vars(self):
    self.assertEqual(config.client_id(), 0)
    self.assertEqual(config.num_clients(), 1)
    self.assertEqual(config.job_name(), 'localhost')
    self.assertEqual(config.full_job_name(), 'localhost/replica:0/task:0')
    self.assertEqual(config.jobs(), [])

  def test_list_devices(self):
    device_type = config.preferred_device_type()

    local_devices = [
        tf_device.DeviceSpec.from_string(
            f'/job:localhost/replica:0/task:0/device:{device_type}:0'),
        tf_device.DeviceSpec.from_string(
            f'/job:localhost/replica:0/task:0/device:{device_type}:1'),
    ]

    self.assertEqual(config.local_devices(device_type), local_devices)
    self.assertEqual(config.num_local_devices(device_type), 2)
    self.assertEqual(config.num_global_devices(device_type), 2)
    # The eager context should not be initialized by any of the calls
    self.assertFalse(context.context()._initialized)  # pylint: disable=protected-access

  def test_sort_jobs_with_bns_names(self):
    # bns names must be sorted in the bns order.
    dtensor_jobs = [
        '/bns/localhost/{task_id}'.format(task_id=i) for i in range(16)
    ]
    os.environ[config._DT_JOBS] = ','.join(dtensor_jobs)
    self.assertListEqual(dtensor_jobs, config.jobs())

    dtensor_jobs = [
        '/bns/localhost/{task_id}:8888'.format(task_id=i) for i in range(16)
    ]
    os.environ[config._DT_JOBS] = ','.join(dtensor_jobs)
    self.assertListEqual(dtensor_jobs, config.jobs())

    dtensor_jobs = [
        '/bns/localhost/{task_id}'.format(task_id=100 - i) for i in range(16)
    ]
    os.environ[config._DT_JOBS] = ','.join(dtensor_jobs)
    with self.assertRaisesRegex(ValueError, 'Unexpected DTENSOR_JOBS'):
      config.jobs()

  def test_jobs_with_ip_port(self):
    # The ip port format is not a bns address, and not required to sorted.
    dtensor_jobs = ['localhost:{port}'.format(port=16 - i) for i in range(16)]
    os.environ[config._DT_JOBS] = ','.join(dtensor_jobs)
    self.assertListEqual(dtensor_jobs, config.jobs())


if __name__ == '__main__':
  tf_test.main()
