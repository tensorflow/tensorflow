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
"""Test for MultiWorkerMirroredStrategy backed by DTensor API."""

import json
import os

from absl import flags

from tensorflow.dtensor.python.tests import multi_client_test_util
from tensorflow.dtensor.python.tests import test_backend_util
from tensorflow.dtensor.python.tests import test_util
from tensorflow.python.distribute.cluster_resolver import tfconfig_cluster_resolver
from tensorflow.python.distribute.experimental import multi_worker_mirrored_strategy as mwms
from tensorflow.python.framework import config
from tensorflow.python.platform import test as tf_test


class StrategyCreationTest(tf_test.TestCase):

  def setUp(self):
    super().setUp()
    self.num_client = flags.FLAGS.num_clients
    self.num_local_devices = flags.FLAGS.num_local_devices

  def test_strategy_creation_with_default_cluster_resolver(self):
    strategy = mwms.MultiWorkerMirroredStrategy()
    mesh = strategy._mesh
    self.assertIsNotNone(mesh)
    self.assertLen(mesh.global_device_ids(),
                   self.num_client * self.num_local_devices)
    self.assertLen(mesh.local_device_ids(), self.num_local_devices)
    self.assertIsInstance(strategy._cluster_resolver,
                          tfconfig_cluster_resolver.TFConfigClusterResolver)

  def test_invalid_init_arguments(self):
    mesh = object()
    cluster_resolver = tfconfig_cluster_resolver.TFConfigClusterResolver()

    with self.assertRaisesRegex(
        ValueError,
        'Mesh and cluster_resolver can not be provided at the same time'):
      mwms.MultiWorkerMirroredStrategy(
          mesh=mesh,
          cluster_resolver=cluster_resolver)

  def test_parse_dtensor_env_var_from_cluster_resolver(self):
    cluster_resolver = tfconfig_cluster_resolver.TFConfigClusterResolver()

    dtensor_env_vars = mwms._parse_dtensor_env_var_from_cluster_resolver(
        cluster_resolver)

    tf_config = json.loads(os.environ['TF_CONFIG'])
    worker_jobs = ','.join(tf_config['cluster']['worker'])
    client_id = tf_config['task']['index']

    self.assertLen(dtensor_env_vars, 4)
    self.assertEqual(dtensor_env_vars['DTENSOR_JOBS'], worker_jobs)
    self.assertEqual(dtensor_env_vars['DTENSOR_NUM_CLIENTS'],
                     str(self.num_client))
    self.assertEqual(dtensor_env_vars['DTENSOR_CLIENT_ID'], client_id)
    self.assertEqual(dtensor_env_vars['DTENSOR_JOB_NAME'], 'worker')


def client_config_function(config_params):
  client_id = config_params['client_id']
  worker_jobs = config_params['worker_jobs']
  num_devices = config_params['num_devices']

  os.environ['TF_CONFIG'] = json.dumps({
      'cluster': {
          'worker': worker_jobs
      },
      'task': {'type': 'worker', 'index': f'{client_id}'}
  })

  if config.list_physical_devices('GPU'):
    device_type = 'GPU'
  elif test_util.is_tpu_present():
    device_type = 'TPU'
  else:
    device_type = 'CPU'

  # reset_logical_devices
  test_util.reset_context()
  if device_type != 'TPU':
    # Configure virtual devices. This does not initialize the TensorFlow
    # context.
    test_util.reset_logical_devices(device_type, num_devices)

  # Validates the correct number of devices are created.
  logical_devices = test_util.list_local_logical_devices(device_type)
  assert len(logical_devices) == num_devices, (
      logical_devices,
      f'Test is mis-configured: expecting {num_devices} logical_devices.')


if __name__ == '__main__':
  test_backend_util.handle_test_main(
      multi_client_test_util.multi_client_main, client_config_function)
