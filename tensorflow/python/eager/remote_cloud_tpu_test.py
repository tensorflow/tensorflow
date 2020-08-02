# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Test that we can connect to a real Cloud TPU."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
from absl.testing import absltest

from tensorflow.python.distribute.cluster_resolver import tpu_cluster_resolver
from tensorflow.python.eager import remote
from tensorflow.python.framework import config
from tensorflow.python.tpu import tpu_strategy_util

FLAGS = flags.FLAGS
flags.DEFINE_string('tpu', '', 'Name of TPU to connect to.')
flags.DEFINE_string('project', None, 'Name of GCP project with TPU.')
flags.DEFINE_string('zone', None, 'Name of GCP zone with TPU.')

flags.DEFINE_integer('num_tpu_devices', 8, 'The expected number of TPUs.')
DEVICES_PER_TASK = 8

EXPECTED_DEVICES_PRE_CONNECT = [
    '/device:CPU:0',
    '/device:XLA_CPU:0',
]
EXPECTED_NEW_DEVICES_AFTER_CONNECT_TEMPLATES = [
    '/job:worker/replica:0/task:{task}/device:CPU:0',
    '/job:worker/replica:0/task:{task}/device:XLA_CPU:0',
    '/job:worker/replica:0/task:{task}/device:TPU_SYSTEM:0',
    '/job:worker/replica:0/task:{task}/device:TPU:0',
    '/job:worker/replica:0/task:{task}/device:TPU:1',
    '/job:worker/replica:0/task:{task}/device:TPU:2',
    '/job:worker/replica:0/task:{task}/device:TPU:3',
    '/job:worker/replica:0/task:{task}/device:TPU:4',
    '/job:worker/replica:0/task:{task}/device:TPU:5',
    '/job:worker/replica:0/task:{task}/device:TPU:6',
    '/job:worker/replica:0/task:{task}/device:TPU:7',
]


class RemoteCloudTPUTest(absltest.TestCase):
  """Test that we can connect to a real Cloud TPU."""

  def test_connect(self):
    # Log full diff on failure.
    self.maxDiff = None  # pylint:disable=invalid-name

    self.assertCountEqual(
        EXPECTED_DEVICES_PRE_CONNECT,
        [device.name for device in config.list_logical_devices()])

    resolver = tpu_cluster_resolver.TPUClusterResolver(
        tpu=FLAGS.tpu, zone=FLAGS.zone, project=FLAGS.project
    )
    remote.connect_to_cluster(resolver)

    expected_devices = EXPECTED_DEVICES_PRE_CONNECT
    for task in range(FLAGS.num_tpu_devices // DEVICES_PER_TASK):
      expected_devices.extend([
          template.format(task=task)
          for template in EXPECTED_NEW_DEVICES_AFTER_CONNECT_TEMPLATES
      ])

    self.assertCountEqual(
        expected_devices,
        [device.name for device in config.list_logical_devices()])

    tpu_strategy_util.initialize_tpu_system(resolver)

if __name__ == '__main__':
  absltest.main()
