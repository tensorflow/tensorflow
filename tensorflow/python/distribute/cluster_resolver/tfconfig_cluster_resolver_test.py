# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for TFCONFIGClusterResolver."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensorflow.python import eager
from tensorflow.python.client import session
from tensorflow.python.distribute.cluster_resolver import TFConfigClusterResolver
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test
from tensorflow.python.training import server_lib

mock = test.mock


@test_util.run_all_in_graph_and_eager_modes
class TFConfigClusterResolverTest(test.TestCase):

  def _verifyClusterSpecEquality(self, cluster_spec, expected_proto):
    self.assertProtoEquals(expected_proto, cluster_spec.as_cluster_def())
    self.assertProtoEquals(
        expected_proto, server_lib.ClusterSpec(cluster_spec).as_cluster_def())
    self.assertProtoEquals(
        expected_proto,
        server_lib.ClusterSpec(cluster_spec.as_cluster_def()).as_cluster_def())
    self.assertProtoEquals(
        expected_proto,
        server_lib.ClusterSpec(cluster_spec.as_dict()).as_cluster_def())

  def testNormalClusterSpecRead(self):
    os.environ['TF_CONFIG'] = """
    {
      "cluster": {
        "ps": ["ps0:2222", "ps1:2222"],
        "worker": ["worker0:2222", "worker1:2222", "worker2:2222"]
      },
      "task": {
        "type": "ps",
        "index": 0
      }
    }
    """

    cluster_resolver = TFConfigClusterResolver()
    expected_proto = """
    job { name: 'ps' tasks { key: 0 value: 'ps0:2222' }
                     tasks { key: 1 value: 'ps1:2222' } }
    job { name: 'worker' tasks { key: 0 value: 'worker0:2222' }
                         tasks { key: 1 value: 'worker1:2222' }
                         tasks { key: 2 value: 'worker2:2222' } }
    """
    actual_cluster_spec = cluster_resolver.cluster_spec()
    self._verifyClusterSpecEquality(actual_cluster_spec, expected_proto)

  def testAutomaticMasterRead(self):
    os.environ['TF_CONFIG'] = """
    {
      "cluster": {
        "ps": ["ps0:2222", "ps1:2222"],
        "worker": ["worker0:2222", "worker1:2222", "worker2:2222"]
      },
      "task": {
        "type": "ps",
        "index": 0
      }
    }
    """

    cluster_resolver = TFConfigClusterResolver()
    self.assertEqual('ps0:2222', cluster_resolver.master())

  def testSpecifiedTaskTypeAndIndexMasterRead(self):
    os.environ['TF_CONFIG'] = """
    {
      "cluster": {
        "ps": ["ps0:2222", "ps1:2222"],
        "worker": ["worker0:2222", "worker1:2222", "worker2:2222"]
      },
      "task": {
        "type": "ps",
        "index": 0
      }
    }
    """

    cluster_resolver = TFConfigClusterResolver()
    self.assertEqual('worker1:2222', cluster_resolver.master('worker', 1))

  def testSessionMasterRead(self):
    os.environ['TF_CONFIG'] = """
    {
      "cluster": {
        "ps": ["ps0:2222", "ps1:2222"],
        "worker": ["worker0:2222", "worker1:2222", "worker2:2222"]
      },
      "session_master": "sessionmaster:2222",
      "task": {
        "type": "ps",
        "index": 0
      }
    }
    """

    cluster_resolver = TFConfigClusterResolver()
    self.assertEqual('sessionmaster:2222', cluster_resolver.master())

  def testRpcLayerRead(self):
    os.environ['TF_CONFIG'] = """
    {
      "cluster": {
        "ps": ["ps0:2222", "ps1:2222"],
        "worker": ["worker0:2222", "worker1:2222", "worker2:2222"]
      },
      "rpc_layer": "grpc",
      "task": {
        "type": "ps",
        "index": 0
      }
    }
    """

    cluster_resolver = TFConfigClusterResolver()
    self.assertEqual('grpc://ps0:2222', cluster_resolver.master())

  def testTaskTypeIndexRpcRead(self):
    os.environ['TF_CONFIG'] = """
    {
      "cluster": {
        "ps": ["ps0:2222", "ps1:2222"],
        "worker": ["worker0:2222", "worker1:2222", "worker2:2222"]
      },
      "rpc_layer": "grpc",
      "task": {
        "type": "ps",
        "index": 0
      }
    }
    """

    cluster_resolver = TFConfigClusterResolver()
    self.assertEqual('ps', cluster_resolver.task_type)
    self.assertEqual(0, cluster_resolver.task_id)
    self.assertEqual('grpc', cluster_resolver.rpc_layer)

  def testParameterOverrides(self):
    os.environ['TF_CONFIG'] = """
    {
      "cluster": {
        "ps": ["ps0:2222", "ps1:2222"],
        "worker": ["worker0:2222", "worker1:2222", "worker2:2222"]
      },
      "rpc_layer": "grpc",
      "task": {
        "type": "ps",
        "index": 1
      }
    }
    """

    cluster_resolver = TFConfigClusterResolver(task_type='ps', task_id=0)

    self.assertEqual('grpc://ps0:2222', cluster_resolver.master())
    self.assertEqual('ps', cluster_resolver.task_type)
    self.assertEqual(0, cluster_resolver.task_id)

    cluster_resolver.task_type = 'worker'
    cluster_resolver.task_id = 1
    cluster_resolver.rpc_layer = 'test'

    self.assertEqual('test://worker1:2222', cluster_resolver.master())
    self.assertEqual('worker', cluster_resolver.task_type)
    self.assertEqual(1, cluster_resolver.task_id)
    self.assertEqual('test', cluster_resolver.rpc_layer)

  def testTaskTypeCastToString(self):
    os.environ['TF_CONFIG'] = """
    {
      "cluster": {
        "123456": ["ps0:2222", "ps1:2222"],
        "worker": ["worker0:2222", "worker1:2222", "worker2:2222"]
      },
      "rpc_layer": "grpc",
      "task": {
        "type": 123456,
        "index": 0
      }
    }
    """
    cluster_resolver = TFConfigClusterResolver()
    self.assertEqual('123456', cluster_resolver.task_type)

  def testTaskIndexCastToInteger(self):
    os.environ['TF_CONFIG'] = """
    {
      "cluster": {
        "ps": ["ps0:2222", "ps1:2222"],
        "worker": ["worker0:2222", "worker1:2222", "worker2:2222"]
      },
      "rpc_layer": "grpc",
      "task": {
        "type": "ps",
        "index": "1"
      }
    }
    """
    cluster_resolver = TFConfigClusterResolver()
    self.assertEqual(1, cluster_resolver.task_id)

  def testZeroItemsInClusterSpecMasterRead(self):
    os.environ['TF_CONFIG'] = """
    {}
    """

    cluster_resolver = TFConfigClusterResolver()
    self.assertEqual('', cluster_resolver.master())

  def testOneItemInClusterSpecMasterRead(self):
    os.environ['TF_CONFIG'] = """
    {
      "cluster": {
        "worker": ["worker0:2222"]
      }
    }
    """

    cluster_resolver = TFConfigClusterResolver()
    self.assertEqual('', cluster_resolver.master())

  @mock.patch.object(eager.context, 'list_devices')
  @mock.patch.object(session.BaseSession, 'list_devices')
  def testNumAcceleratorsFilterTasksByEnvVar(self, mock_list_devices,
                                             mock_eager_list_devices):
    os.environ['TF_CONFIG'] = """
    {
      "cluster": {
        "worker1": ["w10:2222"],
        "worker2": ["w21:2222", "w22:2222", "w23:2222", "w24:2222"]
      },
      "rpc_layer": "grpc",
      "task": {
        "type": "worker1",
        "index": "0"
      }
    }
    """

    device_names = [
        '/job:worker1/task:0/device:TPU:0',
        '/job:worker1/task:0/device:TPU:1',
        '/job:worker1/task:0/device:GPU:0',
        '/job:worker1/task:0/device:GPU:1',
        '/job:worker2/task:1/device:TPU:2',
        '/job:worker2/task:2/device:TPU:3',
        '/job:worker2/task:3/device:GPU:2',
        '/job:worker2/task:4/device:GPU:3',
    ]
    device_list = [
        session._DeviceAttributes(name, name[27:30], 1024, 0)
        for name in device_names
    ]
    mock_eager_list_devices.return_value = device_names
    mock_list_devices.return_value = device_list

    resolver = TFConfigClusterResolver()

    # By default we read from TF_CONFIG
    self.assertEqual(resolver.num_accelerators(), {'TPU': 2, 'GPU': 2})

    # Override still works when we want it to
    self.assertEqual(resolver.num_accelerators(task_type='worker2', task_id=3),
                     {'GPU': 1})


if __name__ == '__main__':
  test.main()
