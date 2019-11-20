# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for Cluster Resolvers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python import framework
from tensorflow.python.client import session
from tensorflow.python.distribute.cluster_resolver import ClusterResolver
from tensorflow.python.distribute.cluster_resolver import SimpleClusterResolver
from tensorflow.python.distribute.cluster_resolver import UnionClusterResolver
from tensorflow.python.eager.context import LogicalDevice
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test
from tensorflow.python.training import server_lib

mock = test.mock


class MockBaseClusterResolver(ClusterResolver):

  def cluster_spec(self):
    return None

  def master(self, task_type=None, task_id=None, rpc_layer=None):
    return ""

  def environment(self):
    return ""


@test_util.run_all_in_graph_and_eager_modes
class BaseClusterResolverTest(test.TestCase):

  @mock.patch.object(framework.config, "list_logical_devices")
  @mock.patch.object(session.BaseSession, "list_devices")
  def testNumAcceleratorsSuccess(self, mock_list_devices,
                                 mock_eager_list_devices):
    devices = [
        LogicalDevice("/job:worker/task:0/device:GPU:0", "GPU"),
        LogicalDevice("/job:worker/task:0/device:GPU:1", "GPU"),
        LogicalDevice("/job:worker/task:0/device:GPU:2", "GPU"),
        LogicalDevice("/job:worker/task:0/device:GPU:3", "GPU"),
    ]
    device_list = [
        session._DeviceAttributes(d.name, d.device_type, 1024, 0)
        for d in devices
    ]
    mock_eager_list_devices.return_value = devices
    mock_list_devices.return_value = device_list

    resolver = MockBaseClusterResolver()
    self.assertEqual(resolver.num_accelerators(), {"GPU": 4})

  @mock.patch.object(framework.config, "list_logical_devices")
  @mock.patch.object(session.BaseSession, "list_devices")
  def testNumAcceleratorsMultiDeviceSuccess(self, mock_list_devices,
                                            mock_eager_list_devices):
    devices = [
        LogicalDevice("/job:worker/task:0/device:TPU:0", "TPU"),
        LogicalDevice("/job:worker/task:0/device:TPU:1", "TPU"),
        LogicalDevice("/job:worker/task:0/device:TPU:2", "TPU"),
        LogicalDevice("/job:worker/task:0/device:TPU:3", "TPU"),
        LogicalDevice("/job:worker/task:0/device:GPU:0", "GPU"),
        LogicalDevice("/job:worker/task:0/device:GPU:1", "GPU"),
        LogicalDevice("/job:worker/task:0/device:GPU:2", "GPU"),
        LogicalDevice("/job:worker/task:0/device:GPU:3", "GPU"),
    ]
    device_list = [
        session._DeviceAttributes(d.name, d.device_type, 1024, 0)
        for d in devices
    ]
    mock_eager_list_devices.return_value = devices
    mock_list_devices.return_value = device_list

    resolver = MockBaseClusterResolver()
    self.assertEqual(resolver.num_accelerators(), {"TPU": 4, "GPU": 4})

  @mock.patch.object(framework.config, "list_logical_devices")
  @mock.patch.object(session.BaseSession, "list_devices")
  def testNumAcceleratorsFilterTasks(self, mock_list_devices,
                                     mock_eager_list_devices):
    devices = [
        LogicalDevice("/job:worker1/task:0/device:TPU:0", "TPU"),
        LogicalDevice("/job:worker1/task:0/device:TPU:1", "TPU"),
        LogicalDevice("/job:worker1/task:0/device:GPU:0", "GPU"),
        LogicalDevice("/job:worker1/task:0/device:GPU:1", "GPU"),
        LogicalDevice("/job:worker2/task:1/device:TPU:2", "TPU"),
        LogicalDevice("/job:worker2/task:2/device:TPU:3", "TPU"),
        LogicalDevice("/job:worker2/task:3/device:GPU:2", "GPU"),
        LogicalDevice("/job:worker2/task:4/device:GPU:3", "GPU"),
    ]
    device_list = [
        session._DeviceAttributes(d.name, d.device_type, 1024, 0)
        for d in devices
    ]
    mock_eager_list_devices.return_value = devices
    mock_list_devices.return_value = device_list

    resolver = MockBaseClusterResolver()
    self.assertEqual(resolver.num_accelerators(task_type="worker1", task_id=0),
                     {"TPU": 2, "GPU": 2})
    self.assertEqual(resolver.num_accelerators(task_type="worker2", task_id=3),
                     {"GPU": 1})
    self.assertEqual(resolver.num_accelerators(task_type="worker2", task_id=4),
                     {"GPU": 1})


class UnionClusterResolverTest(test.TestCase):
  # TODO(frankchn): Transform to parameterized test after it is included in the
  # TF open source codebase.

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

  def testSingleClusterResolver(self):
    base_cluster_spec = server_lib.ClusterSpec({
        "ps": ["ps0:2222", "ps1:2222"],
        "worker": ["worker0:2222", "worker1:2222", "worker2:2222"]
    })
    simple_resolver = SimpleClusterResolver(base_cluster_spec)
    union_resolver = UnionClusterResolver(simple_resolver)

    expected_proto = """
    job { name: 'ps' tasks { key: 0 value: 'ps0:2222' }
                     tasks { key: 1 value: 'ps1:2222' } }
    job { name: 'worker' tasks { key: 0 value: 'worker0:2222' }
                         tasks { key: 1 value: 'worker1:2222' }
                         tasks { key: 2 value: 'worker2:2222' } }
    """
    actual_cluster_spec = union_resolver.cluster_spec()
    self._verifyClusterSpecEquality(actual_cluster_spec, expected_proto)

  def testInitSimpleClusterResolver(self):
    base_cluster_spec = server_lib.ClusterSpec({
        "ps": ["ps0:2222", "ps1:2222"],
        "worker": ["worker0:2222", "worker1:2222", "worker2:2222"]
    })

    simple_resolver = SimpleClusterResolver(base_cluster_spec, task_type="ps",
                                            task_id=1, environment="cloud",
                                            num_accelerators={"GPU": 8},
                                            rpc_layer="grpc")

    self.assertEqual(simple_resolver.task_type, "ps")
    self.assertEqual(simple_resolver.task_id, 1)
    self.assertEqual(simple_resolver.environment, "cloud")
    self.assertEqual(simple_resolver.num_accelerators(), {"GPU": 8})
    self.assertEqual(simple_resolver.rpc_layer, "grpc")

  def testOverrideSimpleClusterResolver(self):
    base_cluster_spec = server_lib.ClusterSpec({
        "ps": ["ps0:2222", "ps1:2222"],
        "worker": ["worker0:2222", "worker1:2222", "worker2:2222"]
    })

    simple_resolver = SimpleClusterResolver(base_cluster_spec, task_type="ps",
                                            task_id=1, environment="cloud",
                                            num_accelerators={"GPU": 8},
                                            rpc_layer="grpc")

    simple_resolver.task_type = "worker"
    simple_resolver.task_id = 2
    simple_resolver.rpc_layer = "http"

    self.assertEqual(simple_resolver.task_type, "worker")
    self.assertEqual(simple_resolver.task_id, 2)
    self.assertEqual(simple_resolver.rpc_layer, "http")

  def testSimpleOverrideMasterWithTaskIndexZero(self):
    base_cluster_spec = server_lib.ClusterSpec({
        "ps": ["ps0:2222", "ps1:2222"],
        "worker": ["worker0:2222", "worker1:2222", "worker2:2222"]
    })

    simple_resolver = SimpleClusterResolver(base_cluster_spec)
    actual_master = simple_resolver.master("worker", 0, rpc_layer="grpc")
    self.assertEqual(actual_master, "grpc://worker0:2222")

  def testSimpleOverrideMasterWithRpcLayer(self):
    base_cluster_spec = server_lib.ClusterSpec({
        "ps": ["ps0:2222", "ps1:2222"],
        "worker": ["worker0:2222", "worker1:2222", "worker2:2222"]
    })

    simple_resolver = SimpleClusterResolver(base_cluster_spec)
    actual_master = simple_resolver.master("worker", 2, rpc_layer="grpc")
    self.assertEqual(actual_master, "grpc://worker2:2222")

  def testSimpleOverrideMaster(self):
    base_cluster_spec = server_lib.ClusterSpec({
        "ps": ["ps0:2222", "ps1:2222"],
        "worker": ["worker0:2222", "worker1:2222", "worker2:2222"]
    })

    simple_resolver = SimpleClusterResolver(base_cluster_spec)
    actual_master = simple_resolver.master("worker", 2)
    self.assertEqual(actual_master, "worker2:2222")

  def testUnionClusterResolverGetProperties(self):
    cluster_spec_1 = server_lib.ClusterSpec({
        "ps": ["ps0:2222", "ps1:2222"],
        "worker": ["worker0:2222", "worker1:2222", "worker2:2222"]
    })
    resolver1 = SimpleClusterResolver(cluster_spec_1, task_type="ps",
                                      task_id=1, environment="cloud",
                                      num_accelerators={"GPU": 8},
                                      rpc_layer="grpc")

    cluster_spec_2 = server_lib.ClusterSpec({
        "ps": ["ps2:2222", "ps3:2222"],
        "worker": ["worker3:2222", "worker4:2222", "worker5:2222"]
    })
    resolver2 = SimpleClusterResolver(cluster_spec_2, task_type="worker",
                                      task_id=2, environment="local",
                                      num_accelerators={"GPU": 16},
                                      rpc_layer="http")

    union_resolver = UnionClusterResolver(resolver1, resolver2)

    self.assertEqual(union_resolver.task_type, "ps")
    self.assertEqual(union_resolver.task_id, 1)
    self.assertEqual(union_resolver.environment, "cloud")
    self.assertEqual(union_resolver.num_accelerators(), {"GPU": 8})
    self.assertEqual(union_resolver.rpc_layer, "grpc")

    union_resolver.task_type = "worker"
    union_resolver.task_id = 2
    union_resolver.rpc_layer = "http"

    self.assertEqual(union_resolver.task_type, "worker")
    self.assertEqual(union_resolver.task_id, 2)
    self.assertEqual(union_resolver.rpc_layer, "http")

  def testTwoNonOverlappingJobMergedClusterResolver(self):
    cluster_spec_1 = server_lib.ClusterSpec({
        "ps": [
            "ps0:2222",
            "ps1:2222"
        ]
    })
    cluster_spec_2 = server_lib.ClusterSpec({
        "worker": [
            "worker0:2222",
            "worker1:2222",
            "worker2:2222"
        ]
    })
    cluster_resolver_1 = SimpleClusterResolver(cluster_spec_1)
    cluster_resolver_2 = SimpleClusterResolver(cluster_spec_2)

    union_cluster = UnionClusterResolver(cluster_resolver_1, cluster_resolver_2)
    cluster_spec = union_cluster.cluster_spec()

    expected_proto = """
    job { name: 'ps' tasks { key: 0 value: 'ps0:2222' }
                     tasks { key: 1 value: 'ps1:2222' } }
    job { name: 'worker' tasks { key: 0 value: 'worker0:2222' }
                         tasks { key: 1 value: 'worker1:2222' }
                         tasks { key: 2 value: 'worker2:2222' } }
    """
    self._verifyClusterSpecEquality(cluster_spec, expected_proto)

  def testMergedClusterResolverMaster(self):
    cluster_spec_1 = server_lib.ClusterSpec({
        "ps": [
            "ps0:2222",
            "ps1:2222"
        ]
    })
    cluster_spec_2 = server_lib.ClusterSpec({
        "worker": [
            "worker0:2222",
            "worker1:2222",
            "worker2:2222"
        ]
    })
    cluster_resolver_1 = SimpleClusterResolver(cluster_spec_1)
    cluster_resolver_2 = SimpleClusterResolver(cluster_spec_2)

    union_cluster = UnionClusterResolver(cluster_resolver_1, cluster_resolver_2)

    unspecified_master = union_cluster.master()
    self.assertEqual(unspecified_master, "")

    specified_master = union_cluster.master("worker", 1)
    self.assertEqual(specified_master, "worker1:2222")

    rpc_master = union_cluster.master("worker", 1, rpc_layer="grpc")
    self.assertEqual(rpc_master, "grpc://worker1:2222")

  def testOverlappingJobMergedClusterResolver(self):
    cluster_spec_1 = server_lib.ClusterSpec({
        "worker": [
            "worker4:2222",
            "worker5:2222"
        ]
    })
    cluster_spec_2 = server_lib.ClusterSpec({
        "worker": [
            "worker0:2222",
            "worker1:2222",
            "worker2:2222"
        ]
    })
    cluster_resolver_1 = SimpleClusterResolver(cluster_spec_1)
    cluster_resolver_2 = SimpleClusterResolver(cluster_spec_2)

    union_cluster = UnionClusterResolver(cluster_resolver_1, cluster_resolver_2)
    cluster_spec = union_cluster.cluster_spec()

    expected_proto = """
    job { name: 'worker' tasks { key: 0 value: 'worker4:2222' }
                         tasks { key: 1 value: 'worker5:2222' }
                         tasks { key: 2 value: 'worker0:2222' }
                         tasks { key: 3 value: 'worker1:2222' }
                         tasks { key: 4 value: 'worker2:2222' } }
    """
    self._verifyClusterSpecEquality(cluster_spec, expected_proto)

  def testOverlappingSparseJobMergedClusterResolverThrowError(self):
    cluster_spec_1 = server_lib.ClusterSpec({
        "worker": {
            7: "worker4:2222",
            9: "worker5:2222"
        }
    })
    cluster_spec_2 = server_lib.ClusterSpec({
        "worker": {
            3: "worker0:2222",
            6: "worker1:2222",
            7: "worker2:2222"
        }
    })
    cluster_resolver_1 = SimpleClusterResolver(cluster_spec_1)
    cluster_resolver_2 = SimpleClusterResolver(cluster_spec_2)

    union_cluster = UnionClusterResolver(cluster_resolver_1, cluster_resolver_2)
    self.assertRaises(KeyError, union_cluster.cluster_spec)

  def testOverlappingDictAndListThrowError(self):
    cluster_spec_1 = server_lib.ClusterSpec({
        "worker": [
            "worker4:2222",
            "worker5:2222"
        ]
    })
    cluster_spec_2 = server_lib.ClusterSpec({
        "worker": {
            1: "worker0:2222",
            2: "worker1:2222",
            3: "worker2:2222"
        }
    })
    cluster_resolver_1 = SimpleClusterResolver(cluster_spec_1)
    cluster_resolver_2 = SimpleClusterResolver(cluster_spec_2)

    union_cluster = UnionClusterResolver(cluster_resolver_1, cluster_resolver_2)
    self.assertRaises(KeyError, union_cluster.cluster_spec)

  def testOverlappingJobNonOverlappingKey(self):
    cluster_spec_1 = server_lib.ClusterSpec({
        "worker": {
            5: "worker4:2222",
            9: "worker5:2222"
        }
    })
    cluster_spec_2 = server_lib.ClusterSpec({
        "worker": {
            3: "worker0:2222",
            6: "worker1:2222",
            7: "worker2:2222"
        }
    })
    cluster_resolver_1 = SimpleClusterResolver(cluster_spec_1)
    cluster_resolver_2 = SimpleClusterResolver(cluster_spec_2)

    union_cluster = UnionClusterResolver(cluster_resolver_1, cluster_resolver_2)
    cluster_spec = union_cluster.cluster_spec()

    expected_proto = """
    job { name: 'worker' tasks { key: 3 value: 'worker0:2222' }
                         tasks { key: 5 value: 'worker4:2222' }
                         tasks { key: 6 value: 'worker1:2222' }
                         tasks { key: 7 value: 'worker2:2222' }
                         tasks { key: 9 value: 'worker5:2222' }}
    """
    self._verifyClusterSpecEquality(cluster_spec, expected_proto)

  def testMixedModeNonOverlappingKey(self):
    cluster_spec_1 = server_lib.ClusterSpec({
        "worker": [
            "worker4:2222",
            "worker5:2222"
        ]
    })
    cluster_spec_2 = server_lib.ClusterSpec({
        "worker": {
            3: "worker0:2222",
            6: "worker1:2222",
            7: "worker2:2222"
        }
    })
    cluster_resolver_1 = SimpleClusterResolver(cluster_spec_1)
    cluster_resolver_2 = SimpleClusterResolver(cluster_spec_2)

    union_cluster = UnionClusterResolver(cluster_resolver_1, cluster_resolver_2)
    cluster_spec = union_cluster.cluster_spec()

    expected_proto = """
    job { name: 'worker' tasks { key: 0 value: 'worker4:2222' }
                         tasks { key: 1 value: 'worker5:2222' }
                         tasks { key: 3 value: 'worker0:2222' }
                         tasks { key: 6 value: 'worker1:2222' }
                         tasks { key: 7 value: 'worker2:2222' }}
    """
    self._verifyClusterSpecEquality(cluster_spec, expected_proto)

  def testRetainSparseJobWithNoMerging(self):
    base_cluster_spec = server_lib.ClusterSpec({
        "worker": {
            1: "worker0:2222",
            3: "worker1:2222",
            5: "worker2:2222"
        }
    })

    base_cluster_resolver = SimpleClusterResolver(base_cluster_spec)
    union_cluster = UnionClusterResolver(base_cluster_resolver)
    cluster_spec = union_cluster.cluster_spec()

    expected_proto = """
    job { name: 'worker' tasks { key: 1 value: 'worker0:2222' }
                         tasks { key: 3 value: 'worker1:2222' }
                         tasks { key: 5 value: 'worker2:2222' } }
    """
    self._verifyClusterSpecEquality(cluster_spec, expected_proto)


# TODO(saeta): Include tests for master resolution

if __name__ == "__main__":
  test.main()
