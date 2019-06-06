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
"""Tests for TPUClusterResolver."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import six
from six.moves.urllib.error import URLError

from tensorflow.python import eager
from tensorflow.python.client import session
from tensorflow.python.distribute.cluster_resolver import tpu_cluster_resolver as resolver
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test
from tensorflow.python.training import server_lib
from tensorflow.python.util import compat

mock = test.mock


class MockRequestClass(object):

  def __init__(self, name, tpu_map):
    self._name = name
    self._tpu_map = tpu_map

  def execute(self):
    if self._name in self._tpu_map:
      return self._tpu_map[self._name]
    else:
      raise KeyError('Resource %s was not found' % self._name)


class MockNodeClass(object):

  def __init__(self, tpu_map):
    self._tpu_map = tpu_map

  def get(self, name):
    return MockRequestClass(name, self._tpu_map)


def mock_request_compute_metadata(cls, *args, **kwargs):
  del cls, kwargs  # Unused.
  if args[0] == 'project/project-id':
    return 'test-project'
  elif args[0] == 'instance/zone':
    return 'projects/test-project/locations/us-central1-c'
  elif args[0] == 'instance/network-interfaces/0/ip':
    return '10.128.1.2'
  return ''


def mock_is_running_in_gce():
  return True


def mock_is_not_running_in_gce():
  return False


def mock_running_in_gce_urlopen(cls, *args, **kwargs):
  del cls, args, kwargs  # Unused.
  mock_response = mock.MagicMock()
  mock_response.info.return_value = {'Metadata-Flavor': 'Google'}
  return mock_response


def mock_not_running_in_gce_urlopen(cls, *args, **kwargs):
  del cls, args, kwargs  # Unused.
  raise URLError(reason='Host does not exist.')


@test_util.run_all_in_graph_and_eager_modes
class TPUClusterResolverTest(test.TestCase):

  def _verifyClusterSpecEquality(self, cluster_spec, expected_proto):
    """Verifies that the ClusterSpec generates the correct proto.

    We are testing this four different ways to ensure that the ClusterSpec
    returned by the TPUClusterResolver behaves identically to a normal
    ClusterSpec when passed into the generic ClusterSpec libraries.

    Args:
      cluster_spec: ClusterSpec returned by the TPUClusterResolver
      expected_proto: Expected protobuf
    """
    self.assertProtoEquals(expected_proto, cluster_spec.as_cluster_def())
    self.assertProtoEquals(
        expected_proto,
        server_lib.ClusterSpec(cluster_spec).as_cluster_def())
    self.assertProtoEquals(expected_proto,
                           server_lib.ClusterSpec(
                               cluster_spec.as_cluster_def()).as_cluster_def())
    self.assertProtoEquals(expected_proto,
                           server_lib.ClusterSpec(
                               cluster_spec.as_dict()).as_cluster_def())

  def mock_service_client(self, tpu_map=None):

    if tpu_map is None:
      tpu_map = {}

    mock_locations = mock.MagicMock()
    mock_locations.nodes.return_value = MockNodeClass(tpu_map)

    mock_project = mock.MagicMock()
    mock_project.locations.return_value = mock_locations

    mock_client = mock.MagicMock()
    mock_client.projects.return_value = mock_project

    return mock_client

  @mock.patch.object(resolver, 'is_running_in_gce',
                     mock_is_running_in_gce)
  def testCheckRunningInGceWithNoTpuName(self):
    with self.assertRaisesRegexp(RuntimeError, '.*Google Cloud.*'):
      resolver.TPUClusterResolver(tpu='')

  @mock.patch.object(six.moves.urllib.request,
                     'urlopen',
                     mock_running_in_gce_urlopen)
  def testIsRunningInGce(self):
    self.assertTrue(resolver.is_running_in_gce())

  @mock.patch.object(six.moves.urllib.request,
                     'urlopen',
                     mock_not_running_in_gce_urlopen)
  def testIsNotRunningInGce(self):
    self.assertFalse(resolver.is_running_in_gce())

  @mock.patch.object(resolver.TPUClusterResolver,
                     '_request_compute_metadata', mock_request_compute_metadata)
  def testRetrieveProjectAndZoneFromMetadata(self):
    tpu_map = {
        'projects/test-project/locations/us-central1-c/nodes/test-tpu-1': {
            'ipAddress': '10.1.2.3',
            'port': '8470',
            'health': 'HEALTHY'
        }
    }

    cluster_resolver = resolver.TPUClusterResolver(
        project=None,
        zone=None,
        tpu=['test-tpu-1'],
        credentials=None,
        service=self.mock_service_client(tpu_map=tpu_map),
        coordinator_name='coordinator')

    actual_cluster_spec = cluster_resolver.cluster_spec()
    expected_proto = """
    job {
      name: 'coordinator'
      tasks { key: 0 value: '10.128.1.2:%s' }
    }
    job {
      name: 'worker'
      tasks { key: 0 value: '10.1.2.3:8470' }
    }
    """ % cluster_resolver._coordinator_port
    self._verifyClusterSpecEquality(actual_cluster_spec, str(expected_proto))
    self.assertEqual(cluster_resolver.master(), 'grpc://10.1.2.3:8470')

  @mock.patch.object(resolver.TPUClusterResolver,
                     '_request_compute_metadata', mock_request_compute_metadata)
  def testRetrieveProjectAndZoneFromMetadataNoCoordinator(self):
    tpu_map = {
        'projects/test-project/locations/us-central1-c/nodes/test-tpu-1': {
            'ipAddress': '10.1.2.3',
            'port': '8470',
            'health': 'HEALTHY'
        }
    }

    cluster_resolver = resolver.TPUClusterResolver(
        project=None,
        zone=None,
        tpu=['test-tpu-1'],
        coordinator_name=None,
        credentials=None,
        service=self.mock_service_client(tpu_map=tpu_map))

    actual_cluster_spec = cluster_resolver.cluster_spec()
    expected_proto = """
    job { name: 'worker' tasks { key: 0 value: '10.1.2.3:8470' } }
    """
    self._verifyClusterSpecEquality(actual_cluster_spec, expected_proto)
    self.assertEqual(cluster_resolver.master(), 'grpc://10.1.2.3:8470')

  @mock.patch.object(resolver.TPUClusterResolver,
                     '_request_compute_metadata', mock_request_compute_metadata)
  def testNotReadyCloudTpu(self):
    tpu_map = {
        'projects/test-project/locations/us-central1-c/nodes/test-tpu-1': {
            'ipAddress': '10.1.2.3',
            'port': '8470',
            'state': 'CREATING'
        }
    }

    cluster_resolver = resolver.TPUClusterResolver(
        project=None,
        zone=None,
        tpu='test-tpu-1',
        coordinator_name=None,
        credentials=None,
        service=self.mock_service_client(tpu_map=tpu_map))

    with self.assertRaises(RuntimeError):
      cluster_resolver.cluster_spec()

  def testSimpleSuccessfulRetrieval(self):
    tpu_map = {
        'projects/test-project/locations/us-central1-c/nodes/test-tpu-1': {
            'ipAddress': '10.1.2.3',
            'port': '8470',
            'health': 'HEALTHY'
        }
    }

    cluster_resolver = resolver.TPUClusterResolver(
        project='test-project',
        zone='us-central1-c',
        tpu=['test-tpu-1'],
        coordinator_name='coordinator',
        coordinator_address='10.128.1.5:10203',
        credentials=None,
        service=self.mock_service_client(tpu_map=tpu_map))

    actual_cluster_spec = cluster_resolver.cluster_spec()
    expected_proto = """
    job { name: 'coordinator' tasks { key: 0 value: '10.128.1.5:10203' } }
    job { name: 'worker' tasks { key: 0 value: '10.1.2.3:8470' } }
    """
    self._verifyClusterSpecEquality(actual_cluster_spec, expected_proto)
    self.assertEqual(cluster_resolver.master(), 'grpc://10.1.2.3:8470')

  def testNewNetworkEndpointFormat(self):
    tpu_map = {
        'projects/test-project/locations/us-central1-c/nodes/test-tpu-1': {
            'health': 'HEALTHY',
            'networkEndpoints': [{
                'ipAddress': '10.2.3.4',
                'port': 8470,
            }]
        }
    }

    cluster_resolver = resolver.TPUClusterResolver(
        project='test-project',
        zone='us-central1-c',
        tpu='test-tpu-1',
        coordinator_name='coordinator',
        coordinator_address='10.128.1.5:10203',
        credentials=None,
        service=self.mock_service_client(tpu_map=tpu_map))

    actual_cluster_spec = cluster_resolver.cluster_spec()
    expected_proto = """
    job { name: 'coordinator' tasks { key: 0 value: '10.128.1.5:10203' } }
    job { name: 'worker' tasks { key: 0 value: '10.2.3.4:8470' } }
    """
    self._verifyClusterSpecEquality(actual_cluster_spec, expected_proto)
    self.assertEqual('grpc://10.2.3.4:8470', cluster_resolver.master())

  @mock.patch.object(resolver.TPUClusterResolver,
                     '_request_compute_metadata', mock_request_compute_metadata)
  def testPodResolution(self):
    tpu_map = {
        'projects/test-project/locations/us-central1-c/nodes/test-tpu-1': {
            'health':
                'HEALTHY',
            'networkEndpoints': [
                {
                    'ipAddress': '10.2.3.4',
                    'port': 8470,
                },
                {
                    'ipAddress': '10.2.3.5',
                    'port': 8470,
                },
                {
                    'ipAddress': '10.2.3.6',
                    'port': 8470,
                },
                {
                    'ipAddress': '10.2.3.7',
                    'port': 8470,
                },
            ]
        }
    }

    cluster_resolver = resolver.TPUClusterResolver(
        tpu='test-tpu-1',
        credentials=None,
        service=self.mock_service_client(tpu_map=tpu_map),
        coordinator_name='coordinator')

    actual_cluster_spec = cluster_resolver.cluster_spec()
    expected_proto = """
    job {
      name: 'coordinator',
      tasks { key: 0 value: '10.128.1.2:%s'}
    }
    job {
      name: 'worker'
      tasks { key: 0 value: '10.2.3.4:8470' }
      tasks { key: 1 value: '10.2.3.5:8470' }
      tasks { key: 2 value: '10.2.3.6:8470' }
      tasks { key: 3 value: '10.2.3.7:8470' }
    }
    """ % cluster_resolver._coordinator_port
    self._verifyClusterSpecEquality(actual_cluster_spec, str(expected_proto))
    self.assertEqual(cluster_resolver.master(), 'grpc://10.2.3.4:8470')

  def testPodResolutionNoCoordinator(self):
    tpu_map = {
        'projects/test-project/locations/us-central1-c/nodes/test-tpu-1': {
            'health':
                'HEALTHY',
            'networkEndpoints': [
                {
                    'ipAddress': '10.2.3.4',
                    'port': 8470,
                },
                {
                    'ipAddress': '10.2.3.5',
                    'port': 8470,
                },
                {
                    'ipAddress': '10.2.3.6',
                    'port': 8470,
                },
                {
                    'ipAddress': '10.2.3.7',
                    'port': 8470,
                },
            ]
        }
    }

    cluster_resolver = resolver.TPUClusterResolver(
        project='test-project',
        zone='us-central1-c',
        tpu='test-tpu-1',
        coordinator_name=None,
        credentials=None,
        service=self.mock_service_client(tpu_map=tpu_map))

    actual_cluster_spec = cluster_resolver.cluster_spec()
    expected_proto = """
    job {
      name: 'worker'
      tasks { key: 0 value: '10.2.3.4:8470' }
      tasks { key: 1 value: '10.2.3.5:8470' }
      tasks { key: 2 value: '10.2.3.6:8470' }
      tasks { key: 3 value: '10.2.3.7:8470' }
    }
    """
    self._verifyClusterSpecEquality(actual_cluster_spec, expected_proto)
    self.assertEqual(cluster_resolver.master(), 'grpc://10.2.3.4:8470')

  def testGetMasterNoEntries(self):
    tpu_map = {}

    with self.assertRaises(ValueError):
      resolver.TPUClusterResolver(
          project='test-project',
          zone='us-central1-c',
          tpu=[],
          coordinator_name=None,
          credentials=None,
          service=self.mock_service_client(tpu_map=tpu_map))

  # TODO(saeta): Convert to parameterized test when included in OSS TF.
  def verifyShouldResolve(self, tpu, should_resolve):
    cluster_resolver = resolver.TPUClusterResolver(
        project='test-project',
        zone='us-central1-c',
        tpu=tpu,
        coordinator_name=None,
        credentials=None,
        service=self.mock_service_client(tpu_map={}))
    self.assertEqual(should_resolve, cluster_resolver._should_resolve(),
                     "TPU: '%s'" % tpu)

  @mock.patch.object(resolver, 'is_running_in_gce',
                     mock_is_not_running_in_gce)
  def testShouldResolveNoName(self):
    self.verifyShouldResolve('', False)

  def testShouldResolveLocal(self):
    self.verifyShouldResolve('local', False)

  def testShouldResolveLocalhost(self):
    self.verifyShouldResolve('localhost:12345', False)

  def testShouldResolveGrpc(self):
    self.verifyShouldResolve('grpc://10.1.2.3:8470', False)

  def testShouldResolveBns(self):
    self.verifyShouldResolve('/bns/foo/bar', False)

  def testShouldResolveName(self):
    self.verifyShouldResolve('mytpu', True)

  def testShouldResolveList(self):
    self.verifyShouldResolve(['myothertpu'], True)

  def testShouldResolveGrpcPrefix(self):
    self.verifyShouldResolve('grpctpu', True)

  def testNoCallComputeMetadata(self):
    cluster_resolver = resolver.TPUClusterResolver(tpu='/bns/foo/bar')
    self.assertEqual('/bns/foo/bar', cluster_resolver.master())
    self.assertEqual(None, cluster_resolver.cluster_spec())

  def testLocalhostMaster(self):
    cluster_resolver = resolver.TPUClusterResolver(tpu='localhost:12345')
    self.assertEqual('localhost:12345', cluster_resolver.master())

  def testGkeEnvironmentForDonut(self):
    os.environ['KUBE_GOOGLE_CLOUD_TPU_ENDPOINTS'] = 'grpc://10.120.27.5:8470'

    self.assertIn('KUBE_GOOGLE_CLOUD_TPU_ENDPOINTS', os.environ)
    self.assertTrue(resolver.TPUClusterResolver._in_gke())
    self.assertEqual(
        compat.as_bytes('grpc://10.120.27.5:8470'),
        compat.as_bytes(
            resolver.TPUClusterResolver._gke_endpoints()))

    cluster_resolver = resolver.TPUClusterResolver()
    self.assertEqual(
        compat.as_bytes('grpc://10.120.27.5:8470'),
        compat.as_bytes(cluster_resolver.master()))
    actual_cluster_spec = cluster_resolver.cluster_spec()
    expected_proto = """
    job {
      name: 'worker'
      tasks { key: 0 value: '10.120.27.5:8470' }
    }
    """
    self._verifyClusterSpecEquality(actual_cluster_spec, expected_proto)

    del os.environ['KUBE_GOOGLE_CLOUD_TPU_ENDPOINTS']

  def testGkeEnvironmentForPod(self):
    os.environ['KUBE_GOOGLE_CLOUD_TPU_ENDPOINTS'] = ('grpc://10.120.27.5:8470,'
                                                     'grpc://10.120.27.6:8470,'
                                                     'grpc://10.120.27.7:8470,'
                                                     'grpc://10.120.27.8:8470')

    self.assertIn('KUBE_GOOGLE_CLOUD_TPU_ENDPOINTS', os.environ)
    self.assertTrue(resolver.TPUClusterResolver._in_gke())
    self.assertEqual(
        compat.as_bytes('grpc://10.120.27.5:8470,'
                        'grpc://10.120.27.6:8470,'
                        'grpc://10.120.27.7:8470,'
                        'grpc://10.120.27.8:8470'),
        compat.as_bytes(
            resolver.TPUClusterResolver._gke_endpoints()))

    cluster_resolver = resolver.TPUClusterResolver()
    self.assertEqual(
        compat.as_bytes('grpc://10.120.27.5:8470'),
        compat.as_bytes(cluster_resolver.master()))
    actual_cluster_spec = cluster_resolver.cluster_spec()
    expected_proto = """
    job {
      name: 'worker'
      tasks { key: 0 value: '10.120.27.5:8470' }
      tasks { key: 1 value: '10.120.27.6:8470' }
      tasks { key: 2 value: '10.120.27.7:8470' }
      tasks { key: 3 value: '10.120.27.8:8470' }
    }
    """
    self._verifyClusterSpecEquality(actual_cluster_spec, expected_proto)

    del os.environ['KUBE_GOOGLE_CLOUD_TPU_ENDPOINTS']

  def testEnvironmentDiscoveryUrl(self):
    os.environ['TPU_API_DISCOVERY_URL'] = 'https://{api}.internal/{apiVersion}'
    self.assertEqual(
        'https://{api}.internal/{apiVersion}',
        (resolver.TPUClusterResolver._environment_discovery_url()))

  def testEnvironmentAndRpcDetectionForGoogle(self):
    cluster_resolver = resolver.TPUClusterResolver(tpu='/bns/ab/cd/ef')
    self.assertEqual(cluster_resolver.environment, 'google')
    self.assertEqual(cluster_resolver.rpc_layer, None)

  def testEnvironmentAndRpcDetectionForGrpcString(self):
    cluster_resolver = resolver.TPUClusterResolver(
        tpu='grpc://10.1.2.3:8470')
    self.assertEqual(cluster_resolver.environment, '')
    self.assertEqual(cluster_resolver.rpc_layer, 'grpc')
    self.assertEqual(cluster_resolver.master(), 'grpc://10.1.2.3:8470')

  def testOverrideTaskTypeAndIndexAndGetMaster(self):
    tpu_map = {
        'projects/test-project/locations/us-central1-c/nodes/test-tpu-1': {
            'health':
                'HEALTHY',
            'networkEndpoints': [
                {
                    'ipAddress': '10.2.3.4',
                    'port': 8470,
                },
                {
                    'ipAddress': '10.2.3.5',
                    'port': 8470,
                },
                {
                    'ipAddress': '10.2.3.6',
                    'port': 8470,
                },
                {
                    'ipAddress': '10.2.3.7',
                    'port': 8470,
                },
            ]
        }
    }

    cluster_resolver = resolver.TPUClusterResolver(
        project='test-project',
        zone='us-central1-c',
        tpu='test-tpu-1',
        coordinator_name=None,
        credentials=None,
        service=self.mock_service_client(tpu_map=tpu_map))

    self.assertEqual(cluster_resolver.master(), 'grpc://10.2.3.4:8470')

    cluster_resolver.task_type = 'worker'
    cluster_resolver.task_id = 3
    self.assertEqual(cluster_resolver.master(), 'grpc://10.2.3.7:8470')

    self.assertEqual(
        cluster_resolver.master(
            task_type='worker', task_id=2, rpc_layer='test'),
        'test://10.2.3.6:8470')

  def testGetDeviceDictAndCoresWithTPUs(self):
    device_names = [
        '/job:tpu_worker/task:0/device:TPU:0',
        '/job:tpu_worker/task:1/device:TPU:1',
        '/job:tpu_worker/task:2/device:TPU:0',
        '/job:tpu_worker/task:3/device:TPU:1',
        '/job:tpu_worker/task:0/device:TPU:4',
        '/job:tpu_worker/task:1/device:TPU:5',
        '/job:tpu_worker/task:2/device:TPU:4',
        '/job:tpu_worker/task:3/device:TPU:5',
    ]
    device_list = [
        session._DeviceAttributes(
            name, 'TPU', 1024, 0) for name in device_names
    ]

    device_details = resolver.TPUClusterResolver._get_device_dict_and_cores(
        device_list)
    self.assertEqual(device_details.total_cores, 8)
    self.assertEqual(device_details.device_map,
                     {'0': ['0', '4'],
                      '1': ['1', '5'],
                      '2': ['0', '4'],
                      '3': ['1', '5']})

  def testGetDeviceDictAndCoresWithCPUsAndGPUs(self):
    device_names = [
        '/job:tpu_worker/task:0/device:CPU:0',
        '/job:tpu_worker/task:1/device:CPU:0',
        '/job:tpu_worker/task:2/device:CPU:0',
        '/job:tpu_worker/task:3/device:CPU:0',
        '/job:tpu_worker/task:0/device:GPU:1',
        '/job:tpu_worker/task:1/device:GPU:1',
        '/job:tpu_worker/task:2/device:GPU:1',
        '/job:tpu_worker/task:3/device:GPU:1',
    ]
    device_list = [
        session._DeviceAttributes(
            name, 'XLA', 1024, 0) for name in device_names
    ]

    device_dict, num_cores =\
        resolver.TPUClusterResolver._get_device_dict_and_cores(device_list)
    self.assertEqual(num_cores, 0)
    self.assertEqual(device_dict, {})

  def testVerifySameCoreCount(self):
    self.assertEqual(
        resolver.TPUClusterResolver
        ._verify_and_return_same_core_count({0: [0, 1, 2, 3, 4, 5, 6, 7]}), 8)
    self.assertEqual(
        resolver.TPUClusterResolver
        ._verify_and_return_same_core_count({
            0: [0, 1],
            1: [2, 3]
        }), 2)
    with self.assertRaises(RuntimeError):
      resolver.TPUClusterResolver._verify_and_return_same_core_count(
          {
              0: [0],
              1: [1, 2]
          })

  @mock.patch.object(eager.context, 'list_devices')
  @mock.patch.object(session.BaseSession, 'list_devices')
  @mock.patch.object(resolver, 'is_running_in_gce',
                     mock_is_not_running_in_gce)
  def testNumAcceleratorsSuccess(self, mock_list_devices,
                                 mock_eager_list_devices):
    device_names = [
        '/job:tpu_worker/task:0/device:TPU:0',
        '/job:tpu_worker/task:1/device:TPU:1',
        '/job:tpu_worker/task:2/device:TPU:0',
        '/job:tpu_worker/task:3/device:TPU:1',
        '/job:tpu_worker/task:0/device:TPU:4',
        '/job:tpu_worker/task:1/device:TPU:5',
        '/job:tpu_worker/task:2/device:TPU:4',
        '/job:tpu_worker/task:3/device:TPU:5',
    ]
    device_list = [
        session._DeviceAttributes(
            name, 'TPU', 1024, 0) for name in device_names
    ]
    mock_eager_list_devices.return_value = device_names
    mock_list_devices.return_value = device_list

    cluster_resolver = resolver.TPUClusterResolver(tpu='')
    self.assertEqual(cluster_resolver.num_accelerators(), {'TPU': 2})

  @mock.patch.object(eager.context, 'list_devices')
  @mock.patch.object(session.BaseSession, 'list_devices')
  @mock.patch.object(resolver, 'is_running_in_gce',
                     mock_is_not_running_in_gce)
  def testNumAcceleratorsRetryFailure(self, mock_list_devices,
                                      mock_eager_list_devices):
    cluster_resolver = resolver.TPUClusterResolver(tpu='')
    mock_list_devices.side_effect = errors.DeadlineExceededError(
        None, None, 'timeout')
    mock_eager_list_devices.side_effect = errors.DeadlineExceededError(
        None, None, 'timeout')
    with self.assertRaises(RuntimeError):
      cluster_resolver.num_accelerators()


if __name__ == '__main__':
  test.main()
