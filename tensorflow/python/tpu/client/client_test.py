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
# Lint as: python3
"""Tests for cloud tpu client."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensorflow.python.platform import test
from tensorflow.python.tpu.client import client

mock = test.mock


def mock_request_compute_metadata(path):
  if path == 'project/project-id':
    return 'test-project'
  elif path == 'instance/zone':
    return 'projects/test-project/locations/us-central1-c'
  elif path == 'instance/network-interfaces/0/ip':
    return '10.128.1.2'
  return ''


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


class CloudTpuClientTest(test.TestCase):

  def setUp(self):
    super(CloudTpuClientTest, self).setUp()
    if 'TPU_API_DISCOVERY_URL' in os.environ:
      del os.environ['TPU_API_DISCOVERY_URL']
    if 'TPU_NAME' in os.environ:
      del os.environ['TPU_NAME']

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

  def testEnvironmentDiscoveryUrl(self):
    os.environ['TPU_API_DISCOVERY_URL'] = 'https://{api}.internal/{apiVersion}'
    self.assertEqual('https://{api}.internal/{apiVersion}',
                     (client._environment_discovery_url()))

  def testEnvironmentVarToNetworkEndpointsSingleIp(self):
    self.assertEqual(
        [{'ipAddress': '1.2.3.4', 'port': '1234'}],
        list(client._environment_var_to_network_endpoints(
            '1.2.3.4:1234')))

  def testEnvironmentVarToNetworkEndpointsSingleGrpcAddress(self):
    self.assertEqual(
        [{'ipAddress': '1.2.3.4', 'port': '2000'}],
        list(
            client._environment_var_to_network_endpoints(
                'grpc://1.2.3.4:2000')))

  def testEnvironmentVarToNetworkEndpointsMultipleIps(self):
    self.assertEqual(
        [{'ipAddress': '1.2.3.4', 'port': '2000'},
         {'ipAddress': '5.6.7.8', 'port': '1234'}],
        list(
            client._environment_var_to_network_endpoints(
                '1.2.3.4:2000,5.6.7.8:1234')))

  def testEnvironmentVarToNetworkEndpointsMultipleGrpcAddresses(self):
    self.assertEqual(
        [{'ipAddress': '1.2.3.4', 'port': '2000'},
         {'ipAddress': '5.6.7.8', 'port': '1234'}],
        list(client._environment_var_to_network_endpoints(
            'grpc://1.2.3.4:2000,grpc://5.6.7.8:1234')))

  def testEnvironmentVarToNetworkEndpointsMissingPortAndMixed(self):
    self.assertEqual(
        [{'ipAddress': '1.2.3.4', 'port': '2000'},
         {'ipAddress': '5.6.7.8', 'port': '8470'}],
        list(client._environment_var_to_network_endpoints(
            '1.2.3.4:2000,grpc://5.6.7.8')))

  def testInitializeNoArguments(self):
    with self.assertRaisesRegex(
        ValueError, 'Please provide a TPU Name to connect to.'):
      client.Client()

  def testInitializeMultiElementTpuArray(self):
    with self.assertRaisesRegex(
        NotImplementedError,
        'Using multiple TPUs in a single session is not yet implemented'):
      client.Client(tpu=['multiple', 'elements'])

  def assertClientContains(self, c):
    self.assertEqual('tpu_name', c._tpu)
    self.assertEqual(True, c._use_api)
    self.assertEqual(None, c._credentials)
    self.assertEqual('test-project', c._project)
    self.assertEqual('us-central1-c', c._zone)
    self.assertEqual(None, c._discovery_url)
    self.assertEqual([{
        'ipAddress': '10.1.2.3',
        'port': '8470'
    }], c.network_endpoints())

  @mock.patch.object(client, '_request_compute_metadata',
                     mock_request_compute_metadata)
  def testNetworkEndpointsNotReadyWithApi(self):
    tpu_map = {
        'projects/test-project/locations/us-central1-c/nodes/tpu_name': {
            'ipAddress': '10.1.2.3',
            'port': '8470',
        }
    }
    c = client.Client(
        tpu='tpu_name', service=self.mock_service_client(tpu_map=tpu_map))
    self.assertRaisesRegex(
        RuntimeError, 'TPU .* is not yet ready; state: "None"',
        c.network_endpoints)

  @mock.patch.object(client, '_request_compute_metadata',
                     mock_request_compute_metadata)
  def testInitializeNoArgumentsWithEnvironmentVariable(self):
    os.environ['TPU_NAME'] = 'tpu_name'
    tpu_map = {
        'projects/test-project/locations/us-central1-c/nodes/tpu_name': {
            'ipAddress': '10.1.2.3',
            'port': '8470',
            'state': 'READY',
            'health': 'HEALTHY',
        }
    }
    c = client.Client(
        service=self.mock_service_client(tpu_map=tpu_map))
    self.assertClientContains(c)

  @mock.patch.object(client, '_request_compute_metadata',
                     mock_request_compute_metadata)
  def testInitializeTpuName(self):
    tpu_map = {
        'projects/test-project/locations/us-central1-c/nodes/tpu_name': {
            'ipAddress': '10.1.2.3',
            'port': '8470',
            'state': 'READY',
            'health': 'HEALTHY',
        }
    }
    c = client.Client(
        tpu='tpu_name', service=self.mock_service_client(tpu_map=tpu_map))
    self.assertClientContains(c)

  @mock.patch.object(client, '_request_compute_metadata',
                     mock_request_compute_metadata)
  def testInitializeIpAddress(self):
    c = client.Client(tpu='grpc://1.2.3.4:8470')
    self.assertEqual('grpc://1.2.3.4:8470', c._tpu)
    self.assertEqual(False, c._use_api)
    self.assertEqual(None, c._service)
    self.assertEqual(None, c._credentials)
    self.assertEqual(None, c._project)
    self.assertEqual(None, c._zone)
    self.assertEqual(None, c._discovery_url)
    self.assertEqual([{
        'ipAddress': '1.2.3.4',
        'port': '8470'
    }], c.network_endpoints())

  def testInitializeWithoutMetadata(self):
    c = client.Client(
        tpu='tpu_name', project='project', zone='zone')
    self.assertEqual('tpu_name', c._tpu)
    self.assertEqual(True, c._use_api)
    self.assertEqual(None, c._service)
    self.assertEqual(None, c._credentials)
    self.assertEqual('project', c._project)
    self.assertEqual('zone', c._zone)
    self.assertEqual(None, c._discovery_url)

  def testRecoverableNoApiAccess(self):
    c = client.Client(tpu='grpc://1.2.3.4:8470')
    self.assertEqual(True, c.recoverable())

  @mock.patch.object(client, '_request_compute_metadata',
                     mock_request_compute_metadata)
  def testRecoverableNoState(self):
    tpu_map = {
        'projects/test-project/locations/us-central1-c/nodes/tpu_name': {
            'ipAddress': '10.1.2.3',
            'port': '8470',
        }
    }
    c = client.Client(
        tpu='tpu_name', service=self.mock_service_client(tpu_map=tpu_map))
    self.assertEqual(True, c.recoverable())

  @mock.patch.object(client, '_request_compute_metadata',
                     mock_request_compute_metadata)
  def testRecoverableReady(self):
    tpu_map = {
        'projects/test-project/locations/us-central1-c/nodes/tpu_name': {
            'ipAddress': '10.1.2.3',
            'port': '8470',
            'state': 'READY',
        }
    }
    c = client.Client(
        tpu='tpu_name', service=self.mock_service_client(tpu_map=tpu_map))
    self.assertEqual(True, c.recoverable())

  @mock.patch.object(client, '_request_compute_metadata',
                     mock_request_compute_metadata)
  def testRecoverablePreempted(self):
    tpu_map = {
        'projects/test-project/locations/us-central1-c/nodes/tpu_name': {
            'ipAddress': '10.1.2.3',
            'port': '8470',
            'state': 'PREEMPTED',
        }
    }
    c = client.Client(
        tpu='tpu_name', service=self.mock_service_client(tpu_map=tpu_map))
    self.assertEqual(False, c.recoverable())

  @mock.patch.object(client, '_request_compute_metadata',
                     mock_request_compute_metadata)
  def testHealthApi(self):
    tpu_map = {
        'projects/test-project/locations/us-central1-c/nodes/tpu_name': {
            'ipAddress': '10.1.2.3',
            'port': '8470',
            'state': 'PREEMPTED',
            'health': 'HEALTHY',
            'acceleratorType': 'v3-8',
            'tensorflowVersion': 'nightly',
        }
    }
    c = client.Client(
        tpu='tpu_name', service=self.mock_service_client(tpu_map=tpu_map))
    self.assertEqual('HEALTHY', c.health())

  @mock.patch.object(client, '_request_compute_metadata',
                     mock_request_compute_metadata)
  def testRuntimeVersionApi(self):
    tpu_map = {
        'projects/test-project/locations/us-central1-c/nodes/tpu_name': {
            'ipAddress': '10.1.2.3',
            'port': '8470',
            'state': 'PREEMPTED',
            'health': 'HEALTHY',
            'acceleratorType': 'v3-8',
            'tensorflowVersion': 'nightly',
        }
    }
    c = client.Client(
        tpu='tpu_name', service=self.mock_service_client(tpu_map=tpu_map))
    self.assertEqual('nightly', c.runtime_version())

  @mock.patch.object(client, '_request_compute_metadata',
                     mock_request_compute_metadata)
  def testAcceleratorTypeApi(self):
    tpu_map = {
        'projects/test-project/locations/us-central1-c/nodes/tpu_name': {
            'ipAddress': '10.1.2.3',
            'port': '8470',
            'state': 'PREEMPTED',
            'health': 'HEALTHY',
            'acceleratorType': 'v3-8',
            'tensorflowVersion': 'nightly',
        }
    }
    c = client.Client(
        tpu='tpu_name', service=self.mock_service_client(tpu_map=tpu_map))
    self.assertEqual('v3-8', c.accelerator_type())

  def testHandlesByteStrings(self):
    self.assertEqual(
        client.Client(
            tpu='tpu_name', zone='zone', project='project')._full_name(),
        client.Client(
            tpu=b'tpu_name', zone=b'zone', project=b'project')._full_name(),
    )


if __name__ == '__main__':
  test.main()
