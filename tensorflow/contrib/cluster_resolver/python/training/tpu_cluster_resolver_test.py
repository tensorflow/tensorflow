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

from tensorflow.contrib.cluster_resolver.python.training.tpu_cluster_resolver import TPUClusterResolver
from tensorflow.python.platform import test
from tensorflow.python.training import server_lib


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
  if args[0] == '/project/project-id':
    return 'test-project'
  elif args[0] == '/instance/zone':
    return 'projects/test-project/locations/us-central1-c'
  return ''


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
        expected_proto, server_lib.ClusterSpec(cluster_spec).as_cluster_def())
    self.assertProtoEquals(
        expected_proto,
        server_lib.ClusterSpec(cluster_spec.as_cluster_def()).as_cluster_def())
    self.assertProtoEquals(
        expected_proto,
        server_lib.ClusterSpec(cluster_spec.as_dict()).as_cluster_def())

  def mock_service_client(
      self,
      tpu_map=None):

    if tpu_map is None:
      tpu_map = {}

    mock_locations = mock.MagicMock()
    mock_locations.nodes.return_value = MockNodeClass(tpu_map)

    mock_project = mock.MagicMock()
    mock_project.locations.return_value = mock_locations

    mock_client = mock.MagicMock()
    mock_client.projects.return_value = mock_project

    return mock_client

  @mock.patch.object(TPUClusterResolver,
                     '_requestComputeMetadata',
                     mock_request_compute_metadata)
  def testRetrieveProjectAndZoneFromMetadata(self):
    tpu_map = {
        'projects/test-project/locations/us-central1-c/nodes/test-tpu-1': {
            'ipAddress': '10.1.2.3',
            'port': '8470'
        }
    }

    tpu_cluster_resolver = TPUClusterResolver(
        project=None,
        zone=None,
        tpu_names=['test-tpu-1'],
        credentials=None,
        service=self.mock_service_client(tpu_map=tpu_map))

    actual_cluster_spec = tpu_cluster_resolver.cluster_spec()
    expected_proto = """
    job { name: 'tpu_worker' tasks { key: 0 value: '10.1.2.3:8470' } }
    """
    self._verifyClusterSpecEquality(actual_cluster_spec, expected_proto)

  def testSimpleSuccessfulRetrieval(self):
    tpu_map = {
        'projects/test-project/locations/us-central1-c/nodes/test-tpu-1': {
            'ipAddress': '10.1.2.3',
            'port': '8470'
        }
    }

    tpu_cluster_resolver = TPUClusterResolver(
        project='test-project',
        zone='us-central1-c',
        tpu_names=['test-tpu-1'],
        credentials=None,
        service=self.mock_service_client(tpu_map=tpu_map))

    actual_cluster_spec = tpu_cluster_resolver.cluster_spec()
    expected_proto = """
    job { name: 'tpu_worker' tasks { key: 0 value: '10.1.2.3:8470' } }
    """
    self._verifyClusterSpecEquality(actual_cluster_spec, expected_proto)

  def testMultipleSuccessfulRetrieval(self):
    tpu_map = {
        'projects/test-project/locations/us-central1-c/nodes/test-tpu-1': {
            'ipAddress': '10.1.2.3',
            'port': '8470'
        },
        'projects/test-project/locations/us-central1-c/nodes/test-tpu-2': {
            'ipAddress': '10.4.5.6',
            'port': '8470'
        }
    }

    tpu_cluster_resolver = TPUClusterResolver(
        project='test-project',
        zone='us-central1-c',
        tpu_names=['test-tpu-2', 'test-tpu-1'],
        credentials=None,
        service=self.mock_service_client(tpu_map=tpu_map))

    actual_cluster_spec = tpu_cluster_resolver.cluster_spec()
    expected_proto = """
    job { name: 'tpu_worker' tasks { key: 0 value: '10.4.5.6:8470' }
                             tasks { key: 1 value: '10.1.2.3:8470' } }
    """
    self._verifyClusterSpecEquality(actual_cluster_spec, expected_proto)

  def testGetMasterMultipleEntries(self):
    tpu_map = {
        'projects/test-project/locations/us-central1-c/nodes/test-tpu-1': {
            'ipAddress': '10.1.2.3',
            'port': '8470'
        },
        'projects/test-project/locations/us-central1-c/nodes/test-tpu-2': {
            'ipAddress': '10.4.5.6',
            'port': '8470'
        }
    }

    tpu_cluster_resolver = TPUClusterResolver(
        project='test-project',
        zone='us-central1-c',
        tpu_names=['test-tpu-2', 'test-tpu-1'],
        credentials=None,
        service=self.mock_service_client(tpu_map=tpu_map))
    self.assertEqual('grpc://10.4.5.6:8470', tpu_cluster_resolver.get_master())

  def testGetMasterNoEntries(self):
    tpu_map = {}

    tpu_cluster_resolver = TPUClusterResolver(
        project='test-project',
        zone='us-central1-c',
        tpu_names=[],
        credentials=None,
        service=self.mock_service_client(tpu_map=tpu_map))
    with self.assertRaises(ValueError):
      tpu_cluster_resolver.get_master()

if __name__ == '__main__':
  test.main()
