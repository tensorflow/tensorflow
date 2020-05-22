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
"""Tests for GCEClusterResolver."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.distribute.cluster_resolver.cluster_resolver import UnionClusterResolver
from tensorflow.python.distribute.cluster_resolver.gce_cluster_resolver import GCEClusterResolver
from tensorflow.python.platform import test
from tensorflow.python.training import server_lib


mock = test.mock


class GCEClusterResolverTest(test.TestCase):

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

  def standard_mock_instance_groups(self, instance_map=None):
    if instance_map is None:
      instance_map = [
          {'instance': 'https://gce.example.com/res/gce-instance-1'}
      ]

    mock_instance_group_request = mock.MagicMock()
    mock_instance_group_request.execute.return_value = {
        'items': instance_map
    }

    service_attrs = {
        'listInstances.return_value': mock_instance_group_request,
        'listInstances_next.return_value': None,
    }
    mock_instance_groups = mock.Mock(**service_attrs)
    return mock_instance_groups

  def standard_mock_instances(self, instance_to_ip_map=None):
    if instance_to_ip_map is None:
      instance_to_ip_map = {
          'gce-instance-1': '10.123.45.67'
      }

    mock_get_request = mock.MagicMock()
    mock_get_request.execute.return_value = {
        'networkInterfaces': [
            {'networkIP': '10.123.45.67'}
        ]
    }

    def get_side_effect(project, zone, instance):
      del project, zone  # Unused

      if instance in instance_to_ip_map:
        mock_get_request = mock.MagicMock()
        mock_get_request.execute.return_value = {
            'networkInterfaces': [
                {'networkIP': instance_to_ip_map[instance]}
            ]
        }
        return mock_get_request
      else:
        raise RuntimeError('Instance %s not found!' % instance)

    service_attrs = {
        'get.side_effect': get_side_effect,
    }
    mock_instances = mock.MagicMock(**service_attrs)
    return mock_instances

  def standard_mock_service_client(
      self,
      mock_instance_groups=None,
      mock_instances=None):

    if mock_instance_groups is None:
      mock_instance_groups = self.standard_mock_instance_groups()
    if mock_instances is None:
      mock_instances = self.standard_mock_instances()

    mock_client = mock.MagicMock()
    mock_client.instanceGroups.return_value = mock_instance_groups
    mock_client.instances.return_value = mock_instances
    return mock_client

  def gen_standard_mock_service_client(self, instances=None):
    name_to_ip = {}
    instance_list = []
    for instance in instances:
      name_to_ip[instance['name']] = instance['ip']
      instance_list.append({
          'instance': 'https://gce.example.com/gce/res/' + instance['name']
      })

    mock_instance = self.standard_mock_instances(name_to_ip)
    mock_instance_group = self.standard_mock_instance_groups(instance_list)

    return self.standard_mock_service_client(mock_instance_group, mock_instance)

  def testSimpleSuccessfulRetrieval(self):
    gce_cluster_resolver = GCEClusterResolver(
        project='test-project',
        zone='us-east1-d',
        instance_group='test-instance-group',
        port=8470,
        credentials=None,
        service=self.standard_mock_service_client())

    actual_cluster_spec = gce_cluster_resolver.cluster_spec()
    expected_proto = """
    job { name: 'worker' tasks { key: 0 value: '10.123.45.67:8470' } }
    """
    self._verifyClusterSpecEquality(actual_cluster_spec, expected_proto)

  def testMasterRetrieval(self):
    gce_cluster_resolver = GCEClusterResolver(
        project='test-project',
        zone='us-east1-d',
        instance_group='test-instance-group',
        task_id=0,
        port=8470,
        credentials=None,
        service=self.standard_mock_service_client())
    self.assertEqual(gce_cluster_resolver.master(), 'grpc://10.123.45.67:8470')

  def testMasterRetrievalWithCustomTasks(self):
    name_to_ip = [
        {'name': 'instance1', 'ip': '10.1.2.3'},
        {'name': 'instance2', 'ip': '10.2.3.4'},
        {'name': 'instance3', 'ip': '10.3.4.5'},
    ]

    gce_cluster_resolver = GCEClusterResolver(
        project='test-project',
        zone='us-east1-d',
        instance_group='test-instance-group',
        port=8470,
        credentials=None,
        service=self.gen_standard_mock_service_client(name_to_ip))

    self.assertEqual(
        gce_cluster_resolver.master('worker', 2, 'test'),
        'test://10.3.4.5:8470')

  def testOverrideParameters(self):
    name_to_ip = [
        {'name': 'instance1', 'ip': '10.1.2.3'},
        {'name': 'instance2', 'ip': '10.2.3.4'},
        {'name': 'instance3', 'ip': '10.3.4.5'},
    ]

    gce_cluster_resolver = GCEClusterResolver(
        project='test-project',
        zone='us-east1-d',
        instance_group='test-instance-group',
        task_type='testworker',
        port=8470,
        credentials=None,
        service=self.gen_standard_mock_service_client(name_to_ip))

    gce_cluster_resolver.task_id = 1
    gce_cluster_resolver.rpc_layer = 'test'

    self.assertEqual(gce_cluster_resolver.task_type, 'testworker')
    self.assertEqual(gce_cluster_resolver.task_id, 1)
    self.assertEqual(gce_cluster_resolver.rpc_layer, 'test')
    self.assertEqual(gce_cluster_resolver.master(), 'test://10.2.3.4:8470')

  def testOverrideParametersWithZeroOrEmpty(self):
    name_to_ip = [
        {'name': 'instance1', 'ip': '10.1.2.3'},
        {'name': 'instance2', 'ip': '10.2.3.4'},
        {'name': 'instance3', 'ip': '10.3.4.5'},
    ]

    gce_cluster_resolver = GCEClusterResolver(
        project='test-project',
        zone='us-east1-d',
        instance_group='test-instance-group',
        task_type='',
        task_id=1,
        port=8470,
        credentials=None,
        service=self.gen_standard_mock_service_client(name_to_ip))

    self.assertEqual(gce_cluster_resolver.master(
        task_type='', task_id=0), 'grpc://10.1.2.3:8470')

  def testCustomJobNameAndPortRetrieval(self):
    gce_cluster_resolver = GCEClusterResolver(
        project='test-project',
        zone='us-east1-d',
        instance_group='test-instance-group',
        task_type='custom',
        port=2222,
        credentials=None,
        service=self.standard_mock_service_client())

    actual_cluster_spec = gce_cluster_resolver.cluster_spec()
    expected_proto = """
    job { name: 'custom' tasks { key: 0 value: '10.123.45.67:2222' } }
    """
    self._verifyClusterSpecEquality(actual_cluster_spec, expected_proto)

  def testMultipleInstancesRetrieval(self):
    name_to_ip = [
        {'name': 'instance1', 'ip': '10.1.2.3'},
        {'name': 'instance2', 'ip': '10.2.3.4'},
        {'name': 'instance3', 'ip': '10.3.4.5'},
    ]

    gce_cluster_resolver = GCEClusterResolver(
        project='test-project',
        zone='us-east1-d',
        instance_group='test-instance-group',
        port=8470,
        credentials=None,
        service=self.gen_standard_mock_service_client(name_to_ip))

    actual_cluster_spec = gce_cluster_resolver.cluster_spec()
    expected_proto = """
    job { name: 'worker' tasks { key: 0 value: '10.1.2.3:8470' }
                         tasks { key: 1 value: '10.2.3.4:8470' }
                         tasks { key: 2 value: '10.3.4.5:8470' } }
    """
    self._verifyClusterSpecEquality(actual_cluster_spec, expected_proto)

  def testUnionMultipleInstanceRetrieval(self):
    worker1_name_to_ip = [
        {'name': 'instance1', 'ip': '10.1.2.3'},
        {'name': 'instance2', 'ip': '10.2.3.4'},
        {'name': 'instance3', 'ip': '10.3.4.5'},
    ]

    worker2_name_to_ip = [
        {'name': 'instance4', 'ip': '10.4.5.6'},
        {'name': 'instance5', 'ip': '10.5.6.7'},
        {'name': 'instance6', 'ip': '10.6.7.8'},
    ]

    ps_name_to_ip = [
        {'name': 'ps1', 'ip': '10.100.1.2'},
        {'name': 'ps2', 'ip': '10.100.2.3'},
    ]

    worker1_gce_cluster_resolver = GCEClusterResolver(
        project='test-project',
        zone='us-east1-d',
        instance_group='test-instance-group',
        task_type='worker',
        port=8470,
        credentials=None,
        service=self.gen_standard_mock_service_client(worker1_name_to_ip))

    worker2_gce_cluster_resolver = GCEClusterResolver(
        project='test-project',
        zone='us-east1-d',
        instance_group='test-instance-group',
        task_type='worker',
        port=8470,
        credentials=None,
        service=self.gen_standard_mock_service_client(worker2_name_to_ip))

    ps_gce_cluster_resolver = GCEClusterResolver(
        project='test-project',
        zone='us-east1-d',
        instance_group='test-instance-group',
        task_type='ps',
        port=2222,
        credentials=None,
        service=self.gen_standard_mock_service_client(ps_name_to_ip))

    union_cluster_resolver = UnionClusterResolver(worker1_gce_cluster_resolver,
                                                  worker2_gce_cluster_resolver,
                                                  ps_gce_cluster_resolver)

    actual_cluster_spec = union_cluster_resolver.cluster_spec()
    expected_proto = """
    job { name: 'ps' tasks { key: 0 value: '10.100.1.2:2222' }
                     tasks { key: 1 value: '10.100.2.3:2222' } }
    job { name: 'worker' tasks { key: 0 value: '10.1.2.3:8470' }
                         tasks { key: 1 value: '10.2.3.4:8470' }
                         tasks { key: 2 value: '10.3.4.5:8470' }
                         tasks { key: 3 value: '10.4.5.6:8470' }
                         tasks { key: 4 value: '10.5.6.7:8470' }
                         tasks { key: 5 value: '10.6.7.8:8470' } }
    """
    self._verifyClusterSpecEquality(actual_cluster_spec, expected_proto)

if __name__ == '__main__':
  test.main()
