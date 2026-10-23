# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for K8sClusterResolver."""

import sys

from tensorflow.python.distribute.cluster_resolver.kubernetes_cluster_resolver import ExecutableLocation
from tensorflow.python.distribute.cluster_resolver.kubernetes_cluster_resolver import KubernetesClusterResolver
from tensorflow.python.platform import test
from tensorflow.python.training import server_lib

mock = test.mock


def _mock_kubernetes_module():
  sys.modules['kubernetes'] = mock.MagicMock()


def _mock_kubernetes_client(ret):
  mock_client = mock.MagicMock()
  mock_client.list_pod_for_all_namespaces.side_effect = (
      lambda *args, **kwargs: ret[kwargs['label_selector']])
  return mock_client


def _get_mock_pod_item(name, phase, host_ip):
  mock_status = mock.Mock()
  mock_status.configure_mock(phase=phase, host_ip=host_ip)

  mock_metadata = mock.Mock()
  mock_metadata.configure_mock(name=name)

  mock_item = mock.Mock()
  mock_item.configure_mock(status=mock_status, metadata=mock_metadata)
  return mock_item


def _create_pod_list(*args):
  return mock.MagicMock(items=[_get_mock_pod_item(*x) for x in args])


class KubernetesClusterResolverTest(test.TestCase):

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

  def testSingleItemSuccessfulRetrievalInCluster(self):
    ret = _create_pod_list(
        ('tensorflow-abc123', 'Running', '10.1.2.3'),
    )

    cluster_resolver = KubernetesClusterResolver(
        override_client=_mock_kubernetes_client({'job-name=tensorflow': ret}),
        executable_location=ExecutableLocation.WITHIN_CLUSTER,
    )

    actual_cluster_spec = cluster_resolver.cluster_spec()
    expected_proto = """
    job {
      name: 'worker'
      tasks { key: 0 value: '10.1.2.3:8470' }
    }
    """
    self._verifyClusterSpecEquality(actual_cluster_spec, str(expected_proto))

  def testValueErrorRaisedOnInvalidExecutableLocation(self):

    _mock_kubernetes_module()

    with self.assertRaisesRegexp(ValueError, '.*'):
      KubernetesClusterResolver(executable_location=None)

  def testSingleItemSuccessfulRetrieval(self):
    ret = _create_pod_list(('tensorflow-abc123', 'Running', '10.1.2.3'),)

    cluster_resolver = KubernetesClusterResolver(
        override_client=_mock_kubernetes_client(
            {'job-name=tensorflow': ret}))

    actual_cluster_spec = cluster_resolver.cluster_spec()
    expected_proto = """
    job {
      name: 'worker'
      tasks { key: 0 value: '10.1.2.3:8470' }
    }
    """
    self._verifyClusterSpecEquality(actual_cluster_spec, str(expected_proto))

  def testSuccessfulRetrievalWithSort(self):
    ret = _create_pod_list(
        ('tensorflow-abc123', 'Running', '10.1.2.3'),
        ('tensorflow-def456', 'Running', '10.1.2.4'),
        ('tensorflow-999999', 'Running', '10.1.2.5'))

    cluster_resolver = KubernetesClusterResolver(
        override_client=_mock_kubernetes_client(
            {'job-name=tensorflow': ret}))

    actual_cluster_spec = cluster_resolver.cluster_spec()
    expected_proto = """
    job {
      name: 'worker'
      tasks { key: 0 value: '10.1.2.5:8470' }
      tasks { key: 1 value: '10.1.2.3:8470' }
      tasks { key: 2 value: '10.1.2.4:8470' }
    }
    """
    self._verifyClusterSpecEquality(actual_cluster_spec, str(expected_proto))

  def testGetMasterWithOverrideParameters(self):
    ret = _create_pod_list(
        ('worker-0', 'Running', '10.1.2.3'),
        ('worker-1', 'Running', '10.1.2.4'),
        ('worker-2', 'Running', '10.1.2.5'))

    cluster_resolver = KubernetesClusterResolver(
        override_client=_mock_kubernetes_client(
            {'job-name=tensorflow': ret}))
    cluster_resolver.task_type = 'worker'
    cluster_resolver.task_id = 0
    self.assertEqual(cluster_resolver.task_type, 'worker')
    self.assertEqual(cluster_resolver.task_id, 0)
    self.assertEqual(cluster_resolver.master(), 'grpc://10.1.2.3:8470')
    self.assertEqual(cluster_resolver.master('worker', 2),
                     'grpc://10.1.2.5:8470')

  def testNonRunningPod(self):
    ret = _create_pod_list(('tensorflow-abc123', 'Failed', '10.1.2.3'),)

    cluster_resolver = KubernetesClusterResolver(
        override_client=_mock_kubernetes_client(
            {'job-name=tensorflow': ret}))

    error_msg = 'Pod "tensorflow-abc123" is not running; phase: "Failed"'
    with self.assertRaisesRegex(RuntimeError, error_msg):
      cluster_resolver.cluster_spec()

  def testMultiplePodSelectorsAndWorkers(self):
    worker1 = _create_pod_list(
        ('tensorflow-abc123', 'Running', '10.1.2.3'),
        ('tensorflow-def456', 'Running', '10.1.2.4'),
        ('tensorflow-999999', 'Running', '10.1.2.5'))
    worker2 = _create_pod_list(
        ('tensorflow-abc124', 'Running', '10.1.2.6'),
        ('tensorflow-def457', 'Running', '10.1.2.7'),
        ('tensorflow-999990', 'Running', '10.1.2.8'))
    ps = _create_pod_list(
        ('tensorflow-ps-1', 'Running', '10.1.2.1'),
        ('tensorflow-ps-2', 'Running', '10.1.2.2'))

    cluster_resolver = KubernetesClusterResolver(
        job_to_label_mapping={
            'worker': ['job-name=worker1', 'job-name=worker2'],
            'ps': ['job-name=ps']
        },
        override_client=_mock_kubernetes_client({
            'job-name=worker1': worker1,
            'job-name=worker2': worker2,
            'job-name=ps': ps
        }))

    actual_cluster_spec = cluster_resolver.cluster_spec()
    expected_proto = """
    job {
      name: 'ps'
      tasks { key: 0 value: '10.1.2.1:8470' }
      tasks { key: 1 value: '10.1.2.2:8470' }
    }
    job {
      name: 'worker'
      tasks { key: 0 value: '10.1.2.5:8470' }
      tasks { key: 1 value: '10.1.2.3:8470' }
      tasks { key: 2 value: '10.1.2.4:8470' }
      tasks { key: 3 value: '10.1.2.8:8470' }
      tasks { key: 4 value: '10.1.2.6:8470' }
      tasks { key: 5 value: '10.1.2.7:8470' }
    }
    """
    self._verifyClusterSpecEquality(actual_cluster_spec, str(expected_proto))

  def _mockKubernetesWithOnePod(self):
    """Mocks the kubernetes module with one running pod and returns it."""
    _mock_kubernetes_module()
    k8s_module = sys.modules['kubernetes']
    k8s_module.client.CoreV1Api.return_value = _mock_kubernetes_client({
        'job-name=tensorflow': _create_pod_list(
            ('tensorflow-abc123', 'Running', '10.1.2.3')
        ),
    })
    return k8s_module

  def _assertClusterSpecLoadsConfig(
      self, cluster_resolver, k8s_module, loaded, not_loaded
  ):
    k8s_module.config.reset_mock()

    actual_cluster_spec = cluster_resolver.cluster_spec()
    self.assertEqual(
        actual_cluster_spec.as_dict(), {'worker': ['10.1.2.3:8470']}
    )
    getattr(k8s_module.config, loaded).assert_called_once_with()
    getattr(k8s_module.config, not_loaded).assert_not_called()

  def testClusterSpecLoadsInClusterConfigWithinCluster(self):
    # The constructor loads the in-cluster config. cluster_spec() must load the
    # same config, not a kubeconfig file, which does not exist inside a pod.
    k8s_module = self._mockKubernetesWithOnePod()
    cluster_resolver = KubernetesClusterResolver(
        executable_location=ExecutableLocation.WITHIN_CLUSTER
    )
    self._assertClusterSpecLoadsConfig(
        cluster_resolver,
        k8s_module,
        loaded='load_incluster_config',
        not_loaded='load_kube_config',
    )

  def testClusterSpecLoadsKubeConfigOffCluster(self):
    k8s_module = self._mockKubernetesWithOnePod()
    cluster_resolver = KubernetesClusterResolver(
        executable_location=ExecutableLocation.OFF_CLUSTER
    )
    self._assertClusterSpecLoadsConfig(
        cluster_resolver,
        k8s_module,
        loaded='load_kube_config',
        not_loaded='load_incluster_config',
    )

  def testClusterSpecWithoutExecutableLocationAttribute(self):
    # Objects created before executable_location existed keep the old behavior.
    k8s_module = self._mockKubernetesWithOnePod()
    cluster_resolver = KubernetesClusterResolver()
    del cluster_resolver._executable_location
    self._assertClusterSpecLoadsConfig(
        cluster_resolver,
        k8s_module,
        loaded='load_kube_config',
        not_loaded='load_incluster_config',
    )


if __name__ == '__main__':
  test.main()
