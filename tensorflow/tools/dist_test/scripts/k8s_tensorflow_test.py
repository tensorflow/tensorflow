# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tensorflow.tools.dist_test.scripts.k8s_tensorflow_lib."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.platform import googletest
from tensorflow.tools.dist_test.scripts import k8s_tensorflow_lib


class K8sTensorflowTest(googletest.TestCase):

  def testGenerateConfig_LoadBalancer(self):
    # Use loadbalancer
    config = k8s_tensorflow_lib.GenerateConfig(
        num_workers=1,
        num_param_servers=1,
        port=5000,
        request_load_balancer=True,
        docker_image='test_image',
        name_prefix='abc',
        use_shared_volume=False)
    self.assertTrue('LoadBalancer' in config)

    # Don't use loadbalancer
    config = k8s_tensorflow_lib.GenerateConfig(
        num_workers=1,
        num_param_servers=1,
        port=5000,
        request_load_balancer=False,
        docker_image='test_image',
        name_prefix='abc',
        use_shared_volume=False)
    self.assertFalse('LoadBalancer' in config)

  def testGenerateConfig_SharedVolume(self):
    # Use shared directory
    config = k8s_tensorflow_lib.GenerateConfig(
        num_workers=1,
        num_param_servers=1,
        port=5000,
        request_load_balancer=False,
        docker_image='test_image',
        name_prefix='abc',
        use_shared_volume=True)
    self.assertTrue('/shared' in config)

    # Don't use shared directory
    config = k8s_tensorflow_lib.GenerateConfig(
        num_workers=1,
        num_param_servers=1,
        port=5000,
        request_load_balancer=False,
        docker_image='test_image',
        name_prefix='abc',
        use_shared_volume=False)
    self.assertFalse('/shared' in config)

  def testEnvVar(self):
    # Use loadbalancer
    config = k8s_tensorflow_lib.GenerateConfig(
        num_workers=1,
        num_param_servers=1,
        port=5000,
        request_load_balancer=True,
        docker_image='test_image',
        name_prefix='abc',
        use_shared_volume=False,
        env_vars={'test1': 'test1_value', 'test2': 'test2_value'})
    self.assertTrue('{name: "test1", value: "test1_value"}' in config)
    self.assertTrue('{name: "test2", value: "test2_value"}' in config)

  def testClusterSpec(self):
    # Use cluster_spec
    config = k8s_tensorflow_lib.GenerateConfig(
        num_workers=1,
        num_param_servers=1,
        port=5000,
        request_load_balancer=True,
        docker_image='test_image',
        name_prefix='abc',
        use_shared_volume=False,
        use_cluster_spec=True)
    self.assertFalse('worker_hosts' in config)
    self.assertFalse('ps_hosts' in config)
    self.assertTrue(
        '"--cluster_spec=worker|abc-worker0:5000,ps|abc-ps0:5000"' in config)

    # Don't use cluster_spec
    config = k8s_tensorflow_lib.GenerateConfig(
        num_workers=1,
        num_param_servers=1,
        port=5000,
        request_load_balancer=True,
        docker_image='test_image',
        name_prefix='abc',
        use_shared_volume=False,
        use_cluster_spec=False)
    self.assertFalse('cluster_spec' in config)
    self.assertTrue('"--worker_hosts=abc-worker0:5000"' in config)
    self.assertTrue('"--ps_hosts=abc-ps0:5000"' in config)

  def testWorkerHosts(self):
    self.assertEquals(
        'test_prefix-worker0:1234',
        k8s_tensorflow_lib.WorkerHosts(1, 1234, 'test_prefix'))
    self.assertEquals(
        'test_prefix-worker0:1234,test_prefix-worker1:1234',
        k8s_tensorflow_lib.WorkerHosts(2, 1234, 'test_prefix'))

  def testPsHosts(self):
    self.assertEquals(
        'test_prefix-ps0:1234,test_prefix-ps1:1234',
        k8s_tensorflow_lib.PsHosts(2, 1234, 'test_prefix'))


if __name__ == '__main__':
  googletest.main()
