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
"""Tests for SlurmClusterResolver."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensorflow.python.distribute.cluster_resolver import SlurmClusterResolver
from tensorflow.python.platform import test
from tensorflow.python.training import server_lib

mock = test.mock


class SlurmClusterResolverTest(test.TestCase):

  def mock_resolve_hostnames_output(self):
    return ['t02n13', 't02n41', 't02n43', 't02n44']

  def _verifyClusterSpecEquality(self, cluster_spec, expected_proto):
    self.assertProtoEquals(expected_proto, cluster_spec.as_cluster_def())
    self.assertProtoEquals(
        expected_proto,
        server_lib.ClusterSpec(cluster_spec).as_cluster_def())
    self.assertProtoEquals(
        expected_proto,
        server_lib.ClusterSpec(cluster_spec.as_cluster_def()).as_cluster_def())
    self.assertProtoEquals(
        expected_proto,
        server_lib.ClusterSpec(cluster_spec.as_dict()).as_cluster_def())

  @mock.patch.dict(os.environ, {'SLURM_PROCID': '0', 'SLURM_NTASKS': '3'})
  @mock.patch.object(SlurmClusterResolver, '_resolve_hostnames',
                     mock_resolve_hostnames_output)
  def testSimpleSuccessfulRetrieval(self):
    slurm_cluster_resolver = SlurmClusterResolver(
        jobs={
            'ps': 1,
            'worker': 2
        },
        port_base=8888,
        tasks_per_node=1,
        gpus_per_node=1,
        gpus_per_task=1,
        auto_set_gpu=False)

    actual_cluster_spec = slurm_cluster_resolver.cluster_spec()
    expected_proto = """
    job { name: 'ps' tasks { value: 't02n13:8888' } }
    job { name: 'worker' tasks { key: 0 value: 't02n41:8888' }
                         tasks { key: 1 value: 't02n43:8888' } }
    """
    self._verifyClusterSpecEquality(actual_cluster_spec, expected_proto)

  @mock.patch.dict(os.environ, {'SLURM_PROCID': '0', 'SLURM_NTASKS': '3'})
  @mock.patch.object(SlurmClusterResolver, '_resolve_hostnames',
                     mock_resolve_hostnames_output)
  def testSimpleMasterRetrieval(self):
    slurm_cluster_resolver = SlurmClusterResolver(
        jobs={
            'ps': 1,
            'worker': 2
        },
        port_base=8888,
        tasks_per_node=1,
        gpus_per_node=1,
        gpus_per_task=1,
        auto_set_gpu=False)

    slurm_cluster_resolver.task_type = 'worker'
    slurm_cluster_resolver.task_index = 1
    self.assertEqual(slurm_cluster_resolver.master(), 'grpc://t02n43:8888')

    slurm_cluster_resolver.rpc_layer = 'ab'
    self.assertEqual(slurm_cluster_resolver.master('ps', 0), 'ab://t02n13:8888')
    self.assertEqual(
        slurm_cluster_resolver.master('ps', 0, rpc_layer='test'),
        'test://t02n13:8888')

  @mock.patch.dict(os.environ, {
      'SLURM_PROCID': '0',
      'SLURM_NTASKS': '3',
      'SLURM_NTASKS_PER_NODE': '1'
  })
  @mock.patch.object(SlurmClusterResolver, '_resolve_hostnames',
                     mock_resolve_hostnames_output)
  def testTaskPerNodeNotSetRetrieval(self):
    slurm_cluster_resolver = SlurmClusterResolver(
        jobs={
            'ps': 1,
            'worker': 2
        },
        port_base=8888,
        gpus_per_node=1,
        gpus_per_task=1,
        auto_set_gpu=False)

    actual_cluster_spec = slurm_cluster_resolver.cluster_spec()
    expected_proto = """
    job { name: 'ps' tasks { value: 't02n13:8888' } }
    job { name: 'worker' tasks { key: 0 value: 't02n41:8888' }
                         tasks { key: 1 value: 't02n43:8888' } }
    """
    self._verifyClusterSpecEquality(actual_cluster_spec, expected_proto)

  @mock.patch.dict(
      os.environ, {
          'SLURM_PROCID': '1',
          'SLURM_NTASKS': '5',
          'SLURM_NTASKS_PER_NODE': '2',
          'CUDA_VISIBLE_DEVICES': ''
      })
  @mock.patch.object(SlurmClusterResolver, '_resolve_hostnames',
                     mock_resolve_hostnames_output)
  def testMultiTaskPerNodeRetrieval(self):
    slurm_cluster_resolver = SlurmClusterResolver(
        jobs={
            'ps': 1,
            'worker': 4
        },
        port_base=8888,
        gpus_per_node=2,
        gpus_per_task=1,
        auto_set_gpu=True)

    actual_cluster_spec = slurm_cluster_resolver.cluster_spec()
    expected_proto = """
    job { name: 'ps' tasks { value: 't02n13:8888' } }
    job { name: 'worker' tasks { key: 0 value: 't02n13:8889' }
                         tasks { key: 1 value: 't02n41:8888' }
                         tasks { key: 2 value: 't02n41:8889' }
                         tasks { key: 3 value: 't02n43:8888' } }
    """

    self._verifyClusterSpecEquality(actual_cluster_spec, expected_proto)
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '1'

  @mock.patch.dict(
      os.environ, {
          'SLURM_PROCID': '1',
          'SLURM_NTASKS': '5',
          'SLURM_NTASKS_PER_NODE': '2',
          'CUDA_VISIBLE_DEVICES': ''
      })
  @mock.patch.object(SlurmClusterResolver, '_resolve_hostnames',
                     mock_resolve_hostnames_output)
  def testMultipleGpusPerTaskRetrieval(self):
    slurm_cluster_resolver = SlurmClusterResolver(
        jobs={
            'ps': 1,
            'worker': 4
        },
        port_base=8888,
        gpus_per_node=4,
        gpus_per_task=2,
        auto_set_gpu=True)

    actual_cluster_spec = slurm_cluster_resolver.cluster_spec()
    expected_proto = """
    job { name: 'ps' tasks { value: 't02n13:8888' } }
    job { name: 'worker' tasks { key: 0 value: 't02n13:8889' }
                         tasks { key: 1 value: 't02n41:8888' }
                         tasks { key: 2 value: 't02n41:8889' }
                         tasks { key: 3 value: 't02n43:8888' } }
    """

    self._verifyClusterSpecEquality(actual_cluster_spec, expected_proto)
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '2,3'


if __name__ == '__main__':
  test.main()
