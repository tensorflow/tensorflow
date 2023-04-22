# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for SageMakerClusterResolver."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensorflow.python.distribute.cluster_resolver.sagemaker_cluster_resolver import SageMakerClusterResolver
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test
from tensorflow.python.training import server_lib

mock = test.mock


@test_util.run_all_in_graph_and_eager_modes
class SageMakerClusterResolverTest(test.TestCase):

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

  def testNormalClusterSpecRead(self):
    os.environ['SM_HOSTS'] = '["algo-1","algo-2"]'
    os.environ['SM_CURRENT_HOST'] = 'algo-2'

    cluster_resolver = SageMakerClusterResolver()
    expected_proto = """
    job { name: 'worker' tasks { key: 0 value: 'algo-1:2223' }
                         tasks { key: 1 value: 'algo-2:2223' } }
    """
    actual_cluster_spec = cluster_resolver.cluster_spec()
    self._verifyClusterSpecEquality(actual_cluster_spec, expected_proto)

  def testAutomaticMasterRead(self):
    os.environ['SM_HOSTS'] = '["algo-1","algo-2"]'
    os.environ['SM_CURRENT_HOST'] = 'algo-1'

    cluster_resolver = SageMakerClusterResolver()
    self.assertEqual('algo-1:2223', cluster_resolver.master())

  def testSpecifiedTaskTypeAndIndexMasterRead(self):
    os.environ['SM_HOSTS'] = '["algo-1","algo-2"]'
    os.environ['SM_CURRENT_HOST'] = 'algo-2'

    cluster_resolver = SageMakerClusterResolver()
    self.assertEqual('algo-2:2223', cluster_resolver.master('worker', 1))

  def testRpcLayerRead(self):
    os.environ['SM_HOSTS'] = '["algo-1","algo-2"]'
    os.environ['SM_CURRENT_HOST'] = 'algo-1'

    cluster_resolver = SageMakerClusterResolver(rpc_layer='grpc')
    self.assertEqual('grpc://algo-1:2223', cluster_resolver.master())

  def testParameterOverrides(self):
    os.environ['SM_HOSTS'] = '["algo-1","algo-2"]'
    os.environ['SM_CURRENT_HOST'] = 'algo-1'

    cluster_resolver = SageMakerClusterResolver(task_type='worker', task_id=0)

    self.assertEqual('algo-1:2223', cluster_resolver.master())
    self.assertEqual('worker', cluster_resolver.task_type)
    self.assertEqual(0, cluster_resolver.task_id)

    cluster_resolver.task_type = 'worker'
    cluster_resolver.task_id = 1
    cluster_resolver.rpc_layer = 'test'

    self.assertEqual('test://algo-2:2223', cluster_resolver.master())
    self.assertEqual('worker', cluster_resolver.task_type)
    self.assertEqual(1, cluster_resolver.task_id)
    self.assertEqual('test', cluster_resolver.rpc_layer)

  def testTaskIndexOverride(self):
    os.environ['SM_HOSTS'] = '["algo-1","algo-2"]'
    os.environ['SM_CURRENT_HOST'] = 'algo-2'

    cluster_resolver = SageMakerClusterResolver(task_id=1)
    self.assertEqual(1, cluster_resolver.task_id)

  def testZeroItemsInClusterSpecMasterRead(self):
    os.environ['SM_HOSTS'] = ''
    os.environ['SM_CURRENT_HOST'] = ''

    cluster_resolver = SageMakerClusterResolver()
    self.assertEqual('', cluster_resolver.master())

  def testOneItemInClusterSpecMasterRead(self):
    os.environ['SM_HOSTS'] = '["algo-1"]'
    os.environ['SM_CURRENT_HOST'] = ''

    cluster_resolver = SageMakerClusterResolver()
    self.assertEqual('', cluster_resolver.master())


if __name__ == '__main__':
  test.main()
