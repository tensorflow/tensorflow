# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for BrainJobsClusterResolver."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from tensorflow.python.distribute.cluster_resolver.brain_jobs_cluster_resolver import BrainJobsClusterResolver
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test
from tensorflow.python.training import server_lib


@test_util.run_all_in_graph_and_eager_modes
class BrainJobsClusterResolverTest(test.TestCase):

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
    brain_job_flags = ('chief|/bns/atlanta/borg/atlanta/bns/my_user/my_job.'
                       'chief|1,worker|/bns/atlanta/borg/atlanta/bns/my_user/'
                       'my_job.worker|2')

    cluster_resolver = BrainJobsClusterResolver(brain_jobs=brain_job_flags,
                                                brain_port=1234,
                                                brain_rpc_layer='rpc2',
                                                task_type='chief',
                                                task_id=0)
    expected_proto = """
    job { name: 'chief' tasks { key: 0 value: '/bns/atlanta/borg/atlanta/bns/my_user/my_job.chief/0' }}
    job { name: 'worker' tasks { key: 0 value: '/bns/atlanta/borg/atlanta/bns/my_user/my_job.worker/0'},
                         tasks { key: 1 value: '/bns/atlanta/borg/atlanta/bns/my_user/my_job.worker/1'}}
    """
    actual_cluster_spec = cluster_resolver.cluster_spec()
    self._verifyClusterSpecEquality(actual_cluster_spec, expected_proto)
    self.assertEqual('', cluster_resolver.rpc_layer)
    self.assertEqual(1234, cluster_resolver.port)
    self.assertEqual('/bns/atlanta/borg/atlanta/bns/my_user/my_job.chief/0',
                     cluster_resolver.master())


if __name__ == '__main__':
  test.main()
