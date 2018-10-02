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

from tensorflow.contrib.cluster_resolver.python.training.cluster_resolver import SimpleClusterResolver
from tensorflow.contrib.cluster_resolver.python.training.cluster_resolver import UnionClusterResolver
from tensorflow.python.platform import test
from tensorflow.python.training import server_lib


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
