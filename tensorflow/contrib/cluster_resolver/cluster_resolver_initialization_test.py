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
"""Tests to ensure ClusterResolvers are usable via the old contrib path."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.cluster_resolver import SimpleClusterResolver
from tensorflow.contrib.cluster_resolver.python.training import cluster_resolver
from tensorflow.contrib.cluster_resolver.python.training import UnionClusterResolver
from tensorflow.python.platform import test
from tensorflow.python.training import server_lib


class ClusterResolverInitializationTest(test.TestCase):

  def testCreateSimpleClusterResolverFromLib(self):
    base_cluster_spec = server_lib.ClusterSpec({
        "ps": ["ps0:2222", "ps1:2222"],
        "worker": ["worker0:2222", "worker1:2222", "worker2:2222"]
    })
    cluster_resolver.SimpleClusterResolver(base_cluster_spec)

  def testCreateSimpleClusterResolver(self):
    base_cluster_spec = server_lib.ClusterSpec({
        "ps": ["ps0:2222", "ps1:2222"],
        "worker": ["worker0:2222", "worker1:2222", "worker2:2222"]
    })
    SimpleClusterResolver(base_cluster_spec)

  def testCreateUnionClusterResolver(self):
    base_cluster_spec = server_lib.ClusterSpec({
        "ps": ["ps0:2222", "ps1:2222"],
        "worker": ["worker0:2222", "worker1:2222", "worker2:2222"]
    })
    simple_cr = SimpleClusterResolver(base_cluster_spec)
    UnionClusterResolver(simple_cr)

if __name__ == "__main__":
  test.main()
