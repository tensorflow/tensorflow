# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for multi_worker_util."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.core.protobuf import cluster_pb2
from tensorflow.python.distribute import multi_worker_util
from tensorflow.python.eager import test
from tensorflow.python.training import server_lib


class NormalizeClusterSpecTest(test.TestCase):

  def assert_same_cluster(self, lhs, rhs):
    self.assertEqual(
        server_lib.ClusterSpec(lhs).as_dict(),
        server_lib.ClusterSpec(rhs).as_dict())

  def testDictAsInput(self):
    cluster_spec = {
        "chief": ["127.0.0.1:1234"],
        "worker": ["127.0.0.1:8964", "127.0.0.1:2333"],
        "ps": ["127.0.0.1:1926", "127.0.0.1:3141"]
    }
    self.assert_same_cluster(
        cluster_spec, multi_worker_util.normalize_cluster_spec(cluster_spec))

  def testClusterDefAsInput(self):
    cluster_def = cluster_pb2.ClusterDef()
    job = cluster_def.job.add()
    job.name = "chief"
    job.tasks[0] = "127.0.0.1:1234"

    job = cluster_def.job.add()
    job.name = "worker"
    job.tasks[0] = "127.0.0.1:8964"
    job.tasks[1] = "127.0.0.1:2333"

    job = cluster_def.job.add()
    job.name = "ps"
    job.tasks[0] = "127.0.0.1:1926"
    job.tasks[1] = "127.0.0.1:3141"

    self.assert_same_cluster(
        cluster_def, multi_worker_util.normalize_cluster_spec(cluster_def))

  def testClusterSpecAsInput(self):
    cluster_spec = server_lib.ClusterSpec({
        "chief": ["127.0.0.1:1234"],
        "worker": ["127.0.0.1:8964", "127.0.0.1:2333"],
        "ps": ["127.0.0.1:1926", "127.0.0.1:3141"]
    })
    self.assert_same_cluster(
        cluster_spec, multi_worker_util.normalize_cluster_spec(cluster_spec))

  def testUnexpectedInput(self):
    cluster_spec = ["127.0.0.1:8964", "127.0.0.1:2333"]

    with self.assertRaisesRegexp(
        ValueError,
        "`cluster_spec' should be dict or a `tf.train.ClusterSpec` or a "
        "`tf.train.ClusterDef` object"):
      multi_worker_util.normalize_cluster_spec(cluster_spec)


class IsChiefTest(test.TestCase):

  def testClusterWithChief(self):
    cluster_spec = {
        "chief": ["127.0.0.1:1234"],
        "worker": ["127.0.0.1:8964", "127.0.0.1:2333"],
        "ps": ["127.0.0.1:1926", "127.0.0.1:3141"]
    }
    self.assertTrue(multi_worker_util.is_chief(cluster_spec, "chief", 0))
    self.assertFalse(multi_worker_util.is_chief(cluster_spec, "worker", 0))

  def testClusterWithoutChief(self):
    cluster_spec = {"worker": ["127.0.0.1:8964", "127.0.0.1:2333"]}
    self.assertTrue(multi_worker_util.is_chief(cluster_spec, "worker", 0))
    self.assertFalse(multi_worker_util.is_chief(cluster_spec, "worker", 1))

    with self.assertRaisesRegexp(
        ValueError, "The task_type \"chief\" is not in the `cluster_spec`."):
      multi_worker_util.is_chief(cluster_spec, "chief", 0)

    with self.assertRaisesRegexp(
        ValueError, "The `task_id` 2 exceeds the maximum id of worker."):
      multi_worker_util.is_chief(cluster_spec, "worker", 2)


if __name__ == "__main__":
  test.main()
