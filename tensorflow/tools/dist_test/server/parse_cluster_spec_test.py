# Copyright 2016 Google Inc. All Rights Reserved.
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

"""Tests for cluster-spec string parser in GRPC TensorFlow server."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.tools.dist_test.server import grpc_tensorflow_server


class ParseClusterSpecStringTest(tf.test.TestCase):

  def setUp(self):
    self._cluster = tf.train.ServerDef(protocol="grpc").cluster

  def test_parse_multi_jobs_sunnyday(self):
    cluster_spec = ("worker|worker0:2220;worker1:2221;worker2:2222,"
                    "ps|ps0:3220;ps1:3221")

    grpc_tensorflow_server.parse_cluster_spec(cluster_spec, self._cluster)

    self.assertEqual(2, len(self._cluster.job))

    self.assertEqual("worker", self._cluster.job[0].name)
    self.assertEqual(3, len(self._cluster.job[0].tasks))
    self.assertEqual("worker0:2220", self._cluster.job[0].tasks[0])
    self.assertEqual("worker1:2221", self._cluster.job[0].tasks[1])
    self.assertEqual("worker2:2222", self._cluster.job[0].tasks[2])

    self.assertEqual("ps", self._cluster.job[1].name)
    self.assertEqual(2, len(self._cluster.job[1].tasks))
    self.assertEqual("ps0:3220", self._cluster.job[1].tasks[0])
    self.assertEqual("ps1:3221", self._cluster.job[1].tasks[1])

  def test_empty_cluster_spec_string(self):
    cluster_spec = ""

    with self.assertRaisesRegexp(ValueError,
                                 "Empty cluster_spec string"):
      grpc_tensorflow_server.parse_cluster_spec(cluster_spec, self._cluster)

  def test_parse_misused_comma_for_semicolon(self):
    cluster_spec = "worker|worker0:2220,worker1:2221"

    with self.assertRaisesRegexp(ValueError,
                                 "Not exactly one instance of \\'\\|\\'"):
      grpc_tensorflow_server.parse_cluster_spec(cluster_spec, self._cluster)

  def test_parse_misused_semicolon_for_comma(self):
    cluster_spec = "worker|worker0:2220;ps|ps0:3220"

    with self.assertRaisesRegexp(ValueError,
                                 "Not exactly one instance of \\'\\|\\'"):
      grpc_tensorflow_server.parse_cluster_spec(cluster_spec, self._cluster)

  def test_parse_empty_job_name(self):
    cluster_spec = "worker|worker0:2220,|ps0:3220"

    with self.assertRaisesRegexp(ValueError,
                                 "Empty job_name in cluster_spec"):
      grpc_tensorflow_server.parse_cluster_spec(cluster_spec, self._cluster)
      print(self._cluster)

  def test_parse_empty_task(self):
    cluster_spec = "worker|worker0:2220,ps|"

    with self.assertRaisesRegexp(ValueError,
                                 "Empty task string at position 0"):
      grpc_tensorflow_server.parse_cluster_spec(cluster_spec, self._cluster)


if __name__ == "__main__":
  tf.test.main()
