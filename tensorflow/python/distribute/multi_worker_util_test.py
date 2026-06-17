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

    with self.assertRaisesRegex(
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

    with self.assertRaisesRegex(
        ValueError, "`task_type` 'chief' not found in cluster_spec."):
      multi_worker_util.is_chief(cluster_spec, "chief", 0)

    with self.assertRaisesRegex(
        ValueError, "The `task_id` 2 exceeds the maximum id of worker."):
      multi_worker_util.is_chief(cluster_spec, "worker", 2)

  def testEvaluatorIsChief(self):
    cluster_spec = {
        "chief": ["127.0.0.1:1234"],
        "worker": ["127.0.0.1:8964", "127.0.0.1:2333"],
        "evaluator": ["127.0.0.1:2019"]
    }
    self.assertTrue(multi_worker_util.is_chief(cluster_spec, "evaluator", 0))


class NumWorkersTest(test.TestCase):

  def testCountWorker(self):
    cluster_spec = {
        "chief": ["127.0.0.1:1234"],
        "worker": ["127.0.0.1:8964", "127.0.0.1:2333"],
        "ps": ["127.0.0.1:1926", "127.0.0.1:3141"]
    }
    self.assertEqual(
        multi_worker_util.worker_count(cluster_spec, task_type="chief"), 3)
    self.assertEqual(
        multi_worker_util.worker_count(cluster_spec, task_type="worker"), 3)

  def testCountEvaluator(self):
    cluster_spec = {
        "chief": ["127.0.0.1:1234"],
        "worker": ["127.0.0.1:8964", "127.0.0.1:2333"],
        "evaluator": ["127.0.0.1:7566"]
    }
    self.assertEqual(
        multi_worker_util.worker_count(cluster_spec, task_type="evaluator"), 1)

  def testTaskTypeNotFound(self):
    cluster_spec = {}
    with self.assertRaisesRegex(
        ValueError, "`task_type` 'worker' not found in cluster_spec."):
      multi_worker_util.worker_count(cluster_spec, task_type="worker")

  def testCountPs(self):
    cluster_spec = {
        "chief": ["127.0.0.1:1234"],
        "ps": ["127.0.0.1:1926", "127.0.0.1:3141"]
    }
    # A "ps" job shouldn't call this method.
    with self.assertRaisesRegex(ValueError, "Unexpected `task_type` 'ps'"):
      multi_worker_util.worker_count(cluster_spec, task_type="ps")


class IdInClusterTest(test.TestCase):

  def testChiefId(self):
    cluster_spec = {
        "chief": ["127.0.0.1:1234"],
        "worker": ["127.0.0.1:8964", "127.0.0.1:2333"],
        "ps": ["127.0.0.1:1926", "127.0.0.1:3141"]
    }
    self.assertEqual(
        multi_worker_util.id_in_cluster(cluster_spec, "chief", 0), 0)

  def testWorkerId(self):
    cluster_spec = {
        "chief": ["127.0.0.1:1234"],
        "worker": ["127.0.0.1:8964", "127.0.0.1:2333"],
        "ps": ["127.0.0.1:1926", "127.0.0.1:3141"]
    }
    self.assertEqual(
        multi_worker_util.id_in_cluster(cluster_spec, "worker", 1), 2)

    cluster_spec = {
        "worker": ["127.0.0.1:8964", "127.0.0.1:2333"],
        "ps": ["127.0.0.1:1926", "127.0.0.1:3141"]
    }
    self.assertEqual(
        multi_worker_util.id_in_cluster(cluster_spec, "worker", 1), 1)

  def testEvaluatorId(self):
    cluster_spec = {
        "chief": ["127.0.0.1:1234"],
        "worker": ["127.0.0.1:8964", "127.0.0.1:2333"],
        "evaluator": ["127.0.0.1:7566"]
    }
    self.assertEqual(
        multi_worker_util.id_in_cluster(cluster_spec, "evaluator", 0), 0)

  def testPsId(self):
    cluster_spec = {"chief": ["127.0.0.1:1234"], "ps": ["127.0.0.1:7566"]}
    with self.assertRaisesRegex(ValueError,
                                "There is no id for task_type 'ps'"):
      multi_worker_util.id_in_cluster(cluster_spec, "ps", 0)

  def testMultipleChiefs(self):
    cluster_spec = {
        "chief": ["127.0.0.1:8258", "127.0.0.1:7566"],
    }
    with self.assertRaisesRegex(ValueError,
                                "There must be at most one 'chief' job."):
      multi_worker_util.id_in_cluster(cluster_spec, "chief", 0)


class CollectiveLeaderTest(test.TestCase):

  def testChiefAsLeader(self):
    cluster_spec = {
        "chief": ["127.0.0.1:1234"],
        "worker": ["127.0.0.1:8964", "127.0.0.1:2333"],
        "ps": ["127.0.0.1:1926", "127.0.0.1:3141"]
    }
    self.assertEqual(
        multi_worker_util.collective_leader(cluster_spec, "worker", 0),
        "/job:chief/replica:0/task:0")

  def testWorkerAsLeader(self):
    cluster_spec = {
        "worker": ["127.0.0.1:8964", "127.0.0.1:2333"],
        "ps": ["127.0.0.1:1926", "127.0.0.1:3141"]
    }
    self.assertEqual(
        multi_worker_util.collective_leader(cluster_spec, "worker", 1),
        "/job:worker/replica:0/task:0")

  def testLeaderForEvaluator(self):
    cluster_spec = {
        "chief": ["127.0.0.1:1234"],
        "worker": ["127.0.0.1:8964", "127.0.0.1:2333"],
        "ps": ["127.0.0.1:1926", "127.0.0.1:3141"],
        "evaluator": ["127.0.0.1:2019"]
    }
    self.assertEqual(
        multi_worker_util.collective_leader(cluster_spec, "evaluator", 0), "")

  def testLocalLeader(self):
    cluster_spec = {}
    self.assertEqual(
        multi_worker_util.collective_leader(cluster_spec, None, 0), "")


# Most of the validation logic is tested by above tests except for some.
class ClusterSpecValidationTest(test.TestCase):

  def testEvaluatorNotInCluster(self):
    cluster_spec = {
        "chief": ["127.0.0.1:1234"],
        "worker": ["127.0.0.1:8964", "127.0.0.1:2333"],
        "ps": ["127.0.0.1:1926", "127.0.0.1:3141"]
    }
    multi_worker_util._validate_cluster_spec(cluster_spec, "chief", 0)
    multi_worker_util._validate_cluster_spec(cluster_spec, "worker", 0)
    multi_worker_util._validate_cluster_spec(cluster_spec, "ps", 0)
    multi_worker_util._validate_cluster_spec(cluster_spec, "evaluator", 0)

  def testWorkerNotInCluster(self):
    cluster_spec = {
        "chief": ["127.0.0.1:1234"],
        "ps": ["127.0.0.1:1926", "127.0.0.1:3141"]
    }
    multi_worker_util._validate_cluster_spec(cluster_spec, "evaluator", 0)
    with self.assertRaisesRegex(
        ValueError, "`task_type` 'worker' not found in cluster_spec."):
      multi_worker_util._validate_cluster_spec(cluster_spec, "worker", 0)


if __name__ == "__main__":
  test.main()
