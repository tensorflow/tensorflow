# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for multi-process clusters."""

from tensorflow.python.distribute import multi_process_runner
from tensorflow.python.distribute import multi_worker_test_base
from tensorflow.python.eager import context
from tensorflow.python.eager import remote
from tensorflow.python.eager import test


class MultiProcessClusterTest(test.TestCase):

  def setUp(self):
    super(MultiProcessClusterTest, self).setUp()
    self._cluster = multi_worker_test_base.create_multi_process_cluster(
        num_workers=2, num_ps=1, has_chief=True, rpc_layer="grpc")
    remote.connect_to_cluster(
        self._cluster.cluster_resolver.cluster_spec(), protocol="grpc")
    context.ensure_initialized()

  def testClusterIsAlive(self):
    self.assertTrue(context.check_alive("/job:worker/replica:0/task:0"))
    self.assertTrue(context.check_alive("/job:worker/replica:0/task:1"))
    self.assertTrue(context.check_alive("/job:ps/replica:0/task:0"))
    self.assertTrue(context.check_alive("/job:chief/replica:0/task:0"))

  def testKillAndStartTask(self):
    self.assertTrue(context.check_alive("/job:worker/replica:0/task:0"))

    # It is not allowed to start a task before killing it.
    with self.assertRaises(ValueError):
      self._cluster.start_task("worker", 0)

    self._cluster.kill_task("worker", 0)
    self.assertFalse(context.check_alive("/job:worker/replica:0/task:0"))

    # The task is already killed.
    with self.assertRaises(ValueError):
      self._cluster.kill_task("worker", 0)

    self._cluster.start_task("worker", 0)

    # Without a call to update_server_def, the next check_alive will return
    # False. Alternatively sleeping for 2 seconds here also works.
    context.context().update_server_def(context.get_server_def())

    self.assertTrue(context.check_alive("/job:worker/replica:0/task:0"))

  def testStop(self):
    self._cluster.stop()
    self.assertFalse(context.check_alive("/job:worker/replica:0/task:0"))
    self.assertFalse(context.check_alive("/job:worker/replica:0/task:1"))
    self.assertFalse(context.check_alive("/job:ps/replica:0/task:0"))
    self.assertFalse(context.check_alive("/job:chief/replica:0/task:0"))

  def testClusterResolverProperty(self):
    cluster_spec = self._cluster.cluster_resolver.cluster_spec().as_dict()

    self.assertEqual(len(cluster_spec["worker"]), 2)
    self.assertEqual(len(cluster_spec["ps"]), 1)
    self.assertEqual(len(cluster_spec["chief"]), 1)


if __name__ == "__main__":
  multi_process_runner.test_main()
