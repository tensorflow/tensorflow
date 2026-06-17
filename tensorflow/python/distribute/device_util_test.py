# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for device utilities."""

from absl.testing import parameterized

from tensorflow.core.protobuf import tensorflow_server_pb2
from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import multi_worker_test_base
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.platform import test
from tensorflow.python.training import server_lib


class DeviceUtilTest(test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(DeviceUtilTest, self).setUp()
    context._reset_context()  # pylint: disable=protected-access

  @combinations.generate(
      combinations.combine(mode="graph")
  )
  def testCurrentDeviceWithGlobalGraph(self):
    with ops.device("/cpu:0"):
      self.assertEqual(device_util.current(), "/device:CPU:0")

    with ops.device("/job:worker"):
      with ops.device("/cpu:0"):
        self.assertEqual(device_util.current(), "/job:worker/device:CPU:0")

    with ops.device("/cpu:0"):
      with ops.device("/gpu:0"):
        self.assertEqual(device_util.current(), "/device:GPU:0")

  def testCurrentDeviceWithNonGlobalGraph(self):
    with ops.Graph().as_default():
      with ops.device("/cpu:0"):
        self.assertEqual(device_util.current(), "/device:CPU:0")

  def testCurrentDeviceWithEager(self):
    with context.eager_mode():
      with ops.device("/cpu:0"):
        self.assertEqual(device_util.current(),
                         "/job:localhost/replica:0/task:0/device:CPU:0")

  @combinations.generate(combinations.combine(mode=["graph", "eager"]))
  def testCanonicalizeWithoutDefaultDevice(self, mode):
    if mode == "graph":
      self.assertEqual(
          device_util.canonicalize("/cpu:0"),
          "/replica:0/task:0/device:CPU:0")
    else:
      self.assertEqual(
          device_util.canonicalize("/cpu:0"),
          "/job:localhost/replica:0/task:0/device:CPU:0")
    self.assertEqual(
        device_util.canonicalize("/job:worker/cpu:0"),
        "/job:worker/replica:0/task:0/device:CPU:0")
    self.assertEqual(
        device_util.canonicalize("/job:worker/task:1/cpu:0"),
        "/job:worker/replica:0/task:1/device:CPU:0")

  @combinations.generate(combinations.combine(mode=["eager"]))
  def testCanonicalizeWithoutDefaultDeviceCollectiveEnabled(self):
    cluster_spec = server_lib.ClusterSpec(
        multi_worker_test_base.create_cluster_spec(
            has_chief=False, num_workers=1, num_ps=0, has_eval=False))
    server_def = tensorflow_server_pb2.ServerDef(
        cluster=cluster_spec.as_cluster_def(),
        job_name="worker",
        task_index=0,
        protocol="grpc",
        port=0)
    context.context().enable_collective_ops(server_def)
    self.assertEqual(
        device_util.canonicalize("/cpu:0"),
        "/job:worker/replica:0/task:0/device:CPU:0")

  def testCanonicalizeWithDefaultDevice(self):
    self.assertEqual(
        device_util.canonicalize("/job:worker/task:1/cpu:0", default="/gpu:0"),
        "/job:worker/replica:0/task:1/device:CPU:0")
    self.assertEqual(
        device_util.canonicalize("/job:worker/task:1", default="/gpu:0"),
        "/job:worker/replica:0/task:1/device:GPU:0")
    self.assertEqual(
        device_util.canonicalize("/cpu:0", default="/job:worker"),
        "/job:worker/replica:0/task:0/device:CPU:0")
    self.assertEqual(
        device_util.canonicalize(
            "/job:worker/replica:0/task:1/device:CPU:0",
            default="/job:chief/replica:0/task:1/device:CPU:0"),
        "/job:worker/replica:0/task:1/device:CPU:0")

  def testResolveWithDeviceScope(self):
    with ops.device("/gpu:0"):
      self.assertEqual(
          device_util.resolve("/job:worker/task:1/cpu:0"),
          "/job:worker/replica:0/task:1/device:CPU:0")
      self.assertEqual(
          device_util.resolve("/job:worker/task:1"),
          "/job:worker/replica:0/task:1/device:GPU:0")
    with ops.device("/job:worker"):
      self.assertEqual(
          device_util.resolve("/cpu:0"),
          "/job:worker/replica:0/task:0/device:CPU:0")


if __name__ == "__main__":
  test.main()
