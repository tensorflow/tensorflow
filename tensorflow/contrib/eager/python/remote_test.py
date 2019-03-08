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
"""Tests for remote eager execution."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os

import numpy as np

from tensorflow.contrib.eager.python import parameter_server
from tensorflow.core.protobuf import cluster_pb2
from tensorflow.core.protobuf import tensorflow_server_pb2
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import function
from tensorflow.python.eager import remote
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import server_lib

JOB_NAME = "remote_device"
ALT_JOB_NAME = "alt_remote_device"


def run_sync_and_async(f):
  """Execute all test methods in the given class in sync and async modes."""

  @functools.wraps(f)
  def decorator(self, *args, **kwargs):
    # TODO(b/117110239): Re-enable.
    # with context.execution_mode(context.ASYNC):
    #   f(self, *args, **kwargs)

    with context.execution_mode(context.SYNC):
      f(self, *args, **kwargs)

  return decorator


def get_server_def(job_name, local_server_port, remote_server_addresses,
                   task_index):
  """Returns a server def with a single job + multiple tasks."""
  cluster_def = cluster_pb2.ClusterDef()
  job_def = cluster_def.job.add()
  job_def.name = job_name
  job_def.tasks[0] = "localhost:%d" % local_server_port

  for i, remote_server_address in enumerate(remote_server_addresses, start=1):
    job_def.tasks[i] = remote_server_address

  server_def = tensorflow_server_pb2.ServerDef(
      cluster=cluster_def,
      job_name=job_name,
      task_index=task_index,
      protocol="grpc")

  return server_def


class RemoteExecutionTest(test.TestCase):

  def __init__(self, methodName="runTest"):  # pylint: disable=invalid-name
    super(RemoteExecutionTest, self).__init__(methodName)
    self._cached_server1 = server_lib.Server.create_local_server()
    self._cached_server2 = server_lib.Server.create_local_server()

    os.environ["TF_EAGER_REMOTE_USE_SEND_TENSOR_RPC"] = "1"

    self._cached_server1_target = self._cached_server1.target[len("grpc://"):]
    self._cached_server2_target = self._cached_server2.target[len("grpc://"):]

  def setUp(self):
    # Start the local server.
    context.set_server_def(
        server_def=get_server_def(
            JOB_NAME,
            local_server_port=0,
            remote_server_addresses=[
                self._cached_server1_target, self._cached_server2_target
            ],
            task_index=0))

  @run_sync_and_async
  def testDefunMatmul(self):
    """Basic remote eager execution with defun."""

    mm_defun = function.defun(math_ops.matmul)
    with ops.device("job:%s/replica:0/task:1/device:CPU:0" % JOB_NAME):
      x1 = array_ops.ones([2, 2])
    with ops.device("job:%s/replica:0/task:2/device:CPU:0" % JOB_NAME):
      x2 = array_ops.ones([2, 2])
      y = mm_defun(x1, x2)
    np.testing.assert_array_equal([[2, 2], [2, 2]], y.numpy())

  @run_sync_and_async
  def testSimpleMatmul(self):
    """Basic remote eager execution."""

    with ops.device("job:%s/replica:0/task:1/device:CPU:0" % JOB_NAME):
      x1 = array_ops.ones([2, 2])
    with ops.device("job:%s/replica:0/task:2/device:CPU:0" % JOB_NAME):
      x2 = array_ops.ones([2, 2])
      y = math_ops.matmul(x1, x2)
    np.testing.assert_array_equal([[2, 2], [2, 2]], y.numpy())

  def testParameterServer(self):
    with parameter_server.parameter_server_scope(
        is_chief=True, ps_job_name=JOB_NAME, num_ps_tasks=3):
      v0 = variables.Variable([1.0], name="v0")
      v1 = variables.Variable([2.0], name="v1")
    v0.assign(v0 * v1)
    self.assertAllEqual(v0.read_value(), [2.0])
    self.assertAllEqual(v0.device,
                        "/job:%s/replica:0/task:0/device:CPU:0" % JOB_NAME)
    self.assertAllEqual(v1.device,
                        "/job:%s/replica:0/task:1/device:CPU:0" % JOB_NAME)
    v1.assign_add(v1)
    # Simulate aliasing another variable of the same name as v1
    with ops.device("/job:%s/replica:0/task:1/device:CPU:0" % JOB_NAME):
      v1_replica = parameter_server.SharedVariable(
          [1.0], name="v1", initialize=False)
    self.assertAllEqual(v1_replica.read_value(), [4.0])

  @run_sync_and_async
  def testSimpleWeightRead(self):
    """Basic remote eager weight read."""

    with ops.device("job:%s/replica:0/task:1/device:CPU:0" % JOB_NAME):
      w = resource_variable_ops.ResourceVariable([[2.0]])
      loss = w * w
    np.testing.assert_array_equal([[4.0]], loss.numpy())

  @run_sync_and_async
  def testTapeWeightRead(self):
    """Remote eager weight read in a tape."""

    with ops.device("job:%s/replica:0/task:1/device:CPU:0" % JOB_NAME):
      w = resource_variable_ops.ResourceVariable([[3.0]])
      with backprop.GradientTape() as tape:
        loss = w * w

      grad = tape.gradient(loss, w)
    np.testing.assert_array_equal([[9.0]], loss.numpy())
    np.testing.assert_array_equal([[6.0]], grad.numpy())

  @run_sync_and_async
  def testServerDefChanged(self):
    """Update server def, and run ops on new cluster."""
    context.set_server_def(
        server_def=get_server_def(
            ALT_JOB_NAME,
            local_server_port=0,
            remote_server_addresses=[
                self._cached_server1_target, self._cached_server2_target
            ],
            task_index=0))

    with ops.device("job:%s/replica:0/task:1/device:CPU:0" % ALT_JOB_NAME):
      x1 = array_ops.ones([2, 2])
    y = math_ops.matmul(x1, x1)
    np.testing.assert_array_equal([[2, 2], [2, 2]], y.numpy())

    # Set the server def back to JOB_NAME
    context.set_server_def(
        server_def=get_server_def(
            JOB_NAME,
            local_server_port=0,
            remote_server_addresses=[
                self._cached_server1_target, self._cached_server2_target
            ],
            task_index=0))

    with ops.device("job:%s/replica:0/task:1/device:CPU:0" % JOB_NAME):
      x1 = array_ops.ones([2, 2])
    y = math_ops.matmul(x1, x1)
    np.testing.assert_array_equal([[2, 2], [2, 2]], y.numpy())

  @run_sync_and_async
  def testConnectToRemoteServer(self):
    """Basic server connection."""
    remote.connect_to_remote_host(self._cached_server1_target)

    with ops.device("job:worker/replica:0/task:1/device:CPU:0"):
      x1 = array_ops.ones([2, 2])
      x2 = array_ops.ones([2, 2])
      y = math_ops.matmul(x1, x2)
    np.testing.assert_array_equal([[2, 2], [2, 2]], y.numpy())

  @run_sync_and_async
  def testContextDeviceUpdated(self):
    """Tests that the context device is correctly updated."""

    with ops.device("cpu:0"):
      x1 = array_ops.ones([2, 2])
      x2 = array_ops.ones([2, 2])
      y = math_ops.matmul(x1, x2)
    np.testing.assert_array_equal([[2, 2], [2, 2]], y.numpy())

    # `y` is placed on the local CPU as expected.
    self.assertEqual(y.device,
                     "/job:%s/replica:0/task:0/device:CPU:0" % JOB_NAME)

  @run_sync_and_async
  def testGPUToRemoteCopy(self):
    """Tests that the remote copy happens satisfactorily."""
    if not context.context().num_gpus():
      self.skipTest("No GPUs.")

    x1 = array_ops.ones([2, 2]).gpu()

    with ops.device("/job:remote_device/replica:0/task:1/device:CPU:0"):
      x2 = x1._copy()  # pylint: disable=protected-access

    np.testing.assert_array_equal(x1.numpy(), x2.numpy())


if __name__ == "__main__":
  ops.enable_eager_execution()
  test.main()
