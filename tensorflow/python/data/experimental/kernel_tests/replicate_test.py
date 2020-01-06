# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Eager mode tests for the experimental `replicate` transformation."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

from tensorflow.core.protobuf import cluster_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import tensorflow_server_pb2
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.data.experimental.ops import distribute
from tensorflow.python.data.experimental.ops import distribute_options
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import context
from tensorflow.python.framework import combinations
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import test
from tensorflow.python.training import server_lib


class LocalReplicateTest(test_base.DatasetTestBase, parameterized.TestCase):

  def __init__(self, methodName="runTest"):  # pylint: disable=invalid-name
    super(LocalReplicateTest, self).__init__(methodName)
    self._device0 = "/device:CPU:0"
    self._device1 = "/device:CPU:1"
    self._device2 = "/device:CPU:2"

  @combinations.generate(
      combinations.combine(tf_api_version=[1], mode=["graph", "eager"]))
  def testBasic(self):
    with ops.device(self._device0):
      dataset0 = dataset_ops.Dataset.range(100)
    replicated_ds = distribute.replicate(dataset0,
                                         [self._device1, self._device2])
    dataset1 = replicated_ds[self._device1]
    dataset2 = replicated_ds[self._device2]

    with ops.device(self._device0):
      self.assertDatasetProduces(dataset0, range(100))
    with ops.device(self._device1):
      self.assertDatasetProduces(dataset1, range(100))
    with ops.device(self._device2):
      self.assertDatasetProduces(dataset2, range(100))

  @combinations.generate(
      combinations.combine(tf_api_version=[1], mode=["graph", "eager"]))
  def testVariableInput(self):
    with ops.device(self._device0):
      counter_var = variable_scope.get_variable(
          "counter", (), dtypes.int32, use_resource=True)
      dataset0 = dataset_ops.Dataset.range(100).map(
          lambda _: counter_var.assign_add(1))
    replicated_ds = distribute.replicate(dataset0,
                                         [self._device1, self._device2])
    dataset1 = replicated_ds[self._device1]
    dataset2 = replicated_ds[self._device2]
    self.evaluate(counter_var.initializer)
    with ops.device(self._device0):
      self.assertDatasetProduces(
          dataset0, range(1, 101), requires_initialization=True)
    with ops.device(self._device1):
      self.assertDatasetProduces(
          dataset1, range(101, 201), requires_initialization=True)
    with ops.device(self._device2):
      self.assertDatasetProduces(
          dataset2, range(201, 301), requires_initialization=True)

  @combinations.generate(
      combinations.combine(tf_api_version=[1], mode=["graph", "eager"]))
  def testExternalStatePolicyIgnore(self):
    with ops.device(self._device0):
      dataset0 = dataset_ops.Dataset.range(100).map(
          lambda _: random_ops.random_uniform(  # pylint:disable=g-long-lambda
              [],
              minval=1,
              maxval=10,
              dtype=dtypes.float32))
      opt = dataset_ops.Options()
      opt.experimental_external_state_policy = (
          distribute_options.ExternalStatePolicy.IGNORE)
      dataset0 = dataset0.with_options(opt)
    replicated_ds = distribute.replicate(dataset0,
                                         [self._device1, self._device2])
    dataset1 = replicated_ds[self._device1]
    dataset2 = replicated_ds[self._device2]

    with ops.device(self._device0):
      get_next0 = self.getNext(dataset0)
    with ops.device(self._device1):
      get_next1 = self.getNext(dataset1)
    with ops.device(self._device2):
      get_next2 = self.getNext(dataset2)

    for _ in range(100):
      self.evaluate(get_next0())
      self.evaluate(get_next1())
      self.evaluate(get_next2())

  @combinations.generate(
      combinations.combine(tf_api_version=[1], mode=["graph", "eager"]))
  def testExternalStatePolicyWarn(self):
    with ops.device(self._device0):
      dataset0 = dataset_ops.Dataset.range(100).map(
          lambda _: random_ops.random_uniform(  # pylint:disable=g-long-lambda
              [],
              minval=1,
              maxval=10,
              dtype=dtypes.float32))
      opt = dataset_ops.Options()
      opt.experimental_external_state_policy = (
          distribute_options.ExternalStatePolicy.WARN)
      dataset0 = dataset0.with_options(opt)
    replicated_ds = distribute.replicate(dataset0,
                                         [self._device1, self._device2])
    dataset1 = replicated_ds[self._device1]
    dataset2 = replicated_ds[self._device2]

    with ops.device(self._device0):
      get_next0 = self.getNext(dataset0)
    with ops.device(self._device1):
      get_next1 = self.getNext(dataset1)
    with ops.device(self._device2):
      get_next2 = self.getNext(dataset2)

    for _ in range(100):
      self.evaluate(get_next0())
      self.evaluate(get_next1())
      self.evaluate(get_next2())

  @combinations.generate(
      combinations.combine(tf_api_version=[1], mode=["graph", "eager"]))
  def testExternalStatePolicyFail(self):
    with ops.device(self._device0):
      dataset0 = dataset_ops.Dataset.range(100).map(
          lambda _: random_ops.random_uniform(  # pylint:disable=g-long-lambda
              [],
              minval=1,
              maxval=10,
              dtype=dtypes.float32))
      opt = dataset_ops.Options()
      opt.experimental_external_state_policy = (
          distribute_options.ExternalStatePolicy.FAIL)
      dataset0 = dataset0.with_options(opt)
    with self.assertRaises(errors.FailedPreconditionError):
      replicated_ds = distribute.replicate(dataset0,
                                           [self._device1, self._device2])
      dataset1 = replicated_ds[self._device1]
      dataset2 = replicated_ds[self._device2]

      with ops.device(self._device0):
        get_next0 = self.getNext(dataset0)
      with ops.device(self._device1):
        get_next1 = self.getNext(dataset1)
      with ops.device(self._device2):
        get_next2 = self.getNext(dataset2)

      for _ in range(100):
        self.evaluate(get_next0())
        self.evaluate(get_next1())
        self.evaluate(get_next2())


JOB_NAME = "remote_device"


def _get_server_def(job_name, local_server_port, remote_server_addresses,
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


# Pure eager mode test that sets up a cluster of processes.
class RemoteReplicateTest(test_base.DatasetTestBase, parameterized.TestCase):

  def __init__(self, methodName="runTest"):  # pylint: disable=invalid-name
    super(RemoteReplicateTest, self).__init__(methodName)
    self._cached_server1 = server_lib.Server.create_local_server()
    self._cached_server2 = server_lib.Server.create_local_server()
    self._cached_server1_target = self._cached_server1.target[len("grpc://"):]
    self._cached_server2_target = self._cached_server2.target[len("grpc://"):]
    self._device0 = "/job:%s/replica:0/task:0/device:CPU:0" % JOB_NAME
    self._device1 = "/job:%s/replica:0/task:1/device:CPU:0" % JOB_NAME
    self._device2 = "/job:%s/replica:0/task:2/device:CPU:0" % JOB_NAME

  def setUp(self):
    super(RemoteReplicateTest, self).setUp()
    # Start the local server.
    local_port = pywrap_tensorflow.TF_PickUnusedPortOrDie()
    context.set_server_def(
        server_def=_get_server_def(
            JOB_NAME,
            local_server_port=local_port,
            remote_server_addresses=[
                self._cached_server1_target, self._cached_server2_target
            ],
            task_index=0))

  @combinations.generate(
      combinations.combine(tf_api_version=[2], mode=["eager"]))
  def testBasic(self):
    with ops.device(self._device0):
      dataset0 = dataset_ops.Dataset.range(100)
    replicated_ds = distribute.replicate(dataset0,
                                         [self._device1, self._device2])
    dataset1 = replicated_ds[self._device1]
    dataset2 = replicated_ds[self._device2]
    with ops.device(self._device0):
      self.assertDatasetProduces(dataset0, range(100))
    with ops.device(self._device1):
      self.assertDatasetProduces(dataset1, range(100))
    with ops.device(self._device2):
      self.assertDatasetProduces(dataset2, range(100))

  @combinations.generate(
      combinations.combine(tf_api_version=[2], mode=["eager"]))
  def testMap(self):
    with ops.device(self._device0):
      dataset0 = dataset_ops.Dataset.range(100).map(lambda x: x * 2)
    replicated_ds = distribute.replicate(dataset0,
                                         [self._device1, self._device2])
    dataset1 = replicated_ds[self._device1]
    dataset2 = replicated_ds[self._device2]
    with ops.device(self._device0):
      self.assertDatasetProduces(dataset0, range(0, 200, 2))
    with ops.device(self._device1):
      self.assertDatasetProduces(dataset1, range(0, 200, 2))
    with ops.device(self._device2):
      self.assertDatasetProduces(dataset2, range(0, 200, 2))

  @combinations.generate(
      combinations.combine(tf_api_version=[2], mode=["eager"]))
  def testVariableInput(self):
    with ops.device(self._device0):
      counter_var = variable_scope.get_variable(
          "counter", (), dtypes.int32, use_resource=True)
      dataset0 = dataset_ops.Dataset.range(100).map(
          lambda _: counter_var.assign_add(1))
    # We don't support stateful ops across processes in functions as of now.
    with self.assertRaises(errors.InvalidArgumentError):
      replicated_ds = distribute.replicate(dataset0,
                                           [self._device1, self._device2])
      dataset1 = replicated_ds[self._device1]
      with ops.device(self._device0):
        get_next0 = self.getNext(dataset0)
      with ops.device(self._device1):
        get_next1 = self.getNext(dataset1)
      for _ in range(100):
        self.evaluate(get_next0())
        self.evaluate(get_next1())


if __name__ == "__main__":
  ops.enable_eager_execution(
      config=config_pb2.ConfigProto(device_count={"CPU": 3}))
  test.main()
