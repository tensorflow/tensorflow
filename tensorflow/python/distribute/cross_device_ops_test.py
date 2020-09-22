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
"""Tests for CrossDeviceOps."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import atexit
import collections
import os

from absl.testing import parameterized

from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import tensorflow_server_pb2
from tensorflow.python.distribute import cluster_resolver as cluster_resolver_lib
from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import cross_device_ops as cross_device_ops_lib
from tensorflow.python.distribute import multi_process_runner
from tensorflow.python.distribute import multi_worker_test_base
from tensorflow.python.distribute import reduce_util
from tensorflow.python.distribute import values as value_lib
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.util import nest

CollectiveCommunication = cross_device_ops_lib.CollectiveCommunication
ReduceOp = reduce_util.ReduceOp


def make_per_replica_value(value_fn, devices):
  """Creates a `PerReplica` object whose values reside in `devices`.

  Args:
    value_fn: a callable that takes one argument (`device_idx`) and should
      return the value that is going to be created on devices[device_idx].
    devices: a list of device strings to create `PerReplica` values on.

  Returns:
    A `PerReplica` object.
  """
  values = []
  for device_idx, device in enumerate(devices):
    v = value_fn(device_idx)
    if isinstance(v, indexed_slices.IndexedSlicesValue):
      with ops.device(device):
        values.append(
            indexed_slices.IndexedSlices(
                values=array_ops.identity(v.values),
                indices=array_ops.identity(v.indices),
                dense_shape=array_ops.identity(v.dense_shape)))
    else:
      with ops.device(device):
        values.append(array_ops.identity(v))
  return value_lib.PerReplica(values)


def enable_collective_ops():
  """Enable collectives in the current process."""
  cluster_resolver = cluster_resolver_lib.TFConfigClusterResolver()
  context.context().configure_collective_ops(
      collective_leader="'/job:worker/replica:0/task:0'")
  config_proto = config_pb2.ConfigProto()
  config_proto.experimental.collective_group_leader = (
      "/job:worker/replica:0/task:0")
  server_def = tensorflow_server_pb2.ServerDef(
      cluster=cluster_resolver.cluster_spec().as_cluster_def(),
      default_session_config=config_proto,
      job_name=cluster_resolver.task_type,
      task_index=cluster_resolver.task_id,
      protocol=cluster_resolver.rpc_layer)
  context.context().enable_collective_ops(server_def)


class MultiProcessPoolRunner():

  def __init__(self, num_processes):
    cluster_spec_dict = multi_worker_test_base.create_cluster_spec(
        num_workers=num_processes)
    self.runner = multi_process_runner.MultiProcessPoolRunner(cluster_spec_dict)


# Global MultiProcessPoolRunners that can be shared by test cases to avoid
# expensive initialization cost of TensorFlow in new processes.
#
# Note that they have to be globals and can't be owned by test classes because
# usually proc_func usually captures the test class instance, and test class
# instance can't be pickled if it has mpr as a member (it is not allowed to
# pickle Process objects).
# TODO(crccw): Use `num_workers` combination once it is ready.
global_mpr_2p = MultiProcessPoolRunner(num_processes=2)
global_mpr_1p = MultiProcessPoolRunner(num_processes=1)


def get_global_mpr(num_processes):
  if num_processes == 1:
    return global_mpr_1p.runner
  elif num_processes == 2:
    return global_mpr_2p.runner
  else:
    raise ValueError("get_global_mpr: num_processes must be 1 or 2, got %d" %
                     num_processes)


# Shutdown the runners gracefully to avoid the processes getting SIGTERM and
# make tsan happy.
def _shutdown_at_exit():
  global_mpr_2p.runner.shutdown()
  global_mpr_1p.runner.shutdown()


atexit.register(_shutdown_at_exit)


class CollectiveOpsTest(test.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    # Enabling collectives can be done in "setUpClass", but requires using
    # different collective_keys in different tests as collectives are reused
    # across tests. Always resetting collective ops before each test offers
    # better test isolation.
    global_mpr_1p.runner.run(enable_collective_ops)
    global_mpr_2p.runner.run(enable_collective_ops)

  def make_collective(self, num_processes, gpu_per_process, communication):
    """Returns collectives and other info to be used in tests.

    Args:
      num_processes: an integer indicating the number of processes that
        participate in the collective.
      gpu_per_process: number of GPUs (0 if no GPUs) used by each process.
      communication: one of `CollectiveCommunication`.

    Returns:
     A tuple of (collective, devices, group_size) where collective is a instance
     of `CollectiveAllReduce`, devices are a list of local devices (str)
     attached to the current process, and group_size is the group_size of
     collective.
    """

    cluster_resolver = cluster_resolver_lib.TFConfigClusterResolver()
    devices = [
        "/job:worker/replica:0/task:%d/device:CPU:0" % cluster_resolver.task_id
    ]
    if gpu_per_process > 0:
      devices = [
          "/job:worker/replica:0/task:%d/device:GPU:%d" %
          (cluster_resolver.task_id, i) for i in range(gpu_per_process)
      ]
    group_size = num_processes * len(devices)
    collective = cross_device_ops_lib.CollectiveAllReduce(
        devices=devices, group_size=group_size, communication=communication)
    return collective, devices, cluster_resolver.task_id

  def as_list(self, value):
    """An utility to convert a `Mirrored`, `Tensor` or `IndexedSlices` to a list.

    The reason it exists is to provide a uniformed view of returned value of
    "reduce" calls, especially across tf.function boundaries. Returning
    `Mirrored` from a tf.function will only evaluate the primary value, which
    makes collective ops of non-primary device being pruned, and will eventually
    cause hanging.

    Args:
      value: the value to convert, can be one of `Mirrored`, `Tensor` and
        `IndexedSlices`.

    Returns:
      A list of `Tensor` or `IndexedSlices`.
    """
    if isinstance(value, ops.Tensor):
      return [value]
    elif isinstance(value, indexed_slices.IndexedSlices):
      return [value]
    elif isinstance(value, value_lib.Mirrored):
      return value.values
    else:
      raise ValueError("unwrap: unsupported input type: %s" % type(value))

  RunOptions = collections.namedtuple(  # pylint: disable=invalid-name
      "RunOptions",
      [
          "mode",  # A list of str from ["eager", "func_graph"]
          "num_processes",
          "gpus_per_process",
          "reduce_op",
          "communication",
      ])
  RunOptions.__new__.__defaults__ = (["eager", "func_graph"], 2, 0,
                                     ReduceOp.SUM, CollectiveCommunication.AUTO)

  def reduce_and_verify(self, inputs, expect, options):
    """Reduce the given `inputs` and verify the output matches `expect`.

    Args:
      inputs: a list of `Tensor` or `IndexedSlices`, where i-th value will be
        fed to i-th replica.
      expect: a `Tensor` or `IndexedSlices`. This should be the expected value
        for one replica.
      options: a `RunOpotions` instance.
    """

    def replica_fn():
      collective, devices, pid = self.make_collective(options.num_processes,
                                                      options.gpus_per_process,
                                                      options.communication)

      def reduce_fn():
        value_fn = lambda device_idx: inputs[pid * len(devices) + device_idx]
        per_replica_value = make_per_replica_value(value_fn, devices)
        reduced_values = collective.reduce(options.reduce_op, per_replica_value,
                                           per_replica_value)
        reduced_values = self.as_list(reduced_values)
        self.assertAllEqual(devices, [v.device for v in reduced_values])
        return [ops.convert_to_tensor(v) for v in reduced_values]

      per_replica_expect = [ops.convert_to_tensor(expect)] * len(devices)

      if "eager" in options.mode:
        got = reduce_fn()
        self.assertAllClose(got, per_replica_expect)

      if "func_graph" in options.mode:
        got = def_function.function(reduce_fn)()
        self.assertAllClose(got, per_replica_expect)

    get_global_mpr(options.num_processes).run(replica_fn)

  def batch_reduce_and_verify(self, inputs, expect, options):
    """Batch reduce the given `inputs` and verify the output matches `expect`.

    Args:
      inputs: a 2-level nested list of `Tensor` or `IndexedSlices`, where i-th
        value will be fed to i-th replica.
      expect: a list of `Tensor` or `IndexedSlices`. This should be the expected
        value for one replica.
      options: a `RunOpotions` instance.
    """

    def replica_fn():
      collective, devices, pid = self.make_collective(options.num_processes,
                                                      options.gpus_per_process,
                                                      options.communication)

      def batch_reduce_fn():
        batch_size = len(inputs[0])
        value_dst_pairs = []
        for i in range(batch_size):

          def value_fn(device_idx, idx=i):
            return inputs[pid * len(devices) + device_idx][idx]

          per_replica_value = make_per_replica_value(value_fn, devices)
          value_dst_pairs.append((per_replica_value, per_replica_value))
        reduced_values = collective.batch_reduce(options.reduce_op,
                                                 value_dst_pairs)
        reduced_values = [self.as_list(v) for v in reduced_values]
        for v in reduced_values:
          self.assertAllEqual(devices, [t.device for t in v])
        return nest.map_structure(ops.convert_to_tensor, reduced_values)

      per_replica_expect = nest.map_structure(
          lambda x: [ops.convert_to_tensor(x)] * len(devices), expect)

      if "eager" in options.mode:
        got = batch_reduce_fn()
        self.assertAllClose(got, per_replica_expect)

      if "func_graph" in options.mode:
        got = def_function.function(batch_reduce_fn)()
        self.assertAllClose(got, per_replica_expect)

    get_global_mpr(options.num_processes).run(replica_fn)

  @combinations.generate(
      combinations.combine(
          num_processes=[1, 2],
          required_gpus=[0, 1, 2],
          communication=[
              # NCCL is only used for batch reduce, so we are not including
              # NCCL combination here.
              CollectiveCommunication.AUTO,
              CollectiveCommunication.RING
          ],
          reduce_op=[ReduceOp.SUM, ReduceOp.MEAN]))
  def testAllReduceDense(self, num_processes, required_gpus, communication,
                         reduce_op):
    options = self.RunOptions(
        num_processes=num_processes,
        gpus_per_process=required_gpus,
        reduce_op=reduce_op,
        communication=communication)
    group_size = options.num_processes * (options.gpus_per_process or 1)

    inputs_data = [1.0, 2.0, 3.0, 4.0]
    inputs = inputs_data[0:group_size]

    if group_size == 1:
      expect = 1.0
    if group_size == 2:
      expect = 3.0 if reduce_op == ReduceOp.SUM else 1.5
    elif group_size == 4:
      expect = 10.0 if reduce_op == ReduceOp.SUM else 2.5

    self.reduce_and_verify(inputs, expect, options)

  @combinations.generate(
      combinations.combine(
          num_processes=[1, 2],
          required_gpus=[0, 1, 2],
          communication=[
              # NCCL is only used for batch reduce, so we are not including
              # NCCL combination here.
              CollectiveCommunication.AUTO,
              CollectiveCommunication.RING
          ],
          # TODO(b/166682130): add MEAN reduce once the bug is fixed.
          reduce_op=ReduceOp.SUM))
  def testAllReduceSparse(self, num_processes, required_gpus, communication,
                          reduce_op):
    options = self.RunOptions(
        mode=["func_graph"],  # Sparse reduce is not supported in eager.
        num_processes=num_processes,
        gpus_per_process=required_gpus,
        reduce_op=reduce_op,
        communication=communication)
    group_size = options.num_processes * (options.gpus_per_process or 1)

    inputs_data = [
        indexed_slices.IndexedSlicesValue(
            values=[[1.], [2.]], indices=[0, 1], dense_shape=[10, 1]),
        indexed_slices.IndexedSlicesValue(
            values=[[3.], [4.]], indices=[1, 2], dense_shape=[10, 1]),
        indexed_slices.IndexedSlicesValue(
            values=[[5.], [6.]], indices=[7, 8], dense_shape=[10, 1]),
        indexed_slices.IndexedSlicesValue(
            values=[[7.], [8.]], indices=[3, 2], dense_shape=[10, 1]),
    ]
    inputs = inputs_data[0:group_size]

    if group_size == 1:
      expect = indexed_slices.IndexedSlices(
          values=[[1.], [2.]], indices=[0, 1], dense_shape=[10, 1])
    elif group_size == 2:
      expect = indexed_slices.IndexedSlices(
          values=[[1.], [2.], [3.], [4.]],
          indices=[0, 1, 1, 2],
          dense_shape=[10, 1])
    elif group_size == 4:
      expect = indexed_slices.IndexedSlices(
          values=[[1.], [2.], [3.], [4.], [5.], [6.], [7.], [8.]],
          indices=[0, 1, 1, 2, 7, 8, 3, 2],
          dense_shape=[10, 1])

    self.reduce_and_verify(inputs, expect, options)

  def testAllReduceSparseVariableLength(self):
    # One device per process, 2 processes, 2 replicas in total.
    inputs = [
        indexed_slices.IndexedSlicesValue(
            values=[[1.]], indices=[0], dense_shape=[10, 1]),
        indexed_slices.IndexedSlicesValue(
            values=[[2.], [3.], [4.]], indices=[0, 1, 2], dense_shape=[10, 1]),
    ]
    expect = indexed_slices.IndexedSlices(
        values=[[1.], [2.], [3.], [4.]],
        indices=[0, 0, 1, 2],
        dense_shape=[10, 1])
    self.reduce_and_verify(
        inputs,
        expect,
        self.RunOptions(
            mode=["func_graph"],  # Sparse reduce is not supported in eager.
            num_processes=2,
            reduce_op=ReduceOp.SUM))

  @combinations.generate(
      combinations.combine(
          num_processes=[1, 2],
          required_gpus=[0, 1, 2],
          communication=[
              CollectiveCommunication.AUTO, CollectiveCommunication.RING,
              CollectiveCommunication.NCCL
          ],
          reduce_op=[ReduceOp.SUM, ReduceOp.MEAN]))
  def testBatchAllReduceDense(self, num_processes, required_gpus, communication,
                              reduce_op):
    if required_gpus == 0 and communication == CollectiveCommunication.NCCL:
      self.skipTest("Skip CPU + NCCL combination")
    if num_processes == 2 and communication == CollectiveCommunication.NCCL:
      self.skipTest("Skip NCCL + 2 processes combination. NCCL requires "
                    "physical GPUs for every process.")

    options = self.RunOptions(
        num_processes=num_processes,
        gpus_per_process=required_gpus,
        reduce_op=reduce_op,
        communication=communication)
    group_size = options.num_processes * (options.gpus_per_process or 1)

    inputs_data = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
    inputs = inputs_data[0:group_size]

    if group_size == 1:
      expect = [1.0, 2.0]
    if group_size == 2:
      expect = [4.0, 6.0] if reduce_op == ReduceOp.SUM else [2.0, 3.0]
    elif group_size == 4:
      expect = [16.0, 20.0] if reduce_op == ReduceOp.SUM else [4.0, 5.0]

    self.batch_reduce_and_verify(inputs, expect, options)

  @combinations.generate(
      combinations.combine(
          num_processes=[1, 2],
          required_gpus=[0, 1, 2],
          communication=[
              CollectiveCommunication.AUTO,
              CollectiveCommunication.RING,
              CollectiveCommunication.NCCL,
          ],
          # TODO(b/166682130): add MEAN reduce once the bug is fixed.
          reduce_op=ReduceOp.SUM))
  def testBatchAllReduceSparse(self, num_processes, required_gpus,
                               communication, reduce_op):
    if required_gpus == 0 and communication == CollectiveCommunication.NCCL:
      self.skipTest("Skip CPU + NCCL combination")
    if num_processes == 2 and communication == CollectiveCommunication.NCCL:
      self.skipTest("Skip NCCL + 2 processes combination. NCCL requires "
                    "physical GPUs for every process.")

    options = self.RunOptions(
        mode=["func_graph"],  # Sparse reduce is not supported in eager.
        num_processes=num_processes,
        gpus_per_process=required_gpus,
        reduce_op=reduce_op,
        communication=communication)
    group_size = options.num_processes * (options.gpus_per_process or 1)

    inputs_data = ([
        indexed_slices.IndexedSlicesValue(
            values=[[1.], [2.]], indices=[0, 1], dense_shape=[10, 1]),
        indexed_slices.IndexedSlicesValue(
            values=[[3.], [4.]], indices=[1, 2], dense_shape=[5, 1])
    ], [
        indexed_slices.IndexedSlicesValue(
            values=[[5.], [6.]], indices=[1, 2], dense_shape=[10, 1]),
        indexed_slices.IndexedSlicesValue(
            values=[[7.], [8.]], indices=[0, 1], dense_shape=[5, 1])
    ], [
        indexed_slices.IndexedSlicesValue(
            values=[[9.], [10.]], indices=[3, 4], dense_shape=[10, 1]),
        indexed_slices.IndexedSlicesValue(
            values=[[11.], [12.]], indices=[3, 4], dense_shape=[5, 1])
    ], [
        indexed_slices.IndexedSlicesValue(
            values=[[13.], [14.]], indices=[8, 9], dense_shape=[10, 1]),
        indexed_slices.IndexedSlicesValue(
            values=[[15.], [16.]], indices=[3, 4], dense_shape=[5, 1])
    ])
    inputs = inputs_data[0:group_size]

    if group_size == 1:
      expect = [
          indexed_slices.IndexedSlices(
              values=[[1.], [2.]], indices=[0, 1], dense_shape=[10, 1]),
          indexed_slices.IndexedSlicesValue(
              values=[[3.], [4.]], indices=[1, 2], dense_shape=[5, 1])
      ]
    if group_size == 2:
      expect = [
          indexed_slices.IndexedSlices(
              values=[[1.], [2.], [5.], [6.]],
              indices=[0, 1, 1, 2],
              dense_shape=[10, 1]),
          indexed_slices.IndexedSlices(
              values=[[3.], [4.], [7.], [8.]],
              indices=[1, 2, 3, 4],
              dense_shape=[5, 1])
      ]
    elif group_size == 4:
      expect = [
          indexed_slices.IndexedSlices(
              values=[[1.], [2.], [5.], [6.], [9.], [10.], [13.], [14.]],
              indices=[0, 1, 1, 2, 3, 4, 8, 9],
              dense_shape=[10, 1]),
          indexed_slices.IndexedSlices(
              values=[[3.], [4.], [7.], [8.], [11.], [12.], [15.], [16.]],
              indices=[1, 2, 0, 1, 3, 4, 3, 4],
              dense_shape=[5, 2])
      ]
      self.batch_reduce_and_verify(inputs, expect, options)

  @combinations.generate(
      combinations.combine(
          num_processes=[1, 2],
          required_gpus=[0, 1, 2],
          axis=[0, 1, 2],
          func_mode=["eager", "func_graph"],
          communication=[
              CollectiveCommunication.NCCL,
              CollectiveCommunication.AUTO,
              CollectiveCommunication.RING
          ]))
  def testAllGatherSameShape(self, num_processes, required_gpus, communication,
                             func_mode, axis):

    def replica_fn():
      collective, devices, _ = self.make_collective(num_processes,
                                                    required_gpus,
                                                    communication)
      value = constant_op.constant([[[1, 2], [1, 2]]], dtype=dtypes.float32)

      def gather_fn():
        value_fn = lambda device_idx: value
        per_replica_value = make_per_replica_value(value_fn, devices)
        gathered_values = collective._gather(
            per_replica_value, per_replica_value, axis=axis)
        gathered_values = self.as_list(gathered_values)
        # Skip checking devices in eager. In eager the device attribute doesn't
        # reflect the actual device of the tensor.
        if not context.executing_eagerly():
          self.assertAllEqual(devices, [v.device for v in gathered_values])
        return [ops.convert_to_tensor(v) for v in gathered_values]

      group_size = num_processes * (required_gpus or 1)
      expect = array_ops.concat([value] * group_size, axis=axis)
      per_replica_expect = [ops.convert_to_tensor(expect)] * len(devices)

      if func_mode == "eager":
        result = gather_fn()
        self.assertAllClose(result, per_replica_expect)

      if func_mode == "func_graph":
        result = def_function.function(gather_fn)()
        self.assertAllClose(result, per_replica_expect)

    get_global_mpr(num_processes).run(replica_fn)

if __name__ == "__main__":
  # Set default inter op thread pool size to one to ensure we don't exhaust the
  # thread pool with the additional executors to run collectives in eager.
  os.environ["TF_NUM_INTEROP_THREADS"] = "1"
  multi_process_runner.test_main()
