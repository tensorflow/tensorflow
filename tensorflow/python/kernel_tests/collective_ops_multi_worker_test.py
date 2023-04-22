# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for multi worker Collective Operations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import os
import threading
import time

from absl.testing import parameterized

from tensorflow.core.protobuf import tensorflow_server_pb2
from tensorflow.python.distribute import cluster_resolver as cluster_resolver_lib
from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import multi_process_runner
from tensorflow.python.distribute import multi_worker_test_base
from tensorflow.python.eager import context
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import collective_ops


def enable_collective_ops(cluster_resolver):
  context.context().configure_collective_ops(
      collective_leader="/job:worker/replica:0/task:0")
  config_proto = copy.deepcopy(context.context().config)
  server_def = tensorflow_server_pb2.ServerDef(
      cluster=cluster_resolver.cluster_spec().as_cluster_def(),
      default_session_config=config_proto,
      job_name=cluster_resolver.task_type,
      task_index=cluster_resolver.task_id,
      protocol=cluster_resolver.rpc_layer or "grpc")
  context.context().enable_collective_ops(server_def)


def enable_collective_ops_with_barrier(cluster_resolver):
  multi_process_runner.get_barrier().wait()
  enable_collective_ops(cluster_resolver)
  multi_process_runner.get_barrier().wait()


device_combination = (
    combinations.combine(device="CPU", communication="RING", required_gpus=0) +
    combinations.combine(
        device="GPU", communication=["RING", "NCCL"], required_gpus=1))


class CollectiveOpTest(test.TestCase):

  def testCheckHealth(self):

    def worker_fn():
      enable_collective_ops(cluster_resolver_lib.TFConfigClusterResolver())
      # There may be some delays before the server startup. Check health should
      # eventually be OK.
      while True:
        try:
          for task in [
              "/job:worker/replica:0/task:0",
              "/job:worker/replica:0/task:1",
          ]:
            context.context().check_collective_ops_peer_health(
                task, timeout_in_ms=1000)
        except (errors.UnavailableError, errors.DeadlineExceededError):
          continue
        break
      multi_process_runner.get_barrier().wait()

    cluster_spec = multi_worker_test_base.create_cluster_spec(num_workers=2)
    mpr = multi_process_runner.MultiProcessRunner(worker_fn, cluster_spec)
    mpr.start()
    mpr.join()

  def testCheckHealthPeerDown(self):

    def worker_fn():
      enable_collective_ops(cluster_resolver_lib.TFConfigClusterResolver())
      context.context().check_collective_ops_peer_health(
          "/job:worker/replica:0/task:1", timeout_in_ms=1000)

    cluster_spec = multi_worker_test_base.create_cluster_spec(num_workers=2)
    mpr = multi_process_runner.MultiProcessRunner(worker_fn, cluster_spec)
    mpr.start_single_process("worker", 0)
    with self.assertRaises(
        (errors.UnavailableError, errors.DeadlineExceededError)):
      mpr.join()

  def testCheckHealthPeerRestart(self):

    def worker_fn():
      cluster_resolver = cluster_resolver_lib.TFConfigClusterResolver()
      enable_collective_ops(cluster_resolver)

      collective_ops.all_reduce(
          constant_op.constant(1.),
          group_size=2,
          group_key=100,
          instance_key=100,
          merge_op="Add",
          final_op="Id",
          communication_hint="ring")

      if cluster_resolver.task_type == "worker":
        # MultiProcessRunner will auto restart worker-0.
        os._exit(1)  # pylint: disable=protected-access
      else:
        # chief should eventually gets FailedPreconditionError after worker-0
        # has restarted.
        while True:
          time.sleep(1)
          try:
            context.context().check_collective_ops_peer_health(
                "/job:worker/replica:0/task:0", timeout_in_ms=1000)
          except errors.UnavailableError:
            pass
          except errors.FailedPreconditionError:
            break

    cluster_spec = multi_worker_test_base.create_cluster_spec(
        has_chief=True, num_workers=1)
    mpr = multi_process_runner.MultiProcessRunner(
        worker_fn, cluster_spec, auto_restart=True)
    mpr.start()
    mpr.join()

  def testCheckHealthInvalidPeer(self):

    def worker_fn():
      enable_collective_ops(cluster_resolver_lib.TFConfigClusterResolver())
      context.context().check_collective_ops_peer_health(
          "localhost:12345", timeout_in_ms=1000)

    cluster_spec = multi_worker_test_base.create_cluster_spec(num_workers=2)
    mpr = multi_process_runner.MultiProcessRunner(worker_fn, cluster_spec)
    mpr.start_single_process("worker", 0)
    with self.assertRaises(errors.InvalidArgumentError):
      mpr.join()


two_worker_pool_runner = multi_process_runner.MultiProcessPoolRunner(
    multi_worker_test_base.create_cluster_spec(num_workers=2),
    initializer=lambda: enable_collective_ops(cluster_resolver_lib.
                                              TFConfigClusterResolver()))


@combinations.generate(
    combinations.times(
        combinations.combine(
            mode="eager", num_workers=2, runner=two_worker_pool_runner),
        device_combination))
class AbortCollectiveOpsTest(test.TestCase, parameterized.TestCase):

  def testAbortCommunication(self, device, communication):
    if communication == "NCCL":
      self.skipTest("b/171358086: cannot test multi worker NCCL")
    dev0 = "/device:%s:0" % device
    cluster_resolver = cluster_resolver_lib.TFConfigClusterResolver()
    enable_collective_ops_with_barrier(cluster_resolver)
    group_size = 2
    group_key = 100
    instance_key = 100
    in_tensor = constant_op.constant([1.])

    # First perform a normal all-reduce to complete the group and instance
    # resolution.
    with ops.device(dev0):
      collective_ops.all_reduce(
          in_tensor,
          group_size,
          group_key,
          instance_key,
          communication_hint=communication)

    if cluster_resolver.task_id == 1:

      def abort_fn():
        time.sleep(2)
        context.context().abort_collective_ops(errors.UNAVAILABLE, "peer down")

      t = threading.Thread(target=abort_fn)
      t.start()

      with self.assertRaisesRegex(errors.UnavailableError, "peer down"):
        with ops.device(dev0):
          collective_ops.all_reduce(
              in_tensor,
              group_size,
              group_key,
              instance_key,
              communication_hint=communication)

      # After abortion, subsequent collectives should fail immediately.
      with self.assertRaisesRegex(errors.UnavailableError, "peer down"):
        with ops.device(dev0):
          collective_ops.all_reduce(
              in_tensor,
              group_size,
              group_key,
              instance_key,
              communication_hint=communication)

      t.join()

    # Enable collective ops again in order to reset the collective executor.
    enable_collective_ops_with_barrier(cluster_resolver)
    with ops.device(dev0):
      collective_ops.all_reduce(
          in_tensor,
          group_size,
          group_key,
          instance_key,
          communication_hint=communication)

  def testAbortGroupParamsResolution(self, device, communication):
    if communication == "NCCL":
      self.skipTest("b/171358086: cannot test multi worker NCCL")
    dev0 = "/device:%s:0" % device
    cluster_resolver = cluster_resolver_lib.TFConfigClusterResolver()
    enable_collective_ops_with_barrier(cluster_resolver)
    group_size = 2
    group_key = 100
    instance_key = 100
    in_tensor = constant_op.constant([1.])

    if cluster_resolver.task_id == 1:

      def abort_fn():
        time.sleep(2)
        context.context().abort_collective_ops(errors.UNAVAILABLE, "peer down")

      t = threading.Thread(target=abort_fn)
      t.start()

      with self.assertRaisesRegex(errors.UnavailableError, "peer down"):
        # This hangs on params resolution since we're only launching one
        # collective for a group size of 2.
        with ops.device(dev0):
          collective_ops.all_reduce(in_tensor, group_size, group_key,
                                    instance_key)

      # After abortion, subsequent collectives should fail immediately.
      with self.assertRaisesRegex(errors.UnavailableError, "peer down"):
        with ops.device(dev0):
          collective_ops.all_reduce(in_tensor, group_size, group_key,
                                    instance_key)

      t.join()

    # Enable collective ops again in order to reset the collective executor.
    enable_collective_ops_with_barrier(cluster_resolver)
    with ops.device(dev0):
      collective_ops.all_reduce(in_tensor, group_size, group_key, instance_key)

  def testAbortInstanceParamsResolution(self, device, communication):
    if communication == "NCCL":
      self.skipTest("b/171358086: cannot test multi worker NCCL")
    dev0 = "/device:%s:0" % device
    cluster_resolver = cluster_resolver_lib.TFConfigClusterResolver()
    enable_collective_ops_with_barrier(cluster_resolver)
    group_size = 2
    group_key = 100
    instance_key = 100
    in_tensor = constant_op.constant([1.])

    # First perform a normal all-reduce to complete the group resolution.
    with ops.device(dev0):
      collective_ops.all_reduce(in_tensor, group_size, group_key, instance_key)

    # We use broadcast to test aborting instance resolution since only broadcast
    # waits for the group.

    if cluster_resolver.task_id == 1:

      def abort_fn():
        time.sleep(2)
        context.context().abort_collective_ops(errors.UNAVAILABLE, "peer down")

      t = threading.Thread(target=abort_fn)
      t.start()

      # Use a different instance key to trigger another instance resolution.
      instance_key = 101
      with self.assertRaisesRegex(errors.UnavailableError, "peer down"):
        # This hangs on params resolution since we're only launching one
        # collective for a group size of 2.
        with ops.device(dev0):
          collective_ops.broadcast_send(in_tensor, (1,), dtypes.float32,
                                        group_size, group_key, instance_key)

      # After abortion, subsequent collectives should fail immediately.
      with self.assertRaisesRegex(errors.UnavailableError, "peer down"):
        with ops.device(dev0):
          collective_ops.broadcast_send(in_tensor, (1,), dtypes.float32,
                                        group_size, group_key, instance_key)

      t.join()

    # Enable collective ops again in order to reset the collective executor.
    enable_collective_ops_with_barrier(cluster_resolver)
    # Reassign instance_key so that it's the same on each worker.
    instance_key = 100
    with ops.device(dev0):
      if cluster_resolver.task_id == 0:
        collective_ops.broadcast_send(in_tensor, (1,), dtypes.float32,
                                      group_size, group_key, instance_key)
      else:
        collective_ops.broadcast_recv((1,), dtypes.float32, group_size,
                                      group_key, instance_key)


if __name__ == "__main__":
  multi_process_runner.test_main()
