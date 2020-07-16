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
# =============================================================================
"""Tests for the functional saver."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensorflow.python.eager import context
from tensorflow.python.eager import remote
from tensorflow.python.eager import test
from tensorflow.python.eager import wrap_function
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.platform import gfile
from tensorflow.python.training import server_lib
from tensorflow.python.training.saving import checkpoint_options
from tensorflow.python.training.saving import functional_saver
from tensorflow.python.training.saving import saveable_hook
from tensorflow.python.training.saving import saveable_object_util

LOCALHOST = "/job:localhost/replica:0/task:0/device:CPU:0"


class SaverTest(test.TestCase):

  def setUp(self):
    super(SaverTest, self).setUp()
    cpus = config.list_physical_devices("CPU")
    # Set 3 virtual CPUs
    config.set_logical_device_configuration(cpus[0], [
        context.LogicalDeviceConfiguration(),
        context.LogicalDeviceConfiguration(),
        context.LogicalDeviceConfiguration()
    ])
    self.local_options = checkpoint_options.CheckpointOptions(
        experimental_io_device=LOCALHOST)

  @test_util.run_in_graph_and_eager_modes
  def test_resource_variable(self):
    v1 = resource_variable_ops.ResourceVariable(2.)
    self.evaluate(v1.initializer)
    saver = functional_saver._SingleDeviceSaver(
        saveable_object_util.saveable_objects_for_op(v1, "x"))
    prefix = os.path.join(self.get_temp_dir(), "ckpt")
    self.evaluate(saver.save(constant_op.constant(prefix)))
    self.assertEqual(2, len(gfile.Glob(prefix + "*")))
    self.evaluate(v1.assign(1.))
    self.evaluate(saver.restore(prefix))
    self.assertEqual(2., self.evaluate(v1))

    v2 = resource_variable_ops.ResourceVariable(3.)
    self.evaluate(v2.initializer)
    second_saver = functional_saver._SingleDeviceSaver(
        saveable_object_util.saveable_objects_for_op(v2, "x"))
    self.evaluate(second_saver.restore(prefix))
    self.assertEqual(2., self.evaluate(v2))

  @test_util.run_in_graph_and_eager_modes
  def test_resource_variable_use_localhost(self):
    v1 = resource_variable_ops.ResourceVariable(2.)
    self.evaluate(v1.initializer)
    saver = functional_saver._SingleDeviceSaver(
        saveable_object_util.saveable_objects_for_op(v1, "x"))
    prefix = os.path.join(self.get_temp_dir(), "ckpt")
    self.evaluate(saver.save(constant_op.constant(prefix), self.local_options))
    self.assertEqual(2, len(gfile.Glob(prefix + "*")))
    self.evaluate(v1.assign(1.))
    self.evaluate(saver.restore(prefix, self.local_options))
    self.assertEqual(2., self.evaluate(v1))

    v2 = resource_variable_ops.ResourceVariable(3.)
    self.evaluate(v2.initializer)
    second_saver = functional_saver._SingleDeviceSaver(
        saveable_object_util.saveable_objects_for_op(v2, "x"))
    self.evaluate(second_saver.restore(prefix, self.local_options))
    self.assertEqual(2., self.evaluate(v2))

    # In graph mode, verify that the save and restore ops were set to run on
    # localhost.
    if not context.executing_eagerly():
      for op in ops.get_default_graph().get_operations():
        if op.type in ("SaveV2", "RestoreV2"):
          self.assertEqual(LOCALHOST, op.device)

  def test_to_proto(self):
    v1 = resource_variable_ops.ResourceVariable(2.)
    saver = functional_saver.MultiDeviceSaver(
        saveable_object_util.saveable_objects_for_op(v1, "x"))
    prefix = os.path.join(self.get_temp_dir(), "ckpt")

    proto_accumulator = []
    wrapped = wrap_function.wrap_function(
        lambda: proto_accumulator.append(saver.to_proto()), signature=())
    self.assertEqual(1, len(proto_accumulator))
    proto = proto_accumulator[0]
    save = wrapped.prune(
        feeds=wrapped.graph.get_tensor_by_name(proto.filename_tensor_name),
        fetches=wrapped.graph.get_tensor_by_name(proto.save_tensor_name))
    restore = wrapped.prune(
        feeds=wrapped.graph.get_tensor_by_name(proto.filename_tensor_name),
        fetches=wrapped.graph.get_operation_by_name(proto.restore_op_name))
    save_path = save(constant_op.constant(prefix))
    v1.assign(1.)
    restore(constant_op.constant(save_path))
    self.assertEqual(2., self.evaluate(v1))

    v2 = resource_variable_ops.ResourceVariable(3.)
    second_saver = functional_saver.MultiDeviceSaver(
        saveable_object_util.saveable_objects_for_op(v2, "x"))
    second_saver.restore(save_path)
    self.assertEqual(2., self.evaluate(v2))

  def test_checkpoint_is_sharded_by_task(self):
    servers = [server_lib.Server.create_local_server() for _ in range(3)]
    cluster_spec = server_lib.ClusterSpec({
        "worker": [s.target[len("grpc://"):] for s in servers]})
    remote.connect_to_cluster(cluster_spec)
    with ops.device("/job:worker/task:0/cpu:0"):
      v0 = resource_variable_ops.ResourceVariable(0.)
    with ops.device("/job:worker/task:1/cpu:0"):
      v1 = resource_variable_ops.ResourceVariable(1.)
    with ops.device("/job:worker/task:2/cpu:0"):
      v2 = resource_variable_ops.ResourceVariable(2.)

    self.evaluate([v0.initializer, v1.initializer, v2.initializer])
    saver = functional_saver.MultiDeviceSaver(
        list(saveable_object_util.saveable_objects_for_op(v0, "v0")) +
        list(saveable_object_util.saveable_objects_for_op(v1, "v1")) +
        list(saveable_object_util.saveable_objects_for_op(v2, "v2")))
    prefix = os.path.join(self.get_temp_dir(), "ckpt")
    self.evaluate(saver.save(constant_op.constant(prefix)))
    self.assertEqual(4, len(gfile.Glob(prefix + "*")))
    self.evaluate(v0.assign(-1.))
    self.evaluate(v1.assign(-1.))
    self.evaluate(v2.assign(-1.))
    self.evaluate(saver.restore(constant_op.constant(prefix)))
    self.assertEqual(0., self.evaluate(v0))
    self.assertEqual(1., self.evaluate(v1))
    self.assertEqual(2., self.evaluate(v2))

  @test_util.run_in_graph_and_eager_modes
  def test_checkpoint_multi_device_using_localhost(self):
    with ops.device("cpu:0"):
      v0 = resource_variable_ops.ResourceVariable(0.)
    with ops.device("cpu:1"):
      v1 = resource_variable_ops.ResourceVariable(1.)
    with ops.device("cpu:2"):
      v2 = resource_variable_ops.ResourceVariable(2.)

    self.evaluate([v0.initializer, v1.initializer, v2.initializer])
    saver = functional_saver.MultiDeviceSaver(
        list(saveable_object_util.saveable_objects_for_op(v0, "v0")) +
        list(saveable_object_util.saveable_objects_for_op(v1, "v1")) +
        list(saveable_object_util.saveable_objects_for_op(v2, "v2")))
    prefix = os.path.join(self.get_temp_dir(), "ckpt")
    self.evaluate(saver.save(constant_op.constant(prefix), self.local_options))
    self.assertEqual(2, len(gfile.Glob(prefix + "*")))
    self.evaluate(v0.assign(-1.))
    self.evaluate(v1.assign(-1.))
    self.evaluate(v2.assign(-1.))
    self.evaluate(
        saver.restore(constant_op.constant(prefix), self.local_options))
    self.assertEqual(0., self.evaluate(v0))
    self.assertEqual(1., self.evaluate(v1))
    self.assertEqual(2., self.evaluate(v2))

    # In graph mode, verify that the save and restore ops were set to run on
    # localhost.
    if not context.executing_eagerly():
      for op in ops.get_default_graph().get_operations():
        if op.type in ("SaveV2", "RestoreV2", "MergeV2Checkpoints"):
          self.assertEqual(LOCALHOST, op.device)

  def test_callbacks_run(self):
    #  Use dict because an int would be shadowed inside callback.
    called = {
        "save": 0,
        "restore": 0,
    }

    class DummyHook(saveable_hook.SaveableHook):

      def before_save(self):
        called["save"] += 1

      def after_restore(self):
        called["restore"] += 1

    saveable = DummyHook(name="dummy")

    saver = functional_saver.MultiDeviceSaver([saveable])
    prefix = os.path.join(self.get_temp_dir(), "ckpt")

    self.evaluate(saver.save(constant_op.constant(prefix)))
    self.assertEqual({"save": 1, "restore": 0}, called)

    self.evaluate(saver.restore(prefix))
    self.assertEqual({"save": 1, "restore": 1}, called)


if __name__ == "__main__":
  ops.enable_eager_execution()
  test.main()
