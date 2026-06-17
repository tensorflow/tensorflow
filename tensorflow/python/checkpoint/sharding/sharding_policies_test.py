# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for checkpoint sharding policies."""

import random
import re
import string

from absl import logging

from tensorflow.python.checkpoint import checkpoint
from tensorflow.python.checkpoint import checkpoint_options
from tensorflow.python.checkpoint import graph_view
from tensorflow.python.checkpoint.sharding import sharding_policies
from tensorflow.python.checkpoint.sharding import sharding_util
from tensorflow.python.eager import def_function
from tensorflow.python.eager import remote
from tensorflow.python.eager import test
from tensorflow.python.framework import device as device_lib
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_util
from tensorflow.python.module import module
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import gfile
from tensorflow.python.training import server_lib
from tensorflow.python.training.saving import saveable_object
from tensorflow.python.training.saving import saveable_object_util


class ShardingPoliciesTest(test.TestCase):

  def _get_shardable_tensors_by_task(self, root):
    serialized_tensors, _, _, _ = (
        checkpoint.TrackableSaver(graph_view.ObjectGraphView(root))
        ._gather_serialized_tensors(None))

    shardable_tensors_by_task = {}
    for obj, tensor_dict in serialized_tensors.items():
      # Divide tensor_dict by device.
      for checkpoint_key, tensor_slice_dict in tensor_dict.items():
        if not isinstance(tensor_slice_dict, dict):
          # Make sure that maybe_tensor is structured as {slice_spec -> tensor}.
          tensor_slice_dict = {"": tensor_slice_dict}
        for slice_spec, tensor_save_spec in tensor_slice_dict.items():
          if not isinstance(tensor_save_spec, saveable_object.SaveSpec):
            tensor_save_spec = saveable_object.SaveSpec(
                tensor=tensor_save_spec,
                slice_spec=slice_spec,
                name=checkpoint_key,
                dtype=tensor_save_spec.dtype,
                device=tensor_save_spec.device)
          save_spec_tensor = tensor_save_spec.tensor
          device = (device_lib.DeviceSpec.from_string(tensor_save_spec.device)
                    if isinstance(tensor_save_spec.device, str)
                    else tensor_save_spec.device)
          task = device_lib.DeviceSpec.from_string(
              saveable_object_util.set_cpu0(device.to_string()))
          shardable_tensors_by_task.setdefault(task, []).append(
              sharding_util.ShardableTensor(
                  _tensor_save_spec=tensor_save_spec,
                  tensor=save_spec_tensor,
                  dtype=tensor_save_spec.dtype,
                  device=device,
                  name=tensor_save_spec.name,
                  shape=save_spec_tensor.shape,
                  slice_spec=slice_spec,
                  checkpoint_key=checkpoint_key,
                  trackable=obj))
    return shardable_tensors_by_task.values()

  def test_ShardByTaskPolicy(self):
    servers = [server_lib.Server.create_local_server() for _ in range(3)]
    cluster_spec = server_lib.ClusterSpec({
        "worker": [s.target[len("grpc://"):] for s in servers]})
    remote.connect_to_cluster(cluster_spec)
    root = module.Module()
    with ops.device("/job:worker/task:0/cpu:0"):
      v0 = resource_variable_ops.ResourceVariable(0.0, name="v0")
    with ops.device("/job:worker/task:1/cpu:0"):
      v1 = resource_variable_ops.ResourceVariable(1.0, name="v1")
    with ops.device("/job:worker/task:2/cpu:0"):
      v2 = resource_variable_ops.ResourceVariable(2.0, name="v2")
    root.v0 = v0
    root.v1 = v1
    root.v2 = v2

    shardable_tensors = self._get_shardable_tensors_by_task(root)

    callback = sharding_policies.ShardByTaskPolicy()
    shards = []
    for tensors in shardable_tensors:
      shards.extend(callback(tensors))

    self.assertAllEqual(
        [set(shard.keys()) for shard in shards],
        [
            {"v0/.ATTRIBUTES/VARIABLE_VALUE"},
            {"v1/.ATTRIBUTES/VARIABLE_VALUE"},
            {"v2/.ATTRIBUTES/VARIABLE_VALUE"},
            {"_CHECKPOINTABLE_OBJECT_GRAPH"}
        ])

    self.assertEqual(
        self.evaluate(shards[0]["v0/.ATTRIBUTES/VARIABLE_VALUE"][""]),
        v0.numpy())
    self.assertEqual(
        self.evaluate(shards[1]["v1/.ATTRIBUTES/VARIABLE_VALUE"][""]),
        v1.numpy())
    self.assertEqual(
        self.evaluate(shards[2]["v2/.ATTRIBUTES/VARIABLE_VALUE"][""]),
        v2.numpy())

  def test_CheckpointOption_ShardByTaskPolicy(self):
    servers = [server_lib.Server.create_local_server() for _ in range(3)]
    cluster_spec = server_lib.ClusterSpec({
        "worker": [s.target[len("grpc://"):] for s in servers]})
    remote.connect_to_cluster(cluster_spec)
    root = module.Module()
    with ops.device("/job:worker/task:0/cpu:0"):
      v0 = resource_variable_ops.ResourceVariable(0.0, name="v0")
    self.evaluate(v0.initializer)
    with ops.device("/job:worker/task:1/cpu:0"):
      v1 = resource_variable_ops.ResourceVariable(1.0, name="v1")
    self.evaluate(v1.initializer)
    with ops.device("/job:worker/task:2/cpu:0"):
      v2 = resource_variable_ops.ResourceVariable(2.0, name="v2")
    self.evaluate(v2.initializer)
    root.v0 = v0
    root.v1 = v1
    root.v2 = v2

    tmp_dir = self.create_tempdir("ckpt")
    ckpt = checkpoint.Checkpoint(root)
    save_path = ckpt.save(
        tmp_dir, options=checkpoint_options.CheckpointOptions(
            experimental_sharding_callback=(
                sharding_policies.ShardByTaskPolicy())))
    self.assertLen(gfile.Glob(save_path + ".data*"), 4)
    ckpt.restore(save_path)

  @test_util.run_in_graph_and_eager_modes
  def test_MaxShardSizePolicy_1D(self):
    root = module.Module()
    with ops.device("cpu:0"):
      v0 = resource_variable_ops.ResourceVariable([0.0, 1.0, 2.0, 3.0],
                                                  name="v0",
                                                  dtype=dtypes.float32)
      v1 = resource_variable_ops.ResourceVariable([[4],
                                                   [5],
                                                   [6],
                                                   [7]],
                                                  name="v1",
                                                  dtype=dtypes.int32)
    self.evaluate(v0.initializer)
    self.evaluate(v1.initializer)
    root.v0 = v0
    root.v1 = v1

    v0_name = "v0/.ATTRIBUTES/VARIABLE_VALUE"
    v1_name = "v1/.ATTRIBUTES/VARIABLE_VALUE"

    class V0SaveSliceInfo(variables.Variable.SaveSliceInfo):
      def __init__(self, var_offset, var_shape):
        super().__init__(
            full_name=v0_name,
            full_shape=tensor_shape.TensorShape(dims=[4]),
            var_offset=var_offset,
            var_shape=var_shape)

    class V1SaveSliceInfo(variables.Variable.SaveSliceInfo):
      def __init__(self, var_offset, var_shape):
        super().__init__(
            full_name=v1_name,
            full_shape=tensor_shape.TensorShape(dims=[4, 1]),
            var_offset=var_offset,
            var_shape=var_shape)

    shardable_tensors = self._get_shardable_tensors_by_task(root)

    # Test sharding the v0 & v1 tensors with different max shard sizes.

    # max_shard_size: 4 bytes
    # Each element of v0/v1 is a 32 bit/4 byte value, so each variable should be
    # split into 4 shards.
    callback = sharding_policies.MaxShardSizePolicy(max_shard_size=4)
    shards = []
    for tensors in shardable_tensors:
      shards.extend(callback(tensors))

    self.assertEqual(
        [set(shard.keys()) for shard in shards],
        [
            {"v0/.ATTRIBUTES/VARIABLE_VALUE",},
            {"v0/.ATTRIBUTES/VARIABLE_VALUE",},
            {"v0/.ATTRIBUTES/VARIABLE_VALUE",},
            {"v0/.ATTRIBUTES/VARIABLE_VALUE",},
            {"v1/.ATTRIBUTES/VARIABLE_VALUE",},
            {"v1/.ATTRIBUTES/VARIABLE_VALUE",},
            {"v1/.ATTRIBUTES/VARIABLE_VALUE",},
            {"v1/.ATTRIBUTES/VARIABLE_VALUE", "_CHECKPOINTABLE_OBJECT_GRAPH",}
        ])

    # V0
    slice_spec = V0SaveSliceInfo(var_offset=[0], var_shape=[1]).spec
    self.assertEqual(self.evaluate(shards[0][v0_name][slice_spec]), 0.0)

    slice_spec = V0SaveSliceInfo(var_offset=[1], var_shape=[1]).spec
    self.assertEqual(self.evaluate(shards[1][v0_name][slice_spec]), 1.0)

    slice_spec = V0SaveSliceInfo(var_offset=[2], var_shape=[1]).spec
    self.assertEqual(self.evaluate(shards[2][v0_name][slice_spec]), 2.0)

    slice_spec = V0SaveSliceInfo(var_offset=[3], var_shape=[1]).spec
    self.assertEqual(self.evaluate(shards[3][v0_name][slice_spec]), 3.0)

    # V1
    slice_spec = V1SaveSliceInfo(var_offset=[0, 0], var_shape=[1, 1]).spec
    self.assertEqual(self.evaluate(shards[4][v1_name][slice_spec]), [4])

    slice_spec = V1SaveSliceInfo(var_offset=[1, 0], var_shape=[1, 1]).spec
    self.assertEqual(self.evaluate(shards[5][v1_name][slice_spec]), [5])

    slice_spec = V1SaveSliceInfo(var_offset=[2, 0], var_shape=[1, 1]).spec
    self.assertEqual(self.evaluate(shards[6][v1_name][slice_spec]), [6])

    slice_spec = V1SaveSliceInfo(var_offset=[3, 0], var_shape=[1, 1]).spec
    self.assertEqual(self.evaluate(shards[7][v1_name][slice_spec]), [7])

    # max_shard_size: 8 bytes
    # v0/v1 haven't changed, so they should now be split into 2 shards each.
    callback = sharding_policies.MaxShardSizePolicy(max_shard_size=8)
    shards = []
    for tensors in shardable_tensors:
      shards.extend(callback(tensors))

    self.assertEqual(
        [set(shard.keys()) for shard in shards],
        [
            {"v0/.ATTRIBUTES/VARIABLE_VALUE",},
            {"v0/.ATTRIBUTES/VARIABLE_VALUE",},
            {"v1/.ATTRIBUTES/VARIABLE_VALUE",},
            {"v1/.ATTRIBUTES/VARIABLE_VALUE", "_CHECKPOINTABLE_OBJECT_GRAPH",}
        ])

    # V0
    slice_spec = V0SaveSliceInfo(var_offset=[0], var_shape=[2]).spec
    self.assertAllEqual(
        self.evaluate(shards[0][v0_name][slice_spec]), [0.0, 1.0])

    slice_spec = V0SaveSliceInfo(var_offset=[2], var_shape=[2]).spec
    self.assertAllEqual(
        self.evaluate(shards[1][v0_name][slice_spec]), [2.0, 3.0])

    # V1
    slice_spec = V1SaveSliceInfo(var_offset=[0, 0], var_shape=[2, 1]).spec
    self.assertAllEqual(
        self.evaluate(shards[2][v1_name][slice_spec]), [[4], [5]])

    slice_spec = V1SaveSliceInfo(var_offset=[2, 0], var_shape=[2, 1]).spec
    self.assertAllEqual(
        self.evaluate(shards[3][v1_name][slice_spec]), [[6], [7]])

    # max_shard_size: 10 bytes
    # 10 bytes is an uneven boundary for 4 byte elements. v0/v1 should be split
    # into 2 shards each.
    callback = sharding_policies.MaxShardSizePolicy(max_shard_size=10)
    shards = []
    for tensors in shardable_tensors:
      shards.extend(callback(tensors))

    self.assertEqual(
        [set(shard.keys()) for shard in shards],
        [
            {"v0/.ATTRIBUTES/VARIABLE_VALUE",},
            {"v0/.ATTRIBUTES/VARIABLE_VALUE",},
            {"v1/.ATTRIBUTES/VARIABLE_VALUE",},
            {"v1/.ATTRIBUTES/VARIABLE_VALUE", "_CHECKPOINTABLE_OBJECT_GRAPH",}
        ])

    # V0
    slice_spec = V0SaveSliceInfo(var_offset=[0], var_shape=[2]).spec
    self.assertAllEqual(
        self.evaluate(shards[0][v0_name][slice_spec]), [0.0, 1.0])

    slice_spec = V0SaveSliceInfo(var_offset=[2], var_shape=[2]).spec
    self.assertAllEqual(
        self.evaluate(shards[1][v0_name][slice_spec]), [2.0, 3.0])

    # V1
    slice_spec = V1SaveSliceInfo(var_offset=[0, 0], var_shape=[2, 1]).spec
    self.assertAllEqual(
        self.evaluate(shards[2][v1_name][slice_spec]), [[4], [5]])

    slice_spec = V1SaveSliceInfo(var_offset=[2, 0], var_shape=[2, 1]).spec
    self.assertAllEqual(
        self.evaluate(shards[3][v1_name][slice_spec]), [[6], [7]])

    # max_shard_size: 16 bytes
    # 16 bytes the exact size of each variable, so they should get 1 shard each.
    callback = sharding_policies.MaxShardSizePolicy(max_shard_size=16)
    shards = []
    for tensors in shardable_tensors:
      shards.extend(callback(tensors))

    self.assertEqual(
        [set(shard.keys()) for shard in shards],
        [
            {"v0/.ATTRIBUTES/VARIABLE_VALUE",},
            {"v1/.ATTRIBUTES/VARIABLE_VALUE", "_CHECKPOINTABLE_OBJECT_GRAPH",}
        ])

    # V0
    slice_spec = V0SaveSliceInfo(var_offset=[0], var_shape=[4]).spec
    self.assertAllEqual(
        self.evaluate(shards[0][v0_name][slice_spec]), [0.0, 1.0, 2.0, 3.0])

    # V1
    slice_spec = V1SaveSliceInfo(var_offset=[0, 0], var_shape=[4, 1]).spec
    self.assertAllEqual(
        self.evaluate(shards[1][v1_name][slice_spec]), [[4], [5], [6], [7]])

    # max_shard_size: 18 bytes
    # 18 bytes slightly larger than the size of each variable, but not large
    # enough to fit another 4 byte element, so they should get 1 shard each.
    callback = sharding_policies.MaxShardSizePolicy(max_shard_size=18)
    shards = []
    for tensors in shardable_tensors:
      shards.extend(callback(tensors))

    self.assertEqual(
        [set(shard.keys()) for shard in shards],
        [
            {"v0/.ATTRIBUTES/VARIABLE_VALUE",},
            {"v1/.ATTRIBUTES/VARIABLE_VALUE", "_CHECKPOINTABLE_OBJECT_GRAPH",}
        ])

    # V0
    slice_spec = V0SaveSliceInfo(var_offset=[0], var_shape=[4]).spec
    self.assertAllEqual(
        self.evaluate(shards[0][v0_name][slice_spec]), [0.0, 1.0, 2.0, 3.0])

    # V1
    slice_spec = V1SaveSliceInfo(var_offset=[0, 0], var_shape=[4, 1]).spec
    self.assertAllEqual(
        self.evaluate(shards[1][v1_name][slice_spec]), [[4], [5], [6], [7]])

  @test_util.run_in_graph_and_eager_modes
  def test_MaxShardSizePolicy_2D(self):
    root = module.Module()
    with ops.device("cpu:0"):
      v0 = resource_variable_ops.ResourceVariable([[0, 1],
                                                   [2, 3],
                                                   [4, 5]],
                                                  name="v0")
      v1 = resource_variable_ops.ResourceVariable([[[6.0], [7.0]],
                                                   [[8.0], [9.0]],
                                                   [[10.0], [11.0]]], name="v1")
    self.evaluate(v0.initializer)
    self.evaluate(v1.initializer)
    root.v0 = v0
    root.v1 = v1

    v0_name = "v0/.ATTRIBUTES/VARIABLE_VALUE"
    v1_name = "v1/.ATTRIBUTES/VARIABLE_VALUE"

    class V0SaveSliceInfo(variables.Variable.SaveSliceInfo):
      def __init__(self, var_offset, var_shape):
        super().__init__(
            full_name=v0_name,
            full_shape=tensor_shape.TensorShape(dims=[3, 2]),
            var_offset=var_offset,
            var_shape=var_shape)

    class V1SaveSliceInfo(variables.Variable.SaveSliceInfo):
      def __init__(self, var_offset, var_shape):
        super().__init__(
            full_name=v1_name,
            full_shape=tensor_shape.TensorShape(dims=[3, 2, 1]),
            var_offset=var_offset,
            var_shape=var_shape)

    shardable_tensors = self._get_shardable_tensors_by_task(root)

    # Test sharding the v0 & v1 tensors with different max shard sizes.

    # max_shard_size: 8 bytes
    # Each element of v0/v1 is a 32 bit/4 byte value, so each variable should be
    # split into 3 shards.
    callback = sharding_policies.MaxShardSizePolicy(max_shard_size=8)
    shards = []
    for tensors in shardable_tensors:
      shards.extend(callback(tensors))

    self.assertEqual(
        [set(shard.keys()) for shard in shards],
        [
            {"v0/.ATTRIBUTES/VARIABLE_VALUE",},
            {"v0/.ATTRIBUTES/VARIABLE_VALUE",},
            {"v0/.ATTRIBUTES/VARIABLE_VALUE",},
            {"v1/.ATTRIBUTES/VARIABLE_VALUE",},
            {"v1/.ATTRIBUTES/VARIABLE_VALUE",},
            {"v1/.ATTRIBUTES/VARIABLE_VALUE", "_CHECKPOINTABLE_OBJECT_GRAPH",}
        ])

    # V0
    slice_spec = V0SaveSliceInfo(var_offset=[0, 0], var_shape=[1, 2]).spec
    self.assertAllEqual(
        self.evaluate(shards[0][v0_name][slice_spec]), [[0, 1]])

    slice_spec = V0SaveSliceInfo(var_offset=[1, 0], var_shape=[1, 2]).spec
    self.assertAllEqual(
        self.evaluate(shards[1][v0_name][slice_spec]), [[2, 3]])

    slice_spec = V0SaveSliceInfo(var_offset=[2, 0], var_shape=[1, 2]).spec
    self.assertAllEqual(
        self.evaluate(shards[2][v0_name][slice_spec]), [[4, 5]])

    # V1
    slice_spec = V1SaveSliceInfo(var_offset=[0, 0, 0], var_shape=[1, 2, 1]).spec
    self.assertAllEqual(
        self.evaluate(shards[3][v1_name][slice_spec]), [[[6.0], [7.0]]])

    slice_spec = V1SaveSliceInfo(var_offset=[1, 0, 0], var_shape=[1, 2, 1]).spec
    self.assertAllEqual(
        self.evaluate(shards[4][v1_name][slice_spec]), [[[8.0], [9.0]]])

    slice_spec = V1SaveSliceInfo(var_offset=[2, 0, 0], var_shape=[1, 2, 1]).spec
    self.assertAllEqual(
        self.evaluate(shards[5][v1_name][slice_spec]), [[[10.0], [11.0]]])

    # max_shard_size: 10 bytes
    # 10 bytes is an uneven boundary for 4 byte elements. v0/v1 should be split
    # into 3 shards each.
    callback = sharding_policies.MaxShardSizePolicy(max_shard_size=10)
    shards = []
    for tensors in shardable_tensors:
      shards.extend(callback(tensors))

    self.assertEqual(
        [set(shard.keys()) for shard in shards],
        [
            {"v0/.ATTRIBUTES/VARIABLE_VALUE",},
            {"v0/.ATTRIBUTES/VARIABLE_VALUE",},
            {"v0/.ATTRIBUTES/VARIABLE_VALUE",},
            {"v1/.ATTRIBUTES/VARIABLE_VALUE",},
            {"v1/.ATTRIBUTES/VARIABLE_VALUE",},
            {"v1/.ATTRIBUTES/VARIABLE_VALUE", "_CHECKPOINTABLE_OBJECT_GRAPH",}
        ])

    # V0
    slice_spec = V0SaveSliceInfo(var_offset=[0, 0], var_shape=[1, 2]).spec
    self.assertAllEqual(
        self.evaluate(shards[0][v0_name][slice_spec]), [[0, 1]])

    slice_spec = V0SaveSliceInfo(var_offset=[1, 0], var_shape=[1, 2]).spec
    self.assertAllEqual(
        self.evaluate(shards[1][v0_name][slice_spec]), [[2, 3]])

    slice_spec = V0SaveSliceInfo(var_offset=[2, 0], var_shape=[1, 2]).spec
    self.assertAllEqual(
        self.evaluate(shards[2][v0_name][slice_spec]), [[4, 5]])

    # V1
    slice_spec = V1SaveSliceInfo(var_offset=[0, 0, 0], var_shape=[1, 2, 1]).spec
    self.assertAllEqual(
        self.evaluate(shards[3][v1_name][slice_spec]), [[[6.0], [7.0]]])

    slice_spec = V1SaveSliceInfo(var_offset=[1, 0, 0], var_shape=[1, 2, 1]).spec
    self.assertAllEqual(
        self.evaluate(shards[4][v1_name][slice_spec]), [[[8.0], [9.0]]])

    slice_spec = V1SaveSliceInfo(var_offset=[2, 0, 0], var_shape=[1, 2, 1]).spec
    self.assertAllEqual(
        self.evaluate(shards[5][v1_name][slice_spec]), [[[10.0], [11.0]]])

    # max_shard_size: 12 bytes
    # 12 bytes is enough to fit 3 elements per variable in each shard.
    # v0/v1 should be split into 2 shards each.
    callback = sharding_policies.MaxShardSizePolicy(max_shard_size=12)
    shards = []
    for tensors in shardable_tensors:
      shards.extend(callback(tensors))

    self.assertEqual(
        [set(shard.keys()) for shard in shards],
        [
            {"v0/.ATTRIBUTES/VARIABLE_VALUE",},
            {"v0/.ATTRIBUTES/VARIABLE_VALUE",},
            {"v1/.ATTRIBUTES/VARIABLE_VALUE",},
            {"v1/.ATTRIBUTES/VARIABLE_VALUE", "_CHECKPOINTABLE_OBJECT_GRAPH",}
        ])

    # V0
    slice_spec = V0SaveSliceInfo(var_offset=[0, 0], var_shape=[3, 1]).spec
    self.assertAllEqual(
        self.evaluate(shards[0][v0_name][slice_spec]), [[0], [2], [4]])

    slice_spec = V0SaveSliceInfo(var_offset=[0, 1], var_shape=[3, 1]).spec
    self.assertAllEqual(
        self.evaluate(shards[1][v0_name][slice_spec]), [[1], [3], [5]])

    # V1
    slice_spec = V1SaveSliceInfo(var_offset=[0, 0, 0], var_shape=[3, 1, 1]).spec
    self.assertAllEqual(
        self.evaluate(shards[2][v1_name][slice_spec]),
        [[[6.0]], [[8.0]], [[10.0]]])

    slice_spec = V1SaveSliceInfo(var_offset=[0, 1, 0], var_shape=[3, 1, 1]).spec
    self.assertAllEqual(
        self.evaluate(shards[3][v1_name][slice_spec]),
        [[[7.0]], [[9.0]], [[11.0]]])

    # max_shard_size: 16 bytes
    # Each variable should be split into 1.5 shards. The middle shard will
    # contain elements from both variables.
    callback = sharding_policies.MaxShardSizePolicy(max_shard_size=16)
    shards = []
    for tensors in shardable_tensors:
      shards.extend(callback(tensors))

    self.assertEqual(
        [set(shard.keys()) for shard in shards],
        [
            {"v0/.ATTRIBUTES/VARIABLE_VALUE",},
            {"v0/.ATTRIBUTES/VARIABLE_VALUE", "v1/.ATTRIBUTES/VARIABLE_VALUE"},
            {"v1/.ATTRIBUTES/VARIABLE_VALUE", "_CHECKPOINTABLE_OBJECT_GRAPH",}
        ])

    # V0
    slice_spec = V0SaveSliceInfo(var_offset=[0, 0], var_shape=[2, 2]).spec
    self.assertAllEqual(
        self.evaluate(shards[0][v0_name][slice_spec]), [[0, 1], [2, 3]])

    slice_spec = V0SaveSliceInfo(var_offset=[2, 0], var_shape=[1, 2]).spec
    self.assertAllEqual(
        self.evaluate(shards[1][v0_name][slice_spec]), [[4, 5]])

    # V1
    slice_spec = V1SaveSliceInfo(var_offset=[0, 0, 0], var_shape=[1, 2, 1]).spec
    self.assertAllEqual(
        self.evaluate(shards[1][v1_name][slice_spec]), [[[6.0], [7.0]]])

    slice_spec = V1SaveSliceInfo(var_offset=[1, 0, 0], var_shape=[2, 2, 1]).spec
    self.assertAllEqual(
        self.evaluate(shards[2][v1_name][slice_spec]),
        [[[8.0], [9.0]], [[10.0], [11.0]]])

  @test_util.run_in_graph_and_eager_modes
  def test_MaxShardSizePolicy_Strings(self):
    v_strings = [
        "".join(random.choices(string.ascii_uppercase + string.digits, k=10))
        for _ in range(4)]

    root = module.Module()
    with ops.device("cpu:0"):
      v0 = resource_variable_ops.ResourceVariable(v_strings, name="v0",
                                                  dtype=dtypes.string)
    self.evaluate(v0.initializer)
    root.v0 = v0

    v0_name = "v0/.ATTRIBUTES/VARIABLE_VALUE"

    class V0SaveSliceInfo(variables.Variable.SaveSliceInfo):
      def __init__(self, var_offset, var_shape):
        super().__init__(
            full_name=v0_name,
            full_shape=tensor_shape.TensorShape(dims=[4]),
            var_offset=var_offset,
            var_shape=var_shape)

    shardable_tensors = self._get_shardable_tensors_by_task(root)

    # Test sharding the v0 & v1 tensors with different max shard sizes.

    # max_shard_size: 10 bytes
    # Each string in v0 is 10 bytes, so there should be 1 string per shard.
    callback = sharding_policies.MaxShardSizePolicy(max_shard_size=10)
    shards = []
    for tensors in shardable_tensors:
      shards.extend(callback(tensors))

    self.assertEqual(
        [set(shard.keys()) for shard in shards],
        [
            {"v0/.ATTRIBUTES/VARIABLE_VALUE",},
            {"v0/.ATTRIBUTES/VARIABLE_VALUE",},
            {"v0/.ATTRIBUTES/VARIABLE_VALUE",},
            {"v0/.ATTRIBUTES/VARIABLE_VALUE", "_CHECKPOINTABLE_OBJECT_GRAPH",}
        ])

    slice_spec = V0SaveSliceInfo(var_offset=[0], var_shape=[1]).spec
    self.assertAllEqual(
        self.evaluate(shards[0][v0_name][slice_spec]), [v_strings[0]])

    slice_spec = V0SaveSliceInfo(var_offset=[1], var_shape=[1]).spec
    self.assertAllEqual(
        self.evaluate(shards[1][v0_name][slice_spec]), [v_strings[1]])

    slice_spec = V0SaveSliceInfo(var_offset=[2], var_shape=[1]).spec
    self.assertAllEqual(
        self.evaluate(shards[2][v0_name][slice_spec]), [v_strings[2]])

    slice_spec = V0SaveSliceInfo(var_offset=[3], var_shape=[1]).spec
    self.assertAllEqual(
        self.evaluate(shards[3][v0_name][slice_spec]), [v_strings[3]])

  @test_util.run_in_graph_and_eager_modes
  def test_MaxShardSizePolicy_LargeScalar(self):
    v_string = "".join(random.choices(
        string.ascii_uppercase + string.digits, k=10)).encode("utf-8")
    root = module.Module()
    with ops.device("cpu:0"):
      v0 = resource_variable_ops.ResourceVariable(
          v_string, name="v0", dtype=dtypes.string)
    self.evaluate(v0.initializer)
    root.v0 = v0

    v0_name = "v0/.ATTRIBUTES/VARIABLE_VALUE"

    shardable_tensors = self._get_shardable_tensors_by_task(root)

    # max_shard_size: 8 bytes
    callback = sharding_policies.MaxShardSizePolicy(max_shard_size=8)
    shards = []
    for tensors in shardable_tensors:
      shards.extend(callback(tensors))

    self.assertEqual(
        [set(shard.keys()) for shard in shards],
        [
            {"_CHECKPOINTABLE_OBJECT_GRAPH",},
            {"v0/.ATTRIBUTES/VARIABLE_VALUE",}
        ])

    tensor_val = (self.evaluate(shards[1][v0_name][""])
                  if ops.context.executing_eagerly()
                  else shards[1][v0_name][""])
    self.assertEqual(tensor_val, v_string)

  @test_util.run_in_graph_and_eager_modes
  def test_CheckpointOption_MaxShardSizePolicy(self):
    root = module.Module()
    with ops.device("cpu:0"):
      v0 = resource_variable_ops.ResourceVariable([[0, 1],
                                                   [2, 3],
                                                   [4, 5]],
                                                  name="v0")
      v1 = resource_variable_ops.ResourceVariable([[[6.0], [7.0]],
                                                   [[8.0], [9.0]],
                                                   [[10.0], [11.0]]], name="v1")
      v2 = resource_variable_ops.ResourceVariable("test_string", name="v1")
    self.evaluate(v0.initializer)
    self.evaluate(v1.initializer)
    self.evaluate(v2.initializer)
    root.v0 = v0
    root.v1 = v1
    root.v2 = v2

    tmp_dir = self.create_tempdir("ckpt")
    ckpt = checkpoint.Checkpoint(root)
    save_path = ckpt.save(
        tmp_dir, options=checkpoint_options.CheckpointOptions(
            experimental_sharding_callback=(
                sharding_policies.MaxShardSizePolicy(max_shard_size=10))))
    # 8 files = 3 shards for v0, 3 for v1, 1 for v2, and 1 for the object graph
    self.assertLen(gfile.Glob(save_path + ".data*"), 8)
    ckpt.restore(save_path)

  @test_util.run_in_graph_and_eager_modes
  def test_MaxShardSizePolicy_PreSlicedTensor(self):
    root = module.Module()

    sliced_v0_name = "sliced_v0/.ATTRIBUTES/VARIABLE_VALUE"

    class V0SaveSliceInfo(variables.Variable.SaveSliceInfo):
      def __init__(self, var_offset, var_shape):
        super().__init__(
            full_name=sliced_v0_name,
            full_shape=tensor_shape.TensorShape(dims=[2, 5]),
            var_offset=var_offset,
            var_shape=var_shape)

    v0_slice_spec = V0SaveSliceInfo(var_offset=[0, 1], var_shape=[2, 3])

    class ResourceVariableWithSliceSpec(resource_variable_ops.ResourceVariable):
      def _serialize_to_tensors(self):
        ckpt_key, tensor = list(super()._serialize_to_tensors().items())[0]
        return {ckpt_key: {v0_slice_spec.spec: tensor}}

    with ops.device("cpu:0"):
      # full_v0 = [[0.0, 1.0, 2.0, 3.0, 4.0],
      #            [5.0, 6.0, 7.0, 8.0, 9.0]]
      sliced_v0 = ResourceVariableWithSliceSpec([[1.0, 2.0, 3.0],
                                                 [6.0, 7.0, 8.0]],
                                                name="sliced_v0",
                                                dtype=dtypes.float32)
      sliced_v0._set_save_slice_info(v0_slice_spec)
    self.evaluate(sliced_v0.initializer)
    root.sliced_v0 = sliced_v0

    shardable_tensors = self._get_shardable_tensors_by_task(root)

    # Test sharding the pre-sliced v0 tensor with different max shard sizes.

    # max_shard_size: 8 bytes
    # Each element of v0 is a 32 bit/4 byte value, so v0 should be split into 3
    # shards containing 2 elements each.
    callback = sharding_policies.MaxShardSizePolicy(max_shard_size=8)
    shards = []
    for tensors in shardable_tensors:
      shards.extend(callback(tensors))

    self.assertEqual(
        [set(shard.keys()) for shard in shards],
        [
            {"sliced_v0/.ATTRIBUTES/VARIABLE_VALUE",},
            {"sliced_v0/.ATTRIBUTES/VARIABLE_VALUE",},
            {"sliced_v0/.ATTRIBUTES/VARIABLE_VALUE",
             "_CHECKPOINTABLE_OBJECT_GRAPH",},
        ])

    slice_spec = V0SaveSliceInfo(var_offset=[0, 1], var_shape=[2, 1]).spec
    self.assertAllEqual(
        self.evaluate(shards[0][sliced_v0_name][slice_spec]), [[1.0], [6.0]])

    slice_spec = V0SaveSliceInfo(var_offset=[0, 2], var_shape=[1, 2]).spec
    self.assertAllEqual(
        self.evaluate(shards[1][sliced_v0_name][slice_spec]), [[2.0, 3.0]])

    slice_spec = V0SaveSliceInfo(var_offset=[1, 2], var_shape=[1, 2]).spec
    self.assertAllEqual(
        self.evaluate(shards[2][sliced_v0_name][slice_spec]), [[7.0, 8.0]])

    # max_shard_size: 12 bytes
    # Each element of v0 is a 32 bit/4 byte value, so v0 should be split into 2
    # shards containing 3 elements each.
    callback = sharding_policies.MaxShardSizePolicy(max_shard_size=12)
    shards = []
    for tensors in shardable_tensors:
      shards.extend(callback(tensors))

    self.assertEqual(
        [set(shard.keys()) for shard in shards],
        [
            {"sliced_v0/.ATTRIBUTES/VARIABLE_VALUE",},
            {"sliced_v0/.ATTRIBUTES/VARIABLE_VALUE",
             "_CHECKPOINTABLE_OBJECT_GRAPH",},
        ])

    slice_spec = V0SaveSliceInfo(var_offset=[0, 1], var_shape=[1, 3]).spec
    self.assertAllEqual(
        self.evaluate(shards[0][sliced_v0_name][slice_spec]), [[1.0, 2.0, 3.0]])

    slice_spec = V0SaveSliceInfo(var_offset=[1, 1], var_shape=[1, 3]).spec
    self.assertAllEqual(
        self.evaluate(shards[1][sliced_v0_name][slice_spec]), [[6.0, 7.0, 8.0]])

  def test_MaxShardSizePolicy_TFFunction(self):
    v_string = "".join(random.choices(
        string.ascii_uppercase + string.digits, k=10)).encode("utf-8")
    root = module.Module()
    with ops.device("cpu:0"):
      v0 = resource_variable_ops.ResourceVariable(
          v_string, name="v0", dtype=dtypes.string)
    self.evaluate(v0.initializer)
    root.v0 = v0

    shardable_tensors = self._get_shardable_tensors_by_task(root)

    @def_function.function
    def wrapped_policy(shardable_tensors):
      callback = sharding_policies.MaxShardSizePolicy(max_shard_size=4)
      shards = []
      for tensors in shardable_tensors:
        shards.extend(callback(tensors))
      return shards

    # TODO(b/326287351): Get string tensor size in tf.function.
    # This test case should be changed when the bug is fixed/warning removed.
    with self.assertLogs(level="WARNING") as log_output:
      log_level = logging.get_verbosity()
      logging.set_verbosity(logging.WARNING)
      try:
        wrapped_policy(shardable_tensors)
      finally:
        logging.set_verbosity(log_level)

    output = log_output[0][0].message
    self.assertTrue(
        re.search("sharding policy is being executed in a tf.function", output))


if __name__ == "__main__":
  ops.enable_eager_execution()
  test.main()
