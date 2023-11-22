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

from tensorflow.python.checkpoint import checkpoint
from tensorflow.python.checkpoint import graph_view
from tensorflow.python.checkpoint.sharding import sharding_policies
from tensorflow.python.checkpoint.sharding import sharding_util
from tensorflow.python.eager import test
from tensorflow.python.framework import ops
from tensorflow.python.module import module
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.training.saving import saveable_object


class ShardingPoliciesTest(test.TestCase):

  def _get_shardable_tensors(self, root):
    serialized_tensors, _, _, _ = (
        checkpoint.TrackableSaver(graph_view.ObjectGraphView(root))
        ._gather_serialized_tensors(None))

    shardable_tensors = []
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
          shardable_tensors.append(
              sharding_util.ShardableTensor(
                  _tensor_save_spec=tensor_save_spec,
                  tensor=save_spec_tensor,
                  dtype=tensor_save_spec.dtype,
                  device=tensor_save_spec.device,
                  name=tensor_save_spec.name,
                  shape=save_spec_tensor.shape,
                  slice_spec=slice_spec,
                  checkpoint_key=checkpoint_key,
                  trackable=obj))
    return shardable_tensors

  def test_ShardByDevicePolicy(self):
    root = module.Module()
    with ops.device("cpu:0"):
      v0 = resource_variable_ops.ResourceVariable(0.0, name="v0")
    with ops.device("cpu:1"):
      v1 = resource_variable_ops.ResourceVariable(1.0, name="v1")
    with ops.device("cpu:2"):
      v2 = resource_variable_ops.ResourceVariable(2.0, name="v2")
    root.v0 = v0
    root.v1 = v1
    root.v2 = v2

    shardable_tensors = self._get_shardable_tensors(root)

    callback = sharding_policies.ShardByDevicePolicy()
    shards = callback(shardable_tensors)

    self.assertAllEqual(
        [list(shard.keys()) for shard in shards],
        [[
            "v0/.ATTRIBUTES/VARIABLE_VALUE",
            "v1/.ATTRIBUTES/VARIABLE_VALUE",
            "v2/.ATTRIBUTES/VARIABLE_VALUE",
            "_CHECKPOINTABLE_OBJECT_GRAPH"
        ]])

    self.assertEqual(shards[0]["v0/.ATTRIBUTES/VARIABLE_VALUE"][""].numpy(),
                     v0.numpy())
    self.assertEqual(shards[0]["v1/.ATTRIBUTES/VARIABLE_VALUE"][""].numpy(),
                     v1.numpy())
    self.assertEqual(shards[0]["v2/.ATTRIBUTES/VARIABLE_VALUE"][""].numpy(),
                     v2.numpy())


if __name__ == "__main__":
  ops.enable_eager_execution()
  test.main()
