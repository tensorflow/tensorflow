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
import gc

from absl.testing import parameterized

import numpy as np

# pylint: disable=g-direct-tensorflow-import
from tensorflow.dtensor.python import api
from tensorflow.dtensor.python import d_variable
from tensorflow.dtensor.python import layout as layout_lib
from tensorflow.dtensor.python import numpy_util
from tensorflow.dtensor.python.tests import test_util
from tensorflow.python.checkpoint import checkpoint
from tensorflow.python.checkpoint import checkpoint_management
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.module import module
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import stateless_random_ops
from tensorflow.python.platform import test

Mesh = layout_lib.Mesh
Layout = layout_lib.Layout
UNSHARDED = layout_lib.UNSHARDED

# Makes a 2D mesh with dimension X(2) and dimension Y(4).
_MESH_DIM_X = 'x'
_MESH_DIM_Y = 'y'
_DEVICE_IDS = test_util.create_device_ids_array((2, 4))
_TWO_D_CPU_MESH = Mesh(
    [_MESH_DIM_X, _MESH_DIM_Y],
    _DEVICE_IDS,
    np.ravel(_DEVICE_IDS).tolist(),
    test_util.create_device_list((2, 4), 'CPU'),
)
_TWO_D_TPU_MESH = Mesh(
    [_MESH_DIM_X, _MESH_DIM_Y],
    _DEVICE_IDS,
    np.ravel(_DEVICE_IDS).tolist(),
    test_util.create_device_list((2, 4), 'TPU'),
)
_TWO_D_GPU_MESH = Mesh(
    [_MESH_DIM_X, _MESH_DIM_Y],
    _DEVICE_IDS,
    np.ravel(_DEVICE_IDS).tolist(),
    test_util.create_device_list((2, 4), 'GPU'),
)


class DTensorSaveRestoreV2Test(test_util.DTensorBaseTest):

  def setUp(self):
    super(DTensorSaveRestoreV2Test, self).setUp()
    self.skipForDeviceType(['TPU'],
                           'all tests require 8 TPU cores.',
                           unless_device_count_equals_to=8)
    mesh_dict = {
        'CPU': _TWO_D_CPU_MESH,
        'GPU': _TWO_D_GPU_MESH,
        'TPU': _TWO_D_TPU_MESH,
    }
    self.mesh = self.configTestMesh(mesh_dict)
    self.skipForTfrt(
        'b/235088250, DTensorCheckpointingV2 requires upcasting TF scalar '
        'variables to replicated DTensor scalar variables, which is not '
        'supported in TFRT.')

  @parameterized.named_parameters(
      ('x_unsharded', [_MESH_DIM_X, UNSHARDED]),
      ('unsharded_x', [UNSHARDED, _MESH_DIM_X]),
      ('x_y', [_MESH_DIM_X, _MESH_DIM_Y]),
      ('unsharded_unsharded', [UNSHARDED, UNSHARDED]),
  )
  def test_checkpoint_simple(self, shard_spec):
    tensor_a = stateless_random_ops.stateless_random_uniform(
        shape=[4, 8], seed=[0, 1]
    )
    tensor_b = stateless_random_ops.stateless_random_uniform(
        shape=[2, 4], seed=[0, 1]
    )

    layout = Layout(shard_spec, self.mesh)

    dvariable_a = d_variable.DVariable(numpy_util.pack_numpy(tensor_a, layout))
    dvariable_b = d_variable.DVariable(numpy_util.pack_numpy(tensor_b, layout))

    # Record a checkpoint with two dvariables.
    ckpt = checkpoint.Checkpoint(a=dvariable_a, b=dvariable_b)

    saved_path = ckpt.save(self.get_temp_dir())

    # Zero out the values of the DVariables so that we can restore
    # and check that the values are restored to the initial random values.
    dvariable_a.assign(
        numpy_util.pack_numpy(
            array_ops.zeros([4, 8], dtype=dtypes.float32), layout
        )
    )
    dvariable_b.assign(
        numpy_util.pack_numpy(
            array_ops.zeros([2, 4], dtype=dtypes.float32), layout
        )
    )

    ckpt.restore(saved_path)

    self.assertDTensorEqual(tensor_a, layout, dvariable_a.read_value())
    self.assertDTensorEqual(tensor_b, layout, dvariable_b.read_value())

  @parameterized.named_parameters(
      ('x_unsharded', [_MESH_DIM_X, UNSHARDED]),
      ('unsharded_x', [UNSHARDED, _MESH_DIM_X]),
      ('x_y', [_MESH_DIM_X, _MESH_DIM_Y]),
      ('unsharded_unsharded', [UNSHARDED, UNSHARDED]),
  )
  def test_checkpoint_write(self, shard_spec):
    tensor_a = stateless_random_ops.stateless_random_uniform(
        shape=[4, 8], seed=[0, 1]
    )
    tensor_b = stateless_random_ops.stateless_random_uniform(
        shape=[2, 4], seed=[0, 1]
    )

    layout = Layout(shard_spec, self.mesh)

    dvariable_a = d_variable.DVariable(numpy_util.pack_numpy(tensor_a, layout))
    dvariable_b = d_variable.DVariable(numpy_util.pack_numpy(tensor_b, layout))

    ckpt = checkpoint.Checkpoint(a=dvariable_a, b=dvariable_b)

    saved_path = ckpt.write(self.get_temp_dir())

    dvariable_a.assign(
        numpy_util.pack_numpy(
            array_ops.zeros([4, 8], dtype=dtypes.float32), layout
        )
    )
    dvariable_b.assign(
        numpy_util.pack_numpy(
            array_ops.zeros([2, 4], dtype=dtypes.float32), layout
        )
    )

    ckpt.restore(saved_path)

    self.assertDTensorEqual(tensor_a, layout, dvariable_a.read_value())
    self.assertDTensorEqual(tensor_b, layout, dvariable_b.read_value())

  @parameterized.named_parameters(
      ('x_unsharded', [_MESH_DIM_X, UNSHARDED]),
      ('unsharded_x', [UNSHARDED, _MESH_DIM_X]),
      ('x_y', [_MESH_DIM_X, _MESH_DIM_Y]),
      ('unsharded_unsharded', [UNSHARDED, UNSHARDED]),
  )
  def test_checkpoint_manager(self, shard_spec):
    tensor_a = stateless_random_ops.stateless_random_uniform(
        shape=[8, 16], seed=[0, 1]
    )
    tensor_b = stateless_random_ops.stateless_random_uniform(
        shape=[4, 4], seed=[0, 1]
    )

    layout = Layout(shard_spec, self.mesh)

    dvariable_a = d_variable.DVariable(numpy_util.pack_numpy(tensor_a, layout))
    dvariable_b = d_variable.DVariable(numpy_util.pack_numpy(tensor_b, layout))

    # Record a checkpoint with two dvariables.
    ckpt = checkpoint.Checkpoint(a=dvariable_a, b=dvariable_b)

    checkpoint_manager = checkpoint_management.CheckpointManager(
        ckpt, self.get_temp_dir(), max_to_keep=None
    )

    saved_path = checkpoint_manager.save()

    # Zero out the values of the DVariables so that we can restore
    # and check that the values are restored to the initial random values.
    dvariable_a.assign(
        numpy_util.pack_numpy(
            array_ops.zeros([8, 16], dtype=dtypes.float32), layout
        )
    )
    dvariable_b.assign(
        numpy_util.pack_numpy(
            array_ops.zeros([4, 4], dtype=dtypes.float32), layout
        )
    )

    ckpt.restore(saved_path)

    self.assertDTensorEqual(tensor_a, layout, dvariable_a.read_value())
    self.assertDTensorEqual(tensor_b, layout, dvariable_b.read_value())

  @parameterized.named_parameters(
      ('x_unsharded', [_MESH_DIM_X, UNSHARDED]),
      ('unsharded_x', [UNSHARDED, _MESH_DIM_X]),
      ('x_y', [_MESH_DIM_X, _MESH_DIM_Y]),
      ('unsharded_unsharded', [UNSHARDED, UNSHARDED]),
  )
  def test_checkpoint_restore_with_different_layout(self, shard_spec):
    tensor_a = stateless_random_ops.stateless_random_uniform(
        shape=[4, 8], seed=[0, 1]
    )
    tensor_b = stateless_random_ops.stateless_random_uniform(
        shape=[2, 4], seed=[0, 1]
    )

    layout = Layout(shard_spec, self.mesh)

    dvariable_a = d_variable.DVariable(numpy_util.pack_numpy(tensor_a, layout))
    dvariable_b = d_variable.DVariable(numpy_util.pack_numpy(tensor_b, layout))

    # Record a checkpoint with two dvariables.
    checkpoint_1 = checkpoint.Checkpoint(a=dvariable_a, b=dvariable_b)

    saved_path = checkpoint_1.save(self.get_temp_dir())

    new_layout = Layout([_MESH_DIM_X, _MESH_DIM_Y], self.mesh)

    # Create new Dvariables, zero'd out with different layouts
    # from the layouts we saved the tensors.
    dvariable_a = d_variable.DVariable(
        numpy_util.pack_numpy(
            array_ops.zeros([4, 8], dtype=dtypes.float32), new_layout
        )
    )
    dvariable_b = d_variable.DVariable(
        numpy_util.pack_numpy(
            array_ops.zeros([2, 4], dtype=dtypes.float32), new_layout
        )
    )

    checkpoint_2 = checkpoint.Checkpoint(a=dvariable_a, b=dvariable_b)

    checkpoint_2.restore(saved_path)

    self.assertDTensorEqual(tensor_a, new_layout, dvariable_a.read_value())
    self.assertDTensorEqual(tensor_b, new_layout, dvariable_b.read_value())

  @parameterized.named_parameters(
      ('x_unsharded', [_MESH_DIM_X, UNSHARDED]),
      ('unsharded_x', [UNSHARDED, _MESH_DIM_X]),
  )
  def test_checkpoint_in_a_train_loop(self, shard_dims):
    # This test is a parallel test with save_restore_test's
    # DTensorSaveRestoreTest.test_checkpoint

    class M(module.Module):

      # Pass in both replicated and sharded for better coverage.
      def __init__(self, replicated_value, sharded_value):
        # This is actually a DVariable.
        self.r = d_variable.DVariable(replicated_value)
        self.s = d_variable.DVariable(sharded_value)

      def __call__(self, x):
        return math_ops.reduce_sum(x + self.r) + math_ops.reduce_sum(x + self.s)

    directory = self.get_temp_dir()

    sharded_np = np.arange(8).reshape((2, 4)).astype(np.float32)
    replicated_np = np.arange(16).reshape((8, 2)).astype(np.float32)

    replicated_layout = Layout.replicated(self.mesh, rank=2)
    one_d_sharded_layout = Layout(shard_dims, self.mesh)

    replicated_value = api.copy_to_mesh(replicated_np, replicated_layout)
    replicated_zeros = api.copy_to_mesh(
        np.zeros((8, 2)).astype(np.float32), replicated_layout
    )

    sharded_value = numpy_util.pack_numpy(sharded_np, one_d_sharded_layout)
    sharded_zeros = numpy_util.pack_numpy(
        np.zeros((2, 4)).astype(np.float32), one_d_sharded_layout)

    # Training loop that just increments the model's variable every "epoch"
    # to test checkpointing.
    for epoch in range(5):
      m = M(replicated_value, sharded_value)

      ckpt = checkpoint.Checkpoint(model=m)
      manager = checkpoint_management.CheckpointManager(
          ckpt, directory=directory, max_to_keep=None
      )

      ckpt.restore(manager.latest_checkpoint)

      # Ensure that the variable is created
      m(api.copy_to_mesh(1.0, Layout.replicated(self.mesh, rank=0)))

      self.assertDTensorEqual(epoch + replicated_np, replicated_layout, m.r)
      self.assertDTensorEqual(epoch + sharded_np, one_d_sharded_layout, m.s)

      m.s.assign_add(
          numpy_util.pack_numpy(
              np.ones((2, 4), dtype=np.float32), one_d_sharded_layout))
      m.r.assign_add(
          api.copy_to_mesh(
              constant_op.constant(np.ones((8, 2), dtype=np.float32)),
              replicated_layout,
          )
      )

      checkpoint_number = epoch + 1

      stats1 = api._dtensor_device()._get_stats()
      manager.save(checkpoint_number=checkpoint_number)

      gc.collect()
      stats2 = api._dtensor_device()._get_stats()
      keys = set(stats2.keys())
      keys.update(stats1.keys())
      diff = {k: stats2.get(k, 0) - stats1.get(k, 0) for k in keys}
      diff = {k: v for k, v in diff.items() if v != 0}

      m.s.assign(sharded_zeros)
      m.r.assign(replicated_zeros)


if __name__ == '__main__':
  test.main()
