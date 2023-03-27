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
"""Tests for layout.py."""

import copy
import itertools
import pickle

from absl.testing import parameterized
import numpy as np

from tensorflow.dtensor.python import layout
from tensorflow.dtensor.python.tests import test_util
from tensorflow.python.platform import test

UNSHARDED = layout.UNSHARDED

# Convenient constants to use for tests.
_MESH_DIM_BATCH = 'batch'
_MESH_DIM_X = 'x'
_MESH_DIM_Y = 'y'
_MESH_2D_STRING = (
    '|batch=2,x=2|0,1,2,3|0,1,2,3|'
    '/job:localhost/replica:0/task:0/device:TPU:0,'
    '/job:localhost/replica:0/task:0/device:TPU:1,'
    '/job:localhost/replica:0/task:0/device:TPU:2,'
    '/job:localhost/replica:0/task:0/device:TPU:3'
)

_2D_GLOBAL_IDS = test_util.create_device_ids_array((2, 2))

_2D_MESH = layout.Mesh([_MESH_DIM_BATCH, _MESH_DIM_X], _2D_GLOBAL_IDS,
                       np.ravel(_2D_GLOBAL_IDS).tolist(),
                       test_util.create_device_list((2, 2), 'TPU'))
_2D_X_Y_MESH = layout.Mesh([_MESH_DIM_X, _MESH_DIM_Y], _2D_GLOBAL_IDS,
                           np.ravel(_2D_GLOBAL_IDS).tolist(),
                           test_util.create_device_list((2, 2), 'CPU'))

_SINGLE_DEVICE_MESH = layout.Mesh.from_device(
    'job:localhost/replica:0/task:0/TPU:0'
)


class MeshTest(test_util.DTensorBaseTest, parameterized.TestCase):

  def test_mesh_single_device(self):
    self.assertTrue(_SINGLE_DEVICE_MESH.is_single_device())

  def test_mesh_single_device_to_string(self):
    roundtrip = layout.Mesh.from_string(_SINGLE_DEVICE_MESH.to_string())
    self.assertTrue(roundtrip.is_single_device())
    self.assertEqual(roundtrip.single_device, _SINGLE_DEVICE_MESH.single_device)

  def test_mesh_single_device_to_proto(self):
    roundtrip = layout.Mesh.from_proto(_SINGLE_DEVICE_MESH.as_proto())
    self.assertTrue(roundtrip.is_single_device())
    self.assertEqual(roundtrip.single_device, _SINGLE_DEVICE_MESH.single_device)

  def test_mesh_reciprocal_mock_string_and_object(self):
    generated_mesh_from_string = layout.Mesh.from_string(_MESH_2D_STRING)
    self.assertProtoEquals(_2D_MESH.as_proto(),
                           generated_mesh_from_string.as_proto())

  def test_mesh_reciprocal_string_rep(self):
    new_mesh_str = layout.Mesh.from_string(_MESH_2D_STRING).to_string()
    self.assertEqual(_MESH_2D_STRING, new_mesh_str)

  def test_mesh_repr(self):
    device_ids = test_util.create_device_ids_array((4, 2))
    mesh = layout.Mesh([_MESH_DIM_BATCH, _MESH_DIM_X], device_ids,
                       np.ravel(device_ids).tolist(),
                       test_util.create_device_list((4, 2), 'CPU'))
    self.assertIn(
        '<Mesh object with dims=[(\'batch\', 4), (\'x\', 2)], '
        'device_type="CPU", num_local_devices=8), size=8', repr(mesh))

  def test_mesh_contains_dim(self):
    self.assertTrue(_2D_MESH.contains_dim('batch'))
    self.assertTrue(_2D_MESH.contains_dim('x'))
    self.assertFalse(_2D_MESH.contains_dim('y'))

  def test_mesh_contains(self):
    self.assertIn('batch', _2D_MESH)
    self.assertIn('x', _2D_MESH)
    self.assertNotIn('y', _2D_MESH)

  def test_mesh_dim_names_property(self):
    self.assertSequenceEqual(_2D_MESH.dim_names, ['batch', 'x'])

  def test_mesh_size_property(self):
    self.assertEqual(_2D_MESH.size, 4)

  def test_mesh_device_type(self):
    self.assertEqual(_2D_MESH.device_type(), 'TPU')
    self.assertEqual(_2D_X_Y_MESH.device_type(), 'CPU')

  def test_mesh_num_local_devices(self):
    self.assertEqual(_2D_MESH.num_local_devices(), 4)

  def test_mesh_min_global_device_id(self):
    self.assertEqual(_2D_MESH.min_global_device_id(), 0)

  def test_mesh_is_remote(self):
    self.assertFalse(_2D_MESH.is_remote())

  def test_mesh_local_device_ids(self):
    self.assertSequenceEqual(_2D_MESH.local_device_ids(), [0, 1, 2, 3])

  def test_mesh_local_devices(self):
    self.assertSequenceEqual(_2D_MESH.local_devices(), [
        '/job:localhost/replica:0/task:0/device:TPU:0',
        '/job:localhost/replica:0/task:0/device:TPU:1',
        '/job:localhost/replica:0/task:0/device:TPU:2',
        '/job:localhost/replica:0/task:0/device:TPU:3'
    ])

  def test_mesh_shape(self):
    self.assertSequenceEqual(_2D_MESH.shape(), [2, 2])

  def test_mesh_pickle(self):
    pickled = pickle.dumps(_2D_MESH)
    unpickled = pickle.loads(pickled)

    self.assertEqual(_2D_MESH, unpickled)

  def test_mesh_pickle_w_modification_after_init(self):
    mesh = copy.copy(_2D_MESH)
    mesh._name = 'fake_name'
    pickled = pickle.dumps(mesh)
    unpickled = pickle.loads(pickled)

    self.assertEqual(mesh, unpickled)

  def test_mesh_dims(self):
    device_ids = test_util.create_device_ids_array((4, 2))
    mesh = layout.Mesh(
        [_MESH_DIM_BATCH, _MESH_DIM_X],
        device_ids,
        np.ravel(device_ids).tolist(),
        test_util.create_device_list((4, 2), 'CPU'))

    self.assertIn(_MESH_DIM_BATCH, mesh)
    self.assertIn(_MESH_DIM_X, mesh)
    self.assertNotIn(_MESH_DIM_Y, mesh)

    self.assertEqual(mesh[_MESH_DIM_BATCH].name, _MESH_DIM_BATCH)
    self.assertEqual(mesh[_MESH_DIM_BATCH].size, 4)
    self.assertEqual(mesh[_MESH_DIM_X].name, _MESH_DIM_X)
    self.assertEqual(mesh[_MESH_DIM_X].size, 2)

  @parameterized.parameters(
      {
          'mesh_dims': ('a',),
          'mesh_shape': (8,),
          'strides': [1],
      }, {
          'mesh_dims': ('a', 'b', 'c'),
          'mesh_shape': (2, 4, 2),
          'strides': [8, 2, 1],
      }, {
          'mesh_dims': ('a', 'b', 'c', 'd'),
          'mesh_shape': (8, 16, 2, 4),
          'strides': [128, 8, 4, 1],
      })
  def test_mesh_strides(self, mesh_dims, mesh_shape, strides):
    device_ids = test_util.create_device_ids_array(mesh_shape)
    mesh = layout.Mesh(
        dim_names=list(mesh_dims),
        global_device_ids=device_ids,
        local_device_ids=np.ravel(device_ids).tolist(),
        local_devices=test_util.create_device_list(mesh_shape, 'CPU'))

    self.assertEqual(mesh.strides, strides)

  def test_mesh_coords(self):
    mesh_shape = (2, 4, 2)
    device_ids = test_util.create_device_ids_array(mesh_shape)
    mesh = layout.Mesh(
        dim_names=['a', 'b', 'c'],
        global_device_ids=device_ids,
        local_device_ids=np.ravel(device_ids).tolist(),
        local_devices=test_util.create_device_list(mesh_shape, 'CPU'))

    coords = itertools.product(range(2), range(4), range(2))
    # Repeat coords to overflow idx, mesh coords should still be correct.
    coords = itertools.chain.from_iterable(
        itertools.repeat(tuple(coords), times=3))
    for idx, (a, b, c) in enumerate(coords):
      self.assertAllEqual(mesh.coords(idx), [a, b, c])

  @parameterized.named_parameters(('use_xla_spmd', True),
                                  ('do_not_use_xla_spmd', False))
  def test_mesh_use_xla_spmd_tpu_mesh(self, use_xla_spmd):
    mesh_shape = (2, 4, 2)
    device_ids = test_util.create_device_ids_array(mesh_shape)
    mesh = layout.Mesh(
        dim_names=['a', 'b', 'c'],
        global_device_ids=device_ids,
        local_device_ids=np.ravel(device_ids).tolist(),
        local_devices=test_util.create_device_list(mesh_shape, 'TPU'),
        use_xla_spmd=use_xla_spmd)
    self.assertEqual(use_xla_spmd, mesh.use_xla_spmd())

  @parameterized.named_parameters(('gpu', 'GPU'), ('cpu', 'CPU'))
  def test_mesh_use_xla_spmd_for_non_tpu_mesh_raises_error(self, mesh_type):
    mesh_shape = (2, 4, 2)
    device_ids = test_util.create_device_ids_array(mesh_shape)
    with self.assertRaisesRegex(ValueError,
                                'XLA SPMD is not currently not supported for'):
      layout.Mesh(
          dim_names=['a', 'b', 'c'],
          global_device_ids=device_ids,
          local_device_ids=np.ravel(device_ids).tolist(),
          local_devices=test_util.create_device_list(mesh_shape, mesh_type),
          use_xla_spmd=True)

  @parameterized.named_parameters(
      ('use_xla_spmd', True), ('do_not_use_xla_spmd', False)
  )
  def test_mesh_as_proto_use_xla_spmd(self, use_xla_spmd):
    mesh_shape = (2, 4, 2)
    device_ids = test_util.create_device_ids_array(mesh_shape)
    mesh = layout.Mesh(
        dim_names=['a', 'b', 'c'],
        global_device_ids=device_ids,
        local_device_ids=np.ravel(device_ids).tolist(),
        local_devices=test_util.create_device_list(mesh_shape, 'TPU'),
        use_xla_spmd=use_xla_spmd,
    )
    mesh_proto = mesh.as_proto()

    self.assertEqual(mesh_proto.use_xla_spmd, mesh.use_xla_spmd())

  def test_mesh_from_string_with_use_xla_spmd(self):
    mesh_str_without_global_device_ids = (
        '|batch=2|0,1|0,1|/job:localhost/replica:0/task:0'
        '/device:TPU:0,/job:localhost/replica:0/task:0/device:TPU:1|use_xla_spmd'
    )
    mesh = layout.Mesh.from_string(mesh_str_without_global_device_ids)
    self.assertTrue(mesh.use_xla_spmd())

  def test_mesh_from_string_with_use_xla_spmd_and_global_devices(self):
    mesh_str_with_global_device_ids = (
        '|batch=2|0,1|0|/job:localhost/replica:0/task:0'
        '/device:TPU:0|/job:localhost/replica:0/task:0/device:TPU:0,'
        '/job:localhost/replica:0/task:0/device:TPU:1|use_xla_spmd'
    )
    mesh = layout.Mesh.from_string(mesh_str_with_global_device_ids)
    self.assertTrue(mesh.use_xla_spmd())

  def test_non_unique_device_type(self):
    a = test_util.create_device_array((2,), 'CPU')
    b = test_util.create_device_array((2,), 'TPU')
    c = np.vstack([a, b])
    global_ids = test_util.create_device_ids_array(c.shape)
    local_ids = np.ravel(global_ids).tolist()
    with self.assertRaisesRegex(ValueError,
                                'Devices containing multiple device_types'):
      layout.Mesh([_MESH_DIM_BATCH, _MESH_DIM_X], global_ids, local_ids,
                  np.ravel(c).tolist())

  def test_duplicated_devices(self):
    a = test_util.create_device_array((2,), 'CPU')
    b = test_util.create_device_array((2,), 'CPU')
    c = np.vstack([a, b])
    global_ids = test_util.create_device_ids_array((2, 2))
    local_ids = global_ids.flatten().tolist()
    with self.assertRaisesRegex(
        ValueError, 'Duplicate devices found in mesh specification'):
      layout.Mesh([_MESH_DIM_BATCH, _MESH_DIM_X], global_ids, local_ids,
                  np.ravel(c).tolist())

  def test_inconsecutive_device_ids(self):
    a = test_util.create_device_array((2,), 'CPU')
    global_ids = test_util.create_device_ids_array((2))
    global_ids = np.flip(global_ids)
    local_ids = global_ids.flatten().tolist()
    with self.assertRaisesRegex(ValueError,
                                'global_device_ids must sequentially increase'):
      layout.Mesh([_MESH_DIM_BATCH], global_ids, local_ids,
                  np.ravel(a).tolist())


class LayoutTest(test_util.DTensorBaseTest, parameterized.TestCase):

  def test_empty_sharding_spec_different_from_single_unsharded(self):
    layout_str_single_unsharded = (
        'sharding_specs:unsharded, mesh:' + _MESH_2D_STRING
    )
    layout_str_empty_sharding_spec = 'sharding_specs: mesh:' + _MESH_2D_STRING

    self.assertNotEqual(
        layout.Layout.from_string(layout_str_single_unsharded).to_string(),
        layout.Layout.from_string(layout_str_empty_sharding_spec).to_string(),
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='sharded_batch_and_x',
          test_layout_str='sharding_specs:batch,x, mesh:' + _MESH_2D_STRING,
      ),
      dict(
          testcase_name='unsharded_explicit',
          test_layout_str='sharding_specs:'
          + UNSHARDED
          + ','
          + UNSHARDED
          + ','
          + ' mesh:'
          + _MESH_2D_STRING,
      ),
  )
  def test_layout_reciprocal_string_rep(self, test_layout_str):
    new_layout_str = layout.Layout.from_string(test_layout_str).to_string()
    self.assertEqual(test_layout_str, new_layout_str)

  def test_layout_pickle(self):
    replicated = layout.Layout.replicated(_2D_MESH, rank=3)
    pickled = pickle.dumps(replicated)
    unpickled = pickle.loads(pickled)

    self.assertEqual(replicated, unpickled)

  def test_layout_repr(self):
    tensor_layout = layout.Layout.batch_sharded(
        _2D_MESH, _MESH_DIM_BATCH, rank=2)
    self.assertIn('Layout(sharding_specs=[\'batch\', \'unsharded\'], mesh=',
                  repr(tensor_layout))

  def test_throws_for_non_mesh(self):
    with self.assertRaisesRegex(ValueError, 'mesh is not a valid Mesh object'):
      layout.Layout([_MESH_DIM_BATCH, _MESH_DIM_X], 'string_mesh')

  def test_throws_for_repeated_dimension(self):
    with self.assertRaisesRegex(ValueError, 'Mesh dimensions must be unique.'):
      layout.Layout([_MESH_DIM_BATCH, _MESH_DIM_BATCH], _2D_MESH)

  def test_throws_for_invalid_sharding_spec(self):
    with self.assertRaisesRegex(
        ValueError,
        'A dimension sharding must either be a valid mesh dimension or ' +
        'UNSHARDED.'):
      layout.Layout(['WRONG_SHARDING_SPEC', 'UNSHARDED'], _2D_MESH)

  def test_data_parallel_layout(self):
    tensor_layout = layout.Layout.batch_sharded(
        _2D_MESH, _MESH_DIM_BATCH, rank=2)
    self.assertEqual(
        tensor_layout.num_shards(0), _2D_MESH.dim_size(_MESH_DIM_BATCH))
    self.assertEqual(tensor_layout.num_shards(1), 1)

  def test_single_device_layout(self):
    tensor_layout = layout.Layout.from_single_device_mesh(_SINGLE_DEVICE_MESH)
    tensor_layout2 = layout.Layout.from_device(
        _SINGLE_DEVICE_MESH.single_device
    )
    self.assertTrue(tensor_layout.is_single_device())
    self.assertEqual(tensor_layout.mesh, _SINGLE_DEVICE_MESH)
    self.assertEqual(tensor_layout, tensor_layout2)

  def test_single_device_layout_from_string(self):
    tensor_layout = layout.Layout.from_single_device_mesh(_SINGLE_DEVICE_MESH)
    roundtrip = layout.Layout.from_string(tensor_layout.to_string())
    self.assertEqual(roundtrip, tensor_layout)

  def test_single_device_layout_from_proto(self):
    tensor_layout = layout.Layout.from_single_device_mesh(_SINGLE_DEVICE_MESH)
    roundtrip = layout.Layout.from_proto(tensor_layout.as_proto())
    self.assertEqual(roundtrip, tensor_layout)


if __name__ == '__main__':
  test.main()
