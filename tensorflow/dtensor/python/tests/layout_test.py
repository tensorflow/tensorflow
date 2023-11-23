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

from tensorflow.dtensor.python import api
from tensorflow.dtensor.python import layout
from tensorflow.dtensor.python.tests import test_util
from tensorflow.python.eager import backprop
from tensorflow.python.eager.polymorphic_function import polymorphic_function
from tensorflow.python.framework import combinations
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import stateless_random_ops
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
    self.assertIn('batch=4,x=2', repr(mesh))
    self.assertIn('CPU:0', repr(mesh))
    self.assertIn('CPU:7', repr(mesh))

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

  def test_host_mesh(self):
    device_ids = test_util.create_device_ids_array((4, 2))
    mesh = layout.Mesh(
        [_MESH_DIM_BATCH, _MESH_DIM_X],
        device_ids,
        np.ravel(device_ids).tolist(),
        test_util.create_device_list((4, 2), 'GPU'),
        mesh_name='name_not_preserved',
    )
    expected = layout.Mesh(
        [_MESH_DIM_BATCH, _MESH_DIM_X],
        device_ids,
        np.ravel(device_ids).tolist(),
        test_util.create_device_list((4, 2), 'CPU'),
    )
    host_mesh = mesh.host_mesh()
    self.assertEqual(host_mesh, expected)


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
    self.assertIn('batch,unsharded', repr(tensor_layout))

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

  def test_global_shape_from_local_shape(self):
    tensor_layout = layout.Layout(
        [_MESH_DIM_BATCH, _MESH_DIM_X, layout.UNSHARDED],
        mesh=_2D_MESH,
    )
    self.assertEqual(
        tensor_layout.global_shape_from_local_shape(
            tensor_shape.TensorShape((1, 3, 5))
        ),
        (2, 6, 5),
    )

  def test_local_shape_from_global_shape(self):
    tensor_layout = layout.Layout(
        [_MESH_DIM_BATCH, _MESH_DIM_X, layout.UNSHARDED],
        mesh=_2D_MESH,
    )
    self.assertEqual(
        tensor_layout.local_shape_from_global_shape(
            tensor_shape.TensorShape((2, 6, 5))
        ),
        (1, 3, 5),
    )

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

  def test_parted_layout(self):
    tensor_layout = layout.Layout.batch_sharded(
        _2D_MESH, _MESH_DIM_BATCH, rank=2
    )
    parted_layout = tensor_layout.to_parted()
    self.assertEqual(parted_layout.type, layout.LayoutType.PARTED)


class RelayoutTest(test_util.DTensorBaseTest):

  def setUp(self):
    super().setUp()
    global_ids = test_util.create_device_ids_array((2, 2))
    local_ids = np.ravel(global_ids).tolist()
    mesh_dict = {  # pylint: disable=g-complex-comprehension
        device: layout.Mesh(
            [_MESH_DIM_X, _MESH_DIM_Y],
            global_ids,
            local_ids,
            test_util.create_device_list((2, 2), device),
        )
        for device in ('CPU', 'GPU', 'TPU')
    }
    self.mesh = self.configTestMesh(mesh_dict)
    # 1D Layouts
    self.x_layout = layout.Layout.batch_sharded(self.mesh, _MESH_DIM_X, rank=1)
    self.y_layout = layout.Layout.batch_sharded(self.mesh, _MESH_DIM_Y, rank=1)
    # 2D Layouts
    self.unsharded_unsharded_layout = layout.Layout.replicated(
        self.mesh, rank=2
    )
    self.x_unsharded_layout = layout.Layout.batch_sharded(
        self.mesh, _MESH_DIM_X, rank=2
    )
    self.unsharded_x_layout = layout.Layout.inner_sharded(
        self.mesh, _MESH_DIM_X, rank=2
    )

  @combinations.generate(
      combinations.combine(is_graph=[False, True], is_replicated=[False, True])
  )
  def test_relayout(self, is_graph, is_replicated):
    inp = stateless_random_ops.stateless_random_uniform([4, 4], seed=[0, 1])
    if is_replicated:
      to_layout = self.unsharded_unsharded_layout
    else:
      to_layout = self.x_unsharded_layout

    def do_relayout():
      return api.relayout(inp, to_layout)

    if is_graph:
      relayout_fn = polymorphic_function.function(do_relayout)
      self.assertRaisesRegex(
          errors_impl.InvalidArgumentError,
          "No OpKernel was registered to support Op 'Relayout'",
          relayout_fn,
      )
    else:
      self.assertDTensorEqual(inp, to_layout, do_relayout())

  @combinations.generate(combinations.combine(is_graph=[False, True]))
  def test_relayout_to_parted(self, is_graph):
    data = np.array([1, 2, 3, 4.0], dtype='f4')
    inp = api.relayout(data, self.y_layout)

    def do_relayout():
      return api.relayout(inp, self.y_layout.to_parted())

    if is_graph:
      do_relayout = polymorphic_function.function(do_relayout)

    with api.default_mesh(self.mesh):
      result = do_relayout()

    self.assertDTensorEqual(data, self.y_layout.to_parted(), result)

  @combinations.generate(combinations.combine(is_graph=[False, True]))
  def test_relayout_like_simple(self, is_graph):
    data = np.array([1, 2, 3, 4.0], dtype='f4')
    inp = api.relayout(data, self.y_layout)
    inp_layout = api.relayout(data, self.x_layout)

    def do_relayout():
      return api.relayout_like(inp, inp_layout)

    if is_graph:
      do_relayout = polymorphic_function.function(do_relayout)

    with api.default_mesh(self.mesh):
      result = do_relayout()

    self.assertDTensorEqual(data, self.x_layout, result)

  def test_relayout_like_init_scope(self):
    data = np.array([1, 2, 3, 4.0], dtype='f4')
    inp = api.relayout(data, self.y_layout)
    inp_layout = api.relayout(data, self.x_layout)

    @polymorphic_function.function
    def do_relayout(x):
      with ops.init_scope():
        return api.relayout_like(inp, x)

    with api.default_mesh(self.mesh):
      with self.assertRaisesRegex(
          TypeError, 'is out of scope and cannot be used here'
      ):
        result = do_relayout(inp_layout)
        result.numpy()

  def test_nested_relayout_gradient_preserves_layout(self):
    # Test that nesting gradient tapes with relayouts preserves the layout of
    # the original DTensor input. The second-order gradient should have a layout
    # equivalent to the original input, even if the inner gradient tape
    # relayouts the DTensor to a different layout.

    @polymorphic_function.function
    def inner(x):
      with backprop.GradientTape() as tape:
        tape.watch(x)
        t = x * 1.0
        t = api.relayout(t, self.unsharded_x_layout)
        cube = t * t * t
      grad = tape.gradient(cube, x)
      return grad

    @polymorphic_function.function
    def outer(x):
      with backprop.GradientTape() as tape:
        tape.watch(x)
        t = api.relayout(x, self.x_unsharded_layout)
        grad = inner(t)
        out = grad + t
      out_grad = tape.gradient(out, x)
      return out_grad

    a = stateless_random_ops.stateless_random_uniform([8, 8], seed=[0, 1])
    a_dt = api.relayout(a, self.unsharded_unsharded_layout)

    with ops.device_v2(api.device_name()):
      inner_grad = inner(a_dt)
      outer_grad = outer(a_dt)

    self.assertDTensorEqual(
        3 * a * a, self.unsharded_unsharded_layout, inner_grad
    )
    self.assertDTensorEqual(
        6 * a + 1, self.unsharded_unsharded_layout, outer_grad
    )

  def test_wus_using_relayout(self):
    sharded_layout = layout.Layout.batch_sharded(self.mesh, _MESH_DIM_X, rank=2)
    w = stateless_random_ops.stateless_random_uniform(
        [4, 4], seed=[0, 1], dtype=dtypes.float32
    )
    sharded_w = api.relayout(w, sharded_layout)
    replicated_layout = layout.Layout(
        [layout.UNSHARDED, layout.UNSHARDED], mesh=self.mesh
    )

    @polymorphic_function.function
    def func_with_relayout(t):
      with backprop.GradientTape() as tape:
        tape.watch(t)
        t = t + t
        out = api.relayout(t, replicated_layout)
        loss = math_ops.reduce_sum(out)
      grad = tape.gradient(loss, t)
      t = t - grad
      return t

    func_with_relayout(sharded_w)

  @combinations.generate(
      combinations.combine(size=[16, 4096], is_graph=[False, True])
  )
  def test_call_with_layout(self, size, is_graph):
    layout_x = layout.Layout.batch_sharded(
        self.mesh, batch_dim=_MESH_DIM_X, rank=1
    )
    layout_y = layout.Layout.batch_sharded(
        self.mesh, batch_dim=_MESH_DIM_Y, rank=1
    )

    expected = array_ops.zeros(shape=[size])

    def func():
      tensor_x = api.call_with_layout(array_ops.zeros, layout_x, shape=[size])
      tensor_y = api.call_with_layout(array_ops.zeros, layout_y, shape=[size])
      return tensor_x, tensor_y

    if is_graph:
      func = polymorphic_function.function(func)

    with api.default_mesh(self.mesh):
      tensor_x, tensor_y = func()

    self.assertDTensorEqual(expected, layout_x, tensor_x)
    self.assertDTensorEqual(expected, layout_y, tensor_y)


if __name__ == '__main__':
  test.main()
