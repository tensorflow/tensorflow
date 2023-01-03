# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Test for MirroredStrategy backed by DTensor API."""

from absl.testing import parameterized
import numpy as np

from tensorflow.dtensor.python import d_variable
from tensorflow.dtensor.python import layout
from tensorflow.dtensor.python import mesh_util
from tensorflow.dtensor.python.tests import test_util
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute.experimental import mirrored_strategy
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import variables


class StrategyBaseTest(test_util.DTensorBaseTest):

  def setUp(self):
    super().setUp()
    global_ids = test_util.create_device_ids_array((2,))
    local_ids = np.ravel(global_ids).tolist()
    mesh_dict = {
        device: layout.Mesh(['batch'], global_ids, local_ids,
                            test_util.create_device_list((2,), device))
        for device in ['TPU', 'GPU', 'CPU']
    }
    self.mesh = self.configTestMesh(mesh_dict)

  @parameterized.named_parameters([
      ('py_floats', lambda: [1.0, 2.0], True),
      ('np_floats', lambda: np.array([1.0, 2.0]), True),
      ('tf_const', lambda: constant_op.constant([1.0, 2.0]), True),
      ('py_floats_callable', lambda: [1.0, 2.0], False),
      ('np_floats_callable', lambda: np.array([1.0, 2.0]), False),
      ('tf_const_callable', lambda: constant_op.constant([1.0, 2.0]), False),
  ])
  def test_variable_creation(self, init_value, convert_callable):
    if convert_callable:
      init_value = init_value()
    strategy = mirrored_strategy.MirroredStrategy(self.mesh)
    with strategy.scope():
      v = variables.Variable(init_value)

    self.assertIsInstance(v, d_variable.DVariable)
    self.assertIsNotNone(v.layout)
    self.assertEqual(v.layout, layout.Layout.replicated(self.mesh, rank=1))

  def test_mesh(self):
    strategy = mirrored_strategy.MirroredStrategy(self.mesh)
    self.assertEqual(strategy._mesh, self.mesh)

  def test_strategy_extension(self):
    strategy = mirrored_strategy.MirroredStrategy(self.mesh)
    self.assertIsInstance(strategy.extended, distribute_lib.StrategyExtendedV2)

  def test_num_replica_in_sync(self):
    strategy = mirrored_strategy.MirroredStrategy(self.mesh)
    self.assertEqual(strategy.num_replicas_in_sync, 2)

  def test_worker_devices(self):
    strategy = mirrored_strategy.MirroredStrategy(self.mesh)
    worker_devices = strategy.extended.worker_devices
    self.assertLen(worker_devices, 2)
    self.assertEqual(worker_devices, tuple(self.mesh.local_devices()))

  def test_parameter_devices(self):
    strategy = mirrored_strategy.MirroredStrategy(self.mesh)
    parameter_devices = strategy.extended.parameter_devices
    self.assertLen(parameter_devices, 2)
    self.assertEqual(parameter_devices, tuple(self.mesh.local_devices()))

  def test_variable_created_in_scope(self):
    strategy1 = mirrored_strategy.MirroredStrategy(self.mesh)
    with strategy1.scope():
      v1 = variables.Variable(constant_op.constant([1.0, 2.0]))

    v2 = variables.Variable(constant_op.constant([1.0, 2.0]))

    strategy2 = mirrored_strategy.MirroredStrategy(self.mesh)
    with strategy2.scope():
      v3 = variables.Variable(constant_op.constant([1.0, 2.0]))

    self.assertTrue(strategy1.extended.variable_created_in_scope(v1))
    self.assertFalse(strategy1.extended.variable_created_in_scope(v2))
    self.assertFalse(strategy1.extended.variable_created_in_scope(v3))
    self.assertTrue(strategy2.extended.variable_created_in_scope(v3))

  def test_colocate_vars_with(self):
    strategy = mirrored_strategy.MirroredStrategy(self.mesh)
    with strategy.scope():
      v1 = variables.Variable(constant_op.constant([1.0, 2.0]))
      with strategy.extended.colocate_vars_with(v1):
        v2 = variables.Variable(constant_op.constant([2.0, 3.0]))

    # We assert the layout for the variable, and make sure they are same.
    self.assertEqual(v1.layout, v2.layout)

  def test_in_multi_worker_mode(self):
    strategy = mirrored_strategy.MirroredStrategy(self.mesh)
    self.assertFalse(strategy.extended._in_multi_worker_mode())


class InvalidMeshTest(test_util.DTensorBaseTest):

  def setUp(self):
    super().setUp()
    global_ids = test_util.create_device_ids_array((2, 1))
    local_ids = np.ravel(global_ids).tolist()
    mesh_dict = {
        device: layout.Mesh(['batch', 'model'], global_ids, local_ids,
                            test_util.create_device_list((2,), device))
        for device in ['TPU', 'GPU', 'CPU']
    }
    self.mesh_2d = self.configTestMesh(mesh_dict)

  def test_invalid_mesh_shape(self):
    with self.assertRaisesRegex(
        ValueError, 'The mesh for MirroredStrategy must be 1D, received: 2D'):
      mirrored_strategy.MirroredStrategy(self.mesh_2d)


class StrategyCreationTest(test_util.DTensorBaseTest):

  def setUp(self):
    super().setUp()
    device_type = test_util.preferred_device_type()
    if device_type != 'TPU':
      test_util.reset_logical_devices(device_type, 2)
    self.device_type = device_type

  def test_explicit_device_list(self):

    device_list = [f'/{self.device_type}:{i}' for i in range(2)]
    strategy = mirrored_strategy.MirroredStrategy(devices=device_list)
    mesh = strategy._mesh
    self.assertEqual(mesh.num_local_devices(), 2)
    self.assertEqual(mesh.shape(), [2,])
    self.assertEqual(mesh.dim_names, ['batch'])
    self.assertIn(
        f'/job:localhost/replica:0/task:0/device:{self.device_type}:0',
        mesh.local_devices()[0])
    self.assertIn(
        f'/job:localhost/replica:0/task:0/device:{self.device_type}:1',
        mesh.local_devices()[1])

  def test_implicit_device_list(self):
    strategy = mirrored_strategy.MirroredStrategy()
    mesh = strategy._mesh
    self.assertEqual(mesh.num_local_devices(), 2)
    self.assertEqual(mesh.shape(), [2,])
    self.assertIn(
        f'/job:localhost/replica:0/task:0/device:{self.device_type}:0',
        mesh.local_devices()[0])
    self.assertIn(
        f'/job:localhost/replica:0/task:0/device:{self.device_type}:1',
        mesh.local_devices()[1])

  def test_mesh_with_device_list(self):
    device_list = [f'/{self.device_type}:{i}' for i in range(2)]
    mesh = mesh_util.create_mesh([('batch', 2)], devices=device_list)
    with self.assertRaisesRegex(
        ValueError, 'Mesh and devices can not be provided at the same time'):
      _ = mirrored_strategy.MirroredStrategy(mesh=mesh, devices=device_list)


if __name__ == '__main__':
  test.main()
