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

from tensorflow.dtensor.python import api as d_api
from tensorflow.dtensor.python import d_variable
from tensorflow.dtensor.python import layout
from tensorflow.dtensor.python import mesh_util
from tensorflow.dtensor.python.tests import test_util
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import distribution_strategy_context
from tensorflow.python.distribute import reduce_util
from tensorflow.python.distribute.experimental import dtensor_util
from tensorflow.python.distribute.experimental import mirrored_strategy
from tensorflow.python.eager import def_function
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import stateless_random_ops
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

  def test_variable_creation_with_dtype(self):
    strategy = mirrored_strategy.MirroredStrategy(self.mesh)
    with strategy.scope():
      v = variables.Variable(
          0, dtype='int64',
          aggregation=variables.VariableAggregationV2.ONLY_FIRST_REPLICA)
    self.assertIsInstance(v, d_variable.DVariable)
    self.assertEqual(v.dtype, dtypes.int64)

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

  def test_run_with_tensor_inputs(self):
    strategy = mirrored_strategy.MirroredStrategy(self.mesh)
    tensor_input = constant_op.constant(3.0)

    @def_function.function
    def replica_fn(inputs):
      return inputs * 2.0

    with self.assertRaisesRegex(
        ValueError, 'Unsupported input types for MirroredStrategy.'):
      strategy.run(replica_fn, args=(tensor_input,))

  def test_run_with_graph_tensor_inputs(self):
    # Note that this is potentially a sharp edge for the user, since the eager
    # test case was raising an error, but the graph context will run, by treat
    # the inputs as a global inputs.
    # TODO(scottzhu): Mitigate this eager/graph behavior difference in future.
    strategy = mirrored_strategy.MirroredStrategy(self.mesh)

    @def_function.function
    def replica_fn(inputs):
      return inputs * 2.0

    @def_function.function
    def run_fn():
      tensor_input = constant_op.constant(3.0)
      return strategy.run(replica_fn, args=(tensor_input,))

    # TODO(b/274647196): Change to use strategy.scope() for default device/mesh.
    # with strategy.scope():
    with d_api.default_mesh(self.mesh):
      result = run_fn()
    self.assertEqual(result, constant_op.constant(6.0))

  def test_run_with_unsupported_input_types(self):
    strategy = mirrored_strategy.MirroredStrategy(self.mesh)
    random_inputs = [123, '456']

    @def_function.function
    def replica_fn(inputs):
      return inputs * 2.0

    with self.assertRaisesRegex(
        ValueError, 'Unsupported input types for MirroredStrategy.'):
      strategy.run(replica_fn, args=(random_inputs,))

  def test_run_with_distribute_value_input(self):
    strategy = mirrored_strategy.MirroredStrategy(self.mesh)

    def value_fn(value_context):
      return value_context.num_replicas_in_sync
    distributed_values = (
        strategy.experimental_distribute_values_from_function(
            value_fn))

    @def_function.function
    def replica_fn(inputs):
      return inputs * 2

    result = strategy.run(replica_fn, args=(distributed_values,))
    self.assertIsInstance(result, dtensor_util.DTensorDistributedValue)
    self.assertLen(result.values, 2)
    # Note that the scalar value from
    # experimental_distribute_values_from_function will be up rank to 1D since
    # batched shared dtensor need at least be 1D. So the result from the
    # strategy.run is [4], instead of just 4.
    self.assertAllClose(result.values[0], constant_op.constant([4]))
    self.assertAllClose(result.values[1], constant_op.constant([4]))

  def test_nested_structure_output(self):
    strategy = mirrored_strategy.MirroredStrategy(self.mesh)
    array_value = np.array([3., 2., 1.])
    def value_fn(ctx):
      value = array_value[ctx.replica_id_in_sync_group]
      return {'a': value,
              'b': constant_op.constant([value + 1.0, value + 2.0])}
    distributed_values = (
        strategy.experimental_distribute_values_from_function(
            value_fn))

    @def_function.function
    def replica_fn(inputs):
      result = {}
      for key in inputs:
        result[key] = inputs[key] * 2.0
      return result

    result = strategy.run(replica_fn, args=(distributed_values,))
    self.assertLen(result.keys(), 2)
    self.assertIsInstance(result['a'], dtensor_util.DTensorDistributedValue)
    self.assertAllClose(result['a'].values[0], constant_op.constant([6.0]))
    self.assertAllClose(result['a'].values[1], constant_op.constant([4.0]))

    self.assertIsInstance(result['b'], dtensor_util.DTensorDistributedValue)
    self.assertAllClose(result['b'].values[0],
                        constant_op.constant([8.0, 10.0]))
    self.assertAllClose(result['b'].values[1], constant_op.constant([6.0, 8.0]))

  def test_inputs_with_dtensor_distribute_values(self):

    @def_function.function
    def replica_fn_1(inputs):
      return inputs * 2.0

    @def_function.function
    def replica_fn_2(inputs):
      return inputs + 1.0

    strategy = mirrored_strategy.MirroredStrategy(self.mesh)
    tensor_input = constant_op.constant(3.0)
    d_tensor_input = strategy.experimental_distribute_values_from_function(
        lambda _: tensor_input)

    result_1 = strategy.run(replica_fn_1, args=(d_tensor_input,))
    self.assertIsInstance(result_1, dtensor_util.DTensorDistributedValue)
    self.assertLen(result_1.values, 2)
    self.assertAllClose(result_1.values[0], constant_op.constant([6.0]))
    self.assertAllClose(result_1.values[1], constant_op.constant([6.0]))

    result_2 = strategy.run(replica_fn_2, args=(result_1,))
    self.assertIsInstance(result_2, dtensor_util.DTensorDistributedValue)
    self.assertLen(result_2.values, 2)
    self.assertAllClose(result_2.values[0], constant_op.constant([7.0]))
    self.assertAllClose(result_2.values[1], constant_op.constant([7.0]))

  def test_run_with_nullary_ops(self):

    @def_function.function
    def replica_fn():
      return constant_op.constant([3.0])

    strategy = mirrored_strategy.MirroredStrategy(self.mesh)
    result = strategy.run(replica_fn)

    self.assertIsInstance(result, dtensor_util.DTensorDistributedValue)
    self.assertAllClose(result.values[0], constant_op.constant([3.0]))
    self.assertAllClose(result.values[1], constant_op.constant([3.0]))

  def test_get_replica_context(self):
    strategy = mirrored_strategy.MirroredStrategy(self.mesh)

    tensor_input = constant_op.constant(3)
    d_tensor_input = strategy.experimental_distribute_values_from_function(
        lambda _: tensor_input)

    @def_function.function
    def replica_fn(inputs):
      replica_context = distribution_strategy_context.get_replica_context()
      self.assertIsInstance(replica_context, dtensor_util.DTensorReplicaContext)
      return inputs * replica_context.num_replicas_in_sync

    # Default replica context
    self.assertIsNotNone(distribution_strategy_context.get_replica_context())
    with strategy.scope():
      self.assertIsNone(distribution_strategy_context.get_replica_context())

      result = strategy.run(replica_fn, args=(d_tensor_input,))

    self.assertLen(result.values, 2)
    self.assertAllClose(result.values[0], constant_op.constant([6]))
    self.assertAllClose(result.values[1], constant_op.constant([6]))

  def test_gather_non_dtensor_value(self):
    strategy = mirrored_strategy.MirroredStrategy(self.mesh)
    tensor_input = constant_op.constant(3.0)

    result = strategy.gather(tensor_input, axis=0)
    self.assertAllClose(result, tensor_input)

  def test_gather_dtensor_value(self):
    strategy = mirrored_strategy.MirroredStrategy(self.mesh)

    def value_fn(value_context):
      start = value_context.replica_id_in_sync_group
      return array_ops.reshape(math_ops.range(start=start, limit=start + 6),
                               shape=(1, 2, 3))

    distribute_result = strategy.experimental_distribute_values_from_function(
        value_fn)
    result = strategy.gather(distribute_result, axis=0)
    self.assertEqual(result.shape, [2, 2, 3])
    self.assertAllClose(result, [[[0, 1, 2], [3, 4, 5]],
                                 [[1, 2, 3], [4, 5, 6]]])
    result = strategy.gather(distribute_result, axis=1)
    self.assertEqual(result.shape, [1, 4, 3])
    self.assertAllClose(result, [[[0, 1, 2], [3, 4, 5], [1, 2, 3], [4, 5, 6]]])
    result = strategy.gather(distribute_result, axis=2)
    self.assertEqual(result.shape, [1, 2, 6])
    self.assertAllClose(result, [[[0, 1, 2, 1, 2, 3], [3, 4, 5, 4, 5, 6]]])

  def test_reduce_mean_non_dtensor_value(self):
    strategy = mirrored_strategy.MirroredStrategy(self.mesh)
    tensor_input = constant_op.constant([[3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])

    with self.assertRaisesRegex(
        ValueError, 'Unsupported input types for MirroredStrategy.'):
      strategy.reduce(reduce_util.ReduceOp.MEAN, tensor_input, axis=0)

  def test_reduce_sum_non_dtensor_value(self):
    strategy = mirrored_strategy.MirroredStrategy(self.mesh)
    tensor_input = constant_op.constant([[3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])

    with self.assertRaisesRegex(
        ValueError, 'Unsupported input types for MirroredStrategy.'):
      strategy.reduce(reduce_util.ReduceOp.SUM, tensor_input, axis=0)

  def test_reduce_mean_distribute_value(self):
    strategy = mirrored_strategy.MirroredStrategy(self.mesh)

    @def_function.function
    def value_fn(value_context):
      i = value_context.replica_id_in_sync_group
      n = value_context.num_replicas_in_sync
      return constant_op.constant([[0.0, 1.0], [2.0, 3.0]]) + i * n * 2.0

    distribute_value = strategy.experimental_distribute_values_from_function(
        value_fn)
    # replica 1 has [[0.0, 1.0],[2.0, 3.0]] and replica 2 has
    # [[4.0, 5.0],[6.0, 7.0]]

    result = strategy.reduce(
        reduce_util.ReduceOp.MEAN, distribute_value, axis=None)
    self.assertAllClose(result, constant_op.constant([[2.0, 3.0], [4.0, 5.0]]))

    result = strategy.reduce(
        reduce_util.ReduceOp.MEAN, distribute_value, axis=0)
    self.assertAllClose(result, constant_op.constant([3.0, 4.0]))

    result = strategy.reduce(
        reduce_util.ReduceOp.MEAN, distribute_value, axis=1)
    self.assertAllClose(result, constant_op.constant([2.5, 4.5]))

  def test_reduce_sum_distribute_value(self):
    strategy = mirrored_strategy.MirroredStrategy(self.mesh)

    @def_function.function
    def value_fn(value_context):
      i = value_context.replica_id_in_sync_group
      n = value_context.num_replicas_in_sync
      return constant_op.constant([[0.0, 1.0], [2.0, 3.0]]) + i * n * 2.0

    distribute_value = strategy.experimental_distribute_values_from_function(
        value_fn)
    # replica 1 has [[0.0, 1.0],[2.0, 3.0]] and replica 2 has
    # [[4.0, 5.0],[6.0, 7.0]]

    result = strategy.reduce(
        reduce_util.ReduceOp.SUM, distribute_value, axis=None)
    self.assertAllClose(result, constant_op.constant([[4.0, 6.0], [8.0, 10.0]]))

    result = strategy.reduce(
        reduce_util.ReduceOp.SUM, distribute_value, axis=0)
    self.assertAllClose(result, constant_op.constant([12.0, 16.0]))

    result = strategy.reduce(
        reduce_util.ReduceOp.SUM, distribute_value, axis=1)
    self.assertAllClose(result, constant_op.constant([10.0, 18.0]))

  def test_reduce_mean_mirrored_value(self):
    strategy = mirrored_strategy.MirroredStrategy(self.mesh)

    with  strategy.scope():
      v = variables.Variable(constant_op.constant([[1.0, 2.0], [3.0, 4.0]]))
    self.assertIsInstance(v, d_variable.DVariable)

    result = strategy.reduce(reduce_util.ReduceOp.MEAN, v, axis=None)
    self.assertAllClose(result, constant_op.constant([[1.0, 2.0], [3.0, 4.0]]))
    result = strategy.reduce(reduce_util.ReduceOp.MEAN, v, axis=0)
    self.assertAllClose(result, constant_op.constant([2.0, 3.0]))
    result = strategy.reduce(reduce_util.ReduceOp.MEAN, v, axis=1)
    self.assertAllClose(result, constant_op.constant([1.5, 3.5]))

  def test_reduce_sum_mirrored_value(self):
    strategy = mirrored_strategy.MirroredStrategy(self.mesh)

    with  strategy.scope():
      v = variables.Variable(constant_op.constant([[1.0, 2.0], [3.0, 4.0]]))
    self.assertIsInstance(v, d_variable.DVariable)

    result = strategy.reduce(reduce_util.ReduceOp.SUM, v, axis=None)
    self.assertAllClose(result, constant_op.constant([[1.0, 2.0], [3.0, 4.0]]))
    result = strategy.reduce(reduce_util.ReduceOp.SUM, v, axis=0)
    self.assertAllClose(result, constant_op.constant([4.0, 6.0]))
    result = strategy.reduce(reduce_util.ReduceOp.SUM, v, axis=1)
    self.assertAllClose(result, constant_op.constant([3.0, 7.0]))

  def test_reduce_value_device(self):
    strategy = mirrored_strategy.MirroredStrategy(self.mesh)
    tensor_input = constant_op.constant([[3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])

    result = strategy.reduce(reduce_util.ReduceOp.MEAN, tensor_input, axis=None)
    self.assertIn('CPU:0', result.device)

  def test_experimental_local_results(self):
    @def_function.function
    def replica_fn():
      return constant_op.constant([3.0])

    strategy = mirrored_strategy.MirroredStrategy(self.mesh)
    result = strategy.run(replica_fn)
    local_result = strategy.experimental_local_results(result)

    self.assertIsInstance(local_result, tuple)
    self.assertLen(local_result, 2)
    self.assertEqual(local_result[0], constant_op.constant([3.0]))
    self.assertEqual(local_result[1], constant_op.constant([3.0]))

  def test_experimental_local_results_with_inputs(self):
    strategy = mirrored_strategy.MirroredStrategy(self.mesh)
    array_value = np.array([3., 2.])
    def value_fn(ctx):
      value = array_value[ctx.replica_id_in_sync_group]
      return {'a': value,
              'b': constant_op.constant([value + 1.0, value + 2.0])}
    distributed_values = (
        strategy.experimental_distribute_values_from_function(
            value_fn))

    @def_function.function
    def replica_fn(inputs):
      result = {}
      for key in inputs:
        result[key] = inputs[key] * 2.0
      return result

    result = strategy.run(replica_fn, args=(distributed_values,))
    local_result = strategy.experimental_local_results(result)
    self.assertIsInstance(local_result, tuple)
    self.assertLen(local_result, 2)
    self.assertDictEqual(local_result[0],
                         {'a': constant_op.constant([6.0]),
                          'b': constant_op.constant([8.0, 10.0])})
    self.assertDictEqual(local_result[1],
                         {'a': constant_op.constant([4.0]),
                          'b': constant_op.constant([6.0, 8.0])})


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


class StrategyDatasetTest(test_util.DTensorBaseTest):

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

    self.images = stateless_random_ops.stateless_random_uniform(
        [8, 8, 3], seed=(1, 2), minval=0, maxval=255)
    self.labels = stateless_random_ops.stateless_random_uniform(
        [1], seed=(1, 2), minval=0, maxval=10)

    self.dataset = dataset_ops.Dataset.from_tensors(
        (self.images, self.labels)).repeat()

  def test_create_batched_dataset(self):
    strategy = mirrored_strategy.MirroredStrategy(self.mesh)
    global_batch_size = 8
    dataset = self.dataset.batch(global_batch_size).prefetch(2)

    distributed_dataset = strategy.experimental_distribute_dataset(dataset)
    element = next(iter(distributed_dataset))
    batched_image, batched_label = element
    self.assertEqual(batched_image.shape, [global_batch_size, 8, 8, 3])
    self.assertEqual(batched_label.shape, [global_batch_size, 1])

    # Make sure when unpack the tensor, each of them has enough shards.
    self.assertLen(d_api.unpack(batched_image), self.mesh.num_local_devices())
    self.assertLen(d_api.unpack(batched_label), self.mesh.num_local_devices())

  def test_uneven_batched_dataset(self):
    elements = [[1, 2, 3], [1, 2], [1, 2, 3, 4]]
    dataset = dataset_ops.Dataset.from_generator(
        lambda: elements, dtypes.int64).repeat()
    strategy = mirrored_strategy.MirroredStrategy(self.mesh)
    with self.assertRaisesRegex(ValueError, 'requires a static batch size'):
      strategy.experimental_distribute_dataset(dataset)

  def test_create_partial_batched_dataset(self):
    # TODO(b/210887657): Support last partial batch.
    self.skipTest('Test failed due to last partial batch')
    dataset = dataset_ops.Dataset.from_tensors(
        (self.images, self.labels)).repeat(30)  # There is a last partial batch

    strategy = mirrored_strategy.MirroredStrategy(self.mesh)
    global_batch_size = 8
    dataset = dataset.batch(global_batch_size).prefetch(2)

    distributed_dataset = strategy.experimental_distribute_dataset(dataset)
    expected_element_batch_size = [8, 8, 8, 6]
    # The last batch with 6 element will fail to produce with StopIteration.
    iterator = iter(distributed_dataset)
    for batch_size in expected_element_batch_size:
      element = next(iterator)
      batched_image, batched_label = element
      self.assertEqual(batched_image.shape, [batch_size, 8, 8, 3])
      self.assertEqual(batched_label.shape, [batch_size, 1])

      # Make sure when unpack the tensor, each of them has enough shards.
      self.assertLen(d_api.unpack(batched_image), self.mesh.num_local_devices())
      self.assertLen(d_api.unpack(batched_label), self.mesh.num_local_devices())

  def test_deprecated_strategy_methods(self):
    strategy = mirrored_strategy.MirroredStrategy(self.mesh)
    with self.assertRaisesRegex(
        NotImplementedError, 'only available in the V1 API'):
      strategy.make_dataset_iterator(self.dataset)

    with self.assertRaisesRegex(
        NotImplementedError, 'only available in the V1 API'):
      strategy.make_input_fn_iterator(lambda _: self.dataset)

  def test_distribute_dataset_from_fn(self):
    local_batch_size = 4
    global_batch_size = 8
    def dataset_fn(option):
      del option
      return dataset_ops.Dataset.from_tensors(
          (self.images, self.labels)).repeat().batch(
              local_batch_size, drop_remainder=True).prefetch(2)
    strategy = mirrored_strategy.MirroredStrategy(self.mesh)
    distributed_dataset = strategy.distribute_datasets_from_function(
        dataset_fn, None)

    element = next(iter(distributed_dataset))
    batched_image, batched_label = element
    self.assertEqual(batched_image.shape, [global_batch_size, 8, 8, 3])
    self.assertEqual(batched_label.shape, [global_batch_size, 1])

    # Make sure there are two shards when unpack, and each of them has 4 as
    # batch size
    unpacked_images = d_api.unpack(batched_image)
    self.assertLen(unpacked_images, self.mesh.num_local_devices())
    self.assertEqual(unpacked_images[0].shape, [local_batch_size, 8, 8, 3])
    self.assertEqual(unpacked_images[1].shape, [local_batch_size, 8, 8, 3])

  def test_distribute_values_from_function(self):
    array_value = np.array([3., 2., 1.])
    def value_fn(ctx):
      return array_value[ctx.replica_id_in_sync_group]
    strategy = mirrored_strategy.MirroredStrategy(self.mesh)
    distributed_values = (
        strategy.experimental_distribute_values_from_function(
            value_fn))
    self.assertDTensorEqual(
        constant_op.constant([3., 2.], dtype=dtypes.float64),
        layout.Layout.batch_sharded(self.mesh, batch_dim='batch', rank=1),
        distributed_values)

  def test_distribute_values_from_function_with_nested_structure(self):
    array_value = np.array([3., 2., 1.])
    def value_fn(ctx):
      value = array_value[ctx.replica_id_in_sync_group]
      return {'a': value,
              'b': constant_op.constant([value + 1.0, value + 2.0])}
    strategy = mirrored_strategy.MirroredStrategy(self.mesh)
    distributed_values = (
        strategy.experimental_distribute_values_from_function(
            value_fn))
    self.assertIsInstance(distributed_values, dict)
    self.assertDTensorEqual(
        constant_op.constant([3., 2.], dtype=dtypes.float64),
        layout.Layout.batch_sharded(self.mesh, batch_dim='batch', rank=1),
        distributed_values['a'])
    unpacked_a = d_api.unpack(distributed_values['a'])
    # Note that this might have a slight behavior difference, the original
    # mirrored strategy may return scalar for each PerReplica. The DTensor
    # implementation is more strict and always ensures the PerReplica
    # value has the same rank as the global-view Tensor.
    self.assertAllClose(unpacked_a[0], [3.])
    self.assertAllClose(unpacked_a[1], [2.])
    self.assertDTensorEqual(
        constant_op.constant([4., 5., 3., 4.], dtype=dtypes.float64),
        layout.Layout.batch_sharded(self.mesh, batch_dim='batch', rank=1),
        distributed_values['b'])

  def test_distribute_dataset_in_tf_function(self):
    strategy = mirrored_strategy.MirroredStrategy(self.mesh)
    local_batch_size = 4
    global_batch_size = 8
    dataset = self.dataset.batch(global_batch_size).prefetch(2)

    distributed_dataset = strategy.experimental_distribute_dataset(dataset)

    @def_function.function
    def step_fn(iterator):
      images, labels = next(iterator)
      del labels
      return images

    result = strategy.run(step_fn, args=(iter(distributed_dataset),))
    self.assertIsInstance(result, dtensor_util.DTensorDistributedValue)
    self.assertLen(result.values, self.mesh.num_local_devices())
    self.assertEqual(result.values[0].shape, [local_batch_size, 8, 8, 3])
    self.assertEqual(result.values[1].shape, [local_batch_size, 8, 8, 3])


if __name__ == '__main__':
  test.main()
