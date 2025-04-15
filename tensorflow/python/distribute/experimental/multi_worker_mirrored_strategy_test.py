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
"""Test for MultiWorkerMirroredStrategy backed by DTensor API."""

import json
import os

from absl import flags
from absl.testing import parameterized
import numpy as np

from tensorflow.dtensor.python import api as d_api
from tensorflow.dtensor.python import d_variable
from tensorflow.dtensor.python import layout
from tensorflow.dtensor.python.tests import multi_client_test_util
from tensorflow.dtensor.python.tests import test_backend_util
from tensorflow.dtensor.python.tests import test_util
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import reduce_util
from tensorflow.python.distribute.cluster_resolver import tfconfig_cluster_resolver
from tensorflow.python.distribute.experimental import dtensor_util
from tensorflow.python.distribute.experimental import multi_worker_mirrored_strategy as mwms
from tensorflow.python.eager import def_function
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import stateless_random_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test as tf_test


class MultiWorkerMirroredStrategyTest(tf_test.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.num_client = flags.FLAGS.num_clients
    self.num_local_devices = flags.FLAGS.num_local_devices

    tf_config = json.loads(os.environ['TF_CONFIG'])
    self.client_id = int(tf_config['task']['index'])

  def test_strategy_creation_with_default_cluster_resolver(self):
    strategy = mwms.MultiWorkerMirroredStrategy()
    mesh = strategy.mesh
    self.assertIsNotNone(mesh)
    self.assertLen(mesh.global_device_ids(),
                   self.num_client * self.num_local_devices)
    self.assertLen(mesh.local_device_ids(), self.num_local_devices)
    self.assertIsInstance(strategy._cluster_resolver,
                          tfconfig_cluster_resolver.TFConfigClusterResolver)

  def test_invalid_init_arguments(self):
    mesh = object()
    cluster_resolver = tfconfig_cluster_resolver.TFConfigClusterResolver()

    with self.assertRaisesRegex(
        ValueError,
        'Mesh and cluster_resolver can not be provided at the same time'):
      mwms.MultiWorkerMirroredStrategy(
          mesh=mesh,
          cluster_resolver=cluster_resolver)

  def test_parse_dtensor_env_var_from_cluster_resolver(self):
    cluster_resolver = tfconfig_cluster_resolver.TFConfigClusterResolver()

    dtensor_env_vars = mwms._parse_dtensor_env_var_from_cluster_resolver(
        cluster_resolver)

    tf_config = json.loads(os.environ['TF_CONFIG'])
    worker_jobs = ','.join(tf_config['cluster']['worker'])
    client_id = tf_config['task']['index']

    self.assertLen(dtensor_env_vars, 4)
    self.assertEqual(dtensor_env_vars['DTENSOR_JOBS'], worker_jobs)
    self.assertEqual(dtensor_env_vars['DTENSOR_NUM_CLIENTS'],
                     str(self.num_client))
    self.assertEqual(dtensor_env_vars['DTENSOR_CLIENT_ID'], client_id)
    self.assertEqual(dtensor_env_vars['DTENSOR_JOB_NAME'], 'worker')

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
    strategy = mwms.MultiWorkerMirroredStrategy()
    with strategy.scope():
      v = variables.Variable(init_value)

    self.assertIsInstance(v, d_variable.DVariable)
    self.assertIsNotNone(v.layout)
    self.assertEqual(v.layout, layout.Layout.replicated(strategy.mesh, rank=1))

  def test_strategy_extension(self):
    strategy = mwms.MultiWorkerMirroredStrategy()
    self.assertIsInstance(strategy.extended, distribute_lib.StrategyExtendedV2)

  def test_num_replica_in_sync(self):
    strategy = mwms.MultiWorkerMirroredStrategy()
    self.assertEqual(strategy.num_replicas_in_sync,
                     self.num_client * self.num_local_devices)

  def test_mesh(self):
    strategy = mwms.MultiWorkerMirroredStrategy()
    self.assertIsNotNone(strategy.mesh)

  def test_worker_devices(self):
    strategy = mwms.MultiWorkerMirroredStrategy()
    worker_devices = strategy.extended.worker_devices
    self.assertLen(worker_devices, self.num_local_devices)
    self.assertEqual(worker_devices, tuple(strategy.mesh.local_devices()))

  def test_parameter_devices(self):
    strategy = mwms.MultiWorkerMirroredStrategy()
    parameter_devices = strategy.extended.parameter_devices
    self.assertLen(parameter_devices, self.num_local_devices)
    self.assertEqual(parameter_devices, tuple(strategy.mesh.local_devices()))

  def test_variable_created_in_scope(self):
    strategy1 = mwms.MultiWorkerMirroredStrategy()
    with strategy1.scope():
      v1 = variables.Variable(constant_op.constant([1.0, 2.0]))

    v2 = variables.Variable(constant_op.constant([1.0, 2.0]))

    strategy2 = mwms.MultiWorkerMirroredStrategy()
    with strategy2.scope():
      v3 = variables.Variable(constant_op.constant([1.0, 2.0]))

    self.assertTrue(strategy1.extended.variable_created_in_scope(v1))
    self.assertFalse(strategy1.extended.variable_created_in_scope(v2))
    self.assertFalse(strategy1.extended.variable_created_in_scope(v3))
    self.assertTrue(strategy2.extended.variable_created_in_scope(v3))

  def test_colocate_vars_with(self):
    strategy = mwms.MultiWorkerMirroredStrategy()
    with strategy.scope():
      v1 = variables.Variable(constant_op.constant([1.0, 2.0]))
      with strategy.extended.colocate_vars_with(v1):
        v2 = variables.Variable(constant_op.constant([2.0, 3.0]))

    # We assert the layout for the variable, and make sure they are same.
    self.assertEqual(v1.layout, v2.layout)

  def test_in_multi_worker_mode(self):
    strategy = mwms.MultiWorkerMirroredStrategy()
    self.assertTrue(strategy.extended._in_multi_worker_mode())

  def test_run_with_distribute_value_input(self):
    strategy = mwms.MultiWorkerMirroredStrategy()

    def value_fn(value_context):
      return value_context.replica_id_in_sync_group
    distributed_values = (
        strategy.experimental_distribute_values_from_function(
            value_fn))

    @def_function.function
    def replica_fn(inputs):
      return inputs * 2

    result = strategy.run(replica_fn, args=(distributed_values,))
    self.assertIsInstance(result, dtensor_util.DTensorDistributedValue)
    self.assertLen(result.values, self.num_local_devices)
    # Note that the scalar value from
    # experimental_distribute_values_from_function will be up rank to 1D since
    # batched shared dtensor need at least be 1D.

    for i in range(self.num_local_devices):
      self.assertAllClose(
          result.values[i],
          constant_op.constant(
              [(self.client_id * self.num_local_devices + i) * 2]))

  def test_nested_structure_output(self):
    strategy = mwms.MultiWorkerMirroredStrategy()
    def value_fn(ctx):
      value = float(ctx.num_replicas_in_sync)
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
    self.assertLen(result['a'].values, self.num_local_devices)
    for i in range(self.num_local_devices):
      self.assertAllClose(
          result['a'].values[i],
          constant_op.constant([strategy.num_replicas_in_sync * 2.0]))

    self.assertIsInstance(result['b'], dtensor_util.DTensorDistributedValue)
    self.assertLen(result['b'].values, self.num_local_devices)
    for i in range(self.num_local_devices):
      self.assertAllClose(
          result['b'].values[i],
          constant_op.constant([(strategy.num_replicas_in_sync + 1.0) * 2.0,
                                (strategy.num_replicas_in_sync + 2.0) * 2.0]))

  def test_inputs_with_dtensor_distribute_values(self):

    @def_function.function
    def replica_fn_1(inputs):
      return inputs * 2.0

    @def_function.function
    def replica_fn_2(inputs):
      return inputs + 1.0

    strategy = mwms.MultiWorkerMirroredStrategy()
    tensor_input = constant_op.constant(3.0)
    d_tensor_input = strategy.experimental_distribute_values_from_function(
        lambda _: tensor_input)

    result_1 = strategy.run(replica_fn_1, args=(d_tensor_input,))
    self.assertIsInstance(result_1, dtensor_util.DTensorDistributedValue)
    self.assertLen(result_1.values, self.num_local_devices)
    for i in range(self.num_local_devices):
      self.assertAllClose(result_1.values[i], constant_op.constant([6.0]))

    result_2 = strategy.run(replica_fn_2, args=(result_1,))
    self.assertIsInstance(result_2, dtensor_util.DTensorDistributedValue)
    self.assertLen(result_2.values, self.num_local_devices)
    for i in range(self.num_local_devices):
      self.assertAllClose(result_2.values[i], constant_op.constant([7.0]))

  def test_get_replica_context(self):
    strategy = mwms.MultiWorkerMirroredStrategy()

    tensor_input = constant_op.constant(3)
    d_tensor_input = strategy.experimental_distribute_values_from_function(
        lambda _: tensor_input)

    @def_function.function
    def replica_fn(inputs):
      replica_context = distribute_lib.get_replica_context()
      self.assertIsInstance(replica_context, dtensor_util.DTensorReplicaContext)
      return inputs * replica_context.num_replicas_in_sync

    # Default replica context
    self.assertIsNotNone(distribute_lib.get_replica_context())
    with strategy.scope():
      self.assertIsNone(distribute_lib.get_replica_context())

      result = strategy.run(replica_fn, args=(d_tensor_input,))

    self.assertLen(result.values, self.num_local_devices)
    for i in range(self.num_local_devices):
      self.assertAllClose(
          result.values[i],
          constant_op.constant([3 * strategy.num_replicas_in_sync]))

  def test_gather_non_dtensor_value(self):
    strategy = mwms.MultiWorkerMirroredStrategy()
    tensor_input = constant_op.constant(3.0)

    result = strategy.gather(tensor_input, axis=0)
    self.assertAllClose(result, tensor_input)

  def test_gather_dtensor_value(self):
    strategy = mwms.MultiWorkerMirroredStrategy()
    stride = self.num_client * self.num_local_devices

    def value_fn(value_context):
      start = value_context.replica_id_in_sync_group * stride
      return array_ops.reshape(
          math_ops.range(start=start, limit=start + stride), shape=(1, stride)
      )
    distribute_result = strategy.experimental_distribute_values_from_function(
        value_fn
    )
    # distribute_result is a DTensorDistributedValue.
    # The shape of the global tensor is [stride, stride],
    # and each worker gets [stride/2, stride].
    result = strategy.gather(distribute_result, axis=0)

    start = stride * self.num_local_devices * self.client_id
    end = start + stride * self.num_local_devices
    self.assertEqual(result.shape, [self.num_local_devices, stride])
    self.assertAllClose(
        result,
        array_ops.reshape(
            math_ops.range(start=start, limit=end),
            shape=(self.num_local_devices, -1),
        ),
    )

    result = strategy.gather(distribute_result, axis=1)
    self.assertEqual(result.shape, [1, self.num_local_devices * stride])
    self.assertAllClose(
        result,
        array_ops.reshape(
            math_ops.range(start=start, limit=end), shape=(1, -1)
        ),
    )

  def test_reduce_mean_non_dtensor_value(self):
    strategy = mwms.MultiWorkerMirroredStrategy()
    tensor_input = constant_op.constant([[3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])

    with self.assertRaisesRegex(
        ValueError, 'Unsupported input types for MirroredStrategy.'
    ):
      strategy.reduce(reduce_util.ReduceOp.MEAN, tensor_input, axis=0)

  def test_reduce_sum_non_dtensor_value(self):
    strategy = mwms.MultiWorkerMirroredStrategy()
    tensor_input = constant_op.constant([[3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])

    with self.assertRaisesRegex(
        ValueError, 'Unsupported input types for MirroredStrategy.'
    ):
      strategy.reduce(reduce_util.ReduceOp.SUM, tensor_input, axis=0)

  def test_reduce_mean_distribute_value(self):
    strategy = mwms.MultiWorkerMirroredStrategy()

    @def_function.function
    def value_fn(value_context):
      i = value_context.replica_id_in_sync_group
      return constant_op.constant([[0.0, 1.0], [2.0, 3.0]]) + i * 4.0

    distribute_value = strategy.experimental_distribute_values_from_function(
        value_fn
    )
    # replica 0 has [[0.0, 1.0],[2.0, 3.0]] and
    # replica 1 has [[4.0, 5.0],[6.0, 7.0]]. Each worker has 4 replicas.
    # For worker 2, it has replica 4 ~ 7.

    result = strategy.reduce(
        reduce_util.ReduceOp.MEAN, distribute_value, axis=None
    )
    # This should be a global reduce and each worker should have same value.
    # [[14.0, 15.0],[16.0, 17.0]]
    final = (self.num_local_devices * self.num_client - 1) * 2.0
    self.assertAllClose(
        result, constant_op.constant([[0.0, 1.0], [2.0, 3.0]]) + final
    )

    result = strategy.reduce(
        reduce_util.ReduceOp.MEAN, distribute_value, axis=0
    )
    # [15.0, 16.0]
    self.assertAllClose(result, constant_op.constant([0.0, 1.0]) + final + 1)

    result = strategy.reduce(
        reduce_util.ReduceOp.MEAN, distribute_value, axis=1
    )

    self.assertAllClose(result, constant_op.constant([0.5, 2.5]) + final)

  def test_reduce_sum_distribute_value(self):
    strategy = mwms.MultiWorkerMirroredStrategy()

    @def_function.function
    def value_fn(value_context):
      i = value_context.replica_id_in_sync_group
      return constant_op.constant([[0.0, 1.0], [2.0, 3.0]]) + i * 4.0

    distribute_value = strategy.experimental_distribute_values_from_function(
        value_fn
    )
    # replica 0 has [[0.0, 1.0],[2.0, 3.0]] and
    # replica 1 has [[4.0, 5.0],[6.0, 7.0]]. Each worker has 4 replicas.
    # For worker 2, it has replica 4 ~ 7.
    # The shape of the global tensor is [16, 2], and each worker gets [8, 2].

    result = strategy.reduce(
        reduce_util.ReduceOp.SUM, distribute_value, axis=None
    )
    self.assertAllClose(result, [[112.0, 120.0], [128.0, 136.0]])

    result = strategy.reduce(reduce_util.ReduceOp.SUM, distribute_value, axis=0)
    self.assertAllClose(result, constant_op.constant([240.0, 256.0]))

    result = strategy.reduce(reduce_util.ReduceOp.SUM, distribute_value, axis=1)
    self.assertAllClose(result, constant_op.constant([232.0, 264.0]))

  def test_reduce_mean_mirrored_value(self):
    strategy = mwms.MultiWorkerMirroredStrategy()

    with strategy.scope():
      v = variables.Variable(constant_op.constant([[1.0, 2.0], [3.0, 4.0]]))
    self.assertIsInstance(v, d_variable.DVariable)

    result = strategy.reduce(reduce_util.ReduceOp.MEAN, v, axis=None)
    self.assertAllClose(result, constant_op.constant([[1.0, 2.0], [3.0, 4.0]]))
    result = strategy.reduce(reduce_util.ReduceOp.MEAN, v, axis=0)
    self.assertAllClose(result, constant_op.constant([2.0, 3.0]))
    result = strategy.reduce(reduce_util.ReduceOp.MEAN, v, axis=1)
    self.assertAllClose(result, constant_op.constant([1.5, 3.5]))

  def test_reduce_sum_mirrored_value(self):
    strategy = mwms.MultiWorkerMirroredStrategy()

    with strategy.scope():
      v = variables.Variable(constant_op.constant([[1.0, 2.0], [3.0, 4.0]]))
    self.assertIsInstance(v, d_variable.DVariable)

    result = strategy.reduce(reduce_util.ReduceOp.SUM, v, axis=None)
    self.assertAllClose(result, constant_op.constant([[1.0, 2.0], [3.0, 4.0]]))
    result = strategy.reduce(reduce_util.ReduceOp.SUM, v, axis=0)
    self.assertAllClose(result, constant_op.constant([4.0, 6.0]))
    result = strategy.reduce(reduce_util.ReduceOp.SUM, v, axis=1)
    self.assertAllClose(result, constant_op.constant([3.0, 7.0]))

  def test_reduce_value_device(self):
    strategy = mwms.MultiWorkerMirroredStrategy()
    tensor_input = constant_op.constant([[3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])

    result = strategy.reduce(reduce_util.ReduceOp.MEAN, tensor_input, axis=None)
    self.assertIn('CPU:0', result.device)

  def test_experimental_local_results(self):
    @def_function.function
    def replica_fn():
      return constant_op.constant([3.0])

    strategy = mwms.MultiWorkerMirroredStrategy()
    result = strategy.run(replica_fn)
    local_result = strategy.experimental_local_results(result)

    self.assertIsInstance(local_result, tuple)
    self.assertLen(local_result, self.num_local_devices)
    for i in range(self.num_local_devices):
      self.assertEqual(local_result[i], constant_op.constant([3.0]))

  def test_experimental_local_results_with_inputs(self):
    strategy = mwms.MultiWorkerMirroredStrategy()

    def value_fn(ctx):
      value = float(ctx.num_replicas_in_sync)
      return {'a': value, 'b': constant_op.constant([value + 1.0, value + 2.0])}

    distributed_values = strategy.experimental_distribute_values_from_function(
        value_fn
    )

    @def_function.function
    def replica_fn(inputs):
      result = {}
      for key in inputs:
        result[key] = inputs[key] * 2.0
      return result

    result = strategy.run(replica_fn, args=(distributed_values,))
    local_result = strategy.experimental_local_results(result)
    self.assertIsInstance(local_result, tuple)
    self.assertLen(local_result, self.num_local_devices)
    for i in range(self.num_local_devices):
      self.assertDictEqual(
          local_result[i],
          {
              'a': constant_op.constant([16.0]),
              'b': constant_op.constant([18.0, 20.0]),
          },
      )


class StrategyDatasetTest(tf_test.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.num_client = flags.FLAGS.num_clients
    self.num_local_devices = flags.FLAGS.num_local_devices

    tf_config = json.loads(os.environ['TF_CONFIG'])
    self.client_id = int(tf_config['task']['index'])

    self.images = stateless_random_ops.stateless_random_uniform(
        [8, 8, 3], seed=(1, 2), minval=0, maxval=255)
    self.labels = stateless_random_ops.stateless_random_uniform(
        [1], seed=(1, 2), minval=0, maxval=10)

    self.dataset = dataset_ops.Dataset.from_tensors(
        (self.images, self.labels)).repeat()

  def test_create_batched_dataset(self):
    strategy = mwms.MultiWorkerMirroredStrategy()
    global_batch_size = self.num_client * self.num_local_devices * 2
    dataset = self.dataset.batch(global_batch_size).prefetch(2)

    distributed_dataset = strategy.experimental_distribute_dataset(dataset)
    element = next(iter(distributed_dataset))
    batched_image, batched_label = element
    self.assertEqual(batched_image.shape, [global_batch_size, 8, 8, 3])
    self.assertEqual(batched_label.shape, [global_batch_size, 1])

    # After unpack, it should only get the local shards.
    self.assertLen(d_api.unpack(batched_image), self.num_local_devices)
    self.assertLen(d_api.unpack(batched_label), self.num_local_devices)

  def test_uneven_batched_dataset(self):
    elements = [[1, 2, 3], [1, 2], [1, 2, 3, 4]]
    dataset = dataset_ops.Dataset.from_generator(
        lambda: elements, dtypes.int64).repeat()
    strategy = mwms.MultiWorkerMirroredStrategy()
    with self.assertRaisesRegex(ValueError, 'requires a static batch size'):
      strategy.experimental_distribute_dataset(dataset)

  def test_deprecated_strategy_methods(self):
    strategy = mwms.MultiWorkerMirroredStrategy()
    with self.assertRaisesRegex(
        NotImplementedError, 'only available in the V1 API'):
      strategy.make_dataset_iterator(self.dataset)

    with self.assertRaisesRegex(
        NotImplementedError, 'only available in the V1 API'):
      strategy.make_input_fn_iterator(lambda _: self.dataset)

  def test_distribute_dataset_from_fn(self):
    strategy = mwms.MultiWorkerMirroredStrategy()
    local_batch_size = 4
    global_batch_size = local_batch_size * strategy.num_replicas_in_sync

    def dataset_fn(option):
      del option
      return dataset_ops.Dataset.from_tensors(
          (self.images, self.labels)).repeat().batch(
              local_batch_size, drop_remainder=True).prefetch(2)

    distributed_dataset = strategy.distribute_datasets_from_function(
        dataset_fn, None)
    iterator = iter(distributed_dataset)

    self.assertEqual(distributed_dataset.element_spec,
                     (tensor_spec.TensorSpec(shape=(global_batch_size, 8, 8, 3),
                                             dtype=dtypes.float32, name=None),
                      tensor_spec.TensorSpec(shape=(global_batch_size, 1),
                                             dtype=dtypes.float32, name=None)))
    self.assertEqual(distributed_dataset.element_spec, iterator.element_spec)

    batched_image, batched_label = next(iterator)
    self.assertEqual(batched_image.shape, [global_batch_size, 8, 8, 3])
    self.assertEqual(batched_label.shape, [global_batch_size, 1])

    # After unpack, it should only get the local shards.
    unpacked_images = d_api.unpack(batched_image)
    self.assertLen(unpacked_images, self.num_local_devices)
    for i in range(self.num_local_devices):
      self.assertEqual(unpacked_images[i].shape, [local_batch_size, 8, 8, 3])

  def test_distribute_values_from_function(self):
    array_value = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    def value_fn(ctx):
      return array_value[ctx.replica_id_in_sync_group]
    strategy = mwms.MultiWorkerMirroredStrategy()
    distributed_values = (
        strategy.experimental_distribute_values_from_function(
            value_fn))

    self.assertEqual(d_api.fetch_layout(distributed_values),
                     layout.Layout.batch_sharded(
                         strategy.mesh, batch_dim='batch', rank=1))
    unpacked_value = d_api.unpack(distributed_values)
    self.assertLen(unpacked_value, self.num_local_devices)
    start = 1.0 + self.num_local_devices * self.client_id
    for i in range(self.num_local_devices):
      self.assertEqual(unpacked_value[i], start + i)

  def test_distribute_dataset_in_tf_function(self):
    strategy = mwms.MultiWorkerMirroredStrategy()
    local_batch_size = 4
    global_batch_size = local_batch_size * strategy.num_replicas_in_sync
    dataset = self.dataset.batch(global_batch_size).prefetch(2)

    distributed_dataset = strategy.experimental_distribute_dataset(dataset)

    @def_function.function
    def step_fn(iterator):
      images, labels = next(iterator)
      del labels
      return images

    result = strategy.run(step_fn, args=(iter(distributed_dataset),))
    self.assertIsInstance(result, dtensor_util.DTensorDistributedValue)
    self.assertLen(result.values, self.num_local_devices)
    for i in range(self.num_local_devices):
      self.assertEqual(result.values[i].shape, [local_batch_size, 8, 8, 3])


def client_config_function(config_params):
  client_id = config_params['client_id']
  worker_jobs = config_params['worker_jobs']
  num_devices = config_params['num_devices']

  os.environ['TF_CONFIG'] = json.dumps({
      'cluster': {
          'worker': worker_jobs
      },
      'task': {'type': 'worker', 'index': f'{client_id}'}
  })

  if config.list_physical_devices('GPU'):
    device_type = 'GPU'
  elif test_util.is_tpu_present():
    device_type = 'TPU'
  else:
    device_type = 'CPU'

  # reset_logical_devices
  test_util.reset_context()
  if device_type != 'TPU':
    # Configure virtual devices. This does not initialize the TensorFlow
    # context.
    test_util.reset_logical_devices(device_type, num_devices)

  # Validates the correct number of devices are created.
  logical_devices = test_util.list_local_logical_devices(device_type)
  assert len(logical_devices) == num_devices, (
      logical_devices,
      f'Test is mis-configured: expecting {num_devices} logical_devices.')


if __name__ == '__main__':
  test_backend_util.handle_test_main(
      multi_client_test_util.multi_client_main, client_config_function)
