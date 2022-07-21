# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for TPUStrategy."""

import os
from tensorflow.python.checkpoint import checkpoint as util
from tensorflow.python.checkpoint import checkpoint_management
from tensorflow.python.distribute import distribution_strategy_context
from tensorflow.python.distribute import strategy_test_lib
from tensorflow.python.distribute import tpu_strategy as tpu_lib
from tensorflow.python.distribute import tpu_values
from tensorflow.python.distribute.cluster_resolver import tpu_cluster_resolver
from tensorflow.python.eager import def_function
from tensorflow.python.eager import remote
from tensorflow.python.eager import test
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.module import module
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import summary_ops_v2 as summary_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import flags
from tensorflow.python.tpu import device_assignment as device_assignment_lib
from tensorflow.python.tpu import tpu
from tensorflow.python.tpu import tpu_strategy_util

FLAGS = flags.FLAGS
flags.DEFINE_string("tpu", "", "Name of TPU to connect to.")
flags.DEFINE_string("project", None, "Name of GCP project with TPU.")
flags.DEFINE_string("zone", None, "Name of GCP zone with TPU.")


def get_tpu_cluster_resolver():
  resolver = tpu_cluster_resolver.TPUClusterResolver(
      tpu=FLAGS.tpu,
      zone=FLAGS.zone,
      project=FLAGS.project,
  )
  return resolver


def get_tpu_strategy(enable_spmd=False):
  resolver = get_tpu_cluster_resolver()
  remote.connect_to_cluster(resolver)
  topology = tpu_strategy_util.initialize_tpu_system(resolver)
  num_replicas = resolver.get_tpu_system_metadata().num_cores // 2
  device_assignment = device_assignment_lib.DeviceAssignment.build(
      topology, num_replicas=num_replicas, computation_shape=[1, 1, 1, 2])
  strategy = tpu_lib.TPUStrategyV2(
      resolver,
      experimental_device_assignment=device_assignment,
      experimental_spmd_xla_partitioning=enable_spmd)
  return strategy, num_replicas


class TPUStrategyModelParallelismTest(
    strategy_test_lib.DistributionTestBase,
    strategy_test_lib.TwoDeviceDistributionTestBase):

  def test_logical_device_assignment(self):
    strategy, num_replicas = get_tpu_strategy()
    with strategy.scope():
      v = variables.Variable(2.)
      with strategy.extended.experimental_logical_device(1):
        w = variables.Variable(3.)

    self.assertLen(strategy.experimental_local_results(v), num_replicas)
    self.assertLen(strategy.experimental_local_results(w), num_replicas)
    self.assertEqual("/job:localhost/replica:0/task:0/device:TPU:0",
                     strategy.experimental_local_results(v)[0].device)
    self.assertEqual("/job:localhost/replica:0/task:0/device:TPU:1",
                     strategy.experimental_local_results(w)[0].device)

    logical_devices = []

    @def_function.function
    def f(x):
      replica_ctx = distribution_strategy_context.get_replica_context()
      with replica_ctx.experimental_logical_device(0):
        y = v * x
      with replica_ctx.experimental_logical_device(1):
        z = w * y
      logical_devices.append((y.device, z.device))
      return z

    result = strategy.run(f, args=(5.,))

    self.assertEqual(
        [("/device:TPU_REPLICATED_CORE:0", "/device:TPU_REPLICATED_CORE:1")],
        logical_devices)

    with self.cached_session():
      self.evaluate(variables.global_variables_initializer())
      self.assertEqual(30. * num_replicas,
                       self.evaluate(strategy.reduce("SUM", result, axis=None)))

  def test_paritioned_model_checkpointing(self):

    class PartitionedModel(module.Module):

      def __init__(self, v, w):
        super(PartitionedModel, self).__init__()

        assert distribution_strategy_context.has_strategy()
        strategy = distribution_strategy_context.get_strategy()

        with strategy.extended.experimental_logical_device(0):
          self.v = variables.Variable(v)
        with strategy.extended.experimental_logical_device(1):
          self.w = variables.Variable(w)

      def __call__(self, x):
        replica_ctx = distribution_strategy_context.get_replica_context()
        with replica_ctx.experimental_logical_device(0):
          y = self.v * x
        with replica_ctx.experimental_logical_device(1):
          z = self.w * y
        return z

      def change_weights_op(self, v_new, w_new):
        return control_flow_ops.group(
            [self.v.assign(v_new), self.w.assign(w_new)])

    strategy, num_replicas = get_tpu_strategy()
    with strategy.scope():
      model = PartitionedModel(2., 3.)

    checkpoint_dir = self.get_temp_dir()
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = util.Checkpoint(model=model)

    with self.cached_session() as sess:
      self.evaluate(variables.global_variables_initializer())
      checkpoint.save(file_prefix=checkpoint_prefix)

      self.evaluate(model.change_weights_op(1., 4.))
      result = strategy.run(def_function.function(model), args=(5.0,))
      self.assertEqual(20. * num_replicas,
                       self.evaluate(strategy.reduce("SUM", result, axis=None)))

      status = checkpoint.restore(
          checkpoint_management.latest_checkpoint(checkpoint_dir))
      status.run_restore_ops(sess)  # must run restore op in non-eager mode.
      status.assert_consumed()
      status.assert_existing_objects_matched()
      result = strategy.run(def_function.function(model), args=(5.0,))
      self.assertEqual(30. * num_replicas,
                       self.evaluate(strategy.reduce("SUM", result, axis=None)))

  def test_spmd_cannot_assign_tensor_to_logical_device(self):
    strategy, _ = get_tpu_strategy(enable_spmd=True)
    x = constant_op.constant([0, 1])
    with self.assertRaises(ValueError):
      strategy.experimental_assign_to_logical_device(x, 0)

  def test_spmd_variable_created_from_callable(self):
    initilizer = lambda: random_ops.random_normal(shape=(16, 16))
    strategy, _ = get_tpu_strategy(enable_spmd=True)
    with strategy.scope():
      w = variables.Variable(initilizer)
    value0 = w.values[0]
    for v in value0.variables:
      self.assertAllEqual(v, value0.variables[0])

  def test_spmd_variable_read(self):
    batch_size = 32
    num_feature_in = 16
    num_feature_out = 8

    x = random_ops.random_uniform((batch_size, num_feature_in),
                                  dtype=dtypes.float32)
    w_init = random_ops.random_uniform((num_feature_in, num_feature_out),
                                       dtype=dtypes.float32)

    strategy, num_replicas = get_tpu_strategy(enable_spmd=True)
    with strategy.scope():
      w = variables.Variable(w_init, dtype=dtypes.float32)

    self.assertEqual(w.values[0].variables[0].shape.as_list(),
                     [num_feature_in, num_feature_out])

    self.assertEqual(w.shape.as_list(), [num_feature_in, num_feature_out])

    def step_fn(batch_features):
      predict = math_ops.matmul(batch_features, w)
      return predict

    @def_function.function
    def train_fn(batch_features):
      return strategy.run(step_fn, args=(batch_features,))

    result = train_fn(x)
    self.assertAllClose(
        strategy.reduce("SUM", result, axis=None),
        math_ops.matmul(x, w_init) * num_replicas,
        rtol=5e-03,
        atol=5e-03)

  def test_spmd_variable_update(self):
    batch_size = 1024
    num_feature_in = 256

    x = random_ops.random_uniform((batch_size, num_feature_in),
                                  dtype=dtypes.float32)
    w_init = random_ops.random_uniform((batch_size, num_feature_in),
                                       dtype=dtypes.float32)

    strategy, num_replicas = get_tpu_strategy(enable_spmd=True)
    with strategy.scope():
      w = variables.Variable(w_init, dtype=dtypes.float32)

    self.assertIsInstance(w, tpu_values.TPUMirroredVariable)
    self.assertTrue(w._is_replicated_or_sharded_to_logical_cores())

    def make_strategy_run(fn):

      def run(value):
        return strategy.run(fn, args=(value,))

      return def_function.function(run)

    result = make_strategy_run(w.assign)(x)
    self.assertAllClose(
        strategy.reduce("SUM", result, axis=None), x * num_replicas)

    delta = random_ops.random_uniform((batch_size, num_feature_in),
                                      dtype=dtypes.float32)
    result = make_strategy_run(w.assign_sub)(delta)
    x -= delta
    self.assertAllClose(
        strategy.reduce("SUM", result, axis=None), x * num_replicas)

    delta = random_ops.random_uniform((batch_size, num_feature_in),
                                      dtype=dtypes.float32)
    result = make_strategy_run(w.assign_add)(delta)
    x += delta
    self.assertAllClose(
        strategy.reduce("SUM", result, axis=None), x * num_replicas)

  def test_spmd_variable_eager_update(self):
    batch_size = 32
    num_feature_in = 16

    x = random_ops.random_uniform((batch_size, num_feature_in),
                                  dtype=dtypes.float32)
    w_init = random_ops.random_uniform((batch_size, num_feature_in),
                                       dtype=dtypes.float32)

    strategy, _ = get_tpu_strategy(enable_spmd=True)
    with strategy.scope():
      w = variables.Variable(w_init, dtype=dtypes.float32)

    w.assign(x)
    result = w.numpy()
    self.assertAllClose(result, x)

    x1 = random_ops.random_uniform((batch_size, num_feature_in),
                                   dtype=dtypes.float32)
    w.assign_sub(x1)
    result = w.numpy()
    self.assertAllClose(result, x - x1)

    x2 = random_ops.random_uniform((batch_size, num_feature_in),
                                   dtype=dtypes.float32)
    w.assign(x)
    w.assign_add(x2)
    result = w.numpy()
    self.assertAllClose(result, x + x2)

  def test_spmd_model_checkpointing(self):

    class LinearModel(module.Module):

      def __init__(self, w):
        super(LinearModel, self).__init__()
        self.w = variables.Variable(w)

      def __call__(self, x):
        return math_ops.matmul(x, self.w)

      def change_weights_op(self, w_new):
        return self.w.assign(w_new)

    batch_size = 32
    num_feature_in = 16
    num_feature_out = 8
    w1 = random_ops.random_uniform((num_feature_in, num_feature_out),
                                   dtype=dtypes.float32)
    w2 = random_ops.random_uniform((num_feature_in, num_feature_out),
                                   dtype=dtypes.float32)
    x = random_ops.random_uniform((batch_size, num_feature_in),
                                  dtype=dtypes.float32)

    strategy, num_replicas = get_tpu_strategy(enable_spmd=True)
    with strategy.scope():
      model = LinearModel(w1)

    checkpoint_dir = self.get_temp_dir()
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = util.Checkpoint(model=model)

    @def_function.function
    def step_fn(x):
      x = strategy.experimental_split_to_logical_devices(x, [1, 2])
      return model(x)

    with self.cached_session() as sess:
      self.evaluate(variables.global_variables_initializer())
      checkpoint.save(file_prefix=checkpoint_prefix)

      self.evaluate(model.change_weights_op(w2))
      result = strategy.run(step_fn, args=(x,))
      self.assertAllClose(
          math_ops.matmul(x, w2) * num_replicas,
          self.evaluate(strategy.reduce("SUM", result, axis=None)),
          rtol=5e-3,
          atol=5e-3)

      status = checkpoint.restore(
          checkpoint_management.latest_checkpoint(checkpoint_dir))
      status.run_restore_ops(sess)  # must run restore op in non-eager mode.
      status.assert_consumed()
      status.assert_existing_objects_matched()
      result = strategy.run(step_fn, args=(x,))
      self.assertAllClose(
          math_ops.matmul(x, w1) * num_replicas,
          self.evaluate(strategy.reduce("SUM", result, axis=None)),
          rtol=5e-3,
          atol=5e-3)

  def test_spmd_with_summary(self):
    original_device_placement = config.get_soft_device_placement()
    config.set_soft_device_placement(True)

    strategy, _ = get_tpu_strategy(enable_spmd=True)
    summary_dir = self.get_temp_dir()
    writer = summary_ops.create_file_writer_v2(summary_dir)

    with strategy.scope():
      step = variables.Variable(0, dtype=dtypes.int64)

    @def_function.function
    def run():
      with writer.as_default():
        summary_ops.scalar("result", step * 2, step=step)
        step.assign_add(1)

    for _ in range(10):
      strategy.run(run, args=())

    for val in step.values:
      for var in val.variables:
        self.assertAllEqual(10, var)

    config.set_soft_device_placement(original_device_placement)

  def test_spmd_with_outside_comp(self):
    strategy, num_replicas = get_tpu_strategy(enable_spmd=True)

    def host_inc(x):
      return x + 1

    @def_function.function
    def fn(x):
      y = x + 1
      z = tpu.outside_compilation(host_inc, y)
      a = z + 1
      return a

    arg = constant_op.constant(0, shape=(), dtype=dtypes.int64)
    result = strategy.run(fn, args=(arg,))
    self.assertEqual(3 * num_replicas,
                     self.evaluate(strategy.reduce("SUM", result, axis=None)))

if __name__ == "__main__":
  test.main()
