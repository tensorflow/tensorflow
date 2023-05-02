# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for parameter_server_strategy_v2.py."""

import contextlib
import functools
import os

os.environ["TF_NUM_INTEROP_THREADS"]="16"

from absl.testing import parameterized
import numpy as np

from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.python.checkpoint import checkpoint as tracking_util
from tensorflow.python.compat import v2_compat
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import multi_process_runner
from tensorflow.python.distribute import multi_worker_test_base
from tensorflow.python.distribute import parameter_server_strategy_v2
from tensorflow.python.distribute import ps_values
from tensorflow.python.distribute import sharded_variable
from tensorflow.python.distribute.cluster_resolver import SimpleClusterResolver
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.module import module
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import init_ops_v2
from tensorflow.python.ops import linalg_ops_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import gfile
from tensorflow.python.saved_model import save as tf_save
from tensorflow.python.trackable import autotrackable
from tensorflow.python.training.server_lib import ClusterSpec


class ParameterServerStrategyV2Test(test.TestCase):

  @classmethod
  def setUpClass(cls):
    super(ParameterServerStrategyV2Test, cls).setUpClass()
    cls.cluster = multi_worker_test_base.create_multi_process_cluster(
        num_workers=2, num_ps=3, rpc_layer="grpc")
    cls.cluster_resolver = cls.cluster.cluster_resolver

  @classmethod
  def tearDownClass(cls):
    super(ParameterServerStrategyV2Test, cls).tearDownClass()
    cls.cluster.stop()

  def testVariablePlacement(self):

    strategy = parameter_server_strategy_v2.ParameterServerStrategyV2(
        self.cluster_resolver)
    v1 = variables.Variable(initial_value=0.0)
    with strategy.scope():
      v2 = variables.Variable(initial_value=1.0)
      v3 = variables.Variable(initial_value=2.0)
      v4 = variables.Variable(initial_value=3.0)
      v5 = variables.Variable(initial_value=4.0)
    # v1 was created outside scope so should be on client.
    gpu_devices = context.num_gpus()
    if gpu_devices:
      # For tests with GPUs
      self.assertEqual(v1.device, "/job:chief/replica:0/task:0/device:GPU:0")
    else:
      self.assertEqual(v1.device, "/job:chief/replica:0/task:0/device:CPU:0")
    # v2 through v5 are created in scope and in a round-robin manner.
    self.assertEqual(v2.device, "/job:ps/replica:0/task:0/device:CPU:0")
    self.assertEqual(v3.device, "/job:ps/replica:0/task:1/device:CPU:0")
    self.assertEqual(v4.device, "/job:ps/replica:0/task:2/device:CPU:0")
    self.assertEqual(v5.device, "/job:ps/replica:0/task:0/device:CPU:0")

  def testInteractionWithDeviceScope(self):
    strategy = parameter_server_strategy_v2.ParameterServerStrategyV2(
        self.cluster_resolver)

    # The strategy scope always wins.
    with strategy.scope():
      with ops.device("/job:ps/replica:0/task:1"):
        v0 = variables.Variable(initial_value=0.0)
      self.assertEqual(v0.device, "/job:ps/replica:0/task:0/device:CPU:0")

      with ops.device("/job:ps/replica:0/task:0"):
        v1 = variables.Variable(initial_value=0.0)
      self.assertEqual(v1.device, "/job:ps/replica:0/task:1/device:CPU:0")

    with ops.device("/job:ps/replica:0/task:1"):
      with strategy.scope():
        v2 = variables.Variable(initial_value=0.0)
        self.assertEqual(v2.device, "/job:ps/replica:0/task:2/device:CPU:0")

        v3 = variables.Variable(initial_value=0.0)
        self.assertEqual(v3.device, "/job:ps/replica:0/task:0/device:CPU:0")

  def testInteractionWithVariableCreatorScope(self):

    def var_creator(next_creator, **kwargs):
      if "colocate_with" in kwargs:
        with ops.device(None):
          with ops.colocate_with(kwargs["colocate_with"]):
            return next_creator(**kwargs)

      self.assertIn("ps1", kwargs["name"])
      with ops.device("/job:ps/task:1"):
        return next_creator(**kwargs)

    strategy = parameter_server_strategy_v2.ParameterServerStrategyV2(
        self.cluster_resolver)

    # variable_creator_scope itself will work.
    with variable_scope.variable_creator_scope(var_creator):
      v0 = variables.Variable(initial_value=0.0, name="ps1_0")
    self.assertEqual(v0.device, "/job:ps/replica:0/task:1/device:CPU:0")

    # variable_creator_scope inside strategy.scope will not work.
    with strategy.scope():
      with variable_scope.variable_creator_scope(var_creator):
        v1 = variables.Variable(initial_value=0.0, name="ps1_1")
    self.assertEqual(v1.device, "/job:ps/replica:0/task:0/device:CPU:0")

    # strategy.scope still assigns variables in a round robin fashion.
    with strategy.scope():
      v2 = variables.Variable(initial_value=0.0, name="ps1_2")
    self.assertEqual(v2.device, "/job:ps/replica:0/task:1/device:CPU:0")

    with strategy.scope():
      v3 = variables.Variable(initial_value=0.0, name="ps1_3")
    self.assertEqual(v3.device, "/job:ps/replica:0/task:2/device:CPU:0")

    # variable_creator_scope outside strategy.scope will work.
    with variable_scope.variable_creator_scope(var_creator):
      with strategy.scope():
        v4 = variables.Variable(initial_value=0.0, name="ps1_4")
    self.assertEqual(v4.device, "/job:ps/replica:0/task:1/device:CPU:0")

    with variable_scope.variable_creator_scope(var_creator):
      with strategy.scope():
        v5 = variables.Variable(initial_value=0.0, name="ps1_5")
    self.assertEqual(v5.device, "/job:ps/replica:0/task:1/device:CPU:0")

    # variable_creator_scope can be made to respect "colocate_with" as well.
    with variable_scope.variable_creator_scope(var_creator):
      with strategy.scope():
        with strategy.extended.colocate_vars_with(v1):
          v6 = variables.Variable(initial_value=0.0, name="ps1_6")
    self.assertEqual(v6.device, "/job:ps/replica:0/task:0/device:CPU:0")

  @contextlib.contextmanager
  def _assertRaisesUsageWarningWithSchedule(self):
    with self.assertLogs(level="WARNING") as logs:
      yield

    self.assertIn(
        "A `tf.distribute.experimental.ParameterServerStrategy` method is "
        "invoked without using `ClusterCoordinator.schedule`. If you are not "
        "tracing a tf.function, this method is possibly executed on the "
        "coordinator, which can be slow. To properly dispatch functions to "
        "run on workers, methods like `run` or `reduce` should be used "
        "within a function passed to `tf.distribute.experimental.coordinator."
        "ClusterCoordinator.schedule`.", "".join(logs.output))

  def testRunNotUsedWithClusterCoordinator(self):
    strategy = parameter_server_strategy_v2.ParameterServerStrategyV2(
        self.cluster_resolver)
    dataset = dataset_ops.DatasetV2.range(8)
    with strategy.scope():
      v = variables.Variable(1, dtype=dtypes.int64)

    def step_fn(iterator):
      return next(iterator) + v

    with self._assertRaisesUsageWarningWithSchedule():
      strategy.run(step_fn, args=(iter(dataset),))

  def testRunUsedWithTestOnlyMode(self):
    strategy = parameter_server_strategy_v2.ParameterServerStrategyV2(
        self.cluster_resolver)
    strategy.extended._allow_run_without_coordinator = True
    dataset = dataset_ops.DatasetV2.range(15)
    with strategy.scope():
      v = variables.Variable(1, dtype=dtypes.int64)

    def step_fn(iterator):
      return next(iterator) + v

    strategy.run(step_fn, args=(iter(dataset),))

  def testReduceNotUsedWithClusterCoordinator(self):
    strategy = parameter_server_strategy_v2.ParameterServerStrategyV2(
        self.cluster_resolver)
    with self._assertRaisesUsageWarningWithSchedule():
      strategy.reduce("SUM", None, axis=None)

  def testDistributeDatasetUsedDirectly(self):
    strategy = parameter_server_strategy_v2.ParameterServerStrategyV2(
        self.cluster_resolver)
    dataset = dataset_ops.DatasetV2.range(3)
    distributed_dataset = strategy.experimental_distribute_dataset(dataset)
    with self.assertRaises(ValueError):
      iter(distributed_dataset)

    distributed_dataset = strategy.distribute_datasets_from_function(
        lambda: dataset)
    with self.assertRaises(ValueError):
      iter(distributed_dataset)

  def testSparselyReadForEmbeddingLookup(self):
    strategy = parameter_server_strategy_v2.ParameterServerStrategyV2(
        self.cluster_resolver)

    class FakeModel(module.Module):

      def __init__(self):
        self._var0 = variables.Variable([1.0, 2.0, 3.0, 4.0])
        self._var1 = variables.Variable([5.0, 6.0, 7.0, 8.0])

      @def_function.function(input_signature=[
          tensor_spec.TensorSpec(shape=[2], dtype=dtypes.int32, name="inputs")
      ])
      def func(self, x):
        return embedding_ops.embedding_lookup([self._var0, self._var1], x)

    with strategy.scope():
      model = FakeModel()

    # Assert that ResourceGather op exists instead of Gather in training
    # function.
    found_resource_gather = False
    found_gather = False

    for n in model.func.get_concrete_function().graph.as_graph_def().node:
      if n.op == "ResourceGather":
        found_resource_gather = True
      elif n.op == "Gather":
        found_gather = True
    self.assertTrue(found_resource_gather)
    self.assertFalse(found_gather)

    # Assert that ResourceGather op exists instead of Gather in saved_model.
    found_resource_gather = False
    found_gather = False

    tmp_dir = self.get_temp_dir()
    tf_save.save(model, tmp_dir, signatures=model.func)

    with gfile.Open("%s/saved_model.pb" % tmp_dir, "rb") as f:
      saved_model_proto = saved_model_pb2.SavedModel().FromString(f.read())

    for function in saved_model_proto.meta_graphs[0].graph_def.library.function:
      for n in function.node_def:
        if n.op == "ResourceGather":
          found_resource_gather = True
          resource_gather_device = n.device
        elif n.op == "Gather":
          found_gather = True
    self.assertTrue(found_resource_gather)
    self.assertFalse(found_gather)

    # We also assert that the colocate_with in embedding_ops will not result in
    # a hard-coded device string.
    self.assertEmpty(resource_gather_device)


class PartitionAwareIdentity(object):

  def __call__(self, shape, dtype, **kwargs):
    value = linalg_ops_impl.eye(*shape, dtype=dtype)
    if "partition_shape" in kwargs and "partition_offset" in kwargs:
      return array_ops.slice(value, kwargs["partition_offset"],
                             kwargs["partition_shape"])
    raise AssertionError("PartitionAwareIdentity do not support "
                         "non-partitioned initialization")


class VariablePartitioningTest(test.TestCase, parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    super(VariablePartitioningTest, cls).setUpClass()
    cls.cluster = multi_worker_test_base.create_multi_process_cluster(
        num_workers=2, num_ps=2, rpc_layer="grpc")
    cls.cluster_resolver = cls.cluster.cluster_resolver

  @classmethod
  def tearDownClass(cls):
    super(VariablePartitioningTest, cls).tearDownClass()
    cls.cluster.stop()

  def testDefaultNoPartition(self):
    strategy = parameter_server_strategy_v2.ParameterServerStrategyV2(
        self.cluster_resolver)
    with strategy.scope():
      v = variables.Variable([0, 1, 2, 3])

    self.assertIsInstance(v, variables.Variable)

  def testBasic(self):
    strategy = parameter_server_strategy_v2.ParameterServerStrategyV2(
        self.cluster_resolver, sharded_variable.FixedShardsPartitioner(2))
    with strategy.scope():
      init1 = init_ops_v2.Constant([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
      v1 = variables.Variable(
          initial_value=lambda: init1(shape=(5, 2), dtype=dtypes.int64),
          shape=(5, 2),
          dtype=dtypes.int64)

      init2 = init_ops_v2.Constant([0, 1, 2, 3, 4, 5])
      v2 = variables.Variable(
          initial_value=lambda: init2(shape=(6, 1), dtype=dtypes.int64),
          shape=(6, 1),
          dtype=dtypes.int64)

    self.assertIsInstance(v1, sharded_variable.ShardedVariable)
    self.assertLen(v1.variables, 2)
    self.assertRegex(v1.variables[0].device, "/job:ps/replica:0/task:0")
    self.assertRegex(v1.variables[1].device, "/job:ps/replica:0/task:1")
    self.assertAllEqual(v1.variables[0], [[0, 1], [2, 3], [4, 5]])
    self.assertAllEqual(v1.variables[1], [[6, 7], [8, 9]])

    self.assertIsInstance(v2, sharded_variable.ShardedVariable)
    self.assertLen(v2.variables, 2)
    self.assertRegex(v2.variables[0].device, "/job:ps/replica:0/task:0")
    self.assertRegex(v2.variables[1].device, "/job:ps/replica:0/task:1")
    self.assertAllEqual(v2.variables[0], [[0], [1], [2]])
    self.assertAllEqual(v2.variables[1], [[3], [4], [5]])

  def testBasicVariableWithAggregation(self):
    strategy = parameter_server_strategy_v2.ParameterServerStrategyV2(
        self.cluster_resolver)
    strategy.extended._allow_run_without_coordinator = True
    with strategy.scope():
      v = variables.Variable(
          initial_value=[0, 0, 0, 0, 0, 0, 0, 0],
          dtype=dtypes.float32,
          aggregation=variable_scope.VariableAggregation.SUM)

    if strategy.num_replicas_in_sync > 1:
      self.assertIsInstance(v, ps_values.AggregatingVariable)
    else:
      self.assertIsInstance(v, variables.Variable)

    def replica_fn():
      replica_id = distribute_lib.get_replica_context(
      ).replica_id_in_sync_group
      val = array_ops.reshape(
          math_ops.cast(replica_id + 10, dtype=v.dtype), [1])
      v.assign(
          array_ops.concat(
              [val, constant_op.constant([1., 2., 3., 4., 5., 6., 7.])], 0))

    strategy.run(replica_fn)

    expected_result = np.arange(8.) * strategy.num_replicas_in_sync
    for i in range(strategy.num_replicas_in_sync):
      expected_result[0] = expected_result[0] + i + 10
    self.assertAllEqual(v, expected_result)

  def testBasicShardedVariableWithAggregation(self):
    strategy = parameter_server_strategy_v2.ParameterServerStrategyV2(
        self.cluster_resolver, sharded_variable.FixedShardsPartitioner(2))
    strategy.extended._allow_run_without_coordinator = True
    with strategy.scope():
      v = variables.Variable(
          initial_value=[0, 0, 0, 0, 0, 0, 0, 0],
          dtype=dtypes.float32,
          aggregation=variable_scope.VariableAggregation.SUM)

    self.assertIsInstance(v, sharded_variable.ShardedVariable)
    self.assertLen(v.variables, 2)
    if strategy.num_replicas_in_sync > 1:
      self.assertIsInstance(v.variables[0], ps_values.AggregatingVariable)
    else:
      self.assertIsInstance(v.variables[0], variables.Variable)

    def replica_fn():
      replica_id = distribute_lib.get_replica_context(
      ).replica_id_in_sync_group
      val = array_ops.reshape(
          math_ops.cast(replica_id + 10, dtype=v.dtype), [1])
      v.assign(
          array_ops.concat(
              [val, constant_op.constant([1., 2., 3., 4., 5., 6., 7.])], 0))

    strategy.run(replica_fn)

    expected_result = np.arange(8.) * strategy.num_replicas_in_sync
    for i in range(strategy.num_replicas_in_sync):
      expected_result[0] = expected_result[0] + i + 10
    expected_result = np.array_split(expected_result, 2)
    self.assertAllEqual(expected_result[0], v.variables[0])
    self.assertAllEqual(expected_result[1], v.variables[1])

  def testNonCallableInitialValue(self):
    strategy = parameter_server_strategy_v2.ParameterServerStrategyV2(
        self.cluster_resolver, sharded_variable.FixedShardsPartitioner(4))
    with strategy.scope():
      v = variables.Variable([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    self.assertIsInstance(v, sharded_variable.ShardedVariable)
    self.assertLen(v.variables, 4)
    self.assertRegex(v.variables[0].device, "/job:ps/replica:0/task:0")
    self.assertRegex(v.variables[1].device, "/job:ps/replica:0/task:1")
    self.assertRegex(v.variables[2].device, "/job:ps/replica:0/task:0")
    self.assertRegex(v.variables[3].device, "/job:ps/replica:0/task:1")
    self.assertAllEqual(v.variables[0], [0, 1, 2])
    self.assertAllEqual(v.variables[1], [3, 4, 5])
    self.assertAllEqual(v.variables[2], [6, 7])
    self.assertAllEqual(v.variables[3], [8, 9])

  def testNumPartitionsLargerThanSize(self):
    strategy = parameter_server_strategy_v2.ParameterServerStrategyV2(
        self.cluster_resolver, sharded_variable.FixedShardsPartitioner(4))
    with strategy.scope():
      v = variables.Variable([0, 1, 2])

    self.assertIsInstance(v, sharded_variable.ShardedVariable)
    self.assertLen(v.variables, 3)
    self.assertRegex(v.variables[0].device, "/job:ps/replica:0/task:0")
    self.assertRegex(v.variables[1].device, "/job:ps/replica:0/task:1")
    self.assertRegex(v.variables[2].device, "/job:ps/replica:0/task:0")
    self.assertAllEqual(v.variables[0], [0])
    self.assertAllEqual(v.variables[1], [1])
    self.assertAllEqual(v.variables[2], [2])

  def testPartitionToOne(self):
    # For small variables there is only one partition.
    variable_partitioner = sharded_variable.MinSizePartitioner(
        min_shard_bytes=64 << 20, max_shards=2)
    strategy = parameter_server_strategy_v2.ParameterServerStrategyV2(
        self.cluster_resolver, variable_partitioner)
    with strategy.scope():
      initializer = init_ops_v2.Constant([0] * 10)
      v1 = variables.Variable(
          initial_value=lambda: initializer(shape=(10,), dtype=dtypes.int64),
          shape=(10,),
          dtype=dtypes.int64)

      v2 = variables.Variable([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    self.assertIsInstance(v1, variables.Variable)
    self.assertNotIsInstance(v1, sharded_variable.ShardedVariable)
    self.assertRegex(v1.device, "/job:ps/replica:0/task:0")
    self.assertAllEqual(v1, [0] * 10)

    self.assertIsInstance(v2, variables.Variable)
    self.assertNotIsInstance(v2, sharded_variable.ShardedVariable)
    self.assertRegex(v2.device, "/job:ps/replica:0/task:1")
    self.assertAllEqual(v2, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

  def testColocateWith(self):
    strategy = parameter_server_strategy_v2.ParameterServerStrategyV2(
        self.cluster_resolver, sharded_variable.FixedShardsPartitioner(2))
    with strategy.scope():
      v1 = variables.Variable([0, 1, 2, 3])

      with strategy.extended.colocate_vars_with(v1.variables[0]):
        v2 = variables.Variable([4, 5])

    self.assertIsInstance(v1, sharded_variable.ShardedVariable)

    self.assertIsInstance(v2, variables.Variable)
    self.assertNotIsInstance(v2, sharded_variable.ShardedVariable)
    self.assertEqual(v2.device, v1.variables[0].device)
    self.assertAllEqual(v2, [4, 5])

  def testCustomPartitionAwareInitializer(self):
    strategy = parameter_server_strategy_v2.ParameterServerStrategyV2(
        self.cluster_resolver, sharded_variable.FixedShardsPartitioner(2))
    with strategy.scope():
      initializer = PartitionAwareIdentity()
      initial_value = functools.partial(
          initializer, shape=(4, 4), dtype=dtypes.int64)
      v = variables.Variable(
          initial_value=initial_value, shape=(4, 4), dtype=dtypes.int64)

    self.assertIsInstance(v, sharded_variable.ShardedVariable)
    self.assertLen(v.variables, 2)
    self.assertRegex(v.variables[0].device, "/job:ps/replica:0/task:0")
    self.assertRegex(v.variables[1].device, "/job:ps/replica:0/task:1")
    self.assertAllEqual(v.variables[0], [[1, 0, 0, 0], [0, 1, 0, 0]])
    self.assertAllEqual(v.variables[1], [[0, 0, 1, 0], [0, 0, 0, 1]])

  def testPartitionWhenLackOfInfo(self):
    strategy = parameter_server_strategy_v2.ParameterServerStrategyV2(
        self.cluster_resolver, sharded_variable.FixedShardsPartitioner(2))
    with strategy.scope():
      initializer = init_ops_v2.Constant([0, 1, 2, 3])
      # Shape is not explicitly specified.
      v1 = variables.Variable(
          initial_value=lambda: initializer(shape=(4,), dtype=dtypes.int64),
          dtype=dtypes.int64)
      # Dtype is not explicitly specified.
      v2 = variables.Variable(
          initial_value=lambda: initializer(shape=(4,), dtype=dtypes.int64),
          shape=(4,))
      # Neither shape nor dtype is explicitly specified.
      v3 = variables.Variable(
          initial_value=lambda: initializer(shape=(4,), dtype=dtypes.int64))

    for v in [v1, v2, v3]:
      self.assertIsInstance(v, sharded_variable.ShardedVariable)
      self.assertLen(v.variables, 2)
      self.assertRegex(v.variables[0].device, "/job:ps/replica:0/task:0")
      self.assertRegex(v.variables[1].device, "/job:ps/replica:0/task:1")
      self.assertAllEqual(v.variables[0], [0, 1])
      self.assertAllEqual(v.variables[1], [2, 3])

  def testInvalidPartitioner(self):
    strategy = parameter_server_strategy_v2.ParameterServerStrategyV2(
        self.cluster_resolver, lambda shape, dtype: None)
    with self.assertRaisesRegex(ValueError, "variable_partitioner"):
      with strategy.scope():
        variables.Variable([[[0, 1], [2, 3]], [[0, 1], [2, 3]]])

    strategy = parameter_server_strategy_v2.ParameterServerStrategyV2(
        self.cluster_resolver, lambda shape, dtype: [])
    with self.assertRaisesRegex(ValueError, "variable_partitioner"):
      with strategy.scope():
        variables.Variable([[[0, 1], [2, 3]], [[0, 1], [2, 3]]])

    strategy = parameter_server_strategy_v2.ParameterServerStrategyV2(
        self.cluster_resolver, lambda shape, dtype: [0, 1, 1])
    with self.assertRaisesRegex(ValueError, "variable_partitioner"):
      with strategy.scope():
        variables.Variable([[[0, 1], [2, 3]], [[0, 1], [2, 3]]])

    strategy = parameter_server_strategy_v2.ParameterServerStrategyV2(
        self.cluster_resolver, lambda shape, dtype: [2, 2, 1])
    with self.assertRaisesRegex(ValueError, "variable_partitioner"):
      with strategy.scope():
        variables.Variable([[[0, 1], [2, 3]], [[0, 1], [2, 3]]])

  def testCreateInsideTFFunction(self):
    if test_util.is_xla_enabled():
      self.skipTest("TODO(b/202760274): Would raise an error that is to be "
                    "investigated.")

    strategy = parameter_server_strategy_v2.ParameterServerStrategyV2(
        self.cluster_resolver, sharded_variable.FixedShardsPartitioner(2))

    collection = []

    @def_function.function
    def create_vars():
      if not collection:
        identity = init_ops_v2.Identity()
        v1 = variables.Variable([[1., 0.], [0., 1.]], dtype=dtypes.float32)
        v2 = variables.Variable(lambda: identity((2, 2), dtypes.float32))
        v3 = variables.Variable(
            lambda: identity((2, 2), dtypes.float32),
            dtype=dtypes.float32,
            shape=(2, 2))
        collection.extend([v1, v2, v3])

    with strategy.scope():
      create_vars()
      for v in collection:
        self.assertIsInstance(v, sharded_variable.ShardedVariable)
        self.assertLen(v.variables, 2)
        self.assertRegex(v.variables[0].device, "/job:ps/replica:0/task:0")
        self.assertRegex(v.variables[1].device, "/job:ps/replica:0/task:1")
        self.assertAllEqual(v.variables[0], [[1., 0.]])
        self.assertAllEqual(v.variables[1], [[0., 1.]])

  @parameterized.named_parameters(
      ("Restore", False, 2),
      ("RestoreDiffShards", False, 4),
      ("DelayedRestore", True, 2),
      ("DelayedRestoreDiffShards", True, 4),
  )
  def testCheckpoint(self, delayed, restore_shards):

    if test_util.is_xla_enabled() and not delayed and restore_shards == 4:
      self.skipTest("TODO(b/202760274): Would raise an error that is to be "
                    "investigated.")

    def make_variable(name, shape, dtype, initializer):
      initial_value = functools.partial(initializer, shape, dtype=dtype)
      return variables.Variable(
          name=name, initial_value=initial_value, shape=shape, dtype=dtype)

    class Model(autotrackable.AutoTrackable):

      def build(self):
        self.w = self._add_variable_with_custom_getter(
            "w",
            shape=(4,),
            initializer=init_ops_v2.Ones(),
            getter=make_variable)

    strategy = parameter_server_strategy_v2.ParameterServerStrategyV2(
        self.cluster_resolver, sharded_variable.FixedShardsPartitioner(2))
    ckpt_dir = os.path.join(self.get_temp_dir(), "checkpoint")

    with strategy.scope():
      model1 = Model()
      model1.build()
      self.assertIsInstance(model1.w, sharded_variable.ShardedVariable)
      self.assertLen(model1.w.variables, 2)
      model1.w.assign([1., 2., 3., 4.])

      cp1 = tracking_util.Checkpoint(model=model1)
      cp1.write(ckpt_dir)

    strategy = parameter_server_strategy_v2.ParameterServerStrategyV2(
        self.cluster_resolver,
        sharded_variable.FixedShardsPartitioner(restore_shards))

    with strategy.scope():
      model2 = Model()
      cp2 = tracking_util.Checkpoint(model=model2)
      if delayed:
        cp2.restore(ckpt_dir)
        model2.build()
      else:
        model2.build()
        cp2.restore(ckpt_dir)
      self.assertIsInstance(model2.w, sharded_variable.ShardedVariable)
      self.assertLen(model2.w.variables, restore_shards)
      if restore_shards == 2:
        self.assertAllEqual(model2.w.variables[0], [1., 2.])
        self.assertAllEqual(model2.w.variables[1], [3., 4.])
      elif restore_shards == 4:
        self.assertAllEqual(model2.w.variables[0], [1.])
        self.assertAllEqual(model2.w.variables[1], [2.])
        self.assertAllEqual(model2.w.variables[2], [3.])
        self.assertAllEqual(model2.w.variables[3], [4.])


class ClusterTypeNameTest(test.TestCase):

  def testArbitraryJobName(self):
    cluster_def = multi_worker_test_base.create_cluster_spec(
        num_workers=1, num_ps=1, has_chief=True)
    cluster_def["some_arbitrary_name"] = [
        "localhost:%d" % multi_worker_test_base.pick_unused_port()
    ]
    cluster_resolver = SimpleClusterResolver(
        ClusterSpec(cluster_def), rpc_layer="grpc")
    with self.assertRaisesRegexp(ValueError, "Disallowed task type found in"):
      parameter_server_strategy_v2.ParameterServerStrategyV2(cluster_resolver)

  def testArbitraryCurrentTaskType(self):
    cluster_def = multi_worker_test_base.create_cluster_spec(
        num_workers=1, num_ps=1, has_chief=True)
    cluster_resolver = SimpleClusterResolver(
        ClusterSpec(cluster_def), rpc_layer="grpc", task_type="foobar")
    with self.assertRaisesRegexp(ValueError, "Unrecognized task_type: foobar"):
      parameter_server_strategy_v2.ParameterServerStrategyV2(cluster_resolver)

  def testMoreThanOneChief(self):
    cluster_def = multi_worker_test_base.create_cluster_spec(
        num_workers=1, num_ps=1)
    chief_ports = [multi_worker_test_base.pick_unused_port() for _ in range(3)]
    cluster_def["chief"] = ["localhost:%s" % port for port in chief_ports]
    cluster_resolver = SimpleClusterResolver(
        ClusterSpec(cluster_def),
        rpc_layer="grpc",
        task_type="chief",
        task_id=1)
    with self.assertRaisesRegexp(ValueError,
                                 "There must be at most one 'chief' job."):
      parameter_server_strategy_v2.ParameterServerStrategyV2(cluster_resolver)

  def testLessThanOneWorker(self):
    cluster_def = multi_worker_test_base.create_cluster_spec(
        num_workers=0, num_ps=1, has_chief=True)
    cluster_resolver = SimpleClusterResolver(
        ClusterSpec(cluster_def), rpc_layer="grpc", task_type="ps", task_id=0)
    with self.assertRaisesRegexp(ValueError,
                                 "There must be at least one worker."):
      parameter_server_strategy_v2.ParameterServerStrategyV2(cluster_resolver)

  def testLessThanOnePs(self):
    cluster_def = multi_worker_test_base.create_cluster_spec(
        num_workers=1, num_ps=0, has_chief=True)
    cluster_resolver = SimpleClusterResolver(
        ClusterSpec(cluster_def),
        rpc_layer="grpc",
        task_type="worker",
        task_id=0)
    with self.assertRaisesRegexp(ValueError, "There must be at least one ps."):
      parameter_server_strategy_v2.ParameterServerStrategyV2(cluster_resolver)


if __name__ == "__main__":
  v2_compat.enable_v2_behavior()
  multi_process_runner.test_main()
