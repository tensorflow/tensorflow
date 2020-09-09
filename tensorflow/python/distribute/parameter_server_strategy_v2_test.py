# Lint as: python3
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import platform
import sys

from tensorflow.python.distribute import multi_worker_test_base
from tensorflow.python.distribute import parameter_server_strategy_v2
from tensorflow.python.distribute import sharded_variable
from tensorflow.python.distribute.cluster_resolver import SimpleClusterResolver
from tensorflow.python.eager import def_function
from tensorflow.python.eager import test
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops_v2
from tensorflow.python.ops import linalg_ops_impl
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import variables
from tensorflow.python.training.server_lib import ClusterSpec


class ParameterServerStrategyV2Test(test.TestCase):

  @classmethod
  def setUpClass(cls):
    super(ParameterServerStrategyV2Test, cls).setUpClass()
    cluster_def = multi_worker_test_base.create_in_process_cluster(
        num_workers=2, num_ps=3)
    cls.cluster_resolver = SimpleClusterResolver(ClusterSpec(cluster_def))

  def testVariablePlacement(self):

    if sys.version_info >= (3, 8) and platform.system() == "Windows":
      # TODO(b/165013260): Fix this
      self.skipTest("Test is currently broken on Windows with Python 3.8")

    strategy = parameter_server_strategy_v2.ParameterServerStrategyV2(
        self.cluster_resolver)
    v1 = variables.Variable(initial_value=0.0)
    with strategy.scope():
      v2 = variables.Variable(initial_value=1.0)
      v3 = variables.Variable(initial_value=2.0)
      v4 = variables.Variable(initial_value=3.0)
      v5 = variables.Variable(initial_value=4.0)
    # v1 was created outside scope so should be on client.
    self.assertEqual(v1.device, "/job:chief/replica:0/task:0/device:CPU:0")
    # v2 through v5 are created in scope and in a round-robin manner.
    self.assertEqual(v2.device, "/job:ps/replica:0/task:0/device:CPU:0")
    self.assertEqual(v3.device, "/job:ps/replica:0/task:1/device:CPU:0")
    self.assertEqual(v4.device, "/job:ps/replica:0/task:2/device:CPU:0")
    self.assertEqual(v5.device, "/job:ps/replica:0/task:0/device:CPU:0")


class PartitionAwareIdentity(object):

  def __call__(self, shape, dtype, partition):
    value = linalg_ops_impl.eye(*shape, dtype=dtype)
    if partition is not None:
      value = array_ops.slice(value, partition.offsets, partition.shape)
    return value


class VariablePartitioningTest(test.TestCase):

  @classmethod
  def setUpClass(cls):
    super(VariablePartitioningTest, cls).setUpClass()
    cluster_def = multi_worker_test_base.create_in_process_cluster(
        num_workers=2, num_ps=2)
    cls.cluster_resolver = SimpleClusterResolver(ClusterSpec(cluster_def))

  def setUp(self):
    super().setUp()
    if sys.version_info >= (3, 8) and platform.system() == "Windows":
      # TODO(b/165013260): Fix this
      self.skipTest("Test is currently broken on Windows with Python 3.8")

  def testDefaultNoPartition(self):
    strategy = parameter_server_strategy_v2.ParameterServerStrategyV2(
        self.cluster_resolver)
    with strategy.scope():
      v = variables.Variable([0, 1, 2, 3])

    self.assertIsInstance(v, variables.Variable)

  def testBasic(self):
    strategy = parameter_server_strategy_v2.ParameterServerStrategyV2(
        self.cluster_resolver, partitioned_variables.fixed_size_partitioner(2))
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
    self.assertAllEqual(v1.variables[0].read_value().numpy(),
                        [[0, 1], [2, 3], [4, 5]])
    self.assertAllEqual(v1.variables[1].read_value().numpy(), [[6, 7], [8, 9]])

    self.assertIsInstance(v2, sharded_variable.ShardedVariable)
    self.assertLen(v2.variables, 2)
    self.assertRegex(v2.variables[0].device, "/job:ps/replica:0/task:0")
    self.assertRegex(v2.variables[1].device, "/job:ps/replica:0/task:1")
    self.assertAllEqual(v2.variables[0].read_value().numpy(), [[0], [1], [2]])
    self.assertAllEqual(v2.variables[1].read_value().numpy(), [[3], [4], [5]])

  def testNonCallableInitialValue(self):
    strategy = parameter_server_strategy_v2.ParameterServerStrategyV2(
        self.cluster_resolver, partitioned_variables.fixed_size_partitioner(4))
    with strategy.scope():
      v = variables.Variable([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    self.assertIsInstance(v, sharded_variable.ShardedVariable)
    self.assertLen(v.variables, 4)
    self.assertRegex(v.variables[0].device, "/job:ps/replica:0/task:0")
    self.assertRegex(v.variables[1].device, "/job:ps/replica:0/task:1")
    self.assertRegex(v.variables[2].device, "/job:ps/replica:0/task:0")
    self.assertRegex(v.variables[3].device, "/job:ps/replica:0/task:1")
    self.assertAllEqual(v.variables[0].read_value().numpy(), [0, 1, 2])
    self.assertAllEqual(v.variables[1].read_value().numpy(), [3, 4, 5])
    self.assertAllEqual(v.variables[2].read_value().numpy(), [6, 7])
    self.assertAllEqual(v.variables[3].read_value().numpy(), [8, 9])

  def testNumPartitionsLargerThanSize(self):
    strategy = parameter_server_strategy_v2.ParameterServerStrategyV2(
        self.cluster_resolver, partitioned_variables.fixed_size_partitioner(4))
    with strategy.scope():
      v = variables.Variable([0, 1, 2])

    self.assertIsInstance(v, sharded_variable.ShardedVariable)
    self.assertLen(v.variables, 3)
    self.assertRegex(v.variables[0].device, "/job:ps/replica:0/task:0")
    self.assertRegex(v.variables[1].device, "/job:ps/replica:0/task:1")
    self.assertRegex(v.variables[2].device, "/job:ps/replica:0/task:0")
    self.assertAllEqual(v.variables[0].read_value().numpy(), [0])
    self.assertAllEqual(v.variables[1].read_value().numpy(), [1])
    self.assertAllEqual(v.variables[2].read_value().numpy(), [2])

  def testPartitionToOne(self):
    # For small variables there is only one partition.
    variable_partitioner = partitioned_variables.min_max_variable_partitioner(
        max_partitions=2, min_slice_size=64 << 20)
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
    self.assertAllEqual(v1.read_value().numpy(), [0] * 10)

    self.assertIsInstance(v2, variables.Variable)
    self.assertNotIsInstance(v2, sharded_variable.ShardedVariable)
    self.assertRegex(v2.device, "/job:ps/replica:0/task:1")
    self.assertAllEqual(v2.read_value().numpy(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

  def testColocateWith(self):
    strategy = parameter_server_strategy_v2.ParameterServerStrategyV2(
        self.cluster_resolver, partitioned_variables.fixed_size_partitioner(2))
    with strategy.scope():
      v1 = variables.Variable([0, 1, 2, 3])

      with strategy.extended.colocate_vars_with(v1.variables[0]):
        v2 = variables.Variable([4, 5])

    self.assertIsInstance(v1, sharded_variable.ShardedVariable)

    self.assertIsInstance(v2, variables.Variable)
    self.assertNotIsInstance(v2, sharded_variable.ShardedVariable)
    self.assertEqual(v2.device, v1.variables[0].device)
    self.assertAllEqual(v2.read_value().numpy(), [4, 5])

  def testPartitionAwareInitializer(self):
    strategy = parameter_server_strategy_v2.ParameterServerStrategyV2(
        self.cluster_resolver, partitioned_variables.fixed_size_partitioner(2))
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
    self.assertAllEqual(v.variables[0].read_value().numpy(),
                        [[1, 0, 0, 0], [0, 1, 0, 0]])
    self.assertAllEqual(v.variables[1].read_value().numpy(),
                        [[0, 0, 1, 0], [0, 0, 0, 1]])

  def testPartitionWhenLackOfInfo(self):
    strategy = parameter_server_strategy_v2.ParameterServerStrategyV2(
        self.cluster_resolver, partitioned_variables.fixed_size_partitioner(2))
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
      self.assertAllEqual(v.variables[0].read_value().numpy(), [0, 1])
      self.assertAllEqual(v.variables[1].read_value().numpy(), [2, 3])

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
    strategy = parameter_server_strategy_v2.ParameterServerStrategyV2(
        self.cluster_resolver, partitioned_variables.fixed_size_partitioner(2))

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
        self.assertAllEqual(v.variables[0].read_value().numpy(), [[1., 0.]])
        self.assertAllEqual(v.variables[1].read_value().numpy(), [[0., 1.]])


if __name__ == "__main__":
  test.main()
