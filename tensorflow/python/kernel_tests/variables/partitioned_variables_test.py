# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for partitioned_variables.py."""

import os

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variable_v1
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import saver as saver_lib


def initialized_value(var):
  return cond.cond(
      variable_v1.is_variable_initialized(var), var.read_value,
      lambda: var.initial_value)


class PartitionerCreatorsTest(test.TestCase):

  def testFixedSizePartitioner(self):
    with self.cached_session():
      partitioner = partitioned_variables.fixed_size_partitioner(5, axis=0)
      with variable_scope.variable_scope("root", partitioner=partitioner):
        v0 = variable_scope.get_variable(
            "v0", dtype=dtypes.float32, shape=(10, 10))
        v0_list = v0._get_variable_list()
        v0_part = v0._get_partitions()
        self.assertEqual(len(v0_list), 5)
        self.assertAllEqual(v0_part, (5, 1))

  def testFixedSizePartitionerInt64(self):
    with self.cached_session():
      partitioner = partitioned_variables.fixed_size_partitioner(4, axis=0)
      with variable_scope.variable_scope("root", partitioner=partitioner):
        v0 = variable_scope.get_variable("v0", dtype=dtypes.int64, shape=[20])
        v0_list = v0._get_variable_list()
        self.assertEqual(len(v0_list), 4)

  def testResourceFixedSizePartitioner(self):
    with self.cached_session():
      partitioner = partitioned_variables.fixed_size_partitioner(5, axis=0)
      with variable_scope.variable_scope(
          "root", partitioner=partitioner, use_resource=True):
        v0 = variable_scope.get_variable(
            "v0", dtype=dtypes.float32, shape=(10, 10))
        v0_list = v0._get_variable_list()
        v0_part = v0._get_partitions()
        self.assertEqual(len(v0_list), 5)
        self.assertAllEqual(v0_part, (5, 1))

  def _testVariableAxisSizePartitioner(self,
                                       name,
                                       axis,
                                       max_shard_bytes,
                                       expected_axis_shards,
                                       expected_partitions,
                                       max_shards=None):
    partitioner = partitioned_variables.variable_axis_size_partitioner(
        axis=axis, max_shard_bytes=max_shard_bytes, max_shards=max_shards)

    with variable_scope.variable_scope("root", partitioner=partitioner):
      v0 = variable_scope.get_variable(
          name, dtype=dtypes.float32, shape=(4, 8, 16, 32))
      v0_list = v0._get_variable_list()
      v0_part = v0._get_partitions()
      self.assertEqual(len(v0_list), expected_axis_shards)
      self.assertAllEqual(v0_part, expected_partitions)

  def testVariableAxisSizePartitioner(self):
    with self.cached_session():
      # Create a partitioned variable of shape (4, 8, 16, 32) type float32
      # Bytes per slice along the given axes:

      # 8 * 16 * 32 * sizeof(float32) = 16384 / slice on axis 0
      # 4 * 16 * 32 * sizeof(float32) = 8192 / slice on axis 1
      # 4 * 8 * 32 * sizeof(float32) = 4096 / slice on axis 2
      # 4 * 8 * 16 * sizeof(float32) = 2048 / slice on axis 3

      # Now partition it in different ways...

      # No need to slice: bytes_per_slice * dim0 = 65536 < max_shard_bytes
      self._testVariableAxisSizePartitioner(
          "v0",
          axis=0,
          max_shard_bytes=131072,
          expected_axis_shards=1,
          expected_partitions=(1, 1, 1, 1))

      # Slice exactly once: bytes_per_slice * dim1 = 65536 = max_shard_bytes
      self._testVariableAxisSizePartitioner(
          "v1",
          axis=1,
          max_shard_bytes=65536,
          expected_axis_shards=1,
          expected_partitions=(1, 1, 1, 1))

      # Slice into 2 parts:
      # bytes_per_slice = 4096
      # slices_per_shard = 32768 / 4096 = 8
      # axis_shards = 16 / 8 = 2
      self._testVariableAxisSizePartitioner(
          "v2",
          axis=2,
          max_shard_bytes=32768,
          expected_axis_shards=2,
          expected_partitions=(1, 1, 2, 1))

      # This partitioner makes sure we maximize the number of shards along
      # axis 3. Slice it into 32 parts:
      # bytes_per_slice = 2048
      # slices_per_shard = 2048 / 2048 = 1
      # axis_shards = 32 / 1 = 32
      self._testVariableAxisSizePartitioner(
          "v3a",
          axis=3,
          max_shard_bytes=2048,
          expected_axis_shards=32,
          expected_partitions=(1, 1, 1, 32))

      # This partitioner makes sure we do not go past the bound of allowable
      # number of shards along axis 3.
      # Slice into 32 parts:
      # bytes_per_slice = 2048
      # slices_per_shard = max(1, 1024 / 2048) = 1
      # axis_shards = 32 / 1 = 32
      # Slice into max of 32 parts because: max_shard_bytes < bytes_per_slice
      self._testVariableAxisSizePartitioner(
          "v3b",
          axis=3,
          max_shard_bytes=1024,
          expected_axis_shards=32,
          expected_partitions=(1, 1, 1, 32))

      # Specify max_shards so that it won't affect sharding.
      self._testVariableAxisSizePartitioner(
          "v3c",
          axis=3,
          max_shard_bytes=1024,
          expected_axis_shards=32,
          expected_partitions=(1, 1, 1, 32),
          max_shards=33)

      # Specify max_shards so that it will affect sharding.
      self._testVariableAxisSizePartitioner(
          "v3d",
          axis=3,
          max_shard_bytes=1024,
          expected_axis_shards=2,
          expected_partitions=(1, 1, 1, 2),
          max_shards=2)

      # Use the partitioner with strings
      partitioner_axis3_str = partitioned_variables.variable_axis_size_partitioner(  # pylint: disable=line-too-long
          axis=3,
          max_shard_bytes=32768,
          bytes_per_string_element=8)

      with variable_scope.variable_scope(
          "root", partitioner=partitioner_axis3_str):
        v3str = variable_scope.get_variable(
            "v3str",
            initializer=np.array([""] * 4 * 8 * 16 * 32).reshape(4, 8, 16, 32),  # pylint: disable=too-many-function-args
            dtype=dtypes.string,
            shape=(4, 8, 16, 32))
        v3str_list = v3str._get_variable_list()
        v3str_part = v3str._get_partitions()

        # Now the estimated bytes_per_slice = 4*8*16*bytes_per_string_element
        # which is equal to 4096.  Setting a max_shard_bytes of 32768
        # and we should get a split of 4.
        # Slice into 4 parts:
        # bytes_per_slice = 4096
        # slices_per_shard = 32768 / 4096 = 8
        # axis_shards = 32 / 8 = 4
        self.assertEqual(len(v3str_list), 4)
        self.assertAllEqual(v3str_part, (1, 1, 1, 4))

  def _testMinMaxVariablePartitioner(self, max_partitions, axis, min_slice_size,
                                     var_name, var_shape, expected_axis_shards,
                                     expected_partitions):
    partitioner = partitioned_variables.min_max_variable_partitioner(
        max_partitions=max_partitions, axis=axis, min_slice_size=min_slice_size)
    with variable_scope.variable_scope("root", partitioner=partitioner):
      v0 = variable_scope.get_variable(
          var_name, dtype=dtypes.float32, shape=var_shape)
      v0_list = v0._get_variable_list()
      v0_part = v0._get_partitions()
      self.assertEqual(len(v0_list), expected_axis_shards)
      self.assertAllEqual(v0_part, expected_partitions)

  def testMinMaxVariablePartitioner(self):
    with self.cached_session():
      # Partitioning a variable of shape=[2048] with a minimum of 2K per slice.
      self._testMinMaxVariablePartitioner(
          max_partitions=100,
          axis=0,
          min_slice_size=2 << 10,
          var_name="v0_0",
          var_shape=[2048],
          expected_axis_shards=4,
          expected_partitions=[4])

      # Partitioning a variable of shape=[2048, 1024] with a minimum of 256K per
      # slice.
      self._testMinMaxVariablePartitioner(
          max_partitions=100,
          axis=0,
          min_slice_size=256 << 10,
          var_name="v0",
          var_shape=[2048, 1024],
          expected_axis_shards=32,
          expected_partitions=[32, 1])

      # max_partitions restricts partitioning of the variable.
      self._testMinMaxVariablePartitioner(
          max_partitions=16,
          axis=0,
          min_slice_size=256 << 10,
          var_name="v1_max",
          var_shape=[2048, 1024],
          expected_axis_shards=16,
          expected_partitions=[16, 1])
      self._testMinMaxVariablePartitioner(
          max_partitions=1,
          axis=0,
          min_slice_size=256 << 10,
          var_name="v2_max",
          var_shape=[2048, 1024],
          expected_axis_shards=1,
          expected_partitions=[1, 1])

      # Reducing/Increasing min_slice_size proportionately increases/reduces the
      # number of partitions.
      self._testMinMaxVariablePartitioner(
          max_partitions=100,
          axis=0,
          min_slice_size=128 << 10,
          var_name="v3_slice",
          var_shape=[2048, 1024],
          expected_axis_shards=64,
          expected_partitions=[64, 1])
      self._testMinMaxVariablePartitioner(
          max_partitions=100,
          axis=0,
          min_slice_size=512 << 10,
          var_name="v4_slice",
          var_shape=[2048, 1024],
          expected_axis_shards=16,
          expected_partitions=[16, 1])

      # Partitioning the variable along a different axis.
      self._testMinMaxVariablePartitioner(
          max_partitions=100,
          axis=1,
          min_slice_size=256 << 10,
          var_name="v5_axis",
          var_shape=[64, 1024, 1, 3],
          expected_axis_shards=3,
          expected_partitions=[1, 3, 1, 1])
      self._testMinMaxVariablePartitioner(
          max_partitions=100,
          axis=3,
          min_slice_size=256 << 10,
          var_name="v6_axis",
          var_shape=[64, 1024, 1, 3],
          expected_axis_shards=3,
          expected_partitions=[1, 1, 1, 3])

      # Can not partition the variable more than what its shape allows.
      self._testMinMaxVariablePartitioner(
          max_partitions=100,
          axis=0,
          min_slice_size=256 << 10,
          var_name="v7_shape",
          var_shape=[16, 128, 1024],
          expected_axis_shards=16,
          expected_partitions=[16, 1, 1])
      self._testMinMaxVariablePartitioner(
          max_partitions=100,
          axis=0,
          min_slice_size=256 << 10,
          var_name="v8_shape",
          var_shape=[4, 512, 1024],
          expected_axis_shards=4,
          expected_partitions=[4, 1, 1])


def _IotaInitializer(shape, dtype=dtypes.float32, partition_info=None):
  assert dtype == dtypes.float32
  if len(shape) == 1:
    return range(shape[0])
  else:
    val = _IotaInitializer(shape[1:], dtype)
    return [[(10**i) * v for v in val] for i in range(shape[0])]


class PartitionedVariablesTestCase(test.TestCase):

  def _TestSaveSpec(self, slices, expected_specs):
    self.assertEqual(len(expected_specs), len(slices))
    for i in range(len(expected_specs)):
      self.assertEqual(expected_specs[i], slices[i]._save_slice_info.spec)

  def testVecConstantInit(self):
    with self.cached_session():
      rnd_par = constant_op.constant([1, 2, 3, 4])
      vs = partitioned_variables.create_partitioned_variables([4], [4], rnd_par)
      self.evaluate(variables.global_variables_initializer())
      val = array_ops.concat(vs, 0)
      rnd = self.evaluate(rnd_par)
      self.assertAllClose(rnd, val)
      self.assertEqual([dtypes.int32] * 4, [v.dtype.base_dtype for v in vs])
      self._TestSaveSpec(vs, ["4 0,1", "4 1,1", "4 2,1", "4 3,1"])

  def testConstantInit(self):
    with self.cached_session():
      rnd_par = constant_op.constant([[1, 2, 3, 4], [5, 6, 7, 8]])
      vs = partitioned_variables.create_partitioned_variables([2, 4], [1, 2],
                                                              rnd_par)
      self.evaluate(variables.global_variables_initializer())
      val = array_ops.concat(vs, 1)
      rnd = self.evaluate(rnd_par)
      self.assertAllClose(rnd, val)
      self.assertEqual([dtypes.int32] * 2, [v.dtype.base_dtype for v in vs])
      self._TestSaveSpec(vs, ["2 4 0,2:0,2", "2 4 0,2:2,2"])

  def _testNameHelper(self, use_resource=False):
    with self.cached_session():
      rnd_par = constant_op.constant([[1, 2, 3, 4], [5, 6, 7, 8]])
      with variable_scope.variable_scope("hi", use_resource=use_resource):
        vs1 = partitioned_variables.create_partitioned_variables([2, 4], [1, 2],
                                                                 rnd_par)
        vs2 = partitioned_variables.create_partitioned_variables([2, 4], [1, 2],
                                                                 rnd_par)
      self.evaluate(variables.global_variables_initializer())
      var1_name = vs1[0]._save_slice_info.full_name
      var2_name = vs2[0]._save_slice_info.full_name
      self.assertEqual("hi/PartitionedVariable", var1_name)
      self.assertEqual("hi/PartitionedVariable_1", var2_name)
      self.assertEqual(var1_name + "/part_0:0", vs1[0].name)
      self.assertEqual(var1_name + "/part_1:0", vs1[1].name)
      self.assertEqual(var2_name + "/part_0:0", vs2[0].name)
      self.assertEqual(var2_name + "/part_1:0", vs2[1].name)
    # Test same variable.
    with self.cached_session():
      rnd_par = constant_op.constant([[1, 2, 3, 4], [5, 6, 7, 8]])
      with variable_scope.variable_scope(
          "hola", use_resource=use_resource) as vs:
        vs1 = partitioned_variables.create_partitioned_variables(
            [2, 4], [1, 2], rnd_par, dtype=dtypes.int32)
      with variable_scope.variable_scope(
          vs, reuse=True, use_resource=use_resource):
        vs2 = partitioned_variables.create_partitioned_variables(
            [2, 4], [1, 2], rnd_par, dtype=dtypes.int32)
      self.evaluate(variables.global_variables_initializer())
      var1_name = vs1[0]._save_slice_info.full_name
      var2_name = vs2[0]._save_slice_info.full_name
      self.assertEqual("hola/PartitionedVariable", var1_name)
      self.assertEqual("hola/PartitionedVariable", var2_name)
      self.assertEqual(var1_name + "/part_0:0", vs1[0].name)
      self.assertEqual(var1_name + "/part_1:0", vs1[1].name)
      self.assertEqual(var2_name + "/part_0:0", vs2[0].name)
      self.assertEqual(var2_name + "/part_1:0", vs2[1].name)
    # Test name_scope
    with self.cached_session():
      rnd_par = constant_op.constant([[1, 2, 3, 4], [5, 6, 7, 8]])
      with ops.name_scope("ola"):
        vs1 = partitioned_variables.create_partitioned_variables([2, 4], [1, 2],
                                                                 rnd_par)
        vs2 = partitioned_variables.create_partitioned_variables([2, 4], [1, 2],
                                                                 rnd_par)
      self.evaluate(variables.global_variables_initializer())
      var1_name = vs1[0]._save_slice_info.full_name
      var2_name = vs2[0]._save_slice_info.full_name
      # Currently, the name scope 'ola' has no effect.
      self.assertEqual("PartitionedVariable", var1_name)
      self.assertEqual("PartitionedVariable_1", var2_name)
      self.assertEqual(var1_name + "/part_0:0", vs1[0].name)
      self.assertEqual(var1_name + "/part_1:0", vs1[1].name)
      self.assertEqual(var2_name + "/part_0:0", vs2[0].name)
      self.assertEqual(var2_name + "/part_1:0", vs2[1].name)

  @test_util.run_deprecated_v1
  def testName(self):
    self._testNameHelper(use_resource=False)

  def testResourceName(self):
    self._testNameHelper(use_resource=True)

  def testRandomInitValue(self):
    with self.cached_session():
      rnd = variables.Variable(random_ops.random_uniform([200, 40]))
      vs = partitioned_variables.create_partitioned_variables(
          rnd.get_shape(), [1, 10], initialized_value(rnd))
      self.evaluate(variables.global_variables_initializer())
      val = array_ops.concat(vs, 1)
      rnd = self.evaluate(rnd)
      self.assertAllClose(rnd, val)
      self.assertEqual([dtypes.float32] * 10, [v.dtype.base_dtype for v in vs])
      self._TestSaveSpec(vs, [
          "200 40 0,200:0,4", "200 40 0,200:4,4", "200 40 0,200:8,4",
          "200 40 0,200:12,4", "200 40 0,200:16,4", "200 40 0,200:20,4",
          "200 40 0,200:24,4", "200 40 0,200:28,4", "200 40 0,200:32,4",
          "200 40 0,200:36,4"
      ])

  def testRandomInitUnevenPartitions(self):
    with self.cached_session():
      rnd = variables.Variable(
          random_ops.random_uniform([20, 43], dtype=dtypes.float64))
      var_lists = [
          partitioned_variables.create_partitioned_variables(
              rnd.get_shape(), [1, i], initialized_value(rnd))
          for i in range(1, 10)
      ]
      self.evaluate(variables.global_variables_initializer())
      rnd_val = self.evaluate(rnd)
      # Only check the slice save specs for the first 5 tf.
      save_specs = [
          # One slice
          ["20 43 0,20:0,43"],
          # Two slices
          ["20 43 0,20:0,22", "20 43 0,20:22,21"],
          # Three slices
          ["20 43 0,20:0,15", "20 43 0,20:15,14", "20 43 0,20:29,14"],
          # Four slices
          [
              "20 43 0,20:0,11", "20 43 0,20:11,11", "20 43 0,20:22,11",
              "20 43 0,20:33,10"
          ],
          # Five slices
          [
              "20 43 0,20:0,9", "20 43 0,20:9,9", "20 43 0,20:18,9",
              "20 43 0,20:27,8", "20 43 0,20:35,8"
          ]
      ]
      for i, vs in enumerate(var_lists):
        var_val = array_ops.concat(vs, 1)
        self.assertAllClose(rnd_val, var_val)
        self.assertEqual([dtypes.float64] * len(vs),
                         [v.dtype.base_dtype for v in vs])
        if i < len(save_specs):
          self._TestSaveSpec(vs, save_specs[i])

  def testDegenerate(self):
    with self.cached_session():
      rnd = variables.Variable(random_ops.random_uniform([10, 43]))
      vs = partitioned_variables.create_partitioned_variables(
          rnd.get_shape(), [1, 1], initialized_value(rnd))
      self.evaluate(variables.global_variables_initializer())
      val = array_ops.concat(vs, 0)
      rnd = self.evaluate(rnd)
      self.assertAllClose(rnd, val)
      self._TestSaveSpec(vs, ["10 43 0,10:0,43"])

  def testSliceSizeOne(self):
    with self.cached_session():
      rnd = variables.Variable(random_ops.random_uniform([10, 43]))
      vs = partitioned_variables.create_partitioned_variables(
          rnd.get_shape(), [10, 1], initialized_value(rnd))
      self.evaluate(variables.global_variables_initializer())
      val = array_ops.concat(vs, 0)
      rnd = self.evaluate(rnd)
      self.assertAllClose(rnd, val)
      self._TestSaveSpec(vs, [
          "10 43 0,1:0,43", "10 43 1,1:0,43", "10 43 2,1:0,43",
          "10 43 3,1:0,43", "10 43 4,1:0,43", "10 43 5,1:0,43",
          "10 43 6,1:0,43", "10 43 7,1:0,43", "10 43 8,1:0,43", "10 43 9,1:0,43"
      ])

  def testIotaInitializer(self):
    self.assertAllClose([0., 1., 2., 3.], _IotaInitializer([4]))
    self.assertAllClose([[0., 1.], [0., 10.], [0., 100.], [0., 1000.]],
                        _IotaInitializer([4, 2]))
    with self.cached_session():
      vs = partitioned_variables.create_partitioned_variables([13, 5], [3, 1],
                                                              _IotaInitializer)
      self.evaluate(variables.global_variables_initializer())
      slice0 = _IotaInitializer([5, 5])
      slice1 = _IotaInitializer([4, 5])
      slice2 = _IotaInitializer([4, 5])
      val = array_ops.concat(vs, 0)
      self.assertAllClose(slice0 + slice1 + slice2, val)
      self._TestSaveSpec(vs, ["13 5 0,5:0,5", "13 5 5,4:0,5", "13 5 9,4:0,5"])

  @test_util.run_deprecated_v1
  def testRandomInitializer(self):
    # Sanity check that the slices uses a different seed when using a random
    # initializer function.
    with self.cached_session():
      var0, var1 = partitioned_variables.create_partitioned_variables(
          [20, 12], [1, 2], init_ops.random_uniform_initializer())
      self.evaluate(variables.global_variables_initializer())
      val0, val1 = self.evaluate(var0).flatten(), self.evaluate(var1).flatten()
      self.assertTrue(np.linalg.norm(val0 - val1) > 1e-6)
    # Negative test that proves that slices have the same values if
    # the random initializer uses a seed.
    with self.cached_session():
      var0, var1 = partitioned_variables.create_partitioned_variables(
          [20, 12], [1, 2], init_ops.random_uniform_initializer(seed=201))
      self.evaluate(variables.global_variables_initializer())
      val0, val1 = self.evaluate(var0).flatten(), self.evaluate(var1).flatten()
      self.assertAllClose(val0, val1)

  def testSomeErrors(self):
    with self.cached_session():
      rnd = variables.Variable(random_ops.random_uniform([10, 43]))
      with self.assertRaises(ValueError):
        partitioned_variables.create_partitioned_variables(
            [10], [1, 1], initialized_value(rnd))
      with self.assertRaises(ValueError):
        partitioned_variables.create_partitioned_variables(
            [10, 20], [1], initialized_value(rnd))
      with self.assertRaises(ValueError):
        partitioned_variables.create_partitioned_variables(
            [10, 43], [1], initialized_value(rnd))
      with self.assertRaises(ValueError):
        partitioned_variables.create_partitioned_variables(
            [10, 43], [1, 2, 3], initialized_value(rnd))
      with self.assertRaises(ValueError):
        partitioned_variables.create_partitioned_variables(
            [10, 43], [11, 1], initialized_value(rnd))
      with self.assertRaises(ValueError):
        partitioned_variables.create_partitioned_variables(
            [10, 43], [20, 1], initialized_value(rnd))
      with self.assertRaises(ValueError):
        partitioned_variables.create_partitioned_variables(
            [10, 43], [1, 50], initialized_value(rnd))

  @test_util.run_deprecated_v1
  def testControlDepsNone(self):
    with self.cached_session() as session:
      c = constant_op.constant(1.0)
      with ops.control_dependencies([c]):
        # d get the control dependency.
        d = constant_op.constant(2.0)
        # Partitioned variables do not.
        var_x = variable_scope.get_variable(
            "x",
            shape=[2],
            initializer=init_ops.ones_initializer(),
            partitioner=partitioned_variables.variable_axis_size_partitioner(4))

        ops_before_read = session.graph.get_operations()
        var_x.as_tensor()  # Caches the ops for subsequent reads.
        reading_ops = [
            op for op in session.graph.get_operations()
            if op not in ops_before_read
        ]

      self.assertEqual([c.op], d.op.control_inputs)
      # Tests that no control dependencies are added to reading a partitioned
      # variable which is similar to reading a variable.
      for op in reading_ops:
        self.assertEqual([], op.control_inputs)

  @test_util.run_deprecated_v1
  def testConcat(self):
    with self.cached_session() as session:
      var_x = variable_scope.get_variable(
          "x",
          initializer=constant_op.constant([1., 2.]),
          partitioner=partitioned_variables.variable_axis_size_partitioner(4))

      c = constant_op.constant(1.0)
      with ops.control_dependencies([c]):
        ops_before_concat = session.graph.get_operations()
        value = var_x._concat()  # pylint: disable=protected-access
        concat_ops = [
            op for op in session.graph.get_operations()
            if op not in ops_before_concat
        ]

      concat_control_inputs = [
          ci for op in concat_ops for ci in op.control_inputs
      ]
      self.assertTrue(
          c.op in concat_control_inputs,
          "var_x._concat() should get control dependencies from its scope.")
      self.evaluate(variables.global_variables_initializer())
      self.assertAllClose(value, var_x.as_tensor())

  def testMetaGraphSaveLoad(self):
    save_prefix = os.path.join(self.get_temp_dir(), "ckpt")
    save_graph = ops.Graph()
    with save_graph.as_default(), self.session(
        graph=save_graph) as session:
      partitioner = partitioned_variables.fixed_size_partitioner(5, axis=0)
      with variable_scope.variable_scope("root", partitioner=partitioner):
        v0 = variable_scope.get_variable(
            "v0", dtype=dtypes.float32, shape=(10, 10))
        v0_list = v0._get_variable_list()
        v0_part = v0._get_partitions()
        self.assertEqual(len(v0_list), 5)
        self.assertAllEqual(v0_part, (5, 1))
        self.evaluate(variables.global_variables_initializer())

        save_graph.get_collection_ref("partvar").append(v0)
        saver = saver_lib.Saver()
        save_graph.finalize()
        save_path = saver.save(sess=session, save_path=save_prefix)
        previous_value = session.run(
            save_graph.get_tensor_by_name(v0.name + ":0"))

    restore_graph = ops.Graph()
    with restore_graph.as_default(), self.session(
        graph=restore_graph) as session:
      saver = saver_lib.import_meta_graph(save_path + ".meta")
      saver.restore(sess=session, save_path=save_path)
      v0, = save_graph.get_collection_ref("partvar")
      self.assertIsInstance(v0, variables.PartitionedVariable)
      self.assertAllEqual(
          previous_value,
          session.run(restore_graph.get_tensor_by_name(v0.name + ":0")))

if __name__ == "__main__":
  test.main()
