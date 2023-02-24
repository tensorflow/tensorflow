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
"""Tests for DTensor collectives."""

import os

from absl.testing import parameterized
import numpy as np

# pylint: disable=g-direct-tensorflow-import
from tensorflow.dtensor.python import api
from tensorflow.dtensor.python import d_variable
from tensorflow.dtensor.python import dtensor_device
from tensorflow.dtensor.python import layout as layout_lib
from tensorflow.dtensor.python.tests import test_util
from tensorflow.python.eager.polymorphic_function import polymorphic_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test

# pylint: enable=g-direct-tensorflow-import

Layout = layout_lib.Layout

_MESH_DIM_X = 'x'
_MESH_DIM_Y = 'y'
_MESH_DIMS = [_MESH_DIM_X, _MESH_DIM_Y]


class CollectiveTest(test_util.DTensorBaseTest):

  def setUp(self):
    super(CollectiveTest, self).setUp()

    global_ids = test_util.create_device_ids_array((2, 1))
    local_ids = np.ravel(global_ids).tolist()
    mesh_dict = {
        device: layout_lib.Mesh(_MESH_DIMS, global_ids, local_ids,
                                test_util.create_device_list((2, 1), device))
        for device in ('CPU', 'GPU', 'TPU')
    }
    self.mesh = self.configTestMesh(mesh_dict)
    self.fully_replicated_layout_2d = Layout.replicated(self.mesh, rank=2)
    self.first_dimension_sharded_layout_2d = Layout.batch_sharded(
        self.mesh, _MESH_DIM_X, 2)
    self.scalar_layout = Layout.replicated(self.mesh, rank=0)

  def testReduceOnBfloat16(self):
    self.skipForDeviceType(['GPU'],
                           'GPUs do not support bfloat16 collective reduce')
    self.skipForDeviceType(['TPU'],
                           'This test only needs to run on 2 cores.',
                           unless_device_count_equals_to=2)

    a = constant_op.constant(
        np.array([[1, 2, 3, 4], [5.0, 6.0, 7.0, 8.0]]), dtype=dtypes.bfloat16)
    expected_result = math_ops.reduce_sum(a)

    sharded_a = api.relayout(a, self.first_dimension_sharded_layout_2d)
    dtensor_result = math_ops.reduce_sum(sharded_a)

    self.assertDTensorEqual(expected_result, self.scalar_layout, dtensor_result)

  def testReduceOnInt32(self):
    a = constant_op.constant(
        np.array([[1, 2, 3, 4], [5, 6, 7, 8]]), dtype=dtypes.int32)

    expected_result = math_ops.reduce_sum(a)

    sharded_a = api.relayout(a, self.first_dimension_sharded_layout_2d)
    dtensor_result = math_ops.reduce_sum(sharded_a)

    self.assertDTensorEqual(expected_result, self.scalar_layout, dtensor_result)

  def testTwoReducesWithAssign(self):
    # FIXME(b/238384852): The purpose of this test is to validate the control
    # dependency added by DTensor.
    # However, as we have no way of testing the per-device graph
    # produced by the DTensor SPMD expansion, we have to use manual logging
    # to verify if the results are correct.
    # For example, this test is used to check cl/459358521.

    # Uses dtypes.float32 to avoid int32 problem with VarHandleOp on GPUs.
    a = constant_op.constant(
        np.array([[1, 2, 3, 4], [5, 6, 7, 8]]), dtype=dtypes.float32)
    b = constant_op.constant(
        np.array([[11, 12, 13, 4], [15, 16, 17, 18]]), dtype=dtypes.float32)

    expected_result = math_ops.reduce_sum(a) * math_ops.reduce_sum(b)

    sharded_a = api.relayout(a, self.first_dimension_sharded_layout_2d)
    sharded_b = api.relayout(b, self.first_dimension_sharded_layout_2d)
    sharded_v = d_variable.DVariable(sharded_b)

    @polymorphic_function.function
    def func(a, b):
      a1 = math_ops.reduce_sum(a, name='reducea')
      sharded_v.assign(a)
      b1 = math_ops.reduce_sum(b, name='reduceb')
      return a1 * b1

    with api.run_on(self.mesh):
      dtensor_result = func(sharded_a, sharded_b)
    self.assertDTensorEqual(expected_result, self.scalar_layout, dtensor_result)

  @parameterized.named_parameters(
      ('_all', math_ops.reduce_all),
      ('_any', math_ops.reduce_any),
  )
  def testReduceOnBool(self, reduction):
    # TODO(b/193531363): Track the work to support int32 reduce.
    self.skipForDeviceType(['GPU'],
                           'GPUs do not support int32 collective reduce')
    self.skipForDeviceType(['TPU'],
                           'This test only needs to run on 2 cores.',
                           unless_device_count_equals_to=2)

    a = constant_op.constant(
        np.array([[True, False, False, True], [False, False, False, True]]),
        dtype=dtypes.bool)
    expected_result = reduction(a)

    sharded_a = api.relayout(a, self.first_dimension_sharded_layout_2d)
    dtensor_result = reduction(sharded_a)

    self.assertDTensorEqual(expected_result, self.scalar_layout, dtensor_result)

  def testAllToAllOnBool(self):
    # TODO(b/193531363): Track the work to support int32 reduce.
    self.skipForDeviceType(['GPU'],
                           'GPUs do not support int32 collective reduce')
    self.skipForDeviceType(['TPU'],
                           'This test only needs to run on 2 cores.',
                           unless_device_count_equals_to=2)

    a = constant_op.constant(
        np.array([[True, False, False, True], [False, False, False, True]]),
        dtype=dtypes.bool)
    sharded_a = api.relayout(a, self.first_dimension_sharded_layout_2d)
    unsharded_a = api.relayout(sharded_a, self.fully_replicated_layout_2d)

    self.assertDTensorEqual(a, self.fully_replicated_layout_2d, unsharded_a)

  def testAllToAllOnInt32(self):
    # TODO(b/193471732): Tracking the work to do int32 all-concat.
    #
    # Currently, the test will fail with segfault.
    self.skipForDeviceType(['GPU'],
                           'GPUs do not support int32 StridedSliceXXX Ops')
    self.skipForDeviceType(['TPU'],
                           'This test only needs to run on 2 cores.',
                           unless_device_count_equals_to=2)

    a = constant_op.constant(np.array([[1, 2], [3, 4]]), dtype=dtypes.int32)
    sharded_a = api.relayout(a, self.first_dimension_sharded_layout_2d)
    unsharded_a = api.relayout(sharded_a, self.fully_replicated_layout_2d)

    self.assertDTensorEqual(a, self.fully_replicated_layout_2d, unsharded_a)

  def testNoOpAllToAll(self):
    self.skipForDeviceType(['TPU'],
                           'This test only needs to run on 2 cores.',
                           unless_device_count_equals_to=2)

    a = constant_op.constant(
        np.array([[1., 2., 3., 4.], [5.0, 6.0, 7.0, 8.0]]),
        dtype=dtypes.float32)
    expected_result = a

    sharded_a = api.relayout(a, self.first_dimension_sharded_layout_2d)
    dtensor_result = api.relayout(sharded_a, self.fully_replicated_layout_2d)

    self.assertDTensorEqual(expected_result, self.fully_replicated_layout_2d,
                            dtensor_result)

  # Regression test for b/184401449.
  def testDeviceIdTensorOnSplitHost(self):
    if not test_util.is_tpu_present():
      self.skipTest('This test only runs on TPUs.')
    self.skipForDeviceType(['TPU'],
                           'This test requires 8 TPU cores.',
                           unless_device_count_equals_to=8)

    global_ids = test_util.create_device_ids_array((2, 4))
    local_ids = [0, 1, 4, 5, 2, 3, 6, 7]  # not sequentially increasing
    mesh = layout_lib.Mesh(_MESH_DIMS, global_ids, local_ids,
                           test_util.create_device_list((2, 4), 'TPU'),
                           'tpu_mesh')
    device = dtensor_device.DTensorDevice(meshes=[mesh])
    # This works because on 2x2, global device IDs are equal to physical TPU
    # core IDs: both are range(8). So local device IDs happen to be usable here.
    # TODO(b/180046115): Add a device.get_tpu_core_ids method and translate
    # device IDs to core IDs before setting the list here.
    device.set_tpu_core_ids('tpu_mesh', local_ids)
    layout_x = Layout.batch_sharded(mesh, _MESH_DIM_X, 2)
    layout_y = Layout.batch_sharded(mesh, _MESH_DIM_Y, 2)

    # Create a 2x4 batch-sharded d-tensor, with batch IDs in its first column
    # and zeros in other columns.
    # pylint: disable=g-complex-comprehension
    replica_ids = [
        constant_op.constant([loc[_MESH_DIM_X], 0, 0, 0],
                             dtype=dtypes.int32,
                             shape=[1, 4])
        for loc in mesh.local_device_locations()
    ]
    # pylint: enable=g-complex-comprehension
    replica_ids = device.pack(replica_ids, layout_x)

    # Create a 4x4 y-sharded d-tensor filled with ones.
    ones = [array_ops.ones([1, 4], dtype=dtypes.int32)] * 8
    ones = device.pack(ones, layout_y)

    # If `a` has a layout of [x, unsharded], and `b` has a layout of
    # [y, unsharded], the matmul will slice `a` to [x, y], do a local matmul,
    # and all-reduce to produce a final result with a layout of [x, unsharded].
    #
    # Because `a` only has non-zero values in its first column, local devices
    # must be given a correct device ID tensor (as the first argument of every
    # function) to produce correct `begin` values for slicing `a`.
    #
    # Although this function only contains a single op, running it in op-by-op
    # mode doesn't produce the intented effect because the output of
    # math_ops.matmul would have a layout of [y, unsharded] instead of
    # [x, unsharded].
    @polymorphic_function.function
    def func(a, b):
      return math_ops.matmul(a, b)

    dtensor_result = func(replica_ids, ones)

    # The correct result is a 2x4 batch-sharded d-tensor, with rows filled with
    # batch IDs.
    expected_result = [
        constant_op.constant(
            [loc[_MESH_DIM_X]] * 4, dtype=dtypes.int32, shape=[1, 4])
        for loc in mesh.local_device_locations()
    ]

    self.assertEqual(device.fetch_layout(dtensor_result), layout_x)
    dtensor_result = [t.numpy() for t in device.unpack(dtensor_result)]
    self.assertAllEqual(expected_result, dtensor_result)

  def testDifferentShapesBetweenCalls(self):
    self.skipForTfrt(
        'b/269333905, TFRT cpu fails due to step_id not propagated.'
    )
    self.skipForDeviceType(
        ['TPU'],
        'Known failure under TPU for legalization requires a static shape.',
    )

    # The error only happens across the batch, where the value of
    # tf.unique are differnet.
    def produce_data(inputs, label):
      inputs = api.relayout(
          inputs, Layout.batch_sharded(self.mesh, _MESH_DIM_X, 1)
      )
      label = api.relayout(
          label, Layout.batch_sharded(self.mesh, _MESH_DIM_X, 1)
      )
      return inputs, label

    @polymorphic_function.function
    def train_fn(inputs, label):
      inputs, indices = array_ops.unique(inputs)
      return math_ops.unsorted_segment_sum(label, indices, len(inputs))

    input1, label1 = produce_data([6, 0, 6, 0], [1, 2, 3, 4])
    input2, label2 = produce_data([2, 1, 2, 0], [1, 2, 3, 4])

    result1 = train_fn(input1, label1)
    result2 = train_fn(input2, label2)
    self.assertAllEqual(
        result1.numpy(),
        [
            4,
            6,
        ],
    )
    self.assertAllEqual(
        result2.numpy(),
        [
            4,
            2,
            4,
        ],
    )


class CollectiveTestWithCustomMesh(test_util.DTensorBaseTest):

  # Create two independent global AllReduce ops that should get combined.
  def testGlobalAllReduceCombiner(self):
    self.skipForDeviceType(['TPU'],
                           'This test requires 8 TPU cores.',
                           unless_device_count_equals_to=8)

    # Create and use an eight-device mesh just for this test.
    global_ids = test_util.create_device_ids_array((8,))
    local_ids = np.ravel(global_ids).tolist()
    mesh_dict = {
        device: layout_lib.Mesh([_MESH_DIM_X], global_ids, local_ids,
                                test_util.create_device_list((8,), device))
        for device in ('CPU', 'GPU', 'TPU')
    }

    mesh = self.configTestMesh(mesh_dict)
    fully_replicated_layout_1d = Layout.replicated(mesh, rank=1)
    first_dimension_sharded_layout_2d = Layout.batch_sharded(
        mesh, _MESH_DIM_X, 2)

    @polymorphic_function.function
    def func(a, b):
      a = math_ops.reduce_sum(a, axis=[0])
      b = math_ops.reduce_sum(b, axis=[0])
      # Do something with the results before adding them, to make sure the MLIR
      # pass can handle dependent ops sandwiched between two all-reduce ops.
      return gen_math_ops.square(a) + gen_math_ops.square(b)

    row = constant_op.constant(np.array([[1., 2.0]]), dtype=dtypes.float32)
    a = array_ops.repeat(row, repeats=[8], axis=0)
    b = gen_array_ops.reverse_v2(a, axis=[1])
    expected_result = func(a, b)

    a = api.relayout(a, first_dimension_sharded_layout_2d)
    b = api.relayout(b, first_dimension_sharded_layout_2d)
    dtensor_result = func(a, b)

    self.assertDTensorEqual(expected_result, fully_replicated_layout_1d,
                            dtensor_result)

  # Create two independent global AllReduce ops that should get combined.
  # Create two independent global AllReduce ops with different reductions, that
  # should not get combined

  def testGlobalAllReduceCombinerDifferentReduce(self):
    self.skipForDeviceType(['TPU'],
                           'This test requires 8 TPU cores.',
                           unless_device_count_equals_to=8)

    # Create and use an eight-device mesh just for this test.
    global_ids = test_util.create_device_ids_array((8,))
    local_ids = np.ravel(global_ids).tolist()
    mesh_dict = {
        device: layout_lib.Mesh([_MESH_DIM_X], global_ids, local_ids,
                                test_util.create_device_list((8,), device))
        for device in ('CPU', 'GPU', 'TPU')
    }

    mesh = self.configTestMesh(mesh_dict)
    fully_replicated_layout_1d = Layout.replicated(mesh, rank=1)
    first_dimension_sharded_layout_2d = Layout.batch_sharded(
        mesh, _MESH_DIM_X, 2)

    @polymorphic_function.function
    def func(a, b):
      a = math_ops.reduce_sum(a, axis=[0])
      b = math_ops.reduce_mean(b, axis=[0])

      # Do something with the results before adding them, to make sure the MLIR
      # pass can handle dependent ops sandwiched between two all-reduce ops.
      return gen_math_ops.square(a) + gen_math_ops.square(b)

    row = constant_op.constant(np.array([[1., 2.0]]), dtype=dtypes.float32)
    a = array_ops.repeat(row, repeats=[8], axis=0)
    b = gen_array_ops.reverse_v2(a, axis=[1])
    expected_result = func(a, b)

    a = api.relayout(a, first_dimension_sharded_layout_2d)
    b = api.relayout(b, first_dimension_sharded_layout_2d)
    dtensor_result = func(a, b)

    self.assertDTensorEqual(expected_result, fully_replicated_layout_1d,
                            dtensor_result)

  # Create two independent subgroup AllReduce ops that should get combined.
  def testSubgroupAllReduceCombiner(self):
    self.skipForDeviceType(['TPU'],
                           'This test requires 8 TPU cores.',
                           unless_device_count_equals_to=8)

    # Create and use an eight-device mesh just for this test.
    global_ids = test_util.create_device_ids_array((4, 2))
    local_ids = np.ravel(global_ids).tolist()
    mesh_dict = {
        device: layout_lib.Mesh(_MESH_DIMS, global_ids, local_ids,
                                test_util.create_device_list((4, 2), device))
        for device in ('CPU', 'GPU', 'TPU')
    }

    mesh = self.configTestMesh(mesh_dict)
    fully_sharded_layout_2d = Layout(_MESH_DIMS, mesh)

    @polymorphic_function.function
    def func(a, b):
      a = math_ops.reduce_sum(a, axis=[0])
      b = math_ops.reduce_sum(b, axis=[0])
      # Do something with the results before adding them, to make sure the MLIR
      # pass can handle dependent ops sandwiched between two all-reduce ops.
      return gen_math_ops.square(a) + gen_math_ops.square(b)

    row = constant_op.constant(np.array([[1., 2.0]]), dtype=dtypes.float32)
    a = array_ops.repeat(row, repeats=[8], axis=0)
    b = gen_array_ops.reverse_v2(a, axis=[1])
    expected_result = func(a, b)

    a = api.relayout(a, fully_sharded_layout_2d)
    b = api.relayout(b, fully_sharded_layout_2d)
    dtensor_result = func(a, b)

    self.assertDTensorEqual(expected_result, Layout([_MESH_DIM_Y], mesh),
                            dtensor_result)

  # TODO(b/188605096): also add a MixedPrecisionReduceScatter test
  def testMixedPrecisionAllReduce(self):
    has_enable_dtensor_mixed_precision_reduce = (
        'DTENSOR_ENABLE_MIXED_PRECISION_REDUCE' in os.environ
    )
    has_dtensor_reduce_in_bfloat16_max_group_size = (
        'DTENSOR_REDUCE_IN_BFLOAT16_MAX_GROUP_SIZE' in os.environ
    )
    if has_dtensor_reduce_in_bfloat16_max_group_size:
      old_dtensor_reduce_in_bfloat16_max_group_size = os.environ[
          'DTENSOR_REDUCE_IN_BFLOAT16_MAX_GROUP_SIZE']
    os.environ['DTENSOR_ENABLE_MIXED_PRECISION_REDUCE'] = ''
    os.environ['DTENSOR_REDUCE_IN_BFLOAT16_MAX_GROUP_SIZE'] = '4'

    self.skipForDeviceType(['GPU'],
                           'GPUs do not support bfloat16 reduce')
    self.skipForDeviceType(['TPU'],
                           'This test requires 8 TPU cores.',
                           unless_device_count_equals_to=8)

    # Create and use an 8-device mesh just for this test. Mixed-precision
    # AllReduce will be in effect since the reduction will be across 8 devices
    # which is larger than the max group size flag value of 4.
    global_ids = test_util.create_device_ids_array((8,))
    local_ids = np.ravel(global_ids).tolist()
    mesh_dict = {
        device: layout_lib.Mesh([_MESH_DIM_X], global_ids, local_ids,
                                test_util.create_device_list((8,), device))
        for device in ('CPU', 'GPU', 'TPU')
    }

    mesh = self.configTestMesh(mesh_dict)
    replicated_layout_1d = Layout.replicated(mesh, rank=1)
    first_dim_sharded_layout_1d = Layout.batch_sharded(
        mesh, _MESH_DIM_X, rank=2)

    @polymorphic_function.function
    def func(x):
      return math_ops.reduce_sum(x, axis=0)

    # Reduce across 8 devices.
    inp = constant_op.constant(
        np.arange(48.).reshape((8, 6)), dtype=dtypes.bfloat16)
    expected_result = np.sum(inp, axis=0)

    inp_dtensor = api.relayout(inp, first_dim_sharded_layout_1d)
    dtensor_result = func(inp_dtensor)

    self.assertDTensorEqual(
        expected_result, replicated_layout_1d, dtensor_result)

    if not has_enable_dtensor_mixed_precision_reduce:
      del os.environ['DTENSOR_ENABLE_MIXED_PRECISION_REDUCE']
    if has_dtensor_reduce_in_bfloat16_max_group_size:
      os.environ['DTENSOR_REDUCE_IN_BFLOAT16_MAX_GROUP_SIZE'] = (
          old_dtensor_reduce_in_bfloat16_max_group_size
      )
    else:
      del os.environ['DTENSOR_REDUCE_IN_BFLOAT16_MAX_GROUP_SIZE']


if __name__ == '__main__':
  test.main()
