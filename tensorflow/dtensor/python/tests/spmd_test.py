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
"""Tests for the open source DTensor Python API."""

import os
from unittest import mock
from absl.testing import parameterized
import numpy as np

# pylint: disable=g-direct-tensorflow-import
from tensorflow.dtensor.python import api
from tensorflow.dtensor.python import d_variable
from tensorflow.dtensor.python import layout as layout_lib
from tensorflow.dtensor.python import numpy_util
from tensorflow.dtensor.python.tests import test_util
from tensorflow.dtensor.python.tests import test_util_ops
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager.polymorphic_function import polymorphic_function
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_bitwise_ops
from tensorflow.python.ops import gen_io_ops
from tensorflow.python.ops import gen_linalg_ops
from tensorflow.python.ops import gen_list_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import gen_stateless_random_ops
from tensorflow.python.ops import gen_string_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import special_math_ops
from tensorflow.python.ops import stateless_random_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test as tf_test
from tensorflow.python.util import nest
# pylint: enable=g-direct-tensorflow-import

# Makes a 2-D mesh with dimensions as, X(2) and Y(4).
_MESH_DIM_X = 'x'
_MESH_DIM_Y = 'y'
_MESH_DIMS = [_MESH_DIM_X, _MESH_DIM_Y]

_MATMUL_IMPLEMENTED = (('_unsharded', 0,
                        0), ('_a_unsharded_b_contracting', 0,
                             1), ('_a_unsharded_b_non_contracting', 0,
                                  2), ('_a_non_contracting_b_unsharded', 1,
                                       0), ('_a_contracting_b_unsharded', 2, 0),
                       ('_a_contracting_b_contracting', 2,
                        1), ('_a_non_contracting_b_contracting', 1,
                             1), ('_a_non_contracting_b_non_contracting', 1, 2),
                       ('_a_contracting_b_non_contracting', 2, 2))
_BATCH_MATMUL_IMPLEMENTED = (('_unsharded', 0,
                              0), ('_a_unsharded_b_contracting', 0,
                                   2), ('_a_unsharded_b_non_contracting', 0,
                                        3), ('_a_batch_b_batch', 1, 1),
                             ('_a_non_contracting_b_unsharded', 2,
                              0), ('_a_contracting_b_unsharded', 3,
                                   0), ('_a_contracting_b_contracting', 3, 2),
                             ('_a_non_contracting_b_contracting', 2,
                              2), ('_a_non_contracting_b_non_contracting', 2,
                                   3), ('_a_contracting_b_non_contracting', 3,
                                        3), ('_a_unsharded_b_batch', 0,
                                             1), ('_a_batch_b_unsharded', 1, 0),
                             ('_a_batch_b_contracting', 1,
                              2), ('_a_batch_b_non_contracting', 1,
                                   3), ('_a_non_contracting_b_batch', 2,
                                        1), ('_a_contracting_b_batch', 3, 1))
_MATMUL_TRANSPOSE = (('', False, False), ('_b_transpose', False, True),
                     ('_a_transpose', True, False), ('_a_transpose_b_transpose',
                                                     True, True))

Layout = layout_lib.Layout
Mesh = layout_lib.Mesh
UNSHARDED = layout_lib.UNSHARDED


def select_tol(op, mesh, default_tol, low_res_tol):
  # Lowers the tol for math_ops.pow,
  # nn_ops.log_softmax_v2 and gen_math_ops.tanh due to
  # resolution on TPU
  if (op not in [
      math_ops.pow, nn_ops.log_softmax_v2, gen_math_ops.tanh,
      gen_math_ops.acosh, gen_math_ops.asinh, gen_math_ops.digamma,
      gen_math_ops.igammac, gen_math_ops.lgamma, gen_math_ops.log1p,
      math_ops.xlog1py, gen_math_ops.xlogy, gen_math_ops.zeta, gen_math_ops.tan,
      gen_math_ops.sin, gen_math_ops.sinh, math_ops.softplus
  ]):
    return default_tol

  if 'TPU' in mesh.local_devices()[0]:
    return low_res_tol
  else:
    return default_tol


def order_broadcastable_operands(op, lhs, rhs):
  # Swaps operands lhs and rhs. Assumes lhs is the broadcasting tensor. Due to
  # ops only with right broadcastable operand like gen_math_ops.truncate_div
  if (op in [gen_math_ops.truncate_div, gen_math_ops.truncate_mod]):
    return rhs, lhs
  return lhs, rhs


class DTensorSPMDTest(test_util.DTensorBaseTest):

  def setUp(self):
    super(DTensorSPMDTest, self).setUp()

    self.skipForDeviceType(['TPU'],
                           'all tests require 8 TPU cores.',
                           unless_device_count_equals_to=8)

    global_ids = test_util.create_device_ids_array((2, 4))
    local_ids = np.ravel(global_ids).tolist()
    mesh_dict = dict()
    for device in ('CPU', 'GPU', 'TPU'):
      mesh_dict[device] = Mesh(
          [_MESH_DIM_X, _MESH_DIM_Y],
          global_ids,
          local_ids,
          test_util.create_device_list((2, 4), device),
          use_xla_spmd=test_util.get_use_xla_spmd(device),
      )
    self.mesh = self.configTestMesh(mesh_dict)

    # Creates a bunch of common layouts used by tests later.
    # - 0-d
    self.scalar_replicated_layout = Layout.replicated(self.mesh, rank=0)
    # - 1-d
    self.replicated_layout_1d = Layout.replicated(self.mesh, rank=1)
    self.first_dimension_sharded_layout_1d = Layout.batch_sharded(
        self.mesh, _MESH_DIM_X, rank=1)
    # - 2-d
    self.replicated_layout_2d = Layout.replicated(self.mesh, rank=2)
    self.first_dimension_sharded_layout = Layout.batch_sharded(
        self.mesh, _MESH_DIM_X, rank=2)
    self.last_dimension_sharded_layout = Layout.inner_sharded(
        self.mesh, _MESH_DIM_X, rank=2)

    self.layouts_2d = [
        self.replicated_layout_2d, self.first_dimension_sharded_layout,
        self.last_dimension_sharded_layout
    ]

    # - 3-d
    self.replicated_layout_3d = Layout.replicated(self.mesh, rank=3)
    self.first_dimension_sharded_layout_3d = Layout.batch_sharded(
        self.mesh, _MESH_DIM_X, rank=3)
    self.middle_dimension_sharded_layout_3d = Layout(
        [layout_lib.UNSHARDED, _MESH_DIM_X, layout_lib.UNSHARDED], self.mesh)
    self.last_dimension_sharded_layout_3d = Layout.inner_sharded(
        self.mesh, _MESH_DIM_X, rank=3)

    self.layouts_3d = [
        self.replicated_layout_3d, self.first_dimension_sharded_layout_3d,
        self.middle_dimension_sharded_layout_3d,
        self.last_dimension_sharded_layout_3d
    ]

    self.shardings = {
        'batch': Layout.batch_sharded,
        'inner': Layout.inner_sharded
    }

  @parameterized.named_parameters(
      ('unsharded_unsharded', [layout_lib.UNSHARDED, layout_lib.UNSHARDED]),
      ('x_unsharded', [_MESH_DIM_X, layout_lib.UNSHARDED]),
      ('unsharded_x', [layout_lib.UNSHARDED, _MESH_DIM_X]),
      ('x,y', [_MESH_DIM_X, _MESH_DIM_Y]),
  )
  @mock.patch.dict(
      os.environ, {'DTENSOR_ENABLE_REPLICATED_SPMD_AS_DEFAULT_TF.MOD': '1'}
  )
  def testDefaultReplicatedSpmd(self, shard_specs):
    x = stateless_random_ops.stateless_random_uniform(
        shape=[4, 8], seed=[0, 1], dtype=dtypes.float32
    )
    y = constant_op.constant(7, dtype=dtypes.float32)

    expected_result = math_ops.Mod(x=x, y=y)
    expected_layout = Layout.replicated(self.mesh, rank=2)
    dtensor_result = math_ops.Mod(
        x=api.relayout(x, layout=Layout(shard_specs, self.mesh)),
        y=api.relayout(y, layout=Layout([], self.mesh)),
    )

    self.assertDTensorEqual(expected_result, expected_layout, dtensor_result)

  @parameterized.product(
      shard_type=['replicated', 'batch_sharded'], full_matrices=[True, False])
  def testQR(self, shard_type, full_matrices):
    np.random.seed(123)
    inputs = constant_op.constant(
        np.random.normal(0.0, 1.0, 8 * 9 * 10).reshape([8, 9, 10]),
        dtype=dtypes.float32)

    expected_result = gen_linalg_ops.qr(
        input=inputs, full_matrices=True, name=None)

    if shard_type == 'replicated':
      layout = self.first_dimension_sharded_layout_3d
    else:
      layout = self.replicated_layout_3d

    inputs = api.relayout(inputs, layout)

    got = gen_linalg_ops.qr(
        input=inputs, full_matrices=full_matrices, name=None)
    self.assertDTensorEqual(expected_result[0], layout, got[0])
    self.assertDTensorEqual(expected_result[1], layout, got[1])

  def testReduceScatter(self,):
    # Generates an AllReduce due to sharding of inner dimensions of Matmul
    # and a Scatter due to the Relayout.  The AllReduce+Scatter can be combined
    # to a single ReduceScatter.
    a, b, c = 128, 128, 128
    seed = [0, 1]
    first_dim_sharded = self.first_dimension_sharded_layout
    second_dim_sharded = self.last_dimension_sharded_layout

    with api.default_mesh(self.mesh):
      m1 = numpy_util.stateless_random_uniform(
          layout=second_dim_sharded, shape=[a, b], seed=seed
      )
      m2 = numpy_util.stateless_random_uniform(
          layout=first_dim_sharded, shape=[b, c], seed=seed
      )

    @polymorphic_function.function
    def func():
      m3 = math_ops.matmul(m1, m2)
      return m3

    @polymorphic_function.function
    def scattered_func():
      m3 = math_ops.matmul(m1, m2)
      return api.relayout(m3, self.first_dimension_sharded_layout)

    dtensor_result = func()
    dtensor_scattered_result = scattered_func()

    self.assertDTensorEqual(dtensor_result, self.first_dimension_sharded_layout,
                            dtensor_scattered_result)

  def testReduceScatterLastDimSharded(
      self,
  ):
    # ReduceScatter on non-0th dimension which requires a transpose.
    a, b, c = 128, 128, 128
    seed = [0, 1]
    first_dim_sharded = self.first_dimension_sharded_layout
    second_dim_sharded = self.last_dimension_sharded_layout

    @polymorphic_function.function
    def uniform(shape, seed, layout):
      return api.relayout(
          stateless_random_ops.stateless_random_uniform(shape=shape, seed=seed),
          layout=layout,
      )

    with api.default_mesh(self.mesh):
      m1 = uniform(layout=second_dim_sharded, shape=[a, b], seed=seed)
      m2 = uniform(layout=first_dim_sharded, shape=[b, c], seed=seed)

    @polymorphic_function.function
    def func():
      m3 = math_ops.matmul(m1, m2)
      return m3

    @polymorphic_function.function
    def scattered_func():
      m3 = math_ops.matmul(m1, m2)
      return api.relayout(m3, self.last_dimension_sharded_layout)

    dtensor_result = func()
    dtensor_scattered_result = scattered_func()

    self.assertDTensorEqual(
        dtensor_result,
        self.last_dimension_sharded_layout,
        dtensor_scattered_result,
    )

  @parameterized.named_parameters(
      (
          'xu_ux',
          [_MESH_DIM_X, layout_lib.UNSHARDED],
          [layout_lib.UNSHARDED, _MESH_DIM_X],
      ),
      (
          'ux_xu',
          [layout_lib.UNSHARDED, _MESH_DIM_X],
          [_MESH_DIM_X, layout_lib.UNSHARDED],
      ),
      (
          'yu_uy',
          [_MESH_DIM_Y, layout_lib.UNSHARDED],
          [layout_lib.UNSHARDED, _MESH_DIM_Y],
      ),
      (
          'uy_yu',
          [layout_lib.UNSHARDED, _MESH_DIM_Y],
          [_MESH_DIM_Y, layout_lib.UNSHARDED],
      ),
  )
  def testAllToAll2D(self, src_spec, tgt_spec):
    a = constant_op.constant(
        np.arange(
            8 * 8,
        ).reshape((8, 8)),
        dtype=dtypes.float32,
    )
    sharded_a = numpy_util.pack_numpy(a, layout=Layout(src_spec, self.mesh))

    @polymorphic_function.function
    def func(a):
      return api.relayout(a, Layout(tgt_spec, self.mesh))

    dtensor_result = func(sharded_a)
    self.assertDTensorEqual(a, Layout(tgt_spec, self.mesh), dtensor_result)

  @parameterized.named_parameters(
      (
          'yuu_uuy',
          [_MESH_DIM_Y, layout_lib.UNSHARDED, layout_lib.UNSHARDED],
          [layout_lib.UNSHARDED, layout_lib.UNSHARDED, _MESH_DIM_Y],
      ),
      (
          'xuu_uux',
          [_MESH_DIM_X, layout_lib.UNSHARDED, layout_lib.UNSHARDED],
          [layout_lib.UNSHARDED, layout_lib.UNSHARDED, _MESH_DIM_X],
      ),
      (
          'uux_xuu',
          [layout_lib.UNSHARDED, layout_lib.UNSHARDED, _MESH_DIM_X],
          [_MESH_DIM_X, layout_lib.UNSHARDED, layout_lib.UNSHARDED],
      ),
      (
          'xuu_uxu',
          [_MESH_DIM_X, layout_lib.UNSHARDED, layout_lib.UNSHARDED],
          [layout_lib.UNSHARDED, _MESH_DIM_X, layout_lib.UNSHARDED],
      ),
      (
          'uxu_xuu',
          [layout_lib.UNSHARDED, _MESH_DIM_X, layout_lib.UNSHARDED],
          [_MESH_DIM_X, layout_lib.UNSHARDED, layout_lib.UNSHARDED],
      ),
      (
          'xuy_uxy',
          [_MESH_DIM_X, layout_lib.UNSHARDED, _MESH_DIM_Y],
          [layout_lib.UNSHARDED, _MESH_DIM_X, _MESH_DIM_Y],
      ),
      (
          'uxy_xuy',
          [layout_lib.UNSHARDED, _MESH_DIM_X, _MESH_DIM_Y],
          [_MESH_DIM_X, layout_lib.UNSHARDED, _MESH_DIM_Y],
      ),
      (
          'xyu_uyx',
          [_MESH_DIM_X, _MESH_DIM_Y, layout_lib.UNSHARDED],
          [layout_lib.UNSHARDED, _MESH_DIM_Y, _MESH_DIM_X],
      ),
      # Requires additional transpose
      (
          'uxu_uux',
          [layout_lib.UNSHARDED, _MESH_DIM_X, layout_lib.UNSHARDED],
          [layout_lib.UNSHARDED, layout_lib.UNSHARDED, _MESH_DIM_X],
      ),
      (
          'uux_uxu',
          [layout_lib.UNSHARDED, layout_lib.UNSHARDED, _MESH_DIM_X],
          [layout_lib.UNSHARDED, _MESH_DIM_X, layout_lib.UNSHARDED],
      ),
      (
          'xyu_xuy',
          [_MESH_DIM_X, _MESH_DIM_Y, layout_lib.UNSHARDED],
          [_MESH_DIM_X, layout_lib.UNSHARDED, _MESH_DIM_Y],
      ),
      (
          'xuy_xyu',
          [_MESH_DIM_X, layout_lib.UNSHARDED, _MESH_DIM_Y],
          [_MESH_DIM_X, _MESH_DIM_Y, layout_lib.UNSHARDED],
      ),
      (
          'yxu_yux',
          [_MESH_DIM_Y, _MESH_DIM_X, layout_lib.UNSHARDED],
          [_MESH_DIM_Y, layout_lib.UNSHARDED, _MESH_DIM_X],
      ),
      (
          'yux_yxu',
          [_MESH_DIM_Y, layout_lib.UNSHARDED, _MESH_DIM_X],
          [_MESH_DIM_Y, _MESH_DIM_X, layout_lib.UNSHARDED],
      ),
  )
  def testAllToAll3D(self, src_spec, tgt_spec):
    a = constant_op.constant(
        np.arange(8 * 8 * 8).reshape((8, 8, 8)), dtype=dtypes.float32
    )
    sharded_a = numpy_util.pack_numpy(a, layout=Layout(src_spec, self.mesh))

    @polymorphic_function.function
    def func(a):
      return api.relayout(a, Layout(tgt_spec, self.mesh))

    dtensor_result = func(sharded_a)

    self.assertDTensorEqual(a, Layout(tgt_spec, self.mesh), dtensor_result)

  def testExpandDimsDifferentInputAndOutputLayouts(self,):
    src_numpy = np.random.uniform(size=[10, 10])
    src = constant_op.constant(src_numpy, dtype=dtypes.float32)

    expected = array_ops.expand_dims_v2(src, axis=-1)

    src = api.relayout(src, self.replicated_layout_2d)

    @polymorphic_function.function
    def expand_dims_fn(src):
      expanded = array_ops.expand_dims_v2(src, axis=-1)
      return api.relayout(expanded, self.first_dimension_sharded_layout_3d)

    dtensor_result = expand_dims_fn(src)
    self.assertDTensorEqual(expected, self.first_dimension_sharded_layout_3d,
                            dtensor_result)

    @polymorphic_function.function
    def expand_dims_list_axis_fn(src):
      expanded = array_ops.expand_dims_v2(src, axis=[-1])
      return api.relayout(expanded, self.first_dimension_sharded_layout_3d)

    dtensor_result_2 = expand_dims_list_axis_fn(src)
    self.assertDTensorEqual(expected, self.first_dimension_sharded_layout_3d,
                            dtensor_result_2)

  def testPackAndUnpackAssertion(self):
    layout = Layout.replicated(self.mesh, rank=3)
    # Due to Perf concerns, `pack` does not check the compatibility of
    # components and layout. Here, we inject a wrong value components.
    with api.default_mesh(self.mesh):
      b = api.pack(
          [constant_op.constant([[[(x + 1) * 1.0]]]) for x in range(8)],
          layout=layout)
      assert b.shape == [1, 1, 1]

    # `to_numpy` assumes all unpacked tensors are compatible with the
    # layout. So, it picks any component to use if that dimension is replicated.
    # In this case, it picks the final one.
    result_dtensor = numpy_util.to_numpy(b)

    self.assertAllEqual(constant_op.constant([[[8.]]]), result_dtensor)

    # assertDTensorEqual does more aggressive check, which respects the layout.
    with self.assertRaisesRegex(AssertionError, 'Mismatched value'):
      self.assertDTensorEqual(constant_op.constant([[[8.]]]), layout, b)

  @parameterized.named_parameters(test_util_ops.UNARY_OPS)
  def testUnaryOpsWithTwoShardedAndOneReplicatedDimension(self, op):
    a = constant_op.constant([[[1.], [2.], [3.], [4.]], [[5.], [6.], [7.],
                                                         [8.]]])
    assert a.shape == [2, 4, 1]
    expected_result = op(a)

    layout = Layout([_MESH_DIM_X, _MESH_DIM_Y, layout_lib.UNSHARDED], self.mesh)
    a = api.relayout(a, layout)
    dtensor_result = op(a)

    tol = select_tol(op, self.mesh, test_util.DEFAULT_TOL, 1e-4)
    self.assertDTensorEqual(expected_result, layout, dtensor_result, tol=tol)

  @parameterized.named_parameters(test_util_ops.UNARY_OPS)
  def testUnaryOpsWithFullyReplicatedInputs(self, op):
    a = constant_op.constant([[1., 2.], [3., 4.]])
    assert a.shape == [2, 2]
    expected_result = op(a)

    a = api.copy_to_mesh(a, self.replicated_layout_2d)
    dtensor_result = op(a)

    tol = select_tol(op, self.mesh, test_util.DEFAULT_TOL, 1e-4)
    self.assertDTensorEqual(
        expected_result, self.replicated_layout_2d, dtensor_result, tol=tol)

  @parameterized.named_parameters(test_util_ops.UNARY_OPS)
  def testUnaryOpsWithFullyShardedInputs(self, op):
    a = constant_op.constant(
        np.arange(16).reshape((2, 4, 2)), dtype=dtypes.float32)
    expected_result = op(a)

    sharded_layout = Layout([_MESH_DIM_X, _MESH_DIM_Y, layout_lib.UNSHARDED],
                            self.mesh)
    a = api.relayout(a, sharded_layout)
    dtensor_result = op(a)

    tol = select_tol(op, self.mesh, test_util.DEFAULT_TOL, 1e-4)
    self.assertDTensorEqual(
        expected_result, sharded_layout, dtensor_result, tol=tol)

  @parameterized.named_parameters(test_util_ops.UNARY_OPS)
  def testUnaryOpsWithBatchShardedInputs(self, op):
    tol = select_tol(op, self.mesh, test_util.DEFAULT_TOL, 1e-3)
    a = constant_op.constant(np.arange(6).reshape((2, 3)), dtype=dtypes.float32)
    expected_result = op(a)

    a = api.relayout(a, self.first_dimension_sharded_layout)
    dtensor_result = op(a)

    self.assertDTensorEqual(
        expected_result,
        self.first_dimension_sharded_layout,
        dtensor_result,
        tol=tol)

  def testInvertOpsWithFullyShardedInputs(self):
    # Invert only support int inputs.
    op = lambda x: gen_bitwise_ops.invert(x=x, name='Invert')

    a = constant_op.constant(
        np.arange(16).reshape((2, 4, 2)), dtype=dtypes.int32)
    expected_result = op(a)

    sharded_layout = Layout([_MESH_DIM_X, _MESH_DIM_Y, layout_lib.UNSHARDED],
                            self.mesh)
    a = api.relayout(a, sharded_layout)
    dtensor_result = op(a)

    tol = select_tol(op, self.mesh, test_util.DEFAULT_TOL, 1e-4)
    self.assertDTensorEqual(
        expected_result, sharded_layout, dtensor_result, tol=tol)

  @parameterized.named_parameters(('replicated', layout_lib.UNSHARDED),
                                  ('sharded', _MESH_DIM_X))
  def testInvertPermutationOp(self, shard):
    self.skipForDeviceType(['GPU', 'TPU'],
                           'Invert Permutation runs in CPU only.')
    op_input = constant_op.constant([3, 4, 0, 2, 1, 5])
    expected_result = gen_array_ops.invert_permutation(op_input)
    # We should always expected the output to be replicated as the
    # expander should relayout both inputs and outputs to replicated.
    expected_layout = Layout.replicated(self.mesh, rank=1)

    self.assertDTensorEqual(
        expected_result,
        expected_layout,
        gen_array_ops.invert_permutation(
            api.relayout(op_input, Layout([shard], self.mesh))
        ),
    )

  def testErfcInvOpsWithFullyShardedInputs(self):
    # By official doc, math_ops.erfcinv is defined on (0, 2]. In addition,
    # math_ops.erfcinv internally calls ndtri internally. So to test the op for
    # spmd expanding, we call raw op here.
    op = lambda x: gen_math_ops.erfinv(x=x, name='erfinv')

    a = constant_op.constant(
        np.arange(16).reshape((2, 4, 2)) / 30 + 0.1, dtype=dtypes.float32)
    expected_result = op(a)

    sharded_layout = Layout([_MESH_DIM_X, _MESH_DIM_Y, layout_lib.UNSHARDED],
                            self.mesh)
    a = api.relayout(a, sharded_layout)
    dtensor_result = op(a)

    tol = select_tol(op, self.mesh, test_util.DEFAULT_TOL, 1e-4)
    self.assertDTensorEqual(
        expected_result, sharded_layout, dtensor_result, tol=tol)

  def testPopulationCountWithFullyShardedInputs(self):
    # By official doc, gen_bitwise_ops.population_count only supports int
    # inputs.
    op = lambda x: gen_bitwise_ops.population_count(x=x, name='pc')

    a = constant_op.constant(
        np.arange(16).reshape((2, 4, 2)), dtype=dtypes.int32)
    expected_result = op(a)

    sharded_layout = Layout([_MESH_DIM_X, _MESH_DIM_Y, layout_lib.UNSHARDED],
                            self.mesh)
    a = api.relayout(a, sharded_layout)
    dtensor_result = op(a)

    tol = select_tol(op, self.mesh, test_util.DEFAULT_TOL, 1e-4)
    self.assertDTensorEqual(
        expected_result, sharded_layout, dtensor_result, tol=tol)

  def testIgammacOpsWithFullyShardedInputs(self):
    # Igammac has super low precision on TPU. So we test it as a separated unit
    # tests to avoid lower the tol of other tests.
    #
    # In addition, according to wiki link below, for s=4, all values are not
    # inf/nan.
    #
    # https://en.wikipedia.org/wiki/Incomplete_gamma_function
    tol = 1e-2
    op = lambda x: gen_math_ops.igammac(4, x)

    a = constant_op.constant(
        np.arange(16).reshape((2, 4, 2)), dtype=dtypes.float32)
    expected_result = op(a)

    sharded_layout = Layout([_MESH_DIM_X, _MESH_DIM_Y, layout_lib.UNSHARDED],
                            self.mesh)
    a = api.relayout(a, sharded_layout)
    dtensor_result = op(a)

    self.assertDTensorEqual(
        expected_result, sharded_layout, dtensor_result, tol=tol)

  @parameterized.parameters(('replicated',), ('sharded',))
  def testBiasAdd2D(self, shard_type):
    value = np.array([[1., 2.], [3., 4.]])
    bias = np.array([0.1, 0.2])
    expected_result = nn_ops.bias_add(value, bias)

    if shard_type == 'replicated':
      layout = self.replicated_layout_2d
    else:
      layout = self.first_dimension_sharded_layout

    value = api.relayout(value, layout)
    bias = api.relayout(bias, self.replicated_layout_1d)
    dtensor_result = nn_ops.bias_add(value, bias)
    self.assertDTensorEqual(expected_result, layout, dtensor_result)

  @parameterized.product(
      shard_type=['replicated', 'batch_sharded'],
      data_format=['N...C', 'NC...'])
  def testBiasAdd4D(self, shard_type, data_format):
    value = np.ones(shape=(6, 2, 4, 2), dtype=np.float32)
    bias = np.array([0.1, 0.2], dtype=np.float32)
    expected_result = nn_ops.bias_add(value, bias, data_format=data_format)

    if shard_type == 'replicated':
      layout = Layout.replicated(self.mesh, rank=4)
    else:
      layout = Layout.batch_sharded(self.mesh, _MESH_DIM_X, rank=4)

    value = api.relayout(value, layout)
    bias = api.relayout(bias, self.replicated_layout_1d)

    dtensor_result = nn_ops.bias_add(value, bias, data_format=data_format)
    self.assertDTensorEqual(expected_result, layout, dtensor_result)

  @parameterized.product(
      data_format=['N...C', 'NC...'],
      bias_sharding=['x', 'y', layout_lib.UNSHARDED],
      c_dim_sharding=['x', layout_lib.UNSHARDED])
  def testBiasAddDataFormatTest(self, data_format, bias_sharding,
                                c_dim_sharding):
    if data_format == 'N...C':
      c_dim = 3
      input_sharding = [
          layout_lib.UNSHARDED, layout_lib.UNSHARDED, 'y', c_dim_sharding
      ]
      a = np.ones(shape=(1, 1, 4, 4), dtype=np.float32)
      layout = Layout(input_sharding, self.mesh)
    else:
      c_dim = 1
      input_sharding = [
          layout_lib.UNSHARDED, c_dim_sharding, 'y', layout_lib.UNSHARDED
      ]
      a = np.ones(shape=(1, 4, 4, 1), dtype=np.float32)
      layout = Layout(input_sharding, self.mesh)

    bias = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    expected_result = nn_ops.bias_add(a, bias, data_format=data_format)
    expected_result_sharding = input_sharding
    if c_dim_sharding == layout_lib.UNSHARDED and bias_sharding != 'y':
      expected_result_sharding[c_dim] = bias_sharding

    expected_layout = Layout(expected_result_sharding, self.mesh)
    a = api.relayout(a, layout)
    bias = api.relayout(bias, Layout([bias_sharding], self.mesh))
    result = nn_ops.bias_add(a, bias=bias, data_format=data_format)

    self.assertDTensorEqual(expected_result, expected_layout, result)

  @parameterized.parameters(('replicated',), ('batch_sharded',))
  def testBiasAddGrad2D(self, shard_type):
    value = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    expected_result = gen_nn_ops.bias_add_grad(out_backprop=value)

    if shard_type == 'replicated':
      layout = self.replicated_layout_2d
    else:
      layout = self.first_dimension_sharded_layout
    expected_layout = self.replicated_layout_1d

    value = api.relayout(value, layout)
    dtensor_result = gen_nn_ops.bias_add_grad(out_backprop=value)
    self.assertDTensorEqual(expected_result, expected_layout, dtensor_result)

  @parameterized.product(
      shard_type=['replicated', 'batch_sharded'], data_format=['NHWC', 'NCHW'])
  def testBiasAddGrad4D(self, shard_type, data_format):
    value = np.ones(shape=(2, 3, 4, 5), dtype=np.float32)
    expected_result = gen_nn_ops.bias_add_grad(
        out_backprop=value, data_format=data_format)

    if shard_type == 'replicated':
      layout = Layout.replicated(self.mesh, rank=4)
    else:
      layout = Layout.batch_sharded(self.mesh, _MESH_DIM_X, rank=4)
    expected_layout = self.replicated_layout_1d

    value = api.relayout(value, layout)
    dtensor_result = gen_nn_ops.bias_add_grad(
        out_backprop=value, data_format=data_format)
    self.assertDTensorEqual(expected_result, expected_layout, dtensor_result)

  @parameterized.named_parameters(test_util_ops.BINARY_FLOAT_OPS)
  def testBinaryOpsWithFullyReplicatedInputs(self, op):
    tol = select_tol(op, self.mesh, test_util.DEFAULT_TOL, low_res_tol=1e-2)
    a = constant_op.constant([[1., 2.], [3., 4.]])
    b = constant_op.constant([[10., 20.], [30., 40.]])
    expected_result = op(a, b)

    a = api.copy_to_mesh(a, self.replicated_layout_2d)
    b = api.copy_to_mesh(b, self.replicated_layout_2d)
    dtensor_result = op(a, b)

    self.assertDTensorEqual(
        expected_result, self.replicated_layout_2d, dtensor_result, tol=tol)

  @parameterized.named_parameters(test_util_ops.BINARY_FLOAT_OPS)
  def testBinaryFloatOpsWithFullyShardedInputs(self, op):
    tol = select_tol(op, self.mesh, test_util.DEFAULT_TOL, low_res_tol=1e-2)
    a = constant_op.constant(np.arange(8).reshape((2, 4)), dtype=dtypes.float32)
    b = constant_op.constant(
        np.arange(8).reshape((2, 4)) + 10.0, dtype=dtypes.float32)
    expected_result = op(a, b)

    sharded_layout_2d = Layout([_MESH_DIM_X, _MESH_DIM_Y], self.mesh)
    a = api.relayout(a, sharded_layout_2d)
    b = api.relayout(b, sharded_layout_2d)
    dtensor_result = op(a, b)

    self.assertDTensorEqual(
        expected_result, sharded_layout_2d, dtensor_result, tol=tol)

  @parameterized.named_parameters(test_util_ops.BINARY_BOOL_OPS)
  def testBinaryBoolOpsWithFullyShardedInputs(self, op):
    a = array_ops.reshape(
        constant_op.constant(
            [True, False, True, False, True, False, True, False]), [2, 4])
    b = array_ops.reshape(
        constant_op.constant(
            [True, True, True, True, False, False, False, False]), [2, 4])
    expected_result = op(a, b)

    sharded_layout_2d = Layout([_MESH_DIM_X, _MESH_DIM_Y], self.mesh)
    a = api.relayout(a, sharded_layout_2d)
    b = api.relayout(b, sharded_layout_2d)
    dtensor_result = op(a, b)

    self.assertDTensorEqual(expected_result, sharded_layout_2d, dtensor_result)

  @parameterized.named_parameters(test_util_ops.BINARY_INT_OPS)
  def testBinaryIntOpsWithFullyShardedInputs(self, op):
    a = constant_op.constant(np.arange(8).reshape((2, 4)))
    b = constant_op.constant(np.arange(8).reshape((2, 4)) + 1)
    expected_result = op(a, b)

    sharded_layout_2d = Layout([_MESH_DIM_X, _MESH_DIM_Y], self.mesh)
    a = api.relayout(a, sharded_layout_2d)
    b = api.relayout(b, sharded_layout_2d)
    dtensor_result = op(a, b)

    self.assertDTensorEqual(expected_result, sharded_layout_2d, dtensor_result)

  @parameterized.named_parameters(test_util_ops.BINARY_FLOAT_OPS)
  def testBinaryFloatOpsWithBatchShardedInputs(self, op):
    tol = select_tol(op, self.mesh, test_util.DEFAULT_TOL, low_res_tol=1e-2)
    a = constant_op.constant(
        np.array([[1., 2.], [3., 4.]]), dtype=dtypes.float32)
    b = constant_op.constant(
        np.array([[10., 20.], [30., 40.]]), dtype=dtypes.float32)
    expected_result = op(a, b)

    a = api.relayout(a, self.first_dimension_sharded_layout)
    b = api.relayout(b, self.first_dimension_sharded_layout)
    dtensor_result = op(a, b)

    self.assertDTensorEqual(
        expected_result,
        self.first_dimension_sharded_layout,
        dtensor_result,
        tol=tol)

  @parameterized.named_parameters(test_util_ops.BINARY_INT_OPS)
  def testBinaryIntOpsWithBatchShardedInputs(self, op):
    a = constant_op.constant(np.array([[1, 2], [3, 4]]))
    b = constant_op.constant(np.array([[5, 6], [7, 4]]))
    expected_result = op(a, b)

    a = api.relayout(a, self.first_dimension_sharded_layout)
    b = api.relayout(b, self.first_dimension_sharded_layout)
    dtensor_result = op(a, b)

    self.assertDTensorEqual(expected_result,
                            self.first_dimension_sharded_layout, dtensor_result)

  @parameterized.named_parameters(
      test_util_ops.BINARY_FLOAT_OPS_WITH_BROADCASTING_SUPPORT
  )
  def testBinaryFloatOpsWithFullyReplicatedBroadcastableInputs(self, op):
    tol = select_tol(op, self.mesh, test_util.DEFAULT_TOL, low_res_tol=1e-2)
    # Currently we only support scalar.
    a = constant_op.constant(23.4)
    b = constant_op.constant([[10., 20.], [30., 40.]])
    expected_result = op(a, b)

    a = api.copy_to_mesh(a, Layout.replicated(self.mesh, rank=a.ndim))
    b = api.copy_to_mesh(b, Layout.replicated(self.mesh, rank=b.ndim))
    dtensor_result = op(a, b)

    self.assertDTensorEqual(
        expected_result, self.replicated_layout_2d, dtensor_result, tol=tol)

  @parameterized.named_parameters(
      test_util_ops.BINARY_INT_OPS_WITH_BROADCASTING_SUPPORT
  )
  def testBinaryIntOpsWithFullyReplicatedBroadcastableInputs(self, op):
    tol = select_tol(op, self.mesh, test_util.DEFAULT_TOL, low_res_tol=1e-2)
    # Currently we only support scalar.
    a = constant_op.constant(3)
    b = constant_op.constant([[0, 1], [2, 3]])
    a, b = order_broadcastable_operands(op, a, b)
    expected_result = op(a, b)

    a = api.copy_to_mesh(a, Layout.replicated(self.mesh, rank=a.ndim))
    b = api.copy_to_mesh(b, Layout.replicated(self.mesh, rank=b.ndim))
    dtensor_result = op(a, b)

    self.assertDTensorEqual(
        expected_result, self.replicated_layout_2d, dtensor_result, tol=tol)

  @parameterized.named_parameters(
      test_util_ops.BINARY_FLOAT_OPS_WITH_BROADCASTING_SUPPORT
  )
  def testBinaryOpsWithFullyShardedBroadcastableInputs(self, op):
    tol = select_tol(op, self.mesh, test_util.DEFAULT_TOL, low_res_tol=1e-2)
    # Currently we only support scalar.
    a = constant_op.constant(23.4)
    b = constant_op.constant(
        10.0 * np.arange(8).reshape((2, 4)), dtype=dtypes.float32)
    expected_result = op(a, b)

    a = api.copy_to_mesh(a, self.scalar_replicated_layout)
    sharded_layout_2d = Layout([_MESH_DIM_X, _MESH_DIM_Y], self.mesh)
    b = api.relayout(b, sharded_layout_2d)

    dtensor_result = op(a, b)

    self.assertDTensorEqual(
        expected_result, sharded_layout_2d, dtensor_result, tol=tol)

  @parameterized.named_parameters(
      test_util_ops.BINARY_FLOAT_OPS_WITH_BROADCASTING_SUPPORT
  )
  def testBinaryOpsWithBatchShardedBroadcastableInputs(self, op):
    tol = select_tol(op, self.mesh, test_util.DEFAULT_TOL, low_res_tol=1e-2)
    # Currently we only support scalar.
    a = constant_op.constant(23.4)
    b = constant_op.constant(
        np.array([[10., 20.], [30., 40.]]), dtype=dtypes.float32)
    expected_result = op(a, b)

    a = api.copy_to_mesh(a, self.scalar_replicated_layout)
    b = api.relayout(b, self.first_dimension_sharded_layout)

    dtensor_result = op(a, b)

    self.assertDTensorEqual(
        expected_result,
        self.first_dimension_sharded_layout,
        dtensor_result,
        tol=tol)

  @parameterized.named_parameters(
      test_util_ops.expand_test_config(
          [
              {
                  'testcase_name': 'Concat',
                  'op': (lambda v: array_ops.concat(values=v, axis=1)),
              },
              {
                  'testcase_name':
                      'ConcatV1',
                  'op':
                      (lambda v: gen_array_ops.concat(concat_dim=1, values=v)),
              },
              {
                  'testcase_name': 'ConcatV2',
                  'op': (lambda v: gen_array_ops.concat_v2(values=v, axis=1)),
              },
          ],
          [
              {
                  'shard_type': 'replicated',
              },
              {
                  'shard_type': 'sharded',
              },
              {
                  'shard_type': 'mixed',
              },
          ],
      ))
  def testConcatOpSPMD(self, op, shard_type):
    layout_a = self.replicated_layout_2d
    layout_b = self.replicated_layout_2d
    layout_output = self.replicated_layout_2d

    if shard_type == 'sharded':
      layout_a = self.first_dimension_sharded_layout
      layout_b = self.first_dimension_sharded_layout
      layout_output = self.first_dimension_sharded_layout
    elif shard_type == 'mixed':
      layout_b = self.first_dimension_sharded_layout
      layout_output = self.first_dimension_sharded_layout

    a = constant_op.constant([[1., 2.], [3., 4.]])
    b = constant_op.constant([[1., 2.], [3., 4.]])
    expected_result = op([a, b])

    with api.default_mesh(self.mesh):
      a = api.relayout(a, layout_a)
      b = api.relayout(b, layout_b)
      c = op([a, b])

    self.assertDTensorEqual(expected_result, layout_output, c)

  @parameterized.named_parameters([{
      'testcase_name': 'ConcatV1',
      'op': (lambda v: gen_array_ops.concat(concat_dim=1, values=v))
  }, {
      'testcase_name': 'ConcatV2',
      'op': (lambda v: gen_array_ops.concat_v2(values=v, axis=1))
  }])
  def testConcatOpShardedOnConcatDim(self, op):
    a = constant_op.constant(
        np.arange(16).reshape((2, 2, 4)), dtype=dtypes.float32)
    b = constant_op.constant(
        np.arange(16).reshape((2, 2, 4)), dtype=dtypes.float32)
    expected_result = op([a, b])

    a_layout = Layout([layout_lib.UNSHARDED, _MESH_DIM_X, _MESH_DIM_Y],
                      self.mesh)
    b_layout = Layout([_MESH_DIM_X, layout_lib.UNSHARDED, layout_lib.UNSHARDED],
                      self.mesh)
    # If any input is sharded on the concat dim, then the concat dim is
    # replicated in the output. Dim 0 in the output is replicated because of
    # broadcast compatibility, mesh dimension X is already used in dim 1 of
    # input a.
    output_layout = Layout(
        [layout_lib.UNSHARDED, layout_lib.UNSHARDED, _MESH_DIM_Y], self.mesh)

    a = api.relayout(a, a_layout)
    b = api.relayout(b, b_layout)

    @polymorphic_function.function
    def concat_fn(a, b):
      return op([a, b])

    dtensor_result = concat_fn(a, b)

    self.assertDTensorEqual(expected_result, output_layout, dtensor_result)

  def testPackWithDifferentInputLayouts(self):
    a = constant_op.constant([[1., 2.], [3., 4.]])
    b = constant_op.constant([[1., 2.], [3., 4.]])
    expected_result = gen_array_ops.pack(values=[a, b], axis=-1)

    a = api.relayout(a, self.replicated_layout_2d)
    b = api.relayout(b, self.first_dimension_sharded_layout)

    @polymorphic_function.function
    def pack_fn(a, b):
      c = gen_array_ops.pack(values=[a, b], axis=-1)
      return api.relayout(c, self.first_dimension_sharded_layout_3d)

    dtensor_result = pack_fn(a, b)

    self.assertDTensorEqual(expected_result,
                            self.first_dimension_sharded_layout_3d,
                            dtensor_result)

  @parameterized.named_parameters(test_util_ops.REDUCTION_OPS)
  def testReductionOpsWithFullyReplicatedInputs(self, op):
    for axis, expected_layout in [([0], self.replicated_layout_1d),
                                  ([1], self.replicated_layout_1d),
                                  ([0, 1], self.scalar_replicated_layout),
                                  (None, self.scalar_replicated_layout)]:
      # Disable the pylint as the cell var is used for this iteration only.
      # pylint: disable=cell-var-from-loop
      reduction_op = lambda x: op(x, axis=axis)
      # pylint: enable=cell-var-from-loop

      a = constant_op.constant([[1., 2.], [3., 4.]])
      expected_result = reduction_op(a)

      a = api.copy_to_mesh(a, self.replicated_layout_2d)
      with api.default_mesh(self.mesh):
        dtensor_result = reduction_op(a)

      self.assertDTensorEqual(expected_result, expected_layout, dtensor_result)

  @parameterized.named_parameters(test_util_ops.REDUCTION_OPS)
  def testReductionOpsWithBatchParallelInputs(self, op):
    sharded_layout_1d = Layout([_MESH_DIM_X], self.mesh)
    for axis, expected_layout in [
        (
            [0],
            self.replicated_layout_1d,
        ),
        ([1], sharded_layout_1d),
        (
            [0, 1],
            self.scalar_replicated_layout,
        ),
        (
            None,
            self.scalar_replicated_layout,
        ),
    ]:
      # Disable the pylint as the cell var is used for this iteration only.
      # pylint: disable=cell-var-from-loop
      reduction_op = lambda x: op(x, axis=axis)
      # pylint: enable=cell-var-from-loop

      a = constant_op.constant(
          np.array([[1., 2.], [3., 4.], [5.0, 6.0], [7.0, 8.0]]),
          dtype=dtypes.float32)
      expected_result = reduction_op(a)

      a = api.relayout(a, self.first_dimension_sharded_layout)

      with api.default_mesh(self.mesh):
        dtensor_result = reduction_op(a)

        self.assertDTensorEqual(expected_result, expected_layout,
                                dtensor_result)

  def testReduceLogSumExpWithBatchParallelInputs(self):
    a = constant_op.constant(
        np.array([[1., 2.], [3., 4.], [5.0, 6.0], [7.0, 8.0]]),
        dtype=dtypes.float32)
    expected_result = math_ops.reduce_logsumexp(a, axis=-1)

    a = api.relayout(a, self.first_dimension_sharded_layout)

    with api.default_mesh(self.mesh):
      dtensor_result = math_ops.reduce_logsumexp(a, axis=-1)

      self.assertDTensorEqual(expected_result,
                              self.first_dimension_sharded_layout_1d,
                              dtensor_result)

  @parameterized.named_parameters(test_util_ops.REDUCTION_OPS)
  def testReductionOpsWithBatchParallelInputsWithInt64Dtype(self, op):
    self.skipForDeviceType(['TPU'], 'reduce on TPU only supports int32')

    sharded_layout_1d = Layout([_MESH_DIM_X], self.mesh)
    for axis, expected_layout in [
        (
            [0],
            self.replicated_layout_1d,
        ),
        (
            [1],
            sharded_layout_1d,
        ),
        (
            [0, 1],
            self.scalar_replicated_layout,
        ),
        (
            None,
            self.scalar_replicated_layout,
        ),
    ]:
      # Disable the pylint as the cell var is used for this iteration only.
      # pylint: disable=cell-var-from-loop
      reduction_op = lambda x: op(x, axis=axis)
      # pylint: enable=cell-var-from-loop

      a = constant_op.constant(
          np.array([[1, 2], [3, 4], [5, 6], [7, 8]]), dtype=dtypes.int64)
      expected_result = reduction_op(a)

      # pylint: disable=g-long-lambda
      a = api.relayout(a, self.first_dimension_sharded_layout)

      with api.default_mesh(self.mesh):
        dtensor_result = reduction_op(a)

        self.assertDTensorEqual(expected_result, expected_layout,
                                dtensor_result)

  @parameterized.named_parameters(test_util_ops.REDUCTION_OPS)
  def testReductionOpsWithBatchParallelInputsWithInt32(self, op):
    self.skipForDeviceType(['GPU'], 'reduce on GPU only supports int64')

    sharded_layout_1d = Layout([_MESH_DIM_X], self.mesh)
    for axis, expected_layout in [
        (
            [0],
            self.replicated_layout_1d,
        ),
        (
            [1],
            sharded_layout_1d,
        ),
        (
            [0, 1],
            self.scalar_replicated_layout,
        ),
        (
            None,
            self.scalar_replicated_layout,
        ),
    ]:
      # Disable the pylint as the cell var is used for this iteration only.
      # pylint: disable=cell-var-from-loop
      reduction_op = lambda x: op(x, axis=axis)
      # pylint: enable=cell-var-from-loop

      a = constant_op.constant(
          np.array([[1, 2], [3, 4], [5, 6], [7, 8]]), dtype=dtypes.int32)
      expected_result = reduction_op(a)

      # pylint: disable=g-long-lambda
      a = api.relayout(a, self.first_dimension_sharded_layout)

      with api.default_mesh(self.mesh):
        dtensor_result = reduction_op(a)

        self.assertDTensorEqual(expected_result, expected_layout,
                                dtensor_result)

  @parameterized.named_parameters(
      test_util_ops.expand_test_config(
          test_util_ops.REDUCTION_OPS,
          [
              {
                  'dtype': dtypes.float32
              },
              {
                  'dtype': dtypes.int32
              },
          ],
      ))
  def testReductionOpsWithReplicatedWithDtypes(self, op, dtype):
    self.skipForDeviceType(['GPU'], 'b/169353279: int32 caused segfault on GPU')

    axis = [0]
    # Disable the pylint as the cell var is used for this iteration only.
    # pylint: disable=cell-var-from-loop
    reduction_op = lambda x: op(x, axis=axis)
    # pylint: enable=cell-var-from-loop

    a = constant_op.constant(
        np.array([[1., 2.], [3., 4.], [5.0, 6.0], [7.0, 8.0]]), dtype=dtype)
    expected_result = reduction_op(a)
    a = api.relayout(a, self.replicated_layout_2d)

    with api.default_mesh(self.mesh):
      dtensor_result = reduction_op(a)

      self.assertDTensorEqual(expected_result, self.replicated_layout_1d,
                              dtensor_result)

  @parameterized.named_parameters(
      test_util_ops.expand_test_config(
          test_util_ops.REDUCTION_OPS,
          [
              {
                  'dtype': dtypes.float32
              },
              {
                  'dtype': dtypes.int32
              },
          ],
      ))
  def testReductionOpsWithBatchShardingWithDTypes(self, op, dtype):
    self.skipForDeviceType(['GPU'], 'b/169353279: int32 caused segfault on GPU')

    axis = [1]
    # Disable the pylint as the cell var is used for this iteration only.
    # pylint: disable=cell-var-from-loop
    reduction_op = lambda x: op(x, axis=axis)
    # pylint: enable=cell-var-from-loop

    a = constant_op.constant(
        np.array([[1., 2.], [3., 4.], [5.0, 6.0], [7.0, 8.0]]), dtype=dtype)
    expected_result = reduction_op(a)
    a = api.relayout(a, self.first_dimension_sharded_layout)

    with api.default_mesh(self.mesh):
      dtensor_result = reduction_op(a)

      self.assertDTensorEqual(expected_result,
                              self.first_dimension_sharded_layout_1d,
                              dtensor_result)

  @parameterized.named_parameters(
      test_util_ops.expand_test_config(
          test_util_ops.REDUCTION_OPS,
          [
              {
                  'axis': [0, 1],
                  'dtype': dtypes.float32
              },
              {
                  'axis': [0, 1],
                  'dtype': dtypes.int32
              },
              {
                  'axis': None,
                  'dtype': dtypes.float32
              },
              {
                  'axis': None,
                  'dtype': dtypes.int32
              },
          ],
      ))
  def testReductionOpsWithReplicatedLayoutAndDTypes(self, op, axis, dtype):
    self.skipForDeviceType(['GPU'], 'b/169353279: int32 caused segfault on GPU')

    # Disable the pylint as the cell var is used for this iteration only.
    # pylint: disable=cell-var-from-loop
    reduction_op = lambda x: op(x, axis=axis)
    # pylint: enable=cell-var-from-loop

    a = constant_op.constant(
        np.array([[1., 2.], [3., 4.], [5.0, 6.0], [7.0, 8.0]]), dtype=dtype)
    expected_result = reduction_op(a)
    a = api.relayout(a, self.first_dimension_sharded_layout)

    with api.default_mesh(self.mesh):
      dtensor_result = reduction_op(a)

      self.assertDTensorEqual(expected_result, self.scalar_replicated_layout,
                              dtensor_result)

  @parameterized.named_parameters(
      test_util_ops.expand_test_config(
          [
              {
                  'testcase_name': 'FullyReplicatedInputs',
                  'shard_type': 'replicated',
              },
              {
                  'testcase_name': 'BatchShardedInputs',
                  'shard_type': 'batch_sharded',
              },
          ],
          [
              {
                  'axis': -1,
              },
              {
                  'axis': 0,
              },
              {
                  'axis': 1,
              },
          ],
      )
  )
  def testOneHotSPMDWith(self, shard_type, axis):
    if axis != -1:
      self.skipTest('b/177569789: fix this test with layout propagation v2')

    indices = constant_op.constant([[1, 2], [3, 4]], dtype=dtypes.int32)
    depth = constant_op.constant(10, dtype=dtypes.int32)
    indices_layout = (
        self.replicated_layout_2d
        if shard_type == 'replicated' else self.first_dimension_sharded_layout)
    output_layout = (
        Layout.replicated(self.mesh, rank=3) if shard_type == 'replicated' else
        Layout.batch_sharded(self.mesh, _MESH_DIM_X, rank=3))

    expected_result = array_ops.one_hot(indices, depth, axis=axis)

    indices = api.relayout(indices, indices_layout)
    depth = api.copy_to_mesh(depth, self.scalar_replicated_layout)
    dtensor_result = array_ops.one_hot(indices, depth, axis=axis)
    if axis == 0 and shard_type == 'batch_sharded':
      output_layout = self.middle_dimension_sharded_layout_3d

    self.assertDTensorEqual(expected_result, output_layout, dtensor_result)

  def testOneHotSPMDWithDifferentLayout(self):
    indices = constant_op.constant([[1, 2], [3, 4]], dtype=dtypes.int32)
    depth = constant_op.constant(10, dtype=dtypes.int32)
    expected_result = array_ops.one_hot(indices, depth, axis=2)

    indices = api.relayout(indices, self.replicated_layout_2d)

    depth = api.copy_to_mesh(depth, self.scalar_replicated_layout)

    @polymorphic_function.function
    def one_hot_fn(indices, depth):
      result = array_ops.one_hot(indices, depth, axis=2)
      return api.relayout(result, self.first_dimension_sharded_layout_3d)

    dtensor_result = one_hot_fn(indices, depth)

    self.assertDTensorEqual(expected_result,
                            self.first_dimension_sharded_layout_3d,
                            dtensor_result)

  def testL2LossOpsWithFullyReplicatedInputs(self):
    loss_op = gen_nn_ops.l2_loss
    a = constant_op.constant([[1., 2.], [3., 4.]])
    expected_result = loss_op(a)
    expected_layout = self.scalar_replicated_layout

    a = api.copy_to_mesh(a, self.replicated_layout_2d)
    dtensor_result = loss_op(a)

    self.assertDTensorEqual(expected_result, expected_layout, dtensor_result)

  def testL2LossOpsWithFullyShardedInputs(self):
    loss_op = gen_nn_ops.l2_loss
    a = constant_op.constant([[1., 2.], [3., 4.]])
    expected_result = loss_op(a)
    expected_layout = self.scalar_replicated_layout

    a = api.relayout(a, self.first_dimension_sharded_layout)
    dtensor_result = loss_op(a)
    self.assertDTensorEqual(expected_result, expected_layout, dtensor_result)

  @parameterized.named_parameters(test_util_ops.EXPANSION_OPS)
  def testExpansionOpsReplicatedLayout(self, inputs, op):
    self.skipTest('b/177569789: fix this test with layout propagation v2')

    global_op_args = inputs()
    expected_result = op(*global_op_args)

    with api.default_mesh(self.mesh):
      dtensor_op_args = inputs()

      def _broadcast_to_replicated(x):
        x = constant_op.constant(x)
        return api.copy_to_mesh(
            x, Layout.replicated(self.mesh, rank=x.shape.ndims))

      dtensor_op_args = nest.map_structure(_broadcast_to_replicated,
                                           dtensor_op_args)

      with api._dtensor_device()._default_layout(self.replicated_layout_2d):
        dtensor_result = op(*dtensor_op_args)
    self.assertDTensorEqual(expected_result, self.replicated_layout_2d,
                            dtensor_result)

  @parameterized.named_parameters(test_util_ops.EXPANSION_OPS)
  def testExpansionOpsFullySharded(self, inputs, op):
    self.skipTest('b/177569789: fix this test with layout propagation v2')

    global_op_args = inputs()
    expected_result = op(*global_op_args)

    with api.default_mesh(self.mesh):
      dtensor_op_args = inputs()

      def _broadcast_to_replicated(x):
        x = constant_op.constant(x)
        return api.copy_to_mesh(
            x, Layout.replicated(self.mesh, rank=x.shape.ndims))

      dtensor_op_args = nest.map_structure(_broadcast_to_replicated,
                                           dtensor_op_args)
      sharded_layout_2d = Layout([_MESH_DIM_X, _MESH_DIM_Y], self.mesh)
      with api._dtensor_device()._default_layout(sharded_layout_2d):
        dtensor_result = op(*dtensor_op_args)

    self.assertDTensorEqual(expected_result, sharded_layout_2d, dtensor_result)

  @parameterized.named_parameters(test_util_ops.EXPANSION_OPS)
  def testExpansionOpsBatchSharded(self, inputs, op):
    self.skipTest('b/177569789: fix this test with layout propagation v2')

    global_op_args = inputs()
    expected_result = op(*global_op_args)

    first_d_shard_layout = Layout([_MESH_DIM_X, layout_lib.UNSHARDED],
                                  self.mesh)

    with api.default_mesh(self.mesh):
      dtensor_op_args = inputs()

      def _broadcast_to_replicated(x):
        x = constant_op.constant(x)
        return api.copy_to_mesh(
            x, Layout.replicated(self.mesh, rank=x.shape.ndims))

      dtensor_op_args = nest.map_structure(_broadcast_to_replicated,
                                           dtensor_op_args)

      with api._dtensor_device().default_layout(first_d_shard_layout):
        dtensor_result = op(*dtensor_op_args)

    self.assertDTensorEqual(expected_result, first_d_shard_layout,
                            dtensor_result)

  def testSliceOpsWithFullyReplicatedInputs(self):
    t = constant_op.constant([[1., 2., 3., 4.], [5., 6., 7., 8.]])
    expected_result = array_ops.slice(t, [0, 0], [-1, 2])

    a = api.copy_to_mesh(t, self.replicated_layout_2d)
    with api.default_mesh(self.mesh):
      dtensor_result = array_ops.slice(a, [0, 0], [-1, 2])

    self.assertDTensorEqual(expected_result, self.replicated_layout_2d,
                            dtensor_result)

  @parameterized.named_parameters(('_minus_one_size', -1), ('_pos_size', 2))
  def testSliceOpsWithFullSlicingOnShardedInputs(self, size):
    t = constant_op.constant([[1., 2., 3., 4.], [5., 6., 7., 8.]])
    expected_result = array_ops.slice(t, [0, 0], [size, 2])
    sharded_layout = self.first_dimension_sharded_layout

    t = api.relayout(t, sharded_layout)
    with api.default_mesh(self.mesh):
      dtensor_result = array_ops.slice(t, [0, 0], [size, 2])

    self.assertDTensorEqual(expected_result, sharded_layout, dtensor_result)

  def testSliceOpsWithDynamicBeginFullSlicingOnShardedInputs(self):
    tensor = constant_op.constant([[1., 2., 3., 4.], [5., 6., 7., 8.]])
    begins = constant_op.constant([0, 0], dtype=dtypes.int32)

    @polymorphic_function.function
    def slice_fn(tensor, begins):
      return array_ops.slice(tensor, begins, [2, 2])

    expected_result = slice_fn(tensor, begins)

    sharded_layout = self.first_dimension_sharded_layout

    tensor = api.relayout(tensor, sharded_layout)
    begins = api.relayout(begins, self.replicated_layout_1d)

    dtensor_result = slice_fn(tensor, begins)

    self.assertDTensorEqual(expected_result, sharded_layout, dtensor_result)

  @parameterized.named_parameters(
      ('FullyReplicatedInputs', {
          'begin': [0, 0],
          'end': [-1, 2],
          'strides': [1, 2]
      }, [layout_lib.UNSHARDED] * 2), ('NewAxisMask', {
          'begin': [0, 0, 0, 0],
          'end': [0, 0, 2, 4],
          'strides': [1, 1, 1, 1],
          'new_axis_mask': 3
      }, [layout_lib.UNSHARDED] * 2, [layout_lib.UNSHARDED] * 4),
      ('ShrinkAxisMask', {
          'begin': [0, 0],
          'end': [-1, 2],
          'strides': [1, 1],
          'shrink_axis_mask': 2
      }, [layout_lib.UNSHARDED] * 2, [layout_lib.UNSHARDED]),
      ('ShardingOnNonSlicedDimension', {
          'begin': [0, 0],
          'end': [2, 2],
          'strides': [1, 2]
      }, [_MESH_DIM_X, layout_lib.UNSHARDED]),
      ('StrideOnShardedDimensionNoRelayout1', {
          'begin': [0, 0],
          'end': [2, 4],
          'strides': [1, 2]
      }, [layout_lib.UNSHARDED, _MESH_DIM_X]),
      ('StrideOnShardedDimensionNoRelayout2', {
          'begin': [0, 1],
          'end': [2, 4],
          'strides': [1, 2]
      }, [layout_lib.UNSHARDED, _MESH_DIM_X]),
      ('StrideOnShardedDimensionNoRelayout3', {
          'begin': [0, 0],
          'end': [2, 3],
          'strides': [1, 2]
      }, [layout_lib.UNSHARDED, _MESH_DIM_X]),
      ('StrideOnShardedDimensionNeedRelayout', {
          'begin': [0, 0],
          'end': [-1, 4],
          'strides': [1, 3]
      }, [_MESH_DIM_X, layout_lib.UNSHARDED], [layout_lib.UNSHARDED] * 2))
  def testStridedSliceOps(self, args, input_layout, expected_layout=None):
    input_tensor = constant_op.constant([[1., 2., 3., 4.], [5., 6., 7., 8.]])
    expected_result = gen_array_ops.strided_slice(input=input_tensor, **args)

    input_layout = Layout(input_layout, self.mesh)
    if expected_layout is None:
      expected_layout = input_layout
    else:
      expected_layout = Layout(expected_layout, self.mesh)

    dtensor_input_tensor = api.relayout(input_tensor, input_layout)
    dtensor_result = gen_array_ops.strided_slice(
        input=dtensor_input_tensor, **args)

    self.assertDTensorEqual(expected_result, expected_layout, dtensor_result)

  @parameterized.named_parameters(
      ('FullyReplicatedInputs', {
          'begin': [0, 0],
          'end': [-1, 2],
          'strides': [1, 2]
      }, [layout_lib.UNSHARDED] * 2), ('NewAxisMask', {
          'begin': [0, 0, 0, 0],
          'end': [0, 0, 2, 4],
          'strides': [1, 1, 1, 1],
          'new_axis_mask': 3
      }, [layout_lib.UNSHARDED] * 2), ('ShrinkAxisMask', {
          'begin': [0, 0],
          'end': [-1, 2],
          'strides': [1, 1],
          'shrink_axis_mask': 2
      }, [layout_lib.UNSHARDED] * 2), ('ShardingOnNonSlicedDimension', {
          'begin': [0, 0],
          'end': [2, 2],
          'strides': [1, 2]
      }, [_MESH_DIM_X, layout_lib.UNSHARDED]),
      ('StrideOnShardedDimensionNoRelayout1', {
          'begin': [0, 0],
          'end': [2, 4],
          'strides': [1, 2]
      }, [layout_lib.UNSHARDED, _MESH_DIM_X]),
      ('StrideOnShardedDimensionNoRelayout2', {
          'begin': [0, 1],
          'end': [2, 4],
          'strides': [1, 2]
      }, [layout_lib.UNSHARDED, _MESH_DIM_X]),
      ('StrideOnShardedDimensionNoRelayout3', {
          'begin': [0, 0],
          'end': [2, 3],
          'strides': [1, 2]
      }, [layout_lib.UNSHARDED, _MESH_DIM_X]),
      ('StrideOnShardedDimensionNeedRelayout', {
          'begin': [0, 0],
          'end': [-1, 4],
          'strides': [1, 3]
      }, [_MESH_DIM_X, layout_lib.UNSHARDED], [layout_lib.UNSHARDED] * 2))
  def testStridedSliceGradOps(self, args, input_layout, expected_layout=None):
    input_tensor = constant_op.constant([[1., 2., 3., 4.], [5., 6., 7., 8.]])
    shape = input_tensor.shape.as_list()
    input_layout = Layout(input_layout, self.mesh)
    if expected_layout is None:
      expected_layout = input_layout
    else:
      expected_layout = Layout(expected_layout, self.mesh)

    grad = gen_array_ops.strided_slice(input=input_tensor, **args)
    expected_result = gen_array_ops.strided_slice_grad(
        shape=shape, **args, dy=grad)

    dtensor_input_tensor = api.relayout(input_tensor, input_layout)
    grad = gen_array_ops.strided_slice(input=dtensor_input_tensor, **args)
    dtensor_result = gen_array_ops.strided_slice_grad(
        shape=shape, **args, dy=grad)

    self.assertDTensorEqual(expected_result, expected_layout, dtensor_result)

  @parameterized.named_parameters(
      ('FullyReplicatedInputs', {
          'begin': [0, 0],
          'end': [-1, 2],
          'strides': [1, 2]
      }, [layout_lib.UNSHARDED] * 2, [layout_lib.UNSHARDED] * 2),
      ('NewAxisMask', {
          'begin': [0, 0, 0, 0],
          'end': [0, 0, 2, 4],
          'strides': [1, 1, 1, 1],
          'new_axis_mask': 3
      }, [layout_lib.UNSHARDED] * 2, [layout_lib.UNSHARDED] * 4),
      ('ShrinkAxisMask', {
          'begin': [0, 0],
          'end': [-1, 2],
          'strides': [1, 1],
          'shrink_axis_mask': 2
      }, [layout_lib.UNSHARDED] * 2, [layout_lib.UNSHARDED]),
      ('ShardingOnNonSlicedDimension', {
          'begin': [0, 0],
          'end': [2, 2],
          'strides': [1, 2]
      }, [_MESH_DIM_X, layout_lib.UNSHARDED
         ], [_MESH_DIM_X, layout_lib.UNSHARDED]),
      ('StrideOnShardedDimensionNoRelayout1', {
          'begin': [0, 0],
          'end': [2, 4],
          'strides': [1, 2]
      }, [layout_lib.UNSHARDED, _MESH_DIM_X
         ], [layout_lib.UNSHARDED, _MESH_DIM_X]),
      ('StrideOnShardedDimensionNoRelayout2', {
          'begin': [0, 1],
          'end': [2, 4],
          'strides': [1, 2]
      }, [layout_lib.UNSHARDED, _MESH_DIM_X
         ], [layout_lib.UNSHARDED, _MESH_DIM_X]),
      ('StrideOnShardedDimensionNoRelayout3', {
          'begin': [0, 0],
          'end': [2, 3],
          'strides': [1, 2]
      }, [layout_lib.UNSHARDED, _MESH_DIM_X
         ], [layout_lib.UNSHARDED, _MESH_DIM_X]),
      ('StrideOnShardedDimensionNeedRelayout', {
          'begin': [0, 0],
          'end': [-1, 4],
          'strides': [1, 3]
      }, [_MESH_DIM_X, layout_lib.UNSHARDED
         ], [layout_lib.UNSHARDED] * 2, [layout_lib.UNSHARDED] * 2))
  def testStridedSliceUpdateOps(self,
                                args,
                                input_layout,
                                value_layout,
                                expected_layout=None):
    self.skipForDeviceType(['TPU'], 'b/123559667; op has no XLA implementation')
    input_tensor = constant_op.constant([[1., 2., 3., 4.], [5., 6., 7., 8.]])
    value_tensor = gen_array_ops.strided_slice(input=input_tensor, **args) * 10.
    expected_result = gen_array_ops.tensor_strided_slice_update(
        input=input_tensor, value=value_tensor, **args)

    input_layout = Layout(input_layout, self.mesh)
    value_layout = Layout(value_layout, self.mesh)
    if expected_layout is None:
      expected_layout = input_layout
    else:
      expected_layout = Layout(expected_layout, self.mesh)

    dtensor_input_tensor = api.relayout(input_tensor, input_layout)
    dtensor_value_tensor = api.relayout(value_tensor, value_layout)
    dtensor_result = gen_array_ops.tensor_strided_slice_update(
        input=dtensor_input_tensor, value=dtensor_value_tensor, **args)

    self.assertDTensorEqual(expected_result, expected_layout, dtensor_result)

  def testBroadcastGradientArgs(self):
    a = constant_op.constant([128, 10])
    b = constant_op.constant([128, 10])
    ea, eb = gen_array_ops.broadcast_gradient_args(s0=a, s1=b)

    a = api.copy_to_mesh(a, self.replicated_layout_1d)
    b = api.copy_to_mesh(b, self.replicated_layout_1d)
    da, db = gen_array_ops.broadcast_gradient_args(s0=a, s1=b)

    self.assertDTensorEqual(ea, self.replicated_layout_1d, da)
    self.assertDTensorEqual(eb, self.replicated_layout_1d, db)

  def _transpose_shape(self, transpose, shape):
    if transpose:
      shape[-1], shape[-2] = shape[-2:]
    return shape

  def _merge_layouts_for_matmul(self, layout_a, layout_b, transpose_a,
                                transpose_b):
    # This merge does no error checking and assumes that mesh dimensions
    # are compatible and that layout_a and b are on the same mesh.
    # Prepend enough layout_lib.UNSHARDED to give both lists the same size.
    a_sharding_spec = (
        [layout_lib.UNSHARDED] * max(0, layout_b.rank - layout_a.rank) +
        layout_a.sharding_specs)
    b_sharding_spec = (
        [layout_lib.UNSHARDED] * max(0, layout_a.rank - layout_b.rank) +
        layout_b.sharding_specs)
    if transpose_a:
      a_sharding_spec[-1], a_sharding_spec[-2] = a_sharding_spec[-2:]
    if transpose_b:
      b_sharding_spec[-1], b_sharding_spec[-2] = b_sharding_spec[-2:]

    def _get_mesh_dim(i):
      if b_sharding_spec[i] == layout_lib.UNSHARDED:
        return a_sharding_spec[i]
      return b_sharding_spec[i]

    final_layout = [_get_mesh_dim(i) for i in range(len(a_sharding_spec) - 2)]
    final_layout.append(a_sharding_spec[-2])
    final_layout.append(b_sharding_spec[-1])
    if final_layout[-2] == final_layout[-1]:
      final_layout[-2] = layout_lib.UNSHARDED
      final_layout[-1] = layout_lib.UNSHARDED
    for i in range(len(final_layout) - 2):
      if (final_layout[i] == a_sharding_spec[-2] or
          final_layout[i] == a_sharding_spec[-1] or
          final_layout[i] == b_sharding_spec[-2] or
          final_layout[i] == b_sharding_spec[-1]):
        final_layout[i] = layout_lib.UNSHARDED

    return Layout(final_layout, layout_a.mesh)

  @parameterized.named_parameters(*test_util.product(_MATMUL_IMPLEMENTED,
                                                     _MATMUL_TRANSPOSE))
  def testMatMul(self, a_layout, b_layout, transpose_a, transpose_b):
    # Swap layout 1 and 2, so that test name is correct (contracting and
    # non_contracting dims switch when transposed).
    if transpose_a and a_layout > 0:
      a_layout = 3 - a_layout
    if transpose_b and b_layout > 0:
      b_layout = 3 - b_layout
    a_layout = self.layouts_2d[a_layout]
    b_layout = self.layouts_2d[b_layout]
    a_numpy = np.random.uniform(size=self._transpose_shape(transpose_a, [4, 8]))
    b_numpy = np.random.uniform(
        size=self._transpose_shape(transpose_b, [8, 12]))
    a = constant_op.constant(a_numpy, dtype=dtypes.float32)
    b = constant_op.constant(b_numpy, dtype=dtypes.float32)

    expected = math_ops.matmul(
        a, b, transpose_a=transpose_a, transpose_b=transpose_b)

    a = api.relayout(a, a_layout)
    b = api.relayout(b, b_layout)
    dtensor_result = math_ops.matmul(
        a, b, transpose_a=transpose_a, transpose_b=transpose_b)
    expected_layout = self._merge_layouts_for_matmul(a_layout, b_layout,
                                                     transpose_a, transpose_b)
    self.assertDTensorEqual(expected, expected_layout, dtensor_result)

  @parameterized.named_parameters(*test_util.product(_BATCH_MATMUL_IMPLEMENTED,
                                                     _MATMUL_TRANSPOSE))
  def testBatchMatMul(self, a_layout, b_layout, transpose_a, transpose_b):
    # Swap layout 2 and 3, so that test name is correct (contracting and
    # non_contracting dims switch when transposed).
    if transpose_a and a_layout > 1:
      a_layout = 5 - a_layout
    if transpose_b and b_layout > 1:
      b_layout = 5 - b_layout
    a_layout = self.layouts_3d[a_layout]
    b_layout = self.layouts_3d[b_layout]
    a_numpy = np.random.uniform(
        size=self._transpose_shape(transpose_a, [2, 4, 8]))
    b_numpy = np.random.uniform(
        size=self._transpose_shape(transpose_b, [2, 8, 12]))
    a = constant_op.constant(a_numpy, dtype=dtypes.float32)  # 2x4x8
    b = constant_op.constant(b_numpy, dtype=dtypes.float32)  # 2x8x12

    # math_ops.matmul should emit a BatchMatMulV2 op here.
    expected = math_ops.matmul(
        a, b, transpose_a=transpose_a, transpose_b=transpose_b)

    a = api.relayout(a, a_layout)
    b = api.relayout(b, b_layout)
    dtensor_result = math_ops.matmul(
        a, b, transpose_a=transpose_a, transpose_b=transpose_b)
    expected_layout = self._merge_layouts_for_matmul(a_layout, b_layout,
                                                     transpose_a, transpose_b)
    self.assertDTensorEqual(expected, expected_layout, dtensor_result)

  @parameterized.named_parameters(
      ('_a_unsharded_b_unsharded', 0, 0), ('_a_batch_b_unsharded', 1, 0),
      ('_a_non_contracting_b_unsharded', 2, 0),
      ('_a_contracting_b_unsharded', 3, 0),
      ('_a_unsharded_b_non_contracting', 0, 2),
      ('_a_unsharded_b_contracting', 0, 1),
      ('_a_contracting_b_contracting', 3, 1),
      ('_a_contracting_b_non_contracting', 3, 2),
      ('_a_non_contracting_b_non_contracting', 2, 2),
      ('_a_non_contracting_b_contracting', 2, 1),
      ('_a_batch_b_non_contracting', 1, 2), ('_a_batch_b_contracting', 1, 1))
  def testBatchMatMulWithBroadcasting(self, a_layout, b_layout):
    a_layout = self.layouts_3d[a_layout]
    b_layout = self.layouts_2d[b_layout]
    a_numpy = np.random.uniform(size=[2, 2, 4])
    b_numpy = np.random.uniform(size=[4, 6])
    a = constant_op.constant(a_numpy, dtype=dtypes.float32)  # 2x2x4
    b = constant_op.constant(b_numpy, dtype=dtypes.float32)  # 4x6

    # math_ops.matmul should emit a BatchMatMulV2 op here.
    expected = math_ops.matmul(a, b)

    a = api.relayout(a, a_layout)
    b = api.relayout(b, b_layout)
    dtensor_result = math_ops.matmul(a, b)
    expected_layout = self._merge_layouts_for_matmul(a_layout, b_layout, False,
                                                     False)
    self.assertDTensorEqual(expected, expected_layout, dtensor_result)

  @parameterized.named_parameters(('_positive_axis_negative_batch', 0, -1),
                                  ('_negative_axis_positive_batch', -2, 0))
  def testGather(self, axis, batch_dims):
    params = np.arange(1000 * 4).reshape((1000, 4))
    # "batch" size = 2, num_indices = 3 per example
    indices = np.random.randint(0, 1000, size=4 * 3).reshape((4, 3))
    expected = array_ops.gather_v2(
        params, indices, axis=axis, batch_dims=batch_dims)

    params = api.relayout(params, layout=Layout.replicated(self.mesh, 2))
    indices = api.relayout(
        indices, Layout.batch_sharded(self.mesh, _MESH_DIM_Y, rank=2)
    )

    dtensor_result = array_ops.gather_v2(
        params, indices, axis=axis, batch_dims=batch_dims)
    expected_layout = Layout.batch_sharded(self.mesh, _MESH_DIM_Y, rank=3)
    self.assertDTensorEqual(expected, expected_layout, dtensor_result)

  def testResourceGather(self):
    if self.mesh.use_xla_spmd():
      self.skipTest('Variables not supported yet with DTensor Xla Spmd.')

    params = np.arange(1000 * 4).reshape((1000, 4))
    indices = np.random.randint(0, 1000, size=1000 * 3).reshape((1000, 3))

    expected = array_ops.gather_v2(variables.Variable(params), indices)

    params = api.relayout(params, layout=Layout.replicated(self.mesh, 2))
    indices = api.relayout(
        indices, Layout.batch_sharded(self.mesh, _MESH_DIM_Y, rank=2)
    )

    dtensor_result = array_ops.gather_v2(d_variable.DVariable(params), indices)
    expected_layout = Layout.batch_sharded(self.mesh, _MESH_DIM_Y, rank=3)

    self.assertDTensorEqual(expected, expected_layout, dtensor_result)

  def testResourceGatherRaisesErrorWhenResourceZeroDimSharded(self):
    if self.mesh.use_xla_spmd():
      self.skipTest('Variables not supported yet with DTensor Xla Spmd.')

    sharded_tensor = api.relayout(
        np.arange(1000 * 4).reshape((1000, 4)),
        layout=Layout.batch_sharded(self.mesh, _MESH_DIM_Y, 2),
    )
    # "batch" size = 2, num_indices = 3 per example
    indices = api.copy_to_mesh(
        np.random.randint(0, 1000, size=4 * 3).reshape((4, 3)),
        Layout.replicated(self.mesh, rank=2))

    with self.assertRaisesRegex(
        errors_impl.UnknownError,
        'DTensor does not support sharded 0th dimension for the resource tensor'
    ):
      array_ops.gather_v2(d_variable.DVariable(sharded_tensor), indices)

  def testUnsortedSegmentSum(self):
    self.skipForDeviceType(['TPU'], 'waiting for cl/344197900')
    num_segments = 12
    data = np.random.uniform(size=[num_segments, 4])
    segment_ids = np.random.randint(0, num_segments, size=num_segments)
    expected = gen_math_ops.unsorted_segment_sum(data, segment_ids,
                                                 num_segments)

    data = api.relayout(data, Layout.replicated(self.mesh, 2))
    segment_ids = api.relayout(
        segment_ids, Layout.batch_sharded(self.mesh, _MESH_DIM_Y, rank=1)
    )
    with api.default_mesh(self.mesh):
      dtensor_result = gen_math_ops.unsorted_segment_sum(
          data, segment_ids, num_segments)
      expected_layout = Layout.replicated(self.mesh, 2)
      self.assertDTensorEqual(expected, expected_layout, dtensor_result)

  def testUnsortedSegmentSumWithFullyShardedIndices(self):
    self.skipForDeviceType(['TPU'], 'waiting for cl/344197900')

    num_segments = 8
    data = np.random.uniform(size=[2, 4, 3])
    segment_ids = np.random.randint(0, num_segments, size=[2, 4])
    expected = gen_math_ops.unsorted_segment_sum(data, segment_ids,
                                                 num_segments)

    data = api.relayout(data, Layout.replicated(self.mesh, 3))
    segment_ids = api.relayout(
        segment_ids, Layout([_MESH_DIM_X, _MESH_DIM_Y], self.mesh)
    )
    with api.default_mesh(self.mesh):
      dtensor_result = gen_math_ops.unsorted_segment_sum(
          data, segment_ids, num_segments)
      expected_layout = Layout.replicated(self.mesh, 2)
      self.assertDTensorEqual(expected, expected_layout, dtensor_result)

  @parameterized.named_parameters(
      ('_same_rank', [2, 2]),
      ('_adding_one_rank', [2, 2, 1]),
      ('_adding_one_rank_and_broadcasting', [2, 2, 2]),
  )
  def testBroadcastOpsWithFullyReplicatedInputs(self, new_shape):
    op = gen_array_ops.broadcast_to
    a = constant_op.constant([[1.], [3.]])
    assert a.shape == [2, 1]

    expected_result = op(a, new_shape)

    a = api.copy_to_mesh(a, self.replicated_layout_2d)
    dtensor_result = op(a, new_shape)

    self.assertDTensorEqual(expected_result,
                            Layout.replicated(self.mesh, len(new_shape)),
                            dtensor_result)

  @parameterized.named_parameters(
      test_util_ops.expand_test_config(
          [
              {
                  'testcase_name': 'FullyReplicatedInputs',
                  'op': array_ops.where_v2
              },
              {
                  'testcase_name': 'BatchShardedInputs',
                  'op': array_ops.where_v2
              },
          ],
          [
              {
                  'shard_type': 'replicated',
              },
              {
                  'shard_type': 'batch_sharded',
              },
          ],
      ))
  def testWhere(self, op, shard_type):
    layout = (
        self.replicated_layout_2d
        if shard_type == 'replicated' else self.first_dimension_sharded_layout)

    a = constant_op.constant([[True, False], [False, True]])
    b = constant_op.constant([[10., 20.], [30., 40.]])
    c = constant_op.constant([[50., 60.], [70., 80.]])
    expected_result = op(a, b, c)

    if shard_type == 'replicated':
      a = api.copy_to_mesh(a, layout)
      b = api.copy_to_mesh(b, layout)
      c = api.copy_to_mesh(c, layout)
    else:
      a = api.relayout(a, layout)
      b = api.relayout(b, layout)
      c = api.relayout(c, layout)
    dtensor_result = op(a, b, c)

    self.assertDTensorEqual(expected_result, layout, dtensor_result)

  def testSqueezeOp(self):
    t = array_ops.ones([1, 2, 1])
    expected_result0 = array_ops.squeeze_v2(t)
    expected_result1 = array_ops.squeeze_v2(t, axis=0)
    expected_result2 = array_ops.squeeze_v2(t, axis=-1)

    # t will have [1,1,1] as locally sharded shape, this covers the case that
    # we should not squeeze the dim that's sharded.
    t = api.relayout(
        t,
        Layout(
            [layout_lib.UNSHARDED, _MESH_DIM_X, layout_lib.UNSHARDED], self.mesh
        ),
    )
    dtensor_result0 = array_ops.squeeze_v2(t)
    dtensor_result1 = array_ops.squeeze_v2(t, axis=0)
    dtensor_result2 = array_ops.squeeze_v2(t, axis=-1)

    self.assertDTensorEqual(expected_result0, Layout([_MESH_DIM_X], self.mesh),
                            dtensor_result0)
    self.assertDTensorEqual(
        expected_result1, Layout([_MESH_DIM_X, layout_lib.UNSHARDED],
                                 self.mesh), dtensor_result1)
    self.assertDTensorEqual(
        expected_result2, Layout([layout_lib.UNSHARDED, _MESH_DIM_X],
                                 self.mesh), dtensor_result2)

  @parameterized.parameters(('replicated',), ('sharded',))
  def testDiagPart(self, shard_type):
    x = stateless_random_ops.stateless_random_uniform(
        shape=(16, 16), seed=[0, 1])
    expected = gen_array_ops.diag_part(input=x)

    if shard_type == 'replicated':
      layout = Layout([_MESH_DIM_X, _MESH_DIM_Y], self.mesh)
    else:
      layout = Layout.replicated(self.mesh, 2)
    x = api.relayout(x, layout)

    got = gen_array_ops.diag_part(input=x)
    self.assertDTensorEqual(expected, Layout.replicated(self.mesh, 1), got)

  @parameterized.product(
      axis_dim=[-3, -2, -1, 0, 1, 2],
      shard_type=['replicated', 'batch_sharded'],
      reverse=[True, False])
  def testCumSum(self, axis_dim, shard_type, reverse):
    input_tensor = stateless_random_ops.stateless_random_uniform(
        shape=(16, 16, 16), seed=[0, 1])
    expected = math_ops.cumsum(x=input_tensor, axis=axis_dim, reverse=reverse)

    if shard_type == 'replicated':
      layout = Layout.replicated(self.mesh, rank=3)
      expected_layout = layout
    else:
      layout = Layout.batch_sharded(self.mesh, batch_dim=_MESH_DIM_X, rank=3)
      # Axis dimension should always be replicated, even on sharding dim.
      if axis_dim in [-3, 0]:
        expected_layout = Layout.replicated(self.mesh, rank=3)
      else:
        expected_layout = layout

    input_tensor = api.relayout(input_tensor, layout)
    got = math_ops.cumsum(x=input_tensor, axis=axis_dim, reverse=reverse)

    self.assertDTensorEqual(expected, expected_layout, got)

  @parameterized.named_parameters(('Sharded', 'sharded'),
                                  ('Replicated', 'replicated'))
  def testStringFormat(self, shard_spec):
    self.skipForDeviceType(['TPU'], 'StringFormat not supported on TPU.')

    np.random.seed(123)
    inputs = constant_op.constant(
        np.random.normal(0.0, 1.0, 8 * 9 * 9).reshape([8, 9, 9, 1]),
        dtype=dtypes.float32)
    expected_result = gen_string_ops.string_format(inputs=[inputs])

    if shard_spec == 'sharded':
      layout = Layout.batch_sharded(self.mesh, _MESH_DIM_X, rank=4)
    else:
      layout = Layout.replicated(self.mesh, rank=4)
    inputs = api.relayout(inputs, layout)
    got = gen_string_ops.string_format(inputs=[inputs])

    # Manually compare instead of assertDTensorEqual since outputs are strings.
    self.assertEqual(
        api.fetch_layout(got), Layout.replicated(self.mesh, rank=0))
    for got_tensor in api.unpack(got):
      self.assertEqual(expected_result, got_tensor)

  @parameterized.named_parameters(('Sharded', 'sharded'),
                                  ('Replicated', 'replicated'))
  def testStringFormatOnTPURequiresCopyToMeshToCPU(self, shard_spec):
    self.skipForDeviceType(['CPU', 'GPU'], 'Testing only for TPU.')

    global_ids = test_util.create_device_ids_array((2, 4))
    local_ids = np.ravel(global_ids).tolist()

    tpu_mesh = Mesh([_MESH_DIM_X, _MESH_DIM_Y], global_ids, local_ids,
                    test_util.create_device_list((2, 4), 'TPU'))
    cpu_mesh = Mesh([_MESH_DIM_X, _MESH_DIM_Y], global_ids, local_ids,
                    test_util.create_device_list((2, 4), 'CPU'))

    cpu_layout = Layout.replicated(cpu_mesh, rank=4)
    if shard_spec == 'sharded':
      tpu_layout = Layout.batch_sharded(tpu_mesh, _MESH_DIM_X, rank=4)
    else:
      tpu_layout = Layout.replicated(tpu_mesh, rank=4)

    inputs = stateless_random_ops.stateless_random_uniform(
        shape=(8, 9, 9, 1), seed=[0, 1])
    expected_result = gen_string_ops.string_format(inputs=[inputs])

    inputs = api.relayout(inputs, tpu_layout)
    # StringFormat is not supported on TPU, so copy_to_mesh to the CPU.
    # Since we cannot eager copy_to_mesh from an input with non-replicated
    # layout yet, relayout to replicated layout first, and then transfer to CPU.
    inputs = api.copy_to_mesh(
        api.relayout(inputs, Layout.replicated(tpu_mesh, rank=4)), cpu_layout)

    got = gen_string_ops.string_format(inputs=[inputs])
    # Manually compare instead of assertDTensorEqual since outputs are strings.
    self.assertEqual(api.fetch_layout(got), Layout.replicated(cpu_mesh, rank=0))
    for got_tensor in api.unpack(got):
      self.assertEqual(expected_result, got_tensor)

  @parameterized.named_parameters(
      # TODO(feyu): to_hash_bucket and to_hash_bucket_strong are not defined
      # in the tf MLIR dialect.
      ('ShardedFast', gen_string_ops.string_to_hash_bucket_fast, 'sharded'),
      ('ReplicatedFast', gen_string_ops.string_to_hash_bucket_fast,
       'replicated'),
  )
  def testStringToHashBucket(self, to_hash_bucket_fn, shard_spec):
    self.skipForDeviceType(
        ['GPU', 'TPU'],
        'StringToHashBucket functions not supported on TPU or GPU.')

    inputs = constant_op.constant(['a', 'b', 'c', 'd'], dtype=dtypes.string)
    expected_result = to_hash_bucket_fn(inputs, num_buckets=32)

    if shard_spec == 'sharded':
      layout = Layout.batch_sharded(self.mesh, _MESH_DIM_X, rank=1)
    else:
      layout = Layout.replicated(self.mesh, rank=1)
    inputs = api.relayout(inputs, layout)
    got = to_hash_bucket_fn(inputs, num_buckets=32)

    self.assertDTensorEqual(expected_result, layout, got)

  @parameterized.named_parameters(
      {
          'testcase_name': 'Replicated',
          'shard_type': 'replicated',
      }, {
          'testcase_name': 'BatchSharded',
          'shard_type': 'batch_sharded',
      })
  def testTensorListReserveSetAndGetRetrievesCorrectTensor(self, shard_type):
    self.skipForDeviceType(['TPU', 'GPU'], 'Testing only for CPU.')

    input_tensor = array_ops.ones([4, 4], dtype=dtypes.int32)

    if shard_type == 'replicated':
      layout = Layout.replicated(self.mesh, rank=2)
    else:
      layout = Layout.batch_sharded(self.mesh, _MESH_DIM_X, rank=2)

    @polymorphic_function.function
    def f(input_tensor):
      list_handle = gen_list_ops.tensor_list_reserve(
          element_shape=constant_op.constant([4, 4], dtype=dtypes.int32),
          num_elements=constant_op.constant(4, dtype=dtypes.int32),
          element_dtype=dtypes.int32)
      list_handle = gen_list_ops.tensor_list_set_item(
          input_handle=list_handle,
          index=constant_op.constant(0, dtype=dtypes.int32),
          item=input_tensor)
      return gen_list_ops.tensor_list_get_item(
          input_handle=list_handle,
          index=constant_op.constant(0, dtype=dtypes.int32),
          element_shape=constant_op.constant([4, 4], dtype=dtypes.int32),
          element_dtype=dtypes.int32)

    got_tensor = f(api.relayout(input_tensor, layout))
    self.assertDTensorEqual(input_tensor, Layout.replicated(self.mesh, rank=2),
                            got_tensor)

  @parameterized.named_parameters(
      ('x_unsharded', [_MESH_DIM_X, layout_lib.UNSHARDED]),
      ('unsharded_x', [layout_lib.UNSHARDED, _MESH_DIM_X]),
      ('x_y', [_MESH_DIM_X, _MESH_DIM_Y]),
      ('unsharded_unsharded', [layout_lib.UNSHARDED, layout_lib.UNSHARDED]))
  def testDisableCopyOnRead(self, sharding_specs):
    self.skipForDeviceType(['TPU'], 'Op not supported on TPUs.')

    def f(d_var):
      gen_resource_variable_ops.disable_copy_on_read(resource=d_var.handle)
      return d_var

    layout = Layout(sharding_specs, self.mesh)

    variable = d_variable.DVariable(
        initial_value=api.relayout(
            array_ops.ones([4, 8], dtype=dtypes.float32), layout
        )
    )

    # Eager
    self.assertEqual(api.fetch_layout(f(variable)), layout)

    # Function
    self.assertEqual(
        api.fetch_layout(polymorphic_function.function(f)(variable)), layout)

  def testShardedFilename(self):
    self.skipForDeviceType(['TPU', 'GPU'],
                           'Strings only for CPUs, this is a host-only op.')

    basename = constant_op.constant('dtensor-file')
    shard = constant_op.constant(1, dtype=dtypes.int32)
    num_shards = constant_op.constant(16, dtype=dtypes.int32)

    layout = Layout.replicated(self.mesh, rank=0)

    expected = gen_io_ops.sharded_filename(
        basename=basename, shard=shard, num_shards=num_shards, name=None)

    result = gen_io_ops.sharded_filename(
        basename=api.relayout(basename, layout),
        shard=api.relayout(shard, layout),
        num_shards=api.relayout(num_shards, layout),
    )

    self.assertEqual(api.fetch_layout(result), layout)
    for result_tensor in api.unpack(result):
      self.assertEqual(expected, result_tensor)

  @parameterized.named_parameters(*test_util.product(
      (('_indices_unsharded', [layout_lib.UNSHARDED, layout_lib.UNSHARDED]),
       ('_indices_x', [_MESH_DIM_X, layout_lib.UNSHARDED])),
      (('_updates_unsharded_unsharded',
        [layout_lib.UNSHARDED, layout_lib.UNSHARDED, layout_lib.UNSHARDED]),
       ('_updates_x_unsharded',
        [layout_lib.UNSHARDED, _MESH_DIM_X, layout_lib.UNSHARDED]),
       ('_updates_unsharded_y',
        [layout_lib.UNSHARDED, layout_lib.UNSHARDED, _MESH_DIM_Y]),
       ('_updates_x_y', [layout_lib.UNSHARDED, _MESH_DIM_X, _MESH_DIM_Y]))))
  def testScatterNd(self, indices_spec, updates_spec):
    indices_layout = Layout(indices_spec, self.mesh)
    updates_layout = Layout(updates_spec, self.mesh)
    indices = constant_op.constant([[0], [15]])
    updates = constant_op.constant([[[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7],
                                     [8, 8, 8, 8]],
                                    [[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7],
                                     [8, 8, 8, 8]]])
    shape = [16, 4, 4]

    expected_result = gen_array_ops.scatter_nd(indices, updates, shape)
    got_result = gen_array_ops.scatter_nd(
        api.relayout(indices, indices_layout),
        api.relayout(updates, updates_layout),
        shape,
    )

    self.assertDTensorEqual(expected_result, updates_layout, got_result)


class DTensorConvSPMDTest(test_util.DTensorBaseTest):

  def setUp(self):
    super(DTensorConvSPMDTest, self).setUp()

    # TODO(b/169436213): Re-enable TPU after figuring out multi-chip story.
    self.skipForDeviceType(['TPU'], 'reserving 4 chips on forge is unreliable')

    if config.list_physical_devices('GPU') or config.list_logical_devices(
        'TPU_SYSTEM'):
      self.skipTest(
          'Skipping as 3D mesh with 18 devices cannot be tested on GPU/TPU.')

    # Builds a 2x3x3 mesh.
    self._mesh_dim_b = 'b'
    self._mesh_dim_x = 'x'
    self._mesh_dim_y = 'y'
    self._dims = [self._mesh_dim_b, self._mesh_dim_x, self._mesh_dim_y]

    global_ids = test_util.create_device_ids_array([2, 3, 3])
    local_ids = np.ravel(global_ids).tolist()

    mesh_dict = {
        device: Mesh(self._dims, global_ids, local_ids,
                     test_util.create_device_list([2, 3, 3], 'CPU'))
        for device in ('CPU', 'GPU', 'TPU')
    }
    self.mesh = self.configTestMesh(mesh_dict)
    self._num_devices = self.mesh.size
    test_util.reset_logical_devices('CPU', self.mesh.size)

  @parameterized.named_parameters(test_util_ops.PADDINGS)
  def testConv2DWithFullReplicatedInputs(self, padding):
    np.random.seed(123)

    x_in = np.random.normal(0.0, 1.0, 9 * 9).reshape([1, 9, 9, 1])
    kernel_in = np.array([
        [[[2, 0.1]], [[3, 0.2]]],
        [[[0, 0.3]], [[1, 0.4]]],
    ])

    x = constant_op.constant(x_in, dtype=dtypes.float32)
    kernel = constant_op.constant(kernel_in, dtype=dtypes.float32)
    expected_result = nn_ops.conv2d_v2(
        x, kernel, strides=[1, 1, 1, 1], padding=padding)

    x = api.copy_to_mesh(x, Layout([layout_lib.UNSHARDED] * 4, self.mesh))
    kernel = api.copy_to_mesh(kernel,
                              Layout([layout_lib.UNSHARDED] * 4, self.mesh))

    got = nn_ops.conv2d_v2(x, kernel, strides=[1, 1, 1, 1], padding=padding)

    self.assertDTensorEqual(expected_result,
                            Layout([layout_lib.UNSHARDED] * 4, self.mesh), got)

  @parameterized.product(shard_type=['replicated', 'batch_sharded'])
  def testConv3DBackpropInput(self, shard_type):
    input_sizes = constant_op.constant([4, 4, 4, 4, 4])
    filter_input = stateless_random_ops.stateless_random_uniform(
        shape=[4, 4, 4, 4, 4], seed=[0, 1])
    out_backprop = stateless_random_ops.stateless_random_uniform(
        shape=[4, 4, 4, 4, 4], seed=[0, 1])
    strides = [1, 1, 1, 1, 1]

    expected_result = gen_nn_ops.conv3d_backprop_input_v2(
        input_sizes=input_sizes,
        filter=filter_input,
        out_backprop=out_backprop,
        strides=strides,
        padding='SAME')

    if shard_type == 'replicated':
      grad_layout = Layout.replicated(self.mesh, rank=5)
    else:
      grad_layout = Layout.batch_sharded(self.mesh, self._mesh_dim_b, rank=5)

    got_result = gen_nn_ops.conv3d_backprop_input_v2(
        input_sizes=api.relayout(
            input_sizes, Layout.replicated(self.mesh, rank=1)
        ),
        filter=api.relayout(filter_input, Layout.replicated(self.mesh, rank=5)),
        out_backprop=api.relayout(out_backprop, grad_layout),
        strides=strides,
        padding='SAME',
    )

    self.assertDTensorEqual(expected_result, grad_layout, got_result)

  @parameterized.named_parameters(test_util_ops.PADDINGS)
  def testConv2DWithBatchShardedInputs(self, padding):
    self.skipTest(
        reason='b/272579753: ensure Conv grad Ops know about input layouts.'
    )
    # Reason to flip same shape policy: The backprop of the nn_ops.conv2d_v2 is
    # simply array_ops.ones_like_v2(conv2d_result). However, as DTensor does not
    # control gradient tape, the tape will not attach the layout from
    # conv2d_result to the ones. In normal computation, the backprop pass has
    # more information to pass from the final scalar loss to the conv2d, so
    # this is not a problem.
    # But this well-design unit tests, without same shape policy, it will get a
    # different layout for the inputs' grad.
    np.random.seed(123)

    x_in = np.random.normal(0.0, 1.0, 2 * 9 * 9).reshape([2, 9, 9, 1])
    kernel_in = np.array([
        [[[2, 0.1]], [[3, 0.2]]],
        [[[0, 0.3]], [[1, 0.4]]],
    ])

    x = constant_op.constant(x_in, dtype=dtypes.float32)
    kernel = constant_op.constant(kernel_in, dtype=dtypes.float32)
    with backprop.GradientTape() as tape:
      tape.watch([x, kernel])
      expected_result = nn_ops.conv2d_v2(
          x, kernel, strides=[1, 1, 1, 1], padding=padding)
    expected_input_gradient, expected_filter_gradient = tape.gradient(
        expected_result, [x, kernel])

    x = api.relayout(
        x, Layout([self._dims[0]] + [layout_lib.UNSHARDED] * 3, self.mesh)
    )
    kernel = api.relayout(kernel, Layout([layout_lib.UNSHARDED] * 4, self.mesh))
    # Explicitly open the scope as ops generated from tape could be broadcasted
    # to replicated by default.
    with api.default_mesh(self.mesh):
      with backprop.GradientTape() as tape:
        tape.watch([x, kernel])
        got = nn_ops.conv2d_v2(x, kernel, strides=[1, 1, 1, 1], padding=padding)
      got_input_gradient, got_filter_filter = tape.gradient(got, [x, kernel])

    self.assertDTensorEqual(
        expected_result,
        Layout([self._dims[0]] + [layout_lib.UNSHARDED] * 3, self.mesh), got)
    self.assertDTensorEqual(
        expected_input_gradient,
        Layout([self._dims[0]] + [layout_lib.UNSHARDED] * 3, self.mesh),
        got_input_gradient)
    self.assertDTensorEqual(expected_filter_gradient,
                            Layout([layout_lib.UNSHARDED] * 4, self.mesh),
                            got_filter_filter)

  @parameterized.named_parameters(test_util_ops.PADDINGS)
  def testMaxPoolWithBatchShardedInputs(self, padding):
    np.random.seed(123)
    row_window_size = 3
    col_window_size = 4
    window_size = [1, row_window_size, col_window_size, 1]
    stride_size = [1, row_window_size - 1, col_window_size - 1, 1]

    num_rows = (row_window_size - 1) * 5 + 1
    num_cols = (col_window_size - 1) * 7 + 1
    x_in = np.random.normal(0.0, 1.0, 2 * num_rows * num_cols * 3).reshape(
        [2, num_rows, num_cols, 3])

    inputs = constant_op.constant(x_in, dtype=dtypes.float32)
    expected_result = nn_ops.max_pool_v2(inputs, window_size, stride_size,
                                         padding)

    x = api.relayout(
        inputs, Layout([self._dims[0]] + [layout_lib.UNSHARDED] * 3, self.mesh)
    )

    got = nn_ops.max_pool_v2(x, window_size, stride_size, padding)

    self.assertDTensorEqual(
        expected_result,
        Layout([self._dims[0]] + [layout_lib.UNSHARDED] * 3, self.mesh), got)

  @parameterized.named_parameters(test_util_ops.PADDINGS)
  def testMaxPoolGradWithBatchShardedInputs(self, padding):
    np.random.seed(123)
    row_window_size = 3
    col_window_size = 4
    window_size = [1, row_window_size, col_window_size, 1]
    stride_size = [1, row_window_size - 1, col_window_size - 1, 1]

    num_rows = (row_window_size - 1) * 5 + 1
    num_cols = (col_window_size - 1) * 7 + 1
    x_in = np.random.normal(0.0, 1.0, 2 * num_rows * num_cols * 3).reshape(
        [2, num_rows, num_cols, 3])
    inputs = constant_op.constant(x_in, dtype=dtypes.float32)
    with backprop.GradientTape() as tape:
      tape.watch([inputs])
      expected_result = nn_ops.max_pool_v2(inputs, window_size, stride_size,
                                           padding)
    expected_grad = tape.gradient(expected_result, [inputs])

    x = api.relayout(
        inputs, Layout([self._dims[0]] + [layout_lib.UNSHARDED] * 3, self.mesh)
    )

    with api.default_mesh(self.mesh):
      with backprop.GradientTape() as tape:
        tape.watch([x])
        dtensor_result = nn_ops.max_pool_v2(x, window_size, stride_size,
                                            padding)
      dtensor_grad = tape.gradient(dtensor_result, [x])

    self.assertDTensorEqual(
        expected_grad[0],
        Layout([self._dims[0]] + [layout_lib.UNSHARDED] * 3, self.mesh),
        dtensor_grad[0])


class DTensorLayoutPropSPMDTest(test_util.DTensorBaseTest):

  def setUp(self):
    super(DTensorLayoutPropSPMDTest, self).setUp()

    self.skipForDeviceType(['TPU'],
                           'all tests require 8 TPU cores.',
                           unless_device_count_equals_to=8)

    global_ids = test_util.create_device_ids_array((2, 4))
    local_ids = np.ravel(global_ids).tolist()
    mesh_dict = {
        device: Mesh([_MESH_DIM_X, _MESH_DIM_Y], global_ids, local_ids,
                     test_util.create_device_list((2, 4), device))
        for device in ('CPU', 'GPU', 'TPU')
    }
    self.mesh = self.configTestMesh(mesh_dict)

    self.scalar_replicated_layout = Layout.replicated(self.mesh, rank=0)

    self.replicated_layout_1d = Layout.replicated(self.mesh, rank=1)
    self.first_dimension_sharded_layout_1d = Layout.batch_sharded(
        self.mesh, _MESH_DIM_X, rank=1)

    self.replicated_layout_2d = Layout.replicated(self.mesh, rank=2)
    self.first_dimension_sharded_layout_2d = Layout.batch_sharded(
        self.mesh, _MESH_DIM_X, rank=2)
    self.last_dimension_sharded_layout_2d = Layout.inner_sharded(
        self.mesh, _MESH_DIM_X, rank=2)

    self.replicated_layout_3d = Layout.replicated(self.mesh, rank=3)
    self.first_dimension_sharded_layout_3d = Layout.batch_sharded(
        self.mesh, _MESH_DIM_X, rank=3)
    self.middle_dimension_sharded_layout_3d = Layout(
        [layout_lib.UNSHARDED, _MESH_DIM_X, layout_lib.UNSHARDED], self.mesh)
    self.last_dimension_sharded_layout_3d = Layout.inner_sharded(
        self.mesh, _MESH_DIM_X, rank=3)

    # Make a list so that we can index layouts by sharding dimension and then
    # by rank.
    self.layouts = [
        [
            None, self.first_dimension_sharded_layout_1d,
            self.first_dimension_sharded_layout_2d,
            self.first_dimension_sharded_layout_3d
        ],
        [
            None, None, self.last_dimension_sharded_layout_2d,
            self.middle_dimension_sharded_layout_3d
        ],
        [None, None, None, self.last_dimension_sharded_layout_3d],
        # Keep this at the end so a sharding dim of -1 corresponds to
        # replicated.
        [
            None, self.replicated_layout_1d, self.replicated_layout_2d,
            self.replicated_layout_3d
        ],
    ]

  @parameterized.named_parameters(
      ('_collapse_all_dims', [2, 4, 3], 0, [24], 0),
      ('_collapse_partial_dims', [2, 4, 3], 0, [8, 3], 0),
      ('_collapse_partial_dims_with_ones', [2, 4, 1], 0, [8, 1], 0),
      ('_collapse_partial_dims_non_batch', [2, 4, 3], 1, [2, 12], 1),
      ('_collapse_partial_dims_non_batch_with_ones', [1, 4, 3], 1, [1, 12], 1),
      ('_uncollapse_all_dims', [24], 0, [2, 4, 3], 0),
      ('_uncollapse_partial_dims', [4, 4], 0, [2, 2, 4], 0),
      ('_uncollapse_partial_dims_non_batch', [2, 12], 1, [2, 4, 3], 1),
      ('_expand_dims', [2, 2], 0, [2, 1, 2], 0),
      ('_squeeze', [2, 1, 2], 0, [2, 2], 0),
  )
  def testReshape(self, src_shape, src_sharding_dim, target_shape,
                  target_sharding_dim):
    src_numpy = np.random.uniform(size=src_shape)
    src = constant_op.constant(src_numpy, dtype=dtypes.float32)

    expected = array_ops.reshape(src, target_shape)

    src = api.relayout(src, self.layouts[src_sharding_dim][len(src_shape)])
    dtensor_result = array_ops.reshape(src, target_shape)
    self.assertDTensorEqual(
        expected, self.layouts[target_sharding_dim][len(target_shape)],
        dtensor_result)

  def testReshapeWithAllConcatOutputLayout(self):
    src_shape = [2, 4, 3]
    target_shape = [2, 12]
    src_layout = Layout(
        [layout_lib.UNSHARDED, _MESH_DIM_X, layout_lib.UNSHARDED], self.mesh)
    target_layout = Layout.replicated(self.mesh, rank=2)

    src_numpy = np.random.uniform(size=src_shape)
    src = constant_op.constant(src_numpy, dtype=dtypes.float32)

    expected = array_ops.reshape(src, target_shape)

    src = api.relayout(src, src_layout)
    with api._dtensor_device()._default_layout(target_layout):
      dtensor_result = array_ops.reshape(src, target_shape)
    self.assertDTensorEqual(expected, target_layout, dtensor_result)

  def testReshapeWithSplitOnOutputLayout(self):
    src_shape = [2, 4, 3]
    target_shape = [2, 12]
    src_layout = Layout.replicated(self.mesh, rank=3)
    target_layout = Layout([layout_lib.UNSHARDED, _MESH_DIM_X], self.mesh)

    src_numpy = np.random.uniform(size=src_shape)
    src = constant_op.constant(src_numpy, dtype=dtypes.float32)

    expected = array_ops.reshape(src, target_shape)

    src = api.relayout(src, src_layout)
    with api._dtensor_device()._default_layout(target_layout):
      dtensor_result = array_ops.reshape(src, target_shape)
    self.assertDTensorEqual(expected, target_layout, dtensor_result)

  @parameterized.named_parameters(
      ('_shard_on_first_output_dim', [4], [2], -1, 0),
      ('_shard_on_first_input_dim', [4, 4], [1, 2], 0, 0),
  )
  def testTile(self, src_shape, multiples, src_sharding_dim,
               target_sharding_dim):
    src_numpy = np.random.uniform(size=src_shape)
    src = constant_op.constant(src_numpy, dtype=dtypes.float32)

    expected = gen_array_ops.tile(src, multiples)

    src = api.relayout(src, self.layouts[src_sharding_dim][len(src_shape)])
    with api._dtensor_device()._default_layout(
        self.layouts[target_sharding_dim][len(src_shape)]):
      dtensor_result = gen_array_ops.tile(src, multiples)
    self.assertDTensorEqual(expected,
                            self.layouts[target_sharding_dim][len(src_shape)],
                            dtensor_result)

  @parameterized.named_parameters(
      ('_unsharded_unsharded', [layout_lib.UNSHARDED, layout_lib.UNSHARDED]),
      ('_x_unsharded', [_MESH_DIM_X, layout_lib.UNSHARDED]),
      ('_unsharded_x', [layout_lib.UNSHARDED, _MESH_DIM_X]),
      ('_y_unsharded', [_MESH_DIM_Y, layout_lib.UNSHARDED]),
      ('_unsharded_y', [layout_lib.UNSHARDED, _MESH_DIM_Y]),
      ('_x_y', [_MESH_DIM_X, _MESH_DIM_Y]),
      ('_y_x', [_MESH_DIM_Y, _MESH_DIM_X]))
  def testConst(self, sharding):
    src_numpy = np.random.uniform(size=[4, 12])
    zero_numpy = np.zeros_like(src_numpy)
    expected = constant_op.constant(src_numpy, dtype=dtypes.float32)  # 4x12

    layout = Layout(sharding, self.mesh)
    zeros = api.relayout(zero_numpy, layout)

    # We can't execute const on dtensor device eagerly, so we wrap it in a
    # function and pass a dtensor (which we ignore) to the function in order to
    # trigger dtensor execution.
    @polymorphic_function.function
    def const_test(_):
      with api._dtensor_device()._default_layout(layout):
        return constant_op.constant(src_numpy, dtype=dtypes.float32)  # 4x12

    dtensor_result = const_test(zeros)

    self.assertDTensorEqual(expected, layout, dtensor_result)

  @parameterized.named_parameters(
      ('_unsharded_unsharded', [layout_lib.UNSHARDED, layout_lib.UNSHARDED]),
      ('_x_unsharded', [_MESH_DIM_X, layout_lib.UNSHARDED]),
      ('_unsharded_x', [layout_lib.UNSHARDED, _MESH_DIM_X]),
      ('_y_unsharded', [_MESH_DIM_Y, layout_lib.UNSHARDED]),
      ('_unsharded_y', [layout_lib.UNSHARDED, _MESH_DIM_Y]),
      ('_x_y', [_MESH_DIM_X, _MESH_DIM_Y]),
      ('_y_x', [_MESH_DIM_Y, _MESH_DIM_X]))
  def testConstScalar(self, sharding):
    src_numpy = np.random.uniform()
    zero_numpy = np.zeros(shape=[4, 12])
    # Note: The diff between this test and the one above is `src_numpy` is
    # single value here; while the one above is full shape numpy array.
    expected = constant_op.constant(
        src_numpy, shape=[4, 12], dtype=dtypes.float32)  # 4x12

    layout = Layout(sharding, self.mesh)
    zeros = api.relayout(zero_numpy, layout)

    # We can't execute const on dtensor device eagerly, so we wrap it in a
    # function and pass a dtensor (which we ignore) to the function in order to
    # trigger dtensor execution.
    @polymorphic_function.function
    def const_test(_):
      with api._dtensor_device()._default_layout(layout):
        return constant_op.constant(
            src_numpy, shape=[4, 12], dtype=dtypes.float32)  # 4x12

    dtensor_result = const_test(zeros)

    self.assertDTensorEqual(expected, layout, dtensor_result)

  def testRandomOpByOp(self):
    with ops.device_v2(self.mesh.device_type()):
      seed = constant_op.constant([1, 2])
      expected = gen_stateless_random_ops.stateless_random_uniform_full_int(
          shape=[2, 2], seed=seed, dtype=dtypes.int64)

    seed = api.copy_to_mesh(seed, Layout.replicated(rank=1, mesh=self.mesh))
    dtensor_result = gen_stateless_random_ops.stateless_random_uniform_full_int(
        shape=[2, 2], seed=seed, dtype=dtypes.int64)
    # Note that we only expect the same result (a) for the same device since
    # this determines the algorithm, and (b) for fully-replicated output layouts
    # since device_id hashing does not reproduce exactly the single-machine
    # numbers, only their distribution.
    self.assertDTensorEqual(expected, Layout.replicated(rank=2, mesh=self.mesh),
                            dtensor_result)

  def testRange(self):
    start = 0
    limit = 3
    expected = math_ops.range(start, limit)

    layout = Layout([layout_lib.UNSHARDED], self.mesh)

    with api._dtensor_device()._default_layout(layout):
      dtensor_result = math_ops.range(start, limit)

    self.assertDTensorEqual(expected, layout, dtensor_result)

  def testBroadcastTo(self):
    inputs = constant_op.constant([1, 2, 3])
    shape = [3, 3]
    expected = gen_array_ops.broadcast_to(inputs, shape)

    inputs = api.copy_to_mesh(inputs, Layout.replicated(self.mesh, rank=1))
    with api.default_mesh(self.mesh):
      dtensor_result = gen_array_ops.broadcast_to(inputs, shape)

    self.assertDTensorEqual(expected, Layout.replicated(self.mesh, rank=2),
                            dtensor_result)

  @parameterized.named_parameters(
      {
          'testcase_name': 'Replicated',
          'sharding': [layout_lib.UNSHARDED, layout_lib.UNSHARDED],
      },
      {
          'testcase_name': 'BatchSharded',
          'sharding': [_MESH_DIM_X, layout_lib.UNSHARDED],
      },
      {
          'testcase_name': 'FullySharded',
          'sharding': [_MESH_DIM_X, _MESH_DIM_Y],
      },
  )
  def testTanhGrad(self, sharding):
    inputs = constant_op.constant([[1, 2, 3, 4], [5, 6, 7, 8]],
                                  dtype=dtypes.float32)

    with backprop.GradientTape() as tape:
      tape.watch([inputs])
      expected_result = gen_math_ops.tanh(inputs)
    expected_grad = tape.gradient(expected_result, [inputs])

    layout = Layout(sharding, self.mesh)
    inputs = api.relayout(inputs.numpy(), layout)
    with api.default_mesh(self.mesh):
      with backprop.GradientTape() as tape:
        tape.watch([inputs])
        dtensor_result = gen_math_ops.tanh(inputs)
      dtensor_grad = tape.gradient(dtensor_result, [inputs])
    # df2x2 lowers the tanh preceision to 1e-4.
    self.assertDTensorEqual(
        expected_grad[0],
        layout,
        dtensor_grad[0],
        tol=1e-4
        if 'TPU' in self.mesh.local_devices()[0] else test_util.DEFAULT_TOL)

  @parameterized.named_parameters(
      {
          'testcase_name': 'Replicated',
          'shard_type': 'replicated',
      }, {
          'testcase_name': 'BatchSharded',
          'shard_type': 'batch_sharded',
      })
  def testIdentityN(self, shard_type):
    layout = (
        Layout.replicated(self.mesh, rank=2) if shard_type == 'replicated' else
        Layout.batch_sharded(self.mesh, _MESH_DIM_X, rank=2))

    a = constant_op.constant([[10., 20.], [30., 40.]])
    b = constant_op.constant([[50., 60.], [70., 80.]])
    expected_c, expected_d = gen_array_ops.identity_n([a, b])

    if shard_type == 'replicated':
      a = api.copy_to_mesh(a, layout)
      b = api.copy_to_mesh(b, layout)
    else:
      a = api.relayout(a, layout)
      b = api.relayout(b, layout)
    dtensor_c, dtensor_d = gen_array_ops.identity_n([a, b])

    self.assertDTensorEqual(expected_c, layout, dtensor_c)
    self.assertDTensorEqual(expected_d, layout, dtensor_d)

  @parameterized.named_parameters(
      {
          'testcase_name': 'Replicated',
          'sharding': [layout_lib.UNSHARDED, layout_lib.UNSHARDED]
      },
      {
          'testcase_name': 'BatchSharded',
          'sharding': [_MESH_DIM_X, layout_lib.UNSHARDED],
      },
      {
          'testcase_name': 'FullySharded',
          'sharding': [_MESH_DIM_X, _MESH_DIM_Y],
      },
  )
  def testArgMax(self, sharding):
    for axis in [0, 1]:
      inputs = constant_op.constant([[1, 2, 3, 4], [5, 6, 7, 8]],
                                    dtype=dtypes.float32)
      expect_result = math_ops.argmax_v2(
          inputs, axis=axis, output_type=dtypes.int32)

      input_layout = Layout(sharding, self.mesh)
      inputs = api.relayout(inputs.numpy(), input_layout)

      output_layout = Layout([sharding[1 - axis]], self.mesh)
      dtensor_result = math_ops.argmax_v2(
          inputs, axis=axis, output_type=dtypes.int32)
      self.assertDTensorEqual(expect_result, output_layout, dtensor_result)

  @parameterized.named_parameters(('PositiveAxis', 0), ('NegativeAxis', -2))
  def testSplitOpsWithFullyReplicatedInputs(self, split_axis):
    t = random_ops.random_uniform([2, 4])
    expected_result = array_ops.split(t, 2, axis=split_axis)
    t = api.copy_to_mesh(t, self.replicated_layout_2d)
    dtensor_result = array_ops.split(t, 2, axis=split_axis)
    self.assertIsInstance(expected_result, list)
    self.assertIsInstance(dtensor_result, list)
    self.assertLen(expected_result, 2)
    self.assertLen(dtensor_result, 2)
    self.assertDTensorEqual(expected_result[0], self.replicated_layout_2d,
                            dtensor_result[0])
    self.assertDTensorEqual(expected_result[1], self.replicated_layout_2d,
                            dtensor_result[1])

  def testSplitOpsWithNonSplitAxisSharded(self):
    t = random_ops.random_uniform([2, 4])
    expected_result = array_ops.split(t, 2, axis=1)
    t = api.relayout(t, self.first_dimension_sharded_layout_2d)
    dtensor_result = array_ops.split(t, 2, axis=1)
    self.assertIsInstance(expected_result, list)
    self.assertIsInstance(dtensor_result, list)
    self.assertLen(expected_result, 2)
    self.assertLen(dtensor_result, 2)
    self.assertDTensorEqual(expected_result[0],
                            self.first_dimension_sharded_layout_2d,
                            dtensor_result[0])
    self.assertDTensorEqual(expected_result[1],
                            self.first_dimension_sharded_layout_2d,
                            dtensor_result[1])

  def testSplitOpsWithSplitAxisShardedRaisesError(self):
    t = random_ops.random_uniform([2, 4])
    t = api.relayout(t, self.last_dimension_sharded_layout_2d)
    with self.assertRaises(errors_impl.UnknownError):
      # Spliting over sharded dimension is not yet supported.
      _ = array_ops.split(t, 2, axis=1)

  @parameterized.named_parameters(('PositiveAxis', 1), ('NegativeAxis', -1))
  def testSplitVOpsWithFullyReplicatedInputs(self, split_axis):
    t = random_ops.random_uniform([4, 5])
    expected_result = array_ops.split(t, [1, 3, 1], axis=split_axis)
    t = api.copy_to_mesh(t, self.replicated_layout_2d)
    dtensor_result = array_ops.split(t, [1, 3, 1], axis=split_axis)
    self.assertIsInstance(expected_result, list)
    self.assertIsInstance(dtensor_result, list)
    self.assertLen(expected_result, 3)
    self.assertLen(dtensor_result, 3)
    self.assertDTensorEqual(expected_result[0], self.replicated_layout_2d,
                            dtensor_result[0])
    self.assertDTensorEqual(expected_result[1], self.replicated_layout_2d,
                            dtensor_result[1])
    self.assertDTensorEqual(expected_result[2], self.replicated_layout_2d,
                            dtensor_result[2])

  def testSplitVOpsWithNonSplitAxisSharded(self):
    t = random_ops.random_uniform([4, 5])
    expected_result = array_ops.split(t, [1, 3, 1], axis=1)
    t = api.relayout(t, self.first_dimension_sharded_layout_2d)
    dtensor_result = array_ops.split(t, [1, 3, 1], axis=1)
    self.assertIsInstance(expected_result, list)
    self.assertIsInstance(dtensor_result, list)
    self.assertLen(expected_result, 3)
    self.assertLen(dtensor_result, 3)
    self.assertDTensorEqual(expected_result[0],
                            self.first_dimension_sharded_layout_2d,
                            dtensor_result[0])
    self.assertDTensorEqual(expected_result[1],
                            self.first_dimension_sharded_layout_2d,
                            dtensor_result[1])
    self.assertDTensorEqual(expected_result[2],
                            self.first_dimension_sharded_layout_2d,
                            dtensor_result[2])

  def testSplitVOpsWithSplitAxisShardedRaisesError(self):
    t = random_ops.random_uniform([2, 4])
    t = api.relayout(t, self.last_dimension_sharded_layout_2d)
    with self.assertRaises(errors_impl.UnknownError):
      # Spliting over sharded dimension is not yet supported.
      _ = array_ops.split(t, [1, 1, 2], axis=1)

  def testUnpackWithFullyReplicatedInputs(self):
    t = constant_op.constant([[1., 2., 3., 4.], [5., 6., 7., 8.]])
    expected_result = array_ops_stack.unstack(t, axis=0)
    t = api.copy_to_mesh(t, self.replicated_layout_2d)
    dtensor_result = array_ops_stack.unstack(t, axis=0)
    self.assertIsInstance(expected_result, list)
    self.assertIsInstance(dtensor_result, list)
    self.assertLen(expected_result, 2)
    self.assertLen(dtensor_result, 2)
    self.assertDTensorEqual(expected_result[0], self.replicated_layout_1d,
                            dtensor_result[0])
    self.assertDTensorEqual(expected_result[1], self.replicated_layout_1d,
                            dtensor_result[1])

  def testUnpackWithShardedInput(self):
    t = constant_op.constant([[1., 2., 3., 4.], [5., 6., 7., 8.]])
    expected_result = array_ops_stack.unstack(t, axis=1)
    t = api.relayout(t, Layout([layout_lib.UNSHARDED, _MESH_DIM_X], self.mesh))
    dtensor_result = array_ops_stack.unstack(t, axis=1)
    self.assertIsInstance(expected_result, list)
    self.assertIsInstance(dtensor_result, list)
    self.assertLen(expected_result, 4)
    self.assertLen(dtensor_result, 4)
    for i in range(4):
      self.assertDTensorEqual(expected_result[i], self.replicated_layout_1d,
                              dtensor_result[i])

  @parameterized.named_parameters(
      (
          '_unshard_input_batch_sharded',
          'bsd,dnh->bsnh',
          [8, 128, 128],
          [_MESH_DIM_X, layout_lib.UNSHARDED, _MESH_DIM_Y],
          [128, 16, 64],
          [layout_lib.UNSHARDED, _MESH_DIM_Y, layout_lib.UNSHARDED],
          [
              _MESH_DIM_X, layout_lib.UNSHARDED, _MESH_DIM_Y,
              layout_lib.UNSHARDED
          ],
      ),
      (
          '_unshard_input_all_reduce_output',
          'bfi,bfd->di',
          [8, 128, 256],
          [_MESH_DIM_X, layout_lib.UNSHARDED, _MESH_DIM_Y],
          [8, 128, 128],
          [_MESH_DIM_X, layout_lib.UNSHARDED, _MESH_DIM_Y],
          [layout_lib.UNSHARDED, _MESH_DIM_Y],
      ),
      (
          '_contracting_dim_sharded_in_output',
          'bfnh,nhd->bfd',
          [8, 128, 16, 8],
          [
              _MESH_DIM_X, layout_lib.UNSHARDED, _MESH_DIM_Y,
              layout_lib.UNSHARDED
          ],
          [16, 8, 128],
          [_MESH_DIM_Y, layout_lib.UNSHARDED, layout_lib.UNSHARDED],
          [_MESH_DIM_X, layout_lib.UNSHARDED, _MESH_DIM_Y],
      ),
  )
  def testEinsum(self, equation, a_shape, a_layout, b_shape, b_layout,
                 output_layout):
    a_numpy = np.random.uniform(size=a_shape)
    b_numpy = np.random.uniform(size=b_shape)
    a = constant_op.constant(a_numpy, dtype=dtypes.float32)
    b = constant_op.constant(b_numpy, dtype=dtypes.float32)

    expected = special_math_ops.einsum(equation, a, b)

    a_layout = Layout(a_layout, self.mesh)
    b_layout = Layout(b_layout, self.mesh)
    output_layout = Layout(output_layout, self.mesh)

    a = api.relayout(a, a_layout)
    b = api.relayout(b, b_layout)

    @polymorphic_function.function
    def einsum_fn(x, y):
      result = special_math_ops.einsum(equation, x, y)
      return api.relayout(result, output_layout)

    dtensor_result = einsum_fn(a, b)
    self.assertDTensorEqual(expected, output_layout, dtensor_result)

  def testAddV2DifferentSharding(self):
    a_numpy = np.random.uniform(size=[16, 8])
    b_numpy = np.random.uniform(size=[16, 8])
    a = constant_op.constant(a_numpy, dtype=dtypes.float32)
    b = constant_op.constant(b_numpy, dtype=dtypes.float32)

    expected = math_ops.add(a, b)

    a_layout = Layout([_MESH_DIM_Y, layout_lib.UNSHARDED], self.mesh)
    b_layout = Layout([_MESH_DIM_X, layout_lib.UNSHARDED], self.mesh)

    a = api.relayout(a, a_layout)
    b = api.relayout(b, b_layout)

    @polymorphic_function.function
    def add_fn(x, y):
      result = math_ops.add(x, y)
      return api.relayout(result, a_layout)

    dtensor_result = add_fn(a, b)
    self.assertDTensorEqual(expected, a_layout, dtensor_result)

  @parameterized.named_parameters(test_util_ops.BINARY_FLOAT_OPS)
  def testBinaryOpsWithArbitrarySharding(self, op):
    tol = select_tol(op, self.mesh, test_util.DEFAULT_TOL, low_res_tol=1e-2)
    a = constant_op.constant(
        np.array([[1., 2.], [3., 4.]]), dtype=dtypes.float32)
    b = constant_op.constant(
        np.array([[10., 20.], [30., 40.]]), dtype=dtypes.float32)
    expected_result = op(a, b)

    layout_x_n = Layout([_MESH_DIM_X, layout_lib.UNSHARDED], self.mesh)
    layout_n_x = Layout([layout_lib.UNSHARDED, _MESH_DIM_X], self.mesh)

    a = api.relayout(a, layout_x_n)
    b = api.relayout(b, layout_n_x)

    with api._dtensor_device()._default_layout(layout_n_x):
      dtensor_result = op(a, b)

    self.assertDTensorEqual(
        expected_result, layout_n_x, dtensor_result, tol=tol)

  def testUnsortedSegmentSum(self):
    self.skipForDeviceType(['GPU'], 'reduce on GPU only supports int64')
    num_segments = 12
    data = np.random.uniform(size=[num_segments, 4])
    segment_ids = np.random.randint(
        0, num_segments, size=num_segments, dtype=np.int32)
    expected = gen_math_ops.unsorted_segment_sum(data, segment_ids,
                                                 num_segments)

    data = api.relayout(data, Layout.replicated(self.mesh, 2))
    segment_ids = api.relayout(
        segment_ids, Layout.batch_sharded(self.mesh, _MESH_DIM_Y, rank=1)
    )
    with api.default_mesh(self.mesh):
      dtensor_result = gen_math_ops.unsorted_segment_sum(
          data, segment_ids, num_segments)
      expected_layout = Layout.replicated(self.mesh, 2)
      self.assertDTensorEqual(expected, expected_layout, dtensor_result)

  def testUnsortedSegmentSumWithShardingData(self):
    self.skipForDeviceType(['GPU'], 'reduce on GPU only supports int64')
    num_segments = 12
    data = np.random.uniform(size=[num_segments, 4]).astype(np.float32)
    segment_ids = np.random.randint(
        0, num_segments, size=num_segments, dtype=np.int32)
    expected = gen_math_ops.unsorted_segment_sum(data, segment_ids,
                                                 num_segments)

    data = api.relayout(
        data, Layout([layout_lib.UNSHARDED, _MESH_DIM_X], self.mesh)
    )
    segment_ids = api.relayout(segment_ids, Layout.replicated(self.mesh, 1))

    with api._dtensor_device()._default_layout(Layout.replicated(self.mesh, 2)):
      with api.default_mesh(self.mesh):
        dtensor_result = gen_math_ops.unsorted_segment_sum(
            data, segment_ids, num_segments)
        expected_layout = Layout.replicated(self.mesh, 2)
        self.assertDTensorEqual(expected, expected_layout, dtensor_result)

  def testGatherShardingParams(self):
    params = np.arange(1000 * 4).reshape((1000, 4)).astype(np.float32)
    # "batch" size = 2, num_indices = 4 per example
    indices = np.random.randint(
        0, 1000, size=4 * 4).reshape((4, 4)).astype(np.int32)
    expected = array_ops.gather_v2(params, indices, axis=0)

    params = api.relayout(
        params, layout=Layout([_MESH_DIM_X, layout_lib.UNSHARDED], self.mesh)
    )
    indices = api.relayout(
        indices, Layout([layout_lib.UNSHARDED, layout_lib.UNSHARDED], self.mesh)
    )

    expected_layout = Layout(
        [layout_lib.UNSHARDED, layout_lib.UNSHARDED, layout_lib.UNSHARDED],
        self.mesh)
    with api.default_mesh(self.mesh):
      dtensor_result = array_ops.gather_v2(params, indices, axis=0)
      self.assertDTensorEqual(expected, expected_layout, dtensor_result)

  @parameterized.named_parameters(
      ('_float32', np.float32),
      ('_int32', np.int32))
  @mock.patch.dict(os.environ,
                   {'LOWER_DTENSOR_GATHER_TO_COLLECTIVE_GATHER_V2': '1'})
  def testGatherIndicesShardingParams(self, data_type):
    params = np.arange(1000 * 4).reshape((1000, 4)).astype(data_type)
    # "batch" size = 2, num_indices = 4 per example
    indices = np.random.randint(
        0, 1000, size=4 * 4).reshape((4, 4)).astype(np.int32)
    expected = array_ops.gather_v2(params, indices, axis=0)

    params = api.relayout(
        params, layout=Layout([_MESH_DIM_X, layout_lib.UNSHARDED], self.mesh)
    )
    indices = api.relayout(
        indices, Layout([_MESH_DIM_X, layout_lib.UNSHARDED], self.mesh)
    )

    expected_layout = Layout(
        [_MESH_DIM_X, layout_lib.UNSHARDED, layout_lib.UNSHARDED], self.mesh)
    with api.default_mesh(self.mesh):
      dtensor_result = array_ops.gather_v2(params, indices, axis=0)
      self.assertDTensorEqual(expected, expected_layout, dtensor_result)

  def testGatherShardingParamsAxisIs1(self):
    params = np.arange(1000 * 32).reshape((1000, 32)).astype(np.float32)
    # "batch" size = 2, num_indices = 4 per example
    indices = np.random.randint(
        0, 32, size=4 * 4).reshape((4, 4)).astype(np.int32)
    expected = array_ops.gather_v2(params, indices, axis=1)

    params = api.relayout(
        params, layout=Layout([_MESH_DIM_X, layout_lib.UNSHARDED], self.mesh)
    )
    indices = api.relayout(
        indices, Layout([layout_lib.UNSHARDED, layout_lib.UNSHARDED], self.mesh)
    )

    expected_layout = Layout(
        [_MESH_DIM_X, layout_lib.UNSHARDED, layout_lib.UNSHARDED], self.mesh)
    with api.default_mesh(self.mesh):
      dtensor_result = array_ops.gather_v2(params, indices, axis=1)
      self.assertDTensorEqual(expected, expected_layout, dtensor_result)

  def testGatherShardingIndices(self):
    params = np.arange(1000 * 4).reshape((1000, 4))
    # "batch" size = 2, num_indices = 4 per example
    indices = np.random.randint(0, 1000, size=4 * 4).reshape((4, 4))
    expected = array_ops.gather_v2(params, indices, axis=0)

    params = api.relayout(params, layout=Layout.replicated(self.mesh, 2))
    indices = api.relayout(
        indices, Layout([layout_lib.UNSHARDED, layout_lib.UNSHARDED], self.mesh)
    )
    expected_layout = Layout(
        [layout_lib.UNSHARDED, _MESH_DIM_Y, layout_lib.UNSHARDED], self.mesh)
    with api.default_mesh(self.mesh):
      with api._dtensor_device()._default_layout(expected_layout):
        dtensor_result = array_ops.gather_v2(params, indices, axis=0)
        self.assertDTensorEqual(expected, expected_layout, dtensor_result)

  def testGatherShardingParamsWithBatchDim(self):
    params = np.arange(128 * 1000 * 2).reshape(
        (128, 1000, 2)).astype(np.float32)
    indices = np.random.randint(
        0, 1000, size=128 * 4 * 4).reshape((128, 4, 4)).astype(np.int32)
    expected = array_ops.gather_v2(params, indices, batch_dims=1, axis=1)

    params = api.relayout(
        params,
        layout=Layout(
            [_MESH_DIM_X, layout_lib.UNSHARDED, layout_lib.UNSHARDED], self.mesh
        ),
    )
    indices = api.relayout(
        indices,
        Layout(
            [layout_lib.UNSHARDED, layout_lib.UNSHARDED, layout_lib.UNSHARDED],
            self.mesh,
        ),
    )

    expected_layout = Layout([
        _MESH_DIM_X, layout_lib.UNSHARDED, layout_lib.UNSHARDED,
        layout_lib.UNSHARDED
    ], self.mesh)
    with api.default_mesh(self.mesh):
      dtensor_result = array_ops.gather_v2(
          params, indices, batch_dims=1, axis=1)
      self.assertDTensorEqual(expected, expected_layout, dtensor_result)

  def testGatherShardingParamsWithBatchDimAxisIs2(self):
    params = np.arange(128 * 1000 * 32).reshape(
        (128, 1000, 32)).astype(np.float32)
    indices = np.random.randint(
        0, 32, size=128 * 4 * 4).reshape((128, 4, 4)).astype(np.int32)
    expected = array_ops.gather_v2(params, indices, batch_dims=1, axis=2)

    params = api.relayout(
        params,
        layout=Layout(
            [_MESH_DIM_X, _MESH_DIM_Y, layout_lib.UNSHARDED], self.mesh
        ),
    )
    indices = api.relayout(
        indices,
        Layout(
            [layout_lib.UNSHARDED, layout_lib.UNSHARDED, layout_lib.UNSHARDED],
            self.mesh,
        ),
    )

    expected_layout = Layout(
        [_MESH_DIM_X, _MESH_DIM_Y, layout_lib.UNSHARDED, layout_lib.UNSHARDED],
        self.mesh)
    with api.default_mesh(self.mesh):
      dtensor_result = array_ops.gather_v2(
          params, indices, batch_dims=1, axis=2)
      self.assertDTensorEqual(expected, expected_layout, dtensor_result)

  def testGatherShardingParamsWithBatchDimAxisIs2ShardingAfterAxis(self):
    params = np.arange(128 * 5 * 32 * 64).reshape(
        (128, 5, 32, 64)).astype(np.float32)
    indices = np.random.randint(
        0, 32, size=128 * 4 * 4).reshape((128, 4, 4)).astype(np.int32)
    expected = array_ops.gather_v2(params, indices, batch_dims=1, axis=2)

    params = api.relayout(
        params,
        layout=Layout(
            [
                _MESH_DIM_X,
                layout_lib.UNSHARDED,
                layout_lib.UNSHARDED,
                _MESH_DIM_Y,
            ],
            self.mesh,
        ),
    )
    indices = api.relayout(
        indices,
        Layout(
            [layout_lib.UNSHARDED, layout_lib.UNSHARDED, layout_lib.UNSHARDED],
            self.mesh,
        ),
    )

    expected_layout = Layout([
        _MESH_DIM_X, layout_lib.UNSHARDED, layout_lib.UNSHARDED,
        layout_lib.UNSHARDED, _MESH_DIM_Y
    ], self.mesh)
    with api.default_mesh(self.mesh):
      dtensor_result = array_ops.gather_v2(
          params, indices, batch_dims=1, axis=2)
      self.assertDTensorEqual(expected, expected_layout, dtensor_result)

  def testGatherShardingParamsWithBatchDimAxisIs2ShardingIndices(self):
    params = np.arange(128 * 5 * 32 * 64).reshape(
        (128, 5, 32, 64)).astype(np.float32)
    indices = np.random.randint(
        0, 32, size=128 * 4 * 4).reshape((128, 4, 4)).astype(np.int32)
    expected = array_ops.gather_v2(params, indices, batch_dims=1, axis=2)

    params = api.relayout(
        params,
        layout=Layout(
            [
                _MESH_DIM_X,
                layout_lib.UNSHARDED,
                layout_lib.UNSHARDED,
                layout_lib.UNSHARDED,
            ],
            self.mesh,
        ),
    )
    indices = api.relayout(
        indices,
        Layout(
            [layout_lib.UNSHARDED, _MESH_DIM_Y, layout_lib.UNSHARDED], self.mesh
        ),
    )

    expected_layout = Layout([
        _MESH_DIM_X, layout_lib.UNSHARDED, _MESH_DIM_Y, layout_lib.UNSHARDED,
        layout_lib.UNSHARDED
    ], self.mesh)
    with api.default_mesh(self.mesh):
      dtensor_result = array_ops.gather_v2(
          params, indices, batch_dims=1, axis=2)
      self.assertDTensorEqual(expected, expected_layout, dtensor_result)

  def testGatherParamsShardingAfterAxisWithBatchDim(self):
    params = np.arange(128 * 1000 * 2).reshape(
        (128, 1000, 2)).astype(np.float32)
    indices = np.random.randint(
        0, 1000, size=128 * 4).reshape((128, 4)).astype(np.int32)
    expected = array_ops.gather_v2(params, indices, batch_dims=1, axis=1)

    params = api.relayout(
        params,
        layout=Layout(
            [layout_lib.UNSHARDED, layout_lib.UNSHARDED, _MESH_DIM_X], self.mesh
        ),
    )
    indices = api.relayout(
        indices, Layout([layout_lib.UNSHARDED, layout_lib.UNSHARDED], self.mesh)
    )

    expected_layout = Layout(
        [layout_lib.UNSHARDED, layout_lib.UNSHARDED, _MESH_DIM_X], self.mesh)
    with api.default_mesh(self.mesh):
      dtensor_result = array_ops.gather_v2(
          params, indices, batch_dims=1, axis=1)
      self.assertDTensorEqual(expected, expected_layout, dtensor_result)

  def testGatherShardingParamsIndicesWithBatchDim(self):
    params = np.arange(128 * 1000 * 2).reshape(
        (128, 1000, 2)).astype(np.float32)
    indices = np.random.randint(
        0, 1000, size=128 * 4 * 4).reshape((128, 4, 4)).astype(np.int32)
    expected = array_ops.gather_v2(params, indices, batch_dims=1, axis=1)

    params = api.relayout(
        params,
        layout=Layout(
            [_MESH_DIM_X, layout_lib.UNSHARDED, layout_lib.UNSHARDED], self.mesh
        ),
    )
    indices = api.relayout(
        indices,
        Layout(
            [_MESH_DIM_X, layout_lib.UNSHARDED, layout_lib.UNSHARDED], self.mesh
        ),
    )

    expected_layout = Layout([
        _MESH_DIM_X, layout_lib.UNSHARDED, layout_lib.UNSHARDED,
        layout_lib.UNSHARDED
    ], self.mesh)
    with api.default_mesh(self.mesh):
      dtensor_result = array_ops.gather_v2(
          params, indices, batch_dims=1, axis=1)
      self.assertDTensorEqual(expected, expected_layout, dtensor_result)

  def testTranspose(self):
    original_a = np.arange(5 * 4 * 6).reshape((5, 4, 6)).astype(np.float32)

    original_layout = Layout([layout_lib.UNSHARDED, _MESH_DIM_Y, _MESH_DIM_X],
                             self.mesh)

    # paris of (perm, expected_layout)
    combinations = [
        ([2, 0, 1], [_MESH_DIM_X, layout_lib.UNSHARDED, _MESH_DIM_Y]),
        ([2, 1, 0], [_MESH_DIM_X, _MESH_DIM_Y, layout_lib.UNSHARDED]),
        ([1, 2, 0], [_MESH_DIM_Y, _MESH_DIM_X, layout_lib.UNSHARDED]),
        ([1, 0, 2], [_MESH_DIM_Y, layout_lib.UNSHARDED, _MESH_DIM_X]),
        ([0, 1, 2], [layout_lib.UNSHARDED, _MESH_DIM_Y, _MESH_DIM_X]),
        ([0, 2, 1], [layout_lib.UNSHARDED, _MESH_DIM_X, _MESH_DIM_Y]),
    ]

    for (perm, expected_spec) in combinations:
      a = original_a
      expected = array_ops.transpose_v2(a, perm)

      a = api.relayout(a, original_layout)
      expected_layout = Layout(expected_spec, self.mesh)
      with api.default_mesh(self.mesh):
        dtensor_result = array_ops.transpose_v2(a, perm)
        self.assertDTensorEqual(expected, expected_layout, dtensor_result)

  def testSliceOpsWithNonFullSlicingOnShardedInputs(self):
    t = constant_op.constant([[1., 2., 3., 4.], [5., 6., 7., 8.]])
    expected_result = array_ops.slice(t, [0, 0], [1, 2])
    sharded_layout = Layout([_MESH_DIM_X, layout_lib.UNSHARDED], self.mesh)

    t = api.relayout(t, sharded_layout)
    expected_layout = Layout([layout_lib.UNSHARDED, layout_lib.UNSHARDED],
                             self.mesh)

    with api.default_mesh(self.mesh):
      dtensor_result = array_ops.slice(t, [0, 0], [1, 2])
      self.assertDTensorEqual(expected_result, expected_layout, dtensor_result)

  def testSliceOpWithNonFullSlicingOnShardedInputsAndFullSlicingOnAnother(self):
    t = constant_op.constant([[1., 2., 3., 4.], [5., 6., 7., 8.]])
    expected_result = array_ops.slice(t, [0, 0], [1, 4])
    sharded_layout = Layout([_MESH_DIM_X, _MESH_DIM_Y], self.mesh)

    t = api.relayout(t, sharded_layout)
    expected_layout = Layout([layout_lib.UNSHARDED, _MESH_DIM_Y], self.mesh)

    with api.default_mesh(self.mesh):
      dtensor_result = array_ops.slice(t, [0, 0], [1, 4])
      self.assertDTensorEqual(expected_result, expected_layout, dtensor_result)

  def testSliceOpsWithNonFullSlicingOnShardedInputsWithShardedOutputs(self):
    t = constant_op.constant([
        [1., 2., 3., 4.],
        [5., 6., 7., 8.],
        [11., 12., 13., 14.],
        [15., 16., 17., 18.],
    ])
    expected_result = array_ops.slice(t, [0, 0], [2, 2])
    sharded_layout = Layout([_MESH_DIM_X, layout_lib.UNSHARDED], self.mesh)

    t = api.relayout(t, sharded_layout)
    expected_layout = Layout([_MESH_DIM_X, layout_lib.UNSHARDED], self.mesh)

    with api.default_mesh(self.mesh):

      @polymorphic_function.function
      def op_fn(x):
        y = array_ops.slice(x, [0, 0], [2, 2])
        return api.relayout(y, expected_layout)

      dtensor_result = op_fn(t)
      self.assertDTensorEqual(expected_result, expected_layout, dtensor_result)

  def testSliceOpsWithFullyReplicatedInputsWithShardedOutputs(self):
    t = constant_op.constant([
        [1., 2., 3., 4.],
        [5., 6., 7., 8.],
        [11., 12., 13., 14.],
        [15., 16., 17., 18.],
    ])
    expected_result = array_ops.slice(t, [0, 0], [2, 2])
    operand_layout = Layout([layout_lib.UNSHARDED, layout_lib.UNSHARDED],
                            self.mesh)

    t = api.relayout(t, operand_layout)
    expected_layout = Layout([_MESH_DIM_X, layout_lib.UNSHARDED], self.mesh)

    with api.default_mesh(self.mesh):

      @polymorphic_function.function
      def op_fn(x):
        y = array_ops.slice(x, [0, 0], [2, 2])
        return api.relayout(y, expected_layout)

      dtensor_result = op_fn(t)
      self.assertDTensorEqual(expected_result, expected_layout, dtensor_result)

  @parameterized.named_parameters(
      test_util.product(
          [('_tensor_unsharded_updates_unsharded', -1, -1, 2),
           ('_tensor_first_updates_unsharded', 0, -1, 2),
           ('_tensor_second_updates_unsharded', 1, -1, 2),
           ('_tensor_unsharded_updates_first', -1, 0, 2),
           ('_tensor_first_updates_first', 0, 0, 2),
           ('_tensor_second_updates_first', 1, 0, 2),
           ('_tensor_unsharded_updates_second', -1, 1, 2),
           ('_tensor_first_updates_second', 0, 1, 2),
           ('_tensor_second_updates_second', 1, 1, 2),
           ('_tensor_second_updates_second_rank_three_indices', 1, 1, 3)], [(
               'update',
               gen_array_ops.tensor_scatter_update,
           ), (
               'add',
               gen_array_ops.tensor_scatter_add,
           )]))
  def testTensorScatterUpdate(self, tensor_dimension, updates_dimension,
                              indices_rank, op_type):
    tensor_layout = self.layouts[tensor_dimension][2]
    updates_layout = self.layouts[updates_dimension][indices_rank]

    # Tensor is shape [4, 2], indices is shape [..., 2, 1], updates is shape
    # [..., 2, 2]
    #
    # Entries in indices should be unique and integers in the range [0, 4).
    # Tensor and updates are float.

    tensor_numpy = np.random.uniform(size=[4, 2])
    padding_axes = [1] * (indices_rank - 2)
    updates_numpy = np.random.uniform(size=padding_axes + [2, 2])
    indices_numpy_flat = np.array(
        [np.random.randint(0, 4),
         np.random.randint(0, 3)])
    if indices_numpy_flat[0] == indices_numpy_flat[1]:
      indices_numpy_flat[1] += 1
    indices_numpy = indices_numpy_flat.reshape(padding_axes + [2, 1])

    tensor = constant_op.constant(tensor_numpy, dtype=dtypes.float32)
    updates = constant_op.constant(updates_numpy, dtype=dtypes.float32)
    indices = constant_op.constant(indices_numpy, dtype=dtypes.int32)

    golden_result = op_type(tensor=tensor, updates=updates, indices=indices)

    tensor = api.relayout(tensor, tensor_layout)
    updates = api.relayout(updates, updates_layout)
    indices = api.relayout(
        indices, Layout.replicated(tensor_layout.mesh, indices_rank)
    )

    dtensor_result = op_type(tensor=tensor, updates=updates, indices=indices)

    # If either of the inputs are sharded in the non-index dimension, then
    # the output is as well, otherwise it is replicated.
    if tensor_dimension == 1 or updates_dimension == 1:
      expected_layout = self.layouts[1][2]
    else:
      expected_layout = self.layouts[-1][2]

    self.assertDTensorEqual(golden_result, expected_layout, dtensor_result)

  @parameterized.named_parameters(
      ('_params_unsharded_indices_unsharded', -1, -1),
      ('_params_first_indices_unsharded', 0, -1),
      ('_params_third_indices_unsharded', 2, -1),
      ('_params_unsharded_indices_first', -1, 0),
      ('_params_first_indices_first', 0, 0),
      ('_params_third_indices_first', 2, 0),
      ('_params_unsharded_indices_second', -1, 1),
      ('_params_first_indices_second', 0, 1),
      ('_params_third_indices_second', 2, 1),
  )
  def testGatherNd(self, params_dimension, indices_dimension):
    self.skipForDeviceType(['GPU'], 'b/179387248 cases with AllConcat crash')
    params_layout = self.layouts[params_dimension][3]
    indices_layout = self.layouts[indices_dimension][2]

    # Params will have shape [6, 4, 4] and indices will have shape [2, 2]
    # this will result in a tensor with final shape [2, 4].

    params_numpy = np.random.uniform(size=[6, 4, 4])
    indices_numpy = np.array(
        [[np.random.randint(0, 6),
          np.random.randint(0, 4)],
         [np.random.randint(0, 6),
          np.random.randint(0, 4)]])

    params = constant_op.constant(params_numpy, dtype=dtypes.float32)
    indices = constant_op.constant(indices_numpy, dtype=dtypes.int32)

    golden_result = gen_array_ops.gather_nd(params=params, indices=indices)

    params = api.relayout(params, params_layout)
    indices = api.relayout(indices, indices_layout)

    dtensor_result = gen_array_ops.gather_nd(params=params, indices=indices)

    if params_dimension < 2 and indices_dimension == 0:
      # if params isn't sharded in the last dimension, then sharding of indices
      # the first dimension gives a first dimension sharding of the output
      expected_layout = self.layouts[0][2]
    elif params_dimension == 2:
      # if params is sharded in the last dimension, then sharding of indices is
      # ignored as they are both sharded on the same dimension.
      expected_layout = self.layouts[1][2]
    else:
      expected_layout = self.layouts[-1][2]

    self.assertDTensorEqual(golden_result, expected_layout, dtensor_result)

  def testGatherNdUnshardedInputShardedOutput(self):
    # Params will have shape [6, 4, 4] and indices will have shape [2, 2]
    # this will result in a tensor with final shape [2, 4].

    params_numpy = np.random.uniform(size=[6, 4, 4])
    indices_numpy = np.array(
        [[np.random.randint(0, 6),
          np.random.randint(0, 4)],
         [np.random.randint(0, 6),
          np.random.randint(0, 4)]])

    params = constant_op.constant(params_numpy, dtype=dtypes.float32)
    indices = constant_op.constant(indices_numpy, dtype=dtypes.int32)

    golden_result = gen_array_ops.gather_nd(params=params, indices=indices)

    params = api.relayout(params, self.replicated_layout_3d)
    indices = api.relayout(indices, self.replicated_layout_2d)

    @polymorphic_function.function
    def gather_with_relayout(params, indices):
      result = gen_array_ops.gather_nd(params=params, indices=indices)
      return api.relayout(result, self.first_dimension_sharded_layout_2d)

    dtensor_result = gather_with_relayout(params, indices)

    self.assertDTensorEqual(golden_result,
                            self.first_dimension_sharded_layout_2d,
                            dtensor_result)

  @parameterized.named_parameters(
      ('_unsharded_unsharded', [layout_lib.UNSHARDED, layout_lib.UNSHARDED]),
      ('_x_unsharded', [_MESH_DIM_X, layout_lib.UNSHARDED]),
      ('_x_y', [_MESH_DIM_X, _MESH_DIM_Y]))
  def testTopK(self, sharding):
    inputs_layout = Layout(sharding, self.mesh)

    inputs = constant_op.constant(
        np.random.uniform(size=[4, 16]), dtype=dtypes.float32)
    topk = constant_op.constant(2, dtype=dtypes.int32)

    expected_topk_values, expected_topk_indices = nn_ops.top_k(inputs, k=topk)

    inputs = api.relayout(inputs, inputs_layout)

    dtensor_topk_values, dtensor_topk_indices = nn_ops.top_k(inputs, k=topk)

    expected_sharding = [sharding[0], layout_lib.UNSHARDED]
    expected_layout = Layout(expected_sharding, self.mesh)

    self.assertDTensorEqual(expected_topk_values, expected_layout,
                            dtensor_topk_values)
    self.assertDTensorEqual(expected_topk_indices, expected_layout,
                            dtensor_topk_indices)

  @parameterized.named_parameters(*test_util.product(
      (('_targets_unsharded', [layout_lib.UNSHARDED]),
       ('_targets_x', [_MESH_DIM_X]), ('_targets_y', [_MESH_DIM_Y])),
      (('_predictions_unsharded_unsharded',
        [layout_lib.UNSHARDED, layout_lib.UNSHARDED]),
       ('_predictions_x_unsharded', [_MESH_DIM_X, layout_lib.UNSHARDED]),
       ('_predictions_unsharded_y', [layout_lib.UNSHARDED, _MESH_DIM_Y]),
       ('_predictions_x_y', [_MESH_DIM_X, _MESH_DIM_Y]))))
  def testInTopK(self, targets_sharding, predictions_sharding):

    # TODO(b/193471732): changed back to dtypes.int32 once it is fixed.
    int_dtype = dtypes.int64 if self.mesh.device_type(
    ) == 'GPU' else dtypes.int32

    targets_layout = Layout(targets_sharding, self.mesh)
    predictions_layout = Layout(predictions_sharding, self.mesh)

    targets = constant_op.constant([2, 2, 1, 0], dtype=int_dtype)
    predictions = constant_op.constant([[0.1, 0.3, 0.2, 0.4],
                                        [0.1, 0.2, 0.3, 0.4],
                                        [0.3, 0.4, 0.1, 0.2],
                                        [0.1, 0.3, 0.4, 0.2]])
    topk = constant_op.constant(2, dtype=int_dtype)

    expected_output = nn_ops.in_top_k_v2(targets, predictions, k=topk)

    targets = api.relayout(targets, targets_layout)
    predictions = api.relayout(predictions, predictions_layout)

    dtensor_output = nn_ops.in_top_k_v2(targets, predictions, k=topk)

    expected_sharding = [layout_lib.UNSHARDED]
    # Select the more sharded layout
    if targets_sharding[0] != layout_lib.UNSHARDED:
      expected_sharding[0] = targets_sharding[0]
    if predictions_sharding[0] != layout_lib.UNSHARDED:
      expected_sharding[0] = predictions_sharding[0]

    expected_layout = Layout(expected_sharding, self.mesh)
    self.assertDTensorEqual(expected_output, expected_layout, dtensor_output)


class DTensorRelayoutTest(test_util.DTensorBaseTest):

  def setUp(self):
    super(DTensorRelayoutTest, self).setUp()

    self.skipForDeviceType(['TPU'],
                           'all tests require 8 TPU cores.',
                           unless_device_count_equals_to=8)

    global_ids = test_util.create_device_ids_array((2, 4))
    local_ids = np.ravel(global_ids).tolist()
    mesh_dict = {
        device: Mesh([_MESH_DIM_X, _MESH_DIM_Y], global_ids, local_ids,
                     test_util.create_device_list((2, 4), device),
                     use_xla_spmd=test_util.get_use_xla_spmd(device))
        for device in ('CPU', 'GPU', 'TPU')
    }
    self.mesh = self.configTestMesh(mesh_dict)
    context.ensure_initialized()

  def testRelayoutEagerAllConcat(self):
    op = gen_nn_ops.relu

    a = constant_op.constant([[[1.], [-2.], [3.], [-4.]],
                              [[5.], [-6.], [-7.], [8.]]])
    assert a.shape == [2, 4, 1]
    expected_result = op(a)

    init_layout = Layout([_MESH_DIM_X, _MESH_DIM_Y, layout_lib.UNSHARDED],
                         self.mesh)
    a = api.relayout(a, init_layout)
    dtensor_output = op(a)

    final_layout = Layout(
        [_MESH_DIM_X, layout_lib.UNSHARDED, layout_lib.UNSHARDED], self.mesh)
    # eager relayout
    dtensor_result = api.relayout(dtensor_output, final_layout)
    self.assertDTensorEqual(expected_result, final_layout, dtensor_result)

  def testRelayoutEagerSlice(self):
    op = gen_nn_ops.relu

    a = constant_op.constant([[[1.], [-2.], [3.], [-4.]],
                              [[5.], [-6.], [-7.], [8.]]])
    assert a.shape == [2, 4, 1]
    expected_result = op(a)

    init_layout = Layout(
        [layout_lib.UNSHARDED, layout_lib.UNSHARDED, layout_lib.UNSHARDED],
        self.mesh)
    a = api.relayout(a, init_layout)
    dtensor_output = op(a)

    final_layout = Layout([_MESH_DIM_X, _MESH_DIM_Y, layout_lib.UNSHARDED],
                          self.mesh)
    # eager relayout
    dtensor_result = api.relayout(dtensor_output, final_layout)
    self.assertDTensorEqual(expected_result, final_layout, dtensor_result)

  def testRelayoutGraphAllConcat(self):
    op = gen_nn_ops.relu

    a = constant_op.constant([[[1.], [-2.], [3.], [-4.]],
                              [[5.], [-6.], [-7.], [8.]]])
    assert a.shape == [2, 4, 1]
    expected_result = op(a)

    init_layout = Layout([_MESH_DIM_X, _MESH_DIM_Y, layout_lib.UNSHARDED],
                         self.mesh)
    a = api.relayout(a, init_layout)

    final_layout = Layout(
        [_MESH_DIM_X, layout_lib.UNSHARDED, layout_lib.UNSHARDED], self.mesh)

    @polymorphic_function.function
    def wrap_fn(x):
      dtensor_output = op(x)
      return api.relayout(dtensor_output, final_layout)

    dtensor_result = wrap_fn(a)
    self.assertDTensorEqual(expected_result, final_layout, dtensor_result)

  def testRelayoutGraphSlice(self):
    op = gen_nn_ops.relu

    a = constant_op.constant([[[1.], [-2.], [3.], [-4.]],
                              [[5.], [-6.], [-7.], [8.]]])
    assert a.shape == [2, 4, 1]
    expected_result = op(a)

    init_layout = Layout(
        [_MESH_DIM_X, layout_lib.UNSHARDED, layout_lib.UNSHARDED], self.mesh)
    a = api.relayout(a, init_layout)

    final_layout = Layout([_MESH_DIM_X, _MESH_DIM_Y, layout_lib.UNSHARDED],
                          self.mesh)

    @polymorphic_function.function
    def wrap_fn(x):
      dtensor_output = op(x)
      return api.relayout(dtensor_output, final_layout)

    dtensor_result = wrap_fn(a)
    self.assertDTensorEqual(expected_result, final_layout, dtensor_result)


if __name__ == '__main__':
  tf_test.main()
