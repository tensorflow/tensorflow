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
"""Tests for DTensor support of Variables."""
import numpy as np

from tensorflow.dtensor.python import api
from tensorflow.dtensor.python import d_variable
from tensorflow.dtensor.python import layout as layout_lib
from tensorflow.dtensor.python.tests import test_util
from tensorflow.python.eager.polymorphic_function import polymorphic_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


# Makes a 1D mesh with dimension X(2).
_MESH_DIM_X = 'x'
_DEVICE_IDS = test_util.create_device_ids_array((2,))
_ONE_D_CPU_MESH = layout_lib.Mesh(
    [_MESH_DIM_X],
    _DEVICE_IDS,
    np.ravel(_DEVICE_IDS).tolist(),
    test_util.create_device_list((2,), 'CPU'),
)
_ONE_D_TPU_MESH = layout_lib.Mesh(
    [_MESH_DIM_X],
    _DEVICE_IDS,
    np.ravel(_DEVICE_IDS).tolist(),
    test_util.create_device_list((2,), 'TPU'),
)
_ONE_D_GPU_MESH = layout_lib.Mesh(
    [_MESH_DIM_X],
    _DEVICE_IDS,
    np.ravel(_DEVICE_IDS).tolist(),
    test_util.create_device_list((2,), 'GPU'),
)

Layout = layout_lib.Layout
UNSHARDED = layout_lib.UNSHARDED
DVariable = d_variable.DVariable


class Var(object):

  def __init__(self):
    self.v = None


class DTensorVariableTest(test_util.DTensorBaseTest):

  def setUp(self):
    super(DTensorVariableTest, self).setUp()
    mesh_dict = {
        'CPU': _ONE_D_CPU_MESH,
        'GPU': _ONE_D_GPU_MESH,
        'TPU': _ONE_D_TPU_MESH,
    }
    self.mesh = self.configTestMesh(mesh_dict)
    self._replicated_layout = Layout([UNSHARDED, UNSHARDED], self.mesh)
    self._one_d_replicated_layout = Layout([UNSHARDED], self.mesh)
    self._scalar_replicated_layout = Layout([], self.mesh)
    self._one_d_shard_layout = Layout([_MESH_DIM_X], self.mesh)
    self._first_d_shard_layout = Layout([_MESH_DIM_X, UNSHARDED], self.mesh)

  def testNonDtensorVariable(self):
    non_dtensor_variable = variables.Variable(1.0)
    with ops.device_v2(api.device_name()):
      with self.assertRaisesRegex(
          errors_impl.InvalidArgumentError,
          'No default mesh has been registered to DTensor',
      ):
        non_dtensor_variable.read_value()

  def testDVariableNoLayout(self):
    with self.assertRaisesRegex(
        errors_impl.InvalidArgumentError,
        'Neither layout nor DTensor initial value are provided.',
    ):
      DVariable(1.0)

  def testDVariableConflictingLayout(self):
    a = api.relayout([1, 2, 3, 4], self._one_d_replicated_layout)
    with self.assertRaisesRegex(
        errors_impl.InvalidArgumentError, 'Conflicting layout are provided'
    ):
      DVariable(a, layout=self._one_d_shard_layout)

  def testVariable(self):
    with ops.device_v2(api.device_name()):
      initial_value = api.relayout([1.0], self._one_d_replicated_layout)
      v = variables.Variable(initial_value)
      v = api.relayout(v, self._one_d_replicated_layout)
      api.check_layout(v, self._one_d_replicated_layout)

  def testVariableWithInitialValue(self):
    a = constant_op.constant([1.0])
    a = api.relayout(a, self._one_d_replicated_layout)
    with ops.device_v2(api.device_name()):
      v = variables.Variable(initial_value=a)
      api.check_layout(v, self._one_d_replicated_layout)
      to_add = api.relayout([1.0], self._one_d_replicated_layout)
      v = v.assign_add(to_add)
      api.check_layout(v, self._one_d_replicated_layout)

  def testVarAssignmentOpByOp(self):
    v = constant_op.constant(1.0)
    v = api.relayout(v, Layout.replicated(self.mesh, rank=0))
    w = d_variable.DVariable(v)
    api.check_layout(w, Layout.replicated(self.mesh, rank=0))
    self.assertEqual(w.numpy(), 1.0)
    w.assign_add(v)
    api.check_layout(w, Layout.replicated(self.mesh, rank=0))
    self.assertEqual(w.numpy(), 2.0)

  def testVarInitOutsideTfFunction(self):
    v = constant_op.constant(1.0)
    v = api.relayout(v, Layout.replicated(self.mesh, rank=0))

    w = d_variable.DVariable(v)

    @polymorphic_function.function()
    def assign_var(x):
      w.assign(x * 2)
      return w + x

    out = assign_var(constant_op.constant(1.0))
    api.check_layout(w, Layout.replicated(self.mesh, rank=0))
    self.assertEqual(w.numpy(), 2.0)
    self.assertEqual(out.numpy(), 3.0)

  def testDVariableInitFromValues(self):
    # Python value 1 will be converted as dtypes.int32 without a dtype.
    non_dtensor_variable = variables.Variable(1, dtype=dtypes.int64)
    with ops.device_v2(api.device_name()):
      with self.assertRaisesRegex(
          errors_impl.InvalidArgumentError,
          'No default mesh has been registered to DTensor',
      ):
        non_dtensor_variable.read_value()

    dtensor_variable = DVariable(
        api.relayout(
            constant_op.constant(1, dtype=dtypes.int64),
            Layout.replicated(self.mesh, rank=0),
        )
    )

    with ops.device_v2(api.device_name()):
      dtensor_variable = DVariable(
          api.relayout(
              constant_op.constant(1, dtype=dtypes.int64),
              Layout.replicated(self.mesh, rank=0),
          )
      )
    self.assertEqual(dtensor_variable.numpy(), 1)

  def testCreateVarInsideFunctionWithInitScope(self):
    var = Var()

    @polymorphic_function.function
    def assign_add():
      with ops.init_scope():
        if var.v is None:
          c = constant_op.constant(1.0)
          c = api.relayout(c, Layout.replicated(self.mesh, rank=0))
          var.v = variables.Variable(c)
      var.v.assign_add(1.0)

    with api._dtensor_device()._default_layout(
        Layout.replicated(self.mesh, rank=0)):
      assign_add()
      output = var.v.read_value()
      api.check_layout(output, Layout.replicated(self.mesh, rank=0))
      self.assertAllEqual(output, 2.)

  def testBufferAliasingOnDF(self):
    # TODO(b/239471086): re-enable once b/239471086 is fixed
    self.skipTest('Disabled due to b/239471086')
    self.skipForDeviceType(['GPU', 'CPU'], 'Test only applies to DF TPU')

    @polymorphic_function.function
    def add_var(v):
      new_v = array_ops_stack.stack([v, v])
      v.assign(math_ops.reduce_sum(new_v, axis=0))
      # Note that this only works with DF. When updated to PF, we need to
      # adjust tensor size accordingly.
      #
      # Without aliasing, the returned tensor is 3.5G * 4 = 14G, plus the
      # reserved 500MB and 3.5G arguments, this exceeds the 16G memory.
      # With aliasing, the 3.5G arguments will be aliased so it lowers the
      # memory pressure to 18-3.5 = 14.5G, which barely fits the memory.
      return v, new_v

    # Around 3.5G tensor.
    v = DVariable(
        initial_value=api.relayout(
            array_ops.ones((7, 512, 1024, 256), dtype=dtypes.float32),
            Layout.replicated(self.mesh, rank=4),
        )
    )

    add_var(v)

    self.assertEqual(api.fetch_layout(v), Layout.replicated(self.mesh, rank=4))


if __name__ == '__main__':
  test.main()
