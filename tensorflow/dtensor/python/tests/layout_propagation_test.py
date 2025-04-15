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
"""Tests for layout propagation."""
from absl.testing import parameterized
import numpy as np

from tensorflow.dtensor.python import api
from tensorflow.dtensor.python import d_variable
from tensorflow.dtensor.python import layout
from tensorflow.dtensor.python import numpy_util
from tensorflow.dtensor.python.tests import test_util
from tensorflow.python.eager import backprop
from tensorflow.python.eager.polymorphic_function import polymorphic_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import stateless_random_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test

UNSHARDED = layout.UNSHARDED

# Convenient constants to use for tests.
_MESH_DIM_X = 'x'
_MESH_DIM_Y = 'y'


class LayoutPropagationV2Test(test_util.DTensorBaseTest):

  def setUp(self):
    super(LayoutPropagationV2Test, self).setUp()
    global_ids = test_util.create_device_ids_array((1, 2))
    local_ids = np.ravel(global_ids).tolist()
    mesh_dict = {  # pylint: disable=g-complex-comprehension
        device: layout.Mesh(
            [_MESH_DIM_X, _MESH_DIM_Y],
            global_ids,
            local_ids,
            test_util.create_device_list((1, 2), device),
        )
        for device in ('CPU', 'GPU', 'TPU')
    }
    self.mesh = self.configTestMesh(mesh_dict)
    # 1D Layouts
    self.unsharded_layout = layout.Layout.replicated(self.mesh, rank=1)
    self.x_layout = layout.Layout.batch_sharded(self.mesh, _MESH_DIM_X, rank=1)
    self.y_layout = layout.Layout.batch_sharded(self.mesh, _MESH_DIM_Y, rank=1)

    # 2D Layouts
    self.unsharded_unsharded_layout = layout.Layout.replicated(
        self.mesh, rank=2
    )
    self.x_unsharded_layout = layout.Layout.batch_sharded(
        self.mesh, _MESH_DIM_X, rank=2
    )
    self.unsharded_x_layout = layout.Layout.inner_sharded(
        self.mesh, _MESH_DIM_X, rank=2
    )
    self.unsharded_y_layout = layout.Layout.inner_sharded(
        self.mesh, _MESH_DIM_Y, rank=2
    )

  def test_layout_prop_v2_with_const_tf_function(self):
    a = constant_op.constant([[1.0, 2.0], [3.0, 4.0]])
    b = constant_op.constant([[10.0, 20.0], [30.0, 40.0]])
    golden_result = math_ops.add(a, b)

    c = api.copy_to_mesh(a, self.unsharded_unsharded_layout)

    @polymorphic_function.function
    def add_function():
      d = constant_op.constant([[10.0, 20.0], [30.0, 40.0]])
      return math_ops.add(c, d)

    dtensor_result = add_function()
    self.assertDTensorEqual(golden_result, self.unsharded_unsharded_layout,
                            dtensor_result)

  def test_layout_prop_v2_while(self):
    a = constant_op.constant([0, 1, 2, 1], dtype=dtypes.float32)
    num_iterations = 10

    @polymorphic_function.function
    def function_with_while(t):
      for _ in math_ops.range(num_iterations):
        random_number = stateless_random_ops.stateless_random_normal(
            shape=[4], seed=[1, 2], dtype=dtypes.float32
        )
        t = t + random_number
      return t

    golden_result = function_with_while(a)

    a = numpy_util.pack_numpy(a, self.unsharded_layout)

    dtensor_result = function_with_while(a)

    self.assertDTensorEqual(golden_result, self.unsharded_layout,
                            dtensor_result)

  @parameterized.named_parameters(
      dict(testcase_name='unsharded', sharded_layout=0, use_split=False),
      dict(testcase_name='x_sharded', sharded_layout=1, use_split=False),
      dict(testcase_name='unsharded_split', sharded_layout=0, use_split=True),
      dict(testcase_name='x_sharded_split', sharded_layout=1, use_split=True))
  def test_while_microbatch(self, sharded_layout, use_split):

    layouts = [self.unsharded_unsharded_layout, self.x_unsharded_layout]
    sharded_layout = layouts[sharded_layout]

    np.random.seed(0)
    random_initial_value = np.random.uniform(size=4 * 4).reshape([4, 4])
    if use_split:
      random_batch = np.random.uniform(size=12 * 4).reshape([12, 4])
    else:
      random_batch = np.random.uniform(size=4 * 4).reshape([4, 4])
    golden_variable = variables.Variable(random_initial_value)

    @polymorphic_function.function
    def update_weights(batch, variable):
      accum_grads = array_ops.zeros_like_v2(variable)
      for i in math_ops.range(3):
        if use_split:
          reshaped = array_ops.reshape(batch, [4, 3, 4])
          mini_batch = array_ops.gather_v2(reshaped, i, axis=1)
        else:
          mini_batch = batch
        with backprop.GradientTape() as tape:
          logits = variable * variable + mini_batch
          loss = math_ops.reduce_sum(logits * logits)
        accum_grads += tape.gradient(loss, variable)
      new_variable = variable + accum_grads
      variable.assign(new_variable)
      return accum_grads

    golden_accum = update_weights(
        constant_op.constant(random_batch), golden_variable
    )

    random_batch = numpy_util.pack_numpy(random_batch, sharded_layout)
    random_initial_value = numpy_util.pack_numpy(random_initial_value,
                                                 sharded_layout)
    dtensor_variable = d_variable.DVariable(random_initial_value)

    dtensor_accum = update_weights(random_batch, dtensor_variable)
    self.assertDTensorEqual(golden_accum, sharded_layout, dtensor_accum)

  @parameterized.named_parameters(
      dict(testcase_name='unsharded_split', sharded_layout=0),
      dict(testcase_name='x_sharded_split', sharded_layout=1))
  def test_while_microbatch_with_reused_gradient_accumulator(
      self, sharded_layout):
    layouts = [
        self.unsharded_unsharded_layout, self.x_unsharded_layout,
        self.unsharded_x_layout
    ]
    sharded_layout_0 = layouts[sharded_layout]
    sharded_layout_1 = layouts[2]

    np.random.seed(0)
    random_initial_value_1 = np.random.uniform(size=4 * 4).reshape(
        [4, 4]).astype(np.float32)
    random_initial_value_2 = np.random.uniform(size=4 * 4).reshape(
        [4, 4]).astype(np.float32)
    random_batch = np.random.uniform(size=12 * 4).reshape([12, 4
                                                          ]).astype(np.float32)

    golden_variable_1 = variables.Variable(random_initial_value_1)
    golden_variable_2 = variables.Variable(random_initial_value_2)

    @polymorphic_function.function
    def update_weights(batch, variable1, variable2):
      accum_grads = array_ops.zeros_like_v2(variable1)
      accum_grads_2 = array_ops.zeros_like_v2(variable2)

      for i in math_ops.range(3):
        reshaped = array_ops.reshape(batch, [4, 3, 4])
        mini_batch = array_ops.gather_v2(reshaped, i, axis=1)

        with backprop.GradientTape() as tape:
          logits_1 = variable1 * variable1 + mini_batch
          logits_2 = variable2 * variable2 + mini_batch
          loss_1 = math_ops.reduce_sum(logits_1 * logits_1)
          loss_2 = math_ops.reduce_sum(logits_2 * logits_2)
          loss = loss_1 + loss_2
        grads = tape.gradient(loss, [variable1, variable2])
        accum_grads += grads[0]
        accum_grads_2 += grads[1]

      new_variable = variable1 + accum_grads
      new_variable_2 = variable2 + accum_grads_2
      variable1.assign(new_variable)
      variable2.assign(new_variable_2)
      return accum_grads, accum_grads_2

    golden_accum, golden_accum_2 = update_weights(
        constant_op.constant(random_batch), golden_variable_1, golden_variable_2
    )

    random_batch = numpy_util.pack_numpy(random_batch, sharded_layout_0)
    random_initial_value_1 = numpy_util.pack_numpy(random_initial_value_1,
                                                   sharded_layout_0)
    dtensor_variable_1 = d_variable.DVariable(random_initial_value_1)

    random_initial_value_2 = numpy_util.pack_numpy(random_initial_value_2,
                                                   sharded_layout_1)
    dtensor_variable_2 = d_variable.DVariable(random_initial_value_2)

    dtensor_accum, dtensor_accum_2 = update_weights(random_batch,
                                                    dtensor_variable_1,
                                                    dtensor_variable_2)
    self.assertDTensorEqual(golden_accum, sharded_layout_0, dtensor_accum)
    self.assertDTensorEqual(golden_accum_2, sharded_layout_1, dtensor_accum_2)

  def test_layout_like(self):
    a = constant_op.constant([1, 2, 3, 4], dtype=dtypes.float32)
    a = numpy_util.pack_numpy(a, self.unsharded_layout)

    b = constant_op.constant([0, 1, 2, 3], dtype=dtypes.int64)
    b = numpy_util.pack_numpy(b, self.x_layout)

    @polymorphic_function.function
    def layout_like_function(i, t):
      i = math_ops.sqrt(i)
      return api.relayout_like(i, t)

    # Triggers relayout_like with different dtype.
    dtensor_result = layout_like_function(a, b)
    api.check_layout(dtensor_result, self.x_layout)

  def test_layout_prop_v2_if(self):
    a = constant_op.constant([0, 1, 2, 1], dtype=dtypes.float32)
    a = numpy_util.pack_numpy(a, self.unsharded_layout)

    @polymorphic_function.function
    def function_with_if(t):
      if math_ops.equal(math_ops.reduce_sum(t), 0):
        t = math_ops.sqrt(t)
        return api.relayout(t, self.x_layout)
      else:
        return array_ops.zeros_like_v2(t)

    dtensor_result = function_with_if(a)
    api.check_layout(dtensor_result, self.x_layout)

  def test_layout_prop_v2_if_with_different_layouts_for_branches(self):
    unsharded_unsharded = layout.Layout.replicated(self.mesh, rank=2)
    unsharded_y = layout.Layout.inner_sharded(self.mesh, _MESH_DIM_Y, rank=2)
    x_unsharded = layout.Layout.batch_sharded(self.mesh, _MESH_DIM_X, rank=2)
    a = np.random.uniform(size=16).reshape([4, 4])
    a = numpy_util.pack_numpy(a, unsharded_unsharded)

    @polymorphic_function.function
    def function_with_if(t):
      if math_ops.equal(math_ops.reduce_sum(t), 0):
        t = math_ops.sqrt(t)
        return api.relayout(t, unsharded_y)
      else:
        t = array_ops.zeros_like_v2(a)
        return api.relayout(t, x_unsharded)

    dtensor_result = function_with_if(a)

    x_y_sharded = layout.Layout([_MESH_DIM_X, _MESH_DIM_Y], self.mesh)
    api.check_layout(dtensor_result, x_y_sharded)

  def test_partial_relayout_in_function(self):
    sharded_layout = layout.Layout([_MESH_DIM_X, _MESH_DIM_Y], self.mesh)

    a = np.random.uniform(size=16).reshape([4, 4])
    a = numpy_util.pack_numpy(a, sharded_layout)
    replicated_layout = layout.Layout(
        [layout.MATCH, layout.UNSHARDED], mesh=self.mesh
    )

    @polymorphic_function.function
    def func_with_relayout(t):
      out = math_ops.cast(t, dtypes.float32)
      out = math_ops.sqrt(out)
      return api.relayout(out, replicated_layout)

    out = func_with_relayout(a)
    expected_layout = layout.Layout([_MESH_DIM_X, layout.UNSHARDED], self.mesh)
    api.check_layout(out, expected_layout)

  def test_partial_relayout_in_eager(self):

    sharded_layout = layout.Layout([_MESH_DIM_X, _MESH_DIM_Y], self.mesh)

    a = np.random.uniform(size=16).reshape([4, 4])
    a = numpy_util.pack_numpy(a, sharded_layout)
    replicated_layout = layout.Layout(
        [layout.MATCH, layout.UNSHARDED], mesh=self.mesh
    )

    a = math_ops.cast(a, dtypes.float32)
    a = math_ops.sqrt(a)
    out = api.relayout(a, replicated_layout)

    expected_layout = layout.Layout([_MESH_DIM_X, layout.UNSHARDED], self.mesh)
    api.check_layout(out, expected_layout)

  @parameterized.named_parameters(
      dict(testcase_name='unsharded_unsharded', sharded_layout=0),
      dict(testcase_name='x_unsharded', sharded_layout=1),
      dict(testcase_name='unsharded_y', sharded_layout=2),
  )
  def test_relayout_equivalent_layouts(self, sharded_layout):
    layouts = [
        self.unsharded_unsharded_layout, self.x_unsharded_layout,
        self.unsharded_y_layout]
    expected_layout = layouts[sharded_layout]
    inputs = constant_op.constant([[0, 1], [2, 1]], dtype=dtypes.float32)
    sharded_layout = layout.Layout([_MESH_DIM_X, _MESH_DIM_Y], self.mesh)
    inputs = numpy_util.pack_numpy(inputs, sharded_layout)
    output = api.relayout(inputs, expected_layout)
    api.check_layout(output, expected_layout)

  def test_strided_slice_grad(self):

    np.random.seed(0)
    random_initial_value = np.random.uniform(size=4).reshape([4])
    random_initial_value = numpy_util.pack_numpy(random_initial_value,
                                                 self.unsharded_layout)

    @polymorphic_function.function
    def fn_with_strided_slice(t):
      a = array_ops.strided_slice(t, [1], [2], shrink_axis_mask=1)
      return math_ops.sqrt(a)

    random_variable = d_variable.DVariable(random_initial_value)
    with backprop.GradientTape() as tape:
      output = fn_with_strided_slice(random_variable)

    grads = tape.gradient(output, [random_variable])
    self.assertTrue(api.fetch_layout(grads[0]).is_fully_replicated())

  def test_layout_prop_v2_infinite_loop(self):
    x_unsharded = layout.Layout([_MESH_DIM_X, layout.UNSHARDED], self.mesh)
    unsharded_x = layout.Layout([layout.UNSHARDED, _MESH_DIM_X], self.mesh)

    @polymorphic_function.function
    def func(input_a, input_b):
      out = array_ops.identity(
          math_ops.matmul(input_a, array_ops.identity(input_b))
      )
      return api.relayout(out, unsharded_x)

    result = func(
        api.call_with_layout(
            array_ops.ones, shape=(16, 16), layout=x_unsharded
        ),
        api.call_with_layout(
            array_ops.ones, shape=(16, 16), layout=unsharded_x
        ),
    )
    api.check_layout(result, unsharded_x)

  def test_layout_prop_parted_layout(self):
    value = np.array([1.0, 2.0, 3.0, 4.0])
    expected = nn_ops.softmax_v2(gen_nn_ops.relu(value))

    @polymorphic_function.function
    def func(value):
      return nn_ops.softmax_v2(gen_nn_ops.relu(value))

    parted_layout = self.x_layout.to_parted()
    result = func(api.relayout(value, parted_layout))

    # Verifies the parted layout can be propagated through a chain of ops to the
    # final output.
    self.assertDTensorEqual(expected, parted_layout, result)


if __name__ == '__main__':
  test.main()
