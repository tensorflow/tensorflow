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
"""Tests for DTensor support in array_ops."""

import numpy as np

from tensorflow.dtensor.python import api
from tensorflow.dtensor.python import layout as layout_lib
from tensorflow.dtensor.python.tests import test_util
from tensorflow.python.eager.polymorphic_function import polymorphic_function
from tensorflow.python.framework import combinations
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test

Layout = layout_lib.Layout

_MESH_DIM_X = 'x'
_MESH_DIM_Y = 'y'
_MESH_DIMS = [_MESH_DIM_X, _MESH_DIM_Y]


class ArrayOpsTest(test_util.DTensorBaseTest):

  def setUp(self):
    super().setUp()

    global_ids = test_util.create_device_ids_array((2, 1))
    local_ids = np.ravel(global_ids).tolist()
    mesh_dict = {  # pylint: disable=g-complex-comprehension
        device: layout_lib.Mesh(
            _MESH_DIMS,
            global_ids,
            local_ids,
            test_util.create_device_list((2, 1), device),
        )
        for device in ('CPU', 'GPU', 'TPU')
    }
    self.mesh = self.configTestMesh(mesh_dict)

  @combinations.generate(
      combinations.combine(is_graph=[False, True], size=[32, 4096])
  )
  def testTwoFills(self, is_graph, size):
    layout_x = Layout.batch_sharded(self.mesh, _MESH_DIM_X, rank=1)
    layout_y = Layout.batch_sharded(self.mesh, _MESH_DIM_Y, rank=1)

    def fn():
      return (
          array_ops.fill([size], 0.0, layout=layout_x),
          array_ops.fill([size], 0.0, layout=layout_y),
      )

    if is_graph:
      fn = polymorphic_function.function(fn)

    with api.default_mesh(self.mesh):
      dtensor_x, dtensor_y = fn()
    tensor = array_ops.zeros([size], layout=None)

    self.assertDTensorEqual(tensor, layout_x, dtensor_x)
    self.assertDTensorEqual(tensor, layout_y, dtensor_y)

  @combinations.generate(
      combinations.combine(
          is_graph=[False, True],
          size=[32, 4096],
          nullary_op=[array_ops.zeros, array_ops.ones],
      )
  )
  def testNullaryOp(self, is_graph, size, nullary_op):
    layout_y = Layout.batch_sharded(self.mesh, _MESH_DIM_Y, rank=1)

    tensor = nullary_op([size], layout=None)

    def fn():
      return nullary_op([size], layout=layout_y)

    if is_graph:
      fn = polymorphic_function.function(fn)

    with api.default_mesh(self.mesh):
      dtensor = fn()

    self.assertDTensorEqual(tensor, layout_y, dtensor)

  @combinations.generate(
      combinations.combine(
          is_graph=[False, True],
          size=[32, 4096],
          nullary_op=[array_ops.zeros_like_v2, array_ops.ones_like_v2],
      )
  )
  def testNullaryLikeOpWithLayout(self, is_graph, size, nullary_op):
    layout_x = Layout.batch_sharded(self.mesh, batch_dim=_MESH_DIM_X, rank=1)
    layout_y = Layout.batch_sharded(self.mesh, batch_dim=_MESH_DIM_Y, rank=1)

    tensor = array_ops.zeros([size], layout=None)
    tensor_like = nullary_op(tensor, layout=None)
    dtensor = array_ops.zeros([size], layout=layout_x)
    self.assertDTensorEqual(tensor, layout_x, dtensor)

    def fn(layout):
      return nullary_op(dtensor, layout=layout)

    if is_graph:
      fn = polymorphic_function.function(fn)

    with api.default_mesh(self.mesh):
      dtensor_like = fn(layout_y)

    self.assertDTensorEqual(tensor_like, layout_y, dtensor_like)

  @combinations.generate(
      combinations.combine(
          is_graph=[True],
          size=[32, 4096],
          nullary_op=[array_ops.zeros_like_v2, array_ops.ones_like_v2],
      )
  )
  def testNullaryLikeOpWithoutLayoutEager(self, is_graph, size, nullary_op):
    layout_x = Layout.batch_sharded(self.mesh, batch_dim=_MESH_DIM_X, rank=1)
    layout_replicated = Layout.replicated(self.mesh, rank=1)

    tensor = array_ops.zeros([size], layout=None)
    tensor_like = nullary_op(tensor, layout=None)
    dtensor = array_ops.zeros([size], layout=layout_x)
    self.assertDTensorEqual(tensor, layout_x, dtensor)

    def fn(layout):
      return nullary_op(dtensor, layout=layout)

    if is_graph:
      fn = polymorphic_function.function(fn)

    with api.default_mesh(self.mesh):
      dtensor_like = fn(None)

    self.assertDTensorEqual(tensor_like, layout_replicated, dtensor_like)

  @combinations.generate(
      combinations.combine(
          is_graph=[False],
          size=[32, 4096],
          nullary_op=[array_ops.zeros_like_v2, array_ops.ones_like_v2],
      )
  )
  def testNullaryLikeOpWithoutLayoutGraph(self, is_graph, size, nullary_op):
    layout_x = Layout.batch_sharded(self.mesh, batch_dim=_MESH_DIM_X, rank=1)

    tensor = array_ops.zeros([size], layout=None)
    tensor_like = nullary_op(tensor, layout=None)
    dtensor = array_ops.zeros([size], layout=layout_x)
    self.assertDTensorEqual(tensor, layout_x, dtensor)

    def fn(layout):
      return nullary_op(dtensor, layout=layout)

    if is_graph:
      fn = polymorphic_function.function(fn)

    with api.default_mesh(self.mesh):
      dtensor_like = fn(None)

    self.assertDTensorEqual(tensor_like, layout_x, dtensor_like)


if __name__ == '__main__':
  test.main()
