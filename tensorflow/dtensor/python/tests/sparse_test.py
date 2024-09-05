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

from absl.testing import parameterized
import numpy as np

from tensorflow.dtensor.python import layout as layout_lib
from tensorflow.dtensor.python import numpy_util
from tensorflow.dtensor.python.tests import test_util
from tensorflow.python.eager.polymorphic_function import polymorphic_function
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


# Convenient constants to use for tests.
_BATCH_DIM = "batch"
_MESH_DIM_X = "x"

# Shorter notation
Layout = layout_lib.Layout
Mesh = layout_lib.Mesh


class DTensorSPMDTest(test_util.DTensorBaseTest):

  def setUp(self):
    super().setUp()

    self.skipForDeviceType(["GPU", "TPU"],
                           "SparseTensors only supported on CPU.")

    global_ids = test_util.create_device_ids_array((2, 2))
    local_ids = np.ravel(global_ids).tolist()
    mesh_dict = {
        device: Mesh(
            [_BATCH_DIM, _MESH_DIM_X],
            global_ids,
            local_ids,
            test_util.create_device_list((2, 2), device),
        )
        for device in ("CPU", "GPU", "TPU")
    }
    self.mesh = self.configTestMesh(mesh_dict)

  @parameterized.parameters(
      [dtypes.int32, dtypes.int64, dtypes.float32, dtypes.float64]
  )
  def testIdentityOpWithSparseTensorInputSimple(self, dtype):
    inputs = array_ops.ones([6, 4], dtype=dtype)
    layout = Layout.batch_sharded(self.mesh, _BATCH_DIM, rank=2)

    @polymorphic_function.function
    def f(x):
      return array_ops.identity(x)

    self.assertDTensorEqual(
        inputs, layout,
        f(numpy_util.pack_numpy(inputs, layout, make_sparse=True)))

  @parameterized.product(
      dtype=[dtypes.int32, dtypes.int64, dtypes.float32, dtypes.float64],
      is_sparse_a=[True, False],
      is_sparse_b=[True, False],
  )
  def testIdentityOpWithSparseTensorInputComplex(self, dtype, is_sparse_a,
                                                 is_sparse_b):
    inputs_a = array_ops.ones([2, 1], dtype=dtype)
    inputs_b = array_ops.ones([32, 16], dtype=dtype)

    layout_a = Layout.batch_sharded(self.mesh, _BATCH_DIM, rank=2)
    layout_b = Layout.replicated(self.mesh, rank=2)

    @polymorphic_function.function
    def f(x, y):
      return array_ops.identity(x), array_ops.identity(y)

    got_a, got_b = f(
        numpy_util.pack_numpy(inputs_a, layout_a, make_sparse=is_sparse_a),
        numpy_util.pack_numpy(inputs_b, layout_b, make_sparse=is_sparse_b))

    self.assertDTensorEqual(inputs_a, layout_a, got_a)
    self.assertDTensorEqual(inputs_b, layout_b, got_b)

  def testMultipleIdentityOpFromOneSparseTensor(self):
    inputs_a = array_ops.ones([2, 1])
    layout_a = Layout.batch_sharded(self.mesh, _BATCH_DIM, rank=2)

    @polymorphic_function.function
    def f(x):
      return array_ops.identity(x), array_ops.identity(x)

    got_a, got_b = f(
        numpy_util.pack_numpy(inputs_a, layout_a, make_sparse=True))

    self.assertDTensorEqual(inputs_a, layout_a, got_a)
    self.assertDTensorEqual(inputs_a, layout_a, got_b)

  @parameterized.product(
      is_sparse_a=[True, False],
      is_sparse_b=[True, False],
      shard_type=["Replicated", "Sharded"])
  def testSparseTensorDenseMatMul(self, is_sparse_a, is_sparse_b, shard_type):
    inputs_a = array_ops.ones([16, 16])
    inputs_b = array_ops.ones([16, 16])

    if shard_type == "Replicated":
      layout_a = Layout.replicated(self.mesh, rank=2)
      layout_b = Layout.replicated(self.mesh, rank=2)
    else:
      layout_a = Layout([_MESH_DIM_X, _BATCH_DIM], self.mesh)
      layout_b = Layout(["unsharded", _MESH_DIM_X], self.mesh)

    expected = math_ops.matmul(inputs_a, inputs_b)

    @polymorphic_function.function
    def f(x, y):
      return math_ops.matmul(x, y)

    got = f(
        numpy_util.pack_numpy(inputs_a, layout_a, make_sparse=is_sparse_a),
        numpy_util.pack_numpy(inputs_b, layout_b, make_sparse=is_sparse_b))

    self.assertDTensorEqual(expected, Layout.replicated(self.mesh, rank=2), got)


if __name__ == "__main__":
  test.main()
