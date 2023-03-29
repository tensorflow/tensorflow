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
"""Tests for DTensorDevice in python."""
from absl.testing import parameterized

import numpy as np

from tensorflow.dtensor.python import api
from tensorflow.dtensor.python import d_variable
from tensorflow.dtensor.python import dtensor_device
from tensorflow.dtensor.python import layout as layout_lib
from tensorflow.dtensor.python.tests import test_util
from tensorflow.python.eager.polymorphic_function import polymorphic_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import stateless_random_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test

# Convenient constants to use for tests.
_BATCH_DIM = "batch"
_MESH_DIM_X = "x"

# Shorter notation
Layout = layout_lib.Layout
Mesh = layout_lib.Mesh
UNSHARDED = layout_lib.UNSHARDED


class DTensorDeviceTest(test_util.DTensorBaseTest, parameterized.TestCase):

  def setUp(self):
    super(DTensorDeviceTest, self).setUp()
    device_ids = test_util.create_device_ids_array((2,))
    local_device_ids = np.ravel(device_ids).tolist()
    mesh_dict = {  # pylint: disable=g-complex-comprehension
        device: Mesh(
            [_BATCH_DIM],
            device_ids,
            local_device_ids,
            test_util.create_device_list((2,), device),
        )
        for device in ("CPU", "GPU", "TPU")
    }
    self.mesh = self.configTestMesh(mesh_dict)

  def testInvalidLayout(self):
    a = api.copy_to_mesh(
        constant_op.constant([1.0]), Layout.replicated(self.mesh, rank=1)
    )
    b = array_ops.identity(a)
    with self.assertRaises(ValueError):
      api.check_layout(b, Layout.batch_sharded(self.mesh, _BATCH_DIM, rank=1))

  @parameterized.parameters(True, False)
  def testAsyncOption(self, is_async):
    # There isn't a great way to test whether something actually executed
    # synchronously; this test just exercises the option.
    device = dtensor_device.DTensorDevice([], is_async=is_async)
    a = device.copy_to_mesh(
        constant_op.constant([1.0]), Layout.replicated(self.mesh, rank=1)
    )
    b = array_ops.identity(a)
    self.assertEqual([1.], b.numpy())

  def testBasicTypeBasedDispatch(self):
    # Tests for b = Op(a).
    a = constant_op.constant([1.0, 2.0, 3.0, 4.0])
    a = api.copy_to_mesh(a, Layout.replicated(self.mesh, rank=1))

    # __getitem__
    b = a[2:-2]
    api.check_layout(b, Layout.replicated(self.mesh, rank=1))

    c = a * 2
    api.check_layout(b, Layout.replicated(self.mesh, rank=1))

    self.assertAllEqual(a.numpy(), [1., 2., 3., 4.])
    self.assertAllEqual(c.numpy(), [2., 4., 6., 8.])

  def testNoImplicitCopyOnForLargeIntegerTensors(self):
    a = array_ops.ones([10, 10], dtype=dtypes.int32)
    a = api.copy_to_mesh(a, Layout.replicated(self.mesh, rank=2))
    big = array_ops.ones([10, 10], dtype=dtypes.int32)
    small = array_ops.ones([10], dtype=dtypes.int32)
    with self.assertRaises(errors_impl.UnimplementedError):
      a + big  # pylint:disable=pointless-statement
    a + small  # pylint:disable=pointless-statement

  def testNoImplicitCopyOnForScalarVariableOnNonCPUMesh(self):
    self.skipForTfrt("b/235088250")
    self.skipForDeviceType(["CPU"], "CPU mesh implicit copy is allowed.")
    init_value = api.call_with_layout(
        array_ops.ones, shape=(1), layout=Layout.replicated(self.mesh, rank=1)
    )
    with self.assertRaisesRegex(
        errors_impl.InvalidArgumentError,
        r"Using a non-DTensor variable with DTensor is only supported for..*\n"
        r".*Shape: \[1\].*\n"
        ".*device_test.py.*",
    ):
      api.copy_to_mesh(
          variables.Variable(init_value), Layout.replicated(self.mesh, rank=1)
      )

  @parameterized.named_parameters(
      test_util.product(
          [
              ("Int32", dtypes.int32),
              ("Float32", dtypes.float32),
              ("Int64", dtypes.int64),
              ("Float64", dtypes.float64),
          ],
          [
              (
                  "Scalar",
                  [],
              ),
              (
                  "RankOne",
                  [1],
              ),
              ("RankTwo", [2, 2]),
          ],
      )
  )
  def testImplicitCopyVariableOnCPUMesh(self, dtype, shape):
    self.skipForTfrt("b/235088250")
    self.skipForDeviceType(
        ["GPU", "TPU"], "Variable implicit copy is only allowed for CPU mesh.")

    variable = d_variable.DVariable(array_ops.ones(shape=shape, dtype=dtype))
    new_value = array_ops.zeros(shape=shape, dtype=dtype)

    @polymorphic_function.function
    def assign_function(v, new_value):
      return v.assign(new_value)

    layout = Layout.replicated(self.mesh, rank=len(shape))
    # Run explicitly on the dtensor device with a default mesh since
    # we do not have any registered mesh to broadcast the inputs to.
    with api.default_mesh(self.mesh):
      assign_function(variable, api.pack([new_value] * self.mesh.size, layout))
      read_value = variable.read_value()
    self.assertDTensorEqual(new_value, layout, read_value)

  def testNumpyCallWithReplicatedInput(self):
    a = constant_op.constant([1.0, 2.0, 3.0, 4.0])
    a = api.copy_to_mesh(a, Layout.replicated(self.mesh, rank=1))
    b = a.numpy()
    self.assertAllEqual(b, [1., 2., 3., 4.])

  def testTensorIteration(self):
    a = constant_op.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    a = api.copy_to_mesh(a, Layout.replicated(self.mesh, rank=1))
    iterator = iter(a)
    self.assertAllClose(1., next(iterator))

  def testCopyToMeshWithSameLayout(self):
    a = constant_op.constant([1.0, 2.0, 3.0, 4.0])
    a = api.copy_to_mesh(a, Layout.replicated(self.mesh, rank=1))
    a = api.copy_to_mesh(a, Layout.replicated(self.mesh, rank=1))
    api.check_layout(a, Layout.replicated(self.mesh, rank=1))

  def testSetDefaultLayoutEager(self):
    tensor = constant_op.constant([[1.0], [1.0]])
    tensor = api.copy_to_mesh(tensor, Layout.replicated(self.mesh, rank=2))
    with api._dtensor_device()._default_layout(
        Layout.replicated(self.mesh, rank=1)):
      tensor = array_ops.reshape(tensor, [-1])
    api.check_layout(tensor, Layout.replicated(self.mesh, rank=1))
    self.assertAllClose([1., 1.], tensor.numpy())

  def testSetDefaultLayoutFunction(self):

    @polymorphic_function.function
    def func():
      tensor = constant_op.constant([[1.0], [1.0]])
      return array_ops.reshape(tensor, [-1]), array_ops.reshape(tensor, [-1])

    with api._dtensor_device()._default_layout(
        Layout.batch_sharded(self.mesh, batch_dim=_BATCH_DIM, rank=1)
    ):
      tensor1, tensor2 = func()

    api.check_layout(
        tensor1, Layout.batch_sharded(self.mesh, batch_dim=_BATCH_DIM, rank=1)
    )
    api.check_layout(tensor2, Layout.replicated(self.mesh, rank=1))
    tensor1 = api.relayout(tensor1, Layout.replicated(self.mesh, rank=1))

    self.assertAllClose([1.0, 1.0], tensor1.numpy())
    self.assertAllClose([1.0, 1.0], tensor2.numpy())

  @parameterized.named_parameters(
      # pylint: disable=unnecessary-lambda
      # Needed for the DVariable monkey patch to work.
      ("Variable", lambda x: d_variable.DVariable(x)),
      # pylint: enable=unnecessary-lambda
      ("Tensor", lambda x: x),
  )
  def testStringRepresentation(self, transform):
    replicated = api.copy_to_mesh(
        constant_op.constant(8.0), Layout.replicated(self.mesh, rank=0)
    )
    replicated = transform(replicated)
    replicated_str = str(replicated)
    self.assertIn("8", replicated_str)
    self.assertIn("layout", replicated_str)

    sharded = api.pack(
        [constant_op.constant([8.0]), constant_op.constant([9.0])],
        layout=Layout([_BATCH_DIM], self.mesh),
    )
    sharded = transform(sharded)
    sharded_str = str(sharded)
    self.assertIn("8", sharded_str)
    self.assertIn("9", sharded_str)
    self.assertIn("layout", sharded_str)

  @parameterized.named_parameters(("Async", True), ("Sync", False))
  def testCancellation(self, is_async):
    self.skipForTfrt("b/181368626: support cancellation in tfrt.")
    self.skipForDeviceType(["TPU"], "b/195552283: Fix cancellation on TPU.")
    device = dtensor_device.DTensorDevice(meshes=[self.mesh], is_async=is_async)

    @polymorphic_function.function
    def f(x):
      # Integer division by 0 on one device, which returns a bad status.
      x = math_ops.cast(gen_math_ops.div(x=x, y=x), dtypes.float32)
      # A reduction requiring a collective, which would normally deadlock with
      # one of its participants missing.
      return math_ops.reduce_sum(x, axis=0)

    a = constant_op.constant([[1, 2]])
    b = constant_op.constant([[0, 1]])
    x = device.pack([a, b], layout=Layout([_BATCH_DIM, UNSHARDED], self.mesh))
    with self.assertRaisesRegex(
        errors_impl.InvalidArgumentError, "Integer division by zero"
    ):
      y = f(x)
      y.numpy()
    z = array_ops.identity(x)
    self.assertAllClose([[[1, 2]], [[0, 1]]], device.unpack(z))

  def testCopyToMeshShapeFn(self):

    @polymorphic_function.function
    def f():
      c = constant_op.constant([1.0, 2.0])
      on_mesh = api.copy_to_mesh(c, Layout.replicated(self.mesh, rank=1))
      return on_mesh

    output, = f.get_concrete_function().outputs
    self.assertEqual([2], output.shape)

  def testFetchLayoutForDVariablesReturnsCorrectLayout(self):
    layout = Layout.replicated(self.mesh, 2)
    with api._dtensor_device()._experimental_default_mesh(self.mesh):
      dvariable = d_variable.DVariable(
          api.call_with_layout(
              array_ops.ones, shape=[2, 3], dtype=dtypes.float32, layout=layout
          )
      )
    self.assertEqual(layout, api.fetch_layout(dvariable))

  def testFetchLayoutForDTensorReturnsCorrectLayout(self):
    layout = Layout.replicated(self.mesh, 2)
    tensor = api.call_with_layout(
        array_ops.ones, shape=[2, 3], dtype=dtypes.float32, layout=layout
    )
    self.assertEqual(layout, api.fetch_layout(tensor))

  def testFetchLayoutForRegularTensorsThrowsError(self):
    with self.assertRaisesRegex(
        errors_impl.InvalidArgumentError,
        "FetchLayout expects a tensor placed on the layout device.",
    ):
      api.fetch_layout(constant_op.constant([2, 3]))

  def testFetchLayoutNotEagerlyRaisesRuntimeError(self):

    @polymorphic_function.function
    def f(dtensor_input):
      api.fetch_layout(dtensor_input)

    with self.assertRaisesRegex(RuntimeError,
                                "`fetch_layout` must be called eagerly."):
      f(
          api.copy_to_mesh(
              constant_op.constant(1.0), Layout.replicated(self.mesh, rank=0)
          )
      )

  def testIsDTensor(self):
    normal_tensor = array_ops.zeros(shape=[10, 10])
    self.assertFalse(api.is_dtensor(normal_tensor))

    layout = Layout.replicated(self.mesh, rank=1)
    d_tensor = api.call_with_layout(array_ops.zeros, layout=layout, shape=[10])
    self.assertTrue(api.is_dtensor(d_tensor))

    self.assertFalse(api.is_dtensor([0, 1]))
    self.assertFalse(api.is_dtensor({False: True}))

    self.assertFalse(api.is_dtensor(1))

    class C:
      pass

    self.assertFalse(api.is_dtensor(C()))

  def testIsDTensorNotEagerlyRaisesRuntimeError(self):

    @polymorphic_function.function
    def f(dtensor_input):
      api.is_dtensor(dtensor_input)

    with self.assertRaisesRegex(
        RuntimeError, "`is_dtensor` must be called eagerly."):
      f(
          api.copy_to_mesh(
              constant_op.constant(1.0), Layout.replicated(self.mesh, 0)
          )
      )


class DTensorPackUnpackOnOneDMeshTest(test_util.DTensorBaseTest):

  def setUp(self):
    super(DTensorPackUnpackOnOneDMeshTest, self).setUp()
    global_ids = test_util.create_device_ids_array((2,))
    local_device_ids = np.ravel(global_ids).tolist()
    mesh_dict = {  # pylint: disable=g-complex-comprehension
        device: Mesh(
            [_BATCH_DIM],
            global_ids,
            local_device_ids,
            test_util.create_device_list((2,), device),
        )
        for device in ("CPU", "GPU", "TPU")
    }
    self.mesh = self.configTestMesh(mesh_dict)

  def testUnpack(self):
    with api.default_mesh(self.mesh):
      v = constant_op.constant(1.0)
      v = api.copy_to_mesh(v, Layout.replicated(self.mesh, rank=0))
      self.assertAllClose([1.0, 1.0], api.unpack(v))

  def testUnpackVariables(self):
    v0 = d_variable.DVariable(
        api.call_with_layout(
            array_ops.ones,
            shape=[2, 3],
            dtype=dtypes.float32,
            layout=Layout.replicated(self.mesh, 2),
        )
    )
    with self.assertRaisesRegex(
        TypeError,
        "Received Variable input to unpack, Variable is not supported."):
      api._dtensor_device().unpack(v0)

  def testUnpackingRegularTensorRaisesInvalidArgumentError(self):
    with self.assertRaisesRegex(
        errors_impl.InvalidArgumentError,
        "DTensorUnpack expects a tensor placed on the DTensor device",
    ):
      api._dtensor_device().unpack(constant_op.constant([1.0, 2.0]))

  def testUnpackingNotEagerlyRaisesRuntimeError(self):

    @polymorphic_function.function
    def f(dtensor_input):
      api._dtensor_device().unpack(dtensor_input)

    with self.assertRaisesRegex(
        RuntimeError, "`unpack` must be called eagerly."):
      f(
          api.copy_to_mesh(
              constant_op.constant(1.0), Layout.replicated(self.mesh, rank=0)
          )
      )

  def testPack(self):
    a = constant_op.constant([1.0, 2.0])
    b = constant_op.constant([3.0, 4.0])
    with ops.device_v2(api.device_name()):
      packed_tensor = api.pack(
          [a, b], layout=Layout.batch_sharded(self.mesh, _BATCH_DIM, rank=1)
      )
      api.check_layout(
          packed_tensor, Layout.batch_sharded(self.mesh, _BATCH_DIM, rank=1)
      )
      self.assertAllEqual([
          4,
      ], packed_tensor.shape)
      unpacked_tensor = api.unpack(packed_tensor)
      self.assertAllClose([1., 2.], unpacked_tensor[0])
      self.assertAllClose([3., 4.], unpacked_tensor[1])

  def testPackingNotEagerlyRaisesRuntimeError(self):

    @polymorphic_function.function
    def f(a):
      api.pack([a, a], layout=Layout.replicated(self.mesh, rank=1))

    with self.assertRaisesRegex(RuntimeError, "`pack` must be called eagerly."):
      f(constant_op.constant([1.0]))

  def testPackingVariablesRaisesTypeError(self):
    with self.assertRaisesRegex(
        TypeError,
        "Received Variable input to Pack, Variable is not supported."):
      api._dtensor_device().pack(
          [
              d_variable.DVariable(array_ops.ones([2, 3])),
              d_variable.DVariable(array_ops.ones([2, 3])),
          ],
          Layout.replicated(self.mesh, rank=2),
      )

  def testPackDevice(self):
    a = constant_op.constant([1.0, 2.0])
    b = constant_op.constant([3.0, 4.0])
    with ops.device_v2(api.device_name()):
      packed_tensor = api.pack(
          [a, b], layout=Layout.batch_sharded(self.mesh, _BATCH_DIM, rank=1)
      )
      unpacked_tensor = api.unpack(packed_tensor)
      self.assertAllEqual(self.mesh.local_devices(),
                          [t.device for t in unpacked_tensor])

  def testPackScalar(self):
    a = constant_op.constant(1.0)
    with ops.device_v2(api.device_name()):
      packed_layout = Layout([], self.mesh)
      packed_tensor = api.pack([a, a], layout=packed_layout)
      api.check_layout(packed_tensor, packed_layout)
      self.assertAllEqual([], packed_tensor.shape)
      unpacked_tensor = api.unpack(packed_tensor)
      self.assertAllClose([a, a], unpacked_tensor)

  def testPackHigherRankValue(self):
    # Pack a rank 3 matrix into a 1d mesh.
    a = constant_op.constant(
        [[[1, 2, 3], [4, 5, 6]], [[2, 3, 4], [5, 6, 7]]]
    )  # 2x2x3
    b = constant_op.constant(
        [[[3, 2, 1], [6, 5, 4]], [[4, 3, 2], [7, 6, 5]]]
    )  # 2x2x3
    pack_layout = Layout([_BATCH_DIM, UNSHARDED, UNSHARDED], self.mesh)
    with ops.device_v2(api.device_name()):
      # pack to 4x2x3
      packed_tensor = api.pack([a, b], layout=pack_layout)
      api.check_layout(packed_tensor, pack_layout)
      self.assertAllEqual([4, 2, 3], packed_tensor.shape)


class DTensorPackUnpackOnTwoDMeshTest(test_util.DTensorBaseTest):

  def setUp(self):
    super().setUp()

    self.skipForDeviceType(["TPU"],
                           "all tests require 8 TPU cores.",
                           unless_device_count_equals_to=8)

    global_ids = test_util.create_device_ids_array((2, 4))
    local_ids = np.ravel(global_ids).tolist()
    mesh_dict = {  # pylint: disable=g-complex-comprehension
        device: Mesh(
            [_BATCH_DIM, _MESH_DIM_X],
            global_ids,
            local_ids,
            test_util.create_device_list((2, 4), device),
        )
        for device in ("CPU", "GPU", "TPU")
    }
    self.mesh = self.configTestMesh(mesh_dict)

  def testPackWithScalars(self):
    a = constant_op.constant(1.23)
    with ops.device_v2(api.device_name()):
      packed_layout = Layout([], self.mesh)
      packed_tensor = api.pack([a, a, a, a, a, a, a, a], layout=packed_layout)

      api.check_layout(packed_tensor, packed_layout)

      self.assertAllEqual([], packed_tensor.shape)
      unpacked_tensor = api.unpack(packed_tensor)
      self.assertAllClose([a, a, a, a, a, a, a, a], unpacked_tensor)

  def testPackWithScalarsWithInvalidRank(self):
    a = constant_op.constant(1.23)
    with ops.device_v2(api.device_name()):
      invalid_packed_layout = Layout([UNSHARDED], self.mesh)
      with self.assertRaisesRegex(
          errors_impl.InvalidArgumentError,
          "Packed layout should have the same rank",
      ):
        _ = api.pack([a, a, a, a, a, a, a, a], layout=invalid_packed_layout)

  def testPackWithBatchSharding(self):
    a = constant_op.constant([[1.0], [2.0]])
    b = constant_op.constant([[3.0], [4.0]])
    with ops.device_v2(api.device_name()):
      packed_tensor = api.pack(
          [a, a, a, a, b, b, b, b],
          layout=Layout.batch_sharded(self.mesh, _BATCH_DIM, rank=2),
      )
      api.check_layout(
          packed_tensor, Layout.batch_sharded(self.mesh, _BATCH_DIM, rank=2)
      )
      self.assertAllEqual([4, 1], packed_tensor.shape)
      unpacked_tensor = api.unpack(packed_tensor)
    self.assertAllClose([a, a, a, a, b, b, b, b], unpacked_tensor)

  def testPackWithFullReplicated(self):
    a = constant_op.constant([[1.0], [2.0]])
    with ops.device_v2(api.device_name()):
      packed_layout = Layout([UNSHARDED, UNSHARDED], self.mesh)
      packed_tensor = api.pack([a, a, a, a, a, a, a, a], layout=packed_layout)
      api.check_layout(packed_tensor, packed_layout)
      self.assertAllEqual([2, 1], packed_tensor.shape)
      unpacked_tensor = api.unpack(packed_tensor)
    self.assertAllClose([a, a, a, a, a, a, a, a], unpacked_tensor)

  def testFillUsesSpecifiedLayout(self):
    with api.default_mesh(self.mesh):
      # TODO(allenl): Figure out why the embedded constant (triggered for
      # dtypes.int32) gets a sharded layout by default.
      dims = constant_op.constant([4, 1], dtype=dtypes.int64)
      value = constant_op.constant(1.0)
      with api._dtensor_device()._default_layout(
          Layout.batch_sharded(self.mesh, _BATCH_DIM, rank=2)):
        filled = array_ops.fill(value=value, dims=dims)
      api.check_layout(
          filled, Layout.batch_sharded(self.mesh, _BATCH_DIM, rank=2)
      )
      unpacked_tensor = api.unpack(filled)
    self.assertAllClose(8 * [[[1.], [1.]]], unpacked_tensor)
    self.assertEqual([4, 1], filled.shape)


class DTensorSparse(test_util.DTensorBaseTest):

  def setUp(self):
    super().setUp()

    self.skipForDeviceType(["TPU"],
                           "all tests require 8 TPU cores.",
                           unless_device_count_equals_to=8)

    global_ids = test_util.create_device_ids_array((2, 4))
    local_ids = np.ravel(global_ids).tolist()
    mesh_dict = {  # pylint: disable=g-complex-comprehension
        device: Mesh(
            [_BATCH_DIM, _MESH_DIM_X],
            global_ids,
            local_ids,
            test_util.create_device_list((2, 4), device),
        )
        for device in ("CPU", "GPU", "TPU")
    }
    self.mesh = self.configTestMesh(mesh_dict)

  @parameterized.named_parameters(
      test_util.product([("Replicated", "replicated"), ("Sharded", "batch")], [(
          "RankTwo",
          [2, 4],
      ), (
          "RankThree",
          [2, 2, 3],
      )]))
  def testPackUnpackReturnsCorrectValuesAndDevices(self, sharding, shape):
    a = sparse_ops.from_dense(
        stateless_random_ops.stateless_random_uniform(shape, seed=[0, 1])
    )
    b = sparse_ops.from_dense(
        stateless_random_ops.stateless_random_uniform(shape, seed=[0, 1])
    )

    if sharding == "replicated":
      layout = Layout(["unsharded"] * len(shape), self.mesh)
      input_tensor = 8 * [a]
      expected = 8 * [sparse_ops.sparse_tensor_to_dense(a)]
      expected_shape = shape
    else:
      layout = Layout([_BATCH_DIM] + ["unsharded"] * (len(shape) - 1),
                      self.mesh)
      input_tensor = 4 * [a] + 4 * [b]
      expected = 4 * [sparse_ops.sparse_tensor_to_dense(a)] + 4 * [
          sparse_ops.sparse_tensor_to_dense(b)
      ]
      expected_shape = [shape[0] * 2] + shape[1:]

    with ops.device_v2(api._dtensor_device().name):
      packed_tensor = api.pack(input_tensor, layout)
      api.check_layout(packed_tensor, layout)
      unpacked_tensor = api.unpack(packed_tensor)

    got = [sparse_ops.sparse_tensor_to_dense(t) for t in unpacked_tensor]

    # Check shape of packed tensor.
    self.assertAllEqual(expected_shape, packed_tensor.shape)
    # Check values.
    self.assertAllClose(expected, got)
    # Check devices.
    self.assertAllEqual(self.mesh.local_devices(),
                        [t.indices.device for t in unpacked_tensor])
    self.assertAllEqual(self.mesh.local_devices(),
                        [t.values.device for t in unpacked_tensor])

  def testPackingMixedTensorTypesRaisesTypeError(self):
    tensor = stateless_random_ops.stateless_random_uniform([2, 4], seed=[0, 1])
    sparse_tensor = sparse_ops.from_dense(tensor)
    with ops.device_v2(api.device_name()):
      with self.assertRaisesRegex(TypeError,
                                  "Cannot Pack SparseTensors with Tensors."):
        api.pack(
            4 * [tensor] + 4 * [sparse_tensor],
            Layout.replicated(self.mesh, rank=2),
        )

  def testPackingTensorsWithDifferentShapesRaisesTypeError(self):
    a = sparse_ops.from_dense(
        stateless_random_ops.stateless_random_uniform([2, 2], seed=[0, 1])
    )
    b = sparse_ops.from_dense(
        stateless_random_ops.stateless_random_uniform([4, 4], seed=[0, 1])
    )
    with ops.device_v2(api.device_name()):
      with self.assertRaisesRegex(
          TypeError, "All input SparseTensors to Pack must be same shape."):
        api.pack(4 * [a] + 4 * [b], Layout.replicated(self.mesh, rank=2))

  def testPackingSparseTensorsReturnsCorrectLayout(self):
    layout = Layout.replicated(self.mesh, 2)
    a = sparse_ops.from_dense(
        stateless_random_ops.stateless_random_uniform([16, 16], seed=[0, 1])
    )
    with ops.device_v2(api.device_name()):
      api.check_layout(api.pack(8 * [a], layout), layout)


if __name__ == "__main__":
  test.main()
