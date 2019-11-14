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
"""Tests for SavedModel utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.core.framework import types_pb2
from tensorflow.core.protobuf import struct_pb2
from tensorflow.python.eager import context
from tensorflow.python.eager import function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.platform import test
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.saved_model import utils


class UtilsTest(test.TestCase):

  @test_util.run_v1_only("b/120545219")
  def testBuildTensorInfoOp(self):
    x = constant_op.constant(1, name="x")
    y = constant_op.constant(2, name="y")
    z = control_flow_ops.group([x, y], name="op_z")
    z_op_info = utils.build_tensor_info_from_op(z)
    self.assertEqual("op_z", z_op_info.name)
    self.assertEqual(types_pb2.DT_INVALID, z_op_info.dtype)
    self.assertEqual(0, len(z_op_info.tensor_shape.dim))

  @test_util.run_v1_only("b/120545219")
  def testBuildTensorInfoDefunOp(self):
    @function.defun
    def my_init_fn(x, y):
      self.x_var = x
      self.y_var = y

    x = constant_op.constant(1, name="x")
    y = constant_op.constant(2, name="y")
    init_op_info = utils.build_tensor_info_from_op(my_init_fn(x, y))
    self.assertEqual("PartitionedFunctionCall", init_op_info.name)
    self.assertEqual(types_pb2.DT_INVALID, init_op_info.dtype)
    self.assertEqual(0, len(init_op_info.tensor_shape.dim))

  @test_util.run_v1_only("b/120545219")
  def testBuildTensorInfoDense(self):
    x = array_ops.placeholder(dtypes.float32, 1, name="x")
    x_tensor_info = utils.build_tensor_info(x)
    self.assertEqual("x:0", x_tensor_info.name)
    self.assertEqual(types_pb2.DT_FLOAT, x_tensor_info.dtype)
    self.assertEqual(1, len(x_tensor_info.tensor_shape.dim))
    self.assertEqual(1, x_tensor_info.tensor_shape.dim[0].size)

  @test_util.run_v1_only("b/120545219")
  def testBuildTensorInfoSparse(self):
    x = array_ops.sparse_placeholder(dtypes.float32, [42, 69], name="x")
    x_tensor_info = utils.build_tensor_info(x)
    self.assertEqual(x.values.name,
                     x_tensor_info.coo_sparse.values_tensor_name)
    self.assertEqual(x.indices.name,
                     x_tensor_info.coo_sparse.indices_tensor_name)
    self.assertEqual(x.dense_shape.name,
                     x_tensor_info.coo_sparse.dense_shape_tensor_name)
    self.assertEqual(types_pb2.DT_FLOAT, x_tensor_info.dtype)
    self.assertEqual(2, len(x_tensor_info.tensor_shape.dim))
    self.assertEqual(42, x_tensor_info.tensor_shape.dim[0].size)
    self.assertEqual(69, x_tensor_info.tensor_shape.dim[1].size)

  @test_util.run_v1_only("b/120545219")
  def testBuildTensorInfoRagged(self):
    x = ragged_factory_ops.constant([[1, 2], [3]])
    x_tensor_info = utils.build_tensor_info(x)
    # Check components
    self.assertEqual(x.values.name,
                     x_tensor_info.composite_tensor.components[0].name)
    self.assertEqual(types_pb2.DT_INT32,
                     x_tensor_info.composite_tensor.components[0].dtype)
    self.assertEqual(x.row_splits.name,
                     x_tensor_info.composite_tensor.components[1].name)
    self.assertEqual(types_pb2.DT_INT64,
                     x_tensor_info.composite_tensor.components[1].dtype)
    # Check type_spec.
    struct_coder = nested_structure_coder.StructureCoder()
    spec_proto = struct_pb2.StructuredValue(
        type_spec_value=x_tensor_info.composite_tensor.type_spec)
    spec = struct_coder.decode_proto(spec_proto)
    self.assertEqual(spec, x._type_spec)

  def testBuildTensorInfoEager(self):
    x = constant_op.constant(1, name="x")
    with context.eager_mode(), self.assertRaisesRegexp(
        RuntimeError, "build_tensor_info is not supported in Eager mode"):
      utils.build_tensor_info(x)

  @test_util.run_v1_only("b/120545219")
  def testGetTensorFromInfoDense(self):
    expected = array_ops.placeholder(dtypes.float32, 1, name="x")
    tensor_info = utils.build_tensor_info(expected)
    actual = utils.get_tensor_from_tensor_info(tensor_info)
    self.assertIsInstance(actual, ops.Tensor)
    self.assertEqual(expected.name, actual.name)

  @test_util.run_v1_only("b/120545219")
  def testGetTensorFromInfoSparse(self):
    expected = array_ops.sparse_placeholder(dtypes.float32, name="x")
    tensor_info = utils.build_tensor_info(expected)
    actual = utils.get_tensor_from_tensor_info(tensor_info)
    self.assertIsInstance(actual, sparse_tensor.SparseTensor)
    self.assertEqual(expected.values.name, actual.values.name)
    self.assertEqual(expected.indices.name, actual.indices.name)
    self.assertEqual(expected.dense_shape.name, actual.dense_shape.name)

  def testGetTensorFromInfoInOtherGraph(self):
    with ops.Graph().as_default() as expected_graph:
      expected = array_ops.placeholder(dtypes.float32, 1, name="right")
      tensor_info = utils.build_tensor_info(expected)
    with ops.Graph().as_default():  # Some other graph.
      array_ops.placeholder(dtypes.float32, 1, name="other")
    actual = utils.get_tensor_from_tensor_info(tensor_info,
                                               graph=expected_graph)
    self.assertIsInstance(actual, ops.Tensor)
    self.assertIs(actual.graph, expected_graph)
    self.assertEqual(expected.name, actual.name)

  def testGetTensorFromInfoInScope(self):
    # Build a TensorInfo with name "bar/x:0".
    with ops.Graph().as_default():
      with ops.name_scope("bar"):
        unscoped = array_ops.placeholder(dtypes.float32, 1, name="x")
        tensor_info = utils.build_tensor_info(unscoped)
        self.assertEqual("bar/x:0", tensor_info.name)
    # Build a graph with node "foo/bar/x:0", akin to importing into scope foo.
    with ops.Graph().as_default():
      with ops.name_scope("foo"):
        with ops.name_scope("bar"):
          expected = array_ops.placeholder(dtypes.float32, 1, name="x")
      self.assertEqual("foo/bar/x:0", expected.name)
      # Test that tensor is found by prepending the import scope.
      actual = utils.get_tensor_from_tensor_info(tensor_info,
                                                 import_scope="foo")
      self.assertEqual(expected.name, actual.name)

  @test_util.run_v1_only("b/120545219")
  def testGetTensorFromInfoRaisesErrors(self):
    expected = array_ops.placeholder(dtypes.float32, 1, name="x")
    tensor_info = utils.build_tensor_info(expected)
    tensor_info.name = "blah:0"  # Nonexistant name.
    with self.assertRaises(KeyError):
      utils.get_tensor_from_tensor_info(tensor_info)
    tensor_info.ClearField("name")  # Malformed (missing encoding).
    with self.assertRaises(ValueError):
      utils.get_tensor_from_tensor_info(tensor_info)

if __name__ == "__main__":
  test.main()
