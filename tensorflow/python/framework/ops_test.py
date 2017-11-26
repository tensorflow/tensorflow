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
"""Tests for tensorflow.python.framework.ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gc
import weakref

from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import types_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.eager import context
from tensorflow.python.framework import common_shapes
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import device as pydev
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import test_ops
from tensorflow.python.framework import test_util
from tensorflow.python.framework import versions
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resources
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
import tensorflow.python.ops.gradients  # pylint: disable=unused-import
from tensorflow.python.platform import googletest
from tensorflow.python.util import compat

ops._set_call_cpp_shape_fn(common_shapes.call_cpp_shape_fn)


@test_util.with_c_api
class ResourceTest(test_util.TensorFlowTestCase):

  def testBuildGraph(self):
    with self.test_session():
      pt = test_ops.stub_resource_handle_op(container="a", shared_name="b")
      test_ops.resource_create_op(pt).run()

  def testInitialize(self):
    with self.test_session():
      handle = test_ops.stub_resource_handle_op(container="a", shared_name="b")
      resources.register_resource(
          handle=handle,
          create_op=test_ops.resource_create_op(handle),
          is_initialized_op=test_ops.resource_initialized_op(handle))
      self.assertEquals(
          len(
              resources.report_uninitialized_resources(
                  resources.shared_resources()).eval()), 1)
      resources.initialize_resources(resources.shared_resources()).run()
      self.assertEquals(
          len(
              resources.report_uninitialized_resources(
                  resources.shared_resources()).eval()), 0)


@test_util.with_c_api
class TensorTest(test_util.TensorFlowTestCase):

  def testShape(self):
    op = ops.Operation(
        ops._NodeDef("FloatOutput", "myop"), ops.Graph(), [], [dtypes.float32])
    t = op.outputs[0]
    self.assertEqual(tensor_shape.unknown_shape(), t.get_shape())
    t.set_shape([1, 2, 3])
    self.assertEqual([1, 2, 3], t.get_shape())

  def testIterable(self):
    op = ops.Operation(
        ops._NodeDef("FloatOutput", "myop"), ops.Graph(), [], [dtypes.float32])
    t = op.outputs[0]
    self.assertTrue(isinstance(t, ops.Tensor))
    with self.assertRaisesRegexp(TypeError, "iter"):
      for _ in t:
        pass


@test_util.with_c_api
class IndexedSlicesTest(test_util.TensorFlowTestCase):

  def testToTensor(self):
    with self.test_session():
      values = constant_op.constant([2, 3, 5, 7], shape=[2, 2])
      indices = constant_op.constant([0, 2])
      dense_shape = constant_op.constant([3, 2])
      x = ops.IndexedSlices(values, indices, dense_shape)
      tensor = ops.convert_to_tensor(x, name="tensor")
      self.assertAllEqual(tensor.eval(), [[2, 3], [0, 0], [5, 7]])

  def testNegation(self):
    with self.test_session():
      values = constant_op.constant([2, 3, 5, 7], shape=[2, 2])
      indices = constant_op.constant([0, 2])
      x = -ops.IndexedSlices(values, indices)
      self.assertAllEqual(x.values.eval(), [[-2, -3], [-5, -7]])
      self.assertAllEqual(x.indices.eval(), [0, 2])

  def testScalarMul(self):
    with self.test_session():
      values = constant_op.constant([2, 3, 5, 7], shape=[2, 2])
      indices = constant_op.constant([0, 2])
      x = math_ops.scalar_mul(-2, ops.IndexedSlices(values, indices))
      self.assertAllEqual(x.values.eval(), [[-4, -6], [-10, -14]])
      self.assertAllEqual(x.indices.eval(), [0, 2])


@test_util.with_c_api
class NodeDefConstructorTest(test_util.TensorFlowTestCase):

  def testNoArgs(self):
    nodedef = ops._NodeDef("None", "bar")
    self.assertProtoEquals("op: 'None' name: 'bar'", nodedef)

  def testArgs(self):
    nodedef = ops._NodeDef("foo", "bar", device="/device:baz:*")
    self.assertProtoEquals("op:'foo' name:'bar' device:'/device:baz:*'",
                           nodedef)
    nodedef = ops._NodeDef("foo", "bar", device=pydev.DeviceSpec(job="j"))
    self.assertProtoEquals("op:'foo' name:'bar' device:'/job:j'", nodedef)


def _apply_op(g, *args, **kwargs):
  op = g.create_op(*args, **kwargs)
  if len(op.outputs) == 1:
    return op.outputs[0]
  else:
    return op.outputs


@test_util.with_c_api
class OperationTest(test_util.TensorFlowTestCase):

  def testNoInputs(self):
    op = test_ops.float_output_string_output(name="myop").a.op
    self.assertEqual(2, len(op.values()))
    self.assertEqual(0, len(op.inputs))
    self.assertEqual("myop", op.name)

    float_t, label_str_t = op.values()
    self.assertEqual(dtypes.float32, float_t.dtype)
    self.assertEqual(op, float_t.op)
    self.assertEqual(0, float_t._value_index)
    self.assertEqual(0, len(float_t._consumers))
    self.assertEqual("myop", float_t._as_node_def_input())

    self.assertEqual(dtypes.string, label_str_t.dtype)
    self.assertEqual(op, label_str_t.op)
    self.assertEqual(1, label_str_t._value_index)
    self.assertEqual(0, len(label_str_t._consumers))
    self.assertEqual("myop:1", label_str_t._as_node_def_input())

    self.assertProtoEquals("op:'FloatOutputStringOutput' name:'myop'",
                           op.node_def)

  def testNoOutputs(self):
    op1 = test_ops.float_output(name="myop1").op
    float_t, = op1.values()
    op2 = test_ops.float_input(float_t, name="myop2")
    self.assertEqual(0, len(op2.values()))
    self.assertEqual(1, len(op2.inputs))
    self.assertIs(float_t, op2.inputs[0])

    self.assertEqual(1, len(float_t._consumers))
    self.assertEqual(op2, float_t._consumers[0])

    self.assertProtoEquals("op:'FloatOutput' name:'myop1'", op1.node_def)
    self.assertProtoEquals("op:'FloatInput' name:'myop2' input:'myop1'",
                           op2.node_def)

  def testInputsAndOutputs(self):
    op1 = test_ops.float_output(name="myop1").op
    self.assertEqual(1, len(op1.values()))
    float1_t, = op1.values()

    op2 = test_ops.float_output_string_output(name="myop2").a.op
    self.assertEqual(2, len(op2.values()))
    float2_t, label2_str_t = op2.values()

    # Note that we consume label2_str_t twice here.
    op3 = test_ops.foo2(float1_t, label2_str_t, label2_str_t, name="myop3").d.op
    self.assertEqual(2, len(op3.values()))

    self.assertEqual(1, len(float1_t._consumers))
    self.assertEqual(op3, float1_t._consumers[0])

    self.assertEqual(0, len(float2_t._consumers))

    self.assertEqual(2, len(label2_str_t._consumers))
    self.assertEqual(op3, label2_str_t._consumers[0])
    self.assertEqual(op3, label2_str_t._consumers[1])

    self.assertProtoEquals("""
    op:'Foo2' name:'myop3'
    input:'myop1' input:'myop2:1' input:'myop2:1'
    """, op3.node_def)

  def testDeviceObject(self):
    op = ops.Operation(ops._NodeDef("None", "myop"), ops.Graph(), [], [])
    op._set_device("/job:goo/device:GPU:0")
    self.assertProtoEquals(
        "op:'None' name:'myop' device:'/job:goo/device:GPU:0' ", op.node_def)
    op = ops.Operation(ops._NodeDef("None", "op2"), ops.Graph(), [], [])
    op._set_device(
        pydev.DeviceSpec(
            job="muu", device_type="CPU", device_index=0))
    self.assertProtoEquals(
        "op:'None' name:'op2' device:'/job:muu/device:CPU:0'", op.node_def)

  def testReferenceInput(self):
    g = ops.Graph()
    op1 = ops.Operation(
        ops._NodeDef("RefOutputFloatOutput", "op1"), g, [],
        [dtypes.float32_ref, dtypes.float32])
    self.assertProtoEquals("op:'RefOutputFloatOutput' name:'op1'", op1.node_def)
    self.assertEquals([], list(op1.inputs))
    ref_t, nonref_t = op1.values()
    # NOTE(mrry): Must specify input_types to preserve ref-typed input.
    op2 = ops.Operation(
        ops._NodeDef("RefInputFloatInput", "op2"),
        g, [ref_t, nonref_t], [],
        input_types=[dtypes.float32_ref, dtypes.float32])
    self.assertProtoEquals(
        "op:'RefInputFloatInput' name:'op2' input:'op1' input:'op1:1'",
        op2.node_def)
    self.assertEquals([ref_t, nonref_t], list(op2.inputs))
    op3 = ops.Operation(
        ops._NodeDef("TwoFloatInputs", "op3"), g, [ref_t, nonref_t], [])
    self.assertProtoEquals(
        "op:'TwoFloatInputs' name:'op3' input:'op1' input:'op1:1'",
        op3.node_def)

  def testInvalidNames(self):
    g = ops.Graph()
    with self.assertRaises(ValueError):
      ops.Operation(ops._NodeDef("op", ""), g)
    with self.assertRaises(ValueError):
      ops.Operation(ops._NodeDef("op", "_invalid"), g)
    with self.assertRaises(ValueError):
      ops.Operation(ops._NodeDef("op", "-invalid"), g)
    with self.assertRaises(ValueError):
      ops.Operation(ops._NodeDef("op", "/invalid"), g)
    with self.assertRaises(ValueError):
      ops.Operation(ops._NodeDef("op", "invalid:0"), g)

  def testNoShapeFunction(self):
    op = test_ops.a()
    self.assertEqual(tensor_shape.unknown_shape(), op.get_shape())

  def testConvertToTensorNestedArray(self):
    with self.test_session():
      values = [[2], [3], [5], [7]]
      tensor = ops.convert_to_tensor(values)
      self.assertAllEqual((4, 1), tensor.get_shape().as_list())
      self.assertAllEqual(values, tensor.eval())

  def testShapeTuple(self):
    with self.test_session():
      c = constant_op.constant(1)
      self.assertEqual(c._shape_tuple(), ())  # pylint: disable=protected-access

  def testConvertToTensorEager(self):
    with context.eager_mode():
      t = constant_op.constant(1)
      self.assertTrue(isinstance(t, ops.EagerTensor))
      converted = ops.convert_to_tensor(t)
      self.assertTrue(isinstance(converted, ops.EagerTensor))
      converted = ops.convert_to_tensor(1)
      self.assertTrue(isinstance(converted, ops.EagerTensor))

  def testConvertToTensorNestedTuple(self):
    with self.test_session():
      values = ((2,), (3,), (5,), (7,))
      tensor = ops.convert_to_tensor(values)
      self.assertAllEqual((4, 1), tensor.get_shape().as_list())
      self.assertAllEqual(values, ops.convert_to_tensor(values).eval())

  def testConvertToTensorNestedTensors(self):
    with self.test_session():
      values = ((2,), (3,), (5,), (7,))
      tensor = ops.convert_to_tensor(
          [constant_op.constant(row) for row in values])
      self.assertAllEqual((4, 1), tensor.get_shape().as_list())
      self.assertAllEqual(values, tensor.eval())
      tensor = ops.convert_to_tensor(
          [[constant_op.constant(v) for v in row] for row in values])
      self.assertAllEqual((4, 1), tensor.get_shape().as_list())
      self.assertAllEqual(values, tensor.eval())

  def testConvertToTensorNestedMix(self):
    with self.test_session():
      values = ([2], (3,), [constant_op.constant(5)], constant_op.constant([7]))
      tensor = ops.convert_to_tensor(values)
      self.assertAllEqual((4, 1), tensor.get_shape().as_list())
      self.assertAllEqual(((2,), (3,), (5,), (7,)), tensor.eval())

  def testConvertToTensorPreferred(self):
    with self.test_session():
      values = [2, 3, 5, 7]
      tensor = ops.convert_to_tensor(values, preferred_dtype=dtypes.float32)
      self.assertEqual(dtypes.float32, tensor.dtype)

    with self.test_session():
      # Convert empty tensor to anything.
      values = []
      tensor = ops.convert_to_tensor(values, preferred_dtype=dtypes.int64)
      self.assertEqual(dtypes.int64, tensor.dtype)

    with self.test_session():
      # The preferred dtype is a type error and will convert to
      # float32 instead.
      values = [1.23]
      tensor = ops.convert_to_tensor(values, preferred_dtype=dtypes.int64)
      self.assertEqual(dtypes.float32, tensor.dtype)

  def testConvertToInvalidTensorType(self):
    with self.assertRaises(TypeError):
      # Forcing an invalid dtype should fail with a type error.
      values = [1.23]
      _ = ops.convert_to_tensor(values, dtype=dtypes.int64)

  def testNoConvert(self):
    # Operation cannot be converted to Tensor.
    op = control_flow_ops.no_op()
    with self.assertRaisesRegexp(TypeError,
                                 r"Can't convert Operation '.*' to Tensor"):
      ops.convert_to_tensor(op)

  def testStr(self):
    node_def = ops._NodeDef("None", "op1")
    op = ops.Operation(node_def, ops.Graph(), [], [dtypes.float32])
    self.assertEqual(str(node_def), str(op))

  def testRepr(self):
    op = ops.Operation(
        ops._NodeDef("None", "op1"), ops.Graph(), [], [dtypes.float32])
    self.assertEqual("<tf.Operation 'op1' type=None>", repr(op))

  def testGetAttr(self):
    op = test_ops.default_attrs()
    self.assertEqual(op.get_attr("string_val"), b"abc")
    self.assertEqual(op.get_attr("string_list_val"), [b"abc", b""])
    self.assertEqual(op.get_attr("int_val"), 123)
    self.assertEqual(op.get_attr("int_list_val"), [1, 2, 3])
    self.assertEqual(op.get_attr("float_val"), 10.0)
    self.assertEqual(op.get_attr("float_list_val"), [10.0])
    self.assertEqual(op.get_attr("bool_val"), True)
    self.assertEqual(op.get_attr("bool_list_val"), [True, False])
    self.assertEqual(op.get_attr("shape_val"),
                     tensor_shape.as_shape([2, 1]).as_proto())
    self.assertEqual(op.get_attr("shape_list_val"),
                     [tensor_shape.as_shape([]).as_proto(),
                      tensor_shape.as_shape([1]).as_proto()])
    self.assertEqual(op.get_attr("tensor_val"),
                     tensor_util.make_tensor_proto(1, dtypes.int32))
    self.assertEqual(op.get_attr("tensor_list_val"),
                     [tensor_util.make_tensor_proto(1, dtypes.int32)])

    type_val = op.get_attr("type_val")
    # First check that type_val is a DType, because the assertEquals will work
    # no matter what since DType overrides __eq__
    self.assertIsInstance(type_val, dtypes.DType)
    self.assertEqual(type_val, dtypes.int32)

    type_list_val = op.get_attr("type_list_val")
    self.assertTrue(all(isinstance(x, dtypes.DType) for x in type_list_val))
    self.assertEqual(type_list_val, [dtypes.int32, dtypes.float32])

    @function.Defun(dtypes.float32, func_name="MyFunc")
    def func(x):
      return x

    op = test_ops.func_attr(func)
    self.assertEqual(op.get_attr("f"),
                     attr_value_pb2.NameAttrList(name="MyFunc"))

    # Try fetching missing attr
    if ops._USE_C_API:
      error_msg = "Operation 'FuncAttr' has no attr named 'FakeAttr'."
    else:
      error_msg = "No attr named 'FakeAttr' in name: \"FuncAttr\""

    with self.assertRaisesRegexp(ValueError, error_msg):
      op.get_attr("FakeAttr")

  # TODO(b/65162920): remove this test when users who are directly mutating the
  # node_def have been updated to proper usage.
  def testSetAttr(self):
    op = test_ops.int_attr().op
    op._set_attr("foo", attr_value_pb2.AttrValue(i=2))
    # TODO(skyewm): add node_def check
    self.assertEqual(op.get_attr("foo"), 2)

  # TODO(nolivia): test all error cases
  def testAddControlInput(self):
    # The C API dedups redundant control edges, pure Python does not
    if ops._USE_C_API: return
    with ops.Graph().as_default():
      x = constant_op.constant(1).op
      y = constant_op.constant(2).op
      z = constant_op.constant(3).op
    z._add_control_input(x)  # pylint: disable=protected-access
    self.assertEqual(z.control_inputs, [x])
    z._add_control_input(x)  # pylint: disable=protected-access
    self.assertEqual(z.control_inputs, [x, x])
    z._add_control_inputs([x, y, y])  # pylint: disable=protected-access
    self.assertEqual(z.control_inputs, [x, x, x, y, y])

  def testAddControlInputC(self):
    # The C API dedups redundant control edges, pure Python does not
    if not ops._USE_C_API: return
    with ops.Graph().as_default():
      x = constant_op.constant(1).op
      y = constant_op.constant(2).op
      z = constant_op.constant(3).op
    z._add_control_input(x)  # pylint: disable=protected-access
    self.assertEqual(z.control_inputs, [x])
    z._add_control_input(x)  # pylint: disable=protected-access
    self.assertEqual(z.control_inputs, [x])
    z._add_control_inputs([x, y, y])  # pylint: disable=protected-access
    self.assertEqual(z.control_inputs, [x, y])

  def testControlInputCycle(self):
    # Non-C API path has a different error message
    if not ops._USE_C_API: return
    graph = ops.Graph()
    with graph.as_default():
      z = constant_op.constant(0)
      x = constant_op.constant(1)
      y = constant_op.constant(2)
      y.op._add_control_input(z.op)  # pylint: disable=protected-access
      y.op._add_control_input(x.op)  # pylint: disable=protected-access
      x.op._add_control_input(y.op)  # pylint: disable=protected-access
    with self.test_session(graph=graph) as sess:
      with self.assertRaisesRegexp(
          errors.InvalidArgumentError,
          "Graph is invalid, contains a cycle with 2 nodes"):
        sess.run(x)

  def testUpdateInput(self):
    g = ops.Graph()
    with g.as_default():
      x = constant_op.constant(1)
      y = constant_op.constant(2)
      z = x + y

    z.op._update_input(0, y)  # pylint: disable=protected-access
    self.assertEquals(list(z.op.inputs), [y, y])
    with session.Session(graph=g) as sess:
      self.assertEquals(sess.run(z), 4)

    z.op._update_input(0, x)  # pylint: disable=protected-access
    self.assertEquals(list(z.op.inputs), [x, y])
    with session.Session(graph=g) as sess:
      self.assertEquals(sess.run(z), 3)

    z.op._update_input(1, y)  # pylint: disable=protected-access
    self.assertEquals(list(z.op.inputs), [x, y])
    with session.Session(graph=g) as sess:
      self.assertEquals(sess.run(z), 3)

  def testUpdateInputGraphError(self):
    g_0 = ops.Graph()
    g_1 = ops.Graph()
    with g_0.as_default():
      x = constant_op.constant(1)
    with g_1.as_default():
      y = constant_op.constant(2)
      z = y * 2
      with self.assertRaisesRegexp(ValueError, "must be from the same graph"):
        z.op._update_input(0, x)  # pylint: disable=protected-access

  def testUpdateInputTypeError(self):
    g = ops.Graph()
    with g.as_default():
      w = constant_op.constant(0)
      x = constant_op.constant("")
      y = constant_op.constant(1)
      z = y + w
      z.op._update_input(0, x)  # pylint: disable=protected-access
    with session.Session(graph=g) as sess:
      with self.assertRaisesRegexp(
          errors.InvalidArgumentError,
          "Input 0 of node add was passed string from Const_1:0 incompatible "
          "with expected int32"):
        sess.run(z)

  def testUpdateInputShapeError(self):
    # C-API throws the error differently.
    if ops._USE_C_API:
      return
    g = ops.Graph()
    with g.as_default():
      w = constant_op.constant(2, shape=[3, 1])
      x = constant_op.constant(0, shape=[3, 1])
      y = constant_op.constant(1, shape=[2, 2])
      z = w + x
      z.op._update_input(0, y)  # pylint: disable=protected-access

    with session.Session(graph=g) as sess:
      with self.assertRaisesRegexp(errors.InvalidArgumentError,
                                   r"Incompatible shapes: \[2,2\] vs. \[3,1\]"):
        sess.run(z)

  def testUpdateInputShapeErrorC(self):
    if not ops._USE_C_API:
      return
    g = ops.Graph()
    with g.as_default():
      w = constant_op.constant(2, shape=[3, 1])
      x = constant_op.constant(0, shape=[3, 1])
      y = constant_op.constant(1, shape=[2, 2])
      z = w + x
    with self.assertRaisesRegexp(
        errors.InvalidArgumentError,
        r"Cannot update edge, incompatible shapes: \[2,2\] and \[3,1\]"):
      z.op._update_input(0, y)  # pylint: disable=protected-access

  def testUpdateInputOutOfRange(self):
    # C-API throws the error differently.
    if ops._USE_C_API: return
    g = ops.Graph()
    with g.as_default():
      x = constant_op.constant(1)
    with self.assertRaisesRegexp(IndexError, "list index out of range"):
      x.op._update_input(1, x)  # pylint: disable=protected-access

  def testUpdateInputOutOfRangeC(self):
    # C-API throws the error differently.
    if not ops._USE_C_API: return
    g = ops.Graph()
    with g.as_default():
      x = constant_op.constant(1)
    with self.assertRaisesRegexp(
        errors.OutOfRangeError,
        r"Cannot update edge. Input index \[1\] is greater than the number of "
        r"total inputs \[0\]."
    ):
      x.op._update_input(1, x)  # pylint: disable=protected-access

  def testOpDef(self):
    x = constant_op.constant(0)
    y = constant_op.constant(1)
    z = x + y

    # Pure Python mode doesn't create OpDefs for constants
    if ops._USE_C_API:
      self.assertEqual(x.op.op_def.name, "Const")
      self.assertEqual(len(x.op.op_def.input_arg), 0)
      self.assertEqual(len(x.op.op_def.output_arg), 1)

    self.assertEqual(z.op.op_def.name, "Add")
    self.assertEqual(len(z.op.op_def.input_arg), 2)
    self.assertEqual(len(z.op.op_def.output_arg), 1)

  def testInputFromDifferentGraphError(self):
    g_0 = ops.Graph()
    g_1 = ops.Graph()
    with g_0.as_default():
      x = constant_op.constant(1)
    with g_1.as_default():
      y = constant_op.constant(2)
      with self.assertRaisesRegexp(ValueError, "must be from the same graph"):
        y * x  # pylint: disable=pointless-statement


@test_util.with_c_api
class CreateOpTest(test_util.TensorFlowTestCase):

  def testNodeDefArgs(self):
    g = ops.Graph()
    op1 = g.create_op("FloatOutput", [], [dtypes.float32], None, name="myop1")
    with g.device("/device:GPU:0"):
      op2 = g.create_op(
          "FloatOutputStringOutput", [], [dtypes.float32, dtypes.string], None,
          name="myop2")
    op3 = g.create_op(
        "Foo3",
        [list(op1.values())[0], list(op2.values())[1], list(op2.values())[0]],
        [dtypes.float32, dtypes.int32],
        None,
        name="myop3")
    self.assertDeviceEqual(None, op1.device)
    self.assertDeviceEqual("/device:GPU:0", op2.device)
    self.assertDeviceEqual(None, op3.device)
    self.assertProtoEquals("name:'myop1' op:'FloatOutput'", op1.node_def)
    self.assertProtoEquals(
        "name:'myop2' op:'FloatOutputStringOutput' device:'/device:GPU:0'",
        op2.node_def)
    self.assertProtoEquals(
        "name:'myop3' input:'myop1' input:'myop2:1' input:'myop2' op:'Foo3'",
        op3.node_def)

  def testReferenceInput(self):
    g = ops.Graph()
    op1 = g.create_op(
        "RefOutputFloatOutput", [], [dtypes.float32_ref, dtypes.float32],
        name="op1")
    self.assertProtoEquals("op:'RefOutputFloatOutput' name:'op1'", op1.node_def)
    ref_t, nonref_t = op1.values()
    # NOTE(mrry): Must specify input_types to preserve ref-typed input.
    op2 = g.create_op(
        "RefInputFloatInput", [ref_t, nonref_t], [],
        input_types=[dtypes.float32_ref, dtypes.float32],
        name="op2")
    self.assertProtoEquals(
        "op:'RefInputFloatInput' name:'op2' input:'op1' input:'op1:1'",
        op2.node_def)
    op3 = g.create_op("TwoFloatInputs", [ref_t, nonref_t], [], name="op3")
    self.assertProtoEquals(
        "op:'TwoFloatInputs' name:'op3' input:'op1' input:'op1:1'",
        op3.node_def)

  def testFinalized(self):
    g = ops.Graph()
    g.finalize()
    with self.assertRaises(RuntimeError):
      g.create_op("FloatOutput", [], [dtypes.float32], None, name="myop1")

    # Test unfinalize.
    g._unsafe_unfinalize()
    g.create_op("FloatOutput", [], [dtypes.float32], None, name="myop1")


# NOTE(skyewm): these cases test the private Graph._create_op_from_tf_operation
# method. Arguably we should only test the public APIs that depend on this
# method. However, this logic is complex and tricky, and it can be difficult to
# ascertain if we have adequate coverage (e.g. a graph may run successfully if
# the control flow context isn't set properly, but a more complicated use case
# that might not be obvious to test will fail). Thus we instead explicitly test
# the low-level behavior.
@test_util.with_c_api
class CreateOpFromTFOperationTest(test_util.TensorFlowTestCase):

  def testBasic(self):
    g = ops.Graph()
    with g.as_default():
      x = test_ops.int_output()
      if ops._USE_C_API:
        c_op = ops._create_c_op(
            g, ops._NodeDef("IntInputIntOutput", "myop"), [x], [])
        op = g._create_op_from_tf_operation(c_op)
      else:
        # Test pure-Python version to make sure C API has same behavior.
        op = test_ops.int_input_int_output(x, name="myop").op

    self.assertEqual(op.name, "myop")
    self.assertEqual(op.type, "IntInputIntOutput")
    self.assertEqual(len(op.outputs), 1)
    self.assertEqual(list(op.inputs), [x])
    self.assertEqual(op.control_inputs, [])
    self.assertEqual(op.graph, g)
    self.assertEqual(x.consumers(), [op])
    self.assertIsNotNone(op.traceback)
    self.assertEqual(g.get_operation_by_name("myop"), op)
    self.assertEqual(g.get_tensor_by_name("myop:0"), op.outputs[0])

  def testCond(self):
    g = ops.Graph()
    with g.as_default():
      x = test_ops.int_output()

      def true_fn():
        if ops._USE_C_API:
          c_op = ops._create_c_op(ops.get_default_graph(),
                                  ops._NodeDef("IntInput", "cond/myop"), [x],
                                  [])
          ops.get_default_graph()._create_op_from_tf_operation(c_op)
        else:
          # Test pure-Python version to make sure C API has same behavior.
          test_ops.int_input(x, name="myop")
        return x

      control_flow_ops.cond(x < 10, true_fn, lambda: x)

    op = g.get_operation_by_name("cond/myop")
    self.assertIsNotNone(op)
    self.assertEqual(op.name, "cond/myop")
    self.assertEqual(op.type, "IntInput")
    self.assertEqual(op.outputs, [])
    op_input = op.inputs[0].op
    self.assertEqual(op_input.type, "Switch")
    self.assertEqual(op_input.inputs[0], x)
    self.assertEqual(op.graph, g)
    # pylint: disable=protected-access
    self.assertIsNotNone(op._get_control_flow_context())
    self.assertEqual(op._get_control_flow_context().name,
                     "cond/cond_text")
    # pylint: enable=protected-access

  def testWhileLoop(self):
    g = ops.Graph()
    with g.as_default():
      x = test_ops.int_output()

      def body(i):
        if ops._USE_C_API:
          c_op = ops._create_c_op(ops.get_default_graph(),
                                  ops._NodeDef("IntInput", "myloop/myop"), [x],
                                  [])
          ops.get_default_graph()._create_op_from_tf_operation(c_op)
        else:
          # Test pure-Python version to make sure C API has same behavior.
          test_ops.int_input(x, name="myop")
        return i

      control_flow_ops.while_loop(lambda i: i < 10, body, [0], name="myloop")

    op = g.get_operation_by_name("myloop/myop")
    self.assertIsNotNone(op)
    self.assertEqual(op.name, "myloop/myop")
    self.assertEqual(op.type, "IntInput")
    self.assertEqual(op.outputs, [])
    op_input = op.inputs[0].op
    self.assertEqual(op_input.type, "Enter")
    self.assertEqual(list(op_input.inputs), [x])
    self.assertEqual(op.graph, g)
    # pylint: disable=protected-access
    self.assertIsNotNone(op._get_control_flow_context())
    self.assertEqual(op._get_control_flow_context().name,
                     "myloop/while_context")
    # pylint: enable=protected-access

  def testWhileLoopWithInternalControlDep(self):
    g = ops.Graph()
    with g.as_default():
      x = test_ops.int_output()

      def body(i):
        c = constant_op.constant(1.0, name="c")
        if ops._USE_C_API:
          c_op = ops._create_c_op(ops.get_default_graph(),
                                  ops._NodeDef("IntInput", "myloop/myop"), [x],
                                  [])
          with ops.control_dependencies([c]):
            ops.get_default_graph()._create_op_from_tf_operation(c_op)
        else:
          with ops.control_dependencies([c]):
            test_ops.int_input(x, name="myop")
        return i

      control_flow_ops.while_loop(lambda i: i < 10, body, [0], name="myloop")

    op = g.get_operation_by_name("myloop/myop")
    self.assertIsNotNone(op)
    c = g.get_operation_by_name("myloop/c")
    self.assertIsNotNone(c)
    # Internal control dep is preserved
    self.assertEqual(op.control_inputs, [c])

  def testWhileLoopWithExternalControlDep(self):
    # TODO(skyewm): enable once ControlFlowContext._RemoveExternalControlEdges
    # works with C API enabled
    if ops._USE_C_API: self.skipTest("Not yet implemented with C API enabled")

    g = ops.Graph()
    with g.as_default():
      x = test_ops.int_output()
      c = constant_op.constant(1.0)

      def body(i):
        if ops._USE_C_API:
          c_op = ops._create_c_op(ops.get_default_graph(),
                                  ops._NodeDef("IntInput", "myloop/myop"), [x],
                                  [])
          with ops.control_dependencies([c]):
            ops.get_default_graph()._create_op_from_tf_operation(c_op)
        else:
          with ops.control_dependencies([c]):
            test_ops.int_input(x, name="myop")
        return i

      control_flow_ops.while_loop(lambda i: i < 10, body, [0], name="myloop")

    op = g.get_operation_by_name("myloop/myop")
    self.assertIsNotNone(op)
    self.assertEqual(len(op.control_inputs), 1)
    # External control dep is removed and replaced with internal control dep
    self.assertNotEqual(op.control_inputs[0], c.op)
    self.assertIsNotNone(op.control_inputs[0]._get_control_flow_context())


@test_util.with_c_api
class ApplyOpTest(test_util.TensorFlowTestCase):

  def testNodeDefArgs(self):
    g = ops.Graph()
    t1 = _apply_op(g, "FloatOutput", [], [dtypes.float32], name="myop1")
    with g.device("/device:GPU:0"):
      t2 = _apply_op(
          g, "TwoIntOutputs", [], [dtypes.int32, dtypes.int32], name="myop2")
    t3 = _apply_op(
        g,
        "Foo1", [t1, t2[1], t2[0]], [dtypes.float32, dtypes.int32],
        name="myop3")
    self.assertTrue(isinstance(t1, ops.Tensor))
    self.assertTrue(isinstance(t2, list))
    self.assertTrue(isinstance(t3, list))
    self.assertTrue(isinstance(t3[0], ops.Tensor))
    self.assertEqual("myop1", t1._as_node_def_input())
    self.assertEqual("myop2", t2[0]._as_node_def_input())
    self.assertEqual("myop2:1", t2[1]._as_node_def_input())
    self.assertEqual("myop3", t3[0]._as_node_def_input())
    # Validate that we got the right ops as well
    self.assertProtoEquals("name:'myop1' op:'FloatOutput'", t1.op.node_def)
    self.assertProtoEquals(
        "name:'myop2' op:'TwoIntOutputs' device:'/device:GPU:0'",
        t2[0].op.node_def)
    self.assertProtoEquals(
        "name:'myop3' input:'myop1' input:'myop2:1' input:'myop2' op:'Foo1'",
        t3[0].op.node_def)

  def testReferenceInput(self):
    g = ops.Graph()
    ref_t, nonref_t = _apply_op(
        g, "RefOutputFloatOutput", [], [dtypes.float32_ref, dtypes.float32],
        name="op1")
    self.assertProtoEquals("op:'RefOutputFloatOutput' name:'op1'",
                           ref_t.op.node_def)
    # NOTE(mrry): Must specify input_types to preserve ref-typed input.
    out_2 = _apply_op(
        g,
        "RefInputFloatInputIntOutput", [ref_t, nonref_t], [dtypes.int32],
        input_types=[dtypes.float32_ref, dtypes.float32],
        name="op2")
    self.assertProtoEquals(
        "op:'RefInputFloatInputIntOutput' name:'op2' input:'op1' input:'op1:1'",
        out_2.op.node_def)
    out_3 = _apply_op(
        g, "TwoFloatInputsIntOutput", [ref_t, nonref_t], [dtypes.int32],
        name="op3")
    self.assertProtoEquals(
        "op:'TwoFloatInputsIntOutput' name:'op3' input:'op1' input:'op1:1'",
        out_3.op.node_def)


@test_util.with_c_api
class NameStackTest(test_util.TensorFlowTestCase):

  def testBasics(self):
    g = ops.Graph()
    self.assertEqual("foo", g.unique_name("foo", mark_as_used=False))
    self.assertEqual("foo", g.unique_name("foo", mark_as_used=False))
    self.assertEqual("foo", g.unique_name("foo"))
    self.assertEqual("foo_1", g.unique_name("foo", mark_as_used=False))
    self.assertEqual("foo_1", g.unique_name("foo"))
    self.assertEqual("foo_2", g.unique_name("foo", mark_as_used=False))
    self.assertEqual("foo_2", g.unique_name("foo"))
    self.assertEqual("foo_1_1", g.unique_name("foo_1", mark_as_used=False))
    self.assertEqual("foo_1_1", g.unique_name("foo_1"))
    self.assertEqual("foo_1_2", g.unique_name("foo_1", mark_as_used=False))
    self.assertEqual("foo_1_2", g.unique_name("foo_1"))
    self.assertEqual("foo_1_2_1", g.unique_name("foo_1_2", mark_as_used=False))
    self.assertEqual("foo_1_2_1", g.unique_name("foo_1_2"))
    with g.name_scope("bar"):
      self.assertEqual("bar/foo", g.unique_name("foo", mark_as_used=False))
      self.assertEqual("bar/foo", g.unique_name("foo"))
      self.assertEqual("bar/foo_1", g.unique_name("foo", mark_as_used=False))
      self.assertEqual("bar/foo_1", g.unique_name("foo"))
      with g.name_scope(None):
        self.assertEqual("foo_3", g.unique_name("foo", mark_as_used=False))
        self.assertEqual("foo_3", g.unique_name("foo"))
      with g.name_scope("baz"):
        self.assertEqual(
            "bar/baz/foo", g.unique_name(
                "foo", mark_as_used=False))
        self.assertEqual("bar/baz/foo", g.unique_name("foo"))
        self.assertEqual(
            "bar/baz/foo_1", g.unique_name(
                "foo", mark_as_used=False))
        self.assertEqual("bar/baz/foo_1", g.unique_name("foo"))
      with g.name_scope("baz"):
        self.assertEqual(
            "bar/baz_1/foo", g.unique_name(
                "foo", mark_as_used=False))
        self.assertEqual("bar/baz_1/foo", g.unique_name("foo"))
        self.assertEqual(
            "bar/baz_1/foo_1", g.unique_name(
                "foo", mark_as_used=False))
        self.assertEqual("bar/baz_1/foo_1", g.unique_name("foo"))
    with g.name_scope("quux"):
      self.assertEqual("quux/foo", g.unique_name("foo", mark_as_used=False))
      self.assertEqual("quux/foo", g.unique_name("foo"))
    with g.name_scope("bar"):
      with g.name_scope("baz"):
        self.assertEqual(
            "bar_1/baz/foo", g.unique_name(
                "foo", mark_as_used=False))
        self.assertEqual("bar_1/baz/foo", g.unique_name("foo"))
    self.assertEqual("foo_4", g.unique_name("foo", mark_as_used=False))
    self.assertEqual("foo_4", g.unique_name("foo"))
    self.assertEqual("bar_2", g.unique_name("bar", mark_as_used=False))
    self.assertEqual("bar_2", g.unique_name("bar"))

  def testNameAndVariableScope(self):
    with self.test_session() as sess:
      with sess.graph.name_scope("l0"):
        with variable_scope.variable_scope("l1"):
          with sess.graph.name_scope("l1") as scope:
            self.assertEqual("l0/l1/l1/", scope)
            self.assertEqual(
                "l0/l1/l1/foo",
                sess.graph.unique_name(
                    "foo", mark_as_used=False))
            self.assertEqual("l0/l1/l1/foo", sess.graph.unique_name("foo"))
          with sess.graph.name_scope("l2") as scope:
            self.assertEqual("l0/l1/l2/", scope)
            self.assertEqual(
                "l0/l1/l2/foo",
                sess.graph.unique_name(
                    "foo", mark_as_used=False))
            self.assertEqual("l0/l1/l2/foo", sess.graph.unique_name("foo"))

  def testOutOfOrderUniqueName(self):
    g = ops.Graph()
    self.assertEqual("foo_2", g.unique_name("foo_2"))
    self.assertEqual("foo", g.unique_name("foo"))
    self.assertEqual("foo_1", g.unique_name("foo"))
    self.assertEqual("foo_3", g.unique_name("foo"))

  def testInvalidNameRaisesError(self):
    g = ops.Graph()
    with g.name_scope(""):  # Should not raise
      pass
    with g.name_scope("foo/"):  # Should not raise
      with g.name_scope("_bar"):  # Should not raise
        pass
    with self.assertRaises(ValueError):
      with g.name_scope("foo:0"):
        pass
    with self.assertRaises(ValueError):
      with g.name_scope("_bar"):
        pass


@test_util.with_c_api
class NameTest(test_util.TensorFlowTestCase):

  def testGenerateName(self):
    g = ops.Graph()
    op0 = g.create_op("TwoFloatOutputs", [], [dtypes.float32, dtypes.float32])
    self.assertEqual("TwoFloatOutputs", op0.name)
    self.assertEqual("TwoFloatOutputs:0", op0.outputs[0].name)
    self.assertEqual("TwoFloatOutputs:1", op0.outputs[1].name)

    op1 = g.create_op("FloatOutput", [], [dtypes.float32])
    self.assertEqual("FloatOutput", op1.name)
    self.assertEqual("FloatOutput:0", op1.outputs[0].name)

    op2 = g.create_op("FloatOutput", [], [dtypes.float32])
    self.assertEqual("FloatOutput_1", op2.name)
    self.assertEqual("FloatOutput_1:0", op2.outputs[0].name)

    op3 = g.create_op("FloatOutput", [], [dtypes.float32], name="my_op")
    self.assertEqual("my_op", op3.name)
    self.assertEqual("my_op:0", op3.outputs[0].name)

  def testNameScope(self):
    g = ops.Graph()

    with g.name_scope("foo") as foo:
      self.assertEqual("foo/", foo)
      with g.name_scope("foo2") as foo2:
        self.assertEqual("foo/foo2/", foo2)
      with g.name_scope(None) as empty1:
        self.assertEqual("", empty1)
        with g.name_scope("foo3") as foo3:
          self.assertEqual("foo3/", foo3)
      with g.name_scope("") as empty2:
        self.assertEqual("", empty2)

    self.assertEqual("FloatOutput",
                     g.create_op("FloatOutput", [], [dtypes.float32]).name)
    with g.name_scope("bar") as scope:
      self.assertEqual("bar/FloatOutput",
                       g.create_op("FloatOutput", [], [dtypes.float32]).name)
      self.assertEqual("bar/FloatOutput_1",
                       g.create_op("FloatOutput", [], [dtypes.float32]).name)
      # If you use the value from "with .. as", that values is used as-is.
      self.assertEqual(
          "bar", g.create_op(
              "FloatOutput", [], [dtypes.float32], name=scope).name)
    with g.name_scope("baz") as scope:
      with g.name_scope("quux"):
        self.assertEqual("baz/quux/FloatOutput",
                         g.create_op("FloatOutput", [], [dtypes.float32]).name)
      # If you use the value from the enclosing "with .. as", nothing is pushed.
      with g.name_scope(scope):
        self.assertEqual("baz/FloatOutput",
                         g.create_op("FloatOutput", [], [dtypes.float32]).name)
        self.assertEqual(
            "baz", g.create_op(
                "FloatOutput", [], [dtypes.float32], name=scope).name)
        self.assertEqual(
            "trailing",
            g.create_op(
                "FloatOutput", [], [dtypes.float32], name="trailing/").name)
    with g.name_scope("bar"):
      self.assertEqual("bar_1/FloatOutput",
                       g.create_op("FloatOutput", [], [dtypes.float32]).name)
    with g.name_scope("bar/"):
      self.assertEqual("bar/FloatOutput_2",
                       g.create_op("FloatOutput", [], [dtypes.float32]).name)


@test_util.with_c_api
class DeviceTest(test_util.TensorFlowTestCase):

  def testNoDevice(self):
    g = ops.Graph()
    op = g.create_op("FloatOutput", [], [dtypes.float32])
    self.assertDeviceEqual(None, op.device)
    gd = g.as_graph_def()
    self.assertProtoEqualsVersion("""
      node { name: "FloatOutput" op: "FloatOutput" }
    """, gd)

  def testDevicePartialString(self):
    g = ops.Graph()
    with g.device("/job:worker/replica:2"):
      g.create_op("FloatOutput", [], [dtypes.float32])
    gd = g.as_graph_def()
    self.assertProtoEqualsVersion("""
      node { name: "FloatOutput" op: "FloatOutput"
             device: "/job:worker/replica:2" }
    """, gd)

  def testDeviceFull(self):
    g = ops.Graph()
    with g.device(
        pydev.DeviceSpec(
            job="worker", replica=2, task=0, device_type="CPU",
            device_index=3)):
      g.create_op("FloatOutput", [], [dtypes.float32])
    gd = g.as_graph_def()
    self.assertProtoEqualsVersion("""
      node { name: "FloatOutput" op: "FloatOutput"
             device: "/job:worker/replica:2/task:0/device:CPU:3" }
    """, gd)

  def testNesting(self):
    g = ops.Graph()
    with g.device("/job:worker/replica:2"):
      g.create_op("FloatOutput", [], [dtypes.float32])
      with g.device("/job:worker/replica:3/task:0"):
        g.create_op("FloatOutput", [], [dtypes.float32])
      g.create_op("FloatOutput", [], [dtypes.float32])
    gd = g.as_graph_def()
    self.assertProtoEqualsVersion("""
      node { name: "FloatOutput" op: "FloatOutput"
             device: "/job:worker/replica:2" }
      node { name: "FloatOutput_1" op: "FloatOutput"
             device: "/job:worker/replica:3/task:0" }
      node { name: "FloatOutput_2" op: "FloatOutput"
             device: "/job:worker/replica:2" }
    """, gd)

  def testNestingString(self):
    g = ops.Graph()
    with g.device("/job:worker/replica:2"):
      g.create_op("FloatOutput", [], [dtypes.float32])
      with g.device("/job:worker/replica:3/task:0"):
        g.create_op("FloatOutput", [], [dtypes.float32])
      g.create_op("FloatOutput", [], [dtypes.float32])
    gd = g.as_graph_def()
    self.assertProtoEqualsVersion("""
      node { name: "FloatOutput" op: "FloatOutput"
             device: "/job:worker/replica:2" }
      node { name: "FloatOutput_1" op: "FloatOutput"
             device: "/job:worker/replica:3/task:0" }
      node { name: "FloatOutput_2" op: "FloatOutput"
             device: "/job:worker/replica:2" }
    """, gd)

  def testNestingOverrideGpuCpu(self):
    g = ops.Graph()
    with g.device("/job:worker/replica:2/device:CPU:1"):
      g.create_op("FloatOutput", [], [dtypes.float32])
      with g.device("/job:worker/replica:2/device:GPU:2"):
        g.create_op("FloatOutput", [], [dtypes.float32])
      g.create_op("FloatOutput", [], [dtypes.float32])
    gd = g.as_graph_def()
    self.assertProtoEqualsVersion("""
      node { name: "FloatOutput" op: "FloatOutput"
             device: "/job:worker/replica:2/device:CPU:1"  }
      node { name: "FloatOutput_1" op: "FloatOutput"
             device: "/job:worker/replica:2/device:GPU:2" }
      node { name: "FloatOutput_2" op: "FloatOutput"
             device: "/job:worker/replica:2/device:CPU:1" }
    """, gd)

  def testNestingWithMergeDeviceFunction(self):
    g = ops.Graph()

    with g.device(pydev.merge_device("/device:GPU:0")):
      g.create_op("FloatOutput", [], [dtypes.float32])
      with g.device(pydev.merge_device("/job:worker")):
        g.create_op("FloatOutput", [], [dtypes.float32])
        with g.device(pydev.merge_device("/device:CPU:0")):
          g.create_op("FloatOutput", [], [dtypes.float32])
          with g.device(pydev.merge_device("/job:ps")):
            g.create_op("FloatOutput", [], [dtypes.float32])
            with g.device(pydev.merge_device(None)):
              g.create_op("FloatOutput", [], [dtypes.float32])

    gd = g.as_graph_def()
    self.assertProtoEqualsVersion("""
      node { name: "FloatOutput" op: "FloatOutput"
             device: "/device:GPU:0" }
      node { name: "FloatOutput_1" op: "FloatOutput"
             device: "/job:worker/device:GPU:0" }
      node { name: "FloatOutput_2" op: "FloatOutput"
             device: "/job:worker/device:CPU:0" }
      node { name: "FloatOutput_3" op: "FloatOutput"
             device: "/job:ps/device:CPU:0" }
      node { name: "FloatOutput_4" op: "FloatOutput"
             device: "/job:ps/device:CPU:0" }
    """, gd)

  def testNestingWithDeviceStrings(self):
    g = ops.Graph()

    with g.device("/device:GPU:0"):
      g.create_op("FloatOutput", [], [dtypes.float32])
      with g.device("/job:worker"):
        g.create_op("FloatOutput", [], [dtypes.float32])
        with g.device("/device:CPU:0"):
          g.create_op("FloatOutput", [], [dtypes.float32])
          with g.device("/job:ps"):
            g.create_op("FloatOutput", [], [dtypes.float32])
            with g.device(""):
              g.create_op("FloatOutput", [], [dtypes.float32])

    gd = g.as_graph_def()
    self.assertProtoEqualsVersion("""
      node { name: "FloatOutput" op: "FloatOutput"
             device: "/device:GPU:0" }
      node { name: "FloatOutput_1" op: "FloatOutput"
             device: "/job:worker/device:GPU:0" }
      node { name: "FloatOutput_2" op: "FloatOutput"
             device: "/job:worker/device:CPU:0" }
      node { name: "FloatOutput_3" op: "FloatOutput"
             device: "/job:ps/device:CPU:0" }
      node { name: "FloatOutput_4" op: "FloatOutput"
             device: "/job:ps/device:CPU:0" }
    """, gd)

  def testNestingWithDeviceStringWildcard(self):
    g = ops.Graph()

    with g.device("/device:GPU:7"):
      g.create_op("FloatOutput", [], [dtypes.float32])
      with g.device("/device:GPU:*"):
        g.create_op("FloatOutput", [], [dtypes.float32])

    with g.device("/device:CPU:*"):
      g.create_op("FloatOutput", [], [dtypes.float32])
      with g.device("/device:CPU:5"):
        g.create_op("FloatOutput", [], [dtypes.float32])

    gd = g.as_graph_def()
    self.assertProtoEqualsVersion("""
      node { name: "FloatOutput" op: "FloatOutput"
             device: "/device:GPU:7" }
      node { name: "FloatOutput_1" op: "FloatOutput"
             device: "/device:GPU:7" }
      node { name: "FloatOutput_2" op: "FloatOutput"
             device: "/device:CPU:*" }
      node { name: "FloatOutput_3" op: "FloatOutput"
             device: "/device:CPU:5" }
    """, gd)

  def testNoneClearsDefault(self):
    g = ops.Graph()
    with g.device("/job:worker/replica:2/device:CPU:1"):
      g.create_op("FloatOutput", [], [dtypes.float32])
      with g.device(None):
        g.create_op("FloatOutput", [], [dtypes.float32])
      g.create_op("FloatOutput", [], [dtypes.float32])
    gd = g.as_graph_def()
    self.assertProtoEqualsVersion("""
      node { name: "FloatOutput" op: "FloatOutput"
             device: "/job:worker/replica:2/device:CPU:1" }
      node { name: "FloatOutput_1" op: "FloatOutput" }
      node { name: "FloatOutput_2" op: "FloatOutput"
             device: "/job:worker/replica:2/device:CPU:1" }
    """, gd)

  def testNoneIgnoresOuterDeviceFunction(self):
    g = ops.Graph()
    with g.device(lambda op: "/job:worker/replica:2/device:CPU:1"):
      g.create_op("FloatOutput", [], [dtypes.float32])
      with g.device(None):
        g.create_op("FloatOutput", [], [dtypes.float32])
      g.create_op("FloatOutput", [], [dtypes.float32])
    gd = g.as_graph_def()
    self.assertProtoEqualsVersion("""
      node { name: "FloatOutput" op: "FloatOutput"
             device: "/job:worker/replica:2/device:CPU:1" }
      node { name: "FloatOutput_1" op: "FloatOutput" }
      node { name: "FloatOutput_2" op: "FloatOutput"
             device: "/job:worker/replica:2/device:CPU:1" }
    """, gd)

  def _overwritingDeviceFunction(self, unused_op):
    # This device function unconditionally overwrites the device of ops.
    #
    # NOTE(mrry): Writing device functions like this is not
    # recommended. Instead, in most cases you should use
    # `pydev.merge_device("/job:ps")` or simply `"/job:ps"` as the
    # argument to `tf.device()` and the device component will be merged in.
    return "/job:overwrite"

  def testOverwritingBehavior(self):
    g = ops.Graph()
    with g.device(self._overwritingDeviceFunction):
      g.create_op("FloatOutput", [], [dtypes.float32])
      with g.device("/job:ps"):  # Will be overwritten.
        g.create_op("FloatOutput", [], [dtypes.float32])
      with g.device(pydev.merge_device("/job:ps")):  # Will be overwritten.
        g.create_op("FloatOutput", [], [dtypes.float32])
      with g.device(None):  # Disables overwriting device function
        with g.device("/job:ps"):
          g.create_op("FloatOutput", [], [dtypes.float32])
      with g.device(None):  # Disables overwriting device function
        with g.device(pydev.merge_device("/job:ps")):
          g.create_op("FloatOutput", [], [dtypes.float32])
    gd = g.as_graph_def()
    self.assertProtoEqualsVersion("""
      node { name: "FloatOutput" op: "FloatOutput"
             device: "/job:overwrite" }
      node { name: "FloatOutput_1" op: "FloatOutput"
             device: "/job:overwrite" }
      node { name: "FloatOutput_2" op: "FloatOutput"
             device: "/job:overwrite" }
      node { name: "FloatOutput_3" op: "FloatOutput"
             device: "/job:ps" }
      node { name: "FloatOutput_4" op: "FloatOutput"
             device: "/job:ps" }
    """, gd)


@test_util.with_c_api
class ObjectWithName(object):

  def __init__(self, name):
    self._name = name

  @property
  def name(self):
    return self._name


@test_util.with_c_api
class CollectionTest(test_util.TensorFlowTestCase):

  def test_get_collections(self):
    g = ops.Graph()
    self.assertSequenceEqual(g.collections, [])
    g.add_to_collection("key", 12)
    g.add_to_collection("key", 15)
    self.assertSequenceEqual(g.collections, ["key"])
    g.add_to_collection("other", "foo")
    self.assertSequenceEqual(sorted(g.collections), ["key", "other"])

  def test_add_to_collection(self):
    g = ops.Graph()
    g.add_to_collection("key", 12)
    g.add_to_collection("other", "foo")
    g.add_to_collection("key", 34)

    # Note that only blank1 is returned.
    g.add_to_collection("blah", 27)
    blank1 = ObjectWithName("prefix/foo")
    g.add_to_collection("blah", blank1)
    blank2 = ObjectWithName("junk/foo")
    g.add_to_collection("blah", blank2)

    self.assertEqual([12, 34], g.get_collection("key"))
    self.assertEqual([], g.get_collection("nothing"))
    self.assertEqual([27, blank1, blank2], g.get_collection("blah"))
    self.assertEqual([blank1], g.get_collection("blah", "prefix"))
    self.assertEqual([blank1], g.get_collection("blah", ".*x"))

    # Make sure that get_collection() returns a first-level
    # copy of the collection, while get_collection_ref() returns
    # the original list.
    other_collection_snapshot = g.get_collection("other")
    other_collection_ref = g.get_collection_ref("other")
    self.assertEqual(["foo"], other_collection_snapshot)
    self.assertEqual(["foo"], other_collection_ref)
    g.add_to_collection("other", "bar")
    self.assertEqual(["foo"], other_collection_snapshot)
    self.assertEqual(["foo", "bar"], other_collection_ref)
    self.assertEqual(["foo", "bar"], g.get_collection("other"))
    self.assertTrue(other_collection_ref is g.get_collection_ref("other"))

    # Verify that getting an empty collection ref returns a modifiable list.
    empty_coll_ref = g.get_collection_ref("empty")
    self.assertEqual([], empty_coll_ref)
    empty_coll = g.get_collection("empty")
    self.assertEqual([], empty_coll)
    self.assertFalse(empty_coll is empty_coll_ref)
    empty_coll_ref2 = g.get_collection_ref("empty")
    self.assertTrue(empty_coll_ref2 is empty_coll_ref)
    # Add to the collection.
    empty_coll_ref.append("something")
    self.assertEqual(["something"], empty_coll_ref)
    self.assertEqual(["something"], empty_coll_ref2)
    self.assertEqual([], empty_coll)
    self.assertEqual(["something"], g.get_collection("empty"))
    empty_coll_ref3 = g.get_collection_ref("empty")
    self.assertTrue(empty_coll_ref3 is empty_coll_ref)

  def test_add_to_collections_uniquify(self):
    g = ops.Graph()
    g.add_to_collections([1, 2, 1], "key")
    # Make sure "key" is not added twice
    self.assertEqual(["key"], g.get_collection(1))

  def test_add_to_collections_from_list(self):
    g = ops.Graph()
    g.add_to_collections(["abc", "123"], "key")
    self.assertEqual(["key"], g.get_collection("abc"))
    self.assertEqual(["key"], g.get_collection("123"))

  def test_add_to_collections_from_tuple(self):
    g = ops.Graph()
    g.add_to_collections(("abc", "123"), "key")
    self.assertEqual(["key"], g.get_collection("abc"))
    self.assertEqual(["key"], g.get_collection("123"))

  def test_add_to_collections_from_generator(self):
    g = ops.Graph()

    def generator():
      yield "abc"
      yield "123"

    g.add_to_collections(generator(), "key")
    self.assertEqual(["key"], g.get_collection("abc"))
    self.assertEqual(["key"], g.get_collection("123"))

  def test_add_to_collections_from_set(self):
    g = ops.Graph()
    g.add_to_collections(set(["abc", "123"]), "key")
    self.assertEqual(["key"], g.get_collection("abc"))
    self.assertEqual(["key"], g.get_collection("123"))

  def test_add_to_collections_from_string(self):
    g = ops.Graph()
    g.add_to_collections("abc", "key")
    self.assertEqual(["key"], g.get_collection("abc"))

  def test_default_graph(self):
    with ops.Graph().as_default():
      ops.add_to_collection("key", 90)
      ops.add_to_collection("key", 100)
      # Collections are ordered.
      self.assertEqual([90, 100], ops.get_collection("key"))


ops.NotDifferentiable("FloatOutput")


@ops.RegisterGradient("CopyOp")
def _CopyGrad(op, x_grad):  # pylint: disable=invalid-name
  _ = op
  return x_grad


@ops.RegisterGradient("copy_override")
def _CopyOverrideGrad(op, x_grad):  # pylint: disable=invalid-name
  _ = op
  return x_grad


@test_util.with_c_api
class RegistrationTest(test_util.TensorFlowTestCase):

  def testRegisterGradients(self):
    x = test_ops.float_output()
    y = test_ops.copy_op(x)
    fn = ops.get_gradient_function(y.op)
    self.assertEqual(_CopyGrad, fn)

  def testOverrideGradients(self):
    g = ops.Graph()
    with g.as_default():
      x = test_ops.float_output()
      with g.gradient_override_map({"CopyOp": "copy_override"}):
        y = test_ops.copy_op(x)
      fn = ops.get_gradient_function(y.op)
      self.assertEqual(_CopyOverrideGrad, fn)

  def testNonExistentOverride(self):
    g = ops.Graph()
    with g.as_default():
      x = test_ops.float_output()
      with g.gradient_override_map({"CopyOp": "unknown_override"}):
        y = test_ops.copy_op(x)
      with self.assertRaisesRegexp(LookupError, "unknown_override"):
        ops.get_gradient_function(y.op)


@test_util.with_c_api
class ComparisonTest(test_util.TensorFlowTestCase):

  def testMembershipAllowed(self):
    g = ops.Graph()
    t1 = _apply_op(g, "FloatOutput", [], [dtypes.float32], name="myop1")
    t2 = _apply_op(g, "FloatOutput", [], [dtypes.float32], name="myop2")
    self.assertTrue(isinstance(t1, ops.Tensor))
    self.assertTrue(isinstance(t2, ops.Tensor))
    self.assertTrue(t1 in [t1])
    self.assertTrue(t1 not in [t2])


@test_util.with_c_api
class ControlDependenciesTest(test_util.TensorFlowTestCase):

  @test_util.enable_c_api
  def testBasic(self):
    g = ops.Graph()
    with g.as_default():
      # Creating unregistered ops with _apply_op() doesn't work with the C API
      # TODO(skyewm): address this more consistently. Possible solutions are
      # to use registered ops in all tests, create a way to register ops in
      # Python tests, or conditionally disable the op registration check in
      # the C API.
      a = constant_op.constant(1.0)
      b = constant_op.constant(1.0)
      with g.control_dependencies([a]):
        c = constant_op.constant(1.0)
        d = array_ops.identity(b)
        e = array_ops.identity(c)

    self.assertEqual(c.op.control_inputs, [a.op])
    self.assertEqual(d.op.control_inputs, [a.op])
    # e should be dominated by c.
    self.assertEqual(e.op.control_inputs, [])

  def testBasicWithConversion(self):
    g = ops.Graph()
    a = _apply_op(g, "FloatOutput", [], [dtypes.float32])

    class ConvertibleObj(object):

      def _as_graph_element(self):
        return a

    with g.control_dependencies([ConvertibleObj()]):
      c = _apply_op(g, "FloatOutput", [], [dtypes.float32])

    self.assertEqual(c.op.control_inputs, [a.op])

  def testNested(self):
    g = ops.Graph()
    a_1 = _apply_op(g, "FloatOutput", [], [dtypes.float32])
    a_2 = _apply_op(g, "FloatOutput", [], [dtypes.float32])
    a_3 = _apply_op(g, "FloatOutput", [], [dtypes.float32])
    a_4 = _apply_op(g, "FloatOutput", [], [dtypes.float32])

    with g.control_dependencies([a_1, a_2, a_3, a_4]):
      b_1 = _apply_op(g, "FloatOutput", [], [dtypes.float32])

    with g.control_dependencies([a_1]):
      with g.control_dependencies([a_2]):
        with g.control_dependencies([a_3]):
          with g.control_dependencies([a_4]):
            b_2 = _apply_op(g, "FloatOutput", [], [dtypes.float32])

    self.assertItemsEqual([a_1.op, a_2.op, a_3.op, a_4.op],
                          b_1.op.control_inputs)
    self.assertItemsEqual(b_1.op.control_inputs, b_2.op.control_inputs)

  def testClear(self):
    g = ops.Graph()
    a_1 = _apply_op(g, "FloatOutput", [], [dtypes.float32])
    a_2 = _apply_op(g, "FloatOutput", [], [dtypes.float32])
    a_3 = _apply_op(g, "FloatOutput", [], [dtypes.float32])
    a_4 = _apply_op(g, "FloatOutput", [], [dtypes.float32])

    with g.control_dependencies([a_1]):
      with g.control_dependencies([a_2]):
        with g.control_dependencies(None):
          with g.control_dependencies([a_3]):
            with g.control_dependencies([a_4]):
              # deps [a_3, a_4]
              b_3_4 = _apply_op(g, "FloatOutput", [], [dtypes.float32])
            # deps = [a_3]
            b_3 = _apply_op(g, "FloatOutput", [], [dtypes.float32])
          # deps back to None
          b_none = _apply_op(g, "FloatOutput", [], [dtypes.float32])
        # deps back to [a_1, a_2]
        b_1_2 = _apply_op(g, "FloatOutput", [], [dtypes.float32])
      # deps back to [a_1]
      b_1 = _apply_op(g, "FloatOutput", [], [dtypes.float32])
      with g.control_dependencies(None):
        # deps are None again
        b_none2 = _apply_op(g, "FloatOutput", [], [dtypes.float32])

    self.assertItemsEqual([a_3.op, a_4.op], b_3_4.op.control_inputs)
    self.assertItemsEqual([a_3.op], b_3.op.control_inputs)
    self.assertItemsEqual([], b_none.op.control_inputs)
    self.assertItemsEqual([a_1.op, a_2.op], b_1_2.op.control_inputs)
    self.assertItemsEqual([a_1.op], b_1.op.control_inputs)
    self.assertItemsEqual([], b_none2.op.control_inputs)

  def testComplex(self):
    g = ops.Graph()

    # Usage pattern:
    # * Nodes a_i are constants defined at the outermost scope, and are used
    #   as control inputs for the ith nested scope.
    # * Nodes b_i are defined as Mul(a_3, a_4) at each scope.
    # * Nodes c_i are defined as Mul(a_1, b_1) at each scope.
    # * Nodes d_i are defined as Mul(b_i, c_i) at each scope.
    # * Nodes e_i are defined as Mul(e_i-1, e_i-1) at each scope i > 1.

    a_1 = _apply_op(g, "FloatOutput", [], [dtypes.float32])
    a_2 = _apply_op(g, "FloatOutput", [], [dtypes.float32])
    a_3 = _apply_op(g, "FloatOutput", [], [dtypes.float32])
    a_4 = _apply_op(g, "FloatOutput", [], [dtypes.float32])

    with g.control_dependencies([a_1]):
      b_1 = _apply_op(g, "TwoFloatInputsFloatOutput", [a_3, a_4],
                      [dtypes.float32])
      c_1 = _apply_op(g, "TwoFloatInputsFloatOutput", [a_1, b_1],
                      [dtypes.float32])
      d_1 = _apply_op(g, "TwoFloatInputsFloatOutput", [b_1, c_1],
                      [dtypes.float32])
      e_1 = _apply_op(g, "FloatOutput", [], [dtypes.float32])
      with g.control_dependencies([a_2]):
        b_2 = _apply_op(g, "TwoFloatInputsFloatOutput", [a_3, a_4],
                        [dtypes.float32])
        c_2 = _apply_op(g, "TwoFloatInputsFloatOutput", [a_1, b_1],
                        [dtypes.float32])
        d_2 = _apply_op(g, "TwoFloatInputsFloatOutput", [b_2, c_2],
                        [dtypes.float32])
        e_2 = _apply_op(g, "TwoFloatInputsFloatOutput", [e_1, e_1],
                        [dtypes.float32])
        with g.control_dependencies([a_3]):
          b_3 = _apply_op(g, "TwoFloatInputsFloatOutput", [a_3, a_4],
                          [dtypes.float32])
          c_3 = _apply_op(g, "TwoFloatInputsFloatOutput", [a_1, b_1],
                          [dtypes.float32])
          d_3 = _apply_op(g, "TwoFloatInputsFloatOutput", [b_3, c_3],
                          [dtypes.float32])
          e_3 = _apply_op(g, "TwoFloatInputsFloatOutput", [e_2, e_2],
                          [dtypes.float32])
          with g.control_dependencies([a_4]):
            b_4 = _apply_op(g, "TwoFloatInputsFloatOutput", [a_3, a_4],
                            [dtypes.float32])
            c_4 = _apply_op(g, "TwoFloatInputsFloatOutput", [a_1, b_1],
                            [dtypes.float32])
            d_4 = _apply_op(g, "TwoFloatInputsFloatOutput", [b_4, c_4],
                            [dtypes.float32])
            e_4 = _apply_op(g, "TwoFloatInputsFloatOutput", [e_3, e_3],
                            [dtypes.float32])

    self.assertItemsEqual([a_1.op], b_1.op.control_inputs)
    self.assertItemsEqual([a_1.op, a_2.op], b_2.op.control_inputs)
    self.assertItemsEqual([a_1.op, a_2.op], b_3.op.control_inputs)
    self.assertItemsEqual([a_1.op, a_2.op], b_4.op.control_inputs)

    self.assertItemsEqual([], c_1.op.control_inputs)
    self.assertItemsEqual([a_2.op], c_2.op.control_inputs)
    self.assertItemsEqual([a_2.op, a_3.op], c_3.op.control_inputs)
    self.assertItemsEqual([a_2.op, a_3.op, a_4.op], c_4.op.control_inputs)

    self.assertItemsEqual([], d_1.op.control_inputs)
    self.assertItemsEqual([], d_2.op.control_inputs)
    self.assertItemsEqual([], d_3.op.control_inputs)
    self.assertItemsEqual([], d_4.op.control_inputs)

    self.assertItemsEqual([a_1.op], e_1.op.control_inputs)
    self.assertItemsEqual([a_2.op], e_2.op.control_inputs)
    self.assertItemsEqual([a_3.op], e_3.op.control_inputs)
    self.assertItemsEqual([a_4.op], e_4.op.control_inputs)

  def testRepeatedDependency(self):
    g = ops.Graph()
    a = g.create_op("TwoFloatOutputs", [], [dtypes.float32, dtypes.float32])
    a_0, a_1 = a.outputs
    with g.control_dependencies([a_0]):
      b = _apply_op(g, "FloatOutput", [], [dtypes.float32])
      with g.control_dependencies([a_1]):
        c = _apply_op(g, "FloatOutput", [], [dtypes.float32])

    self.assertEqual(b.op.control_inputs, [a])
    self.assertEqual(c.op.control_inputs, [a])

  def testNoControlDependencyWithDataDependency(self):
    g = ops.Graph()
    a = _apply_op(g, "FloatOutput", [], [dtypes.float32])
    with g.control_dependencies([a]):
      b = _apply_op(g, "Identity", [a], [dtypes.float32])

    self.assertEqual(b.op.control_inputs, [])


@test_util.with_c_api
class OpScopeTest(test_util.TensorFlowTestCase):

  @test_util.run_in_graph_and_eager_modes()
  def testEagerDefaultScopeName(self):
    with ops.name_scope(None, "default") as scope:
      self.assertEqual(scope, "default/")
      with ops.name_scope(None, "default2") as scope2:
        self.assertEqual(scope2, "default/default2/")

  def testNoScopeName(self):
    g0 = ops.Graph()
    values = [
        g0.create_op("A", [], [dtypes.float32]),
        g0.create_op("B", [], [dtypes.float32])
    ]
    with self.assertRaises(ValueError):
      with ops.name_scope(None, values=values):
        pass
    with self.assertRaises(ValueError):
      with ops.name_scope(None, None, values):
        pass

  def testEmptyScopeName(self):
    g0 = ops.Graph()
    a = g0.create_op("A", [], [dtypes.float32])
    b = g0.create_op("B", [], [dtypes.float32])
    with ops.name_scope("", values=[a, b]) as scope:
      self.assertEqual("", scope)
      self.assertEqual(g0, ops.get_default_graph())
    with ops.name_scope("", "my_default_scope", [a, b]) as scope:
      self.assertEqual("", scope)
      self.assertEqual(g0, ops.get_default_graph())

  def testDefaultScopeName(self):
    g0 = ops.Graph()
    a = g0.create_op("A", [], [dtypes.float32])
    b = g0.create_op("B", [], [dtypes.float32])
    scope_name = "my_scope"
    default_scope_name = "my_default_scope"
    with ops.name_scope(scope_name, default_scope_name, [a, b]) as scope:
      self.assertEqual("%s/" % scope_name, scope)
      self.assertEqual(g0, ops.get_default_graph())
    with ops.name_scope(None, default_scope_name, [a, b]) as scope:
      self.assertEqual("%s/" % default_scope_name, scope)
      self.assertEqual(g0, ops.get_default_graph())

  def _testGraphElements(self, graph_elements):
    scope_name = "my_scope"
    with ops.name_scope(scope_name, values=graph_elements) as scope:
      self.assertEqual("%s/" % scope_name, scope)
      self.assertEqual(graph_elements[0].graph, ops.get_default_graph())
    g1 = ops.Graph()
    a = g1.create_op("A", [], [dtypes.float32])
    with self.assertRaises(ValueError):
      with ops.name_scope(scope_name, values=graph_elements + [a]):
        pass

  def testTensor(self):
    g0 = ops.Graph()
    a = g0.create_op("A", [], [dtypes.float32])
    b = g0.create_op("B", [], [dtypes.float32])
    self._testGraphElements([a, b])

  def testSparseTensor(self):
    g0 = ops.Graph()
    a = g0.create_op("A", [], [dtypes.float32])
    b = g0.create_op("B", [], [dtypes.float32])
    sparse = sparse_tensor.SparseTensor(
        _apply_op(g0, "Int64Output", [], [dtypes.int64]),
        _apply_op(g0, "FloatOutput", [], [dtypes.float32]),
        _apply_op(g0, "Int64Output", [], [dtypes.int64]))
    self._testGraphElements([a, sparse, b])

  def testVariable(self):
    g0 = ops.Graph()
    with g0.as_default():
      variable = variables.Variable([1.0])
    a = g0.create_op("A", [], [dtypes.float32])
    b = g0.create_op("B", [], [dtypes.float32])
    self._testGraphElements([a, variable, b])


@test_util.with_c_api
class GraphTest(test_util.TensorFlowTestCase):

  def setUp(self):
    ops.reset_default_graph()

  def _AssertDefault(self, expected):
    self.assertIs(expected, ops.get_default_graph())

  def testResetDefaultGraphNesting(self):
    g0 = ops.Graph()
    with self.assertRaises(AssertionError):
      with g0.as_default():
        ops.reset_default_graph()

  def testGraphContextManager(self):
    g0 = ops.Graph()
    with g0.as_default() as g1:
      self.assertIs(g0, g1)

  def testDefaultGraph(self):
    orig = ops.get_default_graph()
    self._AssertDefault(orig)
    g0 = ops.Graph()
    self._AssertDefault(orig)
    context_manager_0 = g0.as_default()
    self._AssertDefault(orig)
    with context_manager_0 as g0:
      self._AssertDefault(g0)
      with ops.Graph().as_default() as g1:
        self._AssertDefault(g1)
      self._AssertDefault(g0)
    self._AssertDefault(orig)

  def testPreventFeeding(self):
    g = ops.Graph()
    a = constant_op.constant(2.0)
    self.assertTrue(g.is_feedable(a))
    g.prevent_feeding(a)
    self.assertFalse(g.is_feedable(a))

  def testPreventFetching(self):
    g = ops.Graph()
    a = constant_op.constant(2.0)
    self.assertTrue(g.is_fetchable(a))
    g.prevent_fetching(a.op)
    self.assertFalse(g.is_fetchable(a))

  def testAsGraphElementConversions(self):

    class ConvertibleObj(object):

      def _as_graph_element(self):
        return "FloatOutput:0"

    class NonConvertibleObj(object):

      pass

    g = ops.Graph()
    a = _apply_op(g, "FloatOutput", [], [dtypes.float32])
    self.assertEqual(a, g.as_graph_element(ConvertibleObj()))
    with self.assertRaises(TypeError):
      g.as_graph_element(NonConvertibleObj())

  # Regression test against creating custom __del__ functions in classes
  # involved in cyclic references, e.g. Graph and Operation. (Python won't gc
  # cycles that require calling a __del__ method, because the __del__ method can
  # theoretically increase the object's refcount to "save" it from gc, and any
  # already-deleted objects in the cycle would have be to restored.)
  def testGarbageCollected(self):
    # Create a graph we can delete and a weak reference to monitor if it's gc'd
    g = ops.Graph()
    g_ref = weakref.ref(g)
    # Create some ops
    with g.as_default():
      a = constant_op.constant(2.0)
      b = constant_op.constant(3.0)
      c = math_ops.add(a, b)
    # Create a session we can delete
    with session.Session(graph=g) as sess:
      sess.run(c)
    # Delete all references and trigger gc
    del g
    del a
    del b
    del c
    del sess
    gc.collect()
    self.assertIsNone(g_ref())


@test_util.with_c_api
class AttrScopeTest(test_util.TensorFlowTestCase):

  def _get_test_attrs(self):
    x = control_flow_ops.no_op()
    try:
      a = compat.as_text(x.get_attr("_A"))
    except ValueError:
      a = None
    try:
      b = compat.as_text(x.get_attr("_B"))
    except ValueError:
      b = None
    print(a, b)
    return (a, b)

  def testNoLabel(self):
    with self.test_session():
      self.assertAllEqual((None, None), self._get_test_attrs())

  def testLabelMap(self):
    with self.test_session() as sess:
      a1 = self._get_test_attrs()
      with sess.graph._attr_scope({
          "_A": attr_value_pb2.AttrValue(s=compat.as_bytes("foo"))
      }):
        a2 = self._get_test_attrs()
        with sess.graph._attr_scope({
            "_A": None,
            "_B": attr_value_pb2.AttrValue(s=compat.as_bytes("bar"))
        }):
          a3 = self._get_test_attrs()
          with sess.graph._attr_scope({
              "_A": attr_value_pb2.AttrValue(s=compat.as_bytes("baz"))
          }):
            a4 = self._get_test_attrs()
          a5 = self._get_test_attrs()
        a6 = self._get_test_attrs()
      a7 = self._get_test_attrs()

      self.assertAllEqual((None, None), a1)
      self.assertAllEqual(("foo", None), a2)
      self.assertAllEqual((None, "bar"), a3)
      self.assertAllEqual(("baz", "bar"), a4)
      self.assertAllEqual((None, "bar"), a5)
      self.assertAllEqual(("foo", None), a6)
      self.assertAllEqual((None, None), a7)


ops.RegisterShape("KernelLabel")(common_shapes.scalar_shape)


@test_util.with_c_api
class KernelLabelTest(test_util.TensorFlowTestCase):

  @test_util.enable_c_api
  def testNoLabel(self):
    with self.test_session():
      self.assertAllEqual(b"My label is: default",
                          test_ops.kernel_label().eval())

  def testLabelMap(self):
    with self.test_session() as sess:
      default_1 = test_ops.kernel_label()
      # pylint: disable=protected-access
      with sess.graph._kernel_label_map({"KernelLabel": "overload_1"}):
        overload_1_1 = test_ops.kernel_label()
        with sess.graph._kernel_label_map({"KernelLabel": "overload_2"}):
          overload_2 = test_ops.kernel_label()
          with sess.graph._kernel_label_map({"KernelLabel": ""}):
            default_2 = test_ops.kernel_label()
        overload_1_2 = test_ops.kernel_label()
      # pylint: enable=protected-access
      default_3 = test_ops.kernel_label()

      self.assertAllEqual(b"My label is: default", default_1.eval())
      self.assertAllEqual(b"My label is: default", default_2.eval())
      self.assertAllEqual(b"My label is: default", default_3.eval())
      self.assertAllEqual(b"My label is: overload_1", overload_1_1.eval())
      self.assertAllEqual(b"My label is: overload_1", overload_1_2.eval())
      self.assertAllEqual(b"My label is: overload_2", overload_2.eval())


@test_util.with_c_api
class AsGraphDefTest(test_util.TensorFlowTestCase):

  def testGraphDefVersion(self):
    """Test that the graphdef version is plumbed through to kernels."""
    with ops.Graph().as_default() as g:
      version = g.graph_def_versions.producer
      with self.test_session(graph=g):
        v = test_ops.graph_def_version().eval()
        self.assertEqual(version, v)

  def testAddShapes(self):
    with ops.Graph().as_default() as g:
      t1, t2, t3, t4, t5 = _apply_op(g, "FiveFloatOutputs", [],
                                     [dtypes.float32] * 5)
      t1.set_shape(None)
      t2.set_shape([])
      t3.set_shape([None])
      t4.set_shape([43, 37])
      t5.set_shape([43, None])

      gd = g.as_graph_def(add_shapes=True)
      self.assertProtoEqualsVersion("""
      node { name: "FiveFloatOutputs" op: "FiveFloatOutputs"
        attr {
          key: "_output_shapes"
          value {
            list {
              shape { unknown_rank: true }
              shape { }
              shape { dim { size: -1 } }
              shape { dim { size: 43 } dim { size: 37 } }
              shape { dim { size: 43 } dim { size: -1 } }
            }
          }
        }
      }
      """, gd)


@ops.RegisterStatistics("a", "flops")
def _calc_a_forward_flops(unused_graph, unused_node):
  return ops.OpStats("flops", 20)


@test_util.with_c_api
class StatisticsTest(test_util.TensorFlowTestCase):

  def testRegisteredNode(self):
    graph = ops.Graph()
    node = ops._NodeDef("a", "an_a")
    flops = ops.get_stats_for_node_def(graph, node, "flops")
    self.assertEqual(20, flops.value)
    missing_stat = ops.get_stats_for_node_def(graph, node, "missing_stat")
    self.assertEqual(None, missing_stat.value)

  def testUnregisteredNode(self):
    graph = ops.Graph()
    node = ops._NodeDef("b", "a_b")
    weight_params = ops.get_stats_for_node_def(graph, node, "weight_params")
    self.assertEqual(None, weight_params.value)

  def testAccumulateStatistics(self):
    flops_total = ops.OpStats("flops")
    self.assertEqual(None, flops_total.value)
    second_flops = ops.OpStats("flops", 3)
    flops_total += second_flops
    self.assertEqual(3, flops_total.value)


@test_util.with_c_api
class ColocationGroupTest(test_util.TensorFlowTestCase):

  def testBasic(self):
    a = constant_op.constant([2.0], name="a")
    with ops.colocate_with(a.op):
      b = constant_op.constant(3.0)
    c = constant_op.constant(4.0)
    self.assertEqual([b"loc:@a"], a.op.colocation_groups())
    self.assertEqual([b"loc:@a"], b.op.colocation_groups())
    with self.assertRaises(ValueError):
      c.op.get_attr("_class")

  def testColocationDeviceInteraction(self):
    with ops.device("/cpu:0"):
      with ops.device("/device:GPU:0"):
        a = constant_op.constant([2.0], name="a")
      with ops.colocate_with(a.op):
        # 'b' is created in the scope of /cpu:0, but it is
        # colocated with 'a', which is on '/device:GPU:0'.  colocate_with
        # overrides devices because it is a stronger constraint.
        b = constant_op.constant(3.0)
    self.assertEqual([b"loc:@a"], b.op.colocation_groups())
    self.assertEqual(a.op.device, b.op.device)

  def testColocationCanonicalization(self):
    with ops.device("/device:GPU:0"):
      _ = constant_op.constant(2.0)
    with ops.device(lambda op: "/device:GPU:0"):
      b = constant_op.constant(3.0)
    with ops.get_default_graph().colocate_with(b):
      with ops.device("/device:GPU:0"):
        c = constant_op.constant(4.0)

    # A's device will be /device:GPU:0
    # B's device will be /device:GPU:0
    # C's device will be /device:GPU:0 because it
    # inherits B's device name, after canonicalizing the names.
    self.assertEqual(b.op.device, c.op.device)

  def testLocationOverrides(self):
    with ops.device("/cpu:0"):
      with ops.device("/device:GPU:0"):
        a = constant_op.constant([2.0], name="a")
        # Note that this colocation is "redundant", since we are
        # within the scope of "/device:GPU:0".  However, we would like to
        # preserve in the GraphDef that these two ops should be
        # colocated in a portable way.
        with ops.colocate_with(a.op):
          b = constant_op.constant(3.0)
        c = constant_op.constant(4.0)
      d = constant_op.constant(5.0)

    self.assertEqual([b"loc:@a"], b.op.colocation_groups())
    self.assertEqual("/device:GPU:0", a.op.device)
    self.assertEqual(a.op.device, b.op.device)

    # Test that device function stack is restored.
    self.assertEqual("/device:GPU:0", c.op.device)
    self.assertEqual("/device:CPU:0", d.op.device)

  def testNestedColocateWith(self):
    a = constant_op.constant([2.0], name="a")
    with ops.colocate_with(a.op):
      b = constant_op.constant(3.0)
      with ops.colocate_with(b.op):
        c = constant_op.constant(4.0)
    self.assertEqual([b"loc:@a"], b.op.colocation_groups())
    self.assertEqual([b"loc:@a"], c.op.colocation_groups())

  def testMultiColocationGroups(self):
    a = constant_op.constant([2.0], name="a")
    b = constant_op.constant(3.0, name="b")
    with ops.colocate_with(a.op):
      with ops.colocate_with(b.op):
        c = constant_op.constant(4.0)
    self.assertEqual(set([b"loc:@a", b"loc:@b"]), set(c.op.colocation_groups()))

  def testColocationIgnoreStack(self):
    a = constant_op.constant([2.0], name="a")
    b = constant_op.constant(3.0, name="b")
    with ops.colocate_with(a.op):
      with ops.colocate_with(b.op, ignore_existing=True):
        c = constant_op.constant(4.0)
    self.assertEqual(set([b"loc:@b"]), set(c.op.colocation_groups()))

  def testColocateWithReset(self):
    a = constant_op.constant([2.0], name="a")
    with ops.colocate_with(a.op):
      b = constant_op.constant(3.0, name="b")
      with ops.colocate_with(None, ignore_existing=True):
        c = constant_op.constant(4.0, name="c")
    self.assertEqual([b"loc:@a"], b.op.colocation_groups())
    self.assertEqual([b"loc:@c"], c.op.colocation_groups())

  def testColocateWithInitialNoneThenNested(self):
    a = constant_op.constant([2.0], name="a")
    with ops.colocate_with(a.op):
      with ops.colocate_with(None, ignore_existing=True):
        b = constant_op.constant(3.0, name="b")
        with ops.colocate_with(b.op):
          c = constant_op.constant(4.0, name="c")
    self.assertEqual([b"loc:@b"], b.op.colocation_groups())
    self.assertEqual([b"loc:@b"], c.op.colocation_groups())

  def testColocateVariables(self):
    a = variables.Variable([2.0], name="a")
    with ops.colocate_with(a.op):
      b = variables.Variable([3.0], name="b")
    self.assertEqual([b"loc:@a"], b.op.colocation_groups())

  def testInconsistentDeviceWithinColocate(self):
    with ops.device("/device:GPU:0"):
      a = constant_op.constant([2.0], name="a")
      with ops.colocate_with(a.op):
        # This is allowed due to legacy but clearly wrong, since we
        # should really be colocating with 'a'.  We allow devices to
        # override colocate_with, but we log warnings to suggest that
        # this is probably unintentional or misguided.
        with ops.device("/cpu:0"):
          b = constant_op.constant([3.0], name="b")

    self.assertEqual("/device:CPU:0", b.device)


@test_util.with_c_api
class DeprecatedTest(test_util.TensorFlowTestCase):

  def testSuccess(self):
    # TODO(skyewm): make g.graph_def_versions work with the C API enabled
    if ops._USE_C_API: return

    with ops.Graph().as_default() as g:
      g.graph_def_versions.producer = 7
      old = test_ops.old()
      with self.test_session(graph=g):
        old.run()

  def _error(self):
    return ((r"Op Old is not available in GraphDef version %d\. "
             r"It has been removed in version 8\. For reasons\.") %
            versions.GRAPH_DEF_VERSION)

  def testGraphConstructionFail(self):
    with ops.Graph().as_default():
      with self.assertRaisesRegexp(NotImplementedError, self._error()):
        test_ops.old()

  def testGraphExecutionFail(self):
    # TODO(skyewm): make g.graph_def_versions work with the C API enabled
    if ops._USE_C_API: return

    with ops.Graph().as_default() as g:
      g.graph_def_versions.producer = 7
      old = test_ops.old()
      g.graph_def_versions.producer = versions.GRAPH_DEF_VERSION
      with self.test_session(graph=g):
        with self.assertRaisesRegexp(errors.UnimplementedError, self._error()):
          old.run()


@test_util.with_c_api
class DenseTensorLikeTypeTest(test_util.TensorFlowTestCase):

  def testSuccess(self):
    op = ops.Operation(
        ops._NodeDef("FloatOutput", "myop"), ops.Graph(), [], [dtypes.float32])
    t = op.outputs[0]
    self.assertTrue(ops.is_dense_tensor_like(t))

    v = variables.Variable([17])
    self.assertTrue(ops.is_dense_tensor_like(v))

  class BadClassNoName(object):
    pass

  class BadClassBadName(object):

    def name(self):
      pass

  class BadClassNoDtype(object):

    @property
    def name(self):
      pass

  class BadClassBadDtype(object):

    @property
    def name(self):
      pass

    def dtype(self):
      pass

  def testBadClass(self):
    with self.assertRaisesRegexp(TypeError, "`name`"):
      ops.register_dense_tensor_like_type(
          DenseTensorLikeTypeTest.BadClassNoName)
    with self.assertRaisesRegexp(TypeError, "`name`"):
      ops.register_dense_tensor_like_type(
          DenseTensorLikeTypeTest.BadClassBadName)
    with self.assertRaisesRegexp(TypeError, "`dtype`"):
      ops.register_dense_tensor_like_type(
          DenseTensorLikeTypeTest.BadClassNoDtype)
    with self.assertRaisesRegexp(TypeError, "`dtype`"):
      ops.register_dense_tensor_like_type(
          DenseTensorLikeTypeTest.BadClassBadDtype)


@test_util.with_c_api
class NameScopeTest(test_util.TensorFlowTestCase):

  def testStripAndPrependScope(self):
    strs = [
        "hidden1/hidden1/weights",  # Same prefix. Should strip.
        "hidden1///hidden1/weights",  # Extra "/". Should strip.
        "^hidden1/hidden1/weights",  # Same prefix. Should strip.
        "loc:@hidden1/hidden1/weights",  # Same prefix. Should strip.
        "hhidden1/hidden1/weights",  # Different prefix. Should keep.
        "hidden1"
    ]  # Not a prefix. Should keep.
    expected_striped = [
        "hidden1/weights", "hidden1/weights", "^hidden1/weights",
        "loc:@hidden1/weights", "hhidden1/hidden1/weights", "hidden1"
    ]
    expected_prepended = [
        "hidden2/hidden1/weights", "hidden2/hidden1/weights",
        "^hidden2/hidden1/weights", "loc:@hidden2/hidden1/weights",
        "hidden2/hhidden1/hidden1/weights", "hidden2/hidden1"
    ]
    name_scope_to_strip = "hidden1"
    name_scope_to_add = "hidden2"
    for es, ep, s in zip(expected_striped, expected_prepended, strs):
      striped = ops.strip_name_scope(s, name_scope_to_strip)
      self.assertEqual(es, striped)
      self.assertEqual(ep, ops.prepend_name_scope(striped, name_scope_to_add))

  def testGetNameScope(self):
    with ops.Graph().as_default() as g:
      with ops.name_scope("scope1"):
        with ops.name_scope("scope2"):
          with ops.name_scope("scope3"):
            self.assertEqual("scope1/scope2/scope3", g.get_name_scope())
          self.assertEqual("scope1/scope2", g.get_name_scope())
        self.assertEqual("scope1", g.get_name_scope())
      self.assertEqual("", g.get_name_scope())


@test_util.with_c_api
class TracebackTest(test_util.TensorFlowTestCase):

  def testTracebackWithStartLines(self):
    with self.test_session() as sess:
      a = constant_op.constant(2.0)
      sess.run(
          a,
          options=config_pb2.RunOptions(
              trace_level=config_pb2.RunOptions.FULL_TRACE))
      self.assertTrue(sess.graph.get_operations())

      # Tests that traceback_with_start_lines is the same as traceback
      # but includes one more element at the end.
      for op in sess.graph.get_operations():
        self.assertEquals(len(op.traceback), len(op.traceback_with_start_lines))
        for frame, frame_with_start_line in zip(
            op.traceback, op.traceback_with_start_lines):
          self.assertEquals(5, len(frame_with_start_line))
          self.assertEquals(frame, frame_with_start_line[:-1])


@test_util.with_c_api
class OutputTypesTest(test_util.TensorFlowTestCase):
  """Tests Operation._output_types property.

  This test should not exist as _output_types is a private property.
  This property is used by util.copy_elements and its tests would normally
  cover Operation._output_types. However, we can't yet run these tests in C
  API mode because their use _set_device method. This test will be deleted
  once we port _set_device and run the copy tests with C API on.
  """
  # TODO(iga): Remove this test

  def setUp(self):
    self.prev_use_c_api = ops._USE_C_API  # pylint: disable=protected-access
    ops._USE_C_API = True  # pylint: disable=protected-access

  def tearDown(self):
    ops._USE_C_API = self.prev_use_c_api  # pylint: disable=protected-access

  def testOneOutput(self):
    g = ops.Graph()
    with g.as_default():
      # Using a constant because creating unregistered ops
      # doesn't work with the C API.
      op = constant_op.constant(12, dtype=dtypes.uint16).op
      # pylint: disable=protected-access
      self.assertEqual([types_pb2.DT_UINT16], op._output_types)
      # pylint: enable=protected-access

  def testTwoDifferentOutputs(self):
    g = ops.Graph()
    with g.as_default():
      x = constant_op.constant([1, 1, 2, 4, 4, 4, 7, 8, 8],
                               dtype=dtypes.double)
      y, _ = gen_array_ops.unique(x)
      self.assertEqual([types_pb2.DT_DOUBLE, types_pb2.DT_INT32],
                       y.op._output_types)  # pylint: disable=protected-access

  def testThreeOutputs(self):
    g = ops.Graph()
    with g.as_default():
      # Using a split operationt because creating unregistered ops
      # doesn't work with the C API.
      a = constant_op.constant("abc", dtype=dtypes.string, shape=[5, 30])
      split0, _, _ = array_ops.split(a, [4, 15, 11], 1)
      # pylint: disable=protected-access
      self.assertEqual([types_pb2.DT_STRING] * 3, split0.op._output_types)
      # pylint: enable=protected-access


@test_util.with_c_api
class InputTypesTest(test_util.TensorFlowTestCase):
  """Tests Operation._input_dtypes and Operation._input_types properties.

  This test should not exist as _input_types is a private property.
  This property is used by many tests that would normally cover its
  behavior. However, we can't yet run these tests in C
  API mode because they use _set_device method. This test will be deleted
  once we port _set_device.
  """
  # TODO(iga): Remove this test

  def setUp(self):
    self.prev_use_c_api = ops._USE_C_API  # pylint: disable=protected-access
    ops._USE_C_API = True  # pylint: disable=protected-access

  def tearDown(self):
    ops._USE_C_API = self.prev_use_c_api  # pylint: disable=protected-access

  def testZeroInputs(self):
    g = ops.Graph()
    with g.as_default():
      # Using a constant because creating unregistered ops
      # doesn't work with the C API.
      op = constant_op.constant(12, dtype=dtypes.uint16).op
      # pylint: disable=protected-access
      self.assertEqual([], op._input_types)
      self.assertEqual([], op._input_dtypes)
      # pylint: enable=protected-access

  def testTwoInputs(self):
    g = ops.Graph()
    with g.as_default():
      x = constant_op.constant(1.0, dtype=dtypes.double)
      y = constant_op.constant(2.0, dtype=dtypes.double)
      z = math_ops.multiply(x, y)
      # pylint: disable=protected-access
      self.assertTrue(isinstance(z.op._input_types[0], dtypes.DType))
      self.assertTrue(isinstance(z.op._input_types[1], dtypes.DType))
      self.assertEqual([dtypes.double, dtypes.double], z.op._input_types)
      self.assertEqual([dtypes.double, dtypes.double], z.op._input_dtypes)
      # pylint: enable=protected-access


if __name__ == "__main__":
  googletest.main()
