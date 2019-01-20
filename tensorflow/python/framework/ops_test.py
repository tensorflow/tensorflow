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
import os
import threading
import weakref

from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.eager import context
from tensorflow.python.eager import function as eager_function
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
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import resources
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
import tensorflow.python.ops.gradients  # pylint: disable=unused-import
from tensorflow.python.platform import googletest
from tensorflow.python.util import compat

ops._set_call_cpp_shape_fn(common_shapes.call_cpp_shape_fn)


class ResourceTest(test_util.TensorFlowTestCase):

  @test_util.run_deprecated_v1
  def testBuildGraph(self):
    with self.cached_session():
      pt = test_ops.stub_resource_handle_op(container="a", shared_name="b")
      test_ops.resource_create_op(pt).run()

  @test_util.run_deprecated_v1
  def testInitialize(self):
    with self.cached_session():
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


class TensorAndShapeTest(test_util.TensorFlowTestCase):

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

  def testAddShape(self):
    with self.cached_session():
      a = array_ops.zeros([2, 3])
      b = array_ops.ones([1, 3])
      c = a + b
      self.assertEqual([2, 3], c.shape)

  @test_util.run_deprecated_v1
  def testUnknownDim(self):
    with self.cached_session():
      a = array_ops.placeholder(dtype=dtypes.float32, shape=[2, None, 3])
      b = array_ops.placeholder(dtype=dtypes.float32, shape=[2, None, 3])
      c = a + b
      self.assertEqual([2, None, 3], c.shape.as_list())

  @test_util.run_deprecated_v1
  def testUnknownShape(self):
    with self.cached_session():
      a = array_ops.placeholder(dtype=dtypes.float32, shape=None)
      b = array_ops.ones([1, 3])
      c = a + b
      self.assertEqual(tensor_shape.unknown_shape(), c.shape)

  @test_util.run_deprecated_v1
  def testScalarShape(self):
    with self.cached_session():
      a = array_ops.placeholder(dtype=dtypes.float32, shape=[])
      b = array_ops.ones([])
      c = a + b
      self.assertEqual(tensor_shape.scalar(), c.shape)

  @test_util.run_deprecated_v1
  def testShapeFunctionError(self):
    with self.cached_session():
      a = array_ops.ones([1, 2, 3])
      b = array_ops.ones([4, 5, 6])
      with self.assertRaisesRegexp(
          ValueError,
          r"Dimensions must be equal, but are 2 and 5 for 'add' \(op: 'Add'\) "
          r"with input shapes: \[1,2,3\], \[4,5,6\]."):
        _ = a + b


class IndexedSlicesTest(test_util.TensorFlowTestCase):

  @test_util.run_in_graph_and_eager_modes
  def testToTensor(self):
    values = constant_op.constant([2, 3, 5, 7], shape=[2, 2])
    indices = constant_op.constant([0, 2])
    dense_shape = constant_op.constant([3, 2])
    x = ops.IndexedSlices(values, indices, dense_shape)
    tensor = ops.convert_to_tensor(x, name="tensor")
    self.assertAllEqual(self.evaluate(tensor), [[2, 3], [0, 0], [5, 7]])

  @test_util.run_deprecated_v1
  def testNegation(self):
    with self.cached_session():
      values = constant_op.constant([2, 3, 5, 7], shape=[2, 2])
      indices = constant_op.constant([0, 2])
      x = -ops.IndexedSlices(values, indices)
      self.assertAllEqual(x.values.eval(), [[-2, -3], [-5, -7]])
      self.assertAllEqual(x.indices.eval(), [0, 2])

  @test_util.run_deprecated_v1
  def testScalarMul(self):
    with self.cached_session():
      values = constant_op.constant([2, 3, 5, 7], shape=[2, 2])
      indices = constant_op.constant([0, 2])
      x = math_ops.scalar_mul(-2, ops.IndexedSlices(values, indices))
      self.assertAllEqual(x.values.eval(), [[-4, -6], [-10, -14]])
      self.assertAllEqual(x.indices.eval(), [0, 2])


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


class OperationTest(test_util.TensorFlowTestCase):

  @test_util.run_deprecated_v1
  def testNoInputs(self):
    op = test_ops.float_output_string_output(name="myop").a.op
    self.assertEqual(2, len(op.values()))
    self.assertEqual(0, len(op.inputs))
    self.assertEqual("myop", op.name)

    float_t, label_str_t = op.values()
    self.assertEqual(dtypes.float32, float_t.dtype)
    self.assertEqual(op, float_t.op)
    self.assertEqual(0, float_t._value_index)
    self.assertEqual(0, len(float_t.consumers()))
    self.assertEqual("myop", float_t._as_node_def_input())

    self.assertEqual(dtypes.string, label_str_t.dtype)
    self.assertEqual(op, label_str_t.op)
    self.assertEqual(1, label_str_t._value_index)
    self.assertEqual(0, len(label_str_t.consumers()))
    self.assertEqual("myop:1", label_str_t._as_node_def_input())

    self.assertProtoEquals("op:'FloatOutputStringOutput' name:'myop'",
                           op.node_def)

  @test_util.run_deprecated_v1
  def testNoOutputs(self):
    op1 = test_ops.float_output(name="myop1").op
    float_t, = op1.values()
    op2 = test_ops.float_input(float_t, name="myop2")
    self.assertEqual(0, len(op2.values()))
    self.assertEqual(1, len(op2.inputs))
    self.assertIs(float_t, op2.inputs[0])

    self.assertEqual(1, len(float_t.consumers()))
    self.assertEqual(op2, float_t.consumers()[0])

    self.assertProtoEquals("op:'FloatOutput' name:'myop1'", op1.node_def)
    self.assertProtoEquals("op:'FloatInput' name:'myop2' input:'myop1'",
                           op2.node_def)

  @test_util.run_deprecated_v1
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

    self.assertEqual(1, len(float1_t.consumers()))
    self.assertEqual(op3, float1_t.consumers()[0])

    self.assertEqual(0, len(float2_t.consumers()))

    self.assertEqual(2, len(label2_str_t.consumers()))
    self.assertEqual(op3, label2_str_t.consumers()[0])
    self.assertEqual(op3, label2_str_t.consumers()[1])

    self.assertProtoEquals("""
    op:'Foo2' name:'myop3'
    input:'myop1' input:'myop2:1' input:'myop2:1'
    """, op3.node_def)

  def testDeviceFromNodeDef(self):
    op = ops.Operation(
        ops._NodeDef("None", "myop", device="/job:goo/device:GPU:0"),
        ops.Graph(), [], [])
    self.assertEqual("/job:goo/device:GPU:0", op.device)

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

  @test_util.run_deprecated_v1
  def testNoShapeFunction(self):
    op = test_ops.a()
    self.assertEqual(tensor_shape.unknown_shape(), op.get_shape())

  @test_util.run_in_graph_and_eager_modes
  def testConvertToTensorNestedArray(self):
    values = [[2], [3], [5], [7]]
    tensor = ops.convert_to_tensor(values)
    self.assertAllEqual((4, 1), tensor.get_shape().as_list())
    self.assertAllEqual(values, self.evaluate(tensor))

  def testShapeTuple(self):
    with self.cached_session():
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

  @test_util.run_in_graph_and_eager_modes
  def testConvertToTensorNestedTuple(self):
    values = ((2,), (3,), (5,), (7,))
    tensor = ops.convert_to_tensor(values)
    self.assertAllEqual((4, 1), tensor.get_shape().as_list())
    self.assertAllEqual(values, self.evaluate(ops.convert_to_tensor(values)))

  @test_util.run_in_graph_and_eager_modes
  def testConvertToTensorNestedTensors(self):
    values = ((2,), (3,), (5,), (7,))
    tensor = ops.convert_to_tensor(
        [constant_op.constant(row) for row in values])
    self.assertAllEqual((4, 1), tensor.get_shape().as_list())
    self.assertAllEqual(values, self.evaluate(tensor))
    tensor = ops.convert_to_tensor(
        [[constant_op.constant(v) for v in row] for row in values])
    self.assertAllEqual((4, 1), tensor.get_shape().as_list())
    self.assertAllEqual(values, self.evaluate(tensor))

  @test_util.run_in_graph_and_eager_modes
  def testConvertToTensorNestedMix(self):
    values = ([2], (3,), [constant_op.constant(5)], constant_op.constant([7]))
    tensor = ops.convert_to_tensor(values)
    self.assertAllEqual((4, 1), tensor.get_shape().as_list())
    self.assertAllEqual(((2,), (3,), (5,), (7,)), self.evaluate(tensor))

  @test_util.run_in_graph_and_eager_modes
  def testConvertToTensorPreferred(self):
    values = [2, 3, 5, 7]
    tensor = ops.convert_to_tensor(values, preferred_dtype=dtypes.float32)
    self.assertEqual(dtypes.float32, tensor.dtype)

    # Convert empty tensor to anything.
    values = []
    tensor = ops.convert_to_tensor(values, preferred_dtype=dtypes.int64)
    self.assertEqual(dtypes.int64, tensor.dtype)

    # The preferred dtype is a type error and will convert to
    # float32 instead.
    values = [1.23]
    tensor = ops.convert_to_tensor(values, preferred_dtype=dtypes.int64)
    self.assertEqual(dtypes.float32, tensor.dtype)

  @test_util.run_in_graph_and_eager_modes
  def testConvertToInvalidTensorType(self):
    with self.assertRaises(TypeError):
      # Forcing an invalid dtype should fail with a type error.
      values = [1.23]
      ops.convert_to_tensor(values, dtype=dtypes.int64)

  @test_util.run_in_graph_and_eager_modes
  def testConvertToTensorFromInvalidTensor(self):
    tensor = constant_op.constant(42.0, dtype=dtypes.float32)
    with self.assertRaises(ValueError):
      ops.convert_to_tensor(tensor, dtype=dtypes.int32)

  @test_util.run_deprecated_v1
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

  @test_util.run_deprecated_v1
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
    with self.assertRaisesRegexp(
        ValueError, "Operation 'FuncAttr' has no attr named 'FakeAttr'."):
      op.get_attr("FakeAttr")

  # TODO(b/65162920): remove this test when users who are directly mutating the
  # node_def have been updated to proper usage.
  @test_util.run_deprecated_v1
  def testSetAttr(self):
    op = test_ops.int_attr().op
    op._set_attr("foo", attr_value_pb2.AttrValue(i=2))
    # TODO(skyewm): add node_def check
    self.assertEqual(op.get_attr("foo"), 2)

  # TODO(nolivia): test all error cases
  def testAddControlInput(self):
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
    self.assertEqual(x._control_outputs, [z])

  @test_util.run_deprecated_v1
  def testRemoveAllControlInputs(self):
    a = constant_op.constant(1)
    with ops.control_dependencies([a]):
      b = constant_op.constant(2)
    c = constant_op.constant(3)
    d = constant_op.constant(4)
    e = constant_op.constant(5)
    with ops.control_dependencies([a, c]):
      f = d + e

    self.assertEqual(a.op.control_inputs, [])
    self.assertEqual(b.op.control_inputs, [a.op])
    self.assertEqual(f.op.control_inputs, [a.op, c.op])

    a.op._remove_all_control_inputs()  # pylint: disable=protected-access
    self.assertEqual(a.op.control_inputs, [])

    b.op._remove_all_control_inputs()  # pylint: disable=protected-access
    self.assertEqual(b.op.control_inputs, [])

    f.op._remove_all_control_inputs()  # pylint: disable=protected-access
    self.assertEqual(f.op.control_inputs, [])
    self.assertEqual(list(f.op.inputs), [d, e])

  @test_util.run_deprecated_v1
  def testControlInputCycle(self):
    graph = ops.Graph()
    with graph.as_default():
      z = constant_op.constant(0)
      x = constant_op.constant(1)
      y = constant_op.constant(2)
      y.op._add_control_input(z.op)  # pylint: disable=protected-access
      y.op._add_control_input(x.op)  # pylint: disable=protected-access
      x.op._add_control_input(y.op)  # pylint: disable=protected-access
    with self.session(graph=graph) as sess:
      with self.assertRaisesRegexp(
          errors.InvalidArgumentError,
          "Graph is invalid, contains a cycle with 2 nodes"):
        self.evaluate(x)

  def testUpdateInput(self):
    g = ops.Graph()
    with g.as_default():
      x = constant_op.constant(1)
      y = constant_op.constant(2)
      z = x + y

    z.op._update_input(0, y)  # pylint: disable=protected-access
    self.assertEquals(list(z.op.inputs), [y, y])
    self.assertEquals(x.consumers(), [])
    self.assertEquals(y.consumers(), [z.op, z.op])
    with session.Session(graph=g) as sess:
      self.assertEquals(self.evaluate(z), 4)

    z.op._update_input(0, x)  # pylint: disable=protected-access
    self.assertEquals(list(z.op.inputs), [x, y])
    self.assertEquals(x.consumers(), [z.op])
    self.assertEquals(y.consumers(), [z.op])
    with session.Session(graph=g) as sess:
      self.assertEquals(self.evaluate(z), 3)

    z.op._update_input(1, y)  # pylint: disable=protected-access
    self.assertEquals(list(z.op.inputs), [x, y])
    self.assertEquals(x.consumers(), [z.op])
    self.assertEquals(y.consumers(), [z.op])
    with session.Session(graph=g) as sess:
      self.assertEquals(self.evaluate(z), 3)

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
        self.evaluate(z)

  def testUpdateInputShapeError(self):
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
    g = ops.Graph()
    with g.as_default():
      x = constant_op.constant(1)
    with self.assertRaisesRegexp(
        errors.OutOfRangeError,
        r"Cannot update edge. Input index \[1\] is greater than the number of "
        r"total inputs \[0\]."
    ):
      x.op._update_input(1, x)  # pylint: disable=protected-access

  @test_util.enable_control_flow_v2
  @test_util.run_v1_only("b/120545219")
  def testAddWhileInput(self):
    @eager_function.defun
    def test():
      output = control_flow_ops.while_loop(lambda x: x < 3, lambda x: x + 1,
                                           [1])
      while_op = output.op.inputs[0].op
      self.assertEqual(while_op.type, "While")
      orig_num_inputs = len(while_op.inputs)

      # Make sure we can handle the while op having a control input.
      while_op._add_control_input(constant_op.constant(0).op)

      new_input1 = constant_op.constant(1.0)
      new_input2 = constant_op.constant(True)

      while_op._set_type_list_attr("T",
                                   [t.dtype for t in while_op.inputs] +
                                   [new_input1.dtype, new_input2.dtype])

      while_op._add_while_inputs([new_input1, new_input2])
      # Can't add an edge beyond what's specified by "T"
      with self.assertRaises(errors.OutOfRangeError):
        while_op._add_while_inputs([new_input2])
      self.assertEqual(len(while_op.inputs), orig_num_inputs + 2)  # pylint: disable=g-deprecated-assert

    test()

  @test_util.run_deprecated_v1
  def testOpDef(self):
    x = constant_op.constant(0)
    y = constant_op.constant(1)
    z = x + y

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

  def testInputsAreImmutable(self):
    g = ops.Graph()
    with g.as_default():
      x = test_ops.int_output()
      op = test_ops.int_input_int_output(x, name="myop").op
    with self.assertRaisesRegexp(
        AttributeError, "'_InputList' object has no attribute 'append'"):
      op.inputs.append(None)


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
class CreateOpFromTFOperationTest(test_util.TensorFlowTestCase):

  @test_util.run_deprecated_v1
  def testBasic(self):
    g = ops.Graph()
    with g.as_default():
      x = test_ops.int_output()
      c_op = ops._create_c_op(
          g, ops._NodeDef("IntInputIntOutput", "myop"), [x], [])
      op = g._create_op_from_tf_operation(c_op)

    self.assertEqual(op.name, "myop")
    self.assertEqual(op.type, "IntInputIntOutput")
    self.assertEqual(len(op.outputs), 1)
    self.assertEqual(op.outputs[0].shape, tensor_shape.unknown_shape())
    self.assertEqual(list(op.inputs), [x])
    self.assertEqual(op.control_inputs, [])
    self.assertEqual(op.graph, g)
    self.assertEqual(x.consumers(), [op])
    self.assertIsNotNone(op.traceback)
    self.assertEqual(g.get_operation_by_name("myop"), op)
    self.assertEqual(g.get_tensor_by_name("myop:0"), op.outputs[0])

  def testShape(self):
    g = ops.Graph()
    with g.as_default():
      x = constant_op.constant([[1, 2, 3], [4, 5, 6]])
      c_op = ops._create_c_op(g, ops._NodeDef("Identity", "myop"), [x], [])
      op = g._create_op_from_tf_operation(c_op)

    self.assertEqual(op.name, "myop")
    self.assertEqual(op.type, "Identity")
    self.assertEqual(len(op.outputs), 1)
    self.assertEqual(op.outputs[0].shape, tensor_shape.matrix(2, 3))

  def testUniqueName(self):
    g = ops.Graph()
    with g.as_default():
      c_op = ops._create_c_op(g, ops._NodeDef("IntOutput", "myop"), [], [])
      c_op2 = ops._create_c_op(g, ops._NodeDef("IntOutput", "myop_1"), [], [])
      op = g._create_op_from_tf_operation(c_op)
      op2 = g._create_op_from_tf_operation(c_op2)

      # Create ops with same names as op1 and op2. We expect the new names to be
      # uniquified.
      op3 = test_ops.int_output(name="myop").op
      op4 = test_ops.int_output(name="myop_1").op

    self.assertEqual(op.name, "myop")
    self.assertEqual(op2.name, "myop_1")
    self.assertEqual(op3.name, "myop_2")
    self.assertEqual(op4.name, "myop_1_1")

  @test_util.run_v1_only("b/120545219")
  def testCond(self):
    g = ops.Graph()
    with g.as_default():
      x = test_ops.int_output()

      def true_fn():
        ops._create_c_op(ops.get_default_graph(),
                         ops._NodeDef("IntInput", "cond/myop"), [x], [])
        new_ops = g._add_new_tf_operations()
        self.assertEqual(len(new_ops), 1)
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

  @test_util.run_v1_only("b/120545219")
  def testWhileLoop(self):
    g = ops.Graph()
    with g.as_default():
      x = test_ops.int_output()

      def body(i):
        ops._create_c_op(ops.get_default_graph(),
                         ops._NodeDef("IntInput", "myloop/myop"), [x], [])
        new_ops = g._add_new_tf_operations()
        self.assertEqual(len(new_ops), 1)
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

  @test_util.run_v1_only("b/120545219")
  def testWhileLoopWithInternalControlDep(self):
    g = ops.Graph()
    with g.as_default():
      x = test_ops.int_output()

      def body(i):
        c = constant_op.constant(1.0, name="c")
        ops._create_c_op(ops.get_default_graph(),
                         ops._NodeDef("IntInput", "myloop/myop"), [x], [])
        with ops.control_dependencies([c]):
          new_ops = g._add_new_tf_operations()
          self.assertEqual(len(new_ops), 1)
        return i

      control_flow_ops.while_loop(lambda i: i < 10, body, [0], name="myloop")

    op = g.get_operation_by_name("myloop/myop")
    self.assertIsNotNone(op)
    c = g.get_operation_by_name("myloop/c")
    self.assertIsNotNone(c)
    # Internal control dep is preserved
    self.assertEqual(op.control_inputs, [c])

  @test_util.run_v1_only("b/120545219")
  def testWhileLoopWithExternalControlDep(self):
    g = ops.Graph()
    with g.as_default():
      x = test_ops.int_output()
      c = constant_op.constant(1.0)

      def body(i):
        ops._create_c_op(ops.get_default_graph(),
                         ops._NodeDef("IntInput", "myloop/myop"), [x], [])
        with ops.control_dependencies([c]):
          new_ops = g._add_new_tf_operations()
          self.assertEqual(len(new_ops), 1)
        return i

      control_flow_ops.while_loop(lambda i: i < 10, body, [0], name="myloop")

    op = g.get_operation_by_name("myloop/myop")
    self.assertIsNotNone(op)
    # External control dep is removed and replaced with internal control dep
    self.assertNotEqual(op.control_inputs[0], c.op)
    self.assertIsNotNone(op.control_inputs[0]._get_control_flow_context())


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

  @test_util.run_deprecated_v1
  def testNameAndVariableScope(self):
    with self.cached_session() as sess:
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

  def testUniqueNameCaseInsensitivity(self):
    g = ops.Graph()
    self.assertEqual("foo", g.unique_name("foo"))
    self.assertEqual("Foo_1", g.unique_name("Foo"))
    with g.name_scope("bar"):
      self.assertEqual("bar/foo", g.unique_name("foo"))
    with g.name_scope("Bar"):
      self.assertEqual("Bar_1/foo", g.unique_name("foo"))

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


class DeviceTest(test_util.TensorFlowTestCase):

  def testNoDevice(self):
    g = ops.Graph()
    op = g.create_op("FloatOutput", [], [dtypes.float32])
    self.assertDeviceEqual(None, op.device)
    gd = g.as_graph_def()
    self.assertProtoEqualsVersion("""
      node { name: "FloatOutput" op: "FloatOutput" }
    """, gd)

  def testEagerBackingDevice(self):
    with context.eager_mode():
      with ops.device("/device:CPU:0"):
        t = constant_op.constant(1.0)
        self.assertRegexpMatches(t.device, "/device:CPU:0")
        self.assertRegexpMatches(t.backing_device, "/device:CPU:0")

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


class MultithreadedGraphStateTest(test_util.TensorFlowTestCase):

  class TestThread(threading.Thread):

    def __init__(self, graph, replica_id):
      super(MultithreadedGraphStateTest.TestThread, self).__init__()
      self._graph = graph
      self._replica_id = replica_id
      # This thread sets this event when it mutated the graph.  The caller can
      # wait for that.
      self.has_mutated_graph = threading.Event()
      # This thread waits for when it should continue.  The caller can set this
      # event.
      self.should_continue = threading.Event()

    def run(self):
      # Mutate a graph's stack, then set `has_mutated_graph`, then wait for
      # `should_continue`, then add an op to the graph affected by the graph's
      # stack.
      raise NotImplementedError("must be implemented in descendants")

  def testDeviceFunctionStack(self):

    class DeviceSettingThread(self.TestThread):

      def run(self):
        with g.device("/job:worker/replica:{}".format(self._replica_id)):
          self.has_mutated_graph.set()
          self.should_continue.wait()
          self.should_continue.clear()
          g.create_op(
              "FloatOutput", [], [dtypes.float32],
              name="FloatOutput_{}".format(self._replica_id))

    g = ops.Graph()
    # If `switch_to_thread` isn't called, then device placement of the ops
    # below is not deterministic.
    g.switch_to_thread_local()
    threads = [DeviceSettingThread(g, i) for i in range(3)]
    for t in threads:
      t.start()
      t.has_mutated_graph.wait()
      t.has_mutated_graph.clear()
    for t in threads:
      t.should_continue.set()
      t.join()

    gd = g.as_graph_def()
    self.assertProtoEqualsVersion("""
      node { name: "FloatOutput_0" op: "FloatOutput"
             device: "/job:worker/replica:0" }
      node { name: "FloatOutput_1" op: "FloatOutput"
             device: "/job:worker/replica:1" }
      node { name: "FloatOutput_2" op: "FloatOutput"
             device: "/job:worker/replica:2" }
    """, gd)

  def testColocateWith(self):

    class ColocatingThread(self.TestThread):

      def __init__(self, graph, replica_id, op_to_colocate_with):
        super(ColocatingThread, self).__init__(graph, replica_id)
        self._op_to_colocate_with = op_to_colocate_with

      def run(self):
        with g.colocate_with(self._op_to_colocate_with):
          self.has_mutated_graph.set()
          self.should_continue.wait()
          self.should_continue.clear()
          g.create_op(
              "FloatOutput", [], [dtypes.float32],
              name="FloatOutput_{}".format(self._replica_id))

    g = ops.Graph()
    ops_to_colocate_with = []
    for i in range(3):
      with g.device("/job:worker/replica:{}".format(i)):
        ops_to_colocate_with.append(
            g.create_op(
                "FloatOutput", [], [dtypes.float32],
                name="ColocateWithMe_{}".format(i)))

    # If `switch_to_thread` isn't called, then `device` and `attr` values for
    # the ops below are not deterministic.
    g.switch_to_thread_local()
    threads = [
        ColocatingThread(g, i, ops_to_colocate_with[i]) for i in range(3)
    ]
    for t in threads:
      t.start()
      t.has_mutated_graph.wait()
      t.has_mutated_graph.clear()
    for t in threads:
      t.should_continue.set()
      t.join()

    gd = g.as_graph_def()
    self.assertProtoEqualsVersion("""
      node { name: "ColocateWithMe_0" op: "FloatOutput"
             device: "/job:worker/replica:0" }
      node { name: "ColocateWithMe_1" op: "FloatOutput"
             device: "/job:worker/replica:1" }
      node { name: "ColocateWithMe_2" op: "FloatOutput"
             device: "/job:worker/replica:2" }
      node { name: "FloatOutput_0" op: "FloatOutput"
             device: "/job:worker/replica:0"
             attr { key: "_class"
               value { list {
                 s: "loc:@ColocateWithMe_0"}}}}
      node { name: "FloatOutput_1" op: "FloatOutput"
             device: "/job:worker/replica:1"
             attr { key: "_class"
               value { list {
                 s: "loc:@ColocateWithMe_1"}}}}
      node { name: "FloatOutput_2" op: "FloatOutput"
             device: "/job:worker/replica:2"
             attr { key: "_class"
               value { list {
                 s: "loc:@ColocateWithMe_2"}}}}
    """, gd)

  def testControlDependencies(self):

    class DependingThread(self.TestThread):

      def __init__(self, graph, replica_id, dependency_op):
        super(DependingThread, self).__init__(graph, replica_id)
        self._dependency_op = dependency_op

      def run(self):
        with g.control_dependencies([self._dependency_op]):
          self.has_mutated_graph.set()
          self.should_continue.wait()
          self.should_continue.clear()
          g.create_op(
              "FloatOutput", [], [dtypes.float32],
              name="FloatOutput_{}".format(self._replica_id))

    g = ops.Graph()
    dependency_ops = []
    for i in range(3):
      dependency_ops.append(
          g.create_op(
              "FloatOutput", [], [dtypes.float32],
              name="ColocateWithMe_{}".format(i)))

    # If `switch_to_thread` isn't called, then `input` values for the ops below
    # are not deterministic.
    g.switch_to_thread_local()
    threads = [DependingThread(g, i, dependency_ops[i]) for i in range(3)]
    for t in threads:
      t.start()
      t.has_mutated_graph.wait()
      t.has_mutated_graph.clear()
    for t in threads:
      t.should_continue.set()
      t.join()

    gd = g.as_graph_def()
    self.assertProtoEqualsVersion("""
      node { name: "ColocateWithMe_0" op: "FloatOutput" }
      node { name: "ColocateWithMe_1" op: "FloatOutput" }
      node { name: "ColocateWithMe_2" op: "FloatOutput" }
      node { name: "FloatOutput_0" op: "FloatOutput"
             input: "^ColocateWithMe_0" }
      node { name: "FloatOutput_1" op: "FloatOutput"
             input: "^ColocateWithMe_1" }
      node { name: "FloatOutput_2" op: "FloatOutput"
             input: "^ColocateWithMe_2" }
    """, gd)

  def testNameStack(self):

    class NameSettingThread(self.TestThread):

      def run(self):
        with g.name_scope("foo"):
          op1 = g.create_op("FloatOutput", [], [dtypes.float32])
          self.has_mutated_graph.set()
          self.should_continue.wait()
          self.should_continue.clear()
          op2 = g.create_op("FloatOutput", [], [dtypes.float32])
          self.result = (op1, op2)

    g = ops.Graph()
    threads = [NameSettingThread(g, i) for i in range(3)]
    for t in threads:
      t.start()
      t.has_mutated_graph.wait()
      t.has_mutated_graph.clear()

    for t in threads:
      t.should_continue.set()
      t.join()

    suffixes = ["", "_1", "_2"]
    for t, s in zip(threads, suffixes):
      self.assertEquals("foo" + s + "/FloatOutput", t.result[0].name)
      self.assertEquals("foo" + s + "/FloatOutput_1", t.result[1].name)


class ObjectWithName(object):

  def __init__(self, name):
    self._name = name

  @property
  def name(self):
    return self._name


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

  def test_defun(self):
    with context.eager_mode():

      @eager_function.defun
      def defun():
        ops.add_to_collection("int", 1)
        ops.add_to_collection("tensor", constant_op.constant(2))

        @eager_function.defun
        def inner_defun():
          self.assertEqual(ops.get_collection("int"), [1])
          three = ops.get_collection("tensor")[0] + ops.get_collection("int")[0]
          ops.add_to_collection("int", 2)
          self.assertEqual(ops.get_collection("int"), [1, 2])
          ops.add_to_collection("foo", "bar")
          self.assertEqual(ops.get_collection("foo"), ["bar"])
          return three

        self.assertEqual(ops.get_collection("int"), [1])
        three = inner_defun()
        self.assertEqual(ops.get_collection("int"), [1])
        self.assertEqual(ops.get_collection("foo"), [])
        return three

      three = defun()
      self.assertEqual(three.numpy(), 3)


ops.NotDifferentiable("FloatOutput")


@ops.RegisterGradient("CopyOp")
def _CopyGrad(op, x_grad):  # pylint: disable=invalid-name
  _ = op
  return x_grad


@ops.RegisterGradient("copy_override")
def _CopyOverrideGrad(op, x_grad):  # pylint: disable=invalid-name
  _ = op
  return x_grad


class RegistrationTest(test_util.TensorFlowTestCase):

  @test_util.run_deprecated_v1
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


class ComparisonTest(test_util.TensorFlowTestCase):

  def testMembershipAllowed(self):
    g = ops.Graph()
    t1 = _apply_op(g, "FloatOutput", [], [dtypes.float32], name="myop1")
    t2 = _apply_op(g, "FloatOutput", [], [dtypes.float32], name="myop2")
    self.assertTrue(isinstance(t1, ops.Tensor))
    self.assertTrue(isinstance(t2, ops.Tensor))
    self.assertTrue(t1 in [t1])
    self.assertTrue(t1 not in [t2])


class ControlDependenciesTest(test_util.TensorFlowTestCase):

  @test_util.run_deprecated_v1
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

  @test_util.run_in_graph_and_eager_modes
  def testEager(self):
    def future():
      future.calls += 1
      return constant_op.constant(2.0)
    future.calls = 0

    if context.executing_eagerly():
      a = constant_op.constant(1.0)
      b = future
      with ops.control_dependencies([a, b]):
        c = constant_op.constant(3.0)
      self.assertEqual(future.calls, 1)
    else:
      g = ops.Graph()
      with g.as_default():
        a = constant_op.constant(1.0)
        b = future()
        with g.control_dependencies([a, b]):
          c = constant_op.constant(3.0)
      self.assertEqual(c.op.control_inputs, [a.op, b.op])
      self.assertEqual(future.calls, 1)

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


class OpScopeTest(test_util.TensorFlowTestCase):

  @test_util.run_in_graph_and_eager_modes
  def testNames(self):
    with ops.name_scope("foo") as foo:
      self.assertEqual("foo/", foo)
      with ops.name_scope("foo2") as foo2:
        self.assertEqual("foo/foo2/", foo2)
      with ops.name_scope(None) as empty1:
        self.assertEqual("", empty1)
        with ops.name_scope("foo3") as foo3:
          self.assertEqual("foo3/", foo3)
      with ops.name_scope("") as empty2:
        self.assertEqual("", empty2)
    with ops.name_scope("foo/") as outer_foo:
      self.assertEqual("foo/", outer_foo)
      with ops.name_scope("") as empty3:
        self.assertEqual("", empty3)
      with ops.name_scope("foo4") as foo4:
        self.assertEqual("foo/foo4/", foo4)
      with ops.name_scope("foo5//") as foo5:
        self.assertEqual("foo5//", foo5)
        with ops.name_scope("foo6") as foo6:
          self.assertEqual("foo5//foo6/", foo6)
      with ops.name_scope("/") as foo7:
        self.assertEqual("/", foo7)
      with ops.name_scope("//") as foo8:
        self.assertEqual("//", foo8)
      with ops.name_scope("a//b/c") as foo9:
        self.assertEqual("foo/a//b/c/", foo9)
    with ops.name_scope("a//b/c") as foo10:
      self.assertEqual("a//b/c/", foo10)

  @test_util.run_in_graph_and_eager_modes
  def testEagerDefaultScopeName(self):
    with ops.name_scope(None, "default") as scope:
      self.assertEqual(scope, "default/")
      with ops.name_scope(None, "default2") as scope2:
        self.assertEqual(scope2, "default/default2/")

  @test_util.run_deprecated_v1
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

  @test_util.run_deprecated_v1
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

  @test_util.run_deprecated_v1
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
    with self.assertRaises(TypeError):
      with ops.name_scope(scope_name, [a, b]):
        pass

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

  @test_util.run_deprecated_v1
  def testTensor(self):
    g0 = ops.Graph()
    a = g0.create_op("A", [], [dtypes.float32])
    b = g0.create_op("B", [], [dtypes.float32])
    self._testGraphElements([a, b])

  @test_util.run_deprecated_v1
  def testSparseTensor(self):
    g0 = ops.Graph()
    a = g0.create_op("A", [], [dtypes.float32])
    b = g0.create_op("B", [], [dtypes.float32])
    sparse = sparse_tensor.SparseTensor(
        _apply_op(g0, "Int64Output", [], [dtypes.int64]),
        _apply_op(g0, "FloatOutput", [], [dtypes.float32]),
        _apply_op(g0, "Int64Output", [], [dtypes.int64]))
    self._testGraphElements([a, sparse, b])

  @test_util.run_deprecated_v1
  def testVariable(self):
    g0 = ops.Graph()
    with g0.as_default():
      variable = variables.Variable([1.0])
    a = g0.create_op("A", [], [dtypes.float32])
    b = g0.create_op("B", [], [dtypes.float32])
    self._testGraphElements([a, variable, b])


class InitScopeTest(test_util.TensorFlowTestCase):

  def testClearsControlDependencies(self):
    g = ops.Graph()
    a_1 = _apply_op(g, "FloatOutput", [], [dtypes.float32])
    a_2 = _apply_op(g, "FloatOutput", [], [dtypes.float32])
    a_3 = _apply_op(g, "FloatOutput", [], [dtypes.float32])
    a_4 = _apply_op(g, "FloatOutput", [], [dtypes.float32])

    with g.as_default():
      with g.control_dependencies([a_1]):
        with g.control_dependencies([a_2]):
          with ops.init_scope():
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
        with ops.init_scope():
          # deps are None again
          b_none2 = _apply_op(g, "FloatOutput", [], [dtypes.float32])

    self.assertItemsEqual([a_3.op, a_4.op], b_3_4.op.control_inputs)
    self.assertItemsEqual([a_3.op], b_3.op.control_inputs)
    self.assertItemsEqual([], b_none.op.control_inputs)
    self.assertItemsEqual([a_1.op, a_2.op], b_1_2.op.control_inputs)
    self.assertItemsEqual([a_1.op], b_1.op.control_inputs)
    self.assertItemsEqual([], b_none2.op.control_inputs)

  def testLiftsOpsFromFunctions(self):
    g0 = ops.Graph()
    g1 = ops.Graph()
    g1._building_function = True  # pylint: disable=protected-access
    g2 = ops.Graph()
    g2._building_function = True  # pylint: disable=protected-access

    with g0.as_default():
      with g1.as_default():
        with g2.as_default():
          with ops.init_scope():
            _ = constant_op.constant(1.0)

    self.assertEqual(len(g2.get_operations()), 0)
    self.assertEqual(len(g1.get_operations()), 0)
    self.assertEqual(len(g0.get_operations()), 1)

  def testPreservesDevices(self):
    g0 = ops.Graph()
    with g0.as_default(), ops.device("CPU:0"):
      g1 = ops.Graph()
      g1._building_function = True  # pylint: disable=protected-access
      with g1.as_default(), ops.device("GPU:0"):
        with ops.init_scope():
          # init_scope should preserve device set under `g1`.
          on_gpu = constant_op.constant(1.0)
          self.assertEqual(on_gpu.device, "/device:GPU:0")
        still_on_gpu = constant_op.constant(1.0)
        self.assertEqual(still_on_gpu.device, "/device:GPU:0")
      on_cpu = constant_op.constant(1.0)
      self.assertEqual(on_cpu.device, "/device:CPU:0")

  def testComposes(self):
    g0 = ops.Graph()
    g1 = ops.Graph()
    g1._building_function = True  # pylint: disable=protected-access
    g2 = ops.Graph()
    g2._building_function = True  # pylint: disable=protected-access
    g3 = ops.Graph()
    g3._building_function = False  # pylint: disable=protected-access

    with g0.as_default():
      with g1.as_default():
        with ops.init_scope():
          # This op should be lifted into g0.
          _ = constant_op.constant(1.0)
          self.assertIs(g0, ops.get_default_graph())
          self.assertEqual(len(g2.get_operations()), 0)
          self.assertEqual(len(g1.get_operations()), 0)
          self.assertEqual(len(g0.get_operations()), 1)
        with g2.as_default():
          with ops.init_scope():
            # This op should be lifted into g0.
            _ = constant_op.constant(1.0)
            self.assertIs(g0, ops.get_default_graph())
            with g3.as_default():
              with ops.init_scope():
                # This op should be lifted into g3, because g3 is not building a
                # function.
                _ = constant_op.constant(1.0)
                self.assertIs(g3, ops.get_default_graph())

    self.assertEqual(len(g3.get_operations()), 1)
    self.assertEqual(len(g2.get_operations()), 0)
    self.assertEqual(len(g1.get_operations()), 0)
    self.assertEqual(len(g0.get_operations()), 2)

  def testEscapesToEagerContext(self):
    g = ops.Graph()
    g._building_function = True  # pylint: disable=protected-access
    with context.eager_mode():
      with context.graph_mode():
        with g.as_default():
          with ops.init_scope():
            # Because g is building a function, init_scope should
            # escape out to the eager context.
            self.assertTrue(context.executing_eagerly())
          # g should be reinstated as the default graph, and the
          # graph context should be re-entered.
          self.assertIs(g, ops.get_default_graph())
          self.assertFalse(context.executing_eagerly())

  def testStaysInEagerWhenOnlyEagerContextActive(self):
    with context.eager_mode():
      with ops.init_scope():
        self.assertTrue(context.eager_mode())
      self.assertTrue(context.eager_mode())

  def testEscapesDefunWhenInEagerMode(self):

    def function_with_variables():
      with ops.init_scope():
        self.v = resource_variable_ops.ResourceVariable(3)
      return self.v.assign_add(1)

    with context.eager_mode():
      # Each invocation of function_with_variables recreates a variable.
      self.assertEqual(4, int(function_with_variables()))
      self.assertEqual(4, int(function_with_variables()))

      compiled = eager_function.defun(function_with_variables)
      # The init_scope in function_with_variables lifts the variable out
      # of the graph function constructed by defun; hence,
      # compiled now appears to be stateful.
      self.assertEqual(4, int(compiled()))
      self.assertEqual(5, int(compiled()))

  def testEscapesDefunWhenInGraphMode(self):
    def function_with_variables(name):
      with ops.init_scope():
        _ = variable_scope.get_variable(name, shape=(1,))

    g = ops.Graph()
    with g.as_default():
      with self.cached_session():
        # First ensure that graphs that are not building functions are
        # not escaped.
        function_with_variables("foo")
        with self.assertRaisesRegexp(ValueError,
                                     r"Variable foo already exists.*"):
          # This will fail because reuse is not set to True.
          function_with_variables("foo")

        compiled = eager_function.defun(function_with_variables)
        compiled("bar")
        self.assertEqual(
            len(ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)), 2)

        # The second call to `compiled` should not create variables: the
        # init_scope has lifted the variable creation code out of the defun.
        compiled("bar")
        self.assertEqual(
            len(ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)), 2)

  def testEscapesNestedDefun(self):

    def inner_function():
      with ops.init_scope():
        self.v = resource_variable_ops.ResourceVariable(1)
      return self.v.assign_add(2)

    def outer_function(inner=None):
      with ops.init_scope():
        self.v0 = resource_variable_ops.ResourceVariable(0)
      return self.v0.assign_add(1) + inner()

    with context.eager_mode():
      # Each invocation of outer_function recreates variables.
      self.assertEqual(4, int(outer_function(inner=inner_function)))
      self.assertEqual(4, int(outer_function(inner=inner_function)))

      compiled_inner = eager_function.defun(inner_function)
      compiled_outer = eager_function.defun(outer_function)
      # The init_scope lifts variables out of the graph functions
      # constructed by defun; hence, compiled_outer should now appear to be
      # stateful.
      self.assertEqual(4, int(compiled_outer(inner=compiled_inner)))
      self.assertEqual(7, int(compiled_outer(inner=compiled_inner)))

  @test_util.run_v1_only("b/120545219")
  def testFallsBackToGlobalGraphWhenAllGraphsAreBuildingFunctions(self):
    with context.graph_mode():
      ops.reset_default_graph()
      # This doesn't push anything onto the graph stack, but it does
      # set the stack's global graph.
      global_graph = ops.get_default_graph()
      fn_graph = ops.Graph()

      # pylint: disable=protected-access
      fn_graph._building_function = True
      self.assertEqual(len(ops._default_graph_stack.stack), 0)
      with fn_graph.as_default():
        self.assertEqual(len(ops._default_graph_stack.stack), 1)
        with ops.init_scope():
          self.assertGreater(len(ops._default_graph_stack.stack), 1)
          dummy = constant_op.constant(1.0)
        self.assertEqual(len(ops._default_graph_stack.stack), 1)
      # Note that the global graph is _not_ on the graph stack.
      self.assertEqual(len(ops._default_graph_stack.stack), 0)
      # Ensure that `dummy` was added to the global graph.
      self.assertEqual(global_graph, dummy.graph)
      # pylint: enable=protected-access

  def testInstallsDefaultGraphWhenGraphStackIsEmptyInGraphMode(self):
    with context.graph_mode():
      # pylint: disable=protected-access
      self.assertEqual(len(ops._default_graph_stack.stack), 0)
      with ops.init_scope():
        self.assertGreater(len(ops._default_graph_stack.stack), 0)
      self.assertEqual(len(ops._default_graph_stack.stack), 0)
      # pylint: enable=protected-access

  def testPreservesNameScopeInGraphConstruction(self):
    with ops.Graph().as_default():
      function_graph = ops.Graph()
      with function_graph.as_default():
        with ops.name_scope("inner"), ops.init_scope():
          self.assertEqual(ops.get_name_scope(), "inner")
      self.assertEqual(ops.get_name_scope(), "")

  def testEnteringGraphFromEagerIsSticky(self):
    with context.eager_mode():
      g = ops.Graph()
      with g.as_default():
        with ops.init_scope():
          self.assertFalse(context.executing_eagerly())
          self.assertEqual(g, ops.get_default_graph())

  def testMixGraphEager(self):
    with context.eager_mode():
      c = constant_op.constant(1.0)
      with ops.Graph().as_default():
        with self.assertRaisesRegexp(
            RuntimeError, "Attempting to capture an EagerTensor"):
          math_ops.add(c, c)
        c2 = constant_op.constant(2.0)
      with self.assertRaisesRegexp(
          TypeError, "contains objects other than 'EagerTensor'"):
        math_ops.add(c2, c2)

  def testPreservesNameScopeInEagerExecution(self):
    with context.eager_mode():
      def foo():
        with ops.name_scope("inner"), ops.init_scope():
          if context.executing_eagerly():
            # A trailing slash is always appended when eager execution is
            # enabled.
            self.assertEqual(context.context().scope_name, "inner/")
          else:
            self.assertEqual(ops.get_name_scope(), "inner")

      foo()
      self.assertEqual(ops.get_name_scope(), "")
      foo_compiled = eager_function.defun(foo)
      foo_compiled()
      self.assertEqual(ops.get_name_scope(), "")

  def testExecutingEagerlyOutsideFunctions(self):

    @eager_function.defun
    def f():
      return ops.executing_eagerly_outside_functions()

    with context.eager_mode():
      self.assertTrue(ops.executing_eagerly_outside_functions())
      self.assertTrue(f())
      g = ops.Graph()
      with g.as_default():
        self.assertFalse(ops.executing_eagerly_outside_functions())


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

  def testGraphContextManagerCancelsEager(self):
    with context.eager_mode():
      with ops.Graph().as_default():
        self.assertFalse(context.executing_eagerly())

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

  @test_util.run_deprecated_v1
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
      self.evaluate(c)
    # Delete all references and trigger gc
    del g
    del a
    del b
    del c
    del sess
    gc.collect()
    self.assertIsNone(g_ref())

  def testRunnableAfterInvalidShape(self):
    with ops.Graph().as_default():
      with self.assertRaises(ValueError):
        math_ops.add([1, 2], [1, 2, 3])
      a = constant_op.constant(1)
      with session.Session() as sess:
        self.evaluate(a)

  def testRunnableAfterInvalidShapeWithKernelLabelMap(self):
    g = ops.Graph()
    with g.as_default():
      with g._kernel_label_map({"KernelLabelRequired": "overload_1"}):
        with self.assertRaises(ValueError):
          test_ops.kernel_label_required(1)
      a = constant_op.constant(1)
      with session.Session() as sess:
        self.evaluate(a)


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
    return (a, b)

  @test_util.run_deprecated_v1
  def testNoLabel(self):
    with self.cached_session():
      self.assertAllEqual((None, None), self._get_test_attrs())

  @test_util.run_deprecated_v1
  def testLabelMap(self):
    with self.cached_session() as sess:
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


class KernelLabelTest(test_util.TensorFlowTestCase):

  @test_util.run_deprecated_v1
  def testNoLabel(self):
    with self.cached_session():
      self.assertAllEqual(b"My label is: default",
                          test_ops.kernel_label().eval())

  @test_util.run_deprecated_v1
  def testLabelMap(self):
    with self.cached_session() as sess:
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

      self.assertAllEqual(b"My label is: default", self.evaluate(default_1))
      self.assertAllEqual(b"My label is: default", self.evaluate(default_2))
      self.assertAllEqual(b"My label is: default", self.evaluate(default_3))
      self.assertAllEqual(b"My label is: overload_1",
                          self.evaluate(overload_1_1))
      self.assertAllEqual(b"My label is: overload_1",
                          self.evaluate(overload_1_2))
      self.assertAllEqual(b"My label is: overload_2", self.evaluate(overload_2))


class AsGraphDefTest(test_util.TensorFlowTestCase):

  def testGraphDefVersion(self):
    """Test that the graphdef version is plumbed through to kernels."""
    with ops.Graph().as_default() as g:
      version = g.graph_def_versions.producer
      with self.session(graph=g):
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

      b = constant_op.constant(1.0)  # pylint: disable=unused-variable

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
    node { name: "Const" op: "Const"
      attr {
        key: "_output_shapes"
        value {
          list {
            shape { }
          }
        }
      }
      attr {
        key: "dtype"
        value { type: DT_FLOAT }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_FLOAT
            tensor_shape { }
         float_val: 1.0  } } } }
      """, gd)


@ops.RegisterStatistics("a", "flops")
def _calc_a_forward_flops(unused_graph, unused_node):
  return ops.OpStats("flops", 20)


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


class DeviceStackTest(test_util.TensorFlowTestCase):

  @test_util.run_deprecated_v1
  def testBasicDeviceAssignmentMetadata(self):

    def device_func(unused_op):
      return "/cpu:*"

    const_zero = constant_op.constant([0.0], name="zero")
    with ops.device("/cpu"):
      const_one = constant_op.constant([1.0], name="one")
      with ops.device("/cpu:0"):
        const_two = constant_op.constant([2.0], name="two")
    with ops.device(device_func):
      const_three = constant_op.constant(3.0, name="three")

    self.assertEqual(0, len(const_zero.op._device_assignments))

    one_list = const_one.op._device_assignments
    self.assertEqual(1, len(one_list))
    self.assertEqual("/cpu", one_list[0].obj)
    self.assertEqual("ops_test.py", os.path.basename(one_list[0].filename))

    two_list = const_two.op._device_assignments
    self.assertEqual(2, len(two_list))
    devices = [t.obj for t in two_list]
    self.assertEqual(set(["/cpu", "/cpu:0"]), set(devices))

    three_list = const_three.op._device_assignments
    self.assertEqual(1, len(three_list))
    func_description = three_list[0].obj
    expected_regex = r"device_func<.*ops_test.py, [0-9]+"
    self.assertRegexpMatches(func_description, expected_regex)

  @test_util.run_deprecated_v1
  def testDeviceAssignmentMetadataForGraphDeviceAndTfDeviceFunctions(self):

    with ops.device("/cpu"):
      const_one = constant_op.constant([1.0], name="one")
    with ops.get_default_graph().device("/cpu"):
      const_two = constant_op.constant([2.0], name="two")

    one_metadata = const_one.op._device_assignments[0]
    two_metadata = const_two.op._device_assignments[0]

    # Verify both types of device assignment return the right stack info.
    self.assertRegexpMatches("ops_test.py",
                             os.path.basename(one_metadata.filename))
    self.assertEqual(one_metadata.filename, two_metadata.filename)
    self.assertEqual(one_metadata.lineno + 2, two_metadata.lineno)


class ColocationGroupTest(test_util.TensorFlowTestCase):

  @test_util.run_deprecated_v1
  def testBasic(self):
    a = constant_op.constant([2.0], name="a")
    with ops.colocate_with(a.op):
      b = constant_op.constant(3.0)
    c = constant_op.constant(4.0)
    self.assertEqual([b"loc:@a"], a.op.colocation_groups())
    self.assertEqual([b"loc:@a"], b.op.colocation_groups())
    with self.assertRaises(ValueError):
      c.op.get_attr("_class")

  @test_util.run_deprecated_v1
  def testBasicColocationMetadata(self):
    const_two = constant_op.constant([2.0], name="two")
    with ops.colocate_with(const_two.op):
      const_three = constant_op.constant(3.0, name="three")
    locations_dict = const_three.op._colocation_dict
    self.assertIn("two", locations_dict)
    metadata = locations_dict["two"]
    self.assertIsNone(metadata.obj)
    # Check that this test's filename is recorded as the file containing the
    # colocation statement.
    self.assertEqual("ops_test.py", os.path.basename(metadata.filename))

  @test_util.run_deprecated_v1
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

  @test_util.run_deprecated_v1
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

  @test_util.run_deprecated_v1
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

  @test_util.run_deprecated_v1
  def testNestedColocateWith(self):
    a = constant_op.constant([2.0], name="a")
    with ops.colocate_with(a.op):
      b = constant_op.constant(3.0)
      with ops.colocate_with(b.op):
        c = constant_op.constant(4.0)
    self.assertEqual([b"loc:@a"], b.op.colocation_groups())
    self.assertEqual([b"loc:@a"], c.op.colocation_groups())

  @test_util.run_deprecated_v1
  def testMultiColocationGroups(self):
    a = constant_op.constant([2.0], name="a")
    b = constant_op.constant(3.0, name="b")
    with ops.colocate_with(a.op):
      with ops.colocate_with(b.op):
        c = constant_op.constant(4.0)
    self.assertEqual(set([b"loc:@a", b"loc:@b"]), set(c.op.colocation_groups()))

  @test_util.run_deprecated_v1
  def testColocationIgnoreStack(self):
    a = constant_op.constant([2.0], name="a")
    b = constant_op.constant(3.0, name="b")
    with ops.colocate_with(a.op):
      with ops.colocate_with(b.op, ignore_existing=True):
        c = constant_op.constant(4.0)
    self.assertEqual(set([b"loc:@b"]), set(c.op.colocation_groups()))

  @test_util.run_deprecated_v1
  def testColocateWithReset(self):
    a = constant_op.constant([2.0], name="a")
    with ops.colocate_with(a.op):
      b = constant_op.constant(3.0, name="b")
      with ops.colocate_with(None, ignore_existing=True):
        c = constant_op.constant(4.0, name="c")
    self.assertEqual([b"loc:@a"], b.op.colocation_groups())
    self.assertEqual([b"loc:@c"], c.op.colocation_groups())

  @test_util.run_deprecated_v1
  def testColocateWithInitialNoneThenNested(self):
    a = constant_op.constant([2.0], name="a")
    with ops.colocate_with(a.op):
      with ops.colocate_with(None, ignore_existing=True):
        b = constant_op.constant(3.0, name="b")
        with ops.colocate_with(b.op):
          c = constant_op.constant(4.0, name="c")
    self.assertEqual([b"loc:@b"], b.op.colocation_groups())
    self.assertEqual([b"loc:@b"], c.op.colocation_groups())

  @test_util.run_deprecated_v1
  def testColocateVariables(self):
    a = variables.Variable([2.0], name="a")
    with ops.colocate_with(a.op):
      b = variables.Variable([3.0], name="b")
    self.assertEqual([b"loc:@a"], b.op.colocation_groups())


class DeprecatedTest(test_util.TensorFlowTestCase):

  def testSuccess(self):
    with ops.Graph().as_default() as g:
      test_util.set_producer_version(g, 7)
      old = test_ops.old()
      with self.session(graph=g):
        old.run()

  def _error(self):
    return ((r"Op Old is not available in GraphDef version %d\. "
             r"It has been removed in version 8\. For reasons\.") %
            versions.GRAPH_DEF_VERSION)

  def testGraphConstructionFail(self):
    with ops.Graph().as_default():
      with self.assertRaisesRegexp(NotImplementedError, self._error()):
        test_ops.old()


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

  def testTwoGraphs(self):

    def f():
      g1 = ops.Graph()
      g2 = ops.Graph()
      with g1.as_default():
        with g2.as_default():
          with ops.name_scope("_"):
            pass

    self.assertRaisesRegexp(ValueError, "'_' is not a valid scope name", f)


class TracebackTest(test_util.TensorFlowTestCase):

  @test_util.run_deprecated_v1
  def testTracebackWithStartLines(self):
    with self.cached_session() as sess:
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


class EnableEagerExecutionTest(test_util.TensorFlowTestCase):

  @test_util.run_v1_only("b/120545219")
  def testBadArgumentsToEnableEagerExecution(self):
    with self.assertRaisesRegexp(TypeError, "config must be a tf.ConfigProto"):
      ops.enable_eager_execution(context.DEVICE_PLACEMENT_SILENT)
    with self.assertRaisesRegexp(ValueError, "device_policy must be one of"):
      c = config_pb2.ConfigProto()
      ops.enable_eager_execution(c, c)
    with self.assertRaisesRegexp(ValueError, "execution_mode must be one of"):
      c = config_pb2.ConfigProto()
      ops.enable_eager_execution(c, execution_mode=c)


if __name__ == "__main__":
  googletest.main()
