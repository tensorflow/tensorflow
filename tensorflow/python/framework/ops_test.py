# Copyright 2015 Google Inc. All Rights Reserved.
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

import tensorflow.python.platform

from tensorflow.python.framework import device as pydev
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_kernel_label_op
from tensorflow.python.framework import test_util
from tensorflow.python.ops import common_shapes
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest


class TensorTest(test_util.TensorFlowTestCase):

  def testShape(self):
    op = ops.Operation(ops._NodeDef("noop", "myop"), ops.Graph(),
                       [], [dtypes.float32])
    t = op.outputs[0]
    self.assertEqual(tensor_shape.unknown_shape(), t.get_shape())
    t.set_shape([1, 2, 3])
    self.assertEqual([1, 2, 3], t.get_shape())


class NodeDefConstructorTest(test_util.TensorFlowTestCase):

  def testNoArgs(self):
    nodedef = ops._NodeDef("noop", "bar")
    self.assertProtoEquals("op: 'noop' name: 'bar'", nodedef)

  def testArgs(self):
    nodedef = ops._NodeDef("foo", "bar", device="/device:baz:*")
    self.assertProtoEquals("op:'foo' name:'bar' device:'/device:baz:*'",
                           nodedef)
    nodedef = ops._NodeDef("foo", "bar", device=pydev.Device(job="j"))
    self.assertProtoEquals("op:'foo' name:'bar' device:'/job:j'", nodedef)


# NOTE(mrry): Dummy shape registrations for ops used in the tests.
ops.RegisterShape("a")(None)
ops.RegisterShape("b")(None)
ops.RegisterShape("c")(None)
ops.RegisterShape("add")(None)
ops.RegisterShape("an_op")(None)
ops.RegisterShape("const")(None)
ops.RegisterShape("copy")(None)
ops.RegisterShape("foo")(None)
ops.RegisterShape("identity")(None)
ops.RegisterShape("mul")(None)
ops.RegisterShape("nonrefop")(None)
ops.RegisterShape("noop")(None)
ops.RegisterShape("refop")(None)


def _apply_op(g, *args, **kwargs):
  op = g.create_op(*args, **kwargs)
  if len(op.outputs) == 1:
    return op.outputs[0]
  else:
    return op.outputs


class OperationTest(test_util.TensorFlowTestCase):

  def testNoInputs(self):
    op = ops.Operation(ops._NodeDef("noop", "myop"), ops.Graph(),
                       [],
                       [dtypes.float32, dtypes.string])
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

    self.assertProtoEquals("op:'noop' name:'myop'", op.node_def)

  def testNoOutputs(self):
    g = ops.Graph()
    op1 = ops.Operation(
        ops._NodeDef("noop", "myop1"), g, [], [dtypes.float32])
    float_t, = op1.values()
    op2 = ops.Operation(ops._NodeDef("reop", "myop2"), g, [float_t], [])
    self.assertEqual(0, len(op2.values()))
    self.assertEqual(1, len(op2.inputs))
    self.assertIs(float_t, op2.inputs[0])

    self.assertEqual(1, len(float_t._consumers))
    self.assertEqual(op2, float_t._consumers[0])

    self.assertProtoEquals("op:'noop' name:'myop1'", op1.node_def)
    self.assertProtoEquals("op:'reop' name:'myop2' input:'myop1'",
                           op2.node_def)

  def testInputsAndOutputs(self):
    g = ops.Graph()
    op1 = ops.Operation(
        ops._NodeDef("noop", "myop1"), g, [], [dtypes.float32])
    self.assertEqual(1, len(op1.values()))
    float1_t, = op1.values()

    op2 = ops.Operation(ops._NodeDef("reop", "myop2"), g,
                        [], [dtypes.float32, dtypes.string])
    self.assertEqual(2, len(op2.values()))
    float2_t, label2_str_t = op2.values()

    # Note that we consume label2_str_t twice here.
    op3 = ops.Operation(ops._NodeDef("add", "myop3"), g,
                        [float1_t, label2_str_t, label2_str_t],
                        [dtypes.float32, dtypes.int32])
    self.assertEqual(2, len(op3.values()))

    self.assertEqual(1, len(float1_t._consumers))
    self.assertEqual(op3, float1_t._consumers[0])

    self.assertEqual(0, len(float2_t._consumers))

    self.assertEqual(2, len(label2_str_t._consumers))
    self.assertEqual(op3, label2_str_t._consumers[0])
    self.assertEqual(op3, label2_str_t._consumers[1])

    self.assertProtoEquals("""
    op:'add' name:'myop3'
    input:'myop1' input:'myop2:1' input:'myop2:1'
    """, op3.node_def)

  def testDeviceObject(self):
    op = ops.Operation(ops._NodeDef("noop", "myop"), ops.Graph(), [], [])
    op._set_device("/job:goo/device:GPU:0")
    self.assertProtoEquals(
        "op:'noop' name:'myop' device:'/job:goo/device:GPU:0' ",
        op.node_def)
    op = ops.Operation(ops._NodeDef("noop", "op2"), ops.Graph(), [], [])
    op._set_device(pydev.Device(job="muu", device_type="CPU", device_index=0))
    self.assertProtoEquals(
        "op:'noop' name:'op2' device:'/job:muu/device:CPU:0'",
        op.node_def)

  def testReferenceInput(self):
    g = ops.Graph()
    op1 = ops.Operation(ops._NodeDef("noop", "op1"), g, [],
                        [dtypes.float32_ref, dtypes.float32])
    self.assertProtoEquals("op:'noop' name:'op1'",
                           op1.node_def)
    ref_t, nonref_t = op1.values()
    # NOTE(mrry): Must specify input_types to preserve ref-typed input.
    op2 = ops.Operation(
        ops._NodeDef("refop", "op2"), g, [ref_t, nonref_t], [],
        input_types=[dtypes.float32_ref, dtypes.float32])
    self.assertProtoEquals("op:'refop' name:'op2' input:'op1' input:'op1:1'",
                           op2.node_def)
    op3 = ops.Operation(
        ops._NodeDef("nonrefop", "op3"), g, [ref_t, nonref_t], [])
    self.assertProtoEquals("op:'nonrefop' name:'op3' input:'op1' input:'op1:1'",
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

  def testShapeFunctionAbsence(self):
    def _test():
      pass
    g = ops.Graph()
    with self.assertRaises(RuntimeError):
      g.create_op("shapeless_op", [], [dtypes.float32])

  def testNoShapeFunction(self):
    g = ops.Graph()
    op = ops.Operation(ops._NodeDef("op", "an_op"), g,
                       output_types = [dtypes.float32])
    self.assertEqual(tensor_shape.unknown_shape(),
                     _apply_op(g, "an_op", [], [dtypes.float32]).get_shape())

class CreateOpTest(test_util.TensorFlowTestCase):

  def testNodeDefArgs(self):
    g = ops.Graph()
    op1 = g.create_op("const", [], [dtypes.float32], None, name="myop1")
    with g.device("/device:GPU"):
      op2 = g.create_op("add",
                        [],
                        [dtypes.float32, dtypes.string], None,
                        name="myop2")
    op3 = g.create_op("foo",
                      [list(op1.values())[0], list(op2.values())[1],
                       list(op2.values())[0]],
                      [dtypes.float32, dtypes.int32],
                      None,
                      name="myop3")
    self.assertEqual(None, op1.device)
    self.assertEqual("/device:GPU", op2.device)
    self.assertEqual(None, op3.device)
    self.assertProtoEquals("name:'myop1' op:'const'", op1.node_def)
    self.assertProtoEquals("name:'myop2' op:'add' device:'/device:GPU'",
                           op2.node_def)
    self.assertProtoEquals(
        "name:'myop3' input:'myop1' input:'myop2:1' input:'myop2' op:'foo'",
        op3.node_def)

  def testReferenceInput(self):
    g = ops.Graph()
    op1 = g.create_op("noop", [],
                      [dtypes.float32_ref, dtypes.float32], name="op1")
    self.assertProtoEquals("op:'noop' name:'op1'", op1.node_def)
    ref_t, nonref_t = op1.values()
    # NOTE(mrry): Must specify input_types to preserve ref-typed input.
    op2 = g.create_op("refop", [ref_t, nonref_t], [],
                      input_types=[dtypes.float32_ref, dtypes.float32],
                      name="op2")
    self.assertProtoEquals("op:'refop' name:'op2' input:'op1' input:'op1:1'",
                           op2.node_def)
    op3 = g.create_op("nonrefop", [ref_t, nonref_t], [], name="op3")
    self.assertProtoEquals("op:'nonrefop' name:'op3' input:'op1' input:'op1:1'",
                           op3.node_def)

  def testFinalized(self):
    g = ops.Graph()
    g.finalize()
    with self.assertRaises(RuntimeError):
      g.create_op("const", [], [dtypes.float32], None, name="myop1")


class ApplyOpTest(test_util.TensorFlowTestCase):

  def testNodeDefArgs(self):
    g = ops.Graph()
    t1 = _apply_op(g, "const", [], [dtypes.float32], name="myop1")
    with g.device("/device:GPU"):
      t2 = _apply_op(g, "add",
                     [],
                     [dtypes.float32, dtypes.string],
                     name="myop2")
    t3 = _apply_op(g, "foo", [t1, t2[1], t2[0]],
                   [dtypes.float32, dtypes.int32], name="myop3")
    self.assertTrue(isinstance(t1, ops.Tensor))
    self.assertTrue(isinstance(t2, list))
    self.assertTrue(isinstance(t3, list))
    self.assertTrue(isinstance(t3[0], ops.Tensor))
    self.assertEqual("myop1", t1._as_node_def_input())
    self.assertEqual("myop2", t2[0]._as_node_def_input())
    self.assertEqual("myop2:1", t2[1]._as_node_def_input())
    self.assertEqual("myop3", t3[0]._as_node_def_input())
    # Validate that we got the right ops as well
    self.assertProtoEquals("name:'myop1' op:'const'", t1.op.node_def)
    self.assertProtoEquals("name:'myop2' op:'add' device:'/device:GPU'",
                           t2[0].op.node_def)
    self.assertProtoEquals(
        "name:'myop3' input:'myop1' input:'myop2:1' input:'myop2' op:'foo'",
        t3[0].op.node_def)

  def testReferenceInput(self):
    g = ops.Graph()
    ref_t, nonref_t = _apply_op(
        g, "noop", [], [dtypes.float32_ref, dtypes.float32], name="op1")
    self.assertProtoEquals("op:'noop' name:'op1'", ref_t.op.node_def)
    # NOTE(mrry): Must specify input_types to preserve ref-typed input.
    out_2 = _apply_op(g, "refop", [ref_t, nonref_t], [dtypes.int32],
                      input_types=[dtypes.float32_ref, dtypes.float32],
                      name="op2")
    self.assertProtoEquals("op:'refop' name:'op2' input:'op1' input:'op1:1'",
                           out_2.op.node_def)
    out_3 = _apply_op(g, "nonrefop", [ref_t, nonref_t], [dtypes.int32],
                      name="op3")
    self.assertProtoEquals("op:'nonrefop' name:'op3' input:'op1' input:'op1:1'",
                           out_3.op.node_def)


class NameStackTest(test_util.TensorFlowTestCase):

  def testBasics(self):
    g = ops.Graph()
    self.assertEqual("foo", g.unique_name("foo"))
    self.assertEqual("foo_1", g.unique_name("foo"))
    self.assertEqual("foo_2", g.unique_name("foo"))
    self.assertEqual("foo_1_1", g.unique_name("foo_1"))
    self.assertEqual("foo_1_2", g.unique_name("foo_1"))
    self.assertEqual("foo_1_2_1", g.unique_name("foo_1_2"))
    with g.name_scope("bar"):
      self.assertEqual("bar/foo", g.unique_name("foo"))
      self.assertEqual("bar/foo_1", g.unique_name("foo"))
      with g.name_scope(None):
        self.assertEqual("foo_3", g.unique_name("foo"))
      with g.name_scope("baz"):
        self.assertEqual("bar/baz/foo", g.unique_name("foo"))
        self.assertEqual("bar/baz/foo_1", g.unique_name("foo"))
      with g.name_scope("baz"):
        self.assertEqual("bar/baz_1/foo", g.unique_name("foo"))
        self.assertEqual("bar/baz_1/foo_1", g.unique_name("foo"))
    with g.name_scope("quux"):
      self.assertEqual("quux/foo", g.unique_name("foo"))
    with g.name_scope("bar"):
      with g.name_scope("baz"):
        self.assertEqual("bar_1/baz/foo", g.unique_name("foo"))
    self.assertEqual("foo_4", g.unique_name("foo"))
    self.assertEqual("bar_2", g.unique_name("bar"))

  def testOutOfOrderUniqueName(self):
    g = ops.Graph()
    self.assertEqual("foo_2", g.unique_name("foo_2"))
    self.assertEqual("foo", g.unique_name("foo"))
    self.assertEqual("foo_1", g.unique_name("foo"))
    self.assertEqual("foo_3", g.unique_name("foo"))


class NameTest(test_util.TensorFlowTestCase):

  def testGenerateName(self):
    g = ops.Graph()
    op0 = g.create_op("const", [], [dtypes.float32, dtypes.float32])
    self.assertEqual("const", op0.name)
    self.assertEqual("const:0", op0.outputs[0].name)
    self.assertEqual("const:1", op0.outputs[1].name)

    op1 = g.create_op("const", [], [dtypes.float32])
    self.assertEqual("const_1", op1.name)
    self.assertEqual("const_1:0", op1.outputs[0].name)

    op2 = g.create_op("const", [], [dtypes.float32], name="my_op")
    self.assertEqual("my_op", op2.name)
    self.assertEqual("my_op:0", op2.outputs[0].name)

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

    self.assertEqual("const",
                     g.create_op("const", [], [dtypes.float32]).name)
    with g.name_scope("bar") as scope:
      self.assertEqual("bar/const",
                       g.create_op("const", [], [dtypes.float32]).name)
      self.assertEqual("bar/const_1",
                       g.create_op("const", [], [dtypes.float32]).name)
      # If you use the value from "with .. as", that values is used as-is.
      self.assertEqual(
          "bar",
          g.create_op("const", [], [dtypes.float32], name=scope).name)
    with g.name_scope("baz") as scope:
      with g.name_scope("quux"):
        self.assertEqual("baz/quux/const",
                         g.create_op("const", [], [dtypes.float32]).name)
      # If you use the value from the enclosing "with .. as", nothing is pushed.
      with g.name_scope(scope):
        self.assertEqual("baz/const",
                         g.create_op("const", [], [dtypes.float32]).name)
        self.assertEqual("baz",
                         g.create_op("const", [], [dtypes.float32],
                                     name=scope).name)
        self.assertEqual("trailing",
                         g.create_op("const", [], [dtypes.float32],
                                     name="trailing/").name)
    with g.name_scope("bar"):
      self.assertEqual("bar_1/const",
                       g.create_op("const", [], [dtypes.float32]).name)
    with g.name_scope("bar/"):
      self.assertEqual("bar/const_2",
                       g.create_op("const", [], [dtypes.float32]).name)


class DeviceTest(test_util.TensorFlowTestCase):

  def testNoDevice(self):
    g = ops.Graph()
    op = g.create_op("an_op", [], [dtypes.float32])
    self.assertEqual(None, op.device)
    gd = g.as_graph_def()
    self.assertProtoEqualsVersion("""
      node { name: "an_op" op: "an_op" }
    """, gd)

  def testDevicePartialString(self):
    g = ops.Graph()
    with g.device("/job:worker/replica:2"):
      g.create_op("an_op", [], [dtypes.float32])
    gd = g.as_graph_def()
    self.assertProtoEqualsVersion("""
      node { name: "an_op" op: "an_op" device: "/job:worker/replica:2" }
    """, gd)

  def testDeviceFull(self):
    g = ops.Graph()
    with g.device(pydev.Device(job="worker", replica=2, task=0,
                               device_type="CPU",
                               device_index=3)):
      g.create_op("an_op", [], [dtypes.float32])
    gd = g.as_graph_def()
    self.assertProtoEqualsVersion("""
      node { name: "an_op" op: "an_op"
             device: "/job:worker/replica:2/task:0/device:CPU:3" }
    """, gd)

  def testNesting(self):
    g = ops.Graph()
    with g.device("/job:worker/replica:2"):
      g.create_op("an_op", [], [dtypes.float32])
      with g.device("/job:worker/replica:3/task:0"):
        g.create_op("an_op", [], [dtypes.float32])
      g.create_op("an_op", [], [dtypes.float32])
    gd = g.as_graph_def()
    self.assertProtoEqualsVersion("""
      node { name: "an_op" op: "an_op"
             device: "/job:worker/replica:2" }
      node { name: "an_op_1" op: "an_op"
             device: "/job:worker/replica:3/task:0" }
      node { name: "an_op_2" op: "an_op"
             device: "/job:worker/replica:2" }
    """, gd)

  def testNestingString(self):
    g = ops.Graph()
    with g.device("/job:worker/replica:2"):
      g.create_op("an_op", [], [dtypes.float32])
      with g.device("/job:worker/replica:3/task:0"):
        g.create_op("an_op", [], [dtypes.float32])
      g.create_op("an_op", [], [dtypes.float32])
    gd = g.as_graph_def()
    self.assertProtoEqualsVersion("""
      node { name: "an_op" op: "an_op"
             device: "/job:worker/replica:2" }
      node { name: "an_op_1" op: "an_op"
             device: "/job:worker/replica:3/task:0" }
      node { name: "an_op_2" op: "an_op"
             device: "/job:worker/replica:2" }
    """, gd)

  def testNestingOverrideGpuCpu(self):
    g = ops.Graph()
    with g.device("/job:worker/replica:2/device:CPU:1"):
      g.create_op("an_op", [], [dtypes.float32])
      with g.device("/job:worker/replica:2/device:GPU:2"):
        g.create_op("an_op", [], [dtypes.float32])
      g.create_op("an_op", [], [dtypes.float32])
    gd = g.as_graph_def()
    self.assertProtoEqualsVersion("""
      node { name: "an_op" op: "an_op"
             device: "/job:worker/replica:2/device:CPU:1"  }
      node { name: "an_op_1" op: "an_op"
             device: "/job:worker/replica:2/device:GPU:2" }
      node { name: "an_op_2" op: "an_op"
             device: "/job:worker/replica:2/device:CPU:1" }
    """, gd)

  def testNestingWithMergeDeviceFunction(self):
    g = ops.Graph()

    with g.device(pydev.merge_device("/device:GPU:0")):
      g.create_op("an_op", [], [dtypes.float32])
      with g.device(pydev.merge_device("/job:worker")):
        g.create_op("an_op", [], [dtypes.float32])
        with g.device(pydev.merge_device("/device:CPU:0")):
          g.create_op("an_op", [], [dtypes.float32])
          with g.device(pydev.merge_device("/job:ps")):
            g.create_op("an_op", [], [dtypes.float32])
            with g.device(pydev.merge_device(None)):
              g.create_op("an_op", [], [dtypes.float32])

    gd = g.as_graph_def()
    self.assertProtoEqualsVersion("""
      node { name: "an_op" op: "an_op"
             device: "/device:GPU:0" }
      node { name: "an_op_1" op: "an_op"
             device: "/job:worker/device:GPU:0" }
      node { name: "an_op_2" op: "an_op"
             device: "/job:worker/device:CPU:0" }
      node { name: "an_op_3" op: "an_op"
             device: "/job:ps/device:CPU:0" }
      node { name: "an_op_4" op: "an_op"
             device: "/job:ps/device:CPU:0" }
    """, gd)

  def testNoneClearsDefault(self):
    g = ops.Graph()
    with g.device("/job:worker/replica:2/device:CPU:1"):
      g.create_op("an_op", [], [dtypes.float32])
      with g.device(None):
        g.create_op("an_op", [], [dtypes.float32])
      g.create_op("an_op", [], [dtypes.float32])
    gd = g.as_graph_def()
    self.assertProtoEqualsVersion("""
      node { name: "an_op" op: "an_op"
             device: "/job:worker/replica:2/device:CPU:1" }
      node { name: "an_op_1" op: "an_op" }
      node { name: "an_op_2" op: "an_op"
             device: "/job:worker/replica:2/device:CPU:1" }
    """, gd)


class ObjectWithName(object):

  def __init__(self, name):
    self._name = name

  @property
  def name(self):
    return self._name


class CollectionTest(test_util.TensorFlowTestCase):

  def testadd_to_collection(self):
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

    self.assertEqual(["foo"], g.get_collection("other"))
    self.assertEqual([12, 34], g.get_collection("key"))
    self.assertEqual([], g.get_collection("nothing"))
    self.assertEqual([27, blank1, blank2], g.get_collection("blah"))
    self.assertEqual([blank1], g.get_collection("blah", "prefix"))

  def testDefaulGraph(self):
    with ops.Graph().as_default():
      ops.add_to_collection("key", 90)
      ops.add_to_collection("key", 100)
      # Collections are ordered.
      self.assertEqual([90, 100], ops.get_collection("key"))


def an_op(g):
  return _apply_op(g, "an_op", [], [dtypes.float32])


ops.NoGradient("an_op")


def copy_op(x):
  return _apply_op(x.graph, "copy", [x], [x.dtype])


@ops.RegisterGradient("copy")
def _CopyGrad(op, x_grad):
  _ = op
  return x_grad


@ops.RegisterGradient("copy_override")
def _CopyOverrideGrad(op, x_grad):
  _ = op
  return x_grad


class RegistrationTest(test_util.TensorFlowTestCase):

  def testRegisterGradients(self):
    g = ops.Graph()
    x = an_op(g)
    y = copy_op(x)
    fn = ops.get_gradient_function(y.op)
    self.assertEqual(_CopyGrad, fn)

  def testOverrideGradients(self):
    g = ops.Graph()
    x = an_op(g)
    with g.gradient_override_map({"copy": "copy_override"}):
      y = copy_op(x)
    fn = ops.get_gradient_function(y.op)
    self.assertEqual(_CopyOverrideGrad, fn)

  def testNonExistentOverride(self):
    g = ops.Graph()
    x = an_op(g)
    with g.gradient_override_map({"copy": "unknown_override"}):
      y = copy_op(x)
    with self.assertRaisesRegexp(LookupError, "unknown_override"):
      fn = ops.get_gradient_function(y.op)


class ComparisonTest(test_util.TensorFlowTestCase):

  def testMembershipAllowed(self):
    g = ops.Graph()
    t1 = _apply_op(g, "const", [], [dtypes.float32], name="myop1")
    t2 = _apply_op(g, "const", [], [dtypes.float32], name="myop2")
    self.assertTrue(isinstance(t1, ops.Tensor))
    self.assertTrue(isinstance(t2, ops.Tensor))
    self.assertTrue(t1 in [t1])
    self.assertTrue(t1 not in [t2])


class ControlDependenciesTest(test_util.TensorFlowTestCase):

  def testBasic(self):
    g = ops.Graph()
    a = _apply_op(g, "const", [], [dtypes.float32])
    b = _apply_op(g, "const", [], [dtypes.float32])
    with g.control_dependencies([a]):
      c = _apply_op(g, "const", [], [dtypes.float32])
      d = _apply_op(g, "identity", [b], [dtypes.float32])
      e = _apply_op(g, "identity", [c], [dtypes.float32])

    self.assertEqual(c.op.control_inputs, [a.op])
    self.assertEqual(d.op.control_inputs, [a.op])
    # e should be dominated by c.
    self.assertEqual(e.op.control_inputs, [])

  def testBasicWithConversion(self):
    g = ops.Graph()
    a = _apply_op(g, "const", [], [dtypes.float32])

    class ConvertibleObj(object):

      def _as_graph_element(self):
        return a

    with g.control_dependencies([ConvertibleObj()]):
      c = _apply_op(g, "const", [], [dtypes.float32])

    self.assertEqual(c.op.control_inputs, [a.op])

  def testNested(self):
    g = ops.Graph()
    a_1 = _apply_op(g, "const", [], [dtypes.float32])
    a_2 = _apply_op(g, "const", [], [dtypes.float32])
    a_3 = _apply_op(g, "const", [], [dtypes.float32])
    a_4 = _apply_op(g, "const", [], [dtypes.float32])

    with g.control_dependencies([a_1, a_2, a_3, a_4]):
      b_1 = _apply_op(g, "const", [], [dtypes.float32])

    with g.control_dependencies([a_1]):
      with g.control_dependencies([a_2]):
        with g.control_dependencies([a_3]):
          with g.control_dependencies([a_4]):
            b_2 = _apply_op(g, "const", [], [dtypes.float32])

    self.assertItemsEqual(
        [a_1.op, a_2.op, a_3.op, a_4.op], b_1.op.control_inputs)
    self.assertItemsEqual(b_1.op.control_inputs, b_2.op.control_inputs)

  def testClear(self):
    g = ops.Graph()
    a_1 = _apply_op(g, "const", [], [dtypes.float32])
    a_2 = _apply_op(g, "const", [], [dtypes.float32])
    a_3 = _apply_op(g, "const", [], [dtypes.float32])
    a_4 = _apply_op(g, "const", [], [dtypes.float32])

    with g.control_dependencies([a_1]):
      with g.control_dependencies([a_2]):
        with g.control_dependencies(None):
          with g.control_dependencies([a_3]):
            with g.control_dependencies([a_4]):
              # deps [a_3, a_4]
              b_3_4 = _apply_op(g, "const", [], [dtypes.float32])
            # deps = [a_3]
            b_3 = _apply_op(g, "const", [], [dtypes.float32])
          # deps back to None
          b_none = _apply_op(g, "const", [], [dtypes.float32])
        # deps back to [a_1, a_2]
        b_1_2 = _apply_op(g, "const", [], [dtypes.float32])
      # deps back to [a_1]
      b_1 = _apply_op(g, "const", [], [dtypes.float32])
      with g.control_dependencies(None):
        # deps are None again
        b_none2 = _apply_op(g, "const", [], [dtypes.float32])

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

    a_1 = _apply_op(g, "const", [], [dtypes.float32])
    a_2 = _apply_op(g, "const", [], [dtypes.float32])
    a_3 = _apply_op(g, "const", [], [dtypes.float32])
    a_4 = _apply_op(g, "const", [], [dtypes.float32])

    with g.control_dependencies([a_1]):
      b_1 = _apply_op(g, "mul", [a_3, a_4], [dtypes.float32])
      c_1 = _apply_op(g, "mul", [a_1, b_1], [dtypes.float32])
      d_1 = _apply_op(g, "mul", [b_1, c_1], [dtypes.float32])
      e_1 = _apply_op(g, "const", [], [dtypes.float32])
      with g.control_dependencies([a_2]):
        b_2 = _apply_op(g, "mul", [a_3, a_4], [dtypes.float32])
        c_2 = _apply_op(g, "mul", [a_1, b_1], [dtypes.float32])
        d_2 = _apply_op(g, "mul", [b_2, c_2], [dtypes.float32])
        e_2 = _apply_op(g, "mul", [e_1, e_1], [dtypes.float32])
        with g.control_dependencies([a_3]):
          b_3 = _apply_op(g, "mul", [a_3, a_4], [dtypes.float32])
          c_3 = _apply_op(g, "mul", [a_1, b_1], [dtypes.float32])
          d_3 = _apply_op(g, "mul", [b_3, c_3], [dtypes.float32])
          e_3 = _apply_op(g, "mul", [e_2, e_2], [dtypes.float32])
          with g.control_dependencies([a_4]):
            b_4 = _apply_op(g, "mul", [a_3, a_4], [dtypes.float32])
            c_4 = _apply_op(g, "mul", [a_1, b_1], [dtypes.float32])
            d_4 = _apply_op(g, "mul", [b_4, c_4], [dtypes.float32])
            e_4 = _apply_op(g, "mul", [e_3, e_3], [dtypes.float32])

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
    a = g.create_op("foo", [], [dtypes.float32, dtypes.float32])
    a_0, a_1 = a.outputs
    with g.control_dependencies([a_0]):
      b = _apply_op(g, "const", [], [dtypes.float32])
      with g.control_dependencies([a_1]):
        c = _apply_op(g, "const", [], [dtypes.float32])

    self.assertEqual(b.op.control_inputs, [a])
    self.assertEqual(c.op.control_inputs, [a])

  def testNoControlDependencyWithDataDependency(self):
    g = ops.Graph()
    a = _apply_op(g, "const", [], [dtypes.float32])
    with g.control_dependencies([a]):
      b = _apply_op(g, "identity", [a], [dtypes.float32])

    self.assertEqual(b.op.control_inputs, [])


class OpScopeTest(test_util.TensorFlowTestCase):

  def testNoScopeName(self):
    g0 = ops.Graph()
    values = [
        g0.create_op("a", [], [dtypes.float32]),
        g0.create_op("b", [], [dtypes.float32])]
    with self.assertRaises(ValueError):
      with ops.op_scope(values, None):
        pass
    with self.assertRaises(ValueError):
      with ops.op_scope(values, None, None):
        pass

  def testEmptyScopeName(self):
    g0 = ops.Graph()
    a = g0.create_op("a", [], [dtypes.float32])
    b = g0.create_op("b", [], [dtypes.float32])
    with ops.op_scope([a, b], "") as scope:
      self.assertEqual("", scope)
      self.assertEqual(g0, ops.get_default_graph())
    with ops.op_scope([a, b], "", "my_default_scope") as scope:
      self.assertEqual("", scope)
      self.assertEqual(g0, ops.get_default_graph())

  def testDefaultScopeName(self):
    g0 = ops.Graph()
    a = g0.create_op("a", [], [dtypes.float32])
    b = g0.create_op("b", [], [dtypes.float32])
    scope_name = "my_scope"
    default_scope_name = "my_default_scope"
    with ops.op_scope([a, b], scope_name, default_scope_name) as scope:
      self.assertEqual("%s/" % scope_name, scope)
      self.assertEqual(g0, ops.get_default_graph())
    with ops.op_scope([a, b], None, default_scope_name) as scope:
      self.assertEqual("%s/" % default_scope_name, scope)
      self.assertEqual(g0, ops.get_default_graph())

  def _testGraphElements(self, graph_elements):
    scope_name = "my_scope"
    with ops.op_scope(graph_elements, scope_name) as scope:
      self.assertEqual("%s/" % scope_name, scope)
      self.assertEqual(graph_elements[0].graph, ops.get_default_graph())
    g1 = ops.Graph()
    c = g1.create_op("c", [], [dtypes.float32])
    with self.assertRaises(ValueError):
      with ops.op_scope(graph_elements + [c], scope_name):
        pass

  def testTensor(self):
    g0 = ops.Graph()
    a = g0.create_op("a", [], [dtypes.float32])
    b = g0.create_op("b", [], [dtypes.float32])
    self._testGraphElements([a, b])

  def testSparseTensor(self):
    g0 = ops.Graph()
    a = g0.create_op("a", [], [dtypes.float32])
    b = g0.create_op("b", [], [dtypes.float32])
    sparse = ops.SparseTensor(
        _apply_op(g0, "const", [], [dtypes.int64]),
        _apply_op(g0, "const", [], [dtypes.float32]),
        _apply_op(g0, "const", [], [dtypes.int64]))
    self._testGraphElements([a, sparse, b])

  def testVariable(self):
    g0 = ops.Graph()
    with g0.as_default():
      variable = variables.Variable([1.0])
    a = g0.create_op("a", [], [dtypes.float32])
    b = g0.create_op("b", [], [dtypes.float32])
    self._testGraphElements([a, variable, b])


class GraphTest(test_util.TensorFlowTestCase):

  def setUp(self):
    ops.reset_default_graph()

  def _AssertDefault(self, expected):
    self.assertIs(expected, ops.get_default_graph())

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

  def testAsGraphElementConversions(self):
    class ConvertibleObj(object):

      def _as_graph_element(self):
        return "const:0"

    class NonConvertibleObj(object):

      pass

    g = ops.Graph()
    a = _apply_op(g, "const", [], [dtypes.float32])
    self.assertEqual(a, g.as_graph_element(ConvertibleObj()))
    with self.assertRaises(TypeError):
      g.as_graph_element(NonConvertibleObj())

ops.RegisterShape("KernelLabel")(common_shapes.scalar_shape)


class KernelLabelTest(test_util.TensorFlowTestCase):

  def testNoLabel(self):
    with self.test_session():
      self.assertAllEqual(b"My label is: default",
                          test_kernel_label_op.kernel_label().eval())

  def testLabelMap(self):
    with self.test_session() as sess:
      default_1 = test_kernel_label_op.kernel_label()
      # pylint: disable=protected-access
      with sess.graph._kernel_label_map({"KernelLabel": "overload_1"}):
        overload_1_1 = test_kernel_label_op.kernel_label()
        with sess.graph._kernel_label_map({"KernelLabel": "overload_2"}):
          overload_2 = test_kernel_label_op.kernel_label()
          with sess.graph._kernel_label_map({"KernelLabel": ""}):
            default_2 = test_kernel_label_op.kernel_label()
        overload_1_2 = test_kernel_label_op.kernel_label()
      # pylint: enable=protected-access
      default_3 = test_kernel_label_op.kernel_label()

      self.assertAllEqual(b"My label is: default", default_1.eval())
      self.assertAllEqual(b"My label is: default", default_2.eval())
      self.assertAllEqual(b"My label is: default", default_3.eval())
      self.assertAllEqual(b"My label is: overload_1", overload_1_1.eval())
      self.assertAllEqual(b"My label is: overload_1", overload_1_2.eval())
      self.assertAllEqual(b"My label is: overload_2", overload_2.eval())


if __name__ == "__main__":
  googletest.main()
