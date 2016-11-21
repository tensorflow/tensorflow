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

"""Tests for tensorflow.python.framework.importer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from google.protobuf import text_format

from tensorflow.core.framework import op_def_pb2
from tensorflow.python.framework import device
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import op_def_registry
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape


def _unknown_shape(op):
  return [tensor_shape.unknown_shape() for _ in op.outputs]


# NOTE(cwhipkey): Dummy shape registration for ops used in the tests, since they
# don't have C++ op registrations on which to attach C++ shape fns.
ops.RegisterShape("If")(_unknown_shape)
ops.RegisterShape("Iff")(_unknown_shape)
ops.RegisterShape("Ii")(_unknown_shape)
ops.RegisterShape("Iif")(_unknown_shape)
ops.RegisterShape("Iii")(_unknown_shape)
ops.RegisterShape("In")(_unknown_shape)
ops.RegisterShape("Iri")(_unknown_shape)
ops.RegisterShape("None")(_unknown_shape)
ops.RegisterShape("Of")(_unknown_shape)
ops.RegisterShape("Oi")(_unknown_shape)
ops.RegisterShape("Oif")(_unknown_shape)
ops.RegisterShape("Oii")(_unknown_shape)
ops.RegisterShape("OpWithDefaultAttr")(_unknown_shape)
ops.RegisterShape("OpWithFutureDefaultAttr")(_unknown_shape)
ops.RegisterShape("Or")(_unknown_shape)
ops.RegisterShape("Otl")(_unknown_shape)
ops.RegisterShape("Unary")(_unknown_shape)


_op_list = op_def_pb2.OpList()
text_format.Merge("""
  op {
    name: 'None'
  }
  op {
    name: 'Oi'
    output_arg { name: 'a' type: DT_INT32 }
  }
  op {
    name: 'Or'
    output_arg { name: 'a' type: DT_INT32 is_ref: true }
  }
  op {
    name: 'Of'
    output_arg { name: 'a' type: DT_FLOAT }
  }
  op {
    name: 'Ii'
    input_arg { name: 'a' type: DT_INT32 }
  }
  op {
    name: 'If'
    input_arg { name: 'a' type: DT_FLOAT }
  }
  op {
    name: 'Oii'
    output_arg { name: 'a' type: DT_INT32 }
    output_arg { name: 'b' type: DT_INT32 }
  }
  op {
    name: 'Oif'
    output_arg { name: 'a' type: DT_INT32 }
    output_arg { name: 'b' type: DT_FLOAT }
  }
  op {
    name: 'Iii'
    input_arg { name: 'a' type: DT_INT32 }
    input_arg { name: 'b' type: DT_INT32 }
  }
  op {
    name: 'Iff'
    input_arg { name: 'a' type: DT_FLOAT }
    input_arg { name: 'b' type: DT_FLOAT }
  }
  op {
    name: 'Iif'
    input_arg { name: 'a' type: DT_INT32 }
    input_arg { name: 'b' type: DT_FLOAT }
  }
  op {
    name: 'Iri'
    input_arg { name: 'a' type: DT_INT32 is_ref: true }
    input_arg { name: 'b' type: DT_INT32 }
  }
  op {
    name: 'In'
    input_arg { name: 'a' number_attr: 'N' type_attr: 'T' }
    attr { name: 'N' type: 'int' minimum: 1 }
    attr { name: 'T' type: 'type' }
  }
  op {
    name: 'Otl'
    output_arg { name: 'a' type_list_attr: 't' }
    attr { name: 'T' type: 'list(type)' minimum: 1 }
  }
  op {
    name: 'Unary'
    input_arg { name: 'a' type_attr: 'T' }
    output_arg { name: 'b' type_attr: 'T' }
    attr { name: 'T' type: 'type' }
  }
  op {
    name: 'OpWithDefaultAttr'
    output_arg { name: 'a' type: DT_INT32 }
    attr { name: 'default_float' type: 'float' default_value { f: 123.0 } }
  }
  op {
    name: 'OpWithFutureDefaultAttr'
  }
""", _op_list)
op_def_registry.register_op_list(_op_list)
# NOTE(mrry): Dummy shape registrations for ops used in the tests.
for op_def in _op_list.op:
  ops.RegisterShape(op_def.name)(None)


class ImportGraphDefTest(tf.test.TestCase):

  def _MakeGraphDef(self, text, producer=tf.GRAPH_DEF_VERSION,
                    min_consumer=tf.GRAPH_DEF_VERSION_MIN_CONSUMER):
    text = "versions: { producer: %d min_consumer: %d };\n%s" % (
        producer, min_consumer, text)
    ret = tf.GraphDef()
    text_format.Merge(text, ret)
    return ret

  def testBasic(self):
    with tf.Graph().as_default():
      a, b, c, d = tf.import_graph_def(
          self._MakeGraphDef("""
          node { name: 'A' op: 'Oif' }
          node { name: 'B' op: 'Otl'
                 attr { key: 't'
                        value { list { type: DT_INT32 type: DT_FLOAT } } } }
          node { name: 'C' op: 'In'
                 attr { key: 'N' value { i: 2 } }
                 attr { key: 'T' value { type: DT_INT32 } }
                 input: 'A:0' input: 'B:0' }
          node { name: 'D' op: 'In'
                 attr { key: 'N' value { i: 2 } }
                 attr { key: 'T' value { type: DT_FLOAT } }
                 input: 'A:1' input: 'B:1' }
          """),
          return_elements=["A", "B", "C", "D"],
          name="import")

      # Assert that the import process creates distinct tensors.
      self.assertNotEqual(a.outputs[0].name, a.outputs[1].name)
      self.assertNotEqual(b.outputs[0].name, b.outputs[1].name)
      self.assertNotEqual(a.outputs[0].name, b.outputs[0].name)
      self.assertNotEqual(a.outputs[0].name, b.outputs[1].name)
      self.assertNotEqual(a.outputs[1].name, b.outputs[0].name)
      self.assertNotEqual(a.outputs[1].name, b.outputs[1].name)

      # Assert that the ops are connected according to the GraphDef topology.
      self.assertEqual(c.inputs[0], a.outputs[0])
      self.assertEqual(c.inputs[1], b.outputs[0])
      self.assertEqual(d.inputs[0], a.outputs[1])
      self.assertEqual(d.inputs[1], b.outputs[1])

      # Check the types of the returned ops and tensors.
      self.assertEqual(a.type, "Oif")
      self.assertEqual(b.type, "Otl")
      self.assertEqual(c.type, "In")
      self.assertEqual(d.type, "In")
      self.assertEqual(a.outputs[0].dtype, tf.int32)
      self.assertEqual(a.outputs[1].dtype, tf.float32)
      self.assertEqual(b.outputs[0].dtype, tf.int32)
      self.assertEqual(b.outputs[1].dtype, tf.float32)

      # Check the names of the returned ops.
      self.assertEqual(a.name, "import/A")
      self.assertEqual(b.name, "import/B")
      self.assertEqual(c.name, "import/C")
      self.assertEqual(d.name, "import/D")

      # Check that the op_def is still available.
      self.assertNotEqual(None, a.op_def)

  def testInputMap(self):
    with tf.Graph().as_default():
      feed_a_0 = tf.constant(0, dtype=tf.int32)
      feed_b_1 = tf.constant(1, dtype=tf.int32)

      a, b, c, d = tf.import_graph_def(
          self._MakeGraphDef("""
          node { name: 'A' op: 'Oii' }
          node { name: 'B' op: 'Oii' }
          node { name: 'C' op: 'In'
                 attr { key: 'N' value { i: 2 } }
                 attr { key: 'T' value { type: DT_INT32 } }
                 input: 'A:0' input: 'B:0' }
          node { name: 'D' op: 'In'
                 attr { key: 'N' value { i: 2 } }
                 attr { key: 'T' value { type: DT_INT32 } }
                 input: 'A:1' input: 'B:1' }
          """),
          input_map={"A:0": feed_a_0, "B:1": feed_b_1},
          return_elements=["A", "B", "C", "D"])

      self.assertEqual(c.inputs[0], feed_a_0)
      self.assertEqual(c.inputs[1], b.outputs[0])
      self.assertEqual(d.inputs[0], a.outputs[1])
      self.assertEqual(d.inputs[1], feed_b_1)

  def testInputMapBytes(self):
    with tf.Graph().as_default():
      feed_a_0 = tf.constant(0, dtype=tf.int32)
      feed_b_1 = tf.constant(1, dtype=tf.int32)

      a, b, c, d = tf.import_graph_def(
          self._MakeGraphDef("""
          node { name: 'A' op: 'Oii' }
          node { name: 'B' op: 'Oii' }
          node { name: 'C' op: 'In'
                 attr { key: 'N' value { i: 2 } }
                 attr { key: 'T' value { type: DT_INT32 } }
                 input: 'A:0' input: 'B:0' }
          node { name: 'D' op: 'In'
                 attr { key: 'N' value { i: 2 } }
                 attr { key: 'T' value { type: DT_INT32 } }
                 input: 'A:1' input: 'B:1' }
          """),
          input_map={b"A:0": feed_a_0, b"B:1": feed_b_1},
          return_elements=[b"A", b"B", b"C", b"D"])

      self.assertEqual(c.inputs[0], feed_a_0)
      self.assertEqual(c.inputs[1], b.outputs[0])
      self.assertEqual(d.inputs[0], a.outputs[1])
      self.assertEqual(d.inputs[1], feed_b_1)

  def testInputMapUnicode(self):
    with tf.Graph().as_default():
      feed_a_0 = tf.constant(0, dtype=tf.int32)
      feed_b_1 = tf.constant(1, dtype=tf.int32)

      a, b, c, d = tf.import_graph_def(
          self._MakeGraphDef("""
          node { name: 'A' op: 'Oii' }
          node { name: 'B' op: 'Oii' }
          node { name: 'C' op: 'In'
                 attr { key: 'N' value { i: 2 } }
                 attr { key: 'T' value { type: DT_INT32 } }
                 input: 'A:0' input: 'B:0' }
          node { name: 'D' op: 'In'
                 attr { key: 'N' value { i: 2 } }
                 attr { key: 'T' value { type: DT_INT32 } }
                 input: 'A:1' input: 'B:1' }
          """),
          input_map={u"A:0": feed_a_0, u"B:1": feed_b_1},
          return_elements=[u"A", u"B", u"C", u"D"])

      self.assertEqual(c.inputs[0], feed_a_0)
      self.assertEqual(c.inputs[1], b.outputs[0])
      self.assertEqual(d.inputs[0], a.outputs[1])
      self.assertEqual(d.inputs[1], feed_b_1)

  def testImplicitZerothOutput(self):
    with tf.Graph().as_default():
      a, b = tf.import_graph_def(
          self._MakeGraphDef("""
          node { name: 'A' op: 'Oii' }
          node { name: 'B' op: 'Ii' input: 'A' }
          """),
          return_elements=["A", "B"])

      self.assertEqual(b.inputs[0], a.outputs[0])

  def testInputMapImplicitZerothOutput(self):
    with tf.Graph().as_default():
      feed_a_0 = tf.constant(0, dtype=tf.int32)
      b, = tf.import_graph_def(
          self._MakeGraphDef("""
          node { name: 'A' op: 'Oii' }
          node { name: 'B' op: 'Ii' input: 'A:0' }
          """),
          input_map={"A": feed_a_0},
          return_elements=["B"])

      self.assertEqual(b.inputs[0], feed_a_0)

  def testWithControlDependency(self):
    with tf.Graph().as_default():
      a, b = tf.import_graph_def(
          self._MakeGraphDef("""
          node { name: 'A' op: 'None' }
          node { name: 'B' op: 'None' input: '^A' }
          """),
          return_elements=["A", "B"])

      self.assertEqual(b.control_inputs, [a])

  def testWithRefs(self):
    with tf.Graph().as_default():
      a, b, c, d = tf.import_graph_def(
          self._MakeGraphDef("""
          node { name: 'A' op: 'Or' }
          node { name: 'B' op: 'Oi' }
          node { name: 'C' op: 'Iii' input: 'A:0' input: 'B:0' }
          node { name: 'D' op: 'Iri' input: 'A:0' input: 'B:0' }
          """),
          return_elements=["A", "B", "C", "D"])

      self.assertEqual(c.inputs[0], a.outputs[0])
      self.assertEqual(c.inputs[1], b.outputs[0])
      self.assertEqual(d.inputs[0], a.outputs[0])
      self.assertEqual(d.inputs[1], b.outputs[0])

      self.assertEqual(a.outputs[0].dtype, dtypes.int32_ref)
      self.assertEqual(c._input_dtypes, [tf.int32, tf.int32])
      self.assertEqual(c.outputs, [])
      self.assertEqual(d._input_dtypes,
                       [dtypes.int32_ref, tf.int32])
      self.assertEqual(d.outputs, [])

  def testCyclic(self):
    with tf.Graph().as_default():
      a, b = tf.import_graph_def(
          self._MakeGraphDef("""
          node { name: 'A' op: 'Unary'
                 attr { key: 'T' value { type: DT_INT32 } } input: 'B:0' }
          node { name: 'B' op: 'Unary'
                 attr { key: 'T' value { type: DT_INT32 } } input: 'A:0' }
          """),
          return_elements=["A", "B"])

      self.assertEqual(a.inputs[0], b.outputs[0])
      self.assertEqual(b.inputs[0], a.outputs[0])

  def testTypeMismatchInGraphDef(self):
    with tf.Graph().as_default():
      with self.assertRaises(ValueError) as e:
        tf.import_graph_def(
            self._MakeGraphDef("""
            node { name: 'A' op: 'Oi' }
            node { name: 'B' op: 'If' input: 'A:0' }
            """))
      self.assertTrue(
          "Cannot convert a tensor of type int32 to an input of type float" in
          str(e.exception))

  def testShapeWhitelist(self):
    # Barrier's shape is an output vector of 2, but the
    # graph says it's a scalar.  This is currently whitelisted.
    with tf.Graph().as_default():
      _ = tf.import_graph_def(
          self._MakeGraphDef("""
          node { name: 'A' op: 'Barrier'
                 attr { key: '_output_shapes'
                        value { list { shape { } } } } }
          """),
          return_elements=["A"],
          name="import")

  def testShapeWhitelistViolation(self):
    # L2 loss produces a scalar shape, but the graph
    # has the wrong shape, so raise an error.
    with tf.Graph().as_default():
      with self.assertRaises(ValueError) as e:
        _ = tf.import_graph_def(
            self._MakeGraphDef("""
              node { name: 'A' op: 'Of' }
              node { name: 'B' op: 'L2Loss'
                     input: 'A:0'
                     attr { key: 'T' value { type: DT_FLOAT } }
                     attr { key: '_output_shapes'
                            value { list { shape { dim { size: 43 } } } } } }
            """),
            return_elements=["B"],
            name="import")
        self.assertTrue(
            "Shapes () and (43,) are not compatible" in str(e.exception))

  def testInvalidSignatureTooManyInputsInGraphDef(self):
    with tf.Graph().as_default():
      with self.assertRaises(ValueError) as e:
        tf.import_graph_def(
            self._MakeGraphDef("""
            node { name: 'A' op: 'Oi' }
            node { name: 'B' op: 'None' input: 'A:0' }
            """))
      self.assertTrue("More inputs specified ('A:0') than the op expects" in
                      str(e.exception))

  def testInvalidSignatureNotEnoughInputsInGraphDef(self):
    with tf.Graph().as_default():
      with self.assertRaises(ValueError) as e:
        tf.import_graph_def(
            self._MakeGraphDef("""
            node { name: 'A' op: 'Oi' }
            node { name: 'B' op: 'Iif' input: 'A:0' }
            """))
      self.assertTrue("Input types mismatch (expected 'int32, float32' but "
                      "got 'int32')" in str(e.exception))

  def testMissingInputOpInGraphDef(self):
    with tf.Graph().as_default():
      with self.assertRaises(ValueError) as e:
        tf.import_graph_def(
            self._MakeGraphDef("""
            node { name: 'B' op: 'If' input: 'A:0' }
            """))
      self.assertTrue("Input tensor 'A:0' not found" in str(e.exception))

  def testMissingInputOpInGraphDefButAppearsInInputMap(self):
    with tf.Graph().as_default():
      feed_a_0 = tf.constant(5.0)
      b, = tf.import_graph_def(
          self._MakeGraphDef("""
          node { name: 'B' op: 'If' input: 'A:0' }
          """),
          input_map={"A:0": feed_a_0},
          return_elements=["B"])
      self.assertEqual(b.inputs[0], feed_a_0)

  def testMissingInputTensorInGraphDef(self):
    with tf.Graph().as_default():
      with self.assertRaises(ValueError) as e:
        tf.import_graph_def(
            self._MakeGraphDef("""
            node { name: 'A' op: 'Of' }
            node { name: 'B' op: 'If' input: 'A:1' }
            """))
      self.assertTrue("Input tensor 'A:1' not found" in str(e.exception))

  def testMissingControlInputInGraphDef(self):
    with tf.Graph().as_default():
      with self.assertRaises(ValueError) as e:
        tf.import_graph_def(
            self._MakeGraphDef("""
            node { name: 'B' op: 'None' input: '^A' }
            """))
      self.assertTrue("Control input '^A' not found" in str(e.exception))

  def testInvalidTensorNameOutputIndexInGraphDef(self):
    with tf.Graph().as_default():
      with self.assertRaises(ValueError) as e:
        tf.import_graph_def(
            self._MakeGraphDef("""
            node { name: 'B' op: 'None' input: 'A:B' }
            """))
      self.assertEqual("Cannot convert 'A:B' to a tensor name.",
                       str(e.exception))

  def testInvalidTensorNameInGraphDef(self):
    with tf.Graph().as_default():
      with self.assertRaises(ValueError) as e:
        tf.import_graph_def(
            self._MakeGraphDef("""
            node { name: 'B' op: 'None' input: 'A:B:0' }
            """))
      self.assertEqual("Cannot convert 'A:B:0' to a tensor name.",
                       str(e.exception))

  def testMissingReturnOperation(self):
    with tf.Graph().as_default():
      with self.assertRaises(ValueError) as e:
        tf.import_graph_def(
            self._MakeGraphDef("""
            node { name: 'A' op: 'None' }
            """),
            return_elements=["B"])
      self.assertTrue("return_element 'B' not found in graph_def." in
                      str(e.exception))

  def testMissingReturnTensor(self):
    with tf.Graph().as_default():
      with self.assertRaises(ValueError) as e:
        tf.import_graph_def(
            self._MakeGraphDef("""
            node { name: 'A' op: 'Oi' }
            """),
            return_elements=["A:1"])
      self.assertTrue("return_element 'A:1' not found in graph_def." in
                      str(e.exception))

      with self.assertRaises(ValueError) as e:
        tf.import_graph_def(
            self._MakeGraphDef("""
            node { name: 'A' op: 'Oi' }
            """),
            return_elements=["B:0"])
      self.assertTrue("return_element 'B:0' not found in graph_def." in
                      str(e.exception))

      with self.assertRaises(ValueError) as e:
        tf.import_graph_def(
            self._MakeGraphDef("""
            node { name: 'A' op: 'Oi' }
            """),
            return_elements=["A:B:0"])
      self.assertTrue("return_element 'A:B:0' not found in graph_def." in
                      str(e.exception))

  def testMissingInputMap(self):
    with tf.Graph().as_default():
      with self.assertRaises(ValueError) as e:
        tf.import_graph_def(
            self._MakeGraphDef("""
            node { name: 'A' op: 'None' }
            """),
            input_map={"B:0": tf.constant(5.0)})
      self.assertTrue("not found in graph_def: [B:0]" in str(e.exception))

  def testInputMapTypeMismatch(self):
    with tf.Graph().as_default():
      with self.assertRaises(ValueError) as e:
        tf.import_graph_def(
            self._MakeGraphDef("""
            node { name: 'A' op: 'Oi' }
            node { name: 'B' op: 'Ii' input: 'A:0' }
            """),
            input_map={"A:0": tf.constant(5.0)})
      self.assertTrue(
          "Cannot convert a tensor of type float32 to an input of type int32."
          in str(e.exception))

  def testNoReturns(self):
    with tf.Graph().as_default() as g:
      ret = tf.import_graph_def(
          self._MakeGraphDef("""
          node { name: 'A' op: 'None' }
          """))
      self.assertEqual(ret, None)

      a = g.get_operation_by_name("import/A")
      self.assertEqual(a.type, "None")

  def testOverrideNamePrefix(self):
    with tf.Graph().as_default():
      a, = tf.import_graph_def(
          self._MakeGraphDef("""
          node { name: 'A' op: 'None' }
          """),
          return_elements=["A"], name="imported_graph")
      self.assertEqual(a.name, "imported_graph/A")

  def testNamePrefixColocationAttrs(self):
    original_graph_def = self._MakeGraphDef("""
          node { name: 'A' op: 'None' }
          node { name: 'B' op: 'None'  attr {
            key: '_class'
            value { list { s: 'loc:@A' } }
          } }""")

    with tf.Graph().as_default():
      b, = tf.import_graph_def(original_graph_def,
                               return_elements=["B"], name="imported_graph")
      self.assertProtoEqualsVersion("""
          node { name: 'imported_graph/A' op: 'None' }
          node { name: 'imported_graph/B' op: 'None'  attr {
            key: '_class'
            value { list { s: 'loc:@imported_graph/A' } }
          } }""", b.graph.as_graph_def())

  def testNamePrefixColocationAttrsMultipleImport(self):
    original_graph_def = self._MakeGraphDef("""
          node { name: 'A' op: 'None' }
          node { name: 'B' op: 'None'  attr {
            key: '_class'
            value { list { s: 'loc:@A' } }
          } }""")

    with tf.Graph().as_default():
      b, = tf.import_graph_def(original_graph_def,
                               return_elements=["B"], name="")
      _, = tf.import_graph_def(original_graph_def,
                               return_elements=["B"], name="")
      self.assertProtoEqualsVersion("""
          node { name: 'A' op: 'None' }
          node { name: 'B' op: 'None'  attr {
            key: '_class'
            value { list { s: 'loc:@A' } }
          } }
          node { name: 'A_1' op: 'None' }
          node { name: 'B_1' op: 'None'  attr {
            key: '_class'
            value { list { s: 'loc:@A_1' } }
          } }""", b.graph.as_graph_def())

  def testNamePrefixColocationAttrsNotFound(self):
    original_graph_def = self._MakeGraphDef("""
          node { name: 'B' op: 'None'  attr {
            key: '_class'
            value { list { s: 'loc:@A' } }
          } }""")
    with tf.Graph().as_default():
      with self.assertRaisesRegexp(ValueError, "does not exist during import"):
        tf.import_graph_def(original_graph_def,
                            return_elements=["B"], name="imported_graph")

  def testEmptyGraph(self):
    with tf.Graph().as_default() as g:
      init_version = g.version
      tf.import_graph_def(self._MakeGraphDef(""))
      self.assertEqual(init_version, g.version)

  def testInvalidInputForGraphDef(self):
    with tf.Graph().as_default():
      with self.assertRaises(TypeError) as e:
        tf.import_graph_def("")
      self.assertEqual(
          "graph_def must be a GraphDef proto.", str(e.exception))

  def testInvalidInputForInputMap(self):
    with tf.Graph().as_default():
      with self.assertRaises(TypeError) as e:
        tf.import_graph_def(self._MakeGraphDef(""),
                            input_map=[tf.constant(5.0)])
      self.assertEqual("input_map must be a dictionary mapping strings to "
                       "Tensor objects.", str(e.exception))
      with self.assertRaises(ValueError) as e:
        tf.import_graph_def(self._MakeGraphDef(""),
                            input_map={"a:0": tf.constant(5.0)},
                            name="")
      self.assertEqual("tf.import_graph_def() requires a non-empty `name` "
                       "if `input_map` is used.", str(e.exception))

  def testInvalidInputForReturnOperations(self):
    with tf.Graph().as_default():
      with self.assertRaises(TypeError) as e:
        tf.import_graph_def(self._MakeGraphDef(""), return_elements=[7])
      self.assertEqual(
          "return_elements must be a list of strings.", str(e.exception))

  def testWithExtensionAndAttr(self):
    with tf.Graph().as_default() as g:
      c = tf.constant(5.0, dtype=tf.float32, name="c")
      tf.stack([c, c], name="pack")
    gdef = g.as_graph_def()

    with self.test_session():
      pack, = tf.import_graph_def(gdef, return_elements=["pack"])
      self.assertAllEqual(pack.outputs[0].eval(), [5.0, 5.0])

  def testWithDevice(self):
    with tf.Graph().as_default() as g:
      # No device.
      a = tf.constant(3.0, name="a")

      with tf.device("/cpu:0"):
        b = tf.constant(4.0, name="b")
      with tf.device("/job:worker"):
        c = tf.constant(5.0, name="c")

    gdef = g.as_graph_def()

    with tf.Graph().as_default():
      a2, b2, c2 = tf.import_graph_def(
          gdef, return_elements=["a", "b", "c"])
      self.assertEqual(a.device, a2.device)
      self.assertEqual(b.device, b2.device)
      self.assertEqual(c.device, c2.device)

    with tf.Graph().as_default():
      with tf.device(device.merge_device("/task:0")):
        a3, b3, c3 = tf.import_graph_def(
            gdef, return_elements=["a", "b", "c"])
        self.assertEqual("/task:0", a3.device)
        self.assertEqual("/task:0/device:CPU:0", b3.device)  # canonicalized.
        self.assertEqual(c.device + "/task:0", c3.device)

    with tf.Graph().as_default():
      with tf.device(device.merge_device("/job:ps")):
        a4, b4, c4 = tf.import_graph_def(
            gdef, return_elements=["a", "b", "c"])
        self.assertEqual("/job:ps", a4.device)
        self.assertEqual("/job:ps/device:CPU:0", b4.device)  # canonicalized.
        self.assertEqual(c.device, c4.device)  # worker overrides ps.

    with tf.Graph().as_default():
      with tf.device(device.merge_device("/gpu:0")):
        a5, b5, c5 = tf.import_graph_def(
            gdef, return_elements=["a", "b", "c"])
        self.assertEqual("/device:GPU:0", a5.device)
        self.assertEqual("/device:CPU:0", b5.device)  # cpu overrides gpu.
        self.assertEqual(c.device + "/device:GPU:0", c5.device)

  def testWithDeviceFunctionDependingOnInputs(self):
    with tf.Graph().as_default() as g:
      with tf.device("/job:ps"):
        v = tf.Variable(1.0)
      unused_assign_op = v.assign(2.0)
      unused_assign_2_op = v.assign(3.0)
      unused_add_t = v + v
    gdef = g.as_graph_def()

    # We'll use the following device function to observe ops with two inputs.
    ops_with_two_inputs = []
    def input_counter(op):
      if any(in_t.dtype._is_ref_dtype for in_t in op.inputs):  # pylint: disable=protected-access
        ops_with_two_inputs.append(op)
      return ""

    with tf.Graph().as_default() as g:
      with tf.device(input_counter):
        tf.import_graph_def(gdef)

    # We expect to see the initializer, two assign operations, and the add op.
    self.assertEqual(4, len(ops_with_two_inputs))

  def testGradient(self):
    with tf.Graph().as_default() as g:
      inputs = tf.placeholder(tf.float32, shape=[None, 100], name="input")
      weights = tf.placeholder(tf.float32, shape=[100, 10], name="weights")
      biases = tf.placeholder(tf.float32, shape=[10], name="biases")
      activations = tf.nn.relu(tf.matmul(inputs, weights) + biases,
                               name="activations")
      loss = tf.reduce_mean(activations, name="loss")
    gdef = g.as_graph_def()

    with tf.Graph().as_default() as g:
      input_placeholder = tf.placeholder(tf.float32, shape=[32, 100])
      weights_var = tf.Variable(tf.truncated_normal([100, 10]), name="weights")
      biases_var = tf.Variable(tf.zeros([10]), name="biases")
      activations, loss = tf.import_graph_def(
          gdef,
          input_map={"input:0": input_placeholder,
                     "weights:0": weights_var,
                     "biases:0": biases_var},
          return_elements=["activations:0", "loss:0"])
      self.assertEqual([32, 10], activations.get_shape())
      self.assertEqual([], loss.get_shape())
      weights_grad, biases_grad = tf.gradients(loss, [weights_var, biases_var])
      self.assertEqual([100, 10], weights_grad.get_shape())
      self.assertEqual([10], biases_grad.get_shape())

  def testLargeGraph(self):
    with self.test_session():
      # The default message byte limit is 64M. Ours is 2G with a warning at 512.
      # Adding a 130M entries float32 tensor should exceed the warning, but not
      # the hard limit.
      input_shape = [130, 1000, 1000]
      tensor_input = np.ones(input_shape, dtype=np.float32)
      t = tf.constant(tensor_input, shape=input_shape)
      g = tf.identity(t)
      g.eval()

  def testVersion(self):
    v0 = tf.GRAPH_DEF_VERSION_MIN_CONSUMER
    v2 = tf.GRAPH_DEF_VERSION
    v1 = (v0 + v2) // 2
    for producer in v0, v1, v2:
      for min_consumer in v0, v1, v2:
        with tf.Graph().as_default():
          a, = tf.import_graph_def(
              self._MakeGraphDef("node { name: 'A' op: 'Oii' }",
                                 producer=producer, min_consumer=min_consumer),
              return_elements=["A"])
          self.assertEqual(a.graph.graph_def_versions.producer, producer)
          self.assertEqual(a.graph.graph_def_versions.min_consumer,
                           min_consumer)

  def testVersionLow(self):
    with tf.Graph().as_default() as g:
      pat = (r"GraphDef producer version -1 below min producer %d supported "
             r"by TensorFlow \S+\.  Please regenerate your graph.$" %
             tf.GRAPH_DEF_VERSION_MIN_PRODUCER)
      tf.import_graph_def(self._MakeGraphDef("", producer=-1))
      x = tf.constant(7)  # Need at least one op to get a C++ graph generated
      with self.test_session(graph=g) as sess:
        with self.assertRaisesRegexp(Exception, pat):
          sess.run(x)

  def testVersionHigh(self):
    with tf.Graph().as_default() as g:
      pat = (r"GraphDef min consumer version %d above current version %d "
             r"for TensorFlow \S+\.  Please upgrade TensorFlow\.$" %
             (1 << 30, tf.GRAPH_DEF_VERSION))
      tf.import_graph_def(self._MakeGraphDef("", min_consumer=1 << 30))
      x = tf.constant(7)  # Need at least one op to get a C++ graph generated
      with self.test_session(graph=g) as sess:
        with self.assertRaisesRegexp(Exception, pat):
          sess.run(x)

  def testDefaultAttrsAdded(self):
    with tf.Graph().as_default():
      a = tf.import_graph_def(
          self._MakeGraphDef("""
          node { name: 'A' op: 'OpWithDefaultAttr' }
          """),
          return_elements=["A"])
      self.assertEqual(123.0, a[0].get_attr("default_float"))

  def testDefaultAttrsRemoved(self):
    producer_op_list = op_def_pb2.OpList()
    text_format.Merge("""
      op {
        name: 'OpWithFutureDefaultAttr'
        attr { name: 'default_int' type: 'int' default_value { i: 456 } }
      }
    """, producer_op_list)
    # Attr only in producer_op_list with default value gets removed.
    with tf.Graph().as_default():
      a = tf.import_graph_def(
          self._MakeGraphDef("""
          node { name: 'A' op: 'OpWithFutureDefaultAttr'
                 attr { key: 'default_int' value { i: 456 } } }
          """),
          return_elements=["A"], producer_op_list=producer_op_list)
      with self.assertRaisesRegexp(ValueError, "No attr named 'default_int'"):
        a[0].get_attr("default_int")

    # Attr only in producer_op_list with non-default value is preserved.
    with tf.Graph().as_default():
      a = tf.import_graph_def(
          self._MakeGraphDef("""
          node { name: 'A' op: 'OpWithFutureDefaultAttr'
                 attr { key: 'default_int' value { i: 987 } } }
          """),
          return_elements=["A"], producer_op_list=producer_op_list)
      self.assertEqual(987, a[0].get_attr("default_int"))

if __name__ == "__main__":
  tf.test.main()
