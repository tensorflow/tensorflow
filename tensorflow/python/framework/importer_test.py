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

from google.protobuf import text_format

from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import op_def_pb2
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import device
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import function
from tensorflow.python.framework import importer
from tensorflow.python.framework import op_def_registry
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_ops  # pylint: disable=unused-import
from tensorflow.python.framework import versions
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables
import tensorflow.python.ops.nn_grad  # pylint: disable=unused-import
from tensorflow.python.platform import test


def _UnknownShape(op):
  return [tensor_shape.unknown_shape() for _ in op.outputs]


# NOTE(cwhipkey): Dummy shape registration for ops used in the tests, since they
# don't have C++ op registrations on which to attach C++ shape fns.
ops.RegisterShape("If")(_UnknownShape)
ops.RegisterShape("Iff")(_UnknownShape)
ops.RegisterShape("Ii")(_UnknownShape)
ops.RegisterShape("Iif")(_UnknownShape)
ops.RegisterShape("Iii")(_UnknownShape)
ops.RegisterShape("In")(_UnknownShape)
ops.RegisterShape("Iri")(_UnknownShape)
ops.RegisterShape("None")(_UnknownShape)
ops.RegisterShape("Of")(_UnknownShape)
ops.RegisterShape("Oi")(_UnknownShape)
ops.RegisterShape("Oif")(_UnknownShape)
ops.RegisterShape("Oii")(_UnknownShape)
ops.RegisterShape("OpWithDefaultAttr")(_UnknownShape)
ops.RegisterShape("OpWithFutureDefaultAttr")(_UnknownShape)
ops.RegisterShape("Or")(_UnknownShape)
ops.RegisterShape("Otl")(_UnknownShape)
ops.RegisterShape("Unary")(_UnknownShape)

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


class ImportGraphDefTest(test.TestCase):

  def _MakeGraphDef(self,
                    text,
                    producer=versions.GRAPH_DEF_VERSION,
                    min_consumer=versions.GRAPH_DEF_VERSION_MIN_CONSUMER):
    text = "versions: { producer: %d min_consumer: %d };\n%s" % (producer,
                                                                 min_consumer,
                                                                 text)
    ret = graph_pb2.GraphDef()
    text_format.Merge(text, ret)
    return ret

  def testBasic(self):
    with ops.Graph().as_default():
      a, b, c, d = importer.import_graph_def(
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
      self.assertEqual(a.outputs[0].dtype, dtypes.int32)
      self.assertEqual(a.outputs[1].dtype, dtypes.float32)
      self.assertEqual(b.outputs[0].dtype, dtypes.int32)
      self.assertEqual(b.outputs[1].dtype, dtypes.float32)

      # Check the names of the returned ops.
      self.assertEqual(a.name, "import/A")
      self.assertEqual(b.name, "import/B")
      self.assertEqual(c.name, "import/C")
      self.assertEqual(d.name, "import/D")

      # Check that the op_def is still available.
      self.assertNotEqual(None, a.op_def)

  def testInputMap(self):
    with ops.Graph().as_default():
      feed_a_0 = constant_op.constant(0, dtype=dtypes.int32)
      feed_b_1 = constant_op.constant(1, dtype=dtypes.int32)

      a, b, c, d = importer.import_graph_def(
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
          input_map={"A:0": feed_a_0,
                     "B:1": feed_b_1},
          return_elements=["A", "B", "C", "D"])

      self.assertEqual(c.inputs[0], feed_a_0)
      self.assertEqual(c.inputs[1], b.outputs[0])
      self.assertEqual(d.inputs[0], a.outputs[1])
      self.assertEqual(d.inputs[1], feed_b_1)

  def testInputMapBytes(self):
    with ops.Graph().as_default():
      feed_a_0 = constant_op.constant(0, dtype=dtypes.int32)
      feed_b_1 = constant_op.constant(1, dtype=dtypes.int32)

      a, b, c, d = importer.import_graph_def(
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
          input_map={b"A:0": feed_a_0,
                     b"B:1": feed_b_1},
          return_elements=[b"A", b"B", b"C", b"D"])

      self.assertEqual(c.inputs[0], feed_a_0)
      self.assertEqual(c.inputs[1], b.outputs[0])
      self.assertEqual(d.inputs[0], a.outputs[1])
      self.assertEqual(d.inputs[1], feed_b_1)

  def testInputMapUnicode(self):
    with ops.Graph().as_default():
      feed_a_0 = constant_op.constant(0, dtype=dtypes.int32)
      feed_b_1 = constant_op.constant(1, dtype=dtypes.int32)

      a, b, c, d = importer.import_graph_def(
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
          input_map={u"A:0": feed_a_0,
                     u"B:1": feed_b_1},
          return_elements=[u"A", u"B", u"C", u"D"])

      self.assertEqual(c.inputs[0], feed_a_0)
      self.assertEqual(c.inputs[1], b.outputs[0])
      self.assertEqual(d.inputs[0], a.outputs[1])
      self.assertEqual(d.inputs[1], feed_b_1)

  def testImplicitZerothOutput(self):
    with ops.Graph().as_default():
      a, b = importer.import_graph_def(
          self._MakeGraphDef("""
          node { name: 'A' op: 'Oii' }
          node { name: 'B' op: 'Ii' input: 'A' }
          """),
          return_elements=["A", "B"])

      self.assertEqual(b.inputs[0], a.outputs[0])

  def testInputMapImplicitZerothOutput(self):
    with ops.Graph().as_default():
      feed_a_0 = constant_op.constant(0, dtype=dtypes.int32)
      b, = importer.import_graph_def(
          self._MakeGraphDef("""
          node { name: 'A' op: 'Oii' }
          node { name: 'B' op: 'Ii' input: 'A:0' }
          """),
          input_map={"A": feed_a_0},
          return_elements=["B"])

      self.assertEqual(b.inputs[0], feed_a_0)

  def testWithControlDependency(self):
    with ops.Graph().as_default():
      a, b = importer.import_graph_def(
          self._MakeGraphDef("""
          node { name: 'A' op: 'None' }
          node { name: 'B' op: 'None' input: '^A' }
          """),
          return_elements=["A", "B"])

      self.assertEqual(b.control_inputs, [a])

  def testWithRefs(self):
    with ops.Graph().as_default():
      a, b, c, d = importer.import_graph_def(
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
      self.assertEqual(c._input_dtypes, [dtypes.int32, dtypes.int32])
      self.assertEqual(c.outputs, [])
      self.assertEqual(d._input_dtypes, [dtypes.int32_ref, dtypes.int32])
      self.assertEqual(d.outputs, [])

  def testCyclic(self):
    with ops.Graph().as_default():
      a, b = importer.import_graph_def(
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
    with ops.Graph().as_default():
      with self.assertRaises(ValueError) as e:
        importer.import_graph_def(
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
    with ops.Graph().as_default():
      _ = importer.import_graph_def(
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
    with ops.Graph().as_default():
      with self.assertRaises(ValueError) as e:
        _ = importer.import_graph_def(
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
    with ops.Graph().as_default():
      with self.assertRaises(ValueError) as e:
        importer.import_graph_def(
            self._MakeGraphDef("""
            node { name: 'A' op: 'Oi' }
            node { name: 'B' op: 'None' input: 'A:0' }
            """))
      self.assertTrue("More inputs specified ('A:0') than the op expects" in
                      str(e.exception))

  def testInvalidSignatureNotEnoughInputsInGraphDef(self):
    with ops.Graph().as_default():
      with self.assertRaises(ValueError) as e:
        importer.import_graph_def(
            self._MakeGraphDef("""
            node { name: 'A' op: 'Oi' }
            node { name: 'B' op: 'Iif' input: 'A:0' }
            """))
      self.assertTrue("Input types mismatch (expected 'int32, float32' but "
                      "got 'int32')" in str(e.exception))

  def testMissingInputOpInGraphDef(self):
    with ops.Graph().as_default():
      with self.assertRaises(ValueError) as e:
        importer.import_graph_def(
            self._MakeGraphDef("""
            node { name: 'B' op: 'If' input: 'A:0' }
            """))
      self.assertTrue("Input tensor 'A:0' not found" in str(e.exception))

  def testMissingInputOpInGraphDefButAppearsInInputMap(self):
    with ops.Graph().as_default():
      feed_a_0 = constant_op.constant(5.0)
      b, = importer.import_graph_def(
          self._MakeGraphDef("""
          node { name: 'B' op: 'If' input: 'A:0' }
          """),
          input_map={"A:0": feed_a_0},
          return_elements=["B"])
      self.assertEqual(b.inputs[0], feed_a_0)

  def testMissingInputTensorInGraphDef(self):
    with ops.Graph().as_default():
      with self.assertRaises(ValueError) as e:
        importer.import_graph_def(
            self._MakeGraphDef("""
            node { name: 'A' op: 'Of' }
            node { name: 'B' op: 'If' input: 'A:1' }
            """))
      self.assertTrue("Input tensor 'A:1' not found" in str(e.exception))

  def testMissingControlInputInGraphDef(self):
    with ops.Graph().as_default():
      with self.assertRaises(ValueError) as e:
        importer.import_graph_def(
            self._MakeGraphDef("""
            node { name: 'B' op: 'None' input: '^A' }
            """))
      self.assertTrue("Control input '^A' not found" in str(e.exception))

  def testInvalidTensorNameOutputIndexInGraphDef(self):
    with ops.Graph().as_default():
      with self.assertRaises(ValueError) as e:
        importer.import_graph_def(
            self._MakeGraphDef("""
            node { name: 'B' op: 'None' input: 'A:B' }
            """))
      self.assertEqual("Cannot convert 'A:B' to a tensor name.",
                       str(e.exception))

  def testInvalidTensorNameInGraphDef(self):
    with ops.Graph().as_default():
      with self.assertRaises(ValueError) as e:
        importer.import_graph_def(
            self._MakeGraphDef("""
            node { name: 'B' op: 'None' input: 'A:B:0' }
            """))
      self.assertEqual("Cannot convert 'A:B:0' to a tensor name.",
                       str(e.exception))

  def testMissingReturnOperation(self):
    with ops.Graph().as_default():
      with self.assertRaises(ValueError) as e:
        importer.import_graph_def(
            self._MakeGraphDef("""
            node { name: 'A' op: 'None' }
            """),
            return_elements=["B"])
      self.assertTrue(
          "return_element 'B' not found in graph_def." in str(e.exception))

  def testMissingReturnTensor(self):
    with ops.Graph().as_default():
      with self.assertRaises(ValueError) as e:
        importer.import_graph_def(
            self._MakeGraphDef("""
            node { name: 'A' op: 'Oi' }
            """),
            return_elements=["A:1"])
      self.assertTrue(
          "return_element 'A:1' not found in graph_def." in str(e.exception))

      with self.assertRaises(ValueError) as e:
        importer.import_graph_def(
            self._MakeGraphDef("""
            node { name: 'A' op: 'Oi' }
            """),
            return_elements=["B:0"])
      self.assertTrue(
          "return_element 'B:0' not found in graph_def." in str(e.exception))

      with self.assertRaises(ValueError) as e:
        importer.import_graph_def(
            self._MakeGraphDef("""
            node { name: 'A' op: 'Oi' }
            """),
            return_elements=["A:B:0"])
      self.assertTrue(
          "return_element 'A:B:0' not found in graph_def." in str(e.exception))

  def testMissingInputMap(self):
    with ops.Graph().as_default():
      with self.assertRaises(ValueError) as e:
        importer.import_graph_def(
            self._MakeGraphDef("""
            node { name: 'A' op: 'None' }
            """),
            input_map={"B:0": constant_op.constant(5.0)})
      self.assertTrue("not found in graph_def: [B:0]" in str(e.exception))

  def testInputMapTypeMismatch(self):
    with ops.Graph().as_default():
      with self.assertRaises(ValueError) as e:
        importer.import_graph_def(
            self._MakeGraphDef("""
            node { name: 'A' op: 'Oi' }
            node { name: 'B' op: 'Ii' input: 'A:0' }
            """),
            input_map={"A:0": constant_op.constant(5.0)})
      self.assertTrue(
          "Cannot convert a tensor of type float32 to an input of type int32."
          in str(e.exception))

  def testNoReturns(self):
    with ops.Graph().as_default() as g:
      ret = importer.import_graph_def(
          self._MakeGraphDef("""
          node { name: 'A' op: 'None' }
          """))
      self.assertEqual(ret, None)

      a = g.get_operation_by_name("import/A")
      self.assertEqual(a.type, "None")

  def testOverrideNamePrefix(self):
    with ops.Graph().as_default():
      a, = importer.import_graph_def(
          self._MakeGraphDef("""
          node { name: 'A' op: 'None' }
          """),
          return_elements=["A"],
          name="imported_graph")
      self.assertEqual(a.name, "imported_graph/A")

  def testNamePrefixColocationAttrs(self):
    original_graph_def = self._MakeGraphDef("""
          node { name: 'A' op: 'None' }
          node { name: 'B' op: 'None'  attr {
            key: '_class'
            value { list { s: 'loc:@A' } }
          } }""")

    with ops.Graph().as_default():
      b, = importer.import_graph_def(
          original_graph_def, return_elements=["B"], name="imported_graph")
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

    with ops.Graph().as_default():
      b, = importer.import_graph_def(
          original_graph_def, return_elements=["B"], name="")
      _, = importer.import_graph_def(
          original_graph_def, return_elements=["B"], name="")
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
    with ops.Graph().as_default():
      with self.assertRaisesRegexp(ValueError, "does not exist during import"):
        importer.import_graph_def(
            original_graph_def, return_elements=["B"], name="imported_graph")

  def testEmptyGraph(self):
    with ops.Graph().as_default() as g:
      init_version = g.version
      importer.import_graph_def(self._MakeGraphDef(""))
      self.assertEqual(init_version, g.version)

  def testInvalidInputForGraphDef(self):
    with ops.Graph().as_default():
      with self.assertRaises(TypeError) as e:
        importer.import_graph_def("")
      self.assertEqual("graph_def must be a GraphDef proto.", str(e.exception))

  def testInvalidInputForInputMap(self):
    with ops.Graph().as_default():
      with self.assertRaises(TypeError) as e:
        importer.import_graph_def(
            self._MakeGraphDef(""), input_map=[constant_op.constant(5.0)])
      self.assertEqual("input_map must be a dictionary mapping strings to "
                       "Tensor objects.", str(e.exception))
    graph_def = self._MakeGraphDef("""
         node { name: 'a' op: 'Placeholder'
                attr { key: 'dtype' value { type: DT_FLOAT } }}
         node { name: 'id' op: 'Identity' input: 'a:0'
                attr { key: 'T' value { type: DT_FLOAT } }}""")
    with ops.Graph().as_default():
      with self.assertRaises(ValueError) as e:
        importer.import_graph_def(
            graph_def,
            input_map={"a:0": variables.Variable(5.0)},
            name="")
      self.assertStartsWith(str(e.exception),
                            "tf.import_graph_def() requires a non-empty `name` "
                            "if `input_map` contains non-Tensor values.")
    with ops.Graph().as_default():
      t, = importer.import_graph_def(
          graph_def,
          input_map={"a:0": constant_op.constant(5.0)},
          name="",
          return_elements=["id:0"])
      with self.test_session():
        self.assertEqual(5.0, t.eval())

  def testInvalidInputForReturnOperations(self):
    with ops.Graph().as_default():
      with self.assertRaises(TypeError) as e:
        importer.import_graph_def(self._MakeGraphDef(""), return_elements=[7])
      self.assertEqual("return_elements must be a list of strings.",
                       str(e.exception))

  def testWithExtensionAndAttr(self):
    with ops.Graph().as_default() as g:
      c = constant_op.constant(5.0, dtype=dtypes.float32, name="c")
      array_ops.stack([c, c], name="pack")
    gdef = g.as_graph_def()

    with self.test_session():
      pack, = importer.import_graph_def(gdef, return_elements=["pack"])
      self.assertAllEqual(pack.outputs[0].eval(), [5.0, 5.0])

  def testWithDevice(self):
    with ops.Graph().as_default() as g:
      # No device.
      a = constant_op.constant(3.0, name="a")

      with ops.device("/cpu:0"):
        b = constant_op.constant(4.0, name="b")
      with ops.device("/job:worker"):
        c = constant_op.constant(5.0, name="c")

    gdef = g.as_graph_def()

    with ops.Graph().as_default():
      a2, b2, c2 = importer.import_graph_def(
          gdef, return_elements=["a", "b", "c"])
      self.assertEqual(a.device, a2.device)
      self.assertEqual(b.device, b2.device)
      self.assertEqual(c.device, c2.device)

    with ops.Graph().as_default():
      with ops.device(device.merge_device("/task:0")):
        a3, b3, c3 = importer.import_graph_def(
            gdef, return_elements=["a", "b", "c"])
        self.assertEqual("/task:0", a3.device)
        self.assertEqual("/task:0/device:CPU:0", b3.device)  # canonicalized.
        self.assertEqual(c.device + "/task:0", c3.device)

    with ops.Graph().as_default():
      with ops.device(device.merge_device("/job:ps")):
        a4, b4, c4 = importer.import_graph_def(
            gdef, return_elements=["a", "b", "c"])
        self.assertEqual("/job:ps", a4.device)
        self.assertEqual("/job:ps/device:CPU:0", b4.device)  # canonicalized.
        self.assertEqual(c.device, c4.device)  # worker overrides ps.

    with ops.Graph().as_default():
      with ops.device(device.merge_device("/gpu:0")):
        a5, b5, c5 = importer.import_graph_def(
            gdef, return_elements=["a", "b", "c"])
        self.assertEqual("/device:GPU:0", a5.device)
        self.assertEqual("/device:CPU:0", b5.device)  # cpu overrides gpu.
        self.assertEqual(c.device + "/device:GPU:0", c5.device)

  def testWithDeviceFunctionDependingOnInputs(self):
    with ops.Graph().as_default() as g:
      with ops.device("/job:ps"):
        v = variables.Variable(1.0)
      unused_assign_op = v.assign(2.0)
      unused_assign_2_op = v.assign(3.0)
      unused_add_t = v + v
    gdef = g.as_graph_def()

    # We'll use the following device function to observe ops with two inputs.
    ops_with_two_inputs = []

    def InputCounter(op):
      if any(in_t.dtype._is_ref_dtype for in_t in op.inputs):  # pylint: disable=protected-access
        ops_with_two_inputs.append(op)
      return ""

    with ops.Graph().as_default() as g:
      with ops.device(InputCounter):
        importer.import_graph_def(gdef)

    # We expect to see the initializer, two assign operations, and the add op.
    self.assertEqual(4, len(ops_with_two_inputs))

  def testGradient(self):
    with ops.Graph().as_default() as g:
      inputs = array_ops.placeholder(
          dtypes.float32, shape=[None, 100], name="input")
      weights = array_ops.placeholder(
          dtypes.float32, shape=[100, 10], name="weights")
      biases = array_ops.placeholder(dtypes.float32, shape=[10], name="biases")
      activations = nn_ops.relu(
          math_ops.matmul(inputs, weights) + biases, name="activations")
      loss = math_ops.reduce_mean(activations, name="loss")
    gdef = g.as_graph_def()

    with ops.Graph().as_default() as g:
      input_placeholder = array_ops.placeholder(dtypes.float32, shape=[32, 100])
      weights_var = variables.Variable(
          random_ops.truncated_normal([100, 10]), name="weights")
      biases_var = variables.Variable(array_ops.zeros([10]), name="biases")
      activations, loss = importer.import_graph_def(
          gdef,
          input_map={
              "input:0": input_placeholder,
              "weights:0": weights_var,
              "biases:0": biases_var
          },
          return_elements=["activations:0", "loss:0"])
      self.assertEqual([32, 10], activations.get_shape())
      self.assertEqual([], loss.get_shape())
      weights_grad, biases_grad = gradients_impl.gradients(
          loss, [weights_var, biases_var])
      self.assertEqual([100, 10], weights_grad.get_shape())
      self.assertEqual([10], biases_grad.get_shape())

  def testLargeGraph(self):
    with self.test_session():
      # The default message byte limit is 64M. Ours is 2G with a warning at 512.
      # Adding a 130M entries float32 tensor should exceed the warning, but not
      # the hard limit.
      input_shape = [130, 1000, 1000]
      tensor_input = np.ones(input_shape, dtype=np.float32)
      t = constant_op.constant(tensor_input, shape=input_shape)
      g = array_ops.identity(t)
      g.eval()

  def testVersion(self):
    v0 = versions.GRAPH_DEF_VERSION_MIN_CONSUMER
    v2 = versions.GRAPH_DEF_VERSION
    v1 = (v0 + v2) // 2
    for producer in v0, v1, v2:
      for min_consumer in v0, v1, v2:
        with ops.Graph().as_default():
          a, = importer.import_graph_def(
              self._MakeGraphDef(
                  "node { name: 'A' op: 'Oii' }",
                  producer=producer,
                  min_consumer=min_consumer),
              return_elements=["A"])
          self.assertEqual(a.graph.graph_def_versions.producer, producer)
          self.assertEqual(a.graph.graph_def_versions.min_consumer,
                           min_consumer)

  def testVersionLow(self):
    with ops.Graph().as_default() as g:
      pat = (r"GraphDef producer version -1 below min producer %d supported "
             r"by TensorFlow \S+\.  Please regenerate your graph.$" %
             versions.GRAPH_DEF_VERSION_MIN_PRODUCER)
      importer.import_graph_def(self._MakeGraphDef("", producer=-1))
      x = constant_op.constant(
          7)  # Need at least one op to get a C++ graph generated
      with self.test_session(graph=g) as sess:
        with self.assertRaisesRegexp(Exception, pat):
          sess.run(x)

  def testVersionHigh(self):
    with ops.Graph().as_default() as g:
      pat = (r"GraphDef min consumer version %d above current version %d "
             r"for TensorFlow \S+\.  Please upgrade TensorFlow\.$" %
             (1 << 30, versions.GRAPH_DEF_VERSION))
      importer.import_graph_def(self._MakeGraphDef("", min_consumer=1 << 30))
      x = constant_op.constant(
          7)  # Need at least one op to get a C++ graph generated
      with self.test_session(graph=g) as sess:
        with self.assertRaisesRegexp(Exception, pat):
          sess.run(x)

  def testVersionAppliesToOpConstruction(self):
    """These tests rely on shape fns in test_ops.cc."""
    with ops.Graph().as_default():
      importer.import_graph_def(
          self._MakeGraphDef(
              "node { name: 'A' op: 'RequiresOlderGraphVersion' }",
              producer=versions.GRAPH_DEF_VERSION - 1),
          return_elements=["A"])

    with ops.Graph().as_default():
      with self.assertRaisesWithPredicateMatch(ValueError,
                                               "Wrong graph version.*"):
        importer.import_graph_def(
            self._MakeGraphDef(
                "node { name: 'A' op: 'RequiresOlderGraphVersion' }",
                producer=versions.GRAPH_DEF_VERSION),
            return_elements=["A"])

  def testDefaultAttrsAdded(self):
    with ops.Graph().as_default():
      a = importer.import_graph_def(
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
    with ops.Graph().as_default():
      a = importer.import_graph_def(
          self._MakeGraphDef("""
          node { name: 'A' op: 'OpWithFutureDefaultAttr'
                 attr { key: 'default_int' value { i: 456 } } }
          """),
          return_elements=["A"],
          producer_op_list=producer_op_list)
      with self.assertRaisesRegexp(ValueError, "No attr named 'default_int'"):
        a[0].get_attr("default_int")

    # Attr only in producer_op_list with non-default value is preserved.
    with ops.Graph().as_default():
      a = importer.import_graph_def(
          self._MakeGraphDef("""
          node { name: 'A' op: 'OpWithFutureDefaultAttr'
                 attr { key: 'default_int' value { i: 987 } } }
          """),
          return_elements=["A"],
          producer_op_list=producer_op_list)
      self.assertEqual(987, a[0].get_attr("default_int"))

  def testFunctions(self):
    dtype = dtypes.float32
    @function.Defun(dtype, dtype, dtype, dtype)
    def Grad(x, y, dout1, dout2):  # pylint: disable=unused-argument
      # Return the inputs for simplicity of testing. The correct return value
      # would be (dout1 + dout2, dout1 - dout2)
      return x, y

    @function.Defun(dtype, dtype, grad_func=Grad)
    def FuncWithGrad(x, y):
      return x + y, x - y

    @function.Defun(dtypes.int32)
    def ExternalTensorFunc(x):
      # c must be defined in the containing graph
      return x + c

    @function.Defun(dtypes.int32, dtypes.int32)
    def OuterFunc(x, y):

      @function.Defun(dtypes.int32)
      def InnerFunc(x):
        return x + x

      return InnerFunc(x) + y

    # Create graph with function calls and export to GraphDef
    with ops.Graph().as_default() as g1:
      p1 = array_ops.placeholder(dtype, name="p1")
      p2 = array_ops.placeholder(dtype, name="p2")
      # pylint: disable=unexpected-keyword-arg
      a, b = FuncWithGrad(p1, p2, name="f")

      c = constant_op.constant(10, dtype=dtypes.int32)
      ExternalTensorFunc(1, name="external")

      OuterFunc(10, 1, name="outer")
      # pylint: enable=unexpected-keyword-arg

    gdef = g1.as_graph_def()

    # Import GraphDef into new graph, add imported gradients, and test that
    # imported functions can be run
    with ops.Graph().as_default() as g2:
      p1, p2, a, b = importer.import_graph_def(
          gdef, return_elements=["p1:0", "p2:0", "f:0", "f:1"], name="")
      grad = gradients_impl.gradients([a], [p1, p2])

      with self.test_session(graph=g2) as sess:
        feed_dict = {p1: 1, p2: 2}
        a_val, b_val, grad_val = sess.run([a, b, grad], feed_dict=feed_dict)
        self.assertEqual(a_val, 3.0)
        self.assertEqual(b_val, -1.0)
        # Grad function returns inputs values for testing
        self.assertEqual(grad_val, [1.0, 2.0])
        self.assertEqual(sess.run("external:0"), 11)
        self.assertEqual(sess.run("outer:0"), 21)

    # Export the new graph and reimport to test that imported functions can be
    # successfully exported/imported again
    gdef = g2.as_graph_def()
    with ops.Graph().as_default() as g3:
      p1, p2, a, b = importer.import_graph_def(
          gdef, return_elements=["p1:0", "p2:0", "f:0", "f:1"], name="")
      # Create new gradient functions (in additional to the imported gradient
      # functions created in g2).
      grad = gradients_impl.gradients([a], [p1, p2])

      with self.test_session(graph=g3) as sess:
        feed_dict = {p1: 1, p2: 2}
        a_val, b_val, grad_val = sess.run([a, b, grad], feed_dict=feed_dict)
        self.assertEqual(a_val, 3.0)
        self.assertEqual(b_val, -1.0)
        self.assertEqual(grad_val, [1.0, 2.0])
        self.assertEqual(sess.run("external:0"), 11)
        self.assertEqual(sess.run("outer:0"), 21)

  def testImportInsideDefun(self):
    g = ops.Graph()
    with g.as_default():
      @function.Defun()
      def Add2(x, y):
        return math_ops.add(x, y)

      x = constant_op.constant(3.0, dtype=dtypes.float32)
      y = constant_op.constant(-5.0, dtype=dtypes.float32)
      z = Add2(x, y, name="z")  # pylint: disable=unexpected-keyword-arg

    gdef = g.as_graph_def()

    @function.Defun()
    def TestFunc():
      return importer.import_graph_def(gdef, return_elements=["z:0"])[0]

    z = TestFunc()

    with self.test_session():
      z_val = z.eval()
      self.assertEqual(z_val, -2.0)


if __name__ == "__main__":
  test.main()
