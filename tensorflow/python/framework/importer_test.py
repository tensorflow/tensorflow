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
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_ops  # pylint: disable=unused-import
from tensorflow.python.framework import versions
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
import tensorflow.python.ops.nn_grad  # pylint: disable=unused-import
from tensorflow.python.platform import test


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
          node { name: 'A' op: 'IntOutputFloatOutput' }
          node { name: 'B' op: 'ListOutput'
                 attr { key: 'T'
                        value { list { type: DT_INT32 type: DT_FLOAT } } } }
          node { name: 'C' op: 'ListInput'
                 attr { key: 'N' value { i: 2 } }
                 attr { key: 'T' value { type: DT_INT32 } }
                 input: 'A:0' input: 'B:0' }
          node { name: 'D' op: 'ListInput'
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
      self.assertEqual(a.type, "IntOutputFloatOutput")
      self.assertEqual(b.type, "ListOutput")
      self.assertEqual(c.type, "ListInput")
      self.assertEqual(d.type, "ListInput")
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

  def testMultipleImport(self):
    graph_def = self._MakeGraphDef("""
    node { name: 'A' op: 'IntOutput' }
    node { name: 'B' op: 'IntInput' input: 'A:0' }
    """)

    with ops.Graph().as_default():
      # Initial import
      a, b = importer.import_graph_def(
          graph_def,
          return_elements=["A", "B"],
          name="")
      self.assertEqual(a.name, "A")
      self.assertEqual(b.name, "B")
      self.assertEqual(list(b.inputs), [a.outputs[0]])

      # Repeat the same import
      a1, b1 = importer.import_graph_def(
          graph_def,
          return_elements=["A", "B"],
          name="")
      self.assertEqual(a1.name, "A_1")
      self.assertEqual(b1.name, "B_1")
      self.assertEqual(list(b1.inputs), [a1.outputs[0]])

      # Repeat the same import again
      a2, b2 = importer.import_graph_def(
          graph_def,
          return_elements=["A", "B"],
          name="")
      self.assertEqual(a2.name, "A_2")
      self.assertEqual(b2.name, "B_2")
      self.assertEqual(list(b2.inputs), [a2.outputs[0]])

      # Import with an already-used name
      a3, b3 = importer.import_graph_def(
          graph_def,
          return_elements=["A", "B"],
          name="A")
      self.assertEqual(a3.name, "A_3/A")
      self.assertEqual(b3.name, "A_3/B")
      self.assertEqual(list(b3.inputs), [a3.outputs[0]])

      # Import with an already-used name but with a '/' to indicate an
      # "absolute" name scope (see the Graph.name_scope docstring).
      a_a, a_b = importer.import_graph_def(
          graph_def,
          return_elements=["A", "B"],
          name="A/")
      self.assertEqual(a_a.name, "A/A")
      self.assertEqual(a_b.name, "A/B")
      self.assertEqual(list(a_b.inputs), [a_a.outputs[0]])

      # Repeat the same import.
      a_a1, a_b1 = importer.import_graph_def(
          graph_def,
          return_elements=["A", "B"],
          name="A/")
      self.assertEqual(a_a1.name, "A/A_1")
      self.assertEqual(a_b1.name, "A/B_1")
      self.assertEqual(list(a_b1.inputs), [a_a1.outputs[0]])

      # Import with existing de-duped node names
      a1_1, b1_1 = importer.import_graph_def(
          self._MakeGraphDef("""
          node { name: 'A_1' op: 'IntOutput' }
          node { name: 'B_1' op: 'IntInput' input: 'A_1:0' }
          """),
          return_elements=["A_1", "B_1"],
          name="")
      self.assertEqual(a1_1.name, "A_1_1")
      self.assertEqual(b1_1.name, "B_1_1")
      self.assertEqual(list(b1_1.inputs), [a1_1.outputs[0]])

      # Create a name scope and then import node with same name
      with ops.name_scope("foo"):
        constant_op.constant(1)
      foo, = importer.import_graph_def(
          self._MakeGraphDef("node { name: 'foo' op: 'IntOutput' }"),
          return_elements=["foo"],
          name="")
      self.assertEqual(foo.name, "foo_1")

      # Imported node name can't conflict with intermediate name scope (but can
      # conflict with outer scope and full name scope)
      with ops.name_scope("outer"):
        with ops.name_scope("inner"):
          c = constant_op.constant(1, name="c")
          self.assertEqual(c.op.name, "outer/inner/c")

      outer, inner, new_c, outer_inner, outer_inner_c = (
          importer.import_graph_def(
              self._MakeGraphDef(
                  "node { name: 'outer' op: 'IntOutput' }"
                  "node { name: 'inner' op: 'IntOutput' }"
                  "node { name: 'c' op: 'IntOutput' }"
                  "node { name: 'outer/inner' op: 'IntOutput' }"
                  "node { name: 'outer/inner/c' op: 'IntOutput' }"),
              return_elements=["outer", "inner", "c", "outer/inner",
                               "outer/inner/c"],
              name=""))
      self.assertEqual(outer.name, "outer_1")
      self.assertEqual(inner.name, "inner")
      self.assertEqual(new_c.name, "c")
      self.assertEqual(outer_inner.name, "outer/inner_1")
      self.assertEqual(outer_inner_c.name, "outer/inner/c_1")

  def testEmptyNameScope(self):
    with ops.Graph().as_default():
      # Create name scope but don't create any ops with it
      with ops.name_scope("foo"):
        pass

      # Import graph def that uses name scope name
      op, = importer.import_graph_def(
          self._MakeGraphDef("node { name: 'foo' op: 'IntOutput' }"),
          return_elements=["foo"],
          name="")

      self.assertEqual(op.name, "foo")

  def testInputMap(self):
    with ops.Graph().as_default():
      feed_a_0 = constant_op.constant(0, dtype=dtypes.int32)
      feed_b_1 = constant_op.constant(1, dtype=dtypes.int32)

      a, b, c, d = importer.import_graph_def(
          self._MakeGraphDef("""
          node { name: 'A' op: 'TwoIntOutputs' }
          node { name: 'B' op: 'TwoIntOutputs' }
          node { name: 'C' op: 'ListInput'
                 attr { key: 'N' value { i: 2 } }
                 attr { key: 'T' value { type: DT_INT32 } }
                 input: 'A:0' input: 'B:0' }
          node { name: 'D' op: 'ListInput'
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
          node { name: 'A' op: 'TwoIntOutputs' }
          node { name: 'B' op: 'TwoIntOutputs' }
          node { name: 'C' op: 'ListInput'
                 attr { key: 'N' value { i: 2 } }
                 attr { key: 'T' value { type: DT_INT32 } }
                 input: 'A:0' input: 'B:0' }
          node { name: 'D' op: 'ListInput'
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
          node { name: 'A' op: 'TwoIntOutputs' }
          node { name: 'B' op: 'TwoIntOutputs' }
          node { name: 'C' op: 'ListInput'
                 attr { key: 'N' value { i: 2 } }
                 attr { key: 'T' value { type: DT_INT32 } }
                 input: 'A:0' input: 'B:0' }
          node { name: 'D' op: 'ListInput'
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
          node { name: 'A' op: 'TwoIntOutputs' }
          node { name: 'B' op: 'IntInput' input: 'A' }
          """),
          return_elements=["A", "B"])

      self.assertEqual(b.inputs[0], a.outputs[0])

  def testInputMapImplicitZerothOutput(self):
    with ops.Graph().as_default():
      feed_a_0 = constant_op.constant(0, dtype=dtypes.int32)
      b, = importer.import_graph_def(
          self._MakeGraphDef("""
          node { name: 'A' op: 'TwoIntOutputs' }
          node { name: 'B' op: 'IntInput' input: 'A:0' }
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
          node { name: 'A' op: 'RefOutput' }
          node { name: 'B' op: 'IntOutput' }
          node { name: 'C' op: 'TwoIntInputs' input: 'A:0' input: 'B:0' }
          node { name: 'D' op: 'RefInputIntInput' input: 'A:0' input: 'B:0' }
          """),
          return_elements=["A", "B", "C", "D"])

      self.assertEqual(c.inputs[0], a.outputs[0])
      self.assertEqual(c.inputs[1], b.outputs[0])
      self.assertEqual(d.inputs[0], a.outputs[0])
      self.assertEqual(d.inputs[1], b.outputs[0])

      self.assertEqual(a.outputs[0].dtype, dtypes.int32_ref)
      self.assertEqual(c._input_types, [dtypes.int32, dtypes.int32])
      self.assertEqual(c.outputs, [])
      self.assertEqual(d._input_types, [dtypes.int32_ref, dtypes.int32])
      self.assertEqual(d.outputs, [])

  def testResources(self):
    # Produce GraphDef containing a ops producing and consuming resources.
    graph = ops.Graph()
    with graph.as_default():
      var = resource_variable_ops.ResourceVariable(1.0)
      var_assign = var.assign(2.0)
      # Use an op that requires handle shape to be set.
      var_shape = resource_variable_ops.variable_shape(var.handle)
      init = variables.global_variables_initializer()
    graph_def = graph.as_graph_def()

    # Import the GraphDef.
    with ops.Graph().as_default():
      # pylint: disable=unused-variable
      imported_var, imported_assign, imported_shape, imported_init = (
          importer.import_graph_def(
              graph_def,
              return_elements=[var.name, var_assign.name, var_shape.name,
                               init.name]))

      # Make sure the handle shape is set on the imported variable.
      new_var_shape = resource_variable_ops.variable_shape(imported_var)
      # pylint: enable=unused-variable

      # Run the imported graph.
      # TODO(b/76173421): make this work (currently DCHECKS)
      # with self.cached_session() as sess:
      #   sess.run(imported_init)
      #   self.assertEqual(self.evaluate(imported_var), 1.0)
      #   self.assertEqual(self.evaluate(imported_assign), 2.0)
      #   self.assertEqual(list(self.evaluate(imported_shape)), [])
      #   self.assertEqual(list(self.evaluate(new_var_shape)), [])

  def testWhileLoop(self):
    # Produce GraphDef containing while loop.
    graph = ops.Graph()
    with graph.as_default():
      r = control_flow_ops.while_loop(lambda i: i < 10, lambda i: i + 1, [0])
      # Add an op that consumes the while loop output.
      math_ops.add(r, 1)
    graph_def = graph.as_graph_def()

    # Import the GraphDef and make sure it runs.
    with ops.Graph().as_default():
      imported_r, = importer.import_graph_def(graph_def,
                                              return_elements=[r.name])
      self.assertEqual(imported_r.name, "import/" + r.name)
      with self.cached_session() as sess:
        self.assertEqual(self.evaluate(imported_r), 10)

  def testImportWhileLoopInCond(self):
    # Produce GraphDef containing while loop.
    graph = ops.Graph()
    with graph.as_default():
      r = control_flow_ops.while_loop(lambda i: i < 10, lambda i: i + 1, [0])
    graph_def = graph.as_graph_def()

    # Import the GraphDef inside a cond and make sure it runs.
    with ops.Graph().as_default():

      def ImportFn():
        return importer.import_graph_def(graph_def, return_elements=[r.name])[0]

      pred = array_ops.placeholder(dtypes.bool)
      out = control_flow_ops.cond(pred, ImportFn,
                                  lambda: constant_op.constant(1))
      with self.cached_session() as sess:
        self.assertEqual(sess.run(out, {pred: True}), 10)
        self.assertEqual(sess.run(out, {pred: False}), 1)

  def testImportWhileLoopInWhileLoop(self):
    self.skipTest("b/111757448")
    # Produce GraphDef containing while loop.
    graph = ops.Graph()
    with graph.as_default():
      r = control_flow_ops.while_loop(lambda i: i < 10, lambda i: i + 1, [0])
    graph_def = graph.as_graph_def()

    # Import the GraphDef inside another loop and make sure it runs.
    with ops.Graph().as_default():

      def ImportFn(_):
        return importer.import_graph_def(graph_def, return_elements=[r.name])[0]

      out = control_flow_ops.while_loop(
          lambda i: i < 2, ImportFn, [0],
          shape_invariants=[tensor_shape.TensorShape(None)])
      with self.cached_session() as sess:
        self.assertEqual(self.evaluate(out), 10)

  def testTypeMismatchInGraphDef(self):
    # TODO(skyewm): improve error message
    error_msg = ("Input 0 of node import/B was passed int32 from import/A:0 "
                 "incompatible with expected float.")
    with ops.Graph().as_default():
      with self.assertRaisesRegexp(ValueError, error_msg):
        importer.import_graph_def(
            self._MakeGraphDef("""
            node { name: 'A' op: 'IntOutput' }
            node { name: 'B' op: 'FloatInput' input: 'A:0' }
            """))

  def testShapeWhitelist(self):
    # Barrier's shape is an output vector of 2, but the
    # graph says it's a scalar.  This is currently whitelisted.
    with ops.Graph().as_default():
      _ = importer.import_graph_def(
          self._MakeGraphDef("""
          node { name: 'A' op: 'Barrier'
                 attr { key: '_output_shapes'
                        value { list { shape { } } } }
                 attr { key: 'component_types'
                        value { list { type: DT_FLOAT } } } }
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
              node { name: 'A' op: 'FloatOutput' }
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
      # TODO(skyewm): improve error message
      with self.assertRaisesRegexp(
          ValueError,
          "NodeDef expected inputs '' do not match 1 inputs specified"):
        importer.import_graph_def(
            self._MakeGraphDef("""
            node { name: 'A' op: 'IntOutput' }
            node { name: 'B' op: 'None' input: 'A:0' }
            """))

  def testInvalidSignatureNotEnoughInputsInGraphDef(self):
    with ops.Graph().as_default():
      # TODO(skyewm): improve error message
      with self.assertRaisesRegexp(
          ValueError,
          "NodeDef expected inputs 'int32, float' do not match 1 inputs "
          "specified"):
        importer.import_graph_def(
            self._MakeGraphDef("""
            node { name: 'A' op: 'IntOutput' }
            node { name: 'B' op: 'IntInputFloatInput' input: 'A:0' }
            """))

  def testMissingInputOpInGraphDef(self):
    with ops.Graph().as_default():
      with self.assertRaisesRegexp(ValueError,
                                   "Node 'B': Unknown input node 'A:0'"):
        importer.import_graph_def(
            self._MakeGraphDef("""
            node { name: 'B' op: 'FloatInput' input: 'A:0' }
            """))

  def testMissingInputOpInGraphDefButAppearsInInputMap(self):
    with ops.Graph().as_default():
      feed_a_0 = constant_op.constant(5.0)
      b, = importer.import_graph_def(
          self._MakeGraphDef("""
          node { name: 'B' op: 'FloatInput' input: 'A:0' }
          """),
          input_map={"A:0": feed_a_0},
          return_elements=["B"])
      self.assertEqual(b.inputs[0], feed_a_0)

  def testMissingInputTensorInGraphDef(self):
    with ops.Graph().as_default():
      with self.assertRaisesRegexp(
          ValueError,
          "Node 'B': Connecting to invalid output 1 of source node A "
          "which has 1 outputs"):
        importer.import_graph_def(
            self._MakeGraphDef("""
            node { name: 'A' op: 'FloatOutput' }
            node { name: 'B' op: 'FloatInput' input: 'A:1' }
            """))

  def testMissingControlInputInGraphDef(self):
    with ops.Graph().as_default():
      with self.assertRaisesRegexp(ValueError,
                                   r"Node 'B': Unknown input node '\^A'"):
        importer.import_graph_def(
            self._MakeGraphDef("""
            node { name: 'B' op: 'None' input: '^A' }
            """))

  def testInvalidTensorNameOutputIndexInGraphDef(self):
    with ops.Graph().as_default():
      with self.assertRaisesRegexp(ValueError,
                                   "Node 'B': Unknown input node 'A:B'"):
        importer.import_graph_def(
            self._MakeGraphDef("""
            node { name: 'B' op: 'None' input: 'A:B' }
            """))

  def testInvalidTensorNameInGraphDef(self):
    with ops.Graph().as_default():
      with self.assertRaisesRegexp(ValueError,
                                   "Node 'B': Unknown input node 'A:B:0'"):
        importer.import_graph_def(
            self._MakeGraphDef("""
            node { name: 'B' op: 'None' input: 'A:B:0' }
            """))

  def testMissingReturnOperation(self):
    with ops.Graph().as_default():
      with self.assertRaisesRegexp(
          ValueError, "Requested return node 'B' not found in graph def"):
        importer.import_graph_def(
            self._MakeGraphDef("""
            node { name: 'A' op: 'None' }
            """),
            return_elements=["B"])

  def testMissingReturnTensor(self):
    with ops.Graph().as_default():
      with self.assertRaisesRegexp(
          ValueError,
          r"Invalid return output 1 of node 'A', which has 1 output\(s\)"):
        importer.import_graph_def(
            self._MakeGraphDef("""
            node { name: 'A' op: 'IntOutput' }
            """),
            return_elements=["A:1"])

      with self.assertRaisesRegexp(
          ValueError, "Requested return tensor 'B:0' not found in graph def"):
        importer.import_graph_def(
            self._MakeGraphDef("""
            node { name: 'A' op: 'IntOutput' }
            """),
            return_elements=["B:0"])

      with self.assertRaisesRegexp(ValueError,
                                   "Cannot convert 'A:B:0' to a tensor name."):
        importer.import_graph_def(
            self._MakeGraphDef("""
            node { name: 'A' op: 'IntOutput' }
            """),
            return_elements=["A:B:0"])

  def testMissingInputMap(self):
    with ops.Graph().as_default():
      with self.assertRaisesRegexp(
          ValueError,
          r"Attempted to map inputs that were not found in graph_def: \[B:0\]"):
        importer.import_graph_def(
            self._MakeGraphDef("""
            node { name: 'A' op: 'None' }
            """),
            input_map={"B:0": constant_op.constant(5.0)})

  def testInputMapUnusedAsInput(self):
    with ops.Graph().as_default():
      # Mapping an unused node output should succeed.
      importer.import_graph_def(
          self._MakeGraphDef("""
          node { name: 'A' op: 'IntOutput' }
          """),
          input_map={"A:0": constant_op.constant(5.0)})

      # Mapping a non-existent output of an existing node should fail.
      with self.assertRaisesRegexp(
          ValueError,
          r"Attempted to map inputs that were not found in graph_def: \[A:2\]"):
        importer.import_graph_def(
            self._MakeGraphDef("""
            node { name: 'A' op: 'IntOutput' }
            """),
            input_map={"A:2": constant_op.constant(5.0)})

  def testInputMapTypeMismatch(self):
    with ops.Graph().as_default():
      with self.assertRaisesRegexp(
          ValueError, "Input 0 of node import/B was passed float from Const:0 "
          "incompatible with expected int32."):
        importer.import_graph_def(
            self._MakeGraphDef("""
            node { name: 'A' op: 'IntOutput' }
            node { name: 'B' op: 'IntInput' input: 'A:0' }
            """),
            input_map={"A:0": constant_op.constant(5.0)})

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

  def testDefaultNamePrefix(self):
    with ops.Graph().as_default():
      a, = importer.import_graph_def(
          self._MakeGraphDef("""
          node { name: 'A' op: 'None' }
          """),
          return_elements=["A"],
          name=None)
      self.assertEqual(a.name, "import/A")

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
      self.assertTrue("_class" in b.node_def.attr)
      self.assertProtoEquals(
          "list { s: 'loc:@imported_graph/A' }",
          b.node_def.attr["_class"])

  def testColocationAndDevice(self):
    # A and B are colocated, device set on A.
    original_graph_def = self._MakeGraphDef("""
          node { name: 'A' op: 'None' device: '/device:CPU:0' attr {
            key: '_class'
            value { list { s: 'loc:@A' } }
          } }
          node { name: 'B' op: 'None'  attr {
            key: '_class'
            value { list { s: 'loc:@A' } }
          } }""")

    with ops.Graph().as_default():
      a, b = importer.import_graph_def(original_graph_def,
                                       return_elements=["A", "B"],
                                       name="")
      self.assertEqual(a.device, "/device:CPU:0")
      self.assertEqual(b.device, "/device:CPU:0")
      self.assertEqual(a.colocation_groups(), [b"loc:@A"])
      self.assertEqual(b.colocation_groups(), [b"loc:@A"])

    # A and B are colocated, device set on B.
    original_graph_def = self._MakeGraphDef("""
          node { name: 'A' op: 'None' attr {
            key: '_class'
            value { list { s: 'loc:@A' } }
          } }
          node { name: 'B' op: 'None' device: '/device:CPU:0' attr {
            key: '_class'
            value { list { s: 'loc:@A' } }
          } }""")

    with ops.Graph().as_default():
      a, b = importer.import_graph_def(original_graph_def,
                                       return_elements=["A", "B"],
                                       name="")
      # TODO(skyewm): this behavior seems inconsistent with the above. Why is
      # B's device ignored?
      self.assertEqual(a.device, "")
      self.assertEqual(b.device, "")
      self.assertEqual(a.colocation_groups(), [b"loc:@A"])
      self.assertEqual(b.colocation_groups(), [b"loc:@A"])

  def testColocationWithDeviceFn(self):
    original_graph_def = self._MakeGraphDef("""
          node { name: 'A' op: 'None' attr {
            key: '_class'
            value { list { s: 'loc:@A' } }
          } }
          node { name: 'B' op: 'None'  attr {
            key: '_class'
            value { list { s: 'loc:@A' } }
          } }""")

    # A device function that places "A" on one device and "B" on
    # another device.  Because B is colocated with A, we test that B's
    # device function is overridden by A.
    def CustomDeviceFn(op):
      if "A" in op.name:
        return "/device:A:0"
      else:
        return "/device:B:0"

    with ops.Graph().as_default():
      with ops.device(CustomDeviceFn):
        a, b = importer.import_graph_def(original_graph_def,
                                         return_elements=["A", "B"],
                                         name="imported_graph")
      self.assertEqual(a.device, "/device:A:0")
      self.assertEqual(b.device, "/device:A:0")
      self.assertEqual(a.colocation_groups(), [b"loc:@imported_graph/A"])
      self.assertEqual(b.colocation_groups(), [b"loc:@imported_graph/A"])

    # Test a scenario where 'A' doesn't get a device; 'A' should not have a
    # device, but during runtime will get colocated with 'B' because of the
    # colocation attribute. B's device function is still overridden by A.
    def BDeviceFn(op):
      if "B" in op.name:
        return "/device:B:0"
      return ""

    with ops.Graph().as_default():
      with ops.device(BDeviceFn):
        a, b = importer.import_graph_def(original_graph_def,
                                         return_elements=["A", "B"],
                                         name="imported_graph")
      self.assertEqual(a.device, "")
      self.assertEqual(b.device, "")
      self.assertEqual(a.colocation_groups(), [b"loc:@imported_graph/A"])
      self.assertEqual(b.colocation_groups(), [b"loc:@imported_graph/A"])

    # Only A gets a device, so B inherits it implicitly.
    def ADeviceFn(op):
      if "A" in op.name:
        return "/device:A:0"
      return ""

    with ops.Graph().as_default():
      with ops.device(ADeviceFn):
        a, b = importer.import_graph_def(original_graph_def,
                                         return_elements=["A", "B"],
                                         name="imported_graph")
      self.assertEqual(a.device, "/device:A:0")
      self.assertEqual(b.device, "/device:A:0")
      self.assertEqual(a.colocation_groups(), [b"loc:@imported_graph/A"])
      self.assertEqual(b.colocation_groups(), [b"loc:@imported_graph/A"])

  def testMultipleColocationWithDeviceFn(self):
    original_graph_def = self._MakeGraphDef("""
          node { name: 'A' op: 'None'}
          node { name: 'B' op: 'None'}
          node { name: 'C' op: 'None'  attr {
            key: '_class'
            value { list { s: 'loc:@A' s: 'loc:@B' } }
          } }""")

    # A device function that places "B" on a device, and "A" is empty.
    #
    # B and C should contain "/device:B".  A will not right now.  But
    # because of the colocation property, at runtime it would be
    # placed with B and C.
    def CustomDeviceFn(op):
      if "B" in op.name:
        return "/device:B:0"
      return ""

    with ops.Graph().as_default():
      with ops.device(CustomDeviceFn):
        a, b, c = importer.import_graph_def(original_graph_def,
                                            return_elements=["A", "B", "C"],
                                            name="imported_graph")
      self.assertEqual(a.device, "")
      self.assertEqual(b.device, "/device:B:0")
      self.assertEqual(c.device, "/device:B:0")
      self.assertEqual(a.colocation_groups(), [b"loc:@imported_graph/A"])
      self.assertEqual(b.colocation_groups(), [b"loc:@imported_graph/B"])
      self.assertEqual(c.colocation_groups(),
                       [b"loc:@imported_graph/A", b"loc:@imported_graph/B"])

  def testNamePrefixColocationAttrsMultipleImport(self):
    original_graph_def = self._MakeGraphDef("""
          node { name: 'A' op: 'None' }
          node { name: 'B' op: 'None'  attr {
            key: '_class'
            value { list { s: 'loc:@A' } }
          } }""")

    with ops.Graph().as_default():
      a, b = importer.import_graph_def(
          original_graph_def, return_elements=["A", "B"], name="")
      a_1, b_1 = importer.import_graph_def(
          original_graph_def, return_elements=["A", "B"], name="")

      self.assertEqual(a.name, "A")
      self.assertEqual(b.name, "B")
      self.assertEqual(b.colocation_groups(), [b"loc:@A"])

      self.assertEqual(a_1.name, "A_1")
      self.assertEqual(b_1.name, "B_1")
      self.assertEqual(b_1.colocation_groups(), [b"loc:@A_1"])

  def testNamePrefixColocationAttrsNotFound(self):
    original_graph_def = self._MakeGraphDef("""
          node { name: 'B' op: 'None'  attr {
            key: '_class'
            value { list { s: 'loc:@A' } }
          } }""")

    with ops.Graph().as_default():
      with self.assertRaisesRegexp(
          ValueError, "Node 'B' expects to be colocated with unknown node 'A'"):
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
      with self.cached_session():
        self.assertEqual(5.0, self.evaluate(t))

  def testInvalidInputForReturnOperations(self):
    with ops.Graph().as_default():
      with self.assertRaisesRegexp(
          TypeError, "return_elements must be a list of strings."):
        importer.import_graph_def(self._MakeGraphDef(""), return_elements=[7])

      with self.assertRaisesRegexp(ValueError,
                                   "Cannot convert 'a:b:c' to a tensor name."):
        importer.import_graph_def(
            self._MakeGraphDef(""), return_elements=["a:b:c"])

  def testDuplicateOperationNames(self):
    with self.assertRaisesRegexp(ValueError, "Node 'A' is not unique"):
      importer.import_graph_def(
          self._MakeGraphDef("""
          node { name: 'A' op: 'IntOutput' }
          node { name: 'B' op: 'IntOutput' }
          node { name: 'A' op: 'IntOutput' }
          """))

  def testWithExtensionAndAttr(self):
    with ops.Graph().as_default() as g:
      c = constant_op.constant(5.0, dtype=dtypes.float32, name="c")
      array_ops.stack([c, c], name="pack")
    gdef = g.as_graph_def()

    with self.cached_session():
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
      with ops.device(device.merge_device("/device:GPU:0")):
        a5, b5, c5 = importer.import_graph_def(
            gdef, return_elements=["a", "b", "c"])
        self.assertEqual("/device:GPU:0", a5.device)
        self.assertEqual("/device:CPU:0", b5.device)  # cpu overrides gpu.
        self.assertEqual(c.device + "/device:GPU:0", c5.device)

  def testWithDeviceFunctionDependingOnInputs(self):
    with ops.Graph().as_default() as g:
      with ops.device("/job:ps"):
        v1 = constant_op.constant(1.0)
        v2 = constant_op.constant(1.0)
      _ = v1 + v2
      _ = v1 - v2
      _ = array_ops.identity(v1)
    gdef = g.as_graph_def()

    # We'll use the following device function to observe ops with two inputs.
    ops_with_two_inputs = []

    def InputCounter(op):
      if len(op.inputs) == 2:
        ops_with_two_inputs.append(op)
      return ""

    with ops.Graph().as_default() as g:
      with ops.device(InputCounter):
        importer.import_graph_def(gdef)

    # We expect to see the add and subtract, but not identity.
    self.assertEqual(2, len(ops_with_two_inputs))

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
    with self.cached_session():
      # The default message byte limit is 64M. Ours is 2G with a warning at 512.
      # Adding a 130M entries float32 tensor should exceed the warning, but not
      # the hard limit.
      input_shape = [130, 1000, 1000]
      tensor_input = np.ones(input_shape, dtype=np.float32)
      t = constant_op.constant(tensor_input, shape=input_shape)
      g = array_ops.identity(t)
      self.evaluate(g)

  def testVersion(self):
    v0 = versions.GRAPH_DEF_VERSION_MIN_CONSUMER
    v2 = versions.GRAPH_DEF_VERSION
    v1 = (v0 + v2) // 2
    for producer in v0, v1, v2:
      for min_consumer in v0, v1, v2:
        with ops.Graph().as_default():
          a, = importer.import_graph_def(
              self._MakeGraphDef(
                  "node { name: 'A' op: 'TwoIntOutputs' }",
                  producer=producer,
                  min_consumer=min_consumer),
              return_elements=["A"])
          self.assertEqual(a.graph.graph_def_versions.producer, producer)
          self.assertEqual(a.graph.graph_def_versions.min_consumer,
                           min_consumer)

  def testVersionLow(self):
    with ops.Graph().as_default():
      with self.assertRaisesRegexp(
          Exception,
          r"GraphDef producer version -1 below min producer %d supported "
          r"by TensorFlow \S+\.  Please regenerate your graph.$" %
          versions.GRAPH_DEF_VERSION_MIN_PRODUCER):
        importer.import_graph_def(self._MakeGraphDef("", producer=-1))

  def testVersionHigh(self):
    with ops.Graph().as_default():
      with self.assertRaisesRegexp(
          ValueError,
          r"GraphDef min consumer version %d above current version %d "
          r"for TensorFlow \S+\.  Please upgrade TensorFlow\.$" %
          (1 << 30, versions.GRAPH_DEF_VERSION)):
        importer.import_graph_def(self._MakeGraphDef("", min_consumer=1 << 30))

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
      with self.assertRaisesRegexp(
          ValueError, "Operation 'import/A' has no attr named 'default_int'."):
        a[0].get_attr("default_int")

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

      with self.session(graph=g2) as sess:
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

      with self.session(graph=g3) as sess:
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

    with self.cached_session():
      z_val = self.evaluate(z)
      self.assertEqual(z_val, -2.0)

  def testImportGraphWithFunctionTwice(self):
    g = ops.Graph()
    with g.as_default():

      @function.Defun()
      def Add2(x, y):
        return math_ops.add(x, y)

      x = array_ops.placeholder(dtype=dtypes.float32, name="x")
      y = array_ops.placeholder(dtype=dtypes.float32, name="y")
      _ = Add2(x, y, name="z")  # pylint: disable=unexpected-keyword-arg

    gdef = g.as_graph_def()

    x = random_ops.random_uniform(dtype=dtypes.float32, shape=())
    y = random_ops.random_uniform(dtype=dtypes.float32, shape=())
    input_map = {"x:0": x, "y:0": y}

    with ops.name_scope("first"):
      z1 = importer.import_graph_def(gdef, return_elements=["z:0"],
                                     input_map=input_map)[0]

    with ops.name_scope("second"):
      z2 = importer.import_graph_def(gdef, return_elements=["z:0"],
                                     input_map=input_map)[0]

    with self.cached_session() as sess:
      z1_val, z2_val = sess.run((z1, z2))
      self.assertAllEqual(z1_val, z2_val)


if __name__ == "__main__":
  test.main()
