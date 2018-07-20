# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tensorflow.python.framework.function_def_to_graph."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import function
from tensorflow.python.framework import function_def_to_graph
from tensorflow.python.framework import graph_to_function_def
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class FunctionDefToGraphTest(test.TestCase):

  def _build_function_def(self):
    with ops.Graph().as_default() as g:
      # Inputs
      x = array_ops.placeholder(dtypes.float32, name="x")
      y = array_ops.placeholder(dtypes.float32, name="y")

      # Outputs
      sum_squares = math_ops.add_n(
          [math_ops.pow(x, 2), math_ops.pow(y, 2)], name="sum_squares")
      sum_cubes = math_ops.add_n(
          [math_ops.pow(x, 3), math_ops.pow(y, 3)], name="sum_cubes")
    fdef = graph_to_function_def.graph_to_function_def(
        g,
        g.get_operations(),
        [x, y],  # Inputs
        [sum_squares, sum_cubes])  # Outputs.
    fdef.signature.name = "_whats_in_a_name"
    return fdef

  def testInputsAndOutputs(self):
    fdef = self._build_function_def()
    g = function_def_to_graph.function_def_to_graph(fdef)
    self.assertEqual(g.name, "_whats_in_a_name")
    with self.test_session(graph=g) as sess:
      inputs = sess.run(g.inputs, feed_dict={"x:0": 2, "y:0": 3})
      self.assertSequenceEqual(inputs, [2.0, 3.0])
      outputs = sess.run(g.outputs, feed_dict={"x:0": 2, "y:0": 3})
      self.assertSequenceEqual(outputs, [13.0, 35.0])

  def testShapes(self):
    fdef = self._build_function_def()

    g = function_def_to_graph.function_def_to_graph(fdef)
    self.assertIsNone(g.inputs[0].shape.dims)  # Unknown dims.
    self.assertIsNone(g.inputs[1].shape.dims)  # Unknown dims.
    self.assertIsNone(g.outputs[0].shape.dims)  # Unknown dims.
    self.assertIsNone(g.outputs[1].shape.dims)  # Unknown dims.

    g = function_def_to_graph.function_def_to_graph(
        fdef, input_shapes=[tensor_shape.vector(5),
                            tensor_shape.vector(5)])
    self.assertSequenceEqual(g.inputs[0].shape.dims, [5])
    self.assertSequenceEqual(g.inputs[1].shape.dims, [5])
    self.assertSequenceEqual(g.outputs[0].shape.dims, [5])
    self.assertSequenceEqual(g.outputs[1].shape.dims, [5])

    g = function_def_to_graph.function_def_to_graph(
        fdef, input_shapes=[None, tensor_shape.matrix(5, 7)])
    self.assertIsNone(g.inputs[0].shape.dims)
    self.assertSequenceEqual(g.inputs[1].shape.dims, [5, 7])
    self.assertSequenceEqual(g.outputs[0].shape.dims, [5, 7])
    self.assertSequenceEqual(g.outputs[1].shape.dims, [5, 7])

    # Should raise a ValueError if the length of input_shapes does not match
    # the number of input args in FunctionDef.signature.input_arg.
    with self.assertRaises(ValueError):
      g = function_def_to_graph.function_def_to_graph(
          fdef, input_shapes=[tensor_shape.matrix(5, 7)])


class FunctionDefToGraphDefTest(test.TestCase):

  def _build_function_def(self):
    with ops.Graph().as_default() as g:
      # Inputs:    x    y    z
      #            |\   |   /
      #            | \  |  /
      #            |  foo_1     list_output
      #            |   / \       /       \
      #            | d_1 e_1  a:1        a:0
      #            |  \   |   /           |
      #            |   \  |  /            |
      #            |    foo_2             |
      #            |     / \              |
      # Outputs:   x   d_2 e_2           a:0

      x = array_ops.placeholder(dtypes.float32, name="x")
      y = array_ops.placeholder(dtypes.int32, name="y")
      z = array_ops.placeholder(dtypes.int32, name="z")

      d_1, e_1 = test_ops._op_def_lib.apply_op(
          "Foo1", name="foo_1", a=x, b=y, c=z)

      list_output0, list_output1 = test_ops.list_output(
          T=[dtypes.int32, dtypes.int32], name="list_output")

      d_2, e_2 = test_ops.foo1(a=d_1, b=e_1, c=list_output1, name="foo_2")

    fdef = graph_to_function_def.graph_to_function_def(
        g,
        g.get_operations(),
        [x, y, z],  # Inputs
        [x, d_2, e_2, list_output0])  # Outputs.

    # Assert that the FunctionDef was correctly built.
    assert len(fdef.node_def) == 3  # 2 Foo1 nodes and 1 ListOutput node.
    assert fdef.node_def[0].op == "Foo1"
    assert fdef.node_def[0].input == ["x", "y", "z"]
    assert fdef.node_def[1].op == "ListOutput"
    assert not fdef.node_def[1].input
    assert fdef.node_def[2].op == "Foo1"
    assert fdef.node_def[2].input == [
        "foo_1:d:0", "foo_1:e:0", "list_output:a:1"
    ]
    return fdef

  def testTensorNames(self):
    fdef = self._build_function_def()
    g, tensor_name_map = function_def_to_graph.function_def_to_graph_def(fdef)

    # Verify that inputs of body nodes are correctly renamed.
    # foo_1
    self.assertSequenceEqual(g.node[3].input, ["x:0", "y:0", "z:0"])
    # foo_2
    self.assertSequenceEqual(g.node[5].input,
                             ["foo_1:0", "foo_1:1", "list_output:1"])

    # Verify that the `tensor_name_map` has the correct mapping.
    self.assertDictEqual(
        tensor_name_map, {
            "x": "x:0",
            "y": "y:0",
            "z": "z:0",
            "foo_1:d:0": "foo_1:0",
            "foo_1:e:0": "foo_1:1",
            "list_output:a:0": "list_output:0",
            "list_output:a:1": "list_output:1",
            "foo_2:d:0": "foo_2:0",
            "foo_2:e:0": "foo_2:1",
        })

  def testShapes(self):
    fdef = self._build_function_def()
    g, _ = function_def_to_graph.function_def_to_graph_def(
        fdef,
        input_shapes=[tensor_shape.scalar(),
                      tensor_shape.vector(5), None])
    self.assertEqual("shape" in g.node[0].attr, True)
    self.assertSequenceEqual(
        tensor_shape.TensorShape(g.node[0].attr["shape"].shape).as_list(), [])
    self.assertEqual(g.node[0].attr["shape"].shape.unknown_rank, False)
    self.assertEqual("shape" in g.node[1].attr, True)
    self.assertSequenceEqual(
        tensor_shape.TensorShape(g.node[1].attr["shape"].shape).as_list(), [5])
    self.assertEqual(g.node[0].attr["shape"].shape.unknown_rank, False)
    self.assertFalse("shape" in g.node[2].attr)

  def testFunctionCallsFromFunction(self):
    x = constant_op.constant(5.0)
    y = constant_op.constant(10.0)

    @function.Defun()
    def fn():

      @function.Defun()
      def inner_fn():
        return x + y

      return inner_fn()

    # Instantiate the function in this graph so that
    # `function_def_to_graph` can find it.
    fn()

    def fn2():
      return 2 * fn()

    fdef = function._DefinedFunction(fn2, [], []).definition
    func_graph = function_def_to_graph.function_def_to_graph(fdef)
    with func_graph.as_default():
      x_ph, y_ph = func_graph.inputs
      with self.test_session(graph=func_graph) as sess:
        self.assertEqual(
            sess.run(func_graph.outputs[0], feed_dict={
                x_ph: 5.0,
                y_ph: 10.0
            }), 30.0)


if __name__ == "__main__":
  test.main()
