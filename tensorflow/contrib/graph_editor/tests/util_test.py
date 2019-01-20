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
"""Tests for tensorflow.contrib.graph_editor."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib import graph_editor as ge
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class UtilTest(test.TestCase):

  def test_list_view(self):
    """Test for ge.util.ListView."""
    l = [0, 1, 2]
    lv = ge.util.ListView(l)
    # Should not be the same id.
    self.assertIsNot(l, lv)
    # Should behave the same way than the original list.
    self.assertTrue(len(lv) == 3 and lv[0] == 0 and lv[1] == 1 and lv[2] == 2)
    # Should be read only.
    with self.assertRaises(TypeError):
      lv[0] = 0

  def test_is_iterable(self):
    """Test for ge.util.is_iterable."""
    self.assertTrue(ge.util.is_iterable([0, 1, 2]))
    self.assertFalse(ge.util.is_iterable(3))

  def test_unique_graph(self):
    """Test for ge.util.check_graphs and ge.util.get_unique_graph."""
    g0 = ops.Graph()
    with g0.as_default():
      a0 = constant_op.constant(1)
      b0 = constant_op.constant(2)
    g1 = ops.Graph()
    with g1.as_default():
      a1 = constant_op.constant(1)
      b1 = constant_op.constant(2)
    # Same graph, should be fine.
    self.assertIsNone(ge.util.check_graphs(a0, b0))
    # Two different graphs, should assert.
    with self.assertRaises(ValueError):
      ge.util.check_graphs(a0, b0, a1, b1)
    # a0 and b0 belongs to the same graph, should be fine.
    self.assertEqual(ge.util.get_unique_graph([a0, b0]), g0)
    # Different graph, should raise an error.
    with self.assertRaises(ValueError):
      ge.util.get_unique_graph([a0, b0, a1, b1])

  def test_make_list_of_op(self):
    """Test for ge.util.make_list_of_op."""
    g0 = ops.Graph()
    with g0.as_default():
      a0 = constant_op.constant(1)
      b0 = constant_op.constant(2)
    # Should extract the ops from the graph.
    self.assertEqual(len(ge.util.make_list_of_op(g0)), 2)
    # Should extract the ops from the tuple.
    self.assertEqual(len(ge.util.make_list_of_op((a0.op, b0.op))), 2)

  def test_make_list_of_t(self):
    """Test for ge.util.make_list_of_t."""
    g0 = ops.Graph()
    with g0.as_default():
      a0 = constant_op.constant(1)
      b0 = constant_op.constant(2)
      c0 = math_ops.add(a0, b0)  # pylint: disable=unused-variable
    # Should extract the tensors from tre graph.
    self.assertEqual(len(ge.util.make_list_of_t(g0)), 3)
    # Should extract the tensors from the tuple
    self.assertEqual(len(ge.util.make_list_of_t((a0, b0))), 2)
    # Should extract the tensors and ignore the ops.
    self.assertEqual(
        len(ge.util.make_list_of_t(
            (a0, a0.op, b0), ignore_ops=True)), 2)

  def test_get_generating_consuming(self):
    """Test for ge.util.get_generating_ops and ge.util.get_generating_ops."""
    g0 = ops.Graph()
    with g0.as_default():
      a0 = constant_op.constant(1)
      b0 = constant_op.constant(2)
      c0 = math_ops.add(a0, b0)
    self.assertEqual(len(ge.util.get_generating_ops([a0, b0])), 2)
    self.assertEqual(len(ge.util.get_consuming_ops([a0, b0])), 1)
    self.assertEqual(len(ge.util.get_generating_ops([c0])), 1)
    self.assertEqual(ge.util.get_consuming_ops([c0]), [])

  def test_control_outputs(self):
    """Test for the ge.util.ControlOutputs class."""
    g0 = ops.Graph()
    with g0.as_default():
      a0 = constant_op.constant(1)
      b0 = constant_op.constant(2)
      x0 = constant_op.constant(3)
      with ops.control_dependencies([x0.op]):
        c0 = math_ops.add(a0, b0)  # pylint: disable=unused-variable
    control_outputs = ge.util.ControlOutputs(g0).get_all()
    self.assertEqual(len(control_outputs), 1)
    self.assertEqual(len(control_outputs[x0.op]), 1)
    self.assertIs(list(control_outputs[x0.op])[0], c0.op)

  def test_scope(self):
    """Test simple path scope functionalities."""
    self.assertEqual(ge.util.scope_finalize("foo/bar"), "foo/bar/")
    self.assertEqual(ge.util.scope_dirname("foo/bar/op"), "foo/bar/")
    self.assertEqual(ge.util.scope_basename("foo/bar/op"), "op")

  def test_placeholder(self):
    """Test placeholder functionalities."""
    g0 = ops.Graph()
    with g0.as_default():
      a0 = constant_op.constant(1, name="foo")
    # Test placeholder name.
    self.assertEqual(ge.util.placeholder_name(a0), "geph__foo_0")
    self.assertEqual(ge.util.placeholder_name(None), "geph")
    self.assertEqual(
        ge.util.placeholder_name(
            a0, scope="foo/"), "foo/geph__foo_0")
    self.assertEqual(
        ge.util.placeholder_name(
            a0, scope="foo"), "foo/geph__foo_0")
    self.assertEqual(ge.util.placeholder_name(None, scope="foo/"), "foo/geph")
    self.assertEqual(ge.util.placeholder_name(None, scope="foo"), "foo/geph")
    # Test placeholder creation.
    g0 = ops.Graph()
    with g0.as_default():
      a0 = constant_op.constant(1, dtype=dtypes.float32, name="a0")
      c0 = math_ops.add(
          ge.util.make_placeholder_from_tensor(a0),
          ge.util.make_placeholder_from_dtype_and_shape(dtype=dtypes.float32))
      self.assertEqual(c0.op.inputs[0].op.name, "geph__a0_0")
      self.assertEqual(c0.op.inputs[1].op.name, "geph")


if __name__ == "__main__":
  test.main()
