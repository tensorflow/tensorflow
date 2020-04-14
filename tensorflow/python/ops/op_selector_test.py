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
"""Tests for op_selector.py."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import ops as ops_lib
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import op_selector
from tensorflow.python.platform import test


class SelectTest(test.TestCase):

  def setUp(self):
    self.graph = ops_lib.Graph()
    with self.graph.as_default():
      self.a = constant_op.constant([1., 1.], shape=[2], name="a")
      with ops_lib.name_scope("foo"):
        self.b = constant_op.constant([2., 2.], shape=[2], name="b")
        self.c = math_ops.add(self.a, self.b, name="c")
        self.d = constant_op.constant([3., 3.], shape=[2], name="d")
        with ops_lib.name_scope("bar"):
          self.e = math_ops.add(self.c, self.d, name="e")
          self.f = math_ops.add(self.c, self.d, name="f")
          self.g = math_ops.add(self.c, self.a, name="g")
          with ops_lib.control_dependencies([self.c.op]):
            self.h = math_ops.add(self.f, self.g, name="h")

  def test_is_iterable(self):
    """Test for is_iterable."""
    self.assertTrue(op_selector.is_iterable([0, 1, 2]))
    self.assertFalse(op_selector.is_iterable(3))

  def test_unique_graph(self):
    """Test for check_graphs and get_unique_graph."""
    g0 = ops_lib.Graph()
    with g0.as_default():
      a0 = constant_op.constant(1)
      b0 = constant_op.constant(2)
    g1 = ops_lib.Graph()
    with g1.as_default():
      a1 = constant_op.constant(1)
      b1 = constant_op.constant(2)
    # Same graph, should be fine.
    self.assertIsNone(op_selector.check_graphs(a0, b0))
    # Two different graphs, should assert.
    with self.assertRaises(ValueError):
      op_selector.check_graphs(a0, b0, a1, b1)
    # a0 and b0 belongs to the same graph, should be fine.
    self.assertEqual(op_selector.get_unique_graph([a0, b0]), g0)
    # Different graph, should raise an error.
    with self.assertRaises(ValueError):
      op_selector.get_unique_graph([a0, b0, a1, b1])

  def test_unique_graph_func_graph(self):
    """Test for get_unique_graph with FuncGraph."""
    outer = ops_lib.Graph()
    with outer.as_default():
      k1 = constant_op.constant(1)
      inner = func_graph.FuncGraph("inner")
      inner._graph_key = outer._graph_key
      with inner.as_default():
        k2 = constant_op.constant(2)

    unique_graph = op_selector.get_unique_graph([k1, k2])
    self.assertEqual(unique_graph._graph_key, inner._graph_key)

  def test_make_list_of_op(self):
    """Test for make_list_of_op."""
    g0 = ops_lib.Graph()
    with g0.as_default():
      a0 = constant_op.constant(1)
      b0 = constant_op.constant(2)
    # Should extract the ops from the graph.
    self.assertEqual(len(op_selector.make_list_of_op(g0)), 2)
    # Should extract the ops from the tuple.
    self.assertEqual(len(op_selector.make_list_of_op((a0.op, b0.op))), 2)

  def test_make_list_of_t(self):
    """Test for make_list_of_t."""
    g0 = ops_lib.Graph()
    with g0.as_default():
      a0 = constant_op.constant(1)
      b0 = constant_op.constant(2)
      c0 = math_ops.add(a0, b0)  # pylint: disable=unused-variable
    # Should extract the tensors from tre graph.
    self.assertEqual(len(op_selector.make_list_of_t(g0)), 3)
    # Should extract the tensors from the tuple
    self.assertEqual(len(op_selector.make_list_of_t((a0, b0))), 2)
    # Should extract the tensors and ignore the ops.
    self.assertEqual(
        len(op_selector.make_list_of_t(
            (a0, a0.op, b0), ignore_ops=True)), 2)

  def test_get_generating_consuming(self):
    """Test for get_generating_ops and get_consuming_ops."""
    g0 = ops_lib.Graph()
    with g0.as_default():
      a0 = constant_op.constant(1)
      b0 = constant_op.constant(2)
      c0 = math_ops.add(a0, b0)
    self.assertEqual(len(op_selector.get_generating_ops([a0, b0])), 2)
    self.assertEqual(len(op_selector.get_consuming_ops([a0, b0])), 1)
    self.assertEqual(len(op_selector.get_generating_ops([c0])), 1)
    self.assertEqual(op_selector.get_consuming_ops([c0]), [])

  def test_backward_walk_ops(self):
    seed_ops = [self.h.op]
    # Include all ops except for self.g.op
    within_ops = [
        x.op for x in [self.a, self.b, self.c, self.d, self.e, self.f, self.h]
    ]
    # For the fn, exclude self.c.op.
    within_ops_fn = lambda op: op not in (self.c.op,)
    stop_at_ts = (self.f,)

    with self.graph.as_default():
      # Backward walk only includes h since we stop at f and g is not within.
      ops = op_selector.get_backward_walk_ops(
          seed_ops,
          inclusive=True,
          within_ops=within_ops,
          within_ops_fn=within_ops_fn,
          stop_at_ts=stop_at_ts)
      self.assertEqual(set(ops), set([self.h.op]))

      # If we do inclusive=False, the result is empty.
      ops = op_selector.get_backward_walk_ops(
          seed_ops,
          inclusive=False,
          within_ops=within_ops,
          within_ops_fn=within_ops_fn,
          stop_at_ts=stop_at_ts)
      self.assertEqual(set(ops), set())

      # Removing stop_at_fs adds f.op, d.op.
      ops = op_selector.get_backward_walk_ops(
          seed_ops,
          inclusive=True,
          within_ops=within_ops,
          within_ops_fn=within_ops_fn)
      self.assertEqual(set(ops), set([self.d.op, self.f.op, self.h.op]))

      # Not using within_ops_fn adds back ops for a, b, c.
      ops = op_selector.get_backward_walk_ops(
          seed_ops, inclusive=True, within_ops=within_ops)
      self.assertEqual(
          set(ops),
          set([
              self.a.op, self.b.op, self.c.op, self.d.op, self.f.op, self.h.op
          ]))

      # Vanially backward search via self.h.op includes everything except e.op.
      ops = op_selector.get_backward_walk_ops(seed_ops, inclusive=True)
      self.assertEqual(
          set(ops),
          set([
              self.a.op, self.b.op, self.c.op, self.d.op, self.f.op, self.g.op,
              self.h.op
          ]))


if __name__ == "__main__":
  test.main()
