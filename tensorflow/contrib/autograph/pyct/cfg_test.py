# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for cfg module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.autograph.pyct import cfg
from tensorflow.contrib.autograph.pyct import parser
from tensorflow.python.platform import test


class CountingVisitor(cfg.GraphVisitor):

  def __init__(self, graph):
    super(CountingVisitor, self).__init__(graph)
    self.counts = {}

  def init_state(self, _):
    return None

  def visit_node(self, node):
    self.counts[node.ast_node] = self.counts.get(node.ast_node, 0) + 1
    return False  # visit only once


class GraphVisitorTest(test.TestCase):

  def _build_cfg(self, fn):
    node, _ = parser.parse_entity(fn)
    cfgs = cfg.build(node)
    return cfgs, node

  def test_basic_coverage_forward(self):

    def test_fn(a):
      while a > 0:
        a = 1
        break
        return a  # pylint:disable=unreachable
      a = 2

    graphs, node = self._build_cfg(test_fn)
    graph, = graphs.values()
    visitor = CountingVisitor(graph)
    visitor.visit_forward()
    fn_node = node.body[0]

    self.assertEqual(visitor.counts[fn_node.args], 1)
    self.assertEqual(visitor.counts[fn_node.body[0].test], 1)
    self.assertEqual(visitor.counts[fn_node.body[0].body[0]], 1)
    self.assertEqual(visitor.counts[fn_node.body[0].body[1]], 1)
    # The return node should be unreachable in forward direction.
    self.assertTrue(fn_node.body[0].body[2] not in visitor.counts)
    self.assertEqual(visitor.counts[fn_node.body[1]], 1)

  def test_basic_coverage_reverse(self):

    def test_fn(a):
      while a > 0:
        a = 1
        break
        return a  # pylint:disable=unreachable
      a = 2

    graphs, node = self._build_cfg(test_fn)
    graph, = graphs.values()
    visitor = CountingVisitor(graph)
    visitor.visit_reverse()
    fn_node = node.body[0]

    self.assertEqual(visitor.counts[fn_node.args], 1)
    self.assertEqual(visitor.counts[fn_node.body[0].test], 1)
    self.assertEqual(visitor.counts[fn_node.body[0].body[0]], 1)
    self.assertEqual(visitor.counts[fn_node.body[0].body[1]], 1)
    self.assertTrue(visitor.counts[fn_node.body[0].body[2]], 1)
    self.assertEqual(visitor.counts[fn_node.body[1]], 1)


class AstToCfgTest(test.TestCase):

  def _build_cfg(self, fn):
    node, _ = parser.parse_entity(fn)
    cfgs = cfg.build(node)
    return cfgs

  def _repr_set(self, node_set):
    return frozenset(repr(n) for n in node_set)

  def _as_set(self, elements):
    if elements is None:
      return frozenset()
    elif isinstance(elements, str):
      return frozenset((elements,))
    else:
      return frozenset(elements)

  def assertGraphMatches(self, graph, edges):
    """Tests whether the CFG contains the specified edges."""
    for prev, node_repr, next_ in edges:
      matched = False
      for cfg_node in graph.index.values():
        if repr(cfg_node) == node_repr:
          if (self._as_set(prev) == frozenset(map(repr, cfg_node.prev)) and
              self._as_set(next_) == frozenset(map(repr, cfg_node.next))):
            matched = True
            break
      if not matched:
        self.fail(
            'match failed for node "%s" in graph:\n%s' % (node_repr, graph))

  def assertStatementEdges(self, graph, edges):
    """Tests whether the CFG contains the specified statement edges."""
    for prev_node_reprs, node_repr, next_node_reprs in edges:
      matched = False
      partial_matches = []
      self.assertSetEqual(
          frozenset(graph.stmt_next.keys()), frozenset(graph.stmt_prev.keys()))
      for stmt_ast_node in graph.stmt_next:
        ast_repr = '%s:%s' % (stmt_ast_node.__class__.__name__,
                              stmt_ast_node.lineno)
        if ast_repr == node_repr:
          actual_next = frozenset(map(repr, graph.stmt_next[stmt_ast_node]))
          actual_prev = frozenset(map(repr, graph.stmt_prev[stmt_ast_node]))
          partial_matches.append((actual_prev, node_repr, actual_next))
          if (self._as_set(prev_node_reprs) == actual_prev and
              self._as_set(next_node_reprs) == actual_next):
            matched = True
            break
      if not matched:
        self.fail('edges mismatch for %s: %s' % (node_repr, partial_matches))

  def test_straightline(self):

    def test_fn(a):
      a += 1
      a = 2
      a = 3
      return

    graph, = self._build_cfg(test_fn).values()

    self.assertGraphMatches(
        graph,
        (
            (None, 'a', 'a += 1'),
            ('a += 1', 'a = 2', 'a = 3'),
            ('a = 2', 'a = 3', 'return'),
            ('a = 3', 'return', None),
        ),
    )

  def test_straightline_no_return(self):

    def test_fn(a, b):
      a = b + 1
      a += max(a)

    graph, = self._build_cfg(test_fn).values()

    self.assertGraphMatches(
        graph,
        (
            (None, 'a, b', 'a = b + 1'),
            ('a = b + 1', 'a += max(a)', None),
        ),
    )

  def test_unreachable_code(self):

    def test_fn(a):
      return
      a += 1  # pylint:disable=unreachable

    graph, = self._build_cfg(test_fn).values()

    self.assertGraphMatches(
        graph,
        (
            (None, 'a', 'return'),
            ('a', 'return', None),
            (None, 'a += 1', None),
        ),
    )

  def test_if_straightline(self):

    def test_fn(a):
      if a > 0:
        a = 1
      else:
        a += -1

    graph, = self._build_cfg(test_fn).values()

    self.assertGraphMatches(
        graph,
        (
            (None, 'a', '(a > 0)'),
            ('(a > 0)', 'a = 1', None),
            ('(a > 0)', 'a += -1', None),
        ),
    )
    self.assertStatementEdges(
        graph,
        (('a', 'If:2', None),),
    )

  def test_branch_nested(self):

    def test_fn(a):
      if a > 0:
        if a > 1:
          a = 1
        else:
          a = 2
      else:
        if a > 2:
          a = 3
        else:
          a = 4

    graph, = self._build_cfg(test_fn).values()

    self.assertGraphMatches(
        graph,
        (
            (None, 'a', '(a > 0)'),
            ('a', '(a > 0)', ('(a > 1)', '(a > 2)')),
            ('(a > 0)', '(a > 1)', ('a = 1', 'a = 2')),
            ('(a > 1)', 'a = 1', None),
            ('(a > 1)', 'a = 2', None),
            ('(a > 0)', '(a > 2)', ('a = 3', 'a = 4')),
            ('(a > 2)', 'a = 3', None),
            ('(a > 2)', 'a = 4', None),
        ),
    )
    self.assertStatementEdges(
        graph,
        (
            ('a', 'If:2', None),
            ('(a > 0)', 'If:3', None),
            ('(a > 0)', 'If:8', None),
        ),
    )

  def test_branch_straightline_semi(self):

    def test_fn(a):
      if a > 0:
        a = 1

    graph, = self._build_cfg(test_fn).values()

    self.assertGraphMatches(
        graph,
        (
            (None, 'a', '(a > 0)'),
            ('a', '(a > 0)', 'a = 1'),
            ('(a > 0)', 'a = 1', None),
        ),
    )
    self.assertStatementEdges(
        graph,
        (('a', 'If:2', None),),
    )

  def test_branch_return(self):

    def test_fn(a):
      if a > 0:
        return
      else:
        a = 1
      a = 2

    graph, = self._build_cfg(test_fn).values()

    self.assertGraphMatches(
        graph,
        (
            ('a', '(a > 0)', ('return', 'a = 1')),
            ('(a > 0)', 'a = 1', 'a = 2'),
            ('(a > 0)', 'return', None),
            ('a = 1', 'a = 2', None),
        ),
    )
    self.assertStatementEdges(
        graph,
        (('a', 'If:2', 'a = 2'),),
    )

  def test_branch_return_minimal(self):

    def test_fn(a):
      if a > 0:
        return

    graph, = self._build_cfg(test_fn).values()

    self.assertGraphMatches(
        graph,
        (
            ('a', '(a > 0)', 'return'),
            ('(a > 0)', 'return', None),
        ),
    )
    self.assertStatementEdges(
        graph,
        (('a', 'If:2', None),),
    )

  def test_while_straightline(self):

    def test_fn(a):
      while a > 0:
        a = 1
      a = 2

    graph, = self._build_cfg(test_fn).values()

    self.assertGraphMatches(
        graph,
        (
            (('a', 'a = 1'), '(a > 0)', ('a = 1', 'a = 2')),
            ('(a > 0)', 'a = 1', '(a > 0)'),
            ('(a > 0)', 'a = 2', None),
        ),
    )
    self.assertStatementEdges(
        graph,
        (('a', 'While:2', 'a = 2'),),
    )

  def test_while_else_straightline(self):

    def test_fn(a):
      while a > 0:
        a = 1
      else:  # pylint:disable=useless-else-on-loop
        a = 2
      a = 3

    graph, = self._build_cfg(test_fn).values()

    self.assertGraphMatches(
        graph,
        (
            (('a', 'a = 1'), '(a > 0)', ('a = 1', 'a = 2')),
            ('(a > 0)', 'a = 1', '(a > 0)'),
            ('(a > 0)', 'a = 2', 'a = 3'),
            ('a = 2', 'a = 3', None),
        ),
    )
    self.assertStatementEdges(
        graph,
        (('a', 'While:2', 'a = 3'),),
    )

  def test_while_else_continue(self):

    def test_fn(a):
      while a > 0:
        if a > 1:
          continue
        else:
          a = 0
        a = 1
      else:  # pylint:disable=useless-else-on-loop
        a = 2
      a = 3

    graph, = self._build_cfg(test_fn).values()

    self.assertGraphMatches(
        graph,
        (
            (('a', 'continue', 'a = 1'), '(a > 0)', ('(a > 1)', 'a = 2')),
            ('(a > 0)', '(a > 1)', ('continue', 'a = 0')),
            ('(a > 1)', 'continue', '(a > 0)'),
            ('a = 0', 'a = 1', '(a > 0)'),
            ('(a > 0)', 'a = 2', 'a = 3'),
            ('a = 2', 'a = 3', None),
        ),
    )
    self.assertStatementEdges(
        graph,
        (
            ('a', 'While:2', 'a = 3'),
            ('(a > 0)', 'If:3', ('a = 1', '(a > 0)')),
        ),
    )

  def test_while_else_break(self):

    def test_fn(a):
      while a > 0:
        if a > 1:
          break
        a = 1
      else:
        a = 2
      a = 3

    graph, = self._build_cfg(test_fn).values()

    self.assertGraphMatches(
        graph,
        (
            (('a', 'a = 1'), '(a > 0)', ('(a > 1)', 'a = 2')),
            ('(a > 0)', '(a > 1)', ('break', 'a = 1')),
            ('(a > 1)', 'break', 'a = 3'),
            ('(a > 1)', 'a = 1', '(a > 0)'),
            ('(a > 0)', 'a = 2', 'a = 3'),
            (('break', 'a = 2'), 'a = 3', None),
        ),
    )
    self.assertStatementEdges(
        graph,
        (
            ('a', 'While:2', 'a = 3'),
            ('(a > 0)', 'If:3', ('a = 1', 'a = 3')),
        ),
    )

  def test_while_else_return(self):

    def test_fn(a):
      while a > 0:
        if a > 1:
          return
        a = 1
      else:  # pylint:disable=useless-else-on-loop
        a = 2
      a = 3

    graph, = self._build_cfg(test_fn).values()

    self.assertGraphMatches(
        graph,
        (
            (('a', 'a = 1'), '(a > 0)', ('(a > 1)', 'a = 2')),
            ('(a > 0)', '(a > 1)', ('return', 'a = 1')),
            ('(a > 1)', 'return', None),
            ('(a > 1)', 'a = 1', '(a > 0)'),
            ('(a > 0)', 'a = 2', 'a = 3'),
            ('a = 2', 'a = 3', None),
        ),
    )
    self.assertStatementEdges(
        graph,
        (
            ('a', 'While:2', 'a = 3'),
            ('(a > 0)', 'If:3', 'a = 1'),
        ),
    )

  def test_while_nested_straightline(self):

    def test_fn(a):
      while a > 0:
        while a > 1:
          a = 1
        a = 2
      a = 3

    graph, = self._build_cfg(test_fn).values()

    self.assertGraphMatches(
        graph,
        (
            (('a', 'a = 2'), '(a > 0)', ('(a > 1)', 'a = 3')),
            (('(a > 0)', 'a = 1'), '(a > 1)', ('a = 1', 'a = 2')),
            ('(a > 1)', 'a = 1', '(a > 1)'),
            ('(a > 1)', 'a = 2', '(a > 0)'),
            ('(a > 0)', 'a = 3', None),
        ),
    )
    self.assertStatementEdges(
        graph,
        (
            ('a', 'While:2', 'a = 3'),
            ('(a > 0)', 'While:3', 'a = 2'),
        ),
    )

  def test_while_nested_continue(self):

    def test_fn(a):
      while a > 0:
        while a > 1:
          if a > 3:
            continue
          a = 1
        a = 2
      a = 3

    graph, = self._build_cfg(test_fn).values()

    self.assertGraphMatches(
        graph,
        (
            (('a', 'a = 2'), '(a > 0)', ('(a > 1)', 'a = 3')),
            (('(a > 0)', 'continue', 'a = 1'), '(a > 1)', ('(a > 3)', 'a = 2')),
            ('(a > 1)', '(a > 3)', ('continue', 'a = 1')),
            ('(a > 3)', 'continue', '(a > 1)'),
            ('(a > 3)', 'a = 1', '(a > 1)'),
            ('(a > 1)', 'a = 2', '(a > 0)'),
            ('(a > 0)', 'a = 3', None),
        ),
    )
    self.assertStatementEdges(
        graph,
        (
            ('a', 'While:2', 'a = 3'),
            ('(a > 0)', 'While:3', 'a = 2'),
            ('(a > 1)', 'If:4', ('a = 1', '(a > 1)')),
        ),
    )

  def test_while_nested_break(self):

    def test_fn(a):
      while a > 0:
        while a > 1:
          if a > 2:
            break
          a = 1
        a = 2
      a = 3

    graph, = self._build_cfg(test_fn).values()

    self.assertGraphMatches(graph, (
        (('a', 'a = 2'), '(a > 0)', ('(a > 1)', 'a = 3')),
        (('(a > 0)', 'a = 1'), '(a > 1)', ('(a > 2)', 'a = 2')),
        ('(a > 1)', '(a > 2)', ('break', 'a = 1')),
        ('(a > 2)', 'break', 'a = 2'),
        ('(a > 2)', 'a = 1', '(a > 1)'),
        (('(a > 1)', 'break'), 'a = 2', '(a > 0)'),
        ('(a > 0)', 'a = 3', None),
    ))
    self.assertStatementEdges(
        graph,
        (
            ('a', 'While:2', 'a = 3'),
            ('(a > 0)', 'While:3', 'a = 2'),
            ('(a > 1)', 'If:4', ('a = 1', 'a = 2')),
        ),
    )

  def test_for_straightline(self):

    def test_fn(a):
      for a in range(0, a):
        a = 1
      a = 2

    graph, = self._build_cfg(test_fn).values()

    self.assertGraphMatches(
        graph,
        (
            (('a', 'a = 1'), 'range(0, a)', ('a = 1', 'a = 2')),
            ('range(0, a)', 'a = 1', 'range(0, a)'),
            ('range(0, a)', 'a = 2', None),
        ),
    )
    self.assertStatementEdges(
        graph,
        (('a', 'For:2', 'a = 2'),),
    )

  def test_for_else_straightline(self):

    def test_fn(a):
      for a in range(0, a):
        a = 1
      else:  # pylint:disable=useless-else-on-loop
        a = 2
      a = 3

    graph, = self._build_cfg(test_fn).values()

    self.assertGraphMatches(
        graph,
        (
            (('a', 'a = 1'), 'range(0, a)', ('a = 1', 'a = 2')),
            ('range(0, a)', 'a = 1', 'range(0, a)'),
            ('range(0, a)', 'a = 2', 'a = 3'),
            ('a = 2', 'a = 3', None),
        ),
    )
    self.assertStatementEdges(
        graph,
        (('a', 'For:2', 'a = 3'),),
    )

  def test_for_else_continue(self):

    def test_fn(a):
      for a in range(0, a):
        if a > 1:
          continue
        else:
          a = 0
        a = 1
      else:  # pylint:disable=useless-else-on-loop
        a = 2
      a = 3

    graph, = self._build_cfg(test_fn).values()

    self.assertGraphMatches(
        graph,
        (
            (('a', 'continue', 'a = 1'), 'range(0, a)', ('(a > 1)', 'a = 2')),
            ('range(0, a)', '(a > 1)', ('continue', 'a = 0')),
            ('(a > 1)', 'continue', 'range(0, a)'),
            ('(a > 1)', 'a = 0', 'a = 1'),
            ('a = 0', 'a = 1', 'range(0, a)'),
            ('range(0, a)', 'a = 2', 'a = 3'),
            ('a = 2', 'a = 3', None),
        ),
    )
    self.assertStatementEdges(
        graph,
        (
            ('a', 'For:2', 'a = 3'),
            ('range(0, a)', 'If:3', ('a = 1', 'range(0, a)')),
        ),
    )

  def test_for_else_break(self):

    def test_fn(a):
      for a in range(0, a):
        if a > 1:
          break
        a = 1
      else:
        a = 2
      a = 3

    graph, = self._build_cfg(test_fn).values()

    self.assertGraphMatches(
        graph,
        (
            (('a', 'a = 1'), 'range(0, a)', ('(a > 1)', 'a = 2')),
            ('range(0, a)', '(a > 1)', ('break', 'a = 1')),
            ('(a > 1)', 'break', 'a = 3'),
            ('(a > 1)', 'a = 1', 'range(0, a)'),
            ('range(0, a)', 'a = 2', 'a = 3'),
            (('break', 'a = 2'), 'a = 3', None),
        ),
    )
    self.assertStatementEdges(
        graph,
        (
            ('a', 'For:2', 'a = 3'),
            ('range(0, a)', 'If:3', ('a = 1', 'a = 3')),
        ),
    )

  def test_for_else_return(self):

    def test_fn(a):
      for a in range(0, a):
        if a > 1:
          return
        a = 1
      else:  # pylint:disable=useless-else-on-loop
        a = 2
      a = 3

    graph, = self._build_cfg(test_fn).values()

    self.assertGraphMatches(
        graph,
        (
            (('a', 'a = 1'), 'range(0, a)', ('(a > 1)', 'a = 2')),
            ('range(0, a)', '(a > 1)', ('return', 'a = 1')),
            ('(a > 1)', 'return', None),
            ('(a > 1)', 'a = 1', 'range(0, a)'),
            ('range(0, a)', 'a = 2', 'a = 3'),
            ('a = 2', 'a = 3', None),
        ),
    )
    self.assertStatementEdges(
        graph,
        (
            ('a', 'For:2', 'a = 3'),
            ('range(0, a)', 'If:3', 'a = 1'),
        ),
    )

  def test_for_nested_straightline(self):

    def test_fn(a):
      for a in range(0, a):
        for b in range(1, a):
          b += 1
        a = 2
      a = 3

    graph, = self._build_cfg(test_fn).values()

    self.assertGraphMatches(
        graph,
        (
            (('a', 'a = 2'), 'range(0, a)', ('range(1, a)', 'a = 3')),
            (('range(0, a)', 'b += 1'), 'range(1, a)', ('b += 1', 'a = 2')),
            ('range(1, a)', 'b += 1', 'range(1, a)'),
            ('range(1, a)', 'a = 2', 'range(0, a)'),
            ('range(0, a)', 'a = 3', None),
        ),
    )
    self.assertStatementEdges(
        graph,
        (
            ('a', 'For:2', 'a = 3'),
            ('range(0, a)', 'For:3', 'a = 2'),
        ),
    )

  def test_for_nested_continue(self):

    def test_fn(a):
      for a in range(0, a):
        for b in range(1, a):
          if a > 3:
            continue
          b += 1
        a = 2
      a = 3

    graph, = self._build_cfg(test_fn).values()

    self.assertGraphMatches(
        graph,
        (
            (('a', 'a = 2'), 'range(0, a)', ('range(1, a)', 'a = 3')),
            (('range(0, a)', 'continue', 'b += 1'), 'range(1, a)',
             ('(a > 3)', 'a = 2')),
            ('range(1, a)', '(a > 3)', ('continue', 'b += 1')),
            ('(a > 3)', 'continue', 'range(1, a)'),
            ('(a > 3)', 'b += 1', 'range(1, a)'),
            ('range(1, a)', 'a = 2', 'range(0, a)'),
            ('range(0, a)', 'a = 3', None),
        ),
    )
    self.assertStatementEdges(
        graph,
        (
            ('a', 'For:2', 'a = 3'),
            ('range(0, a)', 'For:3', 'a = 2'),
            ('range(1, a)', 'If:4', ('b += 1', 'range(1, a)')),
        ),
    )

  def test_for_nested_break(self):

    def test_fn(a):
      for a in range(0, a):
        for b in range(1, a):
          if a > 2:
            break
          b += 1
        a = 2
      a = 3

    graph, = self._build_cfg(test_fn).values()

    self.assertGraphMatches(
        graph,
        (
            (('a', 'a = 2'), 'range(0, a)', ('range(1, a)', 'a = 3')),
            (('range(0, a)', 'b += 1'), 'range(1, a)', ('(a > 2)', 'a = 2')),
            ('range(1, a)', '(a > 2)', ('break', 'b += 1')),
            ('(a > 2)', 'break', 'a = 2'),
            ('(a > 2)', 'b += 1', 'range(1, a)'),
            (('range(1, a)', 'break'), 'a = 2', 'range(0, a)'),
            ('range(0, a)', 'a = 3', None),
        ),
    )
    self.assertStatementEdges(
        graph,
        (
            ('a', 'For:2', 'a = 3'),
            ('range(0, a)', 'For:3', 'a = 2'),
            ('range(1, a)', 'If:4', ('b += 1', 'a = 2')),
        ),
    )

  def test_complex(self):

    def test_fn(a):
      b = 0
      while a > 0:
        for b in range(0, a):
          if a > 2:
            break
          if a > 3:
            if a > 4:
              continue
            else:
              max(a)
              break
          b += 1
        else:  # for b in range(0, a):
          return a
        a = 2
      for a in range(1, a):
        return b
      a = 3

    graph, = self._build_cfg(test_fn).values()

    self.assertGraphMatches(
        graph,
        (
            (('b = 0', 'a = 2'), '(a > 0)', ('range(0, a)', 'range(1, a)')),
            (
                ('(a > 0)', 'continue', 'b += 1'),
                'range(0, a)',
                ('(a > 2)', 'return a'),
            ),
            ('range(0, a)', '(a > 2)', ('(a > 3)', 'break')),
            ('(a > 2)', 'break', 'a = 2'),
            ('(a > 2)', '(a > 3)', ('(a > 4)', 'b += 1')),
            ('(a > 3)', '(a > 4)', ('continue', 'max(a)')),
            ('(a > 4)', 'max(a)', 'break'),
            ('max(a)', 'break', 'a = 2'),
            ('(a > 4)', 'continue', 'range(0, a)'),
            ('(a > 3)', 'b += 1', 'range(0, a)'),
            ('range(0, a)', 'return a', None),
            ('break', 'a = 2', '(a > 0)'),
            ('(a > 0)', 'range(1, a)', ('return b', 'a = 3')),
            ('range(1, a)', 'return b', None),
            ('range(1, a)', 'a = 3', None),
        ),
    )
    self.assertStatementEdges(
        graph,
        (
            ('b = 0', 'While:3', 'range(1, a)'),
            ('(a > 0)', 'For:4', 'a = 2'),
            ('range(0, a)', 'If:5', ('(a > 3)', 'a = 2')),
            ('(a > 2)', 'If:7', ('b += 1', 'a = 2', 'range(0, a)')),
            ('(a > 3)', 'If:8', ('a = 2', 'range(0, a)')),
            ('(a > 0)', 'For:17', 'a = 3'),
        ),
    )

  def test_finally_straightline(self):

    def test_fn(a):
      try:
        a += 1
      finally:
        a = 2
      a = 3

    graph, = self._build_cfg(test_fn).values()

    self.assertGraphMatches(
        graph,
        (
            ('a', 'a += 1', 'a = 2'),
            ('a += 1', 'a = 2', 'a = 3'),
            ('a = 2', 'a = 3', None),
        ),
    )

  def test_return_finally(self):

    def test_fn(a):
      try:
        return a
      finally:
        a = 1
      a = 2

    graph, = self._build_cfg(test_fn).values()

    self.assertGraphMatches(
        graph,
        (
            ('a', 'return a', 'a = 1'),
            ('return a', 'a = 1', None),
            (None, 'a = 2', None),
        ),
    )

  def test_break_finally(self):

    def test_fn(a):
      while a > 0:
        try:
          break
        finally:
          a = 1

    graph, = self._build_cfg(test_fn).values()

    self.assertGraphMatches(
        graph,
        (
            ('a', '(a > 0)', 'break'),
            ('(a > 0)', 'break', 'a = 1'),
            ('break', 'a = 1', None),
        ),
    )

  def test_continue_finally(self):

    def test_fn(a):
      while a > 0:
        try:
          continue
        finally:
          a = 1

    graph, = self._build_cfg(test_fn).values()

    self.assertGraphMatches(
        graph,
        (
            (('a', 'a = 1'), '(a > 0)', 'continue'),
            ('(a > 0)', 'continue', 'a = 1'),
            ('continue', 'a = 1', '(a > 0)'),
        ),
    )

  def test_with_straightline(self):

    def test_fn(a):
      with max(a) as b:
        a = 0
        return b

    graph, = self._build_cfg(test_fn).values()

    self.assertGraphMatches(
        graph,
        (
            ('a', 'max(a)', 'a = 0'),
            ('max(a)', 'a = 0', 'return b'),
            ('a = 0', 'return b', None),
        ),
    )


if __name__ == '__main__':
  test.main()
