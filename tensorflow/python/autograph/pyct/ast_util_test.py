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
"""Tests for ast_util module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ast
import collections
import textwrap

import gast

from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import ast_util
from tensorflow.python.autograph.pyct import compiler
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.pyct import qual_names
from tensorflow.python.platform import test


class AstUtilTest(test.TestCase):

  def setUp(self):
    super(AstUtilTest, self).setUp()
    self._invocation_counts = collections.defaultdict(lambda: 0)

  def test_rename_symbols_basic(self):
    node = parser.parse_str('a + b')
    node = qual_names.resolve(node)

    node = ast_util.rename_symbols(
        node, {qual_names.QN('a'): qual_names.QN('renamed_a')})

    self.assertIsInstance(node.value.left.id, str)
    source = compiler.ast_to_source(node)
    self.assertEqual(source.strip(), 'renamed_a + b')

  def test_rename_symbols_attributes(self):
    node = parser.parse_str('b.c = b.c.d')
    node = qual_names.resolve(node)

    node = ast_util.rename_symbols(
        node, {qual_names.from_str('b.c'): qual_names.QN('renamed_b_c')})

    source = compiler.ast_to_source(node)
    self.assertEqual(source.strip(), 'renamed_b_c = renamed_b_c.d')

  def test_rename_symbols_annotations(self):
    node = parser.parse_str('a[i]')
    node = qual_names.resolve(node)
    anno.setanno(node, 'foo', 'bar')
    orig_anno = anno.getanno(node, 'foo')

    node = ast_util.rename_symbols(node,
                                   {qual_names.QN('a'): qual_names.QN('b')})

    self.assertIs(anno.getanno(node, 'foo'), orig_anno)

  def test_copy_clean(self):
    node = parser.parse_str(
        textwrap.dedent("""
      def f(a):
        return a + 1
    """))
    setattr(node, '__foo', 'bar')
    new_node = ast_util.copy_clean(node)
    self.assertIsNot(new_node, node)
    self.assertFalse(hasattr(new_node, '__foo'))

  def test_copy_clean_preserves_annotations(self):
    node = parser.parse_str(
        textwrap.dedent("""
      def f(a):
        return a + 1
    """))
    anno.setanno(node, 'foo', 'bar')
    anno.setanno(node, 'baz', 1)
    new_node = ast_util.copy_clean(node, preserve_annos={'foo'})
    self.assertEqual(anno.getanno(new_node, 'foo'), 'bar')
    self.assertFalse(anno.hasanno(new_node, 'baz'))

  def test_keywords_to_dict(self):
    keywords = parser.parse_expression('f(a=b, c=1, d=\'e\')').keywords
    d = ast_util.keywords_to_dict(keywords)
    # Make sure we generate a usable dict node by attaching it to a variable and
    # compiling everything.
    node = parser.parse_str('def f(b): pass')
    node.body.append(ast.Return(d))
    result, _, _ = compiler.ast_to_object(node)
    self.assertDictEqual(result.f(3), {'a': 3, 'c': 1, 'd': 'e'})

  def assertMatch(self, target_str, pattern_str):
    node = parser.parse_expression(target_str)
    pattern = parser.parse_expression(pattern_str)
    self.assertTrue(ast_util.matches(node, pattern))

  def assertNoMatch(self, target_str, pattern_str):
    node = parser.parse_expression(target_str)
    pattern = parser.parse_expression(pattern_str)
    self.assertFalse(ast_util.matches(node, pattern))

  def test_matches_symbols(self):
    self.assertMatch('foo', '_')
    self.assertNoMatch('foo()', '_')
    self.assertMatch('foo + bar', 'foo + _')
    self.assertNoMatch('bar + bar', 'foo + _')
    self.assertNoMatch('foo - bar', 'foo + _')

  def test_matches_function_args(self):
    self.assertMatch('super(Foo, self).__init__(arg1, arg2)',
                     'super(_).__init__(_)')
    self.assertMatch('super().__init__()', 'super(_).__init__(_)')
    self.assertNoMatch('super(Foo, self).bar(arg1, arg2)',
                       'super(_).__init__(_)')
    self.assertMatch('super(Foo, self).__init__()', 'super(Foo, _).__init__(_)')
    self.assertNoMatch('super(Foo, self).__init__()',
                       'super(Bar, _).__init__(_)')

  def _mock_apply_fn(self, target, source):
    target = compiler.ast_to_source(target)
    source = compiler.ast_to_source(source)
    self._invocation_counts[(target.strip(), source.strip())] += 1

  def test_apply_to_single_assignments_dynamic_unpack(self):
    node = parser.parse_str('a, b, c = d')
    ast_util.apply_to_single_assignments(node.targets, node.value,
                                         self._mock_apply_fn)
    self.assertDictEqual(self._invocation_counts, {
        ('a', 'd[0]'): 1,
        ('b', 'd[1]'): 1,
        ('c', 'd[2]'): 1,
    })

  def test_apply_to_single_assignments_static_unpack(self):
    node = parser.parse_str('a, b, c = d, e, f')
    ast_util.apply_to_single_assignments(node.targets, node.value,
                                         self._mock_apply_fn)
    self.assertDictEqual(self._invocation_counts, {
        ('a', 'd'): 1,
        ('b', 'e'): 1,
        ('c', 'f'): 1,
    })

  def test_parallel_walk(self):
    src = """
      def f(a):
        return a + 1
    """
    node = parser.parse_str(textwrap.dedent(src))
    for child_a, child_b in ast_util.parallel_walk(node, node):
      self.assertEqual(child_a, child_b)

  def test_parallel_walk_string_leaves(self):
    src = """
      def f(a):
        global g
    """
    node = parser.parse_str(textwrap.dedent(src))
    for child_a, child_b in ast_util.parallel_walk(node, node):
      self.assertEqual(child_a, child_b)

  def test_parallel_walk_inconsistent_trees(self):
    node_1 = parser.parse_str(
        textwrap.dedent("""
      def f(a):
        return a + 1
    """))
    node_2 = parser.parse_str(
        textwrap.dedent("""
      def f(a):
        return a + (a * 2)
    """))
    node_3 = parser.parse_str(
        textwrap.dedent("""
      def f(a):
        return a + 2
    """))
    with self.assertRaises(ValueError):
      for _ in ast_util.parallel_walk(node_1, node_2):
        pass
    # There is not particular reason to reject trees that differ only in the
    # value of a constant.
    # TODO(mdan): This should probably be allowed.
    with self.assertRaises(ValueError):
      for _ in ast_util.parallel_walk(node_1, node_3):
        pass

  def assertLambdaNodes(self, matching_nodes, expected_bodies):
    self.assertEqual(len(matching_nodes), len(expected_bodies))
    for node in matching_nodes:
      self.assertIsInstance(node, gast.Lambda)
      self.assertIn(compiler.ast_to_source(node.body).strip(), expected_bodies)

  def test_find_matching_definitions_lambda(self):
    node = parser.parse_str(
        textwrap.dedent("""
      f = lambda x: 1
    """))
    f = lambda x: x
    nodes = ast_util.find_matching_definitions(node, f)
    self.assertLambdaNodes(nodes, ('(1)',))

  def test_find_matching_definitions_lambda_multiple_matches(self):
    node = parser.parse_str(
        textwrap.dedent("""
      f = lambda x: 1, lambda x: 2
    """))
    f = lambda x: x
    nodes = ast_util.find_matching_definitions(node, f)
    self.assertLambdaNodes(nodes, ('(1)', '(2)'))

  def test_find_matching_definitions_lambda_uses_arg_names(self):
    node = parser.parse_str(
        textwrap.dedent("""
      f = lambda x: 1, lambda y: 2
    """))
    f = lambda x: x
    nodes = ast_util.find_matching_definitions(node, f)
    self.assertLambdaNodes(nodes, ('(1)',))

    f = lambda y: y
    nodes = ast_util.find_matching_definitions(node, f)
    self.assertLambdaNodes(nodes, ('(2)',))


if __name__ == '__main__':
  test.main()
