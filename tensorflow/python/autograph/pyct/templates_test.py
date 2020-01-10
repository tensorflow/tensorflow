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
"""Tests for templates module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import imp

from absl.testing import parameterized
import gast

from tensorflow.python.autograph.pyct import loader
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.pyct import qual_names as qn
from tensorflow.python.autograph.pyct import templates
from tensorflow.python.platform import test


class _CtxClearer(gast.NodeTransformer):

  def visit(self, node):
    super(_CtxClearer, self).visit(node)
    if hasattr(node, 'ctx'):
      node.ctx = None
    return node


def _parse_with_unset_ctx(expr_source):
  ast_node = parser.parse_expression(expr_source)
  _CtxClearer().visit(ast_node)
  return ast_node


class _CtxChecker(gast.NodeTransformer):

  def __init__(self, test_instance, expected_ctx):
    self.at_top_level = True
    self.test_instance = test_instance
    self.expected_ctx = expected_ctx

  def visit(self, node):
    if hasattr(node, 'ctx'):
      self.test_instance.assertIsInstance(node.ctx, self.expected_ctx)
    if self.at_top_level:
      self.at_top_level = False
      self.expected_ctx = gast.Load
    return super(_CtxChecker, self).visit(node)


class TemplatesTest(test.TestCase, parameterized.TestCase):

  def assertExpectedCtxSet(self, node, ctx):
    """Assert that node has ctx=ctx at top and ctx=gast.Load everywhere else."""
    checker = _CtxChecker(self, ctx)
    checker.visit(node)

  def test_replace_tuple(self):
    template = """
      def test_fn(a, c):
        return b,
    """

    node = templates.replace(template, b=('a', 'c'))[0]
    result, _, _ = loader.load_ast(node)

    self.assertEqual((2, 3), result.test_fn(2, 3))

  def test_replace_variable(self):
    template = """
      def test_fn(a):
        a += 1
        a = 2 * a + 1
        return b
    """

    node = templates.replace(template, a='b')[0]
    result, _, _ = loader.load_ast(node)
    self.assertEqual(7, result.test_fn(2))

  def test_replace_function_name(self):
    template = """
      def fname(a):
        a += 1
        a = 2 * a + 1
        return a
    """

    node = templates.replace(template, fname='test_fn')[0]
    result, _, _ = loader.load_ast(node)
    self.assertEqual(7, result.test_fn(2))

  def test_replace_code_block(self):
    template = """
      def test_fn(a):
        block
        return a
    """

    node = templates.replace(
        template,
        block=[
            gast.Assign([
                gast.Name('a', None, None)
            ], gast.BinOp(gast.Name('a', None, None), gast.Add(), gast.Num(1))),
        ] * 2)[0]
    result, _, _ = loader.load_ast(node)
    self.assertEqual(3, result.test_fn(1))

  def test_replace_attribute(self):
    template = """
      def test_fn(a):
        return a.foo
    """

    node = templates.replace(template, foo='b')[0]
    result, _, _ = loader.load_ast(node)
    mod = imp.new_module('test')
    mod.b = 3
    self.assertEqual(3, result.test_fn(mod))

    with self.assertRaises(ValueError):
      templates.replace(template, foo=1)

  def test_replace_attribute_context(self):
    template = """
      def test_fn(foo):
        foo = 0
    """

    node = templates.replace(
        template,
        foo=parser.parse_expression('a.b.c'))[0]
    self.assertIsInstance(node.body[0].targets[0].ctx, gast.Store)
    self.assertIsInstance(node.body[0].targets[0].value.ctx, gast.Load)
    self.assertIsInstance(node.body[0].targets[0].value.value.ctx, gast.Load)

  def test_replace_list_context(self):
    template = """
      def test_fn(foo):
        foo = 0
    """

    node = templates.replace(template, foo=parser.parse_expression('[a, b]'))[0]
    self.assertIsInstance(node.body[0].targets[0].ctx, gast.Store)
    self.assertIsInstance(node.body[0].targets[0].elts[0].ctx, gast.Store)
    self.assertIsInstance(node.body[0].targets[0].elts[1].ctx, gast.Store)

  def test_replace_tuple_context(self):
    template = """
      def test_fn(foo):
        foo = 0
    """

    node = templates.replace(template, foo=parser.parse_expression('(a, b)'))[0]
    self.assertIsInstance(node.body[0].targets[0].ctx, gast.Store)
    self.assertIsInstance(node.body[0].targets[0].elts[0].ctx, gast.Store)
    self.assertIsInstance(node.body[0].targets[0].elts[1].ctx, gast.Store)

  def test_replace_expression_context(self):
    template = """
      def test_fn():
        foo
    """

    node = templates.replace(
        template, foo=parser.parse_expression('a + 2 * b / -c'))[0]
    self.assertIsInstance(node.body[0].left.ctx, gast.Load)
    self.assertIsInstance(node.body[0].right.left.right.ctx, gast.Load)

  def test_replace_complex_context(self):
    template = """
      def test_fn():
        foo = 0
    """

    node = templates.replace(
        template, foo=parser.parse_expression('bar(([a, b],)).baz'))[0]
    self.assertIsInstance(node.body[0].targets[0].ctx, gast.Store)
    function_call_arg = node.body[0].targets[0].value.args[0]
    self.assertIsInstance(function_call_arg.elts[0].ctx, gast.Load)
    self.assertIsInstance(function_call_arg.elts[0].elts[0].ctx, gast.Load)
    self.assertIsInstance(function_call_arg.elts[0].elts[1].ctx, gast.Load)

  def test_replace_index(self):
    template = """
      def test_fn():
        foo = 0
    """

    node = templates.replace(
        template, foo=parser.parse_expression('foo(a[b]).bar'))[0]
    function_call_arg = node.body[0].targets[0].value.args[0]
    self.assertIsInstance(function_call_arg.ctx, gast.Load)
    self.assertIsInstance(function_call_arg.slice.value.ctx, gast.Load)

  def test_replace_call_keyword(self):
    template = """
      def test_fn():
        def f(a, d, f):
          return a + d + f
        return f(1, kws=None)
    """

    source = parser.parse_expression('f(d=3, f=5)')
    node = templates.replace(template, kws=source.keywords)[0]
    result, _, _ = loader.load_ast(node)
    self.assertEqual(9, result.test_fn())

    with self.assertRaises(ValueError):
      templates.replace(template, kws=[])
      templates.replace(template, kws=1)

  def test_replace_name_with_call(self):
    template = """
      def test_fn():
        b = 5
        def g(a):
          return 3 * a
        def f():
          return g
        return foo
    """

    source = parser.parse_expression('f()(b)')
    node = templates.replace(template, foo=source)[0]
    result, _, _ = loader.load_ast(node)
    self.assertEqual(15, result.test_fn())

  def test_replace_name_with_dict(self):
    template = """
      def test_fn():
        return foo['bar']
    """

    source = parser.parse_expression('{\'bar\': 3}')
    node = templates.replace(template, foo=source)[0]
    result, _, _ = loader.load_ast(node)
    self.assertEqual(3, result.test_fn())

  def test_replace_as_expression(self):
    template = """
      foo(a)
    """

    node = templates.replace_as_expression(template, foo='bar', a='baz')
    self.assertIsInstance(node, gast.Call)
    self.assertEqual(node.func.id, 'bar')
    self.assertEqual(node.args[0].id, 'baz')

  def test_replace_as_expression_restrictions(self):
    template = """
      foo(a)
      bar(b)
    """
    with self.assertRaises(ValueError):
      templates.replace_as_expression(template)

  def test_function_call_in_list(self):
    template = """
        foo(bar)
    """
    source = parser.parse_expression('[a(b(1))]')
    templates.replace_as_expression(template, bar=source)

  def test_star_comprehension_in_function_call(self):
    template = """
      a = foo(func, args)
    """
    source = parser.parse_expression('bar(*[i for i in range(j)])')
    node = templates.replace(template, func=source.func, args=source.args)
    arg_node = node[0].value.args[1].value
    self.assertIsInstance(arg_node.generators[0].target.ctx, gast.Store)
    self.assertIsInstance(arg_node.elt.ctx, gast.Load)

  def test_lambda_in_function_call(self):
    template = """
      a = foo(arg)
    """
    source = parser.parse_expression('[lambda i: i]')
    node = templates.replace(template, arg=source)
    lambda_arg = node[0].value.args[0].elts[0]
    self.assertIsInstance(lambda_arg.args.args[0].ctx, gast.Param)
    self.assertIsInstance(lambda_arg.body.ctx, gast.Load)

  def test_replace_name_with_subscript(self):
    template = """
        foo = bar
    """
    replacement = qn.QN(qn.QN('dictionary'), subscript=qn.QN('key'))

    node = templates.replace(template, foo=replacement)[0].targets[0]
    self.assertIsInstance(node.ctx, gast.Store)
    self.assertIsInstance(node.value.ctx, gast.Load)

  @parameterized.named_parameters([
      ('mixed_attr_subscript', 'a.b["c"]'),
      ('mixed_subscript_attr', 'a[b.c]'),
      ('nested_subscript', 'a[b[c]]'),
      ('repeated_subscript', 'a[b][c]'),
  ])
  def test_replace_name_mixed_attr_subscript(self, expression_source):
    template = 'foo = bar'
    replacement = _parse_with_unset_ctx(expression_source)

    target_node = templates.replace(template, foo=replacement)[0].targets[0]
    self.assertExpectedCtxSet(target_node, gast.Store)

    value_node = templates.replace(template, bar=replacement)[0].value
    self.assertExpectedCtxSet(value_node, gast.Load)

if __name__ == '__main__':
  test.main()
