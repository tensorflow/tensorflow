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
"""Tests for parser module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import textwrap

import gast

from tensorflow.python.autograph.pyct import ast_util
from tensorflow.python.autograph.pyct import errors
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.pyct import pretty_printer
from tensorflow.python.platform import test


class ParserTest(test.TestCase):

  def assertAstMatches(self, actual_node, expected_node_src, expr=True):
    if expr:
      # Ensure multi-line expressions parse.
      expected_node = gast.parse('({})'.format(expected_node_src)).body[0]
      expected_node = expected_node.value
    else:
      expected_node = gast.parse(expected_node_src).body[0]

    msg = 'AST did not match expected:\n{}\nActual:\n{}'.format(
        pretty_printer.fmt(expected_node),
        pretty_printer.fmt(actual_node))
    self.assertTrue(ast_util.matches(actual_node, expected_node), msg)

  def test_parse_entity(self):

    def f(x):
      return x + 1

    node, _ = parser.parse_entity(f, future_features=())
    self.assertEqual('f', node.name)

  def test_parse_lambda(self):

    l = lambda x: x + 1
    expected_node_src = 'lambda x: (x + 1)'

    node, source = parser.parse_entity(l, future_features=())
    self.assertAstMatches(node, source)
    self.assertAstMatches(node, expected_node_src)

  def test_parse_lambda_prefix_cleanup(self):

    lambda_lam = lambda x: x + 1
    expected_node_src = 'lambda x: (x + 1)'

    node, source = parser.parse_entity(lambda_lam, future_features=())
    self.assertAstMatches(node, source)
    self.assertAstMatches(node, expected_node_src)

  def test_parse_lambda_resolution_by_location(self):

    _ = lambda x: x + 1
    l = lambda x: x + 1
    _ = lambda x: x + 1
    expected_node_src = 'lambda x: (x + 1)'

    node, source = parser.parse_entity(l, future_features=())
    self.assertAstMatches(node, source)
    self.assertAstMatches(node, expected_node_src)
    self.assertEqual(source, 'lambda x: x + 1')

  def test_parse_lambda_resolution_by_signature(self):

    l = lambda x: lambda x, y: x + y

    node, source = parser.parse_entity(l, future_features=())
    expected_node_src = 'lambda x: (lambda x, y: (x + y))'
    self.assertAstMatches(node, source)
    self.assertAstMatches(node, expected_node_src)
    self.assertEqual(source, 'lambda x: lambda x, y: x + y')

    node, source = parser.parse_entity(l(0), future_features=())
    expected_node_src = 'lambda x, y: (x + y)'
    self.assertAstMatches(node, source)
    self.assertAstMatches(node, expected_node_src)
    self.assertEqual(source, 'lambda x, y: x + y')

  def test_parse_lambda_resolution_ambiguous(self):

    l = lambda x: lambda x: 2 * x

    expected_exception_text = re.compile(r'found multiple definitions'
                                         r'.+'
                                         r'\(?lambda x: \(?lambda x'
                                         r'.+'
                                         r'\(?lambda x: \(?2', re.DOTALL)

    with self.assertRaisesRegex(
        errors.UnsupportedLanguageElementError,
        expected_exception_text):
      parser.parse_entity(l, future_features=())

    with self.assertRaisesRegex(
        errors.UnsupportedLanguageElementError,
        expected_exception_text):
      parser.parse_entity(l(0), future_features=())

  def assertMatchesWithPotentialGarbage(self, source, expected, garbage):
    # In runtimes which don't track end_col_number, the source contains the
    # entire line, which in turn may have garbage from the surrounding context.
    self.assertIn(source, (expected, expected + garbage))

  def test_parse_lambda_multiline(self):

    l = (
        lambda x: lambda y: x + y  # pylint:disable=g-long-lambda
        - 1)

    node, source = parser.parse_entity(l, future_features=())
    expected_node_src = 'lambda x: (lambda y: ((x + y) - 1))'
    self.assertAstMatches(node, expected_node_src)
    self.assertMatchesWithPotentialGarbage(
        source, ('lambda x: lambda y: x + y  # pylint:disable=g-long-lambda\n'
                 '        - 1'), ')')

    node, source = parser.parse_entity(l(0), future_features=())
    expected_node_src = 'lambda y: ((x + y) - 1)'
    self.assertAstMatches(node, expected_node_src)
    self.assertMatchesWithPotentialGarbage(
        source, ('lambda y: x + y  # pylint:disable=g-long-lambda\n'
                 '        - 1'), ')')

  def test_parse_lambda_in_expression(self):

    l = (
        lambda x: lambda y: x + y + 1,
        lambda x: lambda y: x + y + 2,
        )

    node, source = parser.parse_entity(l[0], future_features=())
    expected_node_src = 'lambda x: (lambda y: ((x + y) + 1))'
    self.assertAstMatches(node, expected_node_src)
    self.assertMatchesWithPotentialGarbage(
        source, 'lambda x: lambda y: x + y + 1', ',')

    node, source = parser.parse_entity(l[0](0), future_features=())
    expected_node_src = 'lambda y: ((x + y) + 1)'
    self.assertAstMatches(node, expected_node_src)
    self.assertMatchesWithPotentialGarbage(
        source, 'lambda y: x + y + 1', ',')

    node, source = parser.parse_entity(l[1], future_features=())
    expected_node_src = 'lambda x: (lambda y: ((x + y) + 2))'
    self.assertAstMatches(node, expected_node_src)
    self.assertMatchesWithPotentialGarbage(source,
                                           'lambda x: lambda y: x + y + 2', ',')

    node, source = parser.parse_entity(l[1](0), future_features=())
    expected_node_src = 'lambda y: ((x + y) + 2)'
    self.assertAstMatches(node, expected_node_src)
    self.assertMatchesWithPotentialGarbage(source, 'lambda y: x + y + 2', ',')

  def test_parse_lambda_complex_body(self):

    l = lambda x: (  # pylint:disable=g-long-lambda
        x.y(
            [],
            x.z,
            (),
            x[0:2],
        ),
        x.u,
        'abc',
        1,
    )

    node, source = parser.parse_entity(l, future_features=())
    expected_node_src = "lambda x: (x.y([], x.z, (), x[0:2]), x.u, 'abc', 1)"
    self.assertAstMatches(node, expected_node_src)

    base_source = ('lambda x: (  # pylint:disable=g-long-lambda\n'
                   '        x.y(\n'
                   '            [],\n'
                   '            x.z,\n'
                   '            (),\n'
                   '            x[0:2],\n'
                   '        ),\n'
                   '        x.u,\n'
                   '        \'abc\',\n'
                   '        1,')
    # The complete source includes the trailing parenthesis. But that is only
    # detected in runtimes which correctly track end_lineno for ASTs.
    self.assertMatchesWithPotentialGarbage(source, base_source, '\n    )')

  def test_parse_lambda_function_call_definition(self):

    def do_parse_and_test(lam, **unused_kwargs):
      node, source = parser.parse_entity(lam, future_features=())
      expected_node_src = 'lambda x: x'
      self.assertAstMatches(node, expected_node_src)
      self.assertMatchesWithPotentialGarbage(
          source, 'lambda x: x', ', named_arg=1)')

    do_parse_and_test(  # Intentional line break
        lambda x: x, named_arg=1)

  def test_parse_entity_print_function(self):

    def f(x):
      print(x)

    node, _ = parser.parse_entity(f, future_features=('print_function',))
    self.assertEqual('f', node.name)

  def test_parse_comments(self):

    def f():
      # unindented comment
      pass

    node, _ = parser.parse_entity(f, future_features=())
    self.assertEqual('f', node.name)

  def test_parse_multiline_strings(self):

    def f():
      print("""
multiline
string""")

    node, _ = parser.parse_entity(f, future_features=())
    self.assertEqual('f', node.name)

  def _eval_code(self, code, name):
    globs = {}
    exec(code, globs)  # pylint:disable=exec-used
    return globs[name]

  def test_dedent_block_basic(self):

    code = """
    def f(x):
      if x > 0:
        return -x
      return x
    """

    f = self._eval_code(parser.dedent_block(code), 'f')
    self.assertEqual(f(1), -1)
    self.assertEqual(f(-1), -1)

  def test_dedent_block_comments_out_of_line(self):

    code = """
  ###
    def f(x):
###
      if x > 0:
  ###
        return -x
          ###
  ###
      return x
      ###
    """

    f = self._eval_code(parser.dedent_block(code), 'f')
    self.assertEqual(f(1), -1)
    self.assertEqual(f(-1), -1)

  def test_dedent_block_multiline_string(self):

    code = """
    def f():
      '''
      Docstring.
      '''
      return '''
  1
    2
      3'''
    """

    f = self._eval_code(parser.dedent_block(code), 'f')
    self.assertEqual(f.__doc__, '\n      Docstring.\n      ')
    self.assertEqual(f(), '\n  1\n    2\n      3')

  def test_dedent_block_multiline_expression(self):

    code = """
    def f():
      return (1,
2,
        3)
    """

    f = self._eval_code(parser.dedent_block(code), 'f')
    self.assertEqual(f(), (1, 2, 3))

  def test_dedent_block_continuation(self):

    code = r"""
    def f():
      a = \
          1
      return a
    """

    f = self._eval_code(parser.dedent_block(code), 'f')
    self.assertEqual(f(), 1)

  def test_dedent_block_continuation_in_string(self):

    code = r"""
    def f():
      a = "a \
  b"
      return a
    """

    f = self._eval_code(parser.dedent_block(code), 'f')
    self.assertEqual(f(), 'a   b')

  def test_parse_expression(self):
    node = parser.parse_expression('a.b')
    self.assertEqual('a', node.value.id)
    self.assertEqual('b', node.attr)

  def test_unparse(self):
    node = gast.If(
        test=gast.Constant(1, kind=None),
        body=[
            gast.Assign(
                targets=[
                    gast.Name(
                        'a',
                        ctx=gast.Store(),
                        annotation=None,
                        type_comment=None)
                ],
                value=gast.Name(
                    'b', ctx=gast.Load(), annotation=None, type_comment=None))
        ],
        orelse=[
            gast.Assign(
                targets=[
                    gast.Name(
                        'a',
                        ctx=gast.Store(),
                        annotation=None,
                        type_comment=None)
                ],
                value=gast.Constant('c', kind=None))
        ])

    source = parser.unparse(node, indentation='  ')
    self.assertEqual(
        textwrap.dedent("""
            # coding=utf-8
            if 1:
                a = b
            else:
                a = 'c'
        """).strip(), source.strip())

  def test_ext_slice_roundtrip(self):
    def ext_slice(n):
      return n[:, :], n[0, :], n[:, 0]

    node, _ = parser.parse_entity(ext_slice, future_features=())
    source = parser.unparse(node)
    self.assertAstMatches(node, source, expr=False)

if __name__ == '__main__':
  test.main()
