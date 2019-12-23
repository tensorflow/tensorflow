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

import textwrap

import gast

from tensorflow.python.autograph.pyct import parser
from tensorflow.python.platform import test


class ParserTest(test.TestCase):

  def test_parse_entity(self):

    def f(x):
      return x + 1

    node, _ = parser.parse_entity(f, future_features=())
    self.assertEqual('f', node.name)

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

  def test_parse_expression(self):
    node = parser.parse_expression('a.b')
    self.assertEqual('a', node.value.id)
    self.assertEqual('b', node.attr)

  def test_unparse(self):
    node = gast.If(
        test=gast.Num(1),
        body=[
            gast.Assign(
                targets=[gast.Name('a', gast.Store(), None)],
                value=gast.Name('b', gast.Load(), None))
        ],
        orelse=[
            gast.Assign(
                targets=[gast.Name('a', gast.Store(), None)],
                value=gast.Str('c'))
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


if __name__ == '__main__':
  test.main()
