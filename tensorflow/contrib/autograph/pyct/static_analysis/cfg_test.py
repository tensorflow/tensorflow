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

import functools

import gast

from tensorflow.contrib.autograph.pyct import anno
from tensorflow.contrib.autograph.pyct import context
from tensorflow.contrib.autograph.pyct import parser
from tensorflow.contrib.autograph.pyct import qual_names
from tensorflow.contrib.autograph.pyct.static_analysis import cfg
from tensorflow.python.platform import test


class CFGTest(test.TestCase):

  def _parse_and_analyze(self, test_fn, namespace, arg_types=None):
    arg_types = arg_types or {}
    node, source = parser.parse_entity(test_fn)
    ctx = context.EntityContext(
        namer=None,
        source_code=source,
        source_file=None,
        namespace=namespace,
        arg_values=None,
        arg_types=arg_types,
        owner_type=None,
        recursive=True)
    node = qual_names.resolve(node)
    return node, ctx

  def _check_anno_matches(self, node, anno_name, var_names):
    if isinstance(var_names, str):
      var_names = (var_names,)
    qual_vars = set()
    for var_name in var_names:
      if isinstance(var_name, str):
        if '[' in var_name or ']' in var_name:
          raise ValueError('Annotation matching not supported with subscript.')
        if '.' not in var_name:
          qual_vars.add(qual_names.QN(var_name))
        else:
          attrs = var_name.split('.')
          this_qn = functools.reduce(qual_names.QN, attrs[1:],
                                     qual_names.QN(attrs[0]))
          qual_vars.add(this_qn)
    self.assertEqual(anno.getanno(node, anno_name), qual_vars)

  def test_reaching(self):

    def f(x):
      print(x)
      while True:
        x = x
        x = x
      return x

    node, ctx = self._parse_and_analyze(f, {})
    cfg.run_analyses(node, cfg.ReachingDefinitions(ctx))
    body = node.body[0].body
    # Only the argument reaches the expression
    def_in = anno.getanno(body[0], 'definitions_in')
    # One element, x, from arguments
    self.assertEqual(set(type(d[1]) for d in def_in), set((gast.arguments,)))

    while_body = body[1].body
    def_in = anno.getanno(while_body[0], 'definitions_in')
    # One definition, two possible sources.
    # - One from an assignment (if the loop is entered)
    # - The other from the arguments (if loop is not entered)
    self.assertEqual(
        set(type(d[1]) for d in def_in), set((gast.arguments, gast.Assign)))

    def_in = anno.getanno(while_body[1], 'definitions_in')
    # If we've reached this line, the only reaching definition of x is the
    # Assign node in previous line
    self.assertEqual(set(type(d[1]) for d in def_in), set((gast.Assign,)))

    def_in = anno.getanno(body[2], 'definitions_in')
    # Same situation as while_body[0]
    self.assertEqual(
        set(type(d[1]) for d in def_in), set((gast.arguments, gast.Assign)))

  def test_defined(self):

    def f(x):
      if x:
        y = 2  # pylint: disable=unused-variable
      return x

    node, ctx = self._parse_and_analyze(f, {})
    cfg.run_analyses(node, cfg.Defined(ctx))
    body = node.body[0].body
    # only x is for sure defined at the end
    self._check_anno_matches(body[1], 'defined_in', 'x')
    # at the end of the if body both x and y are defined
    if_body = body[0].body
    self._check_anno_matches(if_body[0], 'defined_out', ('x', 'y'))

  # TODO(alexbw): b/73926938 split this test up
  def test_live(self):

    def get_live_annotated_fnbody(f):
      node, ctx = self._parse_and_analyze(f, {})
      cfg.run_analyses(node, cfg.Liveness(ctx))
      body = node.body[0].body
      return body

    def f1(x):
      a = g(x)  # pylint: disable=undefined-variable
      b = h(a)  # pylint: disable=undefined-variable, unused-variable
      return x

    def f2(x, a):  # pylint: disable=unused-argument
      if a > 0:  # x should not be live
        x = 0
      if a > 1:
        x = 1
      else:
        x = 2

    def f3(x, a):
      if a > 0:  # x and a should be live
        x = 0
      if a > 1:  # x and a should be live_in
        x = 1
      return x  # x should be live

    def f4(x, a):
      if a > 0:  # x should be live
        x = 0
      x += 1

    def f5(x, a):
      if a > 0:  # x.y should be live
        x.y = 0
      return x.y

    def f6(x):
      return x  # should this cause x.* to be live?

    def f7(x, n):
      for i in range(n):
        x += i
      return x

    def f8(x, f):
      with f:
        x += 1

    body = get_live_annotated_fnbody(f1)
    self._check_anno_matches(body[1], 'live_in', ('a', 'h', 'x'))
    self._check_anno_matches(body[2], 'live_in', ('x'))
    self._check_anno_matches(body[0], 'live_in', ('g', 'h', 'x'))
    self._check_anno_matches(body[2], 'live_out', ())

    body = get_live_annotated_fnbody(f2)
    self._check_anno_matches(body[0], 'live_in', ('a'))
    self._check_anno_matches(body[1], 'live_in', ('a'))

    body = get_live_annotated_fnbody(f3)
    self._check_anno_matches(body[0], 'live_in', ('a', 'x'))
    self._check_anno_matches(body[1], 'live_in', ('a', 'x'))
    self._check_anno_matches(body[2], 'live_in', ('x'))

    body = get_live_annotated_fnbody(f4)
    self._check_anno_matches(body[0], 'live_in', ('x', 'a'))
    self._check_anno_matches(body[1], 'live_in', ('x'))

    body = get_live_annotated_fnbody(f5)
    self._check_anno_matches(body[0], 'live_in', ('x', 'x.y', 'a'))

    body = get_live_annotated_fnbody(f6)
    self._check_anno_matches(body[0], 'live_in', ('x'))

    body = get_live_annotated_fnbody(f7)
    self._check_anno_matches(body[0], 'live_in', ('x', 'n', 'range'))
    self._check_anno_matches(body[1], 'live_in', ('x'))

    body = get_live_annotated_fnbody(f8)
    self._check_anno_matches(body[0], 'live_in', ('f', 'x'))

  def test_node_equality(self):
    node_a = gast.parse('y = x').body[0]
    node_b = gast.parse('y = x').body[0]
    self.assertNotEqual(node_a, node_b)

  def test_nested_functions_defined(self):

    def f(x):
      y = x * 2

      def g(z):
        return z + y

      return g(x)

    node, ctx = self._parse_and_analyze(f, {})
    cfg.run_analyses(node, cfg.Defined(ctx))

    body = node.body[0].body
    self.assertEqual(
        anno.getanno(body[2], 'defined_in'),
        frozenset(map(qual_names.QN, ('g', 'x', 'y'))))

    # TODO(alexbw): CFG analysis doesn't currently cross FunctionDef boundaries.
    # NOTE: 'z' is easy to find, but 'y' is  not identified as
    # defined, because CFG analysis is applied with each function separately.
    # fndef_body = body[1].body
    # self.assertEqual(
    #     anno.getanno(fndef_body[0], 'defined_in'),
    #     frozenset(map(qual_names.QN, ('z', 'y'))))

  def test_nested_functions_dont_leak_definitions(self):

    def f(x):
      print(x)

      def g():
        y = 2
        return y

      return g()  # y is not defined here

    node, ctx = self._parse_and_analyze(f, {})
    cfg.run_analyses(node, cfg.Defined(ctx))
    body = node.body[0].body
    self.assertEqual(
        anno.getanno(body[2], 'defined_in'),
        frozenset(map(qual_names.QN, ('x', 'g'))))


if __name__ == '__main__':
  test.main()
