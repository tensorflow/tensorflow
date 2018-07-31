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
"""Tests for activity module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gast

from tensorflow.contrib.autograph.pyct import anno
from tensorflow.contrib.autograph.pyct import parser
from tensorflow.contrib.autograph.pyct import qual_names
from tensorflow.contrib.autograph.pyct import transformer
from tensorflow.contrib.autograph.pyct.qual_names import QN
from tensorflow.contrib.autograph.pyct.static_analysis import activity
from tensorflow.contrib.autograph.pyct.static_analysis.annos import NodeAnno
from tensorflow.python.platform import test


class ScopeTest(test.TestCase):

  def test_basic(self):
    scope = activity.Scope(None)
    self.assertFalse(scope.has(QN('foo')))

    scope.mark_read(QN('foo'))
    self.assertFalse(scope.has(QN('foo')))

    scope.mark_write(QN('foo'))
    self.assertTrue(scope.has(QN('foo')))

    scope.mark_read(QN('bar'))
    self.assertFalse(scope.has(QN('bar')))

  def test_copy_from(self):
    scope = activity.Scope(None)
    scope.mark_write(QN('foo'))

    other = activity.Scope(None)
    other.copy_from(scope)

    self.assertTrue(QN('foo') in other.modified)

    scope.mark_write(QN('bar'))
    scope.copy_from(other)

    self.assertFalse(QN('bar') in scope.modified)

    scope.mark_write(QN('bar'))
    scope.merge_from(other)

    self.assertTrue(QN('bar') in scope.modified)
    self.assertFalse(QN('bar') in other.modified)

  def test_copy_of(self):
    scope = activity.Scope(None)
    scope.mark_read(QN('foo'))

    self.assertTrue(QN('foo') in activity.Scope.copy_of(scope).used)

    child_scope = activity.Scope(scope)
    child_scope.mark_read(QN('bar'))

    self.assertTrue(QN('bar') in activity.Scope.copy_of(child_scope).used)

  def test_nesting(self):
    scope = activity.Scope(None)
    scope.mark_write(QN('foo'))
    scope.mark_read(QN('bar'))

    child = activity.Scope(scope)
    self.assertTrue(child.has(QN('foo')))
    self.assertTrue(scope.has(QN('foo')))

    child.mark_write(QN('bar'))
    self.assertTrue(child.has(QN('bar')))
    self.assertFalse(scope.has(QN('bar')))

  def test_referenced(self):
    scope = activity.Scope(None)
    scope.mark_read(QN('a'))

    child = activity.Scope(scope)
    child.mark_read(QN('b'))

    child2 = activity.Scope(child, isolated=False)
    child2.mark_read(QN('c'))

    self.assertTrue(QN('c') in child2.referenced)
    self.assertTrue(QN('b') in child2.referenced)
    self.assertFalse(QN('a') in child2.referenced)

    self.assertTrue(QN('c') in child.referenced)
    self.assertTrue(QN('b') in child.referenced)
    self.assertFalse(QN('a') in child.referenced)


class ActivityAnalyzerTest(test.TestCase):

  def _parse_and_analyze(self, test_fn):
    node, source = parser.parse_entity(test_fn)
    entity_info = transformer.EntityInfo(
        source_code=source,
        source_file=None,
        namespace={},
        arg_values=None,
        arg_types=None,
        owner_type=None)
    node = qual_names.resolve(node)
    node = activity.resolve(node, entity_info)
    return node, entity_info

  def test_local_markers(self):

    def test_fn(a):  # pylint:disable=unused-argument
      b = c  # pylint:disable=undefined-variable
      while b > 0:
        b -= 1
      return b

    node, _ = self._parse_and_analyze(test_fn)
    self.assertFalse(
        anno.getanno(node.body[0].body[0].value,
                     NodeAnno.IS_LOCAL))  # c in b = c
    self.assertTrue(
        anno.getanno(node.body[0].body[1].test.left,
                     NodeAnno.IS_LOCAL))  # b in b > 0
    self.assertTrue(
        anno.getanno(node.body[0].body[2].value,
                     NodeAnno.IS_LOCAL))  # b in return b

  def assertSymbolSetsAre(self, expected, actual, name):
    expected = set(expected)
    actual = set(str(s) for s in actual)
    self.assertSetEqual(
        expected, actual, 'for symbol set: %s\n'
        '  Expected: %s\n'
        '  Got:      %s\n'
        '  Missing:  %s\n'
        '  Extra:    %s\n' % (name.upper(), expected, actual,
                              expected - actual, actual - expected))

  def assertScopeIsRmc(self, scope, used, modified, created):
    """Assert the scope contains specific used, modified & created variables."""
    self.assertSymbolSetsAre(used, scope.used, 'read')
    self.assertSymbolSetsAre(modified, scope.modified, 'modified')
    # Created is deprecated, we're no longer verifying it.
    # self.assertSymbolSetsAre(created, scope.created, 'created')

  def test_print_statement(self):

    def test_fn(a):
      b = 0
      c = 1
      print(a, b)
      return c

    node, _ = self._parse_and_analyze(test_fn)
    print_node = node.body[0].body[2]
    if isinstance(print_node, gast.Print):
      # Python 2
      print_args_scope = anno.getanno(print_node, NodeAnno.ARGS_SCOPE)
    else:
      # Python 3
      assert isinstance(print_node, gast.Expr)
      # The call node should be the one being annotated.
      print_node = print_node.value
      print_args_scope = anno.getanno(print_node, NodeAnno.ARGS_SCOPE)
    # We basically need to detect which variables are captured by the call
    # arguments.
    self.assertScopeIsRmc(print_args_scope, ('a', 'b'), (), ())

  def test_call_args(self):

    def test_fn(a):
      b = 0
      c = 1
      foo(a, b)  # pylint:disable=undefined-variable
      return c

    node, _ = self._parse_and_analyze(test_fn)
    call_node = node.body[0].body[2].value
    # We basically need to detect which variables are captured by the call
    # arguments.
    self.assertScopeIsRmc(
        anno.getanno(call_node, NodeAnno.ARGS_SCOPE), ('a', 'b'), (), ())

  def test_call_args_attributes(self):

    def foo(*_):
      pass

    def test_fn(a):
      a.c = 0
      foo(a.b, a.c)
      return a.d

    node, _ = self._parse_and_analyze(test_fn)
    call_node = node.body[0].body[1].value
    self.assertScopeIsRmc(
        anno.getanno(call_node, NodeAnno.ARGS_SCOPE),
        ('a', 'a.b', 'a.c'),
        (),
        (),
    )

  def test_call_args_subscripts(self):

    def foo(*_):
      pass

    def test_fn(a):
      b = 1
      c = 2
      foo(a[0], a[b])
      return a[c]

    node, _ = self._parse_and_analyze(test_fn)
    call_node = node.body[0].body[2].value
    self.assertScopeIsRmc(
        anno.getanno(call_node, NodeAnno.ARGS_SCOPE),
        ('a', 'a[0]', 'a[b]', 'b'),
        (),
        (),
    )

  def test_while(self):

    def test_fn(a):
      b = a
      while b > 0:
        c = b
        b -= 1
      return b, c

    node, _ = self._parse_and_analyze(test_fn)
    while_node = node.body[0].body[1]
    self.assertScopeIsRmc(
        anno.getanno(while_node, NodeAnno.BODY_SCOPE), ('b',), ('b', 'c'),
        ('c',))
    self.assertScopeIsRmc(
        anno.getanno(while_node, NodeAnno.BODY_SCOPE).parent, ('a', 'b', 'c'),
        ('b', 'c'), ('a', 'b', 'c'))
    self.assertScopeIsRmc(
        anno.getanno(while_node, NodeAnno.COND_SCOPE), ('b',), (), ())

  def test_for(self):

    def test_fn(a):
      b = a
      for _ in a:
        c = b
        b -= 1
      return b, c

    node, _ = self._parse_and_analyze(test_fn)
    for_node = node.body[0].body[1]
    self.assertScopeIsRmc(
        anno.getanno(for_node, NodeAnno.BODY_SCOPE), ('b',), ('b', 'c'), ('c',))
    self.assertScopeIsRmc(
        anno.getanno(for_node, NodeAnno.BODY_SCOPE).parent, ('a', 'b', 'c'),
        ('b', 'c', '_'), ('a', 'b', 'c', '_'))

  def test_if(self):

    def test_fn(x):
      if x > 0:
        x = -x
        y = 2 * x
        z = -y
      else:
        x = 2 * x
        y = -x
        u = -y
      return z, u

    node, _ = self._parse_and_analyze(test_fn)
    if_node = node.body[0].body[0]
    self.assertScopeIsRmc(
        anno.getanno(if_node, NodeAnno.BODY_SCOPE), ('x', 'y'), ('x', 'y', 'z'),
        ('y', 'z'))
    # TODO(mdan): Double check: is it ok to not mark a local symbol as not read?
    self.assertScopeIsRmc(
        anno.getanno(if_node, NodeAnno.BODY_SCOPE).parent, ('x', 'z', 'u'),
        ('x', 'y', 'z', 'u'), ('x', 'y', 'z', 'u'))
    self.assertScopeIsRmc(
        anno.getanno(if_node, NodeAnno.ORELSE_SCOPE), ('x', 'y'),
        ('x', 'y', 'u'), ('y', 'u'))
    self.assertScopeIsRmc(
        anno.getanno(if_node, NodeAnno.ORELSE_SCOPE).parent, ('x', 'z', 'u'),
        ('x', 'y', 'z', 'u'), ('x', 'y', 'z', 'u'))

  def test_if_attributes(self):

    def test_fn(a):
      if a > 0:
        a.b = -a.c
        d = 2 * a
      else:
        a.b = a.c
        d = 1
      return d

    node, _ = self._parse_and_analyze(test_fn)
    if_node = node.body[0].body[0]
    self.assertScopeIsRmc(
        anno.getanno(if_node, NodeAnno.BODY_SCOPE),
        ('a', 'a.c'),
        ('a.b', 'd'),
        ('d',),
    )
    self.assertScopeIsRmc(
        anno.getanno(if_node, NodeAnno.ORELSE_SCOPE),
        ('a', 'a.c'),
        ('a.b', 'd'),
        ('d',),
    )
    self.assertScopeIsRmc(
        anno.getanno(if_node, NodeAnno.BODY_SCOPE).parent,
        ('a', 'a.c', 'd'),
        ('a.b', 'd'),
        ('a', 'd'),
    )

  def test_if_subscripts(self):

    def test_fn(a, b, c, e):
      if a > 0:
        a[b] = -a[c]
        d = 2 * a
      else:
        a[0] = e
        d = 1
      return d

    node, _ = self._parse_and_analyze(test_fn)
    if_node = node.body[0].body[0]
    self.assertScopeIsRmc(
        anno.getanno(if_node, NodeAnno.BODY_SCOPE),
        ('a', 'b', 'c', 'a[c]'),
        ('a[b]', 'd'),
        ('d',),
    )
    # TODO(mdan): Should subscript writes (a[0] = 1) be considered to read "a"?
    self.assertScopeIsRmc(
        anno.getanno(if_node, NodeAnno.ORELSE_SCOPE),
        ('a', 'e'),
        ('a[0]', 'd'),
        ('d',),
    )
    self.assertScopeIsRmc(
        anno.getanno(if_node, NodeAnno.ORELSE_SCOPE).parent,
        ('a', 'b', 'c', 'd', 'e', 'a[c]'),
        ('d', 'a[b]', 'a[0]'),
        ('a', 'b', 'c', 'd', 'e'),
    )

  def test_nested_if(self):

    def test_fn(b):
      if b > 0:
        if b < 5:
          a = b
        else:
          a = b * b
      return a

    node, _ = self._parse_and_analyze(test_fn)
    inner_if_node = node.body[0].body[0].body[0]
    self.assertScopeIsRmc(
        anno.getanno(inner_if_node, NodeAnno.BODY_SCOPE), ('b',), ('a',),
        ('a',))
    self.assertScopeIsRmc(
        anno.getanno(inner_if_node, NodeAnno.ORELSE_SCOPE), ('b',), ('a',),
        ('a',))

  def test_nested_function(self):

    def test_fn(a):

      def f(x):
        y = x * x
        return y

      b = a
      for i in a:
        c = b
        b -= f(i)
      return b, c

    node, _ = self._parse_and_analyze(test_fn)
    fn_def_node = node.body[0].body[0]

    self.assertScopeIsRmc(
        anno.getanno(fn_def_node, NodeAnno.BODY_SCOPE), ('x', 'y'), ('y',), (
            'x',
            'y',
        ))

  def test_constructor_attributes(self):

    class TestClass(object):

      def __init__(self, a):
        self.b = a
        self.b.c = 1

    node, _ = self._parse_and_analyze(TestClass)
    init_node = node.body[0].body[0]
    self.assertScopeIsRmc(
        anno.getanno(init_node, NodeAnno.BODY_SCOPE),
        ('self', 'a', 'self.b'),
        ('self', 'self.b', 'self.b.c'),
        ('self', 'a', 'self.b'),
    )

  def test_aug_assign_subscripts(self):

    def test_fn(a):
      a[0] += 1

    node, _ = self._parse_and_analyze(test_fn)
    fn_node = node.body[0]
    self.assertScopeIsRmc(
        anno.getanno(fn_node, NodeAnno.BODY_SCOPE),
        ('a', 'a[0]'),
        ('a[0]',),
        ('a',),
    )

  def test_return_vars_are_read(self):

    def test_fn(a, b, c):  # pylint: disable=unused-argument
      return c

    node, _ = self._parse_and_analyze(test_fn)
    fn_node = node.body[0]
    self.assertScopeIsRmc(
        anno.getanno(fn_node, NodeAnno.BODY_SCOPE),
        ('c',),
        (),
        (
            'a',
            'b',
            'c',
        ),
    )

  def test_aug_assign(self):

    def test_fn(a, b):
      a += b

    node, _ = self._parse_and_analyze(test_fn)
    fn_node = node.body[0]
    self.assertScopeIsRmc(
        anno.getanno(fn_node, NodeAnno.BODY_SCOPE),
        ('a', 'b'),
        ('a'),
        ('a', 'b'),
    )

  def test_aug_assign_rvalues(self):

    a = dict(bar=3)

    def foo():
      return a

    def test_fn(x):
      foo()['bar'] += x

    node, _ = self._parse_and_analyze(test_fn)
    fn_node = node.body[0]
    self.assertScopeIsRmc(
        anno.getanno(fn_node, NodeAnno.BODY_SCOPE),
        ('foo', 'x'),
        (),
        ('x',),
    )

  def test_params_created(self):

    def test_fn(a, b):  # pylint: disable=unused-argument
      return b

    node, _ = self._parse_and_analyze(test_fn)
    fn_node = node.body[0]
    self.assertScopeIsRmc(
        anno.getanno(fn_node, NodeAnno.BODY_SCOPE), ('b',), (('')),
        (('a', 'b')))


if __name__ == '__main__':
  test.main()
