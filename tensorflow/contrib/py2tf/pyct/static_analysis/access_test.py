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
"""Tests for access module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gast

from tensorflow.contrib.py2tf.pyct import anno
from tensorflow.contrib.py2tf.pyct import parser
from tensorflow.contrib.py2tf.pyct.static_analysis import access
from tensorflow.python.platform import test


class ScopeTest(test.TestCase):

  def test_basic(self):
    scope = access.Scope(None)
    self.assertFalse(scope.has('foo'))

    scope.mark_read('foo')
    self.assertFalse(scope.has('foo'))

    scope.mark_write('foo')
    self.assertTrue(scope.has('foo'))

    scope.mark_read('bar')
    self.assertFalse(scope.has('bar'))

  def test_copy(self):
    scope = access.Scope(None)
    scope.mark_write('foo')

    other = access.Scope(None)
    other.copy_from(scope)

    self.assertTrue('foo' in other.created)

    scope.mark_write('bar')
    scope.copy_from(other)

    self.assertFalse('bar' in scope.created)

    scope.mark_write('bar')
    scope.merge_from(other)

    self.assertTrue('bar' in scope.created)
    self.assertFalse('bar' in other.created)

  def test_nesting(self):
    scope = access.Scope(None)
    scope.mark_write('foo')
    scope.mark_read('bar')

    child = access.Scope(scope)
    self.assertTrue(child.has('foo'))
    self.assertTrue(scope.has('foo'))

    child.mark_write('bar')
    self.assertTrue(child.has('bar'))
    self.assertFalse(scope.has('bar'))


class AccessResolverTest(test.TestCase):

  def test_local_markers(self):

    def test_fn(a):  # pylint:disable=unused-argument
      b = c  # pylint:disable=undefined-variable
      while b > 0:
        b -= 1
      return b

    node = parser.parse_object(test_fn)
    node = access.resolve(node)

    self.assertFalse(anno.getanno(node.body[0].body[0].value,
                                  'is_local'))  # c in b = c
    self.assertTrue(anno.getanno(node.body[0].body[1].test.left,
                                 'is_local'))  # b in b > 0
    self.assertTrue(anno.getanno(node.body[0].body[2].value,
                                 'is_local'))  # b in return b

  def assertScopeIs(self, scope, used, modified, created):
    self.assertItemsEqual(used, scope.used)
    self.assertItemsEqual(modified, scope.modified)
    self.assertItemsEqual(created, scope.created)

  def test_print_statement(self):

    def test_fn(a):
      b = 0
      c = 1
      print(a, b)
      return c

    node = parser.parse_object(test_fn)
    node = access.resolve(node)

    print_node = node.body[0].body[2]
    if isinstance(print_node, gast.Print):
      # Python 2
      print_args_scope = anno.getanno(print_node, 'args_scope')
    else:
      # Python 3
      assert isinstance(print_node, gast.Expr)
      # The call node should be the one being annotated.
      print_node = print_node.value
      print_args_scope = anno.getanno(print_node, 'args_scope')
    # We basically need to detect which variables are captured by the call
    # arguments.
    self.assertScopeIs(print_args_scope, ('a', 'b'), (), ())

  def test_call(self):

    def test_fn(a):
      b = 0
      c = 1
      foo(a, b)  # pylint:disable=undefined-variable
      return c

    node = parser.parse_object(test_fn)
    node = access.resolve(node)

    call_node = node.body[0].body[2].value
    # We basically need to detect which variables are captured by the call
    # arguments.
    self.assertScopeIs(
        anno.getanno(call_node, 'args_scope'), ('a', 'b'), (), ())

  def test_while(self):

    def test_fn(a):
      b = a
      while b > 0:
        c = b
        b -= 1
      return b, c

    node = parser.parse_object(test_fn)
    node = access.resolve(node)

    while_node = node.body[0].body[1]
    self.assertScopeIs(
        anno.getanno(while_node, 'body_scope'), ('b',), ('b', 'c'), ('c',))
    self.assertScopeIs(
        anno.getanno(while_node, 'body_parent_scope'), ('a', 'b', 'c'),
        ('a', 'b', 'c'), ('a', 'b', 'c'))

  def test_for(self):

    def test_fn(a):
      b = a
      for _ in a:
        c = b
        b -= 1
      return b, c

    node = parser.parse_object(test_fn)
    node = access.resolve(node)

    for_node = node.body[0].body[1]
    self.assertScopeIs(
        anno.getanno(for_node, 'body_scope'), ('b',), ('b', 'c'), ('c',))
    self.assertScopeIs(
        anno.getanno(for_node, 'body_parent_scope'), ('a', 'b', 'c'),
        ('a', 'b', 'c', '_'), ('a', 'b', 'c', '_'))

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

    node = parser.parse_object(test_fn)
    node = access.resolve(node)

    if_node = node.body[0].body[0]
    self.assertScopeIs(
        anno.getanno(if_node, 'body_scope'), ('x', 'y'), ('x', 'y', 'z'),
        ('y', 'z'))
    # TODO(mdan): Double check: is it ok to not mark a local symbol as not read?
    self.assertScopeIs(
        anno.getanno(if_node, 'body_parent_scope'), ('x', 'z', 'u'),
        ('x', 'y', 'z', 'u'), ('x', 'y', 'z', 'u'))
    self.assertScopeIs(
        anno.getanno(if_node, 'orelse_scope'), ('x', 'y'), ('x', 'y', 'u'),
        ('y', 'u'))
    self.assertScopeIs(
        anno.getanno(if_node, 'body_parent_scope'), ('x', 'z', 'u'),
        ('x', 'y', 'z', 'u'), ('x', 'y', 'z', 'u'))


if __name__ == '__main__':
  test.main()
