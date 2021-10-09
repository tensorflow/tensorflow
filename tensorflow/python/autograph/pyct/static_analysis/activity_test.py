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

import gast
import six

from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import naming
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.pyct import qual_names
from tensorflow.python.autograph.pyct import transformer
from tensorflow.python.autograph.pyct.static_analysis import activity
from tensorflow.python.autograph.pyct.static_analysis import annos
from tensorflow.python.platform import test


QN = qual_names.QN
NodeAnno = annos.NodeAnno

global_a = 7
global_b = 17


class ScopeTest(test.TestCase):

  def assertMissing(self, qn, scope):
    self.assertNotIn(qn, scope.read)
    self.assertNotIn(qn, scope.modified)

  def assertReadOnly(self, qn, scope):
    self.assertIn(qn, scope.read)
    self.assertNotIn(qn, scope.modified)

  def assertWriteOnly(self, qn, scope):
    self.assertNotIn(qn, scope.read)
    self.assertIn(qn, scope.modified)

  def assertReadWrite(self, qn, scope):
    self.assertIn(qn, scope.read)
    self.assertIn(qn, scope.modified)

  def test_copy_from(self):
    scope = activity.Scope(None)
    scope.modified.add(QN('foo'))
    other = activity.Scope(None)
    other.copy_from(scope)

    self.assertWriteOnly(QN('foo'), other)

    scope.modified.add(QN('bar'))
    scope.copy_from(other)

    self.assertMissing(QN('bar'), scope)

  def test_merge_from(self):
    scope = activity.Scope(None)
    other = activity.Scope(None)

    for col in (scope.modified, scope.read, scope.bound, scope.deleted):
      col.add(QN('foo'))

    for col in (other.modified, other.read, other.bound, other.deleted):
      col.add(QN('foo'))
      col.add(QN('bar'))

    scope.merge_from(other)

    self.assertReadWrite(QN('foo'), scope)
    self.assertReadWrite(QN('bar'), scope)
    self.assertIn(QN('foo'), scope.bound)
    self.assertIn(QN('bar'), scope.bound)
    self.assertIn(QN('foo'), scope.deleted)
    self.assertIn(QN('bar'), scope.deleted)

  def test_copy_of(self):
    scope = activity.Scope(None)
    scope.read.add(QN('foo'))
    other = activity.Scope.copy_of(scope)

    self.assertReadOnly(QN('foo'), other)

    child_scope = activity.Scope(scope)
    child_scope.read.add(QN('bar'))
    other = activity.Scope.copy_of(child_scope)

    self.assertReadOnly(QN('bar'), other)

  def test_referenced(self):
    scope = activity.Scope(None)
    scope.read.add(QN('a'))

    child = activity.Scope(scope)
    child.read.add(QN('b'))

    child2 = activity.Scope(child, isolated=False)
    child2.read.add(QN('c'))

    child2.finalize()
    child.finalize()
    scope.finalize()

    self.assertIn(QN('c'), child2.referenced)
    self.assertIn(QN('b'), child2.referenced)
    self.assertIn(QN('a'), child2.referenced)

    self.assertIn(QN('c'), child.referenced)
    self.assertIn(QN('b'), child.referenced)
    self.assertIn(QN('a'), child.referenced)


class ActivityAnalyzerTestBase(test.TestCase):

  def _parse_and_analyze(self, test_fn):
    # TODO(mdan): Use a custom FunctionTransformer here.
    node, source = parser.parse_entity(test_fn, future_features=())
    entity_info = transformer.EntityInfo(
        name=test_fn.__name__,
        source_code=source,
        source_file=None,
        future_features=(),
        namespace={})
    node = qual_names.resolve(node)
    namer = naming.Namer({})
    ctx = transformer.Context(entity_info, namer, None)
    node = activity.resolve(node, ctx)
    return node, entity_info

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

  def assertScopeIs(self, scope, used, modified):
    """Assert the scope contains specific used, modified & created variables."""
    self.assertSymbolSetsAre(used, scope.read, 'read')
    self.assertSymbolSetsAre(modified, scope.modified, 'modified')


class ActivityAnalyzerTest(ActivityAnalyzerTestBase):

  def test_import(self):

    def test_fn():
      import a, b.x, y as c, z.u as d  # pylint:disable=g-multiple-import,g-import-not-at-top,unused-variable

    node, _ = self._parse_and_analyze(test_fn)
    scope = anno.getanno(node.body[0], anno.Static.SCOPE)
    self.assertScopeIs(scope, (), ('a', 'b', 'c', 'd'))

  def test_import_from(self):

    def test_fn():
      from x import a  # pylint:disable=g-import-not-at-top,unused-variable
      from y import z as b  # pylint:disable=g-import-not-at-top,unused-variable

    node, _ = self._parse_and_analyze(test_fn)
    scope = anno.getanno(node.body[0], anno.Static.SCOPE)
    self.assertScopeIs(scope, (), ('a',))
    scope = anno.getanno(node.body[1], anno.Static.SCOPE)
    self.assertScopeIs(scope, (), ('b',))

  def test_print_statement(self):

    def test_fn(a):
      b = 0
      c = 1
      print(a, b)
      return c

    node, _ = self._parse_and_analyze(test_fn)
    print_node = node.body[2]
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
    self.assertScopeIs(print_args_scope, ('a', 'b'), ())

  def test_call_args(self):

    def test_fn(a):
      b = 0
      c = 1
      foo(a, b)  # pylint:disable=undefined-variable
      return c

    node, _ = self._parse_and_analyze(test_fn)
    call_node = node.body[2].value
    # We basically need to detect which variables are captured by the call
    # arguments.
    self.assertScopeIs(
        anno.getanno(call_node, NodeAnno.ARGS_SCOPE), ('a', 'b'), ())

  def test_call_args_attributes(self):

    def foo(*_):
      pass

    def test_fn(a):
      a.c = 0
      foo(a.b, a.c)
      return a.d

    node, _ = self._parse_and_analyze(test_fn)
    call_node = node.body[1].value
    self.assertScopeIs(
        anno.getanno(call_node, NodeAnno.ARGS_SCOPE), ('a', 'a.b', 'a.c'), ())

  def test_call_args_subscripts(self):

    def foo(*_):
      pass

    def test_fn(a):
      b = 1
      c = 2
      foo(a[0], a[b])
      return a[c]

    node, _ = self._parse_and_analyze(test_fn)
    call_node = node.body[2].value
    self.assertScopeIs(
        anno.getanno(call_node, NodeAnno.ARGS_SCOPE),
        ('a', 'a[0]', 'a[b]', 'b'), ())

  def test_while(self):

    def test_fn(a):
      b = a
      while b > 0:
        c = b
        b -= 1
      return b, c

    node, _ = self._parse_and_analyze(test_fn)
    while_node = node.body[1]
    self.assertScopeIs(
        anno.getanno(while_node, NodeAnno.BODY_SCOPE), ('b',), ('b', 'c'))
    self.assertScopeIs(
        anno.getanno(while_node, NodeAnno.BODY_SCOPE).parent, ('a', 'b', 'c'),
        ('b', 'c'))
    self.assertScopeIs(
        anno.getanno(while_node, NodeAnno.COND_SCOPE), ('b',), ())

  def test_for(self):

    def test_fn(a):
      b = a
      for _ in a:
        c = b
        b -= 1
      return b, c

    node, _ = self._parse_and_analyze(test_fn)
    for_node = node.body[1]
    self.assertScopeIs(
        anno.getanno(for_node, NodeAnno.ITERATE_SCOPE), (), ('_'))
    self.assertScopeIs(
        anno.getanno(for_node, NodeAnno.BODY_SCOPE), ('b',), ('b', 'c'))
    self.assertScopeIs(
        anno.getanno(for_node, NodeAnno.BODY_SCOPE).parent, ('a', 'b', 'c'),
        ('b', 'c', '_'))

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
    if_node = node.body[0]
    self.assertScopeIs(
        anno.getanno(if_node, NodeAnno.BODY_SCOPE), ('x', 'y'), ('x', 'y', 'z'))
    self.assertScopeIs(
        anno.getanno(if_node, NodeAnno.BODY_SCOPE).parent, ('x', 'y', 'z', 'u'),
        ('x', 'y', 'z', 'u'))
    self.assertScopeIs(
        anno.getanno(if_node, NodeAnno.ORELSE_SCOPE), ('x', 'y'),
        ('x', 'y', 'u'))
    self.assertScopeIs(
        anno.getanno(if_node, NodeAnno.ORELSE_SCOPE).parent,
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
    if_node = node.body[0]
    self.assertScopeIs(
        anno.getanno(if_node, NodeAnno.BODY_SCOPE), ('a', 'a.c'), ('a.b', 'd'))
    self.assertScopeIs(
        anno.getanno(if_node, NodeAnno.ORELSE_SCOPE), ('a', 'a.c'),
        ('a.b', 'd'))
    self.assertScopeIs(
        anno.getanno(if_node, NodeAnno.BODY_SCOPE).parent, ('a', 'a.c', 'd'),
        ('a.b', 'd'))

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
    if_node = node.body[0]
    self.assertScopeIs(
        anno.getanno(if_node, NodeAnno.BODY_SCOPE), ('a', 'b', 'c', 'a[c]'),
        ('a[b]', 'd'))
    # TODO(mdan): Should subscript writes (a[0] = 1) be considered to read "a"?
    self.assertScopeIs(
        anno.getanno(if_node, NodeAnno.ORELSE_SCOPE), ('a', 'e'), ('a[0]', 'd'))
    self.assertScopeIs(
        anno.getanno(if_node, NodeAnno.ORELSE_SCOPE).parent,
        ('a', 'b', 'c', 'd', 'e', 'a[c]'), ('d', 'a[b]', 'a[0]'))

  def test_nested_if(self):

    def test_fn(b):
      if b > 0:
        if b < 5:
          a = b
        else:
          a = b * b
      return a

    node, _ = self._parse_and_analyze(test_fn)
    inner_if_node = node.body[0].body[0]
    self.assertScopeIs(
        anno.getanno(inner_if_node, NodeAnno.BODY_SCOPE), ('b',), ('a',))
    self.assertScopeIs(
        anno.getanno(inner_if_node, NodeAnno.ORELSE_SCOPE), ('b',), ('a',))

  def test_nested_function(self):

    def test_fn(a):

      def f(x):
        y = x * x
        return y

      return f(a)

    node, _ = self._parse_and_analyze(test_fn)

    fn_node = node
    scope = anno.getanno(fn_node, NodeAnno.BODY_SCOPE)
    self.assertScopeIs(scope, ('a', 'f'), ('f',))

    fn_def_node = node.body[0]

    scope = anno.getanno(fn_def_node, anno.Static.SCOPE)
    self.assertScopeIs(scope, (), ('f'))

    scope = anno.getanno(fn_def_node, NodeAnno.BODY_SCOPE)
    self.assertScopeIs(scope, ('x', 'y'), ('y',))

    scope = anno.getanno(fn_def_node, NodeAnno.ARGS_AND_BODY_SCOPE)
    self.assertScopeIs(scope, ('x', 'y'), ('y',))
    self.assertSymbolSetsAre(('x', 'y'), scope.bound, 'BOUND')

  def test_nested_lambda(self):

    def test_fn(a):
      return lambda x: (x * a)

    node, _ = self._parse_and_analyze(test_fn)

    fn_node = node
    scope = anno.getanno(fn_node, NodeAnno.BODY_SCOPE)
    self.assertScopeIs(scope, ('a',), ())

    return_node = node.body[0]

    scope = anno.getanno(return_node, anno.Static.SCOPE)
    self.assertScopeIs(scope, ('a',), ())

    lam_def_node = return_node.value

    scope = anno.getanno(lam_def_node, NodeAnno.BODY_SCOPE)
    self.assertScopeIs(scope, ('a', 'x'), ())

    scope = anno.getanno(lam_def_node, NodeAnno.ARGS_AND_BODY_SCOPE)
    self.assertScopeIs(scope, ('a', 'x'), ())
    self.assertSymbolSetsAre(('x',), scope.bound, 'BOUND')

  def test_nested_function_arg_defaults(self):

    def test_fn(a):

      def f(x=a):
        y = x * x
        return y

      return f(a)

    node, _ = self._parse_and_analyze(test_fn)
    fn_def_node = node.body[0]

    self.assertScopeIs(
        anno.getanno(fn_def_node, anno.Static.SCOPE), ('a',), ('f',))

    scope = anno.getanno(fn_def_node, NodeAnno.BODY_SCOPE)
    self.assertScopeIs(scope, ('x', 'y'), ('y',))

    scope = anno.getanno(fn_def_node, NodeAnno.ARGS_AND_BODY_SCOPE)
    self.assertScopeIs(scope, ('x', 'y'), ('y',))
    self.assertSymbolSetsAre(('x', 'y'), scope.bound, 'BOUND')

  def test_constructor_attributes(self):

    class TestClass(object):

      def __init__(self, a):
        self.b = a
        self.b.c = 1

    node, _ = self._parse_and_analyze(TestClass)
    init_node = node.body[0]
    self.assertScopeIs(
        anno.getanno(init_node, NodeAnno.BODY_SCOPE), ('self', 'a', 'self.b'),
        ('self', 'self.b', 'self.b.c'))

  def test_aug_assign_subscripts(self):

    def test_fn(a):
      a[0] += 1

    node, _ = self._parse_and_analyze(test_fn)
    fn_node = node
    self.assertScopeIs(
        anno.getanno(fn_node, NodeAnno.BODY_SCOPE), ('a', 'a[0]'), ('a[0]',))

  def test_return_vars_are_read(self):

    def test_fn(a, b, c):  # pylint: disable=unused-argument
      return c

    node, _ = self._parse_and_analyze(test_fn)
    fn_node = node
    self.assertScopeIs(anno.getanno(fn_node, NodeAnno.BODY_SCOPE), ('c',), ())
    self.assertScopeIs(
        anno.getanno(node.body[0], anno.Static.SCOPE), ('c',), ())

  def test_raise_names_are_read(self):

    def test_fn(a, b, c):  # pylint: disable=unused-argument
      raise b

    node, _ = self._parse_and_analyze(test_fn)
    fn_node = node
    self.assertScopeIs(anno.getanno(fn_node, NodeAnno.BODY_SCOPE), ('b',), ())
    self.assertScopeIs(
        anno.getanno(node.body[0], anno.Static.SCOPE), ('b',), ())

  def test_except_exposes_names(self):

    def test_fn(a, b, c):  # pylint: disable=unused-argument
      try:
        pass
      except:  # pylint: disable=bare-except
        b = c

    node, _ = self._parse_and_analyze(test_fn)
    fn_node = node
    self.assertScopeIs(
        anno.getanno(fn_node, NodeAnno.BODY_SCOPE), ('c',), ('b',))

  def test_except_hides_exception_var_name(self):

    def test_fn(a, b, c):  # pylint: disable=unused-argument
      try:
        pass
      except a as e:
        b = e

    node, _ = self._parse_and_analyze(test_fn)
    fn_node = node
    self.assertScopeIs(
        anno.getanno(fn_node, NodeAnno.BODY_SCOPE), ('a',), ('b',))

  def test_aug_assign(self):

    def test_fn(a, b):
      a += b

    node, _ = self._parse_and_analyze(test_fn)
    fn_node = node
    self.assertScopeIs(
        anno.getanno(fn_node, NodeAnno.BODY_SCOPE), ('a', 'b'), ('a'))

  def test_aug_assign_rvalues(self):

    a = dict(bar=3)

    def foo():
      return a

    def test_fn(x):
      foo()['bar'] += x

    node, _ = self._parse_and_analyze(test_fn)
    fn_node = node
    self.assertScopeIs(
        anno.getanno(fn_node, NodeAnno.BODY_SCOPE), ('foo', 'x'), ())

  def test_lambda(self):

    def test_fn(a, b):
      return lambda: (a + b)

    node, _ = self._parse_and_analyze(test_fn)

    fn_node = node
    scope = anno.getanno(fn_node, NodeAnno.BODY_SCOPE)
    self.assertScopeIs(scope, ('a', 'b'), ())

    lam_def_node = node.body[0].value

    scope = anno.getanno(lam_def_node, anno.Static.SCOPE)
    self.assertScopeIs(scope, (), ())

    scope = anno.getanno(lam_def_node, NodeAnno.BODY_SCOPE)
    self.assertScopeIs(scope, ('a', 'b'), ())

    scope = anno.getanno(lam_def_node, NodeAnno.ARGS_AND_BODY_SCOPE)
    self.assertScopeIs(scope, ('a', 'b'), ())
    self.assertSymbolSetsAre((), scope.bound, 'BOUND')

    scope = anno.getanno(lam_def_node.args, anno.Static.SCOPE)
    self.assertSymbolSetsAre((), scope.params.keys(), 'lambda params')

  def test_lambda_params_args(self):

    def test_fn(a, b):  # pylint: disable=unused-argument
      return lambda a: a + b

    node, _ = self._parse_and_analyze(test_fn)

    fn_node = node
    scope = anno.getanno(fn_node, NodeAnno.BODY_SCOPE)
    # Note: `a` in `a + b` is not "read" here because it's hidden by the `a`
    # argument.
    self.assertScopeIs(scope, ('b',), ())

    lam_def_node = node.body[0].value

    scope = anno.getanno(lam_def_node, anno.Static.SCOPE)
    self.assertScopeIs(scope, (), ())

    scope = anno.getanno(lam_def_node, NodeAnno.BODY_SCOPE)
    self.assertScopeIs(scope, ('a', 'b'), ())

    scope = anno.getanno(lam_def_node, NodeAnno.ARGS_AND_BODY_SCOPE)
    self.assertScopeIs(scope, ('a', 'b'), ())
    self.assertSymbolSetsAre(('a',), scope.bound, 'BOUND')

    scope = anno.getanno(lam_def_node.args, anno.Static.SCOPE)
    self.assertSymbolSetsAre(('a',), scope.params.keys(), 'lambda params')

  def test_lambda_params_arg_defaults(self):

    def test_fn(a, b, c):  # pylint: disable=unused-argument
      return lambda b=c: a + b

    node, _ = self._parse_and_analyze(test_fn)

    fn_node = node
    scope = anno.getanno(fn_node, NodeAnno.BODY_SCOPE)
    # Note: `b` is not "read" here because it's hidden by the argument.
    self.assertScopeIs(scope, ('a', 'c'), ())

    lam_def_node = node.body[0].value

    scope = anno.getanno(lam_def_node, anno.Static.SCOPE)
    self.assertScopeIs(scope, ('c',), ())

    scope = anno.getanno(lam_def_node, NodeAnno.BODY_SCOPE)
    self.assertScopeIs(scope, ('a', 'b'), ())

    scope = anno.getanno(lam_def_node, NodeAnno.ARGS_AND_BODY_SCOPE)
    self.assertScopeIs(scope, ('a', 'b'), ())
    self.assertSymbolSetsAre(('b',), scope.bound, 'BOUND')

    scope = anno.getanno(lam_def_node.args, anno.Static.SCOPE)
    self.assertSymbolSetsAre(('b',), scope.params.keys(), 'lambda params')

  def test_lambda_complex(self):

    def test_fn(a, b, c, d, e):  # pylint: disable=unused-argument
      a = (lambda a, b, c=e: a + b + c)(d, 1, 2) + b

    node, _ = self._parse_and_analyze(test_fn)

    fn_node = node
    scope = anno.getanno(fn_node, NodeAnno.BODY_SCOPE)
    self.assertScopeIs(scope, ('d', 'b', 'e'), ('a',))

    lam_def_node = node.body[0].value.left.func

    scope = anno.getanno(lam_def_node, anno.Static.SCOPE)
    self.assertScopeIs(scope, ('e',), ())

    scope = anno.getanno(lam_def_node, NodeAnno.BODY_SCOPE)
    self.assertScopeIs(scope, ('a', 'b', 'c'), ())

    scope = anno.getanno(lam_def_node, NodeAnno.ARGS_AND_BODY_SCOPE)
    self.assertScopeIs(scope, ('a', 'b', 'c'), ())
    self.assertSymbolSetsAre(('a', 'b', 'c'), scope.bound, 'BOUND')

    scope = anno.getanno(lam_def_node.args, anno.Static.SCOPE)
    self.assertSymbolSetsAre(
        ('a', 'b', 'c'), scope.params.keys(), 'lambda params')

  def test_lambda_nested(self):

    def test_fn(a, b, c, d, e, f):  # pylint: disable=unused-argument
      a = lambda a, b: d(lambda b=f: a + b + c)  # pylint: disable=undefined-variable

    node, _ = self._parse_and_analyze(test_fn)

    fn_node = node
    scope = anno.getanno(fn_node, NodeAnno.BODY_SCOPE)
    self.assertScopeIs(scope, ('d', 'c', 'f'), ('a',))

    outer_lam_def = node.body[0].value

    scope = anno.getanno(outer_lam_def, anno.Static.SCOPE)
    self.assertScopeIs(scope, (), ())

    scope = anno.getanno(outer_lam_def, NodeAnno.BODY_SCOPE)
    self.assertScopeIs(scope, ('d', 'f', 'a', 'c'), ())

    scope = anno.getanno(outer_lam_def, NodeAnno.ARGS_AND_BODY_SCOPE)
    self.assertScopeIs(scope, ('d', 'f', 'a', 'c'), ())
    self.assertSymbolSetsAre(('a', 'b'), scope.bound, 'BOUND')

    scope = anno.getanno(outer_lam_def.args, anno.Static.SCOPE)
    self.assertSymbolSetsAre(('a', 'b'), scope.params.keys(), 'lambda params')

    inner_lam_def = outer_lam_def.body.args[0]

    scope = anno.getanno(inner_lam_def, anno.Static.SCOPE)
    self.assertScopeIs(scope, ('f',), ())

    scope = anno.getanno(inner_lam_def, NodeAnno.BODY_SCOPE)
    self.assertScopeIs(scope, ('a', 'b', 'c'), ())

    scope = anno.getanno(inner_lam_def, NodeAnno.ARGS_AND_BODY_SCOPE)
    self.assertScopeIs(scope, ('a', 'b', 'c'), ())
    self.assertSymbolSetsAre(('b',), scope.bound, 'BOUND')

    scope = anno.getanno(inner_lam_def.args, anno.Static.SCOPE)
    self.assertSymbolSetsAre(('b',), scope.params.keys(), 'lambda params')

  def test_comprehension_targets_are_isolated(self):

    def test_fn(a):
      b = {c for c in a}  # pylint:disable=unused-variable

    node, _ = self._parse_and_analyze(test_fn)
    fn_node = node
    body_scope = anno.getanno(fn_node, NodeAnno.BODY_SCOPE)
    self.assertScopeIs(body_scope, ('a',), ('b',))

  def test_comprehension_targets_are_isolated_list_function_w_generator(self):

    def test_fn(a):
      b = list(c for c in a)  # pylint:disable=unused-variable

    node, _ = self._parse_and_analyze(test_fn)
    fn_node = node
    body_scope = anno.getanno(fn_node, NodeAnno.BODY_SCOPE)
    self.assertScopeIs(body_scope, ('a', 'list'), ('b',))

  def test_list_comprehension_targets_are_sometimes_isolated(self):

    def test_fn(a):
      b = [c for c in a]  # pylint:disable=unused-variable

    node, _ = self._parse_and_analyze(test_fn)
    fn_node = node
    body_scope = anno.getanno(fn_node, NodeAnno.BODY_SCOPE)
    if six.PY2:
      self.assertScopeIs(body_scope, ('a',), ('b', 'c'))
    else:
      self.assertScopeIs(body_scope, ('a',), ('b',))

  def test_comprehension_targets_are_isolated_in_augassign(self):

    def test_fn(a, b):
      b += [c for c in a]  # pylint:disable=unused-variable

    node, _ = self._parse_and_analyze(test_fn)
    fn_node = node
    body_scope = anno.getanno(fn_node, NodeAnno.BODY_SCOPE)
    if six.PY2:
      self.assertScopeIs(body_scope, ('a', 'b'), ('b', 'c'))
    else:
      self.assertScopeIs(body_scope, ('a', 'b'), ('b',))

  def test_comprehension_generator_order(self):

    def test_fn(a, b, c):  # pylint:disable=unused-argument
      e = {d: (a, b) for (a, b) in c for d in b}  # pylint:disable=unused-variable,g-complex-comprehension

    node, _ = self._parse_and_analyze(test_fn)
    fn_node = node
    body_scope = anno.getanno(fn_node, NodeAnno.BODY_SCOPE)
    self.assertScopeIs(body_scope, ('c',), ('e',))

  def test_global_symbol(self):

    def test_fn(c):
      global global_a
      global global_b
      global_a = global_b + c

    node, _ = self._parse_and_analyze(test_fn)
    fn_node = node
    body_scope = anno.getanno(fn_node, NodeAnno.BODY_SCOPE)
    self.assertScopeIs(body_scope, ('global_a', 'global_b', 'c'), ('global_a',))
    self.assertSetEqual(body_scope.globals, set(
        (QN('global_a'), QN('global_b'))))
    global_a_scope = anno.getanno(fn_node.body[0], anno.Static.SCOPE)
    self.assertScopeIs(global_a_scope, ('global_a',), ())

  def test_class_definition_basic(self):

    def test_fn(a, b):
      class C(a(b)):
        d = 1
      return C

    node, _ = self._parse_and_analyze(test_fn)
    fn_node = node
    body_scope = anno.getanno(fn_node, NodeAnno.BODY_SCOPE)
    self.assertScopeIs(body_scope, ('a', 'b', 'C'), ('C',))

  def test_class_definition_isolates_method_writes(self):

    def test_fn(a, b, c):
      class C(a(b)):
        d = 1

        def e(self):
          f = c + 1
          return f
      return C

    node, _ = self._parse_and_analyze(test_fn)
    fn_node = node
    body_scope = anno.getanno(fn_node, NodeAnno.BODY_SCOPE)
    self.assertScopeIs(body_scope, ('a', 'b', 'C', 'c'), ('C',))


if __name__ == '__main__':
  test.main()
