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
"""Tests for anf module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import textwrap

import gast

from tensorflow.python.autograph.pyct import loader
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.pyct import transformer
from tensorflow.python.autograph.pyct.common_transformers import anf
from tensorflow.python.platform import test


# TODO(mdan): These two functions no longer need to be at the top level.
# TODO(mdan): Don't use exec.
def exec_test_function():
  # The point is to test A-normal form conversion of exec
  # pylint: disable=exec-used
  exec('computed' + 5 + 'stuff', globals(), locals())


def exec_expected_result():
  # pylint: disable=exec-used
  tmp_1001 = 'computed' + 5
  tmp_1002 = tmp_1001 + 'stuff'
  tmp_1003 = globals()
  tmp_1004 = locals()
  exec(tmp_1002, tmp_1003, tmp_1004)


class AnfTestBase(test.TestCase):

  def _simple_context(self):
    entity_info = transformer.EntityInfo(
        name='test_fn',
        source_code=None,
        source_file=None,
        future_features=(),
        namespace=None)
    return transformer.Context(entity_info, None, None)

  def assert_same_ast(self, expected_node, node, msg=None):
    expected_source = parser.unparse(expected_node, indentation='  ')
    expected_str = textwrap.dedent(expected_source).strip()
    got_source = parser.unparse(node, indentation='  ')
    got_str = textwrap.dedent(got_source).strip()
    self.assertEqual(expected_str, got_str, msg=msg)

  def assert_body_anfs_as_expected(self, expected_fn, test_fn, config=None):
    # Testing the code bodies only.  Wrapping them in functions so the
    # syntax highlights nicely, but Python doesn't try to execute the
    # statements.
    exp_node, _ = parser.parse_entity(expected_fn, future_features=())
    node, _ = parser.parse_entity(test_fn, future_features=())
    node = anf.transform(node, self._simple_context(), config=config)
    exp_name = exp_node.name
    # Ignoring the function names in the result because they can't be
    # the same (because both functions have to exist in the same scope
    # at the same time).
    node.name = exp_name
    self.assert_same_ast(exp_node, node)
    # Check that ANF is idempotent
    node_repeated = anf.transform(node, self._simple_context())
    self.assert_same_ast(node_repeated, node)


class AnfTransformerTest(AnfTestBase):

  def test_basic(self):
    def test_function():
      a = 0
      return a

    node, _ = parser.parse_entity(test_function, future_features=())
    node = anf.transform(node, self._simple_context())
    result, _, _ = loader.load_ast(node)
    self.assertEqual(test_function(), result.test_function())

  def test_binop_basic(self):

    def test_function(x, y, z):
      a = x + y + z
      return a

    def expected_result(x, y, z):
      tmp_1001 = x + y
      a = tmp_1001 + z
      return a

    self.assert_body_anfs_as_expected(expected_result, test_function)

  def test_if_basic(self):

    def test_function(a, b, c, e, f, g):
      if a + b + c:
        d = e + f + g
        return d

    def expected_result(a, b, c, e, f, g):
      tmp_1001 = a + b
      tmp_1002 = tmp_1001 + c
      if tmp_1002:
        tmp_1003 = e + f
        d = tmp_1003 + g
        return d

    self.assert_body_anfs_as_expected(expected_result, test_function)

  def test_nested_binop_and_return(self):

    def test_function(b, c, d, e):
      return (2 * b + c) + (d + e)

    def expected_result(b, c, d, e):
      tmp_1001 = 2 * b
      tmp_1002 = tmp_1001 + c
      tmp_1003 = d + e
      tmp_1004 = tmp_1002 + tmp_1003
      return tmp_1004

    self.assert_body_anfs_as_expected(expected_result, test_function)

  def test_function_call_and_expr(self):

    def test_function(call_something, a, b, y, z, c, d, e, f, g, h, i):
      call_something(a + b, y * z, kwarg=c + d, *(e + f), **(g + h + i))

    def expected_result(call_something, a, b, y, z, c, d, e, f, g, h, i):
      tmp_1001 = g + h
      tmp_1002 = a + b
      tmp_1003 = y * z
      tmp_1004 = e + f
      tmp_1005 = c + d
      tmp_1006 = tmp_1001 + i
      call_something(tmp_1002, tmp_1003, kwarg=tmp_1005, *tmp_1004, **tmp_1006)

    self.assert_body_anfs_as_expected(expected_result, test_function)

  def test_with_and_print(self):

    def test_function(a, b, c):
      with a + b + c as d:
        print(2 * d + 1)

    def expected_result(a, b, c):
      tmp_1001 = a + b
      tmp_1002 = tmp_1001 + c
      with tmp_1002 as d:
        tmp_1003 = 2 * d
        tmp_1004 = tmp_1003 + 1
        print(tmp_1004)

    self.assert_body_anfs_as_expected(expected_result, test_function)

  def test_nested_multi_value_assign(self):

    def test_function(a, b, c):
      x, y = a, a + b
      (z, y), x = (c, y + b), x + a
      return z, (y, x)

    def expected_result(a, b, c):
      tmp_1001 = a + b
      x, y = a, tmp_1001
      tmp_1002 = y + b
      tmp_1003 = (c, tmp_1002)
      tmp_1004 = x + a
      (z, y), x = tmp_1003, tmp_1004
      tmp_1005 = y, x
      tmp_1006 = z, tmp_1005
      return tmp_1006

    self.assert_body_anfs_as_expected(expected_result, test_function)

  def test_deeply_nested_multi_value_assign(self):

    def test_function(a):
      [([(b, c), [d, e]], (f, g)), [(h, i, j), k]] = a
      return [([(b, c), [d, e]], (f, g)), [(h, i, j), k]]

    def expected_result(a):
      [([(b, c), [d, e]], (f, g)), [(h, i, j), k]] = a
      tmp_1001 = b, c
      tmp_1002 = [d, e]
      tmp_1003 = [tmp_1001, tmp_1002]
      tmp_1004 = f, g
      tmp_1005 = h, i, j
      tmp_1006 = tmp_1003, tmp_1004
      tmp_1007 = [tmp_1005, k]
      tmp_1008 = [tmp_1006, tmp_1007]
      return tmp_1008

    self.assert_body_anfs_as_expected(expected_result, test_function)

  def test_local_definition_and_binary_compare(self):

    def test_function():
      def foo(a, b):
        return 2 * a < b
      return foo

    def expected_result():
      def foo(a, b):
        tmp_1001 = 2 * a
        tmp_1002 = tmp_1001 < b
        return tmp_1002
      return foo

    self.assert_body_anfs_as_expected(expected_result, test_function)

  def test_list_literal(self):

    def test_function(a, b, c, d, e, f):
      return [a + b, c + d, e + f]

    def expected_result(a, b, c, d, e, f):
      tmp_1001 = a + b
      tmp_1002 = c + d
      tmp_1003 = e + f
      tmp_1004 = [tmp_1001, tmp_1002, tmp_1003]
      return tmp_1004

    self.assert_body_anfs_as_expected(expected_result, test_function)

  def test_tuple_literal_and_unary(self):

    def test_function(a, b, c, d, e, f):
      return (a + b, -(c + d), e + f)

    def expected_result(a, b, c, d, e, f):
      tmp_1001 = c + d
      tmp_1002 = a + b
      tmp_1003 = -tmp_1001
      tmp_1004 = e + f
      tmp_1005 = (tmp_1002, tmp_1003, tmp_1004)
      return tmp_1005

    self.assert_body_anfs_as_expected(expected_result, test_function)

  def test_set_literal(self):

    def test_function(a, b, c, d, e, f):
      return set(a + b, c + d, e + f)

    def expected_result(a, b, c, d, e, f):
      tmp_1001 = a + b
      tmp_1002 = c + d
      tmp_1003 = e + f
      tmp_1004 = set(tmp_1001, tmp_1002, tmp_1003)
      return tmp_1004

    self.assert_body_anfs_as_expected(expected_result, test_function)

  def test_dict_literal_and_repr(self):

    def test_function(foo, bar, baz):
      return repr({foo + bar + baz: 7 | 8})

    def expected_result(foo, bar, baz):
      tmp_1001 = foo + bar
      tmp_1002 = tmp_1001 + baz
      tmp_1003 = 7 | 8
      tmp_1004 = {tmp_1002: tmp_1003}
      tmp_1005 = repr(tmp_1004)
      return tmp_1005

    self.assert_body_anfs_as_expected(expected_result, test_function)

  def test_field_read_and_write(self):

    def test_function(a, d):
      a.b.c = d.e.f + 3

    def expected_result(a, d):
      tmp_1001 = a.b
      tmp_1002 = d.e
      tmp_1003 = tmp_1002.f
      tmp_1001.c = tmp_1003 + 3

    self.assert_body_anfs_as_expected(expected_result, test_function)

  def test_subscript_read_and_write(self):

    def test_function(a, b, c, d, e, f):
      a[b][c] = d[e][f] + 3

    def expected_result(a, b, c, d, e, f):
      tmp_1001 = a[b]
      tmp_1002 = d[e]
      tmp_1003 = tmp_1002[f]
      tmp_1001[c] = tmp_1003 + 3

    self.assert_body_anfs_as_expected(expected_result, test_function)

  def test_augassign_and_delete(self):

    def test_function(a, x, y, z):
      a += x + y + z
      del a
      del z[y][x]

    def expected_result(a, x, y, z):
      tmp_1001 = x + y
      a += tmp_1001 + z
      del a
      tmp_1002 = z[y]
      del tmp_1002[x]

    self.assert_body_anfs_as_expected(expected_result, test_function)

  def test_raise_yield_and_raise(self):

    def test_function(a, c, some_computed, exception):
      yield a ** c
      raise some_computed('complicated' + exception)

    def expected_result(a, c, some_computed, exception):
      tmp_1001 = a ** c
      yield tmp_1001
      tmp_1002 = 'complicated' + exception
      tmp_1003 = some_computed(tmp_1002)
      raise tmp_1003

    self.assert_body_anfs_as_expected(expected_result, test_function)

  def test_with_and_if_with_expressions(self):

    def test_function(foo, bar, function, quux, quozzle, w, x, y, z):
      with foo + bar:
        function(x + y)
      if quux + quozzle:
        function(z / w)

    def expected_result(foo, bar, function, quux, quozzle, w, x, y, z):
      tmp_1001 = foo + bar
      with tmp_1001:
        tmp_1002 = x + y
        function(tmp_1002)
      tmp_1003 = quux + quozzle
      if tmp_1003:
        tmp_1004 = z / w
        function(tmp_1004)

    self.assert_body_anfs_as_expected(expected_result, test_function)

  def test_exec(self):
    self.assert_body_anfs_as_expected(exec_expected_result, exec_test_function)

  def test_simple_while_and_assert(self):

    def test_function(foo, quux):
      while foo:
        assert quux
        foo = foo + 1 * 3

    def expected_result(foo, quux):
      while foo:
        assert quux
        tmp_1001 = 1 * 3
        foo = foo + tmp_1001

    self.assert_body_anfs_as_expected(expected_result, test_function)

  def test_for(self):

    def test_function(compute, something, complicated, foo):
      for foo in compute(something + complicated):
        bar = foo + 1 * 3
      return bar

    def expected_result(compute, something, complicated, foo):
      tmp_1001 = something + complicated
      tmp_1002 = compute(tmp_1001)
      for foo in tmp_1002:
        tmp_1003 = 1 * 3
        bar = foo + tmp_1003
      return bar

    self.assert_body_anfs_as_expected(expected_result, test_function)

  # This test collects several examples where the definition of A-normal form
  # implemented by this transformer is questionable.  Mostly it's here to spell
  # out what the definition is in these cases.
  def test_controversial(self):

    def test_function(b, c, d, f):
      a = c + d
      a.b = c + d
      a[b] = c + d
      a += c + d
      a, b = c
      a, b = c, d
      a = f(c)
      a = f(c + d)
      a[b + d] = f.e(c + d)

    def expected_result(b, c, d, f):
      a = c + d
      a.b = c + d  # Should be a.b = tmp?  (Definitely not tmp = c + d)
      a[b] = c + d  # Should be a[b] = tmp?  (Definitely not tmp = c + d)
      a += c + d  # Should be a += tmp?  (Definitely not tmp = c + d)
      a, b = c  # Should be a = c[0], b = c[1]?  Or not?
      a, b = c, d  # Should be a = c, b = d?  Or not?
      a = f(c)
      tmp_1001 = c + d
      a = f(tmp_1001)
      tmp_1002 = b + d
      tmp_1003 = f.e
      tmp_1004 = c + d
      a[tmp_1002] = tmp_1003(tmp_1004)  # Or should be a[tmp1] = tmp2?

    self.assert_body_anfs_as_expected(expected_result, test_function)


class AnfNonTransformationTest(AnfTransformerTest):
  """Test that specifying "no transformation" does nothing.

  Reuses all the examples of AnfTransformerTest by overriding
  `assert_body_anfs_as_expected_`.
  """

  def assert_body_anfs_as_expected(self, expected_fn, test_fn):
    # Testing the code bodies only.  Wrapping them in functions so the
    # syntax highlights nicely, but Python doesn't try to execute the
    # statements.
    node, _ = parser.parse_entity(test_fn, future_features=())
    orig_source = parser.unparse(node, indentation='  ')
    orig_str = textwrap.dedent(orig_source).strip()
    config = [(anf.ANY, anf.LEAVE)]  # Configuration to transform nothing
    node = anf.transform(node, self._simple_context(), config=config)
    new_source = parser.unparse(node, indentation='  ')
    new_str = textwrap.dedent(new_source).strip()
    self.assertEqual(orig_str, new_str)


class AnfConfiguredTest(AnfTestBase):

  def test_constants_in_function_calls(self):
    # An example specific configuration that differs from the default: Moving
    # literals out of being directly passed to functions, but nothing else.
    try:
      # TODO(b/140808434): Fix this.
      # gast pre-0.3
      literals = (gast.Num, gast.Str, gast.Bytes, gast.NameConstant, gast.Name)
    except AttributeError:
      # gast 0.3+
      literals = (gast.Constant, gast.Name)
    config = [(anf.ASTEdgePattern(gast.Call, anf.ANY, literals), anf.REPLACE)]

    def test_function(x, frob):
      return frob(x, x+1, 2)

    def expected_result(x, frob):
      tmp_1001 = 2
      return frob(x, x+1, tmp_1001)

    self.assert_body_anfs_as_expected(expected_result, test_function, config)

  def test_anf_some_function_calls(self):
    # Another example specific configuration that differs from the default:
    # Moving all arguments out of some function calls but leaving others be.
    allowlist = ['foo']

    def transform(parent, field, child):
      del field
      del child
      func_name = parent.func.id
      return str(func_name) in allowlist

    config = [(anf.ASTEdgePattern(gast.Call, anf.ANY, anf.ANY), transform)]

    def test_function(x, foo, bar):
      y = foo(x, x+1, 2)
      return bar(y, y+1, 2)

    def expected_result(x, foo, bar):
      tmp_1001 = x+1
      tmp_1002 = 2
      y = foo(x, tmp_1001, tmp_1002)
      return bar(y, y+1, 2)

    self.assert_body_anfs_as_expected(expected_result, test_function, config)

  def test_touching_name_constant(self):
    # Checking that the nodes for `True`, `False`, and `None` can be manipulated
    # by a configuration.  This is non-trivial, because in Python 2 those are
    # represented as `Name`, which is the same node type as variable references.
    specials = (gast.Name, gast.Constant)
    config = [(anf.ASTEdgePattern(gast.Call, anf.ANY, specials), anf.REPLACE)]

    def test_function(f):
      return f(True, False, None)

    def expected_result(f):
      tmp_1001 = True
      tmp_1002 = False
      tmp_1003 = None
      return f(tmp_1001, tmp_1002, tmp_1003)

    self.assert_body_anfs_as_expected(expected_result, test_function, config)


if __name__ == '__main__':
  test.main()
