# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for type_inference module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Any, Callable, List

from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import cfg
from tensorflow.python.autograph.pyct import qual_names
from tensorflow.python.autograph.pyct import transpiler
from tensorflow.python.autograph.pyct.static_analysis import activity
from tensorflow.python.autograph.pyct.static_analysis import reaching_definitions
from tensorflow.python.autograph.pyct.static_analysis import reaching_fndefs
from tensorflow.python.autograph.pyct.static_analysis import type_inference
from tensorflow.python.platform import test


class BasicTestResolver(type_inference.Resolver):
  """A very basic resolver for testing."""

  def res_name(self, ns, types_ns, name):
    str_name = str(name)
    if str_name == 'int':
      return {int}, int
    return {type(ns[str_name])}, ns[str_name]

  def res_value(self, ns, value):
    return {type(value)}

  def res_arg(self, ns, types_ns, f_name, name, type_anno, f_is_local):
    if type_anno is None:
      return None
    return {str(type_anno)}


class TestTranspiler(transpiler.GenericTranspiler):

  def __init__(self, resolver_type):
    super().__init__()
    self.resolver = resolver_type()

  def get_transformed_name(self, _):
    return 'test_item'

  def transform_ast(self, node, ctx):
    node = qual_names.resolve(node)
    node = activity.resolve(node, ctx)
    graphs = cfg.build(node)
    node = reaching_definitions.resolve(node, ctx, graphs)
    node = reaching_fndefs.resolve(node, ctx, graphs)
    node = type_inference.resolve(node, ctx, graphs, self.resolver)
    return node


class TypeInferenceAnalyzerTest(test.TestCase):

  def assertTypes(self, node, expected):
    if not isinstance(expected, tuple):
      expected = expected,
    self.assertSetEqual(
        set(anno.getanno(node, anno.Static.TYPES)), set(expected))

  def assertClosureTypes(self, node, expected):
    actual = anno.getanno(node, anno.Static.CLOSURE_TYPES)
    actual = {str(k): v for k, v in actual.items()}
    for k, v in expected.items():
      self.assertIn(k, actual)
      self.assertEqual(actual[k], v)

  def test_no_inference_on_unknown_operand_types(self):

    class Resolver(type_inference.Resolver):

      def res_arg(self, ns, types_ns, f_name, name, type_anno, f_is_local):
        return None

    def test_fn(a, b):
      return a < b, a - b

    node, _ = TestTranspiler(Resolver).transform(test_fn, None)
    fn_body = node.body

    # With no information on operand types, the operators will infer nothing.
    self.assertFalse(
        anno.hasanno(fn_body[0].value.elts[0], anno.Static.TYPES))
    self.assertFalse(
        anno.hasanno(fn_body[0].value.elts[1], anno.Static.TYPES))

  def test_resolver_output_checked(self):

    class Resolver(type_inference.Resolver):

      def res_arg(self, ns, types_ns, f_name, name, type_anno, f_is_local):
        return 1

    def test_fn(a):
      del a
      pass

    with self.assertRaisesRegex(ValueError, 'expected to return set'):
      TestTranspiler(Resolver).transform(test_fn, None)

  def test_argument(self):

    test_self = self

    class Resolver(type_inference.Resolver):

      def res_arg(self, ns, types_ns, f_name, name, type_anno, f_is_local):
        test_self.assertFalse(f_is_local)
        if name == qual_names.QN('a'):
          test_self.assertEqual(type_anno, qual_names.QN('int'))
        return {str(name) + '_type'}

    def test_fn(a: int, b):
      return a, b

    node, _ = TestTranspiler(Resolver).transform(test_fn, None)
    fn_body = node.body

    self.assertTypes(fn_body[0].value.elts[0], 'a_type')
    self.assertTypes(fn_body[0].value.elts[1], 'b_type')

  def test_argument_of_local_function(self):

    test_self = self

    class Resolver(type_inference.Resolver):

      def res_arg(self, ns, types_ns, f_name, name, type_anno, f_is_local):
        if f_name == 'test_fn':
          test_self.assertFalse(f_is_local)
          test_self.assertEqual(name, qual_names.QN('a'))
          test_self.assertEqual(type_anno, qual_names.QN('int'))
        elif f_name == 'foo':
          test_self.assertTrue(f_is_local)
          if name == qual_names.QN('x'):
            test_self.assertEqual(type_anno, qual_names.QN('float'))
          elif name == qual_names.QN('y'):
            test_self.assertIsNone(type_anno)
          else:
            test_self.fail('unexpected argument {} for {}'.format(name, f_name))
        else:
          test_self.fail('unexpected function name {}'.format(f_name))
        return {str(name) + '_type'}

    def test_fn(a: int):

      def foo(x: float, y):
        return x, y

      return foo(a, a)

    tr = TestTranspiler(Resolver)
    node, _ = tr.transform(test_fn, None)
    fn_body = node.body

    self.assertTypes(fn_body[0].body[0].value, (('x_type', 'y_type'),))
    self.assertTypes(fn_body[0].body[0].value.elts[0], 'x_type')
    self.assertTypes(fn_body[0].body[0].value.elts[1], 'y_type')

  def test_assign_straightline(self):

    def test_fn(a: int, c: float):
      b = a
      return a, b, c

    node, _ = TestTranspiler(BasicTestResolver).transform(test_fn, None)
    fn_body = node.body

    self.assertTypes(fn_body[0].targets[0], 'int')
    self.assertTypes(fn_body[0].value, 'int')
    self.assertTypes(fn_body[1].value.elts[0], 'int')
    self.assertTypes(fn_body[1].value.elts[1], 'int')
    self.assertTypes(fn_body[1].value.elts[2], 'float')

  def test_expr(self):

    test_self = self

    class Resolver(type_inference.Resolver):

      def res_value(self, ns, value):
        test_self.assertEqual(value, tc.a)
        return {str}

      def res_name(self, ns, types_ns, name):
        test_self.assertEqual(name, qual_names.QN('tc'))
        return {TestClass}, tc

      def res_call(self, ns, types_ns, node, f_type, args, keywords):
        test_self.assertEqual(f_type, (str,))
        return {int}, None

    class TestClass:

      def a(self):
        pass

    tc = TestClass()

    def test_fn():
      tc.a()

    node, _ = TestTranspiler(Resolver).transform(test_fn, None)
    fn_body = node.body

    self.assertEqual(
        anno.getanno(fn_body[0].value.func, anno.Static.VALUE), tc.a)
    self.assertTypes(fn_body[0].value.func, str)
    self.assertTypes(fn_body[0].value, int)
    self.assertTypes(fn_body[0], int)

  def test_assign_overwriting(self):

    def test_fn(a: int, b: float):
      c = a
      c = b
      return c

    node, _ = TestTranspiler(BasicTestResolver).transform(test_fn, None)
    fn_body = node.body

    self.assertTypes(fn_body[0].targets[0], 'int')
    self.assertTypes(fn_body[0].value, 'int')
    self.assertTypes(fn_body[1].targets[0], 'float')
    self.assertTypes(fn_body[1].value, 'float')

  def test_dynamic_attribute_of_static_value(self):

    test_self = self

    class Resolver(type_inference.Resolver):

      def res_value(self, ns, value):
        test_self.assertEqual(value, tc.a)
        return {int}

      def res_name(self, ns, types_ns, name):
        test_self.assertEqual(name, qual_names.QN('tc'))
        return {TestClass}, tc

    class TestClass:

      def __init__(self):
        self.a = 1

    tc = TestClass()

    def test_fn():
      return tc.a

    node, _ = TestTranspiler(Resolver).transform(test_fn, None)
    fn_body = node.body

    self.assertTypes(fn_body[0].value.value, TestClass)
    self.assertTypes(fn_body[0].value, int)
    self.assertIs(anno.getanno(fn_body[0].value.value, anno.Static.VALUE), tc)
    self.assertEqual(anno.getanno(fn_body[0].value, anno.Static.VALUE), tc.a)

  def test_static_attribute_of_typed_value(self):

    test_self = self

    class TestClass:

      a = 1

    tc = TestClass()

    class Resolver(type_inference.Resolver):

      def res_name(self, ns, types_ns, name):
        test_self.assertEqual(name, qual_names.QN('tc'))
        return {TestClass}, None

      def res_value(self, ns, value):
        test_self.assertIs(value, tc.a)
        return {str}

    def test_fn():
      return tc.a

    node, _ = TestTranspiler(Resolver).transform(test_fn, None)
    fn_body = node.body

    self.assertTypes(fn_body[0].value.value, TestClass)
    self.assertTypes(fn_body[0].value, str)  # Resolver is SOT
    self.assertFalse(anno.hasanno(fn_body[0].value.value, anno.Static.VALUE))
    self.assertEqual(anno.getanno(fn_body[0].value, anno.Static.VALUE), 1)

  def test_static_attribute_of_ambiguous_type(self):

    test_self = self

    class TestClass1:

      a = 1

    class TestClass2:

      a = 2

    tc = TestClass1()

    class Resolver(type_inference.Resolver):

      def res_name(self, ns, types_ns, name):
        test_self.assertEqual(name, qual_names.QN('tc'))
        return {TestClass1, TestClass2}, None

      def res_value(self, ns, value):
        test_self.assertIn(value, (1, 2))
        return {str}

    def test_fn():
      return tc.a

    node, _ = TestTranspiler(Resolver).transform(test_fn, None)
    fn_body = node.body

    self.assertTypes(fn_body[0].value.value, (TestClass1, TestClass2))
    self.assertFalse(anno.hasanno(fn_body[0].value, anno.Static.TYPES))
    self.assertFalse(anno.hasanno(fn_body[0].value.value, anno.Static.VALUE))
    self.assertFalse(anno.hasanno(fn_body[0].value, anno.Static.VALUE))

  def test_property_of_typed_value(self):

    test_self = self

    class TestClass:

      @property
      def a(self):
        return 1

    tc = TestClass()

    class Resolver(type_inference.Resolver):

      def res_name(self, ns, types_ns, name):
        test_self.assertEqual(name, qual_names.QN('tc'))
        return {TestClass}, None

      def res_value(self, ns, value):
        test_self.assertIs(value, TestClass.a)
        test_self.assertNotEqual(value, 1)  # Can't evaluate property of class.
        return {property}

    def test_fn():
      return tc.a

    node, _ = TestTranspiler(Resolver).transform(test_fn, None)
    fn_body = node.body

    self.assertTypes(fn_body[0].value.value, TestClass)
    self.assertTypes(fn_body[0].value, property)
    self.assertFalse(anno.hasanno(fn_body[0].value.value, anno.Static.VALUE))
    self.assertEqual(
        anno.getanno(fn_body[0].value, anno.Static.VALUE), TestClass.a)

  def test_dynamic_attribute_of_typed_value(self):

    test_self = self

    class TestClass:

      def __init__(self):
        self.a = 1

    tc = TestClass()

    class Resolver(type_inference.Resolver):

      def res_name(self, ns, types_ns, name):
        test_self.assertEqual(name, qual_names.QN('tc'))
        return {TestClass}, None

    def test_fn():
      return tc.a

    node, _ = TestTranspiler(Resolver).transform(test_fn, None)
    fn_body = node.body

    self.assertTypes(fn_body[0].value.value, TestClass)
    self.assertFalse(anno.hasanno(fn_body[0].value, anno.Static.TYPES))
    self.assertFalse(anno.hasanno(fn_body[0].value.value, anno.Static.VALUE))
    self.assertFalse(anno.hasanno(fn_body[0].value, anno.Static.VALUE))

  def test_external_value(self):

    a = 'foo'

    def test_fn():
      b = a
      return b

    node, _ = TestTranspiler(BasicTestResolver).transform(test_fn, None)
    fn_body = node.body

    self.assertTypes(fn_body[0].targets[0], str)
    self.assertTypes(fn_body[1].value, str)

  def test_external_function(self):

    test_self = self

    class Resolver(type_inference.Resolver):

      def res_name(self, ns, types_ns, name):
        test_self.assertEqual(name, qual_names.QN('g'))
        return {str}, g

      def res_call(self, ns, types_ns, node, f_type, args, keywords):
        test_self.assertEqual(f_type, (str,))
        test_self.assertEqual(
            anno.getanno(node.func, anno.Basic.QN), qual_names.QN('g'))
        return {float}, None

    def g() -> float:
      return 1.0

    def test_fn():
      a = g()
      return a

    node, _ = TestTranspiler(Resolver).transform(test_fn, None)
    fn_body = node.body

    self.assertTypes(fn_body[0].value.func, str)
    self.assertTypes(fn_body[0].targets[0], float)
    self.assertTypes(fn_body[1].value, float)

  def test_external_function_side_effects(self):

    test_self = self

    class Resolver(type_inference.Resolver):

      def res_name(self, ns, types_ns, name):
        test_self.assertEqual(name, qual_names.QN('g'))
        return None, g

      def res_arg(self, ns, types_ns, f_name, name, type_anno, f_is_local):
        return {str(type_anno)}

      def res_call(self, ns, types_ns, node, f_type, args, keywords):
        test_self.assertIsNone(f_type)
        return None, {qual_names.QN('x'): {str}}

    def g():
      # The resolver will pretend that this function has the following body:
      #
      #   nonlocal x
      #   x = 'a'
      pass

    def test_fn(x: int):
      y = x
      g()
      return x, y

    node, _ = TestTranspiler(Resolver).transform(test_fn, None)
    fn_body = node.body

    self.assertTypes(fn_body[0].targets[0], 'int')
    self.assertTypes(fn_body[0].value, 'int')
    self.assertTypes(fn_body[2].value.elts[0], str)
    self.assertTypes(fn_body[2].value.elts[1], 'int')

  def test_local_function_closure(self):

    def test_fn(x: int):

      def foo():
        return x

      foo()

    node, _ = TestTranspiler(BasicTestResolver).transform(test_fn, None)
    fn_body = node.body

    self.assertTypes(fn_body[0].body[0].value, 'int')
    self.assertClosureTypes(fn_body[0], {'x': {'int'}})

  def test_local_function_closure_nested(self):

    def test_fn(x: int):

      def foo():

        def bar():
          return x

        bar()

      foo()

    node, _ = TestTranspiler(BasicTestResolver).transform(test_fn, None)
    fn_body = node.body

    self.assertTypes(fn_body[0].body[0].body[0].value, 'int')
    self.assertClosureTypes(fn_body[0], {'x': {'int'}})
    self.assertClosureTypes(fn_body[0].body[0], {'x': {'int'}})

  def test_local_function_closure_mutable_var(self):

    def test_fn(x: int):

      def foo():
        nonlocal x
        return x

      foo()

    node, _ = TestTranspiler(BasicTestResolver).transform(test_fn, None)
    fn_body = node.body

    self.assertTypes(fn_body[0].body[1].value, 'int')
    self.assertClosureTypes(fn_body[0], {'x': {'int'}})

  def test_local_function_closure_ignored_for_bound_symbols(self):

    def test_fn(x: float):  # pylint:disable=unused-argument

      def foo():
        x = x + 1  # pylint:disable=used-before-assignment

      foo()

    node, _ = TestTranspiler(BasicTestResolver).transform(test_fn, None)
    fn_body = node.body

    self.assertFalse(
        anno.hasanno(fn_body[0].body[0].value.left, anno.Static.TYPES))
    self.assertClosureTypes(fn_body[0], {'x': {'float'}})

  def test_local_function_closure_uses_call_site_types(self):

    def test_fn(x: int):

      def foo():
        return x

      x = 1.0
      foo()

    node, _ = TestTranspiler(BasicTestResolver).transform(test_fn, None)
    fn_body = node.body

    self.assertTypes(fn_body[0].body[0].value, float)
    self.assertTypes(fn_body[1].targets[0], float)
    self.assertClosureTypes(fn_body[0], {'x': {float}})

  def test_local_function_hides_locals(self):

    def test_fn(a: int):  # pylint:disable=unused-argument

      def local_fn(v):
        a = v
        return a

      local_fn(1)

    node, _ = TestTranspiler(BasicTestResolver).transform(test_fn, None)
    fn_body = node.body

    self.assertFalse(
        anno.hasanno(fn_body[0].body[0].targets[0], anno.Static.TYPES))

  def test_local_function_type(self):

    def test_fn(x: int):

      def foo() -> int:
        return x

      foo()

    node, _ = TestTranspiler(BasicTestResolver).transform(test_fn, None)
    fn_body = node.body

    self.assertTypes(fn_body[1].value.func, Callable[[Any], int])
    self.assertTypes(fn_body[1].value, int)
    self.assertTypes(fn_body[1], int)

  def test_side_effects_on_arg_function_closure(self):

    test_self = self

    class Resolver(type_inference.Resolver):

      def res_name(self, ns, types_ns, name):
        test_self.assertEqual(name, qual_names.QN('g'))
        return {Callable[[Callable], None]}, g

      def res_value(self, ns, value):
        test_self.assertEqual(value, 1.0)
        return {float}

      def res_arg(self, ns, types_ns, f_name, name, type_anno, f_is_local):
        return {str(type_anno)}

      def res_call(self, ns, types_ns, node, f_type, args, keywords):
        test_self.assertEqual(node.func.id, 'g')
        test_self.assertEqual(f_type, (Callable[[Callable], None],))
        return None, {qual_names.QN('x'): {str}}

    def g(foo):
      # The resolver will convey that this function has the following body:
      #
      #   nonlocal x
      #   x = 'a'
      #   foo()
      del foo
      pass

    def test_fn(x: int):  # pylint:disable=unused-argument

      def foo():
        return x

      x = 1.0
      g(foo)

    node, _ = TestTranspiler(Resolver).transform(test_fn, None)
    fn_body = node.body

    self.assertTypes(fn_body[0].body[0].value, str)

  def test_subscript(self):

    test_self = self

    class Resolver(type_inference.Resolver):

      def res_arg(self, ns, types_ns, f_name, name, type_anno, f_is_local):
        return {list}

      def res_value(self, ns, value):
        return {int}

      def res_slice(self, ns, types_ns, node, value, slice_):
        test_self.assertSetEqual(value, {list})
        test_self.assertSetEqual(slice_, {int})
        return {str}

    def test_fn(a):
      return a[1]

    node, _ = TestTranspiler(Resolver).transform(test_fn, None)
    fn_body = node.body

    self.assertTypes(fn_body[0].value, str)
    self.assertTypes(fn_body[0].value.value, list)
    self.assertTypes(fn_body[0].value.slice, int)

  def test_tuple_unpacking(self):

    test_self = self

    class Resolver(type_inference.Resolver):

      def res_arg(self, ns, types_ns, f_name, name, type_anno, f_is_local):
        return {list}

      def res_value(self, ns, value):
        return {int}

      def res_slice(self, ns, types_ns, node_or_slice, value, slice_):
        test_self.assertIn(node_or_slice, (0, 1))
        test_self.assertSetEqual(value, {list})
        test_self.assertSetEqual(slice_, {int})
        if node_or_slice == 0:
          return {float}
        else:
          return {str}

    def test_fn(t):
      a, b = t
      return a, b

    node, _ = TestTranspiler(Resolver).transform(test_fn, None)
    fn_body = node.body

    self.assertTypes(fn_body[1].value, ((float, str),))
    self.assertTypes(fn_body[1].value.elts[0], float)
    self.assertTypes(fn_body[1].value.elts[1], str)

  def test_compare(self):

    test_self = self

    class Resolver(type_inference.Resolver):

      def res_arg(self, ns, types_ns, f_name, name, type_anno, f_is_local):
        return {int}

      def res_compare(self, ns, types_ns, node, left, right):
        test_self.assertSetEqual(left, {int})
        test_self.assertListEqual(right, [{int}])
        return {bool}

    def test_fn(a, b):
      return a < b

    node, _ = TestTranspiler(Resolver).transform(test_fn, None)
    fn_body = node.body

    self.assertTypes(fn_body[0].value, bool)
    self.assertTypes(fn_body[0].value.left, int)
    self.assertTypes(fn_body[0].value.comparators[0], int)

  def test_binop(self):

    test_self = self

    class Resolver(type_inference.Resolver):

      def res_arg(self, ns, types_ns, f_name, name, type_anno, f_is_local):
        return {list}

      def res_binop(self, ns, types_ns, node, left, right):
        test_self.assertSetEqual(left, {list})
        test_self.assertSetEqual(right, {list})
        return {float}

    def test_fn(a, b):
      return a @ b

    node, _ = TestTranspiler(Resolver).transform(test_fn, None)
    fn_body = node.body

    self.assertTypes(fn_body[0].value, float)
    self.assertTypes(fn_body[0].value.left, list)
    self.assertTypes(fn_body[0].value.right, list)

  def test_unop(self):

    class Resolver(type_inference.Resolver):

      def res_arg(self, ns, types_ns, f_name, name, type_anno, f_is_local):
        return {list}

      def res_unop(self, ns, types_ns, node, opnd):
        return {float}

    def test_fn(a):
      return -a

    node, _ = TestTranspiler(Resolver).transform(test_fn, None)
    fn_body = node.body

    self.assertTypes(fn_body[0].value, float)
    self.assertTypes(fn_body[0].value.operand, list)

  def test_tuple_literal(self):

    class Resolver(type_inference.Resolver):

      def res_arg(self, ns, types_ns, f_name, name, type_anno, f_is_local):
        return {int}

    def test_fn(a, b):
      return a, b

    node, _ = TestTranspiler(Resolver).transform(test_fn, None)
    fn_body = node.body

    self.assertTypes(fn_body[0].value, ((int, int),))
    self.assertTypes(fn_body[0].value.elts[0], int)
    self.assertTypes(fn_body[0].value.elts[1], int)

  def test_list_literal(self):

    class Resolver(type_inference.Resolver):

      def res_arg(self, ns, types_ns, f_name, name, type_anno, f_is_local):
        return {int}

      def res_list_literal(self, ns, elt_types):
        all_types = set()
        for s in elt_types:
          all_types |= s
        return {List[t] for t in all_types}

    def test_fn(a, b):
      return [a, b]

    node, _ = TestTranspiler(Resolver).transform(test_fn, None)
    fn_body = node.body

    self.assertTypes(fn_body[0].value, List[int])
    self.assertTypes(fn_body[0].value.elts[0], int)
    self.assertTypes(fn_body[0].value.elts[1], int)

  def test_tuple_unpacking_syntactic(self):

    test_self = self

    class Resolver(type_inference.Resolver):

      def res_arg(self, ns, types_ns, f_name, name, type_anno, f_is_local):
        if name == qual_names.QN('a'):
          return {int}
        else:
          return {float}

      def res_value(self, ns, value):
        test_self.assertIn(value, (0, 1))
        return int

      def res_slice(self, ns, types_ns, node_or_slice, value, slice_):
        test_self.assertIn(node_or_slice, (0, 1))
        test_self.assertSetEqual(value, {(int, float)})
        test_self.assertEqual(slice_, int)
        return {t[node_or_slice] for t in value}

    def test_fn(a, b):
      c, d = a, b
      return c, d

    node, _ = TestTranspiler(Resolver).transform(test_fn, None)
    fn_body = node.body

    self.assertTypes(fn_body[1].value, ((int, float),))
    self.assertTypes(fn_body[1].value.elts[0], int)
    self.assertTypes(fn_body[1].value.elts[1], float)

  def test_tuple_unpacking_operational(self):

    test_self = self

    class Resolver(type_inference.Resolver):

      def res_arg(self, ns, types_ns, f_name, name, type_anno, f_is_local):
        return {(int, float)}

      def res_value(self, ns, value):
        test_self.assertIn(value, (0, 1))
        return int

      def res_slice(self, ns, types_ns, node_or_slice, value, slice_):
        test_self.assertIn(node_or_slice, (0, 1))
        test_self.assertSetEqual(value, {(int, float)})
        test_self.assertEqual(slice_, int)
        return {t[node_or_slice] for t in value}

    def test_fn(a):
      c, d = a
      return c, d

    node, _ = TestTranspiler(Resolver).transform(test_fn, None)
    fn_body = node.body

    self.assertTypes(fn_body[1].value, ((int, float),))
    self.assertTypes(fn_body[1].value.elts[0], int)
    self.assertTypes(fn_body[1].value.elts[1], float)

  def test_list_expansion_syntactic(self):

    test_self = self

    class Resolver(type_inference.Resolver):

      def res_arg(self, ns, types_ns, f_name, name, type_anno, f_is_local):
        if name == qual_names.QN('a'):
          return {int}
        else:
          return {float}

      def res_value(self, ns, value):
        test_self.assertIn(value, (0, 1))
        return int

      def res_slice(self, ns, types_ns, node_or_slice, value, slice_):
        test_self.assertIn(node_or_slice, (0, 1))
        test_self.assertSetEqual(value, {(int, float)})
        test_self.assertEqual(slice_, int)
        return {t[node_or_slice] for t in value}

    def test_fn(a, b):
      [c, d] = a, b
      return c, d

    node, _ = TestTranspiler(Resolver).transform(test_fn, None)
    fn_body = node.body

    # TODO(mdan): Whether it's List or Tuple might be open for interpretation.
    self.assertTypes(fn_body[1].value, ((int, float),))
    self.assertTypes(fn_body[1].value.elts[0], int)
    self.assertTypes(fn_body[1].value.elts[1], float)

  def test_list_expansion_operational(self):

    test_self = self

    class Resolver(type_inference.Resolver):

      def res_arg(self, ns, types_ns, f_name, name, type_anno, f_is_local):
        if name == qual_names.QN('a'):
          return {int}
        else:
          return {float}

      def res_value(self, ns, value):
        test_self.assertIn(value, (0, 1))
        return int

      def res_slice(self, ns, types_ns, node_or_slice, value, slice_):
        test_self.assertIn(node_or_slice, (0, 1))
        test_self.assertSetEqual(value, {(int, float)})
        test_self.assertEqual(slice_, int)
        return {t[node_or_slice] for t in value}

    def test_fn(a, b):
      [c, d] = a, b
      return c, d

    node, _ = TestTranspiler(Resolver).transform(test_fn, None)
    fn_body = node.body

    # TODO(mdan): Whether it's List or Tuple might be open for interpretation.
    self.assertTypes(fn_body[1].value, ((int, float),))
    self.assertTypes(fn_body[1].value.elts[0], int)
    self.assertTypes(fn_body[1].value.elts[1], float)


if __name__ == '__main__':
  test.main()
