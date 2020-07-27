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

from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import cfg
from tensorflow.python.autograph.pyct import qual_names
from tensorflow.python.autograph.pyct import transpiler
from tensorflow.python.autograph.pyct.static_analysis import activity
from tensorflow.python.autograph.pyct.static_analysis import reaching_definitions
from tensorflow.python.autograph.pyct.static_analysis import reaching_fndefs
from tensorflow.python.autograph.pyct.static_analysis import type_inference
from tensorflow.python.platform import test


class TestResolver(type_inference.Resolver):
  """A very basic resolver for testing."""

  def res_name(self, ns, name):
    return {type(ns[str(name)])}

  def res_value(self, ns, value):
    del ns
    return {type(value)}

  def res_call(self, ns, name, target, args, keywords, starargs, kwargs):
    name_str = str(name)
    if name_str in ns:
      return {ns[name_str].__annotations__['return']}
    if target is None:
      return {'unk_{}'.format(name_str)}
    return {'{}_{}'.format(list(target)[0], name_str)}

  def res_arg(self, ns, f_name, arg_name, type_anno):
    if f_name == 'magic_no_types':
      return None
    if type_anno is not None:
      return {{'int': int, 'float': float}[str(type_anno)]}
    return {'{}_{}'.format(f_name, arg_name)}


class TestTranspiler(transpiler.GenericTranspiler):

  def __init__(self):
    super().__init__()
    self.resolver = TestResolver()

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
    self.assertDictEqual(actual, expected)

  def test_argument(self):

    def test_fn(a: int, b):
      return a, b

    node, _ = TestTranspiler().transform(test_fn, None)
    fn_body = node.body

    self.assertTypes(fn_body[0].value.elts[0], int)
    self.assertTypes(fn_body[0].value.elts[1], 'test_fn_b')

  def test_argument_of_local_function(self):

    def test_fn(a: int):

      def foo(x: float):
        return x

      return foo(a)

    tr = TestTranspiler()
    node, _ = tr.transform(test_fn, None)
    fn_body = node.body

    self.assertTypes(fn_body[0].body[0].value, float)
    self.assertClosureTypes(fn_body[0], {'a': {int}})

  def test_straightline_assignment(self):

    def test_fn(a: int, c):
      b = a
      return a, b, c

    node, _ = TestTranspiler().transform(test_fn, None)
    fn_body = node.body

    self.assertTypes(fn_body[0].targets[0], int)
    self.assertTypes(fn_body[0].value, int)
    self.assertTypes(fn_body[1].value.elts[0], int)
    self.assertTypes(fn_body[1].value.elts[1], int)
    self.assertTypes(fn_body[1].value.elts[2], 'test_fn_c')

  def test_assignment_overwrite(self):

    def test_fn(a: int, b: float):
      c = a
      c = b
      return c

    node, _ = TestTranspiler().transform(test_fn, None)
    fn_body = node.body

    self.assertTypes(fn_body[0].targets[0], int)
    self.assertTypes(fn_body[0].value, int)
    self.assertTypes(fn_body[1].targets[0], float)
    self.assertTypes(fn_body[1].value, float)

  def test_external_value(self):

    a = 'foo'

    def test_fn():
      b = a
      return b

    node, _ = TestTranspiler().transform(test_fn, None)
    fn_body = node.body

    self.assertTypes(fn_body[0].targets[0], str)
    self.assertTypes(fn_body[1].value, str)

  def test_external_function(self):

    def g() -> float:
      return 1.0

    def test_fn():
      a = g()
      return a

    node, _ = TestTranspiler().transform(test_fn, None)
    fn_body = node.body

    self.assertTypes(fn_body[0].targets[0], float)
    self.assertTypes(fn_body[1].value, float)

  def test_local_function_closure(self):

    def test_fn(x: int):

      def foo():
        return x

      foo()

    node, _ = TestTranspiler().transform(test_fn, None)
    fn_body = node.body

    self.assertTypes(fn_body[0].body[0].value, int)
    self.assertClosureTypes(fn_body[0], {'x': {int}})

  def test_local_function_closure_ignored_for_bound_symbols(self):

    def test_fn(x: int):  # pylint:disable=unused-argument

      def foo():
        x = x + 1  # pylint:disable=used-before-assignment

      foo()

    node, _ = TestTranspiler().transform(test_fn, None)
    fn_body = node.body

    self.assertFalse(
        anno.hasanno(fn_body[0].body[0].value.left, anno.Static.TYPES))
    self.assertClosureTypes(fn_body[0], {'x': {int}})

  def test_local_function_closure_uses_call_site_types(self):

    def test_fn(x: int):

      def foo():
        return x

      x = 1.0
      foo()

    node, _ = TestTranspiler().transform(test_fn, None)
    fn_body = node.body

    self.assertTypes(fn_body[0].body[0].value, float)
    self.assertTypes(fn_body[1].targets[0], float)
    self.assertClosureTypes(fn_body[0], {'x': {float}})

  def test_subscript(self):

    def test_fn(a):
      return a[1]

    node, _ = TestTranspiler().transform(test_fn, None)
    fn_body = node.body

    self.assertTypes(fn_body[0].value, 'test_fn_a___getitem__')
    self.assertTypes(fn_body[0].value.value, 'test_fn_a')
    self.assertTypes(fn_body[0].value.slice.value, int)

  def test_compare(self):

    def test_fn(a, b):
      return a < b

    node, _ = TestTranspiler().transform(test_fn, None)
    fn_body = node.body

    self.assertTypes(fn_body[0].value, 'test_fn_a___lt__')
    self.assertTypes(fn_body[0].value.left, 'test_fn_a')
    self.assertTypes(fn_body[0].value.comparators[0], 'test_fn_b')

  def test_binop(self):

    def test_fn(a, b):
      return a @ b

    node, _ = TestTranspiler().transform(test_fn, None)
    fn_body = node.body

    self.assertTypes(fn_body[0].value, 'test_fn_a___matmul__')
    self.assertTypes(fn_body[0].value.left, 'test_fn_a')
    self.assertTypes(fn_body[0].value.right, 'test_fn_b')

  def test_no_inference_on_unknown_operand_types(self):

    # No information on types of a and b, see TestResolver.
    def magic_no_types(a, b):
      return a < b, a - b

    node, _ = TestTranspiler().transform(magic_no_types, None)
    fn_body = node.body

    # With no information on operand types, the operators will assert nothing.
    self.assertFalse(
        anno.hasanno(fn_body[0].value.elts[0], anno.Static.TYPES))
    self.assertFalse(
        anno.hasanno(fn_body[0].value.elts[1], anno.Static.TYPES))


if __name__ == '__main__':
  test.main()
