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
from tensorflow.python.autograph.pyct.static_analysis import type_inference
from tensorflow.python.platform import test


class TestResolver(type_inference.Resolver):

  def resolve_external_name(self, ns, name):
    return {type(ns[str(name)])}

  def resolve_external_call(self, ns, f_name):
    return {ns[str(f_name)].__annotations__['return']}

  def resolve_external_arg(self, ns, f_name, arg_name, type_anno):
    if type_anno is not None:
      return {{'int': int, 'float': float}[str(type_anno)]}
    return {'{}_{}'.format(f_name, arg_name)}


class TestTranspiler(transpiler.GenericTranspiler):

  def get_transformed_name(self, _):
    return 'test_item'

  def transform_ast(self, node, ctx):
    node = qual_names.resolve(node)
    node = activity.resolve(node, ctx)
    graphs = cfg.build(node)
    node = type_inference.resolve(node, ctx, graphs, TestResolver())
    return node


class TypeInferenceAnalyzerTest(test.TestCase):

  def assertTypes(self, node, expected):
    if not isinstance(expected, tuple):
      expected = expected,
    self.assertSetEqual(
        set(anno.getanno(node, anno.Static.TYPES)), set(expected))

  def test_argument(self):

    def test_fn(a: int, b):
      return a, b

    node, _ = TestTranspiler().transform(test_fn, None)
    fn_body = node.body

    self.assertTypes(fn_body[0].value.elts[0], int)
    self.assertTypes(fn_body[0].value.elts[1], 'test_fn_b')

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


if __name__ == '__main__':
  test.main()
