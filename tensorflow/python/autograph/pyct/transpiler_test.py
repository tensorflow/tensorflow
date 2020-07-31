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
"""Tests for transpiler module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import threading

import gast

from tensorflow.python.autograph.pyct import transformer
from tensorflow.python.autograph.pyct import transpiler
from tensorflow.python.platform import test


class FlipSignTransformer(transformer.Base):

  def visit_BinOp(self, node):
    if isinstance(node.op, gast.Add):
      node.op = gast.Sub()
    return self.generic_visit(node)


class TestTranspiler(transpiler.PyToPy):

  def get_caching_key(self, ctx):
    del ctx
    return 0

  def get_extra_locals(self):
    return {}

  def transform_ast(self, node, ctx):
    return FlipSignTransformer(ctx).visit(node)


global_var_for_test_global = 1
global_var_for_test_namespace_collisions = object()


class PyToPyTest(test.TestCase):

  def test_basic(self):
    def f(a):
      return a + 1

    tr = TestTranspiler()
    f, _, _ = tr.transform(f, None)

    self.assertEqual(f(1), 0)

  def test_closure(self):
    b = 1

    def f(a):
      return a + b

    tr = TestTranspiler()
    f, _, _ = tr.transform(f, None)

    self.assertEqual(f(1), 0)
    b = 2
    self.assertEqual(f(1), -1)

  def test_global(self):
    def f(a):
      return a + global_var_for_test_global

    tr = TestTranspiler()
    f, _, _ = tr.transform(f, None)

    global global_var_for_test_global
    global_var_for_test_global = 1
    self.assertEqual(f(1), 0)
    global_var_for_test_global = 2
    self.assertEqual(f(1), -1)

  def test_defaults(self):
    b = 2
    c = 1

    def f(a, d=c + 1):
      return a + b + d

    tr = TestTranspiler()
    f, _, _ = tr.transform(f, None)

    self.assertEqual(f(1), 1 - 2 - 2)
    c = 0
    self.assertEqual(f(1), 1 - 2 - 2)  # Defaults are evaluated at definition.
    b = 1
    self.assertEqual(f(1), 1 - 2 - 1)

  def test_call_tree(self):

    def g(a):
      return a + 1

    def f(a):
      return g(a) + 1

    tr = TestTranspiler()
    f, _, _ = tr.transform(f, None)

    self.assertEqual(f(1), 1 - 1 + 1)  # Only f is converted.

  def test_lambda(self):
    b = 2
    f = lambda x: (b + (x if x > 0 else -x))

    tr = TestTranspiler()
    f, _, _ = tr.transform(f, None)

    self.assertEqual(f(1), 2 - 1)
    self.assertEqual(f(-1), 2 - 1)

    b = 3

    self.assertEqual(f(1), 3 - 1)
    self.assertEqual(f(-1), 3 - 1)

  def test_multiple_lambdas(self):
    a, b = 1, 2
    # This can be disambiguated by the argument names.
    f, _ = (lambda x: a + x, lambda y: b * y)

    tr = TestTranspiler()
    f, _, _ = tr.transform(f, None)

    self.assertEqual(f(1), 1 - 1)

  def test_nested_functions(self):
    b = 2

    def f(x):

      def g(x):
        return b + x

      return g(x)

    tr = TestTranspiler()
    f, _, _ = tr.transform(f, None)

    self.assertEqual(f(1), 2 - 1)

  def test_nested_lambda(self):
    b = 2

    def f(x):
      g = lambda x: b + x
      return g(x)

    tr = TestTranspiler()
    f, _, _ = tr.transform(f, None)

    self.assertEqual(f(1), 2 - 1)

  def test_concurrency(self):

    def f():
      pass

    outputs = []

    tr = TestTranspiler()
    # Note: this is not a test, it's a required invariant.
    assert tr.get_caching_key(None) == tr.get_caching_key(None)

    def conversion_thread():
      _, mod, _ = tr.transform(f, None)
      outputs.append(mod.__name__)

    threads = tuple(
        threading.Thread(target=conversion_thread) for _ in range(10))
    for t in threads:
      t.start()
    for t in threads:
      t.join()

    # Races would potentially create multiple functions / modules
    # (non-deterministically, but with high likelihood).
    self.assertEqual(len(set(outputs)), 1)

  def test_reentrance(self):

    def test_fn():
      return 1 + 1

    class ReentrantTranspiler(transpiler.PyToPy):

      def __init__(self):
        super(ReentrantTranspiler, self).__init__()
        self._recursion_depth = 0

      def get_caching_key(self, ctx):
        del ctx
        return 0

      def get_extra_locals(self):
        return {}

      def transform_ast(self, node, ctx):
        self._recursion_depth += 1
        if self._recursion_depth < 2:
          self.transform(test_fn, None)
        return FlipSignTransformer(ctx).visit(node)

    tr = ReentrantTranspiler()

    f, _, _ = tr.transform(test_fn, None)
    self.assertEqual(f(), 0)

  def test_namespace_collisions_avoided(self):

    class TestClass(object):

      def global_var_for_test_namespace_collisions(self):
        return global_var_for_test_namespace_collisions

    tr = TestTranspiler()
    obj = TestClass()

    f, _, _ = tr.transform(
        obj.global_var_for_test_namespace_collisions, None)
    self.assertIs(f(obj), global_var_for_test_namespace_collisions)


if __name__ == '__main__':
  test.main()
