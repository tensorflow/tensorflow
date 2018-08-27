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
"""Tests for decorators module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import wraps

from tensorflow.contrib.autograph.converters import decorators
from tensorflow.contrib.autograph.core import converter_testing
from tensorflow.contrib.autograph.pyct import compiler
from tensorflow.contrib.autograph.pyct import transformer
from tensorflow.python.platform import test


# The Python parser only briefly captures decorators into the AST.
# The interpreter desugars them on load, and the decorated function loses any
# trace of the decorator (which is normally what you would expect, since
# they are meant to be transparent).
# However, decorators are still visible when you analyze the function
# from inside a decorator, before it was applied - as is the case
# with our conversion decorators.


def simple_decorator(f):
  return lambda a: f(a) + 1


def self_transform_decorator(transform):

  def decorator(f):
    @wraps(f)
    def wrapper(*args):
      # This removing wrapper is defined in the test below. This setup is so
      # intricate in order to simulate how we use the transformer in practice.
      transformed_f = transform(f, (self_transform_decorator,))
      return transformed_f(*args) + 1
    return wrapper
  return decorator


class DecoratorsTest(converter_testing.TestCase):

  def _transform(self, f, autograph_decorators):
    namespace = {
        'self_transform_decorator': self_transform_decorator,
        'simple_decorator': simple_decorator,
        'converter_testing': converter_testing,
    }
    node, ctx = self.prepare(
        f,
        namespace,
        recursive=False,
        autograph_decorators=autograph_decorators)
    node = decorators.transform(node, ctx)
    import_line = '\n'.join(ctx.program.additional_imports)
    result, _ = compiler.ast_to_object(node, source_prefix=import_line)
    return getattr(result, f.__name__)

  def test_noop(self):

    def test_fn(a):
      return a

    with self.converted(test_fn, decorators, {}) as result:
      self.assertEqual(1, result.test_fn(1))

  def test_function(self):

    @self_transform_decorator(self._transform)
    def test_fn(a):
      return a

    # 2 = 1 (a) + 1 (decorator applied exactly once)
    self.assertEqual(2, test_fn(1))

  def test_method(self):

    class TestClass(object):

      @self_transform_decorator(self._transform)
      def test_fn(self, a):
        return a

    # 2 = 1 (a) + 1 (decorator applied exactly once)
    self.assertEqual(2, TestClass().test_fn(1))

  def test_multiple_decorators(self):

    class TestClass(object):

      # Note that reversing the order of this two doesn't work.
      @classmethod
      @self_transform_decorator(self._transform)
      def test_fn(cls, a):
        return a

    # 2 = 1 (a) + 1 (decorator applied exactly once)
    self.assertEqual(2, TestClass.test_fn(1))

  def test_nested_decorators_local(self):

    @self_transform_decorator(self._transform)
    def test_fn(a):
      @simple_decorator
      def inner_fn(b):
        return b + 11
      return inner_fn(a)

    # Expected to fail because simple_decorator could not be imported.
    with self.assertRaises(transformer.AutographParseError):
      test_fn(1)

  def test_nested_decorators_imported(self):

    @self_transform_decorator(self._transform)
    def test_fn(a):

      @converter_testing.imported_decorator
      def inner_fn(b):
        return b + 11

      return inner_fn(a)

    # 14 = 1 (a) + 1 (simple_decorator) + 11 (inner_fn)
    self.assertEqual(14, test_fn(1))


if __name__ == '__main__':
  test.main()
