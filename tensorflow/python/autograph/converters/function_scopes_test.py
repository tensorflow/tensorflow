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
"""Tests for function_scopes module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.autograph.converters import function_scopes
from tensorflow.python.autograph.core import converter_testing
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test


class FunctionBodyTransformerTest(converter_testing.TestCase):

  @test_util.run_deprecated_v1
  def test_basic(self):

    def test_fn(l):
      """Docstring."""
      a = 1
      l += a
      return l

    with self.converted(test_fn, function_scopes, {}) as result:
      result_op = result.test_fn(constant_op.constant(1))
      self.assertIn('test_fn/', result_op.op.name)
      self.assertEqual('Docstring.', result.test_fn.__doc__)

  @test_util.run_deprecated_v1
  def test_multiline_docstring(self):

    tf = None

    def test_fn():
      """First sentence.

      Second sentence.
      """
      return tf.constant(1)

    with self.converted(test_fn, function_scopes, {},
                        constant_op.constant) as result:
      result_op = result.test_fn()
      self.assertIn('test_fn/', result_op.op.name)
      self.assertIn('First sentence.', result.test_fn.__doc__)
      self.assertIn('Second sentence.', result.test_fn.__doc__)

  @test_util.run_deprecated_v1
  def test_nested_functions(self):

    def test_fn(l):

      def inner_fn(i):
        return i + 1

      l += 1
      return l, inner_fn(l)

    with self.converted(test_fn, function_scopes, {}, ops.name_scope) as result:
      first, second = result.test_fn(constant_op.constant(1))
      self.assertIn('test_fn/', first.op.name)
      self.assertNotIn('inner_fn', first.op.name)
      self.assertIn('test_fn/inner_fn/', second.op.name)

  @test_util.run_deprecated_v1
  def test_method(self):

    class TestClass(object):

      def test_fn(self, l):

        def inner_fn(i):
          return i + 1

        l += 1
        return l, inner_fn(l)

    ns = {'TestClass': TestClass}
    node, ctx = self.prepare(TestClass, ns)
    node = function_scopes.transform(node, ctx)

    with self.compiled(node, {}, ops.name_scope) as result:
      first, second = result.TestClass().test_fn(constant_op.constant(1))
      self.assertIn('TestClass/test_fn/', first.op.name)
      self.assertNotIn('inner_fn', first.op.name)
      self.assertIn('TestClass/test_fn/inner_fn/', second.op.name)


if __name__ == '__main__':
  test.main()
