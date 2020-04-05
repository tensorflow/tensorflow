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
"""Tests for functions module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.autograph.converters import functions
from tensorflow.python.autograph.core import ag_ctx
from tensorflow.python.autograph.core import converter
from tensorflow.python.autograph.core import converter_testing
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test


class FunctionTransformer(converter_testing.TestCase):

  @test_util.run_deprecated_v1
  def test_basic(self):

    def test_fn(l):
      """Docstring."""
      a = 1
      l += a
      return l

    with self.converted(test_fn, functions, {}) as result:
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

    with self.converted(test_fn, functions, {},
                        (constant_op.constant,)) as result:
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

    with self.converted(test_fn, functions, {},
                        (ops.name_scope,)) as result:
      first, second = result.test_fn(constant_op.constant(1))
      self.assertIn('test_fn/', first.op.name)
      self.assertNotIn('inner_fn', first.op.name)
      self.assertIn('test_fn/inner_fn/', second.op.inputs[0].name)

  @test_util.run_deprecated_v1
  def test_conversion_context_preserves_in_inner_functions(self):

    def inner_fn_callee():
      self.assertEqual(
          ag_ctx.control_status_ctx().status, ag_ctx.Status.DISABLED)

    def test_fn():
      def inner_fn():
        inner_fn_callee()
      with ag_ctx.ControlStatusCtx(
          ag_ctx.Status.DISABLED, converter.ConversionOptions(recursive=True)):
        inner_fn()

    ns = {
        'inner_fn_callee': inner_fn_callee,
        'ag_ctx': ag_ctx,
        'converter': converter
    }
    with self.converted(test_fn, functions, ns) as result:
      result.test_fn()

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
    node = functions.transform(node, ctx)

    with self.compiled(node, {}, (ops.name_scope,)) as result:
      first, second = result.TestClass().test_fn(constant_op.constant(1))
      self.assertIn('test_fn/', first.op.name)
      self.assertNotIn('inner_fn', first.op.name)
      self.assertIn('test_fn/inner_fn/', second.op.inputs[0].name)

  def test_lambda_in_return_value(self):

    def test_fn():
      return lambda x: x + 1

    with self.converted(test_fn, functions, {}) as result:
      result_l = result.test_fn()
      self.assertTrue(result_l.fake_autograph_artifact)


if __name__ == '__main__':
  test.main()
