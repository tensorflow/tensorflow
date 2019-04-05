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
"""Tests for conversion module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gast

from tensorflow.python.autograph import utils
from tensorflow.python.autograph.core import converter
from tensorflow.python.autograph.impl import api
from tensorflow.python.autograph.impl import conversion
from tensorflow.python.autograph.pyct import compiler
from tensorflow.python.framework import constant_op
from tensorflow.python.keras.engine import training
from tensorflow.python.platform import test


class ConversionTest(test.TestCase):

  def _simple_program_ctx(self):
    return converter.ProgramContext(
        options=converter.ConversionOptions(recursive=True),
        autograph_module=api)

  def test_is_whitelisted_for_graph(self):

    def test_fn():
      return constant_op.constant(1)

    self.assertFalse(conversion.is_whitelisted_for_graph(test_fn))
    self.assertTrue(conversion.is_whitelisted_for_graph(utils))
    self.assertTrue(conversion.is_whitelisted_for_graph(constant_op.constant))

  def test_convert_entity_to_ast_unsupported_types(self):
    with self.assertRaises(NotImplementedError):
      program_ctx = self._simple_program_ctx()
      conversion.convert_entity_to_ast('dummy', program_ctx)

  def test_convert_entity_to_ast_callable(self):
    b = 2

    def f(a):
      return a + b

    program_ctx = self._simple_program_ctx()
    nodes, name, info = conversion.convert_entity_to_ast(f, program_ctx)
    fn_node, = nodes
    self.assertIsInstance(fn_node, gast.FunctionDef)
    self.assertEqual('tf__f', name)
    self.assertIs(info.namespace['b'], b)

  def test_convert_entity_to_ast_function_with_defaults(self):
    b = 2
    c = 1

    def f(a, d=c + 1):
      return a + b + d

    program_ctx = self._simple_program_ctx()
    nodes, name, _ = conversion.convert_entity_to_ast(f, program_ctx)
    fn_node, = nodes
    self.assertIsInstance(fn_node, gast.FunctionDef)
    self.assertEqual('tf__f', name)
    self.assertEqual(
        compiler.ast_to_source(fn_node.args.defaults[0]).strip(), 'None')

  def test_convert_entity_to_ast_call_tree(self):

    def g(a):
      return a

    def f(a):
      return g(a)

    program_ctx = self._simple_program_ctx()
    nodes, _, _ = conversion.convert_entity_to_ast(f, program_ctx)
    f_node, = nodes
    self.assertEqual('tf__f', f_node.name)

  def test_convert_entity_to_ast_class_hierarchy(self):

    class TestBase(object):

      def __init__(self, x='base'):
        self.x = x

      def foo(self):
        return self.x

      def bar(self):
        return self.x

    class TestSubclass(TestBase):

      def __init__(self, y):
        super(TestSubclass, self).__init__('sub')
        self.y = y

      def foo(self):
        return self.y

      def baz(self):
        return self.y

    program_ctx = self._simple_program_ctx()
    with self.assertRaisesRegex(NotImplementedError, 'classes.*whitelisted'):
      conversion.convert_entity_to_ast(TestSubclass, program_ctx)

  def test_convert_entity_to_ast_class_hierarchy_whitelisted(self):

    class TestSubclass(training.Model):

      def __init__(self, y):
        super(TestSubclass, self).__init__()
        self.built = False

      def call(self, x):
        return 3 * x

    program_ctx = self._simple_program_ctx()
    (import_node, class_node), name, _ = conversion.convert_entity_to_ast(
        TestSubclass, program_ctx)
    self.assertEqual(import_node.names[0].name, 'Model')
    self.assertEqual(name, 'TfTestSubclass')
    self.assertEqual(class_node.name, 'TfTestSubclass')

  def test_convert_entity_to_ast_lambda(self):
    b = 2
    f = lambda x: b * x if x > 0 else -x

    program_ctx = self._simple_program_ctx()
    (fn_node,), name, entity_info = conversion.convert_entity_to_ast(
        f, program_ctx)
    self.assertIsInstance(fn_node, gast.Assign)
    self.assertIsInstance(fn_node.value, gast.Lambda)
    self.assertEqual('tf__lambda', name)
    self.assertIs(entity_info.namespace['b'], b)

  def test_convert_entity_to_ast_multiple_lambdas(self):
    a, b = 1, 2
    f, _ = (lambda x: a * x, lambda y: b * y)

    program_ctx = self._simple_program_ctx()
    (fn_node,), name, entity_info = conversion.convert_entity_to_ast(
        f, program_ctx)
    self.assertIsInstance(fn_node, gast.Assign)
    self.assertIsInstance(fn_node.value, gast.Lambda)
    self.assertEqual('tf__lambda', name)
    self.assertIs(entity_info.namespace['a'], a)

  def test_convert_entity_to_ast_multiple_lambdas_ambiguous_definitions(self):
    a, b = 1, 2
    f, _ = (lambda x: a * x, lambda x: b * x)

    program_ctx = self._simple_program_ctx()
    with self.assertRaises(ValueError):
      conversion.convert_entity_to_ast(f, program_ctx)

  def test_convert_entity_to_ast_lambda_code_with_garbage(self):
    # pylint:disable=g-long-lambda
    f = (  # intentional wrap
        lambda x: (
            x  # intentional wrap
            + 1),)[0]
    # pylint:enable=g-long-lambda

    program_ctx = self._simple_program_ctx()
    (fn_node,), name, _ = conversion.convert_entity_to_ast(f, program_ctx)
    self.assertIsInstance(fn_node, gast.Assign)
    self.assertIsInstance(fn_node.value, gast.Lambda)
    self.assertEqual('tf__lambda', name)

  def test_convert_entity_to_ast_nested_functions(self):
    b = 2

    def f(x):

      def g(x):
        return b * x

      return g(x)

    program_ctx = self._simple_program_ctx()
    (fn_node,), name, entity_info = conversion.convert_entity_to_ast(
        f, program_ctx)
    self.assertIsInstance(fn_node, gast.FunctionDef)
    self.assertEqual(fn_node.name, 'tf__f')
    self.assertEqual('tf__f', name)
    self.assertIs(entity_info.namespace['b'], b)


if __name__ == '__main__':
  test.main()
