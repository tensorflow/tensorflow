# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for py_builtins_py3 module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.autograph.core import converter
from tensorflow.python.autograph.core import function_wrappers
from tensorflow.python.autograph.operators import py_builtins
from tensorflow.python.platform import test


class TestBaseClass(object):

  def overridden_method(self, x):
    return x + 20


class PyBuiltinsTest(test.TestCase):

  def _basic_function_scope(self):
    return function_wrappers.FunctionScope(
        'test_function_name',
        'test_scope',  # Note: this must match the name in the `with` statement.
        converter.ConversionOptions())

  def test_super_in_original_context_niladic_call(self):
    test_case_self = self

    class TestSubclass(TestBaseClass):

      def overridden_method(self, x):
        test_case_self.fail('This should never be called.')

      def test_method(self):
        with test_case_self._basic_function_scope() as test_scope:
          b = py_builtins.super_in_original_context(super, (), test_scope)
          return b.overridden_method(1)

    tc = TestSubclass()
    self.assertEqual(tc.test_method(), 21)

  def test_super_in_original_context_caller_with_locals(self):
    test_case_self = self

    class TestSubclass(TestBaseClass):

      def overridden_method(self, x):
        test_case_self.fail('This should never be called.')

      def test_method(self, x):
        y = 7
        with test_case_self._basic_function_scope() as test_scope:
          z = 7
          return py_builtins.super_in_original_context(
              super, (), test_scope).overridden_method(x + y - z)

    tc = TestSubclass()
    self.assertEqual(tc.test_method(1), 21)

  def test_super_in_original_context_inner_function(self):
    test_case_self = self

    class TestSubclass(TestBaseClass):

      def overridden_method(self, x):
        test_case_self.fail('This should never be called.')

      def test_method(self, x):
        with test_case_self._basic_function_scope() as test_scope:
          # Oddly, it's sufficient to use `self` in an inner function
          # to gain access to __class__ in this scope.
          # TODO(mdan): Is this true across implementations?
          # Note: normally, it's illegal to use super() in inner functions (it
          # throws an error), but the generated code may create them.
          def inner_fn():
            return py_builtins.super_in_original_context(
                super, (), test_scope).overridden_method(x)

          return inner_fn()

    tc = TestSubclass()
    self.assertEqual(tc.test_method(1), 21)

  def test_super_in_original_context_inner_lambda(self):
    test_case_self = self

    class TestSubclass(TestBaseClass):

      def overridden_method(self, x):
        test_case_self.fail('This should never be called.')

      def test_method(self, x):
        with test_case_self._basic_function_scope() as test_scope:
          # Oddly, it's sufficient to use `self` in an inner function
          # to gain access to __class__ in this scope.
          # TODO(mdan): Is this true across implementations?
          # Note: normally, it's illegal to use super() in inner functions (it
          # throws an error), but the generated code may create them.
          l = lambda: py_builtins.super_in_original_context(  # pylint:disable=g-long-lambda
              super, (), test_scope).overridden_method(x)
          return l()

    tc = TestSubclass()
    self.assertEqual(tc.test_method(1), 21)


if __name__ == '__main__':
  test.main()
