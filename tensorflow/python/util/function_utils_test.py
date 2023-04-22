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
"""Tests for Estimator related util."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

from tensorflow.python.platform import test
from tensorflow.python.util import function_utils


def silly_example_function():
  pass


class SillyCallableClass(object):

  def __call__(self):
    pass


class FnArgsTest(test.TestCase):

  def test_simple_function(self):
    def fn(a, b):
      return a + b
    self.assertEqual(('a', 'b'), function_utils.fn_args(fn))

  def test_callable(self):

    class Foo(object):

      def __call__(self, a, b):
        return a + b

    self.assertEqual(('a', 'b'), function_utils.fn_args(Foo()))

  def test_bound_method(self):

    class Foo(object):

      def bar(self, a, b):
        return a + b

    self.assertEqual(('a', 'b'), function_utils.fn_args(Foo().bar))

  def test_bound_method_no_self(self):

    class Foo(object):

      def bar(*args):  # pylint:disable=no-method-argument
        return args[1] + args[2]

    self.assertEqual((), function_utils.fn_args(Foo().bar))

  def test_partial_function(self):
    expected_test_arg = 123

    def fn(a, test_arg):
      if test_arg != expected_test_arg:
        return ValueError('partial fn does not work correctly')
      return a

    wrapped_fn = functools.partial(fn, test_arg=123)

    self.assertEqual(('a',), function_utils.fn_args(wrapped_fn))

  def test_partial_function_with_positional_args(self):
    expected_test_arg = 123

    def fn(test_arg, a):
      if test_arg != expected_test_arg:
        return ValueError('partial fn does not work correctly')
      return a

    wrapped_fn = functools.partial(fn, 123)

    self.assertEqual(('a',), function_utils.fn_args(wrapped_fn))

    self.assertEqual(3, wrapped_fn(3))
    self.assertEqual(3, wrapped_fn(a=3))

  def test_double_partial(self):
    expected_test_arg1 = 123
    expected_test_arg2 = 456

    def fn(a, test_arg1, test_arg2):
      if test_arg1 != expected_test_arg1 or test_arg2 != expected_test_arg2:
        return ValueError('partial does not work correctly')
      return a

    wrapped_fn = functools.partial(fn, test_arg2=456)
    double_wrapped_fn = functools.partial(wrapped_fn, test_arg1=123)

    self.assertEqual(('a',), function_utils.fn_args(double_wrapped_fn))

  def test_double_partial_with_positional_args_in_outer_layer(self):
    expected_test_arg1 = 123
    expected_test_arg2 = 456

    def fn(test_arg1, a, test_arg2):
      if test_arg1 != expected_test_arg1 or test_arg2 != expected_test_arg2:
        return ValueError('partial fn does not work correctly')
      return a

    wrapped_fn = functools.partial(fn, test_arg2=456)
    double_wrapped_fn = functools.partial(wrapped_fn, 123)

    self.assertEqual(('a',), function_utils.fn_args(double_wrapped_fn))

    self.assertEqual(3, double_wrapped_fn(3))  # pylint: disable=no-value-for-parameter
    self.assertEqual(3, double_wrapped_fn(a=3))  # pylint: disable=no-value-for-parameter

  def test_double_partial_with_positional_args_in_both_layers(self):
    expected_test_arg1 = 123
    expected_test_arg2 = 456

    def fn(test_arg1, test_arg2, a):
      if test_arg1 != expected_test_arg1 or test_arg2 != expected_test_arg2:
        return ValueError('partial fn does not work correctly')
      return a

    wrapped_fn = functools.partial(fn, 123)  # binds to test_arg1
    double_wrapped_fn = functools.partial(wrapped_fn, 456)  # binds to test_arg2

    self.assertEqual(('a',), function_utils.fn_args(double_wrapped_fn))

    self.assertEqual(3, double_wrapped_fn(3))  # pylint: disable=no-value-for-parameter
    self.assertEqual(3, double_wrapped_fn(a=3))  # pylint: disable=no-value-for-parameter


class HasKwargsTest(test.TestCase):

  def test_simple_function(self):

    fn_has_kwargs = lambda **x: x
    self.assertTrue(function_utils.has_kwargs(fn_has_kwargs))

    fn_has_no_kwargs = lambda x: x
    self.assertFalse(function_utils.has_kwargs(fn_has_no_kwargs))

  def test_callable(self):

    class FooHasKwargs(object):

      def __call__(self, **x):
        del x
    self.assertTrue(function_utils.has_kwargs(FooHasKwargs()))

    class FooHasNoKwargs(object):

      def __call__(self, x):
        del x
    self.assertFalse(function_utils.has_kwargs(FooHasNoKwargs()))

  def test_bound_method(self):

    class FooHasKwargs(object):

      def fn(self, **x):
        del x
    self.assertTrue(function_utils.has_kwargs(FooHasKwargs().fn))

    class FooHasNoKwargs(object):

      def fn(self, x):
        del x
    self.assertFalse(function_utils.has_kwargs(FooHasNoKwargs().fn))

  def test_partial_function(self):
    expected_test_arg = 123

    def fn_has_kwargs(test_arg, **x):
      if test_arg != expected_test_arg:
        return ValueError('partial fn does not work correctly')
      return x

    wrapped_fn = functools.partial(fn_has_kwargs, test_arg=123)
    self.assertTrue(function_utils.has_kwargs(wrapped_fn))
    some_kwargs = dict(x=1, y=2, z=3)
    self.assertEqual(wrapped_fn(**some_kwargs), some_kwargs)

    def fn_has_no_kwargs(x, test_arg):
      if test_arg != expected_test_arg:
        return ValueError('partial fn does not work correctly')
      return x

    wrapped_fn = functools.partial(fn_has_no_kwargs, test_arg=123)
    self.assertFalse(function_utils.has_kwargs(wrapped_fn))
    some_arg = 1
    self.assertEqual(wrapped_fn(some_arg), some_arg)

  def test_double_partial(self):
    expected_test_arg1 = 123
    expected_test_arg2 = 456

    def fn_has_kwargs(test_arg1, test_arg2, **x):
      if test_arg1 != expected_test_arg1 or test_arg2 != expected_test_arg2:
        return ValueError('partial does not work correctly')
      return x

    wrapped_fn = functools.partial(fn_has_kwargs, test_arg2=456)
    double_wrapped_fn = functools.partial(wrapped_fn, test_arg1=123)

    self.assertTrue(function_utils.has_kwargs(double_wrapped_fn))
    some_kwargs = dict(x=1, y=2, z=3)
    self.assertEqual(double_wrapped_fn(**some_kwargs), some_kwargs)

    def fn_has_no_kwargs(x, test_arg1, test_arg2):
      if test_arg1 != expected_test_arg1 or test_arg2 != expected_test_arg2:
        return ValueError('partial does not work correctly')
      return x

    wrapped_fn = functools.partial(fn_has_no_kwargs, test_arg2=456)
    double_wrapped_fn = functools.partial(wrapped_fn, test_arg1=123)

    self.assertFalse(function_utils.has_kwargs(double_wrapped_fn))
    some_arg = 1
    self.assertEqual(double_wrapped_fn(some_arg), some_arg)  # pylint: disable=no-value-for-parameter

  def test_raises_type_error(self):
    with self.assertRaisesRegex(TypeError,
                                'fn should be a function-like object'):
      function_utils.has_kwargs('not a function')


class GetFuncNameTest(test.TestCase):

  def testWithSimpleFunction(self):
    self.assertEqual(
        'silly_example_function',
        function_utils.get_func_name(silly_example_function))

  def testWithClassMethod(self):
    self.assertEqual(
        'GetFuncNameTest.testWithClassMethod',
        function_utils.get_func_name(self.testWithClassMethod))

  def testWithCallableClass(self):
    callable_instance = SillyCallableClass()
    self.assertRegex(
        function_utils.get_func_name(callable_instance),
        '<.*SillyCallableClass.*>')

  def testWithFunctoolsPartial(self):
    partial = functools.partial(silly_example_function)
    self.assertRegex(
        function_utils.get_func_name(partial), '<.*functools.partial.*>')

  def testWithLambda(self):
    anon_fn = lambda x: x
    self.assertEqual('<lambda>', function_utils.get_func_name(anon_fn))

  def testRaisesWithNonCallableObject(self):
    with self.assertRaises(ValueError):
      function_utils.get_func_name(None)


class GetFuncCodeTest(test.TestCase):

  def testWithSimpleFunction(self):
    code = function_utils.get_func_code(silly_example_function)
    self.assertIsNotNone(code)
    self.assertRegex(code.co_filename, 'function_utils_test.py')

  def testWithClassMethod(self):
    code = function_utils.get_func_code(self.testWithClassMethod)
    self.assertIsNotNone(code)
    self.assertRegex(code.co_filename, 'function_utils_test.py')

  def testWithCallableClass(self):
    callable_instance = SillyCallableClass()
    code = function_utils.get_func_code(callable_instance)
    self.assertIsNotNone(code)
    self.assertRegex(code.co_filename, 'function_utils_test.py')

  def testWithLambda(self):
    anon_fn = lambda x: x
    code = function_utils.get_func_code(anon_fn)
    self.assertIsNotNone(code)
    self.assertRegex(code.co_filename, 'function_utils_test.py')

  def testWithFunctoolsPartial(self):
    partial = functools.partial(silly_example_function)
    code = function_utils.get_func_code(partial)
    self.assertIsNone(code)

  def testRaisesWithNonCallableObject(self):
    with self.assertRaises(ValueError):
      function_utils.get_func_code(None)


if __name__ == '__main__':
  test.main()
