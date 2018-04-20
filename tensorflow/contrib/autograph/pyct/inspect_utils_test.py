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
"""Tests for unspect_utils module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import wraps

import six

from tensorflow.contrib.autograph.pyct import inspect_utils
from tensorflow.python.platform import test


def decorator(f):
  return f


def function_decorator():
  def dec(f):
    return f
  return dec


def wrapping_decorator():
  def dec(f):
    def replacement(*_):
      return None

    @wraps(f)
    def wrapper(*args, **kwargs):
      return replacement(*args, **kwargs)
    return wrapper
  return dec


class TestClass(object):

  def member_function(self):
    pass

  @decorator
  def decorated_member(self):
    pass

  @function_decorator()
  def fn_decorated_member(self):
    pass

  @wrapping_decorator()
  def wrap_decorated_member(self):
    pass

  @staticmethod
  def static_method():
    pass

  @classmethod
  def class_method(cls):
    pass


def free_function():
  pass


def factory():
  return free_function


def free_factory():
  def local_function():
    pass
  return local_function


class InspectUtilsTest(test.TestCase):

  def test_getnamespace_globals(self):
    ns = inspect_utils.getnamespace(factory)
    self.assertEqual(ns['free_function'], free_function)

  def test_getnamespace_hermetic(self):

    # Intentionally hiding the global function to make sure we don't overwrite
    # it in the global namespace.
    free_function = object()  # pylint:disable=redefined-outer-name

    def test_fn():
      return free_function

    ns = inspect_utils.getnamespace(test_fn)
    globs = six.get_function_globals(test_fn)
    self.assertTrue(ns['free_function'] is free_function)
    self.assertFalse(globs['free_function'] is free_function)

  def test_getnamespace_locals(self):

    def called_fn():
      return 0

    closed_over_list = []
    closed_over_primitive = 1

    def local_fn():
      closed_over_list.append(1)
      local_var = 1
      return called_fn() + local_var + closed_over_primitive

    ns = inspect_utils.getnamespace(local_fn)
    self.assertEqual(ns['called_fn'], called_fn)
    self.assertEqual(ns['closed_over_list'], closed_over_list)
    self.assertEqual(ns['closed_over_primitive'], closed_over_primitive)
    self.assertTrue('local_var' not in ns)

  def test_getmethodclass(self):

    self.assertEqual(
        inspect_utils.getmethodclass(free_function), None)
    self.assertEqual(
        inspect_utils.getmethodclass(free_factory()), None)

    self.assertEqual(
        inspect_utils.getmethodclass(TestClass.member_function),
        TestClass)
    self.assertEqual(
        inspect_utils.getmethodclass(TestClass.decorated_member),
        TestClass)
    self.assertEqual(
        inspect_utils.getmethodclass(TestClass.fn_decorated_member),
        TestClass)
    self.assertEqual(
        inspect_utils.getmethodclass(TestClass.wrap_decorated_member),
        TestClass)
    self.assertEqual(
        inspect_utils.getmethodclass(TestClass.static_method),
        TestClass)
    self.assertEqual(
        inspect_utils.getmethodclass(TestClass.class_method),
        TestClass)

    test_obj = TestClass()
    self.assertEqual(
        inspect_utils.getmethodclass(test_obj.member_function),
        TestClass)
    self.assertEqual(
        inspect_utils.getmethodclass(test_obj.decorated_member),
        TestClass)
    self.assertEqual(
        inspect_utils.getmethodclass(test_obj.fn_decorated_member),
        TestClass)
    self.assertEqual(
        inspect_utils.getmethodclass(test_obj.wrap_decorated_member),
        TestClass)
    self.assertEqual(
        inspect_utils.getmethodclass(test_obj.static_method),
        TestClass)
    self.assertEqual(
        inspect_utils.getmethodclass(test_obj.class_method),
        TestClass)

  def test_getmethodclass_locals(self):

    def local_function():
      pass

    class LocalClass(object):

      def member_function(self):
        pass

      @decorator
      def decorated_member(self):
        pass

      @function_decorator()
      def fn_decorated_member(self):
        pass

      @wrapping_decorator()
      def wrap_decorated_member(self):
        pass

    self.assertEqual(
        inspect_utils.getmethodclass(local_function), None)

    self.assertEqual(
        inspect_utils.getmethodclass(LocalClass.member_function),
        LocalClass)
    self.assertEqual(
        inspect_utils.getmethodclass(LocalClass.decorated_member),
        LocalClass)
    self.assertEqual(
        inspect_utils.getmethodclass(LocalClass.fn_decorated_member),
        LocalClass)
    self.assertEqual(
        inspect_utils.getmethodclass(LocalClass.wrap_decorated_member),
        LocalClass)

    test_obj = LocalClass()
    self.assertEqual(
        inspect_utils.getmethodclass(test_obj.member_function),
        LocalClass)
    self.assertEqual(
        inspect_utils.getmethodclass(test_obj.decorated_member),
        LocalClass)
    self.assertEqual(
        inspect_utils.getmethodclass(test_obj.fn_decorated_member),
        LocalClass)
    self.assertEqual(
        inspect_utils.getmethodclass(test_obj.wrap_decorated_member),
        LocalClass)

  def test_getmethodclass_callables(self):
    class TestCallable(object):

      def __call__(self):
        pass

    c = TestCallable()
    self.assertEqual(inspect_utils.getmethodclass(c), TestCallable)

  def test_getdefiningclass(self):
    class Superclass(object):

      def foo(self):
        pass

      def bar(self):
        pass

    class Subclass(Superclass):

      def foo(self):
        pass

      def baz(self):
        pass

    self.assertTrue(
        inspect_utils.getdefiningclass(Subclass.foo, Subclass) is Subclass)
    self.assertTrue(
        inspect_utils.getdefiningclass(Subclass.bar, Subclass) is Superclass)
    self.assertTrue(
        inspect_utils.getdefiningclass(Subclass.baz, Subclass) is Subclass)

  def test_isbuiltin(self):
    self.assertTrue(inspect_utils.isbuiltin(range))
    self.assertTrue(inspect_utils.isbuiltin(float))
    self.assertTrue(inspect_utils.isbuiltin(int))
    self.assertTrue(inspect_utils.isbuiltin(len))
    self.assertFalse(inspect_utils.isbuiltin(function_decorator))


if __name__ == '__main__':
  test.main()
