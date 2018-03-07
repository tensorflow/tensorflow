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

from tensorflow.contrib.py2tf.pyct import inspect_utils
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


def free_factory():
  def local_function():
    pass
  return local_function


class InspectUtilsTest(test.TestCase):

  def test_getcallargs_constructor(self):

    class TestSuperclass(object):

      def __init__(self, x):
        pass

    class TestCallable(TestSuperclass):
      pass

    self.assertDictEqual({
        'x': 1
    }, inspect_utils.getcallargs(TestCallable, 1))

  def test_getcallargs_object(self):

    class TestCallable(object):

      def __call__(self, x):
        pass

    obj = TestCallable()
    self.assertDictEqual({
        'self': obj,
        'x': 1
    }, inspect_utils.getcallargs(obj, 1))

  def test_getcallargs_function(self):

    def test_fn(x):
      return x + 1

    self.assertDictEqual({
        'x': 1
    }, inspect_utils.getcallargs(test_fn, 1))

  def test_getmethodclass(self):

    self.assertEqual(
        inspect_utils.getmethodclass(free_function, {}), None)
    self.assertEqual(
        inspect_utils.getmethodclass(free_factory(), {}), None)

    ns = {'TestClass': TestClass}
    self.assertEqual(
        inspect_utils.getmethodclass(TestClass.member_function, ns),
        TestClass)
    self.assertEqual(
        inspect_utils.getmethodclass(TestClass.decorated_member, ns),
        TestClass)
    self.assertEqual(
        inspect_utils.getmethodclass(TestClass.fn_decorated_member, ns),
        TestClass)
    self.assertEqual(
        inspect_utils.getmethodclass(TestClass.wrap_decorated_member, ns),
        TestClass)
    self.assertEqual(
        inspect_utils.getmethodclass(TestClass.static_method, ns),
        TestClass)
    self.assertEqual(
        inspect_utils.getmethodclass(TestClass.class_method, ns),
        TestClass)

    test_obj = TestClass()
    self.assertEqual(
        inspect_utils.getmethodclass(test_obj.member_function, ns),
        TestClass)
    self.assertEqual(
        inspect_utils.getmethodclass(test_obj.decorated_member, ns),
        TestClass)
    self.assertEqual(
        inspect_utils.getmethodclass(test_obj.fn_decorated_member, ns),
        TestClass)
    self.assertEqual(
        inspect_utils.getmethodclass(test_obj.wrap_decorated_member, ns),
        TestClass)
    self.assertEqual(
        inspect_utils.getmethodclass(test_obj.static_method, ns),
        TestClass)
    self.assertEqual(
        inspect_utils.getmethodclass(test_obj.class_method, ns),
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
        inspect_utils.getmethodclass(local_function, {}), None)

    ns = {'LocalClass': LocalClass}
    self.assertEqual(
        inspect_utils.getmethodclass(LocalClass.member_function, ns),
        LocalClass)
    self.assertEqual(
        inspect_utils.getmethodclass(LocalClass.decorated_member, ns),
        LocalClass)
    self.assertEqual(
        inspect_utils.getmethodclass(LocalClass.fn_decorated_member, ns),
        LocalClass)
    self.assertEqual(
        inspect_utils.getmethodclass(LocalClass.wrap_decorated_member, ns),
        LocalClass)

    test_obj = LocalClass()
    self.assertEqual(
        inspect_utils.getmethodclass(test_obj.member_function, ns),
        LocalClass)
    self.assertEqual(
        inspect_utils.getmethodclass(test_obj.decorated_member, ns),
        LocalClass)
    self.assertEqual(
        inspect_utils.getmethodclass(test_obj.fn_decorated_member, ns),
        LocalClass)
    self.assertEqual(
        inspect_utils.getmethodclass(test_obj.wrap_decorated_member, ns),
        LocalClass)


if __name__ == '__main__':
  test.main()
