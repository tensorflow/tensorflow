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

import collections
import functools
import imp
import types
import weakref

import six

from tensorflow.python import lib
from tensorflow.python.autograph.pyct import inspect_utils
from tensorflow.python.framework import constant_op
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

    @functools.wraps(f)
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

  def test_islambda(self):
    def test_fn():
      pass

    self.assertTrue(inspect_utils.islambda(lambda x: x))
    self.assertFalse(inspect_utils.islambda(test_fn))

  def test_isnamedtuple(self):
    nt = collections.namedtuple('TestNamedTuple', ['a', 'b'])

    class NotANamedTuple(tuple):
      pass

    self.assertTrue(inspect_utils.isnamedtuple(nt))
    self.assertFalse(inspect_utils.isnamedtuple(NotANamedTuple))

  def test_isnamedtuple_confounder(self):
    """This test highlights false positives when detecting named tuples."""

    class NamedTupleLike(tuple):
      _fields = ('a', 'b')

    self.assertTrue(inspect_utils.isnamedtuple(NamedTupleLike))

  def test_isnamedtuple_subclass(self):
    """This test highlights false positives when detecting named tuples."""

    class NamedTupleSubclass(collections.namedtuple('Test', ['a', 'b'])):
      pass

    self.assertTrue(inspect_utils.isnamedtuple(NamedTupleSubclass))

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

  def test_getqualifiedname(self):
    foo = object()
    qux = imp.new_module('quxmodule')
    bar = imp.new_module('barmodule')
    baz = object()
    bar.baz = baz

    ns = {
        'foo': foo,
        'bar': bar,
        'qux': qux,
    }

    self.assertIsNone(inspect_utils.getqualifiedname(ns, inspect_utils))
    self.assertEqual(inspect_utils.getqualifiedname(ns, foo), 'foo')
    self.assertEqual(inspect_utils.getqualifiedname(ns, bar), 'bar')
    self.assertEqual(inspect_utils.getqualifiedname(ns, baz), 'bar.baz')

  def test_getqualifiedname_efficiency(self):
    foo = object()

    # We create a densely connected graph consisting of a relatively small
    # number of modules and hide our symbol in one of them. The path to the
    # symbol is at least 10, and each node has about 10 neighbors. However,
    # by skipping visited modules, the search should take much less.
    ns = {}
    prev_level = []
    for i in range(10):
      current_level = []
      for j in range(10):
        mod_name = 'mod_{}_{}'.format(i, j)
        mod = imp.new_module(mod_name)
        current_level.append(mod)
        if i == 9 and j == 9:
          mod.foo = foo
      if prev_level:
        # All modules at level i refer to all modules at level i+1
        for prev in prev_level:
          for mod in current_level:
            prev.__dict__[mod.__name__] = mod
      else:
        for mod in current_level:
          ns[mod.__name__] = mod
      prev_level = current_level

    self.assertIsNone(inspect_utils.getqualifiedname(ns, inspect_utils))
    self.assertIsNotNone(
        inspect_utils.getqualifiedname(ns, foo, max_depth=10000000000))

  def test_getqualifiedname_cycles(self):
    foo = object()

    # We create a graph of modules that contains circular references. The
    # search process should avoid them. The searched object is hidden at the
    # bottom of a path of length roughly 10.
    ns = {}
    mods = []
    for i in range(10):
      mod = imp.new_module('mod_{}'.format(i))
      if i == 9:
        mod.foo = foo
      # Module i refers to module i+1
      if mods:
        mods[-1].__dict__[mod.__name__] = mod
      else:
        ns[mod.__name__] = mod
      # Module i refers to all modules j < i.
      for prev in mods:
        mod.__dict__[prev.__name__] = prev
      mods.append(mod)

    self.assertIsNone(inspect_utils.getqualifiedname(ns, inspect_utils))
    self.assertIsNotNone(
        inspect_utils.getqualifiedname(ns, foo, max_depth=10000000000))

  def test_getqualifiedname_finds_via_parent_module(self):
    # TODO(mdan): This test is vulnerable to change in the lib module.
    # A better way to forge modules should be found.
    self.assertEqual(
        inspect_utils.getqualifiedname(
            lib.__dict__, lib.io.file_io.FileIO, max_depth=1),
        'io.file_io.FileIO')

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

  def test_getmethodclass_weakref_mechanism(self):
    test_obj = TestClass()

    class WeakrefWrapper(object):

      def __init__(self):
        self.ag_self_weakref__ = weakref.ref(test_obj)

    def test_fn(self):
      return self

    bound_method = types.MethodType(test_fn, WeakrefWrapper())
    self.assertEqual(inspect_utils.getmethodclass(bound_method), TestClass)

  def test_getmethodclass_no_bool_conversion(self):

    tensor = constant_op.constant([1])
    self.assertEqual(
        inspect_utils.getmethodclass(tensor.get_shape), type(tensor))

  def test_getdefiningclass(self):
    class Superclass(object):

      def foo(self):
        pass

      def bar(self):
        pass

      @classmethod
      def class_method(cls):
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
    self.assertTrue(
        inspect_utils.getdefiningclass(Subclass.class_method, Subclass) is
        Superclass)

  def test_isbuiltin(self):
    self.assertTrue(inspect_utils.isbuiltin(range))
    self.assertTrue(inspect_utils.isbuiltin(float))
    self.assertTrue(inspect_utils.isbuiltin(int))
    self.assertTrue(inspect_utils.isbuiltin(len))
    self.assertFalse(inspect_utils.isbuiltin(function_decorator))

  def test_super_wrapper_for_dynamic_attrs(self):

    a = object()
    b = object()

    class Base(object):

      def __init__(self):
        self.a = a

    class Subclass(Base):

      def __init__(self):
        super(Subclass, self).__init__()
        self.b = b

    base = Base()
    sub = Subclass()

    sub_super = super(Subclass, sub)
    sub_super_wrapped = inspect_utils.SuperWrapperForDynamicAttrs(sub_super)

    self.assertIs(base.a, a)
    self.assertIs(sub.a, a)

    self.assertFalse(hasattr(sub_super, 'a'))
    self.assertIs(sub_super_wrapped.a, a)

    # TODO(mdan): Is this side effect harmful? Can it be avoided?
    # Note that `b` was set in `Subclass.__init__`.
    self.assertIs(sub_super_wrapped.b, b)


if __name__ == '__main__':
  test.main()
