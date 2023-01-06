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
"""Tests for api module."""

import abc
import collections
import contextlib
import functools
import gc
import imp
import inspect
import io
import os
import re
import sys
import textwrap
import types

import numpy as np

from tensorflow.python.autograph.core import ag_ctx
from tensorflow.python.autograph.core import converter
from tensorflow.python.autograph.core import converter_testing
from tensorflow.python.autograph.impl import api
from tensorflow.python.autograph.impl import conversion
from tensorflow.python.autograph.pyct import errors
from tensorflow.python.autograph.pyct import inspect_utils
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.utils import ag_logging
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import def_function
from tensorflow.python.framework import _errors_test_helper
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors as tf_errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.util import function_utils
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect

global_n = 2

DEFAULT_RECURSIVE = converter.ConversionOptions(recursive=True)


class TestResource:

  def __init__(self):
    self.x = 3


class ApiTest(test.TestCase):

  @contextlib.contextmanager
  def assertPrints(self, expected, not_expected):
    try:
      out_capturer = io.StringIO()
      sys.stdout = out_capturer
      yield
      self.assertIn(expected, out_capturer.getvalue())
      self.assertNotIn(not_expected, out_capturer.getvalue())
    finally:
      sys.stdout = sys.__stdout__

  def assertNoMemoryLeaks(self, f):
    object_ids_before = {id(o) for o in gc.get_objects()}
    f()
    gc.collect()
    objects_after = tuple(
        o for o in gc.get_objects() if id(o) not in object_ids_before)
    self.assertEmpty(
        tuple(o for o in objects_after if isinstance(o, TestResource)))

  def test_converted_call_kwonly_args(self):

    def test_fn(*, a):
      return a

    x = api.converted_call(
        test_fn, (), {'a': constant_op.constant(-1)}, options=DEFAULT_RECURSIVE)
    self.assertEqual(-1, self.evaluate(x))

  def test_super_with_no_arg(self):
    test_case_self = self

    class TestBase:

      def plus_three(self, x):
        return x + 3

    class TestSubclass(TestBase):

      def plus_three(self, x):
        test_case_self.fail('This should never be called.')

      def no_arg(self, x):
        return super().plus_three(x)

    tc = api.converted_call(TestSubclass, (), None, options=DEFAULT_RECURSIVE)

    self.assertEqual(5, tc.no_arg(2))

  def test_converted_call_avoids_triggering_operators(self):

    test_self = self

    class Pair(collections.namedtuple('Pair', ['a', 'b'])):

      def __call__(self):
        return self.a + self.b

      def __eq__(self, other):
        test_self.fail('Triggered operator')

    p = Pair(constant_op.constant(1), constant_op.constant(2))

    x = api.converted_call(p, (), {}, options=DEFAULT_RECURSIVE)
    self.assertIsNotNone(self.evaluate(x), 3)

  @test_util.run_deprecated_v1
  def test_decorator_recursive(self):

    class TestClass:

      def called_member(self, a):
        if a < 0:
          a = -a
        return a

      @api.convert(recursive=True)
      def test_method(self, x, s, a):
        while math_ops.reduce_sum(x) > s:
          x //= self.called_member(a)
        return x

    tc = TestClass()
    x = tc.test_method(
        constant_op.constant([2, 4]), constant_op.constant(1),
        constant_op.constant(-2))
    self.assertListEqual([0, 1], self.evaluate(x).tolist())

  @test_util.run_deprecated_v1
  def test_decorator_not_recursive(self):

    class TestClass:

      def called_member(self, a):
        return math_ops.negative(a)

      @api.convert(recursive=False)
      def test_method(self, x, s, a):
        while math_ops.reduce_sum(x) > s:
          x //= self.called_member(a)
        return x

    tc = TestClass()
    x = tc.test_method(
        constant_op.constant([2, 4]), constant_op.constant(1),
        constant_op.constant(-2))
    self.assertListEqual([0, 1], self.evaluate(x).tolist())

  @test_util.run_deprecated_v1
  def test_convert_then_do_not_convert(self):

    class TestClass:

      @api.do_not_convert
      def called_member(self, a):
        return math_ops.negative(a)

      @api.convert(recursive=True)
      def test_method(self, x, s, a):
        while math_ops.reduce_sum(x) > s:
          x //= self.called_member(a)
        return x

    tc = TestClass()
    x = tc.test_method(
        constant_op.constant((2, 4)), constant_op.constant(1),
        constant_op.constant(-2))
    self.assertAllEqual((0, 1), self.evaluate(x))

  @test_util.run_deprecated_v1
  def test_decorator_calls_decorated(self):

    class TestClass:

      @api.convert()
      def called_member(self, a):
        if a < 0:
          a = -a
        return a

      @api.convert(recursive=True)
      def test_method(self, x, s, a):
        while math_ops.reduce_sum(x) > s:
          x //= self.called_member(a)
        return x

    tc = TestClass()
    x = tc.test_method(
        constant_op.constant([2, 4]), constant_op.constant(1),
        constant_op.constant(-2))
    self.assertListEqual([0, 1], self.evaluate(x).tolist())

  def test_decorator_preserves_argspec(self):

    class TestClass:

      def test_method(self, a):
        if a < 0:
          a = -a
        return a

      test_method_converted = api.convert()(test_method)

    tc = TestClass()
    self.assertListEqual(
        list(tf_inspect.getfullargspec(tc.test_method)),
        list(tf_inspect.getfullargspec(tc.test_method_converted)))

  def test_do_not_convert_argspec(self):

    class TestClass:

      def test_method(self, x, y):
        z = x + y
        return z

      test_method_allowlisted = api.do_not_convert(test_method)

    tc = TestClass()
    self.assertTrue(tf_inspect.ismethod(tc.test_method_allowlisted))
    # Because the wrapped function is not generated, we can't preserve its
    # arg spec.
    self.assertEqual((),
                     tuple(function_utils.fn_args(tc.test_method_allowlisted)))

  def test_do_not_convert_callable_object(self):

    class TestClass:

      def __call__(self):
        return 1

    tc = TestClass()
    self.assertEqual(1, api.do_not_convert(tc)())

  @test_util.run_deprecated_v1
  def test_convert_call_site_decorator(self):

    class TestClass:

      def called_member(self, a):
        if a < 0:
          a = -a
        return a

      @api.convert(recursive=True)
      def test_method(self, x, s, a):
        while math_ops.reduce_sum(x) > s:
          x //= api.converted_call(
              self.called_member, (a,), None, options=DEFAULT_RECURSIVE)
        return x

    tc = TestClass()
    x = tc.test_method(
        constant_op.constant([2, 4]), constant_op.constant(1),
        constant_op.constant(-2))
    self.assertListEqual([0, 1], self.evaluate(x).tolist())

  def test_converted_call_builtin(self):
    x = api.converted_call(range, (3,), None, options=DEFAULT_RECURSIVE)
    self.assertEqual((0, 1, 2), tuple(x))

    x = api.converted_call(
        re.compile, ('mnas_v4_a.*\\/.*(weights|kernel):0$',),
        None,
        options=DEFAULT_RECURSIVE)
    self.assertIsNotNone(x.match('mnas_v4_a/weights:0'))

  def test_converted_call_function(self):

    def test_fn(x):
      if x < 0:
        return -x
      return x

    x = api.converted_call(
        test_fn, (constant_op.constant(-1),), None, options=DEFAULT_RECURSIVE)
    self.assertEqual(1, self.evaluate(x))

  @test_util.run_v1_only('b/120545219')
  def test_converted_call_functools_partial(self):

    def test_fn(x, y, z):
      if x < 0:
        return -x, -y, -z
      return x, y, z

    x = api.converted_call(
        functools.partial(test_fn, constant_op.constant(-1), z=-3),
        (constant_op.constant(-2),),
        None,
        options=DEFAULT_RECURSIVE)
    self.assertEqual((1, 2, 3), self.evaluate(x))

    x = api.converted_call(
        functools.partial(
            functools.partial(test_fn, constant_op.constant(-1)), z=-3),
        (constant_op.constant(-2),),
        None,
        options=DEFAULT_RECURSIVE)
    self.assertEqual((1, 2, 3), self.evaluate(x))

  @test_util.run_v1_only('b/120545219')
  def test_converted_call_functools_partial_kwarg_mutation(self):

    def test_fn(x, y, z):
      if x < 0:
        return -x, -y, -z
      return x, y, z

    partial_fn = functools.partial(test_fn, constant_op.constant(-1), z=-3)
    # Call using kwargs to assign y first to ensure that partial_fn.keywords is
    # not mutated for subsequent calls (where y is assign through args).
    x = api.converted_call(
        partial_fn,
        args=(),
        kwargs={
            'y': constant_op.constant(-2),
        },
        options=DEFAULT_RECURSIVE)
    self.assertEqual((1, 2, 3), self.evaluate(x))

    x = api.converted_call(
        partial_fn,
        args=(constant_op.constant(-4),),
        kwargs=None,
        options=DEFAULT_RECURSIVE)
    self.assertEqual((1, 4, 3), self.evaluate(x))

  def test_converted_call_method(self):

    class TestClass:

      def __init__(self, x):
        self.x = x

      def test_method(self):
        if self.x < 0:
          return -self.x
        return self.x

    tc = TestClass(constant_op.constant(-1))
    x = api.converted_call(tc.test_method, (), None, options=DEFAULT_RECURSIVE)
    self.assertEqual(1, self.evaluate(x))

  def test_converted_call_synthetic_method(self):

    class TestClass:

      def __init__(self, x):
        self.x = x

    def test_function(self):
      if self.x < 0:
        return -self.x
      return self.x

    tc = TestClass(constant_op.constant(-1))
    test_method = types.MethodType(test_function, tc)

    x = api.converted_call(test_method, (), None, options=DEFAULT_RECURSIVE)
    self.assertEqual(1, self.evaluate(x))

  def test_converted_call_method_wrapper(self):

    class TestClass:

      def foo(self):
        pass

    tc = TestClass()

    # `method.__get__()` returns a so-called method-wrapper.
    wrapper = api.converted_call(
        tc.foo.__get__, (tc,), None, options=DEFAULT_RECURSIVE)
    self.assertEqual(wrapper, tc.foo)

  def test_converted_call_method_as_object_attribute(self):

    class AnotherClass:

      def __init__(self):
        self.another_class_attr = constant_op.constant(1)

      def method(self):
        if self.another_class_attr > 0:
          return self.another_class_attr + 1
        return self.another_class_attr + 10

    class TestClass:

      def __init__(self, another_obj_method):
        self.another_obj_method = another_obj_method

    obj = AnotherClass()
    tc = TestClass(obj.method)

    x = api.converted_call(
        tc.another_obj_method, (), None, options=DEFAULT_RECURSIVE)
    self.assertEqual(self.evaluate(x), 2)

  def test_converted_call_method_converts_recursively(self):

    class TestClass:

      def __init__(self, x):
        self.x = x

      def other_method(self):
        if self.x < 0:
          return -self.x
        return self.x

      def test_method(self):
        return self.other_method()

    tc = TestClass(constant_op.constant(-1))
    x = api.converted_call(tc.test_method, (), None, options=DEFAULT_RECURSIVE)
    self.assertEqual(1, self.evaluate(x))

  def test_converted_call_method_by_class(self):

    class TestClass:

      def __init__(self, x):
        self.x = x

      def test_method(self):
        if self.x < 0:
          return -self.x
        return self.x

    tc = TestClass(constant_op.constant(-1))
    x = api.converted_call(
        TestClass.test_method, (tc,), None, options=DEFAULT_RECURSIVE)
    self.assertEqual(1, self.evaluate(x))

  def test_converted_call_callable_object(self):

    class TestClass:

      def __init__(self, x):
        self.x = x

      def __call__(self):
        if self.x < 0:
          return -self.x
        return self.x

    tc = TestClass(constant_op.constant(-1))
    x = api.converted_call(tc, (), None, options=DEFAULT_RECURSIVE)
    self.assertEqual(1, self.evaluate(x))

  def test_converted_call_callable_metaclass(self):

    test_self = self

    class TestMetaclass(type):

      def __call__(cls):  # pylint: disable=method-hidden
        self.assertTrue(converter_testing.is_inside_generated_code())
        inst = object.__new__(cls)
        inst.__init__()

        def instance_call(unused_self):
          test_self.fail(
              'The class-bound __call__ should be called, not the instance'
              ' bound one.')

        inst.__call__ = instance_call
        return inst

    tmc = TestMetaclass('TestClass', (), {})
    tc = api.converted_call(tmc, (), None, options=DEFAULT_RECURSIVE)
    self.assertIsInstance(tc, tmc)

  def test_converted_call_callable_abc(self):

    test_self = self

    class TestBase(metaclass=abc.ABCMeta):

      @abc.abstractmethod
      def __call__(self):
        test_self.fail('This should not be called')

    class TestSubclass(TestBase):

      def __init__(self):
        test_self.assertFalse(converter_testing.is_inside_generated_code())

      def __call__(self, expected):
        test_self.assertTrue(expected)
        test_self.assertTrue(converter_testing.is_inside_generated_code())

    tc = api.converted_call(TestSubclass, (), None, options=DEFAULT_RECURSIVE)
    api.converted_call(tc, (True,), None, options=DEFAULT_RECURSIVE)

  @test_util.run_deprecated_v1
  def test_converted_call_constructor(self):

    test_self = self

    class TestClass:

      def __init__(self):
        test_self.assertFalse(converter_testing.is_inside_generated_code())

    tc = api.converted_call(TestClass, (), None, options=DEFAULT_RECURSIVE)
    self.assertIsInstance(tc, TestClass)

  def test_converted_call_mangled_properties(self):

    class TestClass:

      def __init__(self):
        self.__private = constant_op.constant(-1)

      def test_method(self):
        return self.__private

    tc = TestClass()
    with self.assertRaisesRegex(errors.UnsupportedLanguageElementError,
                                'mangled names'):
      api.converted_call(tc.test_method, (), None, options=DEFAULT_RECURSIVE)

    # TODO(mdan): Refactor to avoid this use of global state.
    ag_logging.set_verbosity(0, True)
    os.environ['AUTOGRAPH_STRICT_CONVERSION'] = '0'
    with self.assertPrints('could not transform', 'bug'):
      api.converted_call(tc.test_method, (), None, options=DEFAULT_RECURSIVE)
    ag_logging.set_verbosity(0, False)
    os.environ['AUTOGRAPH_STRICT_CONVERSION'] = '1'

  def test_converted_call_partial_of_allowlisted_function(self):

    def test_fn(_):
      self.assertFalse(converter_testing.is_inside_generated_code())

    converter_testing.allowlist(test_fn)
    api.converted_call(
        functools.partial(test_fn, None), (), None, options=DEFAULT_RECURSIVE)

  def test_converted_call_already_converted(self):

    def f(x):
      return x == 0

    x = api.converted_call(
        f, (constant_op.constant(0),), None, options=DEFAULT_RECURSIVE)
    self.assertTrue(self.evaluate(x))

    converted_f = api.to_graph(
        f, experimental_optional_features=converter.Feature.ALL)
    x = api.converted_call(
        converted_f, (constant_op.constant(0),),
        None,
        options=DEFAULT_RECURSIVE)
    self.assertTrue(self.evaluate(x))

  def test_converted_call_then_already_converted_dynamic(self):

    @api.convert()
    def g(x):
      if x > 0:
        return x
      else:
        return -x

    def f(g, x):
      return g(x)

    x = api.converted_call(
        f, (g, constant_op.constant(1)), None, options=DEFAULT_RECURSIVE)
    self.assertEqual(self.evaluate(x), 1)

  def test_converted_call_forced_when_explicitly_allowlisted(self):

    @api.do_not_convert()
    def f(x):
      return x + 1

    opts = converter.ConversionOptions(recursive=True, user_requested=True)
    x = api.converted_call(f, (constant_op.constant(0),), None, options=opts)
    self.assertTrue(self.evaluate(x))

    converted_f = api.to_graph(
        f, experimental_optional_features=converter.Feature.ALL)
    x = api.converted_call(converted_f, (0,), None, options=DEFAULT_RECURSIVE)
    self.assertEqual(x, 1)

  @test_util.run_deprecated_v1
  def test_converted_call_no_user_code(self):

    def f(x):
      return len(x)

    opts = converter.ConversionOptions(internal_convert_user_code=False)

    # f should not be converted, causing len to error out.
    with self.assertRaisesRegex(Exception, 'len is not well defined'):
      api.converted_call(f, (constant_op.constant([0]),), None, options=opts)

    # len on the other hand should work fine.
    x = api.converted_call(
        len, (constant_op.constant([0]),), None, options=opts)
    # The constant has static shape so the result is a primitive not a Tensor.
    self.assertEqual(x, 1)

  def test_converted_call_no_kwargs_allowed(self):

    def f(*args):
      # Note: np.broadcast rejects any **kwargs, even *{}
      return np.broadcast(args[:1])

    opts = converter.ConversionOptions(internal_convert_user_code=False)
    self.assertIsNotNone(
        api.converted_call(f, (1, 2, 3, 4), None, options=opts))

  def test_converted_call_allowlisted_method(self):

    class TestClass:

      def method(self):
        return converter_testing.is_inside_generated_code()

    obj = TestClass()
    converter_testing.allowlist(obj.method.__func__)

    self.assertFalse(
        api.converted_call(obj.method, (), {}, options=DEFAULT_RECURSIVE))

  def test_converted_call_allowlisted_method_via_owner(self):

    class TestClass:

      def method(self):
        return converter_testing.is_inside_generated_code()

    converter_testing.allowlist(TestClass)

    obj = TestClass()
    self.assertFalse(
        api.converted_call(obj.method, (), {}, options=DEFAULT_RECURSIVE))

  def test_converted_call_numpy(self):

    x = api.converted_call(np.arange, (5,), None, options=DEFAULT_RECURSIVE)

    self.assertAllEqual(x, list(range(5)))

  def test_converted_call_tf_op_forced(self):

    # TODO(mdan): Add the missing level of support to LOGICAL_EXPRESSIONS.
    opts = converter.ConversionOptions(
        user_requested=True, optional_features=None)

    x = api.converted_call(math_ops.add, (1, 1), None, options=opts)

    self.assertAllEqual(self.evaluate(x), 2)

  def test_converted_call_exec_generated_code(self):

    temp_mod = imp.new_module('test_module')
    dynamic_code = """
      def foo(x):
        return x + 1
    """
    exec(textwrap.dedent(dynamic_code), temp_mod.__dict__)  # pylint:disable=exec-used
    opts = converter.ConversionOptions(optional_features=None)

    x = api.converted_call(temp_mod.foo, (1,), None, options=opts)

    self.assertAllEqual(x, 2)

  def test_converted_call_namedtuple(self):

    x = api.converted_call(
        collections.namedtuple, ('TestNamedtuple', ('a', 'b')),
        None,
        options=DEFAULT_RECURSIVE)

    self.assertTrue(inspect_utils.isnamedtuple(x))

  def test_converted_call_namedtuple_via_collections(self):

    x = api.converted_call(
        collections.namedtuple, ('TestNamedtuple', ('a', 'b')),
        None,
        options=DEFAULT_RECURSIVE)

    self.assertTrue(inspect_utils.isnamedtuple(x))

  def test_converted_call_namedtuple_subclass_bound_method(self):

    class TestClass(collections.namedtuple('TestNamedtuple', ('a', 'b'))):

      def test_method(self, x):
        while math_ops.reduce_sum(x) > self.a:
          x //= self.b
        return x

    obj = TestClass(5, 2)
    x = api.converted_call(
        obj.test_method, (constant_op.constant([2, 4]),),
        None,
        options=DEFAULT_RECURSIVE)

    self.assertAllEqual(self.evaluate(x), [1, 2])

  def test_converted_call_namedtuple_method(self):

    class TestClass(collections.namedtuple('TestNamedtuple', ('a', 'b'))):
      pass

    obj = TestClass(5, 2)
    # _asdict is a documented method of namedtuple.
    x = api.converted_call(obj._asdict, (), None, options=DEFAULT_RECURSIVE)

    self.assertDictEqual(x, {'a': 5, 'b': 2})

  def test_converted_call_namedtuple_subclass_unbound_method(self):

    class TestClass(collections.namedtuple('TestNamedtuple', ('a', 'b'))):

      def test_method(self, x):
        while math_ops.reduce_sum(x) > self.a:
          x //= self.b
        return x

    obj = TestClass(5, 2)
    x = api.converted_call(
        TestClass.test_method, (obj, constant_op.constant([2, 4])),
        None,
        options=DEFAULT_RECURSIVE)

    self.assertAllEqual(self.evaluate(x), [1, 2])

  def test_converted_call_lambda(self):

    l = lambda x: x == 0

    x = api.converted_call(
        l, (constant_op.constant(0),), None, options=DEFAULT_RECURSIVE)

    self.evaluate(variables.global_variables_initializer())
    self.assertAllEqual(True, self.evaluate(x))

  def test_converted_call_function_object_method(self):

    # pylint:disable=method-hidden
    class TestClass:

      def method(self):
        return 1

      def prepare(self):
        self.method = def_function.function(self.method)

    # pylint:enable=method-hidden

    tc = TestClass()
    tc.prepare()

    x = api.converted_call(tc.method, (), None, options=DEFAULT_RECURSIVE)

    self.assertAllEqual(1, self.evaluate(x))

  def test_converted_call_native_binding(self):
    x = api.converted_call(np.power, (2, 2), None, options=DEFAULT_RECURSIVE)
    self.assertAllEqual(x, 4)

  def test_converted_call_native_binding_errorneous(self):

    class FaultyBinding:

      def __array__(self):
        raise ValueError('fault')

    bad_obj = FaultyBinding()

    def fail_if_warning(*_):
      self.fail('No warning should be issued')

    with test.mock.patch.object(ag_logging, 'warning', fail_if_warning):
      with self.assertRaisesRegex(ValueError, 'fault'):
        api.converted_call(
            np.power, (bad_obj, 2), None, options=DEFAULT_RECURSIVE)

  def test_converted_call_through_tf_dataset(self):

    def other_fn(x):
      if x > 0:
        return x
      return -x

    def f():
      return dataset_ops.Dataset.range(-3, 3).map(other_fn)

    # Dataset iteration only works inside math_ops.
    @def_function.function
    def graph_fn():
      ds = api.converted_call(f, (), None, options=DEFAULT_RECURSIVE)
      itr = iter(ds)
      return next(itr), next(itr), next(itr)

    self.assertAllEqual(self.evaluate(graph_fn()), (3, 2, 1))

  def test_converted_call_no_leaks_via_closure(self):

    def test_fn():
      res = TestResource()

      def f(y):
        return res.x + y

      api.converted_call(f, (1,), None, options=DEFAULT_RECURSIVE)

    self.assertNoMemoryLeaks(test_fn)

  def test_converted_call_no_leaks_via_inner_function_closure(self):

    def test_fn():
      res = TestResource()

      def f(y):

        def inner_f():
          return res.x + y

        return inner_f

      api.converted_call(f, (1,), None, options=DEFAULT_RECURSIVE)()

    self.assertNoMemoryLeaks(test_fn)

  def test_converted_call_no_caching_on_abort(self):

    def test_fn(needs_autograph):
      if needs_autograph:
        if constant_op.constant(True):
          x = constant_op.constant(1)
        else:
          x = constant_op.constant(2)
      else:
        x = 3
      return x

    def call_in_disabled_context():
      with ag_ctx.ControlStatusCtx(status=ag_ctx.Status.DISABLED):
        return api.converted_call(
            test_fn, (False,), None, options=DEFAULT_RECURSIVE)

    def call_in_default_context():
      with ag_ctx.ControlStatusCtx(status=ag_ctx.Status.ENABLED):
        return api.converted_call(
            test_fn, (True,), None, options=DEFAULT_RECURSIVE)

    # Note: this is an invariant, not a test (see above).
    assert call_in_disabled_context() == 3

    # If api.convert placed test_fn in the unconverted cache, this second
    # invocation would fail.
    self.assertEqual(self.evaluate(call_in_default_context()), 1)

  def test_converted_call_caching_of_allowlisted_bound_methods(self):

    class TestClass:

      def __init__(self):
        self.__private = constant_op.constant(-1)

      def test_method(self):
        return self.__private

    # TODO(mdan): Refactor to avoid this use of global state.
    cache_size_before = len(conversion._ALLOWLIST_CACHE)

    # First invocation with fallback on, to allow recording it into cache.
    os.environ['AUTOGRAPH_STRICT_CONVERSION'] = '0'
    tc = TestClass()
    api.converted_call(tc.test_method, (), None, options=DEFAULT_RECURSIVE)
    os.environ['AUTOGRAPH_STRICT_CONVERSION'] = '1'

    # Entry should be added to the allowlist cache.
    self.assertEqual(len(conversion._ALLOWLIST_CACHE), cache_size_before + 1)

    # A second invocation should go through even with fallback off.
    tc = TestClass()
    api.converted_call(tc.test_method, (), None, options=DEFAULT_RECURSIVE)

    # No new entries should appear in the allowlist cache.
    self.assertEqual(len(conversion._ALLOWLIST_CACHE), cache_size_before + 1)

  def test_context_tracking_direct_calls(self):

    @api.do_not_convert()
    def unconverted_fn():
      self.assertEqual(ag_ctx.control_status_ctx().status,
                       ag_ctx.Status.DISABLED)

    @api.convert()
    def converted_fn():
      self.assertEqual(ag_ctx.control_status_ctx().status,
                       ag_ctx.Status.ENABLED)
      unconverted_fn()
      self.assertEqual(ag_ctx.control_status_ctx().status,
                       ag_ctx.Status.ENABLED)

    self.assertEqual(ag_ctx.control_status_ctx().status,
                     ag_ctx.Status.UNSPECIFIED)
    converted_fn()
    self.assertEqual(ag_ctx.control_status_ctx().status,
                     ag_ctx.Status.UNSPECIFIED)

    @api.call_with_unspecified_conversion_status
    def unspecified_fn():
      self.assertEqual(ag_ctx.control_status_ctx().status,
                       ag_ctx.Status.UNSPECIFIED)

    unspecified_fn()

  def test_to_graph_basic(self):

    def test_fn(x, s):
      while math_ops.reduce_sum(x) > s:
        x //= 2
      return x

    compiled_fn = api.to_graph(test_fn)

    with ops.Graph().as_default():
      x = compiled_fn(constant_op.constant((4, 8)), 4)
      self.assertAllEqual(self.evaluate(x), (1, 2))

  @test_util.run_deprecated_v1
  def test_to_graph_with_defaults(self):

    foo = 4

    def test_fn(x, s=foo):
      while math_ops.reduce_sum(x) > s:
        x //= 2
      return x

    compiled_fn = api.to_graph(test_fn)

    x = compiled_fn(constant_op.constant([4, 8]))
    self.assertListEqual([1, 2], self.evaluate(x).tolist())

  def test_to_graph_with_globals(self):

    def test_fn(x):
      global global_n
      global_n = x + global_n
      return global_n

    converted_fn = api.to_graph(test_fn)
    prev_val = global_n
    converted_fn(10)
    self.assertGreater(global_n, prev_val)

  def test_to_graph_with_kwargs_clashing_converted_call(self):

    def called_fn(**kwargs):
      return kwargs['f'] + kwargs['owner']

    def test_fn():
      # These arg names intentionally match converted_call's
      return called_fn(f=1, owner=2)

    compiled_fn = api.to_graph(test_fn)

    self.assertEqual(compiled_fn(), 3)

  def test_to_graph_with_kwargs_clashing_unconverted_call(self):

    @api.do_not_convert
    def called_fn(**kwargs):
      return kwargs['f'] + kwargs['owner']

    def test_fn():
      # These arg names intentionally match _call_unconverted's
      return called_fn(f=1, owner=2)

    compiled_fn = api.to_graph(test_fn)

    self.assertEqual(compiled_fn(), 3)

  def test_to_graph_caching(self):

    def test_fn(x):
      if x > 0:
        return x
      else:
        return -x

    converted_functions = tuple(api.to_graph(test_fn) for _ in (-1, 0, 1))

    # All outputs are from the same module. We can't use __module__ because
    # that's reset when we instantiate the function (see conversion.py).
    # TODO(mdan): Can and should we overwrite __module__ instead?
    module_names = frozenset(f.ag_module for f in converted_functions)
    self.assertEqual(len(module_names), 1)
    self.assertNotIn('__main__', module_names)

    self.assertEqual(len(frozenset(id(f) for f in converted_functions)), 3)

  def test_to_graph_caching_different_options(self):

    def called_fn():
      pass

    def test_fn():
      return called_fn()

    converted_recursive = api.to_graph(test_fn, recursive=True)
    converted_non_recursive = api.to_graph(test_fn, recursive=False)

    self.assertNotEqual(converted_recursive.ag_module,
                        converted_non_recursive.ag_module)
    self.assertRegex(
        tf_inspect.getsource(converted_recursive),
        'FunctionScope(.*recursive=True.*)')
    self.assertRegex(
        tf_inspect.getsource(converted_non_recursive),
        'FunctionScope(.*recursive=False.*)')

  def test_to_graph_preserves_bindings(self):
    y = 3

    def test_fn():
      return y

    converted = api.to_graph(test_fn)

    self.assertEqual(converted(), 3)

    y = 7

    self.assertEqual(converted(), 7)

  def test_to_graph_source_map(self):

    def test_fn(y):
      return y**2

    self.assertTrue(hasattr(api.to_graph(test_fn), 'ag_source_map'))

  def test_to_graph_sets_conversion_context(self):

    def g():
      self.assertEqual(ag_ctx.control_status_ctx().status,
                       ag_ctx.Status.ENABLED)
      return 0

    # Note: the autograph=False sets the connect to Status.DISABLED. The test
    # verifies that to_graph overrides that.
    @def_function.function(autograph=False)
    def f():
      converted_g = api.to_graph(g)
      converted_g()

    f()

  def test_to_code_basic(self):

    def test_fn(x, s):
      while math_ops.reduce_sum(x) > s:
        x /= 2
      return x

    # Just check that the output is parseable Python code.
    self.assertIsNotNone(parser.parse(api.to_code(test_fn)))

  def test_to_code_with_wrapped_function(self):

    @def_function.function
    def test_fn(x, s):
      while math_ops.reduce_sum(x) > s:
        x /= 2
      return x

    with self.assertRaisesRegex(Exception, 'try passing.*python_function'):
      api.to_code(test_fn)

  def test_tf_convert_overrides_current_context(self):

    def f(expect_converted):
      self.assertEqual(converter_testing.is_inside_generated_code(),
                       expect_converted)

    @api.do_not_convert
    def test_fn(ctx, expect_converted):
      return api.tf_convert(f, ctx)(expect_converted)

    test_fn(ag_ctx.ControlStatusCtx(status=ag_ctx.Status.ENABLED), True)
    test_fn(ag_ctx.ControlStatusCtx(status=ag_ctx.Status.DISABLED), False)

  def test_tf_convert_unspecified_not_converted_by_default(self):

    def f():
      self.assertEqual(ag_ctx.control_status_ctx().status,
                       ag_ctx.Status.UNSPECIFIED)
      self.assertFalse(converter_testing.is_inside_generated_code())

    @def_function.function
    def test_fn(ctx):
      return api.tf_convert(f, ctx, convert_by_default=False)()

    test_fn(ag_ctx.ControlStatusCtx(status=ag_ctx.Status.UNSPECIFIED))

  def test_tf_convert_allowlisted_method(self):

    class TestClass:

      def method(self):
        return converter_testing.is_inside_generated_code()

    converter_testing.allowlist(TestClass.method)

    obj = TestClass()
    converted_call = api.tf_convert(
        obj.method, ag_ctx.ControlStatusCtx(status=ag_ctx.Status.ENABLED))
    _, converted_target = tf_decorator.unwrap(converted_call)
    self.assertIs(converted_target.__func__, obj.method.__func__)

  def test_tf_convert_tf_decorator_unwrapping_context_enabled(self):

    def f():
      self.assertTrue(converter_testing.is_inside_generated_code())

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
      return wrapper.__wrapped__(*args, **kwargs)

    decorated_f = tf_decorator.make_decorator(f, wrapper)

    def test_fn(ctx):
      return api.tf_convert(decorated_f, ctx)()

    test_fn(ag_ctx.ControlStatusCtx(status=ag_ctx.Status.ENABLED))

  def test_tf_convert_tf_decorator_unwrapping_context_disabled(self):

    def f():
      self.assertFalse(converter_testing.is_inside_generated_code())

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
      return wrapper.__wrapped__(*args, **kwargs)

    decorated_f = tf_decorator.make_decorator(f, wrapper)

    def test_fn(ctx):
      return api.tf_convert(decorated_f, ctx)()

    test_fn(ag_ctx.ControlStatusCtx(status=ag_ctx.Status.DISABLED))

  def test_tf_convert_tf_decorator_allowlist_method(self):

    def wrap(f):

      def wrapper(*args, **kwargs):
        return wrapper.__wrapped__(*args, **kwargs)

      return tf_decorator.make_decorator(f, wrapper)

    class TestClass:

      @wrap
      def method(self):
        return converter_testing.is_inside_generated_code()

    converter_testing.allowlist(TestClass.method)

    obj = TestClass()
    # It's intended that tf_convert modifies the original method in this case.
    # This is not desirable, but options are limited.
    converted = api.tf_convert(
        obj.method, ag_ctx.ControlStatusCtx(status=ag_ctx.Status.ENABLED))
    self.assertTrue(converted())
    self.assertTrue(obj.method())

  def test_super_with_one_arg(self):
    test_case_self = self

    class TestBase:

      def plus_three(self, x):
        return x + 3

    class TestSubclass(TestBase):

      def plus_three(self, x):
        test_case_self.fail('This should never be called.')

      def one_arg(self, x):
        test_base_unbound = super(TestSubclass)
        test_base = test_base_unbound.__get__(self, TestSubclass)
        return test_base.plus_three(x)

    tc = api.converted_call(TestSubclass, (), None, options=DEFAULT_RECURSIVE)

    self.assertEqual(5, tc.one_arg(2))

  def test_super_with_two_args(self):
    test_case_self = self

    class TestBase:

      def plus_three(self, x):
        return x + 3

    class TestSubclass(TestBase):

      def plus_three(self, x):
        test_case_self.fail('This should never be called.')

      def two_args(self, x):
        return super(TestSubclass, self).plus_three(x)

    tc = api.converted_call(TestSubclass, (), None, options=DEFAULT_RECURSIVE)

    self.assertEqual(5, tc.two_args(2))

  def test_raise_from_func_graph(self):

    @def_function.function
    def raise_from_tf_function(n):
      _errors_test_helper.TestRaiseFromStatus(n)

    for code, expected_exception in [
        (1, tf_errors.CancelledError),
        (2, tf_errors.UnknownError),
        (3, tf_errors.InvalidArgumentError),
        (4, tf_errors.DeadlineExceededError),
        (5, tf_errors.NotFoundError),
        (6, tf_errors.AlreadyExistsError),
        (7, tf_errors.PermissionDeniedError),
        (16, tf_errors.UnauthenticatedError),
        (8, tf_errors.ResourceExhaustedError),
        (9, tf_errors.FailedPreconditionError),
        (10, tf_errors.AbortedError),
        (11, tf_errors.OutOfRangeError),
        (12, tf_errors.UnimplementedError),
        (13, tf_errors.InternalError),
        (14, tf_errors.UnavailableError),
        (15, tf_errors.DataLossError),
    ]:
      with self.assertRaises(expected_exception) as error:
        raise_from_tf_function(code)
      self.assertEqual(error.exception.experimental_payloads[b'key1'],
                       b'value1')
      self.assertEqual(error.exception.experimental_payloads[b'key2'],
                       b'value2')

  def test_inspect_source_unsupported(self):

    @def_function.function
    def test_func(a):
      if constant_op.constant(True):
        return a
      else:
        return a + a

    patch = test.mock.patch
    with patch.dict(os.environ, {'AUTOGRAPH_STRICT_CONVERSION': '0'}), \
         patch.object(inspect, 'findsource', side_effect=OSError()), \
         patch.object(ag_logging, 'warning') as warning_log_mock:

      with patch.object(ag_ctx, 'INSPECT_SOURCE_SUPPORTED', False):
        with self.assertRaisesRegex(tf_errors.OperatorNotAllowedInGraphError,
                                    'AutoGraph is unavailable in this runtime'):
          test_func(2)
      warning_log_mock.assert_not_called()

      with patch.object(ag_ctx, 'INSPECT_SOURCE_SUPPORTED', True):
        with self.assertRaisesRegex(tf_errors.OperatorNotAllowedInGraphError,
                                    'AutoGraph did convert this function'):
          test_func(2)
      warning_log_mock.called_once_with('AutoGraph could not transform')


if __name__ == '__main__':
  os.environ['AUTOGRAPH_STRICT_CONVERSION'] = '1'
  test.main()
