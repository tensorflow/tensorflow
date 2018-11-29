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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import gc

import numpy as np

from tensorflow.python.autograph import utils
from tensorflow.python.autograph.core import converter
from tensorflow.python.autograph.impl import api
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.utils import py_func
from tensorflow.python.framework import constant_op
from tensorflow.python.keras.engine import sequential
from tensorflow.python.keras.layers import core
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.util import tf_inspect

tf = utils.fake_tf()


class TestResource(str):
  pass


class ApiTest(test.TestCase):

  def test_decorator_recurses(self):

    class TestClass(object):

      def called_member(self, a):
        if a < 0:
          a = -a
        return a

      @api.convert(recursive=True)
      def test_method(self, x, s, a):
        while tf.reduce_sum(x) > s:
          x //= self.called_member(a)
        return x

    tc = TestClass()
    with self.cached_session() as sess:
      x = tc.test_method(
          constant_op.constant([2, 4]), constant_op.constant(1),
          constant_op.constant(-2))
      self.assertListEqual([0, 1], self.evaluate(x).tolist())

  def test_decorator_does_not_recurse(self):

    class TestClass(object):

      def called_member(self, a):
        return tf.negative(a)

      @api.convert(recursive=False)
      def test_method(self, x, s, a):
        while tf.reduce_sum(x) > s:
          x //= self.called_member(a)
        return x

    tc = TestClass()
    with self.cached_session() as sess:
      x = tc.test_method(
          constant_op.constant([2, 4]), constant_op.constant(1),
          constant_op.constant(-2))
      self.assertListEqual([0, 1], self.evaluate(x).tolist())

  def test_decorator_calls_unconverted_graph(self):

    class TestClass(object):

      @api.do_not_convert(api.RunMode.GRAPH)
      def called_member(self, a):
        return tf.negative(a)

      @api.convert(recursive=True)
      def test_method(self, x, s, a):
        while tf.reduce_sum(x) > s:
          x //= self.called_member(a)
        return x

    tc = TestClass()
    with self.cached_session() as sess:
      x = tc.test_method(
          constant_op.constant([2, 4]), constant_op.constant(1),
          constant_op.constant(-2))
      self.assertListEqual([0, 1], self.evaluate(x).tolist())

  def test_decorator_calls_unconverted_py_func(self):

    class TestClass(object):

      @api.do_not_convert(
          api.RunMode.PY_FUNC, return_dtypes=py_func.MatchDType(1))
      def called_member(self, a):
        return np.negative(a)

      @api.convert(recursive=True)
      def test_method(self, x, s, a):
        while tf.reduce_sum(x) > s:
          y = self.called_member(a)
          # set_shape works around while_loop's limitations.
          # TODO(mdan): Allow specifying shapes (or ShapeLike) instead.
          y.set_shape(a.shape)
          x //= y
        return x

    tc = TestClass()
    with self.cached_session() as sess:
      x = tc.test_method(
          constant_op.constant([2, 4]), constant_op.constant(1),
          constant_op.constant(-2))
      self.assertListEqual([0, 1], self.evaluate(x).tolist())

  def test_decorator_calls_decorated(self):

    class TestClass(object):

      @api.convert()
      def called_member(self, a):
        if a < 0:
          a = -a
        return a

      @api.convert(recursive=True)
      def test_method(self, x, s, a):
        while tf.reduce_sum(x) > s:
          x //= self.called_member(a)
        return x

    tc = TestClass()
    with self.cached_session() as sess:
      x = tc.test_method(
          constant_op.constant([2, 4]), constant_op.constant(1),
          constant_op.constant(-2))
      self.assertListEqual([0, 1], self.evaluate(x).tolist())

  def test_decorator_preserves_argspec(self):

    class TestClass(object):

      def called_member(self, a):
        if a < 0:
          a = -a
        return a

      called_member_converted = api.convert()(called_member)

    tc = TestClass()
    self.assertListEqual(
        list(tf_inspect.getfullargspec(tc.called_member)),
        list(tf_inspect.getfullargspec(tc.called_member_converted)))

  def test_convert_call_site_decorator(self):

    class TestClass(object):

      def called_member(self, a):
        if a < 0:
          a = -a
        return a

      @api.convert(recursive=True)
      def test_method(self, x, s, a):
        while tf.reduce_sum(x) > s:
          x //= api.converted_call(self.called_member, None,
                                   converter.ConversionOptions(), self, a)
        return x

    tc = TestClass()
    with self.cached_session() as sess:
      x = tc.test_method(
          constant_op.constant([2, 4]), constant_op.constant(1),
          constant_op.constant(-2))
      self.assertListEqual([0, 1], self.evaluate(x).tolist())

  def test_converted_call_builtin(self):
    x = api.converted_call(range, None, converter.ConversionOptions(), 3)
    self.assertEqual((0, 1, 2), tuple(x))

  def test_converted_call_function(self):

    def test_fn(x):
      if x < 0:
        return -x
      return x

    with self.cached_session() as sess:
      x = api.converted_call(test_fn, None, converter.ConversionOptions(),
                             constant_op.constant(-1))
      self.assertEqual(1, self.evaluate(x))

  def test_converted_call_functools_partial(self):

    def test_fn(x, y, z):
      if x < 0:
        return -x, -y, -z
      return x, y, z

    x = api.converted_call(
        functools.partial(test_fn, constant_op.constant(-1), z=-3),
        None, converter.ConversionOptions(),
        constant_op.constant(-2))
    self.assertEqual((1, 2, 3), self.evaluate(x))

    x = api.converted_call(
        functools.partial(
            functools.partial(test_fn, constant_op.constant(-1)), z=-3),
        None, converter.ConversionOptions(),
        constant_op.constant(-2))
    self.assertEqual((1, 2, 3), self.evaluate(x))

  def test_converted_call_method_explicit_owner(self):
    # TODO(mdan): Implement.
    pass

  def test_converted_call_method_explicit_super_owner(self):
    # TODO(mdan): Implement.
    pass

  def test_converted_call_method(self):

    class TestClass(object):

      def __init__(self, x):
        self.x = x

      def test_method(self):
        if self.x < 0:
          return -self.x
        return self.x

    with self.cached_session() as sess:
      tc = TestClass(constant_op.constant(-1))
      x = api.converted_call(tc.test_method, None,
                             converter.ConversionOptions(), tc)
      self.assertEqual(1, self.evaluate(x))

  def test_converted_call_method_by_class(self):

    class TestClass(object):

      def __init__(self, x):
        self.x = x

      def test_method(self):
        if self.x < 0:
          return -self.x
        return self.x

    with self.cached_session() as sess:
      tc = TestClass(constant_op.constant(-1))
      x = api.converted_call(TestClass.test_method, None,
                             converter.ConversionOptions(), tc)
      self.assertEqual(1, self.evaluate(x))

  def test_converted_call_callable_object(self):

    class TestClass(object):

      def __init__(self, x):
        self.x = x

      def __call__(self):
        if self.x < 0:
          return -self.x
        return self.x

    with self.cached_session() as sess:
      tc = TestClass(constant_op.constant(-1))
      x = api.converted_call(tc, None, converter.ConversionOptions())
      self.assertEqual(1, self.evaluate(x))

  def test_converted_call_constructor(self):

    class TestClass(object):

      def __init__(self, x):
        self.x = x

      def test_method(self):
        if self.x < 0:
          return -self.x
        return self.x

    with self.cached_session() as sess:
      tc = api.converted_call(TestClass, None, converter.ConversionOptions(),
                              constant_op.constant(-1))
      # tc is now a converted object.
      x = tc.test_method()
      self.assertEqual(1, self.evaluate(x))

  def test_converted_call_already_converted(self):

    def f(x):
      return x == 0

    with self.cached_session() as sess:
      x = api.converted_call(f, None, converter.ConversionOptions(),
                             constant_op.constant(0))
      self.assertTrue(self.evaluate(x))

      converted_f = api.to_graph(f)
      x = api.converted_call(converted_f, None, converter.ConversionOptions(),
                             constant_op.constant(0))
      self.assertTrue(self.evaluate(x))

  def test_converted_call_no_user_code(self):

    def f(x):
      return len(x)

    opts = converter.ConversionOptions(internal_convert_user_code=False)

    # f should not be converted, causing len to error out.
    with self.assertRaisesRegexp(Exception,
                                 'object of type \'Tensor\' has no len()'):
      api.converted_call(f, None, opts, constant_op.constant([0]))

    # len on the other hand should work fine.
    x = api.converted_call(len, None, opts, constant_op.constant([0]))
    # The constant has static shape so the result is a primitive not a Tensor.
    self.assertEqual(x, 1)

  def test_converted_call_whitelisted_method(self):

    opts = converter.ConversionOptions()

    model = sequential.Sequential([
        core.Dense(2)
    ])

    x = api.converted_call(model.call, None, opts,
                           constant_op.constant([[0.0]]), training=True)

    with self.cached_session() as sess:
      self.evaluate(variables.global_variables_initializer())
      self.assertAllEqual([[0.0, 0.0]], self.evaluate(x))

  def test_converted_call_whitelisted_method_extra_self(self):

    opts = converter.ConversionOptions()

    model = sequential.Sequential([
        core.Dense(2)
    ])

    x = api.converted_call(model.call, None, opts,
                           model, constant_op.constant([[0.0]]), training=True)

    with self.cached_session() as sess:
      self.evaluate(variables.global_variables_initializer())
      self.assertAllEqual([[0.0, 0.0]], self.evaluate(x))

  def test_converted_call_whitelisted_method_via_owner(self):

    opts = converter.ConversionOptions()

    model = sequential.Sequential([
        core.Dense(2)
    ])

    x = api.converted_call('call', model, opts,
                           constant_op.constant([[0.0]]), training=True)

    with self.cached_session() as sess:
      self.evaluate(variables.global_variables_initializer())
      self.assertAllEqual([[0.0, 0.0]], self.evaluate(x))

  def test_converted_call_lambda(self):

    opts = converter.ConversionOptions()

    l = lambda x: x == 0

    x = api.converted_call(l, None, opts, constant_op.constant(0))

    with self.cached_session() as sess:
      self.evaluate(variables.global_variables_initializer())
      self.assertAllEqual(True, self.evaluate(x))

  def test_to_graph_basic(self):

    def test_fn(x, s):
      while tf.reduce_sum(x) > s:
        x //= 2
      return x

    compiled_fn = api.to_graph(test_fn)

    with self.cached_session() as sess:
      x = compiled_fn(constant_op.constant([4, 8]), 4)
      self.assertListEqual([1, 2], self.evaluate(x).tolist())

  def test_to_graph_with_defaults(self):

    foo = 4

    def test_fn(x, s=foo):
      while tf.reduce_sum(x) > s:
        x //= 2
      return x

    compiled_fn = api.to_graph(test_fn)

    with self.cached_session() as sess:
      x = compiled_fn(constant_op.constant([4, 8]))
      self.assertListEqual([1, 2], self.evaluate(x).tolist())

  def test_to_code_basic(self):

    def test_fn(x, s):
      while tf.reduce_sum(x) > s:
        x /= 2
      return x

    compiled_code = api.to_code(test_fn)

    # Just check that it is parseable Python code.
    self.assertIsNotNone(parser.parse_str(compiled_code))

  def test_source_map_attribute_present(self):

    def test_fn(y):
      return y**2

    self.assertTrue(hasattr(api.to_graph(test_fn), 'ag_source_map'))

  def assertNoMemoryLeaks(self, target_f):
    refs_before = set(id(obj) for obj in gc.get_objects())
    target_f()
    gc.collect()
    objs_after = [obj for obj in gc.get_objects() if id(obj) not in refs_before]
    leaked = [obj for obj in objs_after if isinstance(obj, TestResource)]
    self.assertFalse(leaked,
                     'Resources {} were leaked by AutoGraph.'.format(leaked))

  def test_no_module_memory_leak(self):
    def f():
      resource = TestResource('some-resource')
      @api.convert()
      def target(x):
        return x + resource, 42
      self.assertEqual(target('foo'), ('foosome-resource', 42))

    self.assertNoMemoryLeaks(f)

  def test_no_module_memory_leak_deferred_call(self):
    def f():
      resource = TestResource('some-resource')
      @api.convert()
      def target(x):
        def inner_fn():
          return x + resource
        return inner_fn, 42
      self.assertEqual(target('foo')[0](), 'foosome-resource')

    f()
    # TODO(brianklee): Reenable when we've revised module loading approach.
    # self.assertNoMemoryLeaks(f)


if __name__ == '__main__':
  test.main()
