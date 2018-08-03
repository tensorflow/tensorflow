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

import numpy as np

from tensorflow.contrib.autograph import utils
from tensorflow.contrib.autograph.core import config
from tensorflow.contrib.autograph.impl import api
from tensorflow.contrib.autograph.pyct import parser
from tensorflow.contrib.autograph.utils import py_func
from tensorflow.python.framework import constant_op
from tensorflow.python.platform import test
from tensorflow.python.util import tf_inspect


tf = utils.fake_tf()


class ApiTest(test.TestCase):

  def setUp(self):
    config.COMPILED_IMPORT_STATEMENTS = (
        'from __future__ import print_function',
        'from tensorflow.contrib.autograph import utils'
        ' as autograph_utils',
        'tf = autograph_utils.fake_tf()',
    )

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
    with self.test_session() as sess:
      x = tc.test_method(
          constant_op.constant([2, 4]), constant_op.constant(1),
          constant_op.constant(-2))
      self.assertListEqual([0, 1], sess.run(x).tolist())

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
    with self.test_session() as sess:
      x = tc.test_method(
          constant_op.constant([2, 4]), constant_op.constant(1),
          constant_op.constant(-2))
      self.assertListEqual([0, 1], sess.run(x).tolist())

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
    with self.test_session() as sess:
      x = tc.test_method(
          constant_op.constant([2, 4]), constant_op.constant(1),
          constant_op.constant(-2))
      self.assertListEqual([0, 1], sess.run(x).tolist())

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
    with self.test_session() as sess:
      x = tc.test_method(
          constant_op.constant([2, 4]), constant_op.constant(1),
          constant_op.constant(-2))
      self.assertListEqual([0, 1], sess.run(x).tolist())

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
    with self.test_session() as sess:
      x = tc.test_method(
          constant_op.constant([2, 4]), constant_op.constant(1),
          constant_op.constant(-2))
      self.assertListEqual([0, 1], sess.run(x).tolist())

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
          x //= api.converted_call(self.called_member, False, False, False, {},
                                   self, a)
        return x

    tc = TestClass()
    with self.test_session() as sess:
      x = tc.test_method(
          constant_op.constant([2, 4]), constant_op.constant(1),
          constant_op.constant(-2))
      self.assertListEqual([0, 1], sess.run(x).tolist())

  def test_converted_call_builtin(self):
    x = api.converted_call(range, False, False, False, {}, 3)
    self.assertEqual((0, 1, 2), tuple(x))

  def test_converted_call_function(self):

    def test_fn(x):
      if x < 0:
        return -x
      return x

    with self.test_session() as sess:
      x = api.converted_call(test_fn, False, False, False, {},
                             constant_op.constant(-1))
      self.assertEqual(1, sess.run(x))

  def test_converted_call_method(self):

    class TestClass(object):

      def __init__(self, x):
        self.x = x

      def test_method(self):
        if self.x < 0:
          return -self.x
        return self.x

    with self.test_session() as sess:
      tc = TestClass(constant_op.constant(-1))
      x = api.converted_call(tc.test_method, False, False, False, {}, tc)
      self.assertEqual(1, sess.run(x))

  def test_converted_call_method_by_class(self):

    class TestClass(object):

      def __init__(self, x):
        self.x = x

      def test_method(self):
        if self.x < 0:
          return -self.x
        return self.x

    with self.test_session() as sess:
      tc = TestClass(constant_op.constant(-1))
      x = api.converted_call(TestClass.test_method, False, False, False, {}, tc)
      self.assertEqual(1, sess.run(x))

  def test_converted_call_callable_object(self):

    class TestClass(object):

      def __init__(self, x):
        self.x = x

      def __call__(self):
        if self.x < 0:
          return -self.x
        return self.x

    with self.test_session() as sess:
      tc = TestClass(constant_op.constant(-1))
      x = api.converted_call(tc, False, False, False, {})
      self.assertEqual(1, sess.run(x))

  def test_converted_call_constructor(self):

    class TestClass(object):

      def __init__(self, x):
        self.x = x

      def test_method(self):
        if self.x < 0:
          return -self.x
        return self.x

    with self.test_session() as sess:
      tc = api.converted_call(TestClass, False, False, False, {},
                              constant_op.constant(-1))
      # tc is now a converted object.
      x = tc.test_method()
      self.assertEqual(1, sess.run(x))

  def test_converted_call_already_converted(self):

    def f(x):
      return x == 0

    with self.test_session() as sess:
      x = api.converted_call(f, False, False, False, {},
                             constant_op.constant(0))
      self.assertTrue(sess.run(x))

      converted_f = api.to_graph(f)
      x = api.converted_call(converted_f, False, False, False, {},
                             constant_op.constant(0))
      self.assertTrue(sess.run(x))

  def test_to_graph_basic(self):

    def test_fn(x, s):
      while tf.reduce_sum(x) > s:
        x //= 2
      return x

    compiled_fn = api.to_graph(test_fn)

    with self.test_session() as sess:
      x = compiled_fn(constant_op.constant([4, 8]), 4)
      self.assertListEqual([1, 2], sess.run(x).tolist())

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


if __name__ == '__main__':
  test.main()
