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
from tensorflow.contrib.autograph.impl import api
from tensorflow.contrib.autograph.impl import config
from tensorflow.contrib.autograph.pyct import parser
from tensorflow.contrib.autograph.utils import py_func
from tensorflow.python.framework import constant_op
from tensorflow.python.platform import test


tf = utils.fake_tf()


class ApiTest(test.TestCase):

  def setUp(self):
    config.COMPILED_IMPORT_STATEMENTS = (
        'from __future__ import print_function',
        'from tensorflow.contrib.autograph import utils as '
        'autograph_utils', 'tf = autograph_utils.fake_tf()')

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

  def test_convert_call_site_decorator(self):

    class TestClass(object):

      def called_member(self, a):
        if a < 0:
          a = -a
        return a

      @api.convert(recursive=True)
      def test_method(self, x, s, a):
        while tf.reduce_sum(x) > s:
          x //= api.converted_call(self.called_member, False, False, {}, self,
                                   a)
        return x

    tc = TestClass()
    with self.test_session() as sess:
      x = tc.test_method(
          constant_op.constant([2, 4]), constant_op.constant(1),
          constant_op.constant(-2))
      self.assertListEqual([0, 1], sess.run(x).tolist())

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

    # Just check for some key words and that it is parseable Python code.
    self.assertRegexpMatches(compiled_code, 'autograph_utils\\.run_while')
    self.assertIsNotNone(parser.parse_str(compiled_code))


if __name__ == '__main__':
  test.main()
