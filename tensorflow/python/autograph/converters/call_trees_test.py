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
"""Tests for call_trees module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.autograph.converters import call_trees
from tensorflow.python.autograph.core import converter_testing
from tensorflow.python.platform import test


class CallTreesTest(converter_testing.TestCase):

  def test_normal_function(self):

    def test_fn(f):
      return f() + 3

    with self.converted(test_fn, call_trees, {}) as result:
      self.assertEqual(
          result.test_fn(None),
          converter_testing.RESULT_OF_MOCK_CONVERTED_CALL + 3)
      self.assertListEqual(self.dynamic_calls, [((), None)])

  def test_function_with_expression_in_argument(self):

    def test_fn(f, g):
      return f(g() + 7) + 3

    with self.converted(test_fn, call_trees, {}) as result:
      self.assertEqual(
          result.test_fn(None, None),
          converter_testing.RESULT_OF_MOCK_CONVERTED_CALL + 3)
      self.assertListEqual(self.dynamic_calls, [
          ((), None),
          ((converter_testing.RESULT_OF_MOCK_CONVERTED_CALL + 7,), None),
      ])

  def test_function_with_call_in_argument(self):

    def test_fn(f, g):
      return f(g()) + 3

    with self.converted(test_fn, call_trees, {}) as result:
      self.assertEqual(
          result.test_fn(None, None),
          converter_testing.RESULT_OF_MOCK_CONVERTED_CALL + 3)
      self.assertListEqual(self.dynamic_calls, [
          ((), None),
          ((converter_testing.RESULT_OF_MOCK_CONVERTED_CALL,), None),
      ])

  def test_function_with_kwarg(self):

    def test_fn(f, a, b):
      return f(a, c=b) + 3

    with self.converted(test_fn, call_trees, {}) as result:
      self.assertEqual(
          result.test_fn(None, 1, 2),
          converter_testing.RESULT_OF_MOCK_CONVERTED_CALL + 3)
      self.assertListEqual(self.dynamic_calls, [((1,), {'c': 2})])

  def test_function_with_kwargs_starargs(self):

    def test_fn(f, a, *args, **kwargs):
      return f(a, *args, **kwargs) + 5

    with self.converted(test_fn, call_trees, {}) as result:
      self.assertEqual(
          result.test_fn(None, 1, *[2, 3], **{
              'b': 4,
              'c': 5
          }), converter_testing.RESULT_OF_MOCK_CONVERTED_CALL + 5)
      self.assertListEqual(self.dynamic_calls, [((1, 2, 3), {'b': 4, 'c': 5})])

  def test_function_with_kwargs_starargs_only(self):

    def f(*unused_args):  # Will not be called.
      pass

    def test_fn():
      args = [1, 2, 3]
      return f(*args) + 11

    with self.converted(test_fn, call_trees, {'f': f}) as result:
      self.assertEqual(result.test_fn(),
                       converter_testing.RESULT_OF_MOCK_CONVERTED_CALL + 11)
      self.assertListEqual(self.dynamic_calls, [((1, 2, 3), None)])

  def test_function_with_kwargs_keywords(self):

    def test_fn(f, a, b, **kwargs):
      return f(a, b=b, **kwargs) + 5

    with self.converted(test_fn, call_trees, {}) as result:
      self.assertEqual(
          result.test_fn(None, 1, 2, **{'c': 3}),
          converter_testing.RESULT_OF_MOCK_CONVERTED_CALL + 5)
      self.assertListEqual(self.dynamic_calls, [((1,), {'b': 2, 'c': 3})])

  def test_class_method(self):

    class TestClass(object):

      def test_method(self, a):
        return self.other_method(a) + 1

    tc = TestClass()
    with self.converted(TestClass.test_method, call_trees, {}) as result:
      self.assertEqual(converter_testing.RESULT_OF_MOCK_CONVERTED_CALL + 1,
                       result.test_method(tc, 1))
      self.assertListEqual(self.dynamic_calls, [((1,), None)])

  def test_object_method(self):

    class TestClass(object):

      def test_method(self, a):
        return self.other_method(a) + 1

    tc = TestClass()
    with self.converted(tc.test_method, call_trees, {}) as result:
      self.assertEqual(converter_testing.RESULT_OF_MOCK_CONVERTED_CALL + 1,
                       result.test_method(tc, 1))
      self.assertListEqual(self.dynamic_calls, [((1,), None)])


if __name__ == '__main__':
  test.main()
