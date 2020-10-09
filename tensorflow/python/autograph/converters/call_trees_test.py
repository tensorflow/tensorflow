# python3
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

import imp

from tensorflow.python.autograph.converters import call_trees
from tensorflow.python.autograph.converters import functions
from tensorflow.python.autograph.core import converter_testing
from tensorflow.python.platform import test


class MockConvertedCall(object):

  def __init__(self):
    self.calls = []

  def __call__(self, f, args, kwargs, caller_fn_scope=None, options=None):
    del caller_fn_scope, options
    self.calls.append((args, kwargs))
    kwargs = kwargs or {}
    return f(*args, **kwargs)


class CallTreesTest(converter_testing.TestCase):

  def _transform_with_mock(self, f):
    mock = MockConvertedCall()
    tr = self.transform(
        f, (functions, call_trees),
        ag_overrides={'converted_call': mock})
    return tr, mock

  def test_function_no_args(self):

    def f(f):
      return f() + 20

    tr, mock = self._transform_with_mock(f)

    self.assertEqual(tr(lambda: 1), 21)
    self.assertListEqual(mock.calls, [((), None)])

  def test_function_with_expression_in_argument(self):

    def f(f, g):
      return f(g() + 20) + 4000

    tr, mock = self._transform_with_mock(f)

    self.assertEqual(tr(lambda x: x + 300, lambda: 1), 4321)
    self.assertListEqual(mock.calls, [
        ((), None),
        ((21,), None),
    ])

  def test_function_with_call_in_argument(self):

    def f(f, g):
      return f(g()) + 300

    tr, mock = self._transform_with_mock(f)

    self.assertEqual(tr(lambda x: x + 20, lambda: 1), 321)
    self.assertListEqual(mock.calls, [
        ((), None),
        ((1,), None),
    ])

  def test_function_chaining(self):

    def get_one():
      return 1

    def f():
      return get_one().__add__(20)

    tr, mock = self._transform_with_mock(f)

    self.assertEqual(tr(), 21)
    self.assertListEqual(mock.calls, [
        ((), None),
        ((20,), None),
    ])

  def test_function_with_single_arg(self):

    def f(f, a):
      return f(a) + 20

    tr, mock = self._transform_with_mock(f)

    self.assertEqual(tr(lambda a: a, 1), 21)
    self.assertListEqual(mock.calls, [((1,), None)])

  def test_function_with_args_only(self):

    def f(f, a, b):
      return f(a, b) + 300

    tr, mock = self._transform_with_mock(f)

    self.assertEqual(tr(lambda a, b: a + b, 1, 20), 321)
    self.assertListEqual(mock.calls, [((1, 20), None)])

  def test_function_with_kwarg(self):

    def f(f, a, b):
      return f(a, c=b) + 300

    tr, mock = self._transform_with_mock(f)

    self.assertEqual(tr(lambda a, c: a + c, 1, 20), 321)
    self.assertListEqual(mock.calls, [((1,), {'c': 20})])

  def test_function_with_kwargs_starargs(self):

    def f(f, a, *args, **kwargs):
      return f(a, *args, **kwargs) + 5

    tr, mock = self._transform_with_mock(f)

    self.assertEqual(
        tr(lambda *args, **kwargs: 7, 1, *[2, 3], **{
            'b': 4,
            'c': 5
        }), 12)
    self.assertListEqual(mock.calls, [((1, 2, 3), {'b': 4, 'c': 5})])

  def test_function_with_starargs_only(self):

    def g(*args):
      return sum(args)

    def f():
      args = [1, 20, 300]
      return g(*args) + 4000

    tr, mock = self._transform_with_mock(f)

    self.assertEqual(tr(), 4321)
    self.assertListEqual(mock.calls, [((1, 20, 300), None)])

  def test_function_with_starargs_mixed(self):

    def g(a, b, c, d):
      return a * 1000 + b * 100 + c * 10 + d

    def f():
      args1 = (1,)
      args2 = [3]
      return g(*args1, 2, *args2, 4)

    tr, mock = self._transform_with_mock(f)

    self.assertEqual(tr(), 1234)
    self.assertListEqual(mock.calls, [((1, 2, 3, 4), None)])

  def test_function_with_kwargs_keywords(self):

    def f(f, a, b, **kwargs):
      return f(a, b=b, **kwargs) + 5

    tr, mock = self._transform_with_mock(f)

    self.assertEqual(
        tr(lambda *args, **kwargs: 7, 1, 2, **{'c': 3}), 12)
    self.assertListEqual(mock.calls, [((1,), {'b': 2, 'c': 3})])

  def test_function_with_multiple_kwargs(self):

    def f(f, a, b, c, kwargs1, kwargs2):
      return f(a, b=b, **kwargs1, c=c, **kwargs2) + 5

    tr, mock = self._transform_with_mock(f)

    self.assertEqual(
        tr(lambda *args, **kwargs: 7, 1, 2, 3, {'d': 4}, {'e': 5}), 12)
    self.assertListEqual(mock.calls, [((1,), {
        'b': 2,
        'c': 3,
        'd': 4,
        'e': 5
    })])

  def test_function_with_call_in_lambda_argument(self):

    def h(l, a):
      return l(a) + 4000

    def g(a, *args):
      return a + sum(args)

    def f(h, g, a, *args):
      return h(lambda x: g(x, *args), a)

    tr, _ = self._transform_with_mock(f)

    self.assertEqual(tr(h, g, 1, *(20, 300)), 4321)

  def test_debugger_set_trace(self):

    tracking_list = []

    pdb = imp.new_module('fake_pdb')
    pdb.set_trace = lambda: tracking_list.append(1)

    def f():
      return pdb.set_trace()

    tr, _ = self._transform_with_mock(f)

    tr()
    self.assertListEqual(tracking_list, [1])

  def test_class_method(self):

    class TestClass(object):

      def other_method(self, x):
        return x + 20

      def test_method(self, a):
        return self.other_method(a) + 300

    tc = TestClass()
    tr, mock = self._transform_with_mock(TestClass.test_method)

    self.assertEqual(321, tr(tc, 1))
    self.assertListEqual(mock.calls, [((1,), None)])

  def test_object_method(self):

    class TestClass(object):

      def other_method(self, x):
        return x + 20

      def test_method(self, a):
        return self.other_method(a) + 300

    tc = TestClass()
    tr, mock = self._transform_with_mock(tc.test_method)

    self.assertEqual(321, tr(tc, 1))
    self.assertListEqual(mock.calls, [((1,), None)])


if __name__ == '__main__':
  test.main()
