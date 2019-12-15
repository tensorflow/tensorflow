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
"""Tests for wrap_py_func module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.autograph.utils import py_func
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.platform import test


class PyFuncTest(test.TestCase):

  def test_wrap_py_func_simple(self):

    def test_fn(a, b, c):
      return a + b + c

    with self.cached_session() as sess:
      result = py_func.wrap_py_func(test_fn, dtypes.int32,
                                    (1, constant_op.constant(1), 1))
      self.assertEqual(3, self.evaluate(result))
      result = py_func.wrap_py_func(test_fn, dtypes.int32, (1, 1, 1))
      self.assertEqual(3, self.evaluate(result))
      result = py_func.wrap_py_func(
          test_fn, dtypes.int32,
          (constant_op.constant(1), 1, constant_op.constant(1)))
      self.assertEqual(3, self.evaluate(result))

  def test_wrap_py_func_complex_args(self):

    class TestClass(object):

      def __init__(self):
        self.foo = 5

    def test_fn(a, b):
      return a * b.foo

    with self.cached_session() as sess:
      result = py_func.wrap_py_func(test_fn, dtypes.int32, (7, TestClass()))
      self.assertEqual(35, self.evaluate(result))
      result = py_func.wrap_py_func(test_fn, dtypes.int32,
                                    (constant_op.constant(7), TestClass()))
      self.assertEqual(35, self.evaluate(result))

  def test_wrap_py_func_kwargs(self):

    class TestClass(object):

      def __init__(self, foo):
        self.foo = foo

    def test_fn(a, b, c, d):
      return a * b.foo + c * d.foo

    with self.cached_session() as sess:
      result = py_func.wrap_py_func(test_fn, dtypes.int32, (7, TestClass(5)), {
          'c': 11,
          'd': TestClass(13)
      })
      self.assertEqual(178, self.evaluate(result))
      result = py_func.wrap_py_func(test_fn, dtypes.int32,
                                    (constant_op.constant(7), TestClass(5)), {
                                        'c': constant_op.constant(11),
                                        'd': TestClass(13)
                                    })
      self.assertEqual(178, self.evaluate(result))

  def test_wrap_py_func_dummy_return(self):

    side_counter = [0]

    def test_fn(_):
      side_counter[0] += 1

    with self.cached_session() as sess:
      result = py_func.wrap_py_func(test_fn, None, (5,), use_dummy_return=True)
      self.assertEqual(1, self.evaluate(result))
      self.assertEqual([1], side_counter)
      result = py_func.wrap_py_func(
          test_fn, None, (constant_op.constant(5),), use_dummy_return=True)
      self.assertEqual(1, self.evaluate(result))
      self.assertEqual([2], side_counter)


if __name__ == '__main__':
  test.main()
