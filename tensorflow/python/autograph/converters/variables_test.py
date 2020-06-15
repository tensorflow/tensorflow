# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for variables module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib

from tensorflow.python.autograph.converters import variables
from tensorflow.python.autograph.core import converter_testing
from tensorflow.python.platform import test


class VariablesTest(converter_testing.TestCase):

  @contextlib.contextmanager
  def apply_add_one_conversion(self, fn):
    """Generates code which adds 1 to all variable reads."""
    with self.converted(fn, variables, {}) as result:
      result.ag__.__dict__['ld'] = lambda x: x + 1
      yield result

  def test_read(self):

    def test_fn(l):
      return l

    with self.apply_add_one_conversion(test_fn) as result:
      self.assertEqual(result.test_fn(1), 2)

  def test_aug_assign(self):

    def test_fn(l):
      l *= 10
      return l

    with self.apply_add_one_conversion(test_fn) as result:
      self.assertEqual(result.test_fn(1), (1 + 1) * 10 + 1)  # two reads

  def test_del(self):

    def test_fn(l):
      del l
      return l

    with self.converted(test_fn, variables, {}) as result:
      with self.assertRaisesRegex(
          NameError, "'l' is used before assignment"):
        result.test_fn(1)

  def test_del_getitem_ignored(self):

    def basic_slice(l):
      del l[0]
      return l

    with self.converted(basic_slice, variables, {}) as result:
      self.assertListEqual([2], result.basic_slice([1, 2]))

    def range_slice(l):
      del l[0:2]
      return l

    with self.converted(range_slice, variables, {}) as result:
      self.assertListEqual([], result.range_slice([1, 2]))

  def test_del_getattr_ignored(self):

    def test_fn(l):
      del l.a
      return l

    class TestClass(object):

      def __init__(self):
        self.a = 1
        self.b = 2

    with self.converted(test_fn, variables, {}) as result:
      self.assertFalse(hasattr(result.test_fn(TestClass()), 'a'))
      self.assertEqual(result.test_fn(TestClass()).b, 2)

  def test_del_packing_ignored(self):
    # Note: test for UnboundLocalError, not NameError because in this case we
    # don't rewrite the del.

    def list_(a, b):
      del [a, b]
      return a

    with self.converted(list_, variables, {}) as result:
      with self.assertRaises(UnboundLocalError):
        result.list_(1, 2)

    def nested(a, b, c):
      del [a, (b, c)]
      return c

    with self.converted(nested, variables, {}) as result:
      with self.assertRaises(UnboundLocalError):
        result.nested(1, 2, 3)

  def test_del_item_multiple_mixed(self):

    def test_fn_failing(a, b, c):
      del a, b, c[0]
      a = 1
      return a, b, c

    with self.converted(test_fn_failing, variables, {}) as result:
      with self.assertRaisesRegex(
          NameError, "'b' is used before assignment"):
        result.test_fn_failing(1, 2, [1, 2])

    def test_fn_passing(a, b, c):
      del a, b, c[0]
      a = 1
      b = 2
      return c

    with self.converted(test_fn_passing, variables, {}) as result:
      self.assertListEqual([2], result.test_fn_passing(1, 2, [1, 2]))

  def test_attribute(self):

    class TestClass(object):

      def __init__(self):
        self.v = 1

      def __add__(self, other):
        self.v += other
        return self

    def test_fn(l):
      return l.v

    tc = TestClass()
    with self.apply_add_one_conversion(test_fn) as result:
      self.assertEqual(result.test_fn(tc), 2)

  def test_subscript(self):

    class TestClass(object):

      def __init__(self):
        self.v = 1

      def __add__(self, other):
        self.v += other
        return self

      def __getitem__(self, _):
        return self.v

    def test_fn(l):
      return l[0]

    tc = TestClass()
    with self.apply_add_one_conversion(test_fn) as result:
      self.assertEqual(result.test_fn(tc), 2)

  def test_call(self):

    class TestClass(object):

      def __init__(self):
        self.v = 1

      def __add__(self, other):
        self.v += other
        return self

      def __call__(self):
        return self.v

    def test_fn(l):
      return l()

    tc = TestClass()
    with self.apply_add_one_conversion(test_fn) as result:
      self.assertEqual(result.test_fn(tc), 2)


if __name__ == '__main__':
  test.main()
