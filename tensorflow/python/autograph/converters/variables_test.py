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

from tensorflow.python.autograph.converters import variables
from tensorflow.python.autograph.core import converter_testing
from tensorflow.python.platform import test


class VariablesTest(converter_testing.TestCase):

  def transform_with_test_ld(self, f):
    """Generates code which adds 1 to all variable reads."""
    return self.transform(f, variables, ag_overrides={'ld': lambda x: x + 1})

  def test_read(self):

    def f(l):
      return l

    tr = self.transform_with_test_ld(f)

    self.assertEqual(tr(1), 2)

  def test_aug_assign(self):

    def f(l):
      l *= 10
      return l

    tr = self.transform_with_test_ld(f)

    self.assertEqual(tr(1), (1 + 1) * 10 + 1)  # two reads

  def test_del(self):

    def f(l):
      del l
      return l

    tr = self.transform(f, variables)

    with self.assertRaisesRegex(NameError, "'l' is used before assignment"):
      tr(1)

  def test_del_getitem_ignored_basic_slice(self):

    def f(l):
      del l[0]
      return l

    tr = self.transform(f, variables)

    self.assertListEqual([2], tr([1, 2]))

  def test_del_getitem_ignored_range_slice(self):

    def f(l):
      del l[0:2]
      return l

    tr = self.transform(f, variables)

    self.assertListEqual([], tr([1, 2]))

  def test_del_getattr_ignored(self):

    def f(l):
      del l.a
      return l

    class TestClass(object):

      def __init__(self):
        self.a = 1
        self.b = 2

    tr = self.transform(f, variables)

    self.assertFalse(hasattr(tr(TestClass()), 'a'))
    self.assertEqual(tr(TestClass()).b, 2)

  def test_del_packing_ignored_list(self):
    # Note: testing for UnboundLocalError, not NameError because in this case we
    # don't rewrite the del.

    def f(a, b):
      del [a, b]
      return a

    tr = self.transform(f, variables)

    with self.assertRaises(UnboundLocalError):
      tr(1, 2)

  def test_del_packing_ignored_nested(self):
    # Note: testing for UnboundLocalError, not NameError because in this case we
    # don't rewrite the del.

    def f(a, b, c):
      del [a, (b, c)]
      return c

    tr = self.transform(f, variables)

    with self.assertRaises(UnboundLocalError):
      tr(1, 2, 3)

  def test_del_item_multiple_mixed_used_after(self):

    def f(a, b, c):
      del a, b, c[0]
      a = 1
      return a, b, c

    tr = self.transform(f, variables)

    with self.assertRaisesRegex(NameError, "'b' is used before assignment"):
      tr(1, 2, [1, 2])

  def test_del_item_multiple_mixed_unused_after(self):

    def f(a, b, c):
      del a, b, c[0]
      a = 1
      b = 2
      return c

    tr = self.transform(f, variables)

    self.assertListEqual([2], tr(1, 2, [1, 2]))

  def test_attribute(self):

    class TestClass(object):

      def __init__(self):
        self.v = 1

      def __add__(self, other):
        self.v += other
        return self

    def f(l):
      return l.v

    tc = TestClass()
    tr = self.transform_with_test_ld(f)

    self.assertEqual(tr(tc), 2)

  def test_subscript(self):

    class TestClass(object):

      def __init__(self):
        self.v = 1

      def __add__(self, other):
        self.v += other
        return self

      def __getitem__(self, _):
        return self.v

    def f(l):
      return l[0]

    tc = TestClass()
    tr = self.transform_with_test_ld(f)

    self.assertEqual(tr(tc), 2)

  def test_call(self):

    class TestClass(object):

      def __init__(self):
        self.v = 1

      def __add__(self, other):
        self.v += other
        return self

      def __call__(self):
        return self.v

    def f(l):
      return l()

    tc = TestClass()
    tr = self.transform_with_test_ld(f)

    self.assertEqual(tr(tc), 2)


if __name__ == '__main__':
  test.main()
