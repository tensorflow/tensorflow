# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for special symbol handling."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.autograph.operators import special_values
from tensorflow.python.autograph.operators import symbols
from tensorflow.python.platform import test

Undefined = special_values.Undefined
AttributeAccessSymbol = symbols.AttributeAccessSymbol
SubscriptSymbol = symbols.SubscriptSymbol
ValueSymbol = symbols.ValueSymbol


class SymbolsTest(test.TestCase):

  def test_value_symbol_returns_value(self):
    a = 42
    a_symbol = ValueSymbol('a', a)
    self.assertEqual(a_symbol.maybe_compute_value(), a)
    self.assertEqual(a_symbol.name, 'a')

  def test_attribute_access_missing_attribute(self):
    class Foo(object):
      pass
    a = Foo()

    a_symbol = ValueSymbol('a', a)
    a_b_symbol = AttributeAccessSymbol(a_symbol, 'b')

    self.assertEqual(a_symbol.maybe_compute_value(), a)
    self.assertIsInstance(a_b_symbol.maybe_compute_value(), Undefined)
    self.assertEqual(a_b_symbol.maybe_compute_value().symbol_name, 'a.b')

  def test_attribute_access_undefined_target(self):
    a = Undefined('a')
    a_symbol = ValueSymbol('a', a)
    a_b_symbol = AttributeAccessSymbol(a_symbol, 'b')

    self.assertEqual(a_symbol.maybe_compute_value(), a)
    self.assertIsInstance(a_b_symbol.maybe_compute_value(), Undefined)
    self.assertEqual(a_b_symbol.maybe_compute_value().symbol_name, 'a.b')

  def test_attribute_access_basic(self):
    class Foo(object):

      def __init__(self):
        self.b = 'this is an attribute'

    a = Foo()
    a_symbol = ValueSymbol('a', a)
    a_b_symbol = AttributeAccessSymbol(a_symbol, 'b')

    self.assertEqual(a_symbol.maybe_compute_value(), a)
    self.assertEqual(a_b_symbol.maybe_compute_value(), a.b)

  def test_item_access_undefined_index(self):
    class Foo(object):

      def __getitem__(self, key):
        return 'this is an item'

    a = Foo()
    b = Undefined('b')
    a_symbol = ValueSymbol('a', a)
    b_symbol = ValueSymbol('b', b)
    a_b_symbol = SubscriptSymbol(a_symbol, b_symbol)

    self.assertEqual(a_symbol.maybe_compute_value(), a)
    self.assertEqual(b_symbol.maybe_compute_value(), b)
    self.assertIsInstance(a_b_symbol.maybe_compute_value(), Undefined)
    self.assertEqual(a_b_symbol.maybe_compute_value().symbol_name, 'a[b]')

  def test_item_access_no_getitem(self):
    class Foo(object):
      pass

    a = Foo()
    b = 42
    a_symbol = ValueSymbol('a', a)
    b_symbol = ValueSymbol('b', b)
    a_b_symbol = SubscriptSymbol(a_symbol, b_symbol)

    self.assertEqual(a_symbol.maybe_compute_value(), a)
    self.assertEqual(b_symbol.maybe_compute_value(), b)
    self.assertIsInstance(a_b_symbol.maybe_compute_value(), Undefined)
    self.assertEqual(a_b_symbol.maybe_compute_value().symbol_name, 'a[b]')

  def test_item_access_undefined_root(self):
    a = Undefined('a')
    b = 42
    a_symbol = ValueSymbol('a', a)
    b_symbol = ValueSymbol('b', b)
    a_b_symbol = SubscriptSymbol(a_symbol, b_symbol)

    self.assertEqual(a_symbol.maybe_compute_value(), a)
    self.assertEqual(b_symbol.maybe_compute_value(), b)
    self.assertIsInstance(a_b_symbol.maybe_compute_value(), Undefined)
    self.assertEqual(a_b_symbol.maybe_compute_value().symbol_name, 'a[b]')

  def test_item_access_basic(self):
    class Foo(object):

      def __getitem__(self, key):
        return 'this is an item'

    a = Foo()
    b = 42
    a_symbol = ValueSymbol('a', a)
    b_symbol = ValueSymbol('b', b)
    a_b_symbol = SubscriptSymbol(a_symbol, b_symbol)

    self.assertEqual(a_symbol.maybe_compute_value(), a)
    self.assertEqual(b_symbol.maybe_compute_value(), b)
    self.assertEqual(a_b_symbol.maybe_compute_value(), a[b])

  def test_item_access_after_attribute_access(self):
    class Foo(object):

      def __getitem__(self, key):
        return 'this is an item'

    class Bar(object):

      def __init__(self):
        self.b = Foo()

    a = Bar()
    c = 42
    a_symbol = ValueSymbol('a', a)
    c_symbol = ValueSymbol('c', c)
    a_b_symbol = AttributeAccessSymbol(a_symbol, 'b')
    a_b_c_symbol = SubscriptSymbol(a_b_symbol, c_symbol)

    self.assertEqual(a_symbol.maybe_compute_value(), a)
    self.assertEqual(c_symbol.maybe_compute_value(), c)
    self.assertEqual(a_b_symbol.maybe_compute_value(), a.b)
    self.assertEqual(a_b_c_symbol.maybe_compute_value(), a.b[c])

  def test_attribute_access_after_item_access(self):
    class Bar(object):

      def __init__(self):
        self.c = object()

    item = Bar()

    class Foo(object):

      def __getitem__(self, key):
        return item

    a = Foo()
    b = 42
    a_symbol = ValueSymbol('a', a)
    b_symbol = ValueSymbol('b', b)
    a_b_symbol = SubscriptSymbol(a_symbol, b_symbol)
    a_b_c_symbol = AttributeAccessSymbol(a_b_symbol, 'c')

    self.assertEqual(a_symbol.maybe_compute_value(), a)
    self.assertEqual(b_symbol.maybe_compute_value(), b)
    self.assertEqual(a_b_symbol.maybe_compute_value(), a[b])
    self.assertEqual(a_b_c_symbol.maybe_compute_value(), a[b].c)

  def test_item_access_after_item_access(self):
    class Bar(object):

      def __getitem__(self, key):
        return 'this is an item'

    item = Bar()

    class Foo(object):

      def __getitem__(self, key):
        return item

    a = Foo()
    b = 42
    c = 43
    a_symbol = ValueSymbol('a', a)
    b_symbol = ValueSymbol('b', b)
    c_symbol = ValueSymbol('b', c)
    a_b_symbol = SubscriptSymbol(a_symbol, b_symbol)
    a_b_c_symbol = SubscriptSymbol(a_b_symbol, c_symbol)

    self.assertEqual(a_symbol.maybe_compute_value(), a)
    self.assertEqual(b_symbol.maybe_compute_value(), b)
    self.assertEqual(a_b_symbol.maybe_compute_value(), a[b])
    self.assertEqual(a_b_c_symbol.maybe_compute_value(), a[b][c])

  def test_attribute_access_after_attribute_access(self):
    class Bar(object):

      def __init__(self):
        self.c = object()

    class Foo(object):

      def __init__(self):
        self.b = Bar()

    a = Foo()
    a_symbol = ValueSymbol('a', a)
    a_b_symbol = AttributeAccessSymbol(a_symbol, 'b')
    a_b_c_symbol = AttributeAccessSymbol(a_b_symbol, 'c')

    self.assertEqual(a_symbol.maybe_compute_value(), a)
    self.assertEqual(a_b_symbol.maybe_compute_value(), a.b)
    self.assertEqual(a_b_c_symbol.maybe_compute_value(), a.b.c)


if __name__ == '__main__':
  test.main()
