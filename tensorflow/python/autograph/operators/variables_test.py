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
"""Tests for python_lang_utils module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.autograph.operators import variables
from tensorflow.python.platform import test


class SpecialValuesTest(test.TestCase):

  def test_undefined(self):
    undefined_symbol = variables.Undefined('name')
    undefined_symbol2 = variables.Undefined('name')

    self.assertEqual(undefined_symbol.symbol_name, 'name')
    self.assertEqual(undefined_symbol2.symbol_name, 'name')
    self.assertNotEqual(undefined_symbol, undefined_symbol2)

  def test_undefined_operations(self):
    undefined_symbol = variables.Undefined('name')

    self.assertIsInstance(undefined_symbol.foo, variables.Undefined)
    self.assertIsInstance(undefined_symbol[0], variables.Undefined)
    self.assertNotIsInstance(undefined_symbol.__class__, variables.Undefined)

  def test_read(self):
    self.assertEqual(variables.ld(1), 1)
    o = object()
    self.assertEqual(variables.ld(o), o)

    self.assertIsNone(variables.ld(None))

  def test_read_undefined(self):
    with self.assertRaisesRegex(UnboundLocalError, 'used before assignment'):
      variables.ld(variables.Undefined('a'))


if __name__ == '__main__':
  test.main()
