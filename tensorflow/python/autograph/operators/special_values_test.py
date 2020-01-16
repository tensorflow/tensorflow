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

from tensorflow.python.autograph.operators import special_values
from tensorflow.python.platform import test


class SpecialValuesTest(test.TestCase):

  def test_undefined(self):
    undefined_symbol = special_values.Undefined('name')
    self.assertEqual(undefined_symbol.symbol_name, 'name')

    undefined_symbol2 = special_values.Undefined('name')
    self.assertNotEqual(undefined_symbol, undefined_symbol2)

    self.assertTrue(special_values.is_undefined(undefined_symbol))
    self.assertTrue(special_values.is_undefined(undefined_symbol2))

  def test_undefined_operations(self):
    undefined_symbol = special_values.Undefined('name')

    self.assertTrue(special_values.is_undefined(undefined_symbol.foo))
    self.assertTrue(special_values.is_undefined(undefined_symbol[0]))
    self.assertFalse(special_values.is_undefined(undefined_symbol.__class__))

if __name__ == '__main__':
  test.main()
