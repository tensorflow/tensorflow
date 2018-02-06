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
"""Tests for conversion module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gast

from tensorflow.contrib.py2tf.impl import conversion
from tensorflow.python.platform import test


class ConversionTest(test.TestCase):

  def test_entity_to_graph_unsupported_types(self):
    with self.assertRaises(ValueError):
      conversion_map = conversion.ConversionMap(True, (), ())
      conversion.entity_to_graph('dummy', conversion_map, None, None)

  def test_entity_to_graph_callable(self):

    def f(a):
      return a

    conversion_map = conversion.ConversionMap(True, (), ())
    ast, new_name = conversion.entity_to_graph(f, conversion_map, None, None)
    self.assertTrue(isinstance(ast, gast.FunctionDef), ast)
    self.assertEqual('tf__f', new_name)

  def test_entity_to_graph_call_tree(self):

    def g(a):
      return a

    def f(a):
      return g(a)

    conversion_map = conversion.ConversionMap(True, (), ())
    conversion.entity_to_graph(f, conversion_map, None, None)

    self.assertTrue(f in conversion_map.dependency_cache)
    self.assertTrue(g in conversion_map.dependency_cache)
    self.assertEqual('tf__f', conversion_map.dependency_cache[f].name)
    self.assertEqual(
        'tf__g', conversion_map.dependency_cache[f].body[0].value.func.id)
    self.assertEqual('tf__g', conversion_map.dependency_cache[g].name)


if __name__ == '__main__':
  test.main()
