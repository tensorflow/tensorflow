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

from tensorflow.contrib.autograph import utils
from tensorflow.contrib.autograph.impl import api
from tensorflow.contrib.autograph.impl import conversion
from tensorflow.python.framework import constant_op
from tensorflow.python.keras._impl.keras.engine import training
from tensorflow.python.platform import test


class ConversionTest(test.TestCase):

  def _simple_conversion_map(self):
    return conversion.ConversionMap(True, (), (), api)

  def test_is_whitelisted_for_graph(self):

    def test_fn():
      return constant_op.constant(1)

    self.assertFalse(conversion.is_whitelisted_for_graph(test_fn))
    self.assertTrue(conversion.is_whitelisted_for_graph(utils))
    self.assertTrue(conversion.is_whitelisted_for_graph(constant_op.constant))

  def test_entity_to_graph_unsupported_types(self):
    with self.assertRaises(ValueError):
      conversion_map = self._simple_conversion_map()
      conversion.entity_to_graph('dummy', conversion_map, None, None)

  def test_entity_to_graph_callable(self):
    b = 2
    def f(a):
      return a + b

    conversion_map = self._simple_conversion_map()
    ast, name, ns = conversion.entity_to_graph(f, conversion_map, None, None)
    self.assertTrue(isinstance(ast, gast.FunctionDef), ast)
    self.assertEqual('tf__f', name)
    self.assertTrue(ns['b'] is b)

  def test_entity_to_graph_call_tree(self):

    def g(a):
      return a

    def f(a):
      return g(a)

    conversion_map = self._simple_conversion_map()
    conversion.entity_to_graph(f, conversion_map, None, None)

    self.assertTrue(f in conversion_map.dependency_cache)
    self.assertTrue(g in conversion_map.dependency_cache)
    self.assertEqual('tf__f', conversion_map.dependency_cache[f].name)
    # need the extra .body[0] in order to step past the with tf.name_scope('f')
    # that is added automatically
    self.assertEqual(
        'tf__g',
        conversion_map.dependency_cache[f].body[0].body[0].value.func.id)
    self.assertEqual('tf__g', conversion_map.dependency_cache[g].name)

  def test_entity_to_graph_class_hierarchy(self):

    class TestBase(object):

      def __init__(self, x='base'):
        self.x = x

      def foo(self):
        return self.x

      def bar(self):
        return self.x

    class TestSubclass(TestBase):

      def __init__(self, y):
        super(TestSubclass, self).__init__('sub')
        self.y = y

      def foo(self):
        return self.y

      def baz(self):
        return self.y

    conversion_map = self._simple_conversion_map()
    conversion.entity_to_graph(TestSubclass, conversion_map, None, None)

    self.assertTrue(TestBase in conversion_map.dependency_cache)
    self.assertTrue(TestSubclass in conversion_map.dependency_cache)
    self.assertEqual('TfTestBase',
                     conversion_map.dependency_cache[TestBase].body[-1].name)
    self.assertEqual(
        'TfTestSubclass',
        conversion_map.dependency_cache[TestSubclass].body[-1].name)

  def test_entity_to_graph_class_hierarchy_whitelisted(self):

    class TestSubclass(training.Model):

      def __init__(self, y):
        super(TestSubclass, self).__init__()
        self.built = False

      def call(self, x):
        return 3 * x

    conversion_map = self._simple_conversion_map()
    conversion.entity_to_graph(TestSubclass, conversion_map, None, None)

    self.assertTrue(TestSubclass in conversion_map.dependency_cache)
    self.assertFalse(training.Model in conversion_map.dependency_cache)
    self.assertEqual(
        'Model',
        conversion_map.dependency_cache[TestSubclass].body[0].names[0].name)
    self.assertEqual(
        'TfTestSubclass',
        conversion_map.dependency_cache[TestSubclass].body[-1].name)

  def test_entity_to_graph_lambda(self):
    f = lambda a: a

    with self.assertRaises(NotImplementedError):
      conversion_map = self._simple_conversion_map()
      conversion.entity_to_graph(f, conversion_map, None, None)

  def test_ag_module_cached(self):
    def callee():
      return range(3)

    def caller(a):
      return a()

    conversion_map = self._simple_conversion_map()
    _, _, callee_ns = conversion.entity_to_graph(
        callee, conversion_map, None, None)
    _, _, caller_ns = conversion.entity_to_graph(
        caller, conversion_map, None, None)

    self.assertTrue(callee_ns['ag__'] is caller_ns['ag__'])


if __name__ == '__main__':
  test.main()
