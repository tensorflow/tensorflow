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
"""Tests for type_info module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.py2tf.pyct import anno
from tensorflow.contrib.py2tf.pyct import parser
from tensorflow.contrib.py2tf.pyct.static_analysis import access
from tensorflow.contrib.py2tf.pyct.static_analysis import live_values
from tensorflow.contrib.py2tf.pyct.static_analysis import type_info
from tensorflow.python.client import session
from tensorflow.python.platform import test
from tensorflow.python.training import training


class ScopeTest(test.TestCase):

  def test_basic(self):
    scope = type_info.Scope(None)
    self.assertFalse(scope.hasval('foo'))

    scope.setval('foo', 'bar')
    self.assertTrue(scope.hasval('foo'))

    self.assertFalse(scope.hasval('baz'))

  def test_nesting(self):
    scope = type_info.Scope(None)
    scope.setval('foo', '')

    child = type_info.Scope(scope)
    self.assertTrue(child.hasval('foo'))
    self.assertTrue(scope.hasval('foo'))

    child.setval('bar', '')
    self.assertTrue(child.hasval('bar'))
    self.assertFalse(scope.hasval('bar'))


class TypeInfoResolverTest(test.TestCase):

  def test_constructor_detection(self):

    def test_fn():
      opt = training.GradientDescentOptimizer(0.1)
      return opt

    node = parser.parse_object(test_fn)
    node = access.resolve(node)
    node = live_values.resolve(node, {'training': training}, {})
    node = type_info.resolve(node, None)

    call_node = node.body[0].body[0].value
    self.assertEquals(training.GradientDescentOptimizer,
                      anno.getanno(call_node, 'type'))
    self.assertEquals((training.__name__, 'GradientDescentOptimizer'),
                      anno.getanno(call_node, 'type_fqn'))

  def test_class_members(self):

    def test_fn():
      opt = training.GradientDescentOptimizer(0.1)
      opt.minimize(0)

    node = parser.parse_object(test_fn)
    node = access.resolve(node)
    node = live_values.resolve(node, {'training': training}, {})
    node = type_info.resolve(node, None)

    attr_call_node = node.body[0].body[1].value.func
    self.assertEquals((training.__name__, 'GradientDescentOptimizer'),
                      anno.getanno(attr_call_node, 'type_fqn'))

  def test_class_members_in_with_stmt(self):

    def test_fn(x):
      with session.Session() as sess:
        sess.run(x)

    node = parser.parse_object(test_fn)
    node = access.resolve(node)
    node = live_values.resolve(node, {'session': session}, {})
    node = type_info.resolve(node, None)

    constructor_call = node.body[0].body[0].items[0].context_expr
    self.assertEquals(session.Session, anno.getanno(constructor_call, 'type'))
    self.assertEquals((session.__name__, 'Session'),
                      anno.getanno(constructor_call, 'type_fqn'))

    member_call = node.body[0].body[0].body[0].value.func
    self.assertEquals((session.__name__, 'Session'),
                      anno.getanno(member_call, 'type_fqn'))

  def test_constructor_deta_dependent(self):

    def test_fn(x):
      if x > 0:
        opt = training.GradientDescentOptimizer(0.1)
      else:
        opt = training.GradientDescentOptimizer(0.01)
      opt.minimize(0)

    node = parser.parse_object(test_fn)
    node = access.resolve(node)
    node = live_values.resolve(node, {'training': training}, {})
    with self.assertRaises(ValueError):
      node = type_info.resolve(node, None)

  def test_parameter_class_members(self):

    def test_fn(opt):
      opt.minimize(0)

    node = parser.parse_object(test_fn)
    node = access.resolve(node)
    node = live_values.resolve(node, {'training': training}, {})
    with self.assertRaises(ValueError):
      node = type_info.resolve(node, None)

  def test_parameter_class_members_with_value_hints(self):

    def test_fn(opt):
      opt.minimize(0)

    node = parser.parse_object(test_fn)
    node = access.resolve(node)
    node = live_values.resolve(node, {'training': training}, {})
    node = type_info.resolve(
        node, {
            'opt': (('%s.GradientDescentOptimizer' % training.__name__),
                    training.GradientDescentOptimizer(0.1))
        })

    attr_call_node = node.body[0].body[0].value.func
    self.assertEquals(
        tuple(training.__name__.split('.')) + ('GradientDescentOptimizer',),
        anno.getanno(attr_call_node, 'type_fqn'))

  def test_function_variables(self):

    def bar():
      pass

    def test_fn():
      foo = bar
      foo()

    node = parser.parse_object(test_fn)
    node = access.resolve(node)
    node = live_values.resolve(node, {'bar': bar}, {})
    with self.assertRaises(ValueError):
      node = type_info.resolve(node, None)

  def test_nested_members(self):

    def test_fn():
      foo = training.GradientDescentOptimizer(0.1)
      foo.bar.baz()

    node = parser.parse_object(test_fn)
    node = access.resolve(node)
    node = live_values.resolve(node, {'training': training}, {})
    with self.assertRaises(ValueError):
      node = type_info.resolve(node, None)


if __name__ == '__main__':
  test.main()
