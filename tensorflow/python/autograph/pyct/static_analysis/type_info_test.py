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

from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import cfg
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.pyct import qual_names
from tensorflow.python.autograph.pyct import transformer
from tensorflow.python.autograph.pyct.static_analysis import activity
from tensorflow.python.autograph.pyct.static_analysis import live_values
from tensorflow.python.autograph.pyct.static_analysis import reaching_definitions
from tensorflow.python.autograph.pyct.static_analysis import type_info
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

  def _parse_and_analyze(self,
                         test_fn,
                         namespace,
                         arg_types=None):
    node, source = parser.parse_entity(test_fn)
    entity_info = transformer.EntityInfo(
        source_code=source,
        source_file=None,
        namespace=namespace,
        arg_values=None,
        arg_types=arg_types,
        owner_type=None)
    node = qual_names.resolve(node)
    graphs = cfg.build(node)
    ctx = transformer.Context(entity_info)
    node = activity.resolve(node, ctx)
    node = reaching_definitions.resolve(node, ctx, graphs,
                                        reaching_definitions.Definition)
    node = live_values.resolve(node, ctx, {})
    node = type_info.resolve(node, ctx)
    node = live_values.resolve(node, ctx, {})
    return node

  def test_constructor_detection(self):

    def test_fn():
      opt = training.GradientDescentOptimizer(0.1)
      return opt

    node = self._parse_and_analyze(test_fn, {'training': training})
    call_node = node.body[0].body[0].value
    self.assertTrue(anno.getanno(call_node, 'is_constructor'))
    self.assertEquals(training.GradientDescentOptimizer,
                      anno.getanno(call_node, 'type'))
    self.assertEquals((training.__name__, 'GradientDescentOptimizer'),
                      anno.getanno(call_node, 'type_fqn'))

  def test_constructor_detection_builtin_class(self):

    def test_fn(x):
      res = zip(x)
      return res

    node = self._parse_and_analyze(test_fn, {})
    call_node = node.body[0].body[0].value
    self.assertFalse(anno.hasanno(call_node, 'is_constructor'))

  def test_class_members_of_detected_constructor(self):

    def test_fn():
      opt = training.GradientDescentOptimizer(0.1)
      opt.minimize(0)

    node = self._parse_and_analyze(test_fn, {'training': training})
    method_call = node.body[0].body[1].value.func
    self.assertEquals(training.GradientDescentOptimizer.minimize,
                      anno.getanno(method_call, 'live_val'))

  def test_class_members_in_with_stmt(self):

    def test_fn(x):
      with session.Session() as sess:
        sess.run(x)

    node = self._parse_and_analyze(test_fn, {'session': session})
    constructor_call = node.body[0].body[0].items[0].context_expr
    self.assertEquals(session.Session, anno.getanno(constructor_call, 'type'))
    self.assertEquals((session.__name__, 'Session'),
                      anno.getanno(constructor_call, 'type_fqn'))

    method_call = node.body[0].body[0].body[0].value.func
    self.assertEquals(session.Session.run, anno.getanno(method_call,
                                                        'live_val'))

  def test_constructor_data_dependent(self):

    def test_fn(x):
      if x > 0:
        opt = training.GradientDescentOptimizer(0.1)
      else:
        opt = training.GradientDescentOptimizer(0.01)
      opt.minimize(0)

    node = self._parse_and_analyze(test_fn, {'training': training})
    method_call = node.body[0].body[1].value.func
    self.assertFalse(anno.hasanno(method_call, 'live_val'))

  def test_parameter_class_members(self):

    def test_fn(opt):
      opt.minimize(0)

    node = self._parse_and_analyze(test_fn, {})
    method_call = node.body[0].body[0].value.func
    self.assertFalse(anno.hasanno(method_call, 'live_val'))

  def test_parameter_class_members_with_value_hints(self):

    def test_fn(opt):
      opt.minimize(0)

    node = self._parse_and_analyze(
        test_fn, {},
        arg_types={
            'opt': (training.GradientDescentOptimizer.__name__,
                    training.GradientDescentOptimizer)
        })

    method_call = node.body[0].body[0].value.func
    self.assertEquals(training.GradientDescentOptimizer.minimize,
                      anno.getanno(method_call, 'live_val'))

  def test_function_variables(self):

    def bar():
      pass

    def test_fn():
      foo = bar
      foo()

    node = self._parse_and_analyze(test_fn, {'bar': bar})
    method_call = node.body[0].body[1].value.func
    self.assertFalse(anno.hasanno(method_call, 'live_val'))

  def test_nested_members(self):

    def test_fn():
      foo = training.GradientDescentOptimizer(0.1)
      foo.bar.baz()

    node = self._parse_and_analyze(test_fn, {'training': training})
    method_call = node.body[0].body[1].value.func
    self.assertFalse(anno.hasanno(method_call, 'live_val'))

  def test_nested_unpacking(self):

    class Foo(object):
      pass

    class Bar(object):
      pass

    def test_fn():
      a, (b, c) = (Foo(), (Bar(), Foo()))
      return a, b, c

    node = self._parse_and_analyze(test_fn, {'Foo': Foo, 'Bar': Bar})
    a, b, c = node.body[0].body[1].value.elts
    self.assertEquals(anno.getanno(a, 'type'), Foo)
    self.assertEquals(anno.getanno(b, 'type'), Bar)
    self.assertEquals(anno.getanno(c, 'type'), Foo)
    self.assertFalse(anno.hasanno(a, 'live_val'))
    self.assertFalse(anno.hasanno(b, 'live_val'))
    self.assertFalse(anno.hasanno(c, 'live_val'))


if __name__ == '__main__':
  test.main()
