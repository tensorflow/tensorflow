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

from tensorflow.contrib.py2tf import utils
from tensorflow.contrib.py2tf.pyct import anno
from tensorflow.contrib.py2tf.pyct import context
from tensorflow.contrib.py2tf.pyct import parser
from tensorflow.contrib.py2tf.pyct import qual_names
from tensorflow.contrib.py2tf.pyct.static_analysis import activity
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

  def _parse_and_analyze(self,
                         test_fn,
                         namespace,
                         arg_types=None):
    node, source = parser.parse_entity(test_fn)
    ctx = context.EntityContext(
        namer=None,
        source_code=source,
        source_file=None,
        namespace=namespace,
        arg_values=None,
        arg_types=arg_types,
        owner_type=None,
        recursive=True,
        type_annotation_func=utils.set_element_type)
    node = qual_names.resolve(node)
    node = activity.resolve(node, ctx)
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
    self.assertEquals(training.GradientDescentOptimizer,
                      anno.getanno(call_node, 'type'))
    self.assertEquals((training.__name__, 'GradientDescentOptimizer'),
                      anno.getanno(call_node, 'type_fqn'))

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
        test_fn, {'training': training},
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

  def test_type_annotation(self):

    class Foo(object):
      pass

    def test_fn():
      f = []
      f = utils.set_element_type(f, Foo)
      return f

    node = self._parse_and_analyze(test_fn, {'Foo': Foo, 'utils': utils})
    f_def = node.body[0].body[0].value
    self.assertEqual(anno.getanno(f_def, 'element_type'), Foo)
    f_ref = node.body[0].body[1].value
    self.assertEqual(anno.getanno(f_ref, 'element_type'), Foo)

  def test_nested_assignment(self):

    def test_fn(foo):
      a, (b, c) = foo
      return a, b, c

    node = self._parse_and_analyze(test_fn, {'foo': (1, 2, 3)})
    lhs = node.body[0].body[1].value.elts
    a = lhs[0]
    b = lhs[1]
    c = lhs[2]
    # TODO(mdan): change these once we have the live values propagating
    # correctly
    self.assertFalse(anno.hasanno(a, 'live_val'))
    self.assertFalse(anno.hasanno(b, 'live_val'))
    self.assertFalse(anno.hasanno(c, 'live_val'))


if __name__ == '__main__':
  test.main()
