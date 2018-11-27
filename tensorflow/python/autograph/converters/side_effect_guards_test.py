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
"""Tests for side_effect_guards module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.autograph.converters import side_effect_guards
from tensorflow.python.autograph.core import converter_testing
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import test


tf = None  # Will be replaced by a mock.


class SideEffectGuardsTest(converter_testing.TestCase):

  def test_side_effect_on_return_only_variable(self):

    def test_fn(a):
      tf.assign(a, a + 1)
      return a

    node, ctx = self.prepare(test_fn, {})
    node = side_effect_guards.transform(node, ctx)

    self.assertEqual(len(node.body), 1)

    with self.compiled(node, {}, state_ops.assign) as result:
      with self.cached_session() as sess:
        v = variable_scope.get_variable('test', initializer=2)
        self.evaluate(v.initializer)
        self.evaluate(result.test_fn(v))
        # TODO(mdan): Add support for this use case.
        # Right now the variable `a` is not conditioned on the `assign` because
        # there's no way to add control dependencies to a variable object.
        self.assertEqual(2, self.evaluate(v))

  def test_side_effect_on_used_variable(self):

    def test_fn(a):
      tf.assign(a, a + 1)
      return a + 1

    node, ctx = self.prepare(test_fn, {})
    node = side_effect_guards.transform(node, ctx)

    self.assertEqual(len(node.body), 1)

    with self.compiled(node, {}, state_ops.assign) as result:
      with self.cached_session() as sess:
        v = variable_scope.get_variable('test', initializer=2)
        self.evaluate(v.initializer)
        self.evaluate(result.test_fn(v))
        # TODO(mdan): Ensure the result of test_fn(v) is also deterministic.
        # Right now it's 3 or 4 based on whether the read is synchronized.
        self.assertEqual(3, self.evaluate(v))

  def test_side_effect_on_tensor(self):

    def test_fn(a):
      tf.Assert(a > 0, ['expected in throw'])
      return a

    node, ctx = self.prepare(test_fn, {})
    node = side_effect_guards.transform(node, ctx)

    self.assertEqual(len(node.body), 1)

    with self.compiled(node, {}, control_flow_ops.Assert) as result:
      with self.cached_session() as sess:
        with self.assertRaisesRegexp(errors_impl.InvalidArgumentError,
                                     'expected in throw'):
          sess.run(result.test_fn(constant_op.constant(-1)))

  def test_multiline_block(self):

    def test_fn(a):
      tf.assign_add(a, 1)
      b = a + 1
      tf.assign_add(a, 1)
      b += 1
      return b

    node, ctx = self.prepare(test_fn, {})
    node = side_effect_guards.transform(node, ctx)

    self.assertEqual(len(node.body), 1)

    with self.compiled(node, {}, state_ops.assign_add) as result:
      with self.cached_session() as sess:
        v = variable_scope.get_variable('test', initializer=2)
        self.evaluate(v.initializer)
        self.evaluate(result.test_fn(v))
        # TODO(mdan): Ensure the result of test_fn(v) is also deterministic.
        self.assertEqual(4, self.evaluate(v))

  def test_multiline_nested_block(self):

    def test_fn(a):
      with tf.name_scope('foo'):
        tf.assign(a, a + 1)
        b = a + 1
      return b

    node, ctx = self.prepare(test_fn, {})
    node = side_effect_guards.transform(node, ctx)

    self.assertEqual(len(node.body[0].body), 1)

    with self.compiled(node, {}, state_ops.assign, ops.name_scope) as result:
      with self.cached_session() as sess:
        v = variable_scope.get_variable('test', initializer=2)
        self.evaluate(v.initializer)
        self.evaluate(result.test_fn(v))
        # TODO(mdan): Ensure the result of test_fn(v) is also deterministic.
        self.assertEqual(3, self.evaluate(v))

  def test_multiline_block_unsafe(self):

    def test_fn(a):
      tf.assign(a, a + 1)
      b = a + 1
      tf.assign_add(a, 1)
      c = b + 1
      return c

    node, ctx = self.prepare(test_fn, {})
    node = side_effect_guards.transform(node, ctx)

    self.assertEqual(len(node.body), 1)

    with self.compiled(node, {}, state_ops.assign,
                       state_ops.assign_add) as result:
      with self.cached_session() as sess:
        v = variable_scope.get_variable('test', initializer=2)
        self.evaluate(v.initializer)
        self.evaluate(result.test_fn(v))
        # TODO(mdan): Ensure the result of test_fn(v) is also deterministic.
        self.assertEqual(4, self.evaluate(v))


if __name__ == '__main__':
  test.main()
