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

from tensorflow.contrib.autograph.converters import side_effect_guards
from tensorflow.contrib.autograph.core import converter_testing
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


class SideEffectGuardsTest(converter_testing.TestCase):

  def test_side_effect_on_return_only_variable(self):

    tf = None

    def test_fn(a):
      tf.assign(a, a + 1)
      return a

    node = self.parse_and_analyze(test_fn, {})
    node = side_effect_guards.transform(node, self.ctx)

    with self.compiled(node, state_ops.assign) as result:
      self.assertEqual(len(node.body[0].body), 1)
      with self.test_session() as sess:
        v = variables.Variable(2)
        sess.run(v.initializer)
        # NOTE: We don't expect the assignment to execute in this case, because
        # variables cannot be reliably guarded.
        self.assertEqual(2, sess.run(result.test_fn(v)))

  def test_side_effect_on_used_variable(self):

    tf = None

    def test_fn(a):
      tf.assign(a, a + 1)
      return a + 1

    node = self.parse_and_analyze(test_fn, {})
    node = side_effect_guards.transform(node, self.ctx)

    with self.compiled(node, state_ops.assign) as result:
      self.assertEqual(len(node.body[0].body), 1)
      with self.test_session() as sess:
        v = variables.Variable(2)
        sess.run(v.initializer)
        # NOTE: Unlike test_side_effect_on_return_only_variable, the variable
        # was used in the local scope and so we could catch the assign's side
        # effect.
        self.assertEqual(4, sess.run(result.test_fn(v)))

  def test_side_effect_on_tensor(self):

    tf = None

    def test_fn(a):
      tf.Assert(a > 0, ['expected in throw'])
      return a

    node = self.parse_and_analyze(test_fn, {})
    node = side_effect_guards.transform(node, self.ctx)

    with self.compiled(node, control_flow_ops.Assert) as result:
      self.assertEqual(len(node.body[0].body), 1)
      with self.test_session() as sess:
        # NOTE: In this case we can also capture the side effect because the
        # argument is a tensor ans we can wrap it inside an identity.
        with self.assertRaisesRegexp(errors_impl.InvalidArgumentError,
                                     'expected in throw'):
          sess.run(result.test_fn(constant_op.constant(-1)))

  def test_multiline_block(self):

    tf = None

    def test_fn(a):
      tf.assign(a, a + 1)
      b = a + 1
      tf.assign(a, b + 1)
      c = b + 1
      d = c + 1
      return d

    node = self.parse_and_analyze(test_fn, {})
    node = side_effect_guards.transform(node, self.ctx)

    with self.compiled(node, state_ops.assign) as result:
      self.assertEqual(len(node.body[0].body), 1)
      with self.test_session() as sess:
        v = variables.Variable(2)
        sess.run(v.initializer)
        self.assertEqual(6, sess.run(result.test_fn(v)))

  def test_multiline_nested_block(self):

    tf = None

    def test_fn(a):
      with tf.name_scope('foo'):
        tf.assign(a, a + 1)
        b = a + 1
        c = b + 1
        d = c + 1
      return d

    node = self.parse_and_analyze(test_fn, {})
    node = side_effect_guards.transform(node, self.ctx)

    with self.compiled(node, state_ops.assign, ops.name_scope) as result:
      self.assertEqual(len(node.body[0].body[0].body), 1)
      with self.test_session() as sess:
        v = variables.Variable(2)
        sess.run(v.initializer)
        self.assertEqual(6, sess.run(result.test_fn(v)))

  def test_multiline_block_unsafe(self):

    tf = None

    def test_fn(a):
      tf.assign(a, a + 1)
      b = a + 1
      tf.assign(a, a + 1)
      c = b + 1
      d = c + 1
      return d

    node = self.parse_and_analyze(test_fn, {})
    node = side_effect_guards.transform(node, self.ctx)

    with self.compiled(node, state_ops.assign) as result:
      self.assertEqual(len(node.body[0].body), 1)
      with self.test_session() as sess:
        v = variables.Variable(2)
        sess.run(v.initializer)
        # NOTE: This intentionally highlights the flakiness. The test should be
        # tightened down once that is solved.
        self.assertTrue(sess.run(result.test_fn(v)) in (6, 7))


if __name__ == '__main__':
  test.main()
