# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
# =============================================================================
"""Tests for contrib.compiler.xla."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.compiler import xla
from tensorflow.contrib.tpu.python.tpu import tpu_feed
from tensorflow.python import summary
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import test


class XLACompileContextTest(test.TestCase):

  def create_test_xla_compile_context(self):
    computation_name = ops.get_default_graph().unique_name('computation')
    pivot = control_flow_ops.no_op(name=computation_name + '/pivot')
    return xla.XLACompileContext(name=computation_name, pivot=pivot)

  def test_report_unsupported_operations(self):
    """Tests that unsupported operations are detected."""
    context = self.create_test_xla_compile_context()
    context.Enter()
    dummy_tensor = constant_op.constant(1.1)
    audio_summary = summary.audio('audio_summary', dummy_tensor, 0.5)
    histogram_summary = summary.histogram('histogram_summary', dummy_tensor)
    image_summary = summary.image('image_summary', dummy_tensor)
    scalar_summary = summary.scalar('scalar_summary', dummy_tensor)
    tensor_summary = summary.tensor_summary('tensor_summary', dummy_tensor)
    summary.merge(
        [
            audio_summary, histogram_summary, image_summary, scalar_summary,
            tensor_summary
        ],
        name='merge_summary')
    logging_ops.Print(dummy_tensor, [dummy_tensor], name='print_op')
    context.Exit()

    unsupported_ops_names = [op.name for op in context._unsupported_ops]
    self.assertEqual(unsupported_ops_names, [
        u'audio_summary', u'histogram_summary', u'image_summary',
        u'scalar_summary', u'tensor_summary', u'merge_summary/merge_summary',
        u'print_op'
    ])

  def test_resource_variable(self):
    """Tests that resource variable usage is allowed."""
    a = variable_scope.get_variable(
        name='variable_a', shape=(1), use_resource=True)

    context = self.create_test_xla_compile_context()
    context.Enter()
    state_ops.assign(a, a + 1)
    context.Exit()

  def test_non_resource_variable_error(self):
    """Tests that non-resource variable usage is disallowed."""
    a = variable_scope.get_variable(
        name='variable_a', shape=(1), use_resource=False)

    context = self.create_test_xla_compile_context()
    context.Enter()
    with self.assertRaisesRegexp(
        NotImplementedError, 'Non-resource Variables are not supported inside '
        r'XLA computations \(operator name: Assign\)'):
      state_ops.assign(a, a + 1)
    context.Exit()

  def test_nested_xla_compile_error(self):
    """Tests that nested XLA computation leads to fatal error."""
    context1 = self.create_test_xla_compile_context()
    context1.Enter()

    context2 = self.create_test_xla_compile_context()
    context2.Enter()
    with self.assertRaisesRegexp(ValueError,
                                 'XLA compiled computations cannot be nested'):
      constant_op.constant(1)
    context2.Exit()
    context1.Exit()

  def test_xla_compile_attr(self):
    """Tests that ops are tagged with XLA compile ID attribute."""
    context = self.create_test_xla_compile_context()
    context.Enter()
    op = constant_op.constant(1)
    context.Exit()
    self.assertIn('_xla_compile_id', op.op.node_def.attr)

  def test_op_without_input(self):
    """Tests that ops without inputs depend on pivot correctly."""
    context = self.create_test_xla_compile_context()
    context.Enter()
    op = constant_op.constant(1)
    context.Exit()

    self.assertIn(context._pivot, op.op.control_inputs)

  def test_external_control_edges(self):
    """Tests that external control edges are handled correctly."""
    i = constant_op.constant(1)
    op1 = constant_op.constant(1)

    with ops.control_dependencies([op1]):
      op2 = constant_op.constant(1)
    self.assertIn(op1.op, op2.op.control_inputs)

    def while_body(i):
      del i  # unused
      context = self.create_test_xla_compile_context()
      context.Enter()
      with ops.control_dependencies([op1]):
        op3 = constant_op.constant(1)
      context.Exit()
      self.assertNotIn(op1.op, op3.op.control_inputs)
      return op3

    control_flow_ops.while_loop(
        cond=lambda i: math_ops.less(i, 10), body=while_body, loop_vars=[i])

  def test_op_output_marked_as_seen(self):
    """Tests that any op output is marked as seen in context."""
    context = self.create_test_xla_compile_context()
    context.Enter()
    op = constant_op.constant(1)
    context.Exit()

    self.assertIn(op.name, context._values)

  def testOpIsInContext(self):
    """Tests that XLACompileContext is recognized as an XLA context."""
    op1 = constant_op.constant(1)
    context = self.create_test_xla_compile_context()
    context.Enter()
    op2 = constant_op.constant(2)
    context.Exit()
    self.assertFalse(control_flow_util.IsInXLAContext(op1.op))
    self.assertTrue(control_flow_util.IsInXLAContext(op2.op))

  def testOpPreventFeeding(self):
    """Tests that ops created inside XLACompileContext can not be fed."""
    context = self.create_test_xla_compile_context()
    context.Enter()
    op = constant_op.constant(1)
    context.Exit()
    self.assertFalse(op.graph.is_feedable(op.op))

  def testOpPreventFetching(self):
    """Tests that ops created inside XLACompileContext can not be fetched."""
    context = self.create_test_xla_compile_context()
    context.Enter()
    op = constant_op.constant(1)
    context.Exit()
    self.assertFalse(op.graph.is_fetchable(op.op))


class CheckFunctionArgumentCountTest(test.TestCase):

  def testSimple(self):
    """Tests that arg checker works for functions with no varargs or defaults.
    """

    def func(x, y, z):
      return x + y + z

    self.assertEqual(None, xla.check_function_argument_count(func, 3, None))
    self.assertEqual('exactly 3 arguments',
                     xla.check_function_argument_count(func, 2, None))
    queue = tpu_feed.InfeedQueue(2)
    self.assertEqual(None, xla.check_function_argument_count(func, 1, queue))
    self.assertEqual('exactly 3 arguments',
                     xla.check_function_argument_count(func, 2, queue))

  def testDefaultArgs(self):
    """Tests that arg checker works for a function with no varargs."""

    def func(x, y, z=17):
      return x + y + z

    self.assertEqual(None, xla.check_function_argument_count(func, 3, None))
    self.assertEqual(None, xla.check_function_argument_count(func, 2, None))
    self.assertEqual('at least 2 arguments',
                     xla.check_function_argument_count(func, 1, None))
    self.assertEqual('at most 3 arguments',
                     xla.check_function_argument_count(func, 4, None))
    queue = tpu_feed.InfeedQueue(1)
    self.assertEqual(None, xla.check_function_argument_count(func, 2, queue))
    self.assertEqual(None, xla.check_function_argument_count(func, 1, queue))
    self.assertEqual('at least 2 arguments',
                     xla.check_function_argument_count(func, 0, queue))
    self.assertEqual('at most 3 arguments',
                     xla.check_function_argument_count(func, 4, queue))

  def testVarArgs(self):
    """Tests that arg checker works for a function with varargs."""

    def func(x, y, *z):
      return x + y + len(z)

    self.assertEqual(None, xla.check_function_argument_count(func, 2, None))
    self.assertEqual(None, xla.check_function_argument_count(func, 3, None))
    self.assertEqual(None, xla.check_function_argument_count(func, 4, None))
    self.assertEqual('at least 2 arguments',
                     xla.check_function_argument_count(func, 1, None))
    queue = tpu_feed.InfeedQueue(1)
    self.assertEqual(None, xla.check_function_argument_count(func, 1, queue))
    self.assertEqual(None, xla.check_function_argument_count(func, 2, queue))
    self.assertEqual(None, xla.check_function_argument_count(func, 3, queue))
    self.assertEqual('at least 2 arguments',
                     xla.check_function_argument_count(func, 0, queue))

  def testVarArgsAndDefaults(self):
    """Tests that arg checker works for a function with varargs and defaults."""

    def func(x, y, z=17, *q):  # pylint: disable=keyword-arg-before-vararg
      return x + y + z + len(q)

    self.assertEqual(None, xla.check_function_argument_count(func, 2, None))
    self.assertEqual(None, xla.check_function_argument_count(func, 3, None))
    self.assertEqual(None, xla.check_function_argument_count(func, 4, None))
    self.assertEqual(None, xla.check_function_argument_count(func, 5, None))
    self.assertEqual('at least 2 arguments',
                     xla.check_function_argument_count(func, 1, None))
    queue = tpu_feed.InfeedQueue(1)
    self.assertEqual(None, xla.check_function_argument_count(func, 1, queue))
    self.assertEqual(None, xla.check_function_argument_count(func, 2, queue))
    self.assertEqual(None, xla.check_function_argument_count(func, 3, queue))
    self.assertEqual(None, xla.check_function_argument_count(func, 4, queue))
    self.assertEqual('at least 2 arguments',
                     xla.check_function_argument_count(func, 0, queue))


if __name__ == '__main__':
  test.main()
