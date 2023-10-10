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
"""Tests for python.compiler.xla.xla."""

from absl.testing import parameterized

from tensorflow.python import summary
from tensorflow.python.compiler.xla import xla
from tensorflow.python.eager import def_function
from tensorflow.python.estimator import model_fn as model_fn_lib
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import while_loop
from tensorflow.python.platform import test
from tensorflow.python.tpu import tpu_feed


_TRAIN = model_fn_lib.ModeKeys.TRAIN
_EVAL = model_fn_lib.ModeKeys.EVAL
_EXPECTED_LOSS = 1
_EXPECTED_FEATURE = 2
_EXPECTED_LABEL = 3


class XLACompileContextTest(test.TestCase, parameterized.TestCase):

  def create_test_xla_compile_context(self):
    computation_name = ops.get_default_graph().unique_name('computation')
    pivot = control_flow_ops.no_op(name=computation_name + '/pivot')
    return xla.XLACompileContext(name=computation_name, pivot=pivot)

  @test_util.run_v1_only('Testing graph mode behavior only')
  def test_report_unsupported_operations_graph_mode(self):
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

  @test_util.run_v1_only('Testing graph mode behavior only')
  def test_resource_variable_graph_mode(self):
    """Tests that resource variable usage is allowed."""
    a = variable_scope.get_variable(
        name='variable_a', use_resource=True, initializer=1)

    context = self.create_test_xla_compile_context()
    context.Enter()
    a.assign(2)
    context.Exit()

  def test_resource_variable_in_function(self):
    """Tests that resource variable usage is allowed."""
    a = variable_scope.get_variable(
        name='variable_a', use_resource=True, initializer=1)

    @def_function.function
    def func():
      context = self.create_test_xla_compile_context()
      context.Enter()
      o = a.assign(2)
      context.Exit()
      return o

    self.assertEqual(self.evaluate(func()), 2)

  @test_util.run_v1_only('Testing v1-only ref variable handling.')
  def test_non_resource_variable_error(self):
    """Tests that non-resource variable usage is disallowed."""
    a = variable_scope.get_variable(
        name='variable_a', shape=(1), use_resource=False)

    context = self.create_test_xla_compile_context()
    context.Enter()
    with self.assertRaisesRegex(
        NotImplementedError, 'Non-resource Variables are not supported inside '
        r'XLA computations \(operator name: Assign\)'):
      state_ops.assign(a, a + 1)
    context.Exit()

  @test_util.build_as_function_and_v1_graph
  def test_nested_xla_compile_error(self):
    """Tests that nested XLA computation leads to fatal error."""
    context1 = self.create_test_xla_compile_context()
    context1.Enter()

    context2 = self.create_test_xla_compile_context()
    context2.Enter()
    with self.assertRaisesRegex(ValueError,
                                'XLA compiled computations cannot be nested'):
      constant_op.constant(1)
    context2.Exit()
    context1.Exit()

  @test_util.build_as_function_and_v1_graph
  def test_xla_compile_attr(self):
    """Tests that ops are tagged with XLA compile ID attribute."""
    context = self.create_test_xla_compile_context()
    context.Enter()
    op = constant_op.constant(1)
    context.Exit()
    self.assertIn('_xla_compile_id', op.op.node_def.attr)

  @test_util.build_as_function_and_v1_graph
  def test_op_without_input(self):
    """Tests that ops without inputs depend on pivot correctly."""
    context = self.create_test_xla_compile_context()
    context.Enter()
    op = constant_op.constant(1)
    context.Exit()

    self.assertIn(context._pivot, op.op.control_inputs)

  @test_util.run_v1_only('Testing graph mode behavior only')
  def test_external_control_edges_graph_mode(self):
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

    while_loop.while_loop(
        cond=lambda i: math_ops.less(i, 10), body=while_body, loop_vars=[i])

  @test_util.build_as_function_and_v1_graph
  def test_op_output_marked_as_seen(self):
    """Tests that any op output is marked as seen in context."""
    context = self.create_test_xla_compile_context()
    context.Enter()
    op = constant_op.constant(1)
    context.Exit()

    self.assertIn(op.name, context._values)

  @test_util.build_as_function_and_v1_graph
  def test_op_is_in_context(self):
    """Tests that XLACompileContext is recognized as an XLA context."""
    op1 = constant_op.constant(1)
    context = self.create_test_xla_compile_context()
    context.Enter()
    op2 = constant_op.constant(2)
    context.Exit()
    self.assertFalse(control_flow_util.IsInXLAContext(op1.op))
    self.assertTrue(control_flow_util.IsInXLAContext(op2.op))

  @test_util.build_as_function_and_v1_graph
  def test_op_prevent_feeding(self):
    """Tests that ops created inside XLACompileContext can not be fed."""
    context = self.create_test_xla_compile_context()
    context.Enter()
    op = constant_op.constant(1)
    context.Exit()
    self.assertFalse(op.graph.is_feedable(op.op))

  @test_util.build_as_function_and_v1_graph
  def test_op_prevent_fetching(self):
    """Tests that ops created inside XLACompileContext can not be fetched."""
    context = self.create_test_xla_compile_context()
    context.Enter()
    op = constant_op.constant(1)
    context.Exit()
    self.assertFalse(op.graph.is_fetchable(op.op))


class XlaCompileTest(test.TestCase):

  @test_util.run_v2_only
  @test_util.disable_tfrt(
      'Legacy XLA test. It depends on EncapsulateXlaComputationsPass.')
  def test_xla_compile_eager(self):
    """Tests that xla.compile raises proper exception when used eagerly."""

    def computation(a, b):
      return a + b

    self.assertEqual(self.evaluate(xla.compile(computation, [1, 2])[0]), 3)

  @test_util.disable_tfrt(
      'Legacy XLA test. It depends on EncapsulateXlaComputationsPass.')
  def test_xla_compile_in_function(self):
    """Tests that xla.compile works in tf.function."""

    @def_function.function
    def func_wrapper(a):

      def compute(a):
        return a + 1

      return xla.compile(compute, [a])

    self.assertEqual(self.evaluate(func_wrapper(1))[0], 2)

  @test_util.disable_tfrt(
      'Legacy XLA test. It depends on EncapsulateXlaComputationsPass.')
  def test_xla_compile_write_variable_in_function(self):
    """Tests that xla.compile works with variable in tf.function."""
    a = variable_scope.get_variable(
        name='variable_a', use_resource=True, initializer=1)

    @def_function.function
    def func_wrapper():

      def compute():
        a.assign_add(1)
        a.assign_sub(2)
        return a.read_value()

      return xla.compile(compute)

    self.evaluate(a.initializer)
    self.assertEqual(self.evaluate(func_wrapper())[0], 0)


class CheckFunctionArgumentCountTest(test.TestCase):

  def test_simple(self):
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

  def test_default_args(self):
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

  def test_var_args(self):
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

  def test_var_args_and_defaults(self):
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
