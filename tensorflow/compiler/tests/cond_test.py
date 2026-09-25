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
"""Tests for tf.cond in XLA."""

from tensorflow.compiler.tests import xla_test
from tensorflow.python.client import session
from tensorflow.python.compiler.xla import xla
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_switch_case
from tensorflow.python.ops import image_ops
from tensorflow.python.ops import io_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops.signal import fft_ops
from tensorflow.python.platform import test


@test_util.with_control_flow_v2
class CondTest(xla_test.XLATestCase):

  def testCondAndTensorArrayInDefun(self):
    # TODO(b/132430685): Make test more useful. Also b/129396295, b/127846988
    with self.session(), self.test_scope():
      xla_context = control_flow_ops.XLAControlFlowContext()
      xla_context.Enter()

      @def_function.function
      def f():
        ta = tensor_array_ops.TensorArray(dtype=dtypes.float32, size=1)
        output = cond.cond(
            constant_op.constant(True),
            lambda: ta.write(0, 5.), lambda: ta.write(0, 10.))

        return output.stack()

      output_t = f()
      self.assertAllEqual([5.], self.evaluate(output_t))

      xla_context.Exit()

  def testCondAndTensorArrayInDefun_constFolding(self):
    g = ops.Graph()
    with session.Session(graph=g), g.as_default(), self.test_scope():
      xla_context = control_flow_ops.XLAControlFlowContext()
      xla_context.Enter()

      @def_function.function
      def f():
        ta = tensor_array_ops.TensorArray(dtype=dtypes.float32, size=1)
        output = cond.cond(
            constant_op.constant(False),
            lambda: ta.write(0, 5.), lambda: ta.write(0, 10.))

        return output.stack()

      output_t = f()
      self.assertAllEqual([10.], self.evaluate(output_t))

      xla_context.Exit()

  def testCondAndTensorArray_xlaCompile(self):
    self.skipTest("b/127846988")
    # Fails with "Uninitialized arguments" in XlaIfOp::Compile
    with self.session(), self.test_scope():
      xla_context = control_flow_ops.XLAControlFlowContext()
      xla_context.Enter()

      def f():
        ta = tensor_array_ops.TensorArray(dtype=dtypes.float32, size=1)
        output = cond.cond(
            constant_op.constant(True),
            lambda: ta.write(0, 5.), lambda: ta.write(0, 10.))

        return output.stack()

      output_t, = xla.compile(f)
      self.assertAllEqual([5.], self.evaluate(output_t))

      xla_context.Exit()

  def testCondConstPropagation(self):
    with self.session() as sess, self.test_scope():
      xla_context = control_flow_ops.XLAControlFlowContext()
      xla_context.Enter()

      x = array_ops.placeholder(dtypes.float32)
      p = array_ops.placeholder(dtypes.int32)

      # TODO(b/129021699): Wrapping this in a tf.function does not work.
      def if_true():
        # This emits a StridedSlice op which expects the index to be a
        # compile-time const.
        return x[p]

      def if_false():
        return 5.

      output = cond.cond(
          constant_op.constant(True), if_true, if_false)

      self.assertAllEqual(1.,
                          sess.run(output, feed_dict={
                              x: [0., 1., 2.],
                              p: 1
                          }))

      xla_context.Exit()

  def testCondConstPropagation_xlaCompile(self):
    self.skipTest("b/132430685")
    with self.session(), self.test_scope():
      xla_context = control_flow_ops.XLAControlFlowContext()
      xla_context.Enter()

      x = array_ops.placeholder_with_default([0., 1., 2.], shape=[3])
      p = constant_op.constant(1)

      def f():
        # TODO(b/129021699): Wrapping this in a tf.function does not work.
        def if_true():
          # This emits a StridedSlice op which expects the index to be a
          # compile-time const.
          return x[p]

        def if_false():
          return 5.

        return cond.cond(
            constant_op.constant(True), if_true, if_false)

      output = xla.compile(f)

      self.assertAllEqual(1., self.evaluate(output))

      xla_context.Exit()

  def testCondConstPropagation_errorMsg(self):
    self.skipTest("b/132430685")
    with self.session() as sess, self.test_scope():
      xla_context = control_flow_ops.XLAControlFlowContext()
      xla_context.Enter()

      x = array_ops.placeholder(dtypes.float32)
      p = random_ops.random_uniform([], minval=1, maxval=3, dtype=dtypes.int32)

      # TODO(b/129021699): Wrapping this in a tf.function does not work.
      def if_true():
        # This emits a StridedSlice op which expects the index to be a
        # compile-time const.
        return x[:p]

      def if_false():
        return array_ops.fill([p], 5.)

      output = cond.cond(
          constant_op.constant(True), if_true, if_false)

      with self.assertRaisesRegex(errors.InvalidArgumentError,
                                  "must be a compile-time constant"):
        sess.run(
            output, feed_dict={
                x: [0., 1., 2.],
            })

      xla_context.Exit()

  def testCondConstPropagation_errorMsg_xlaCompile(self):
    with self.session() as sess, self.test_scope():
      xla_context = control_flow_ops.XLAControlFlowContext()
      xla_context.Enter()

      x = array_ops.placeholder(dtypes.float32)
      p = random_ops.random_uniform([], minval=1, maxval=3, dtype=dtypes.int32)
      condition = math_ops.cast(
          random_ops.random_uniform([], minval=0, maxval=2, dtype=dtypes.int32),
          dtypes.bool)

      def f():
        # TODO(b/129021699): Wrapping this in a tf.function does not work.
        def if_true():
          # This emits a StridedSlice op which expects the index to be a
          # compile-time const.
          return x[:p]

        def if_false():
          return array_ops.fill([p], 5.)

        return cond.cond(condition, if_true, if_false)

      output = xla.compile(f)

      with self.assertRaisesRegex(errors.InvalidArgumentError,
                                  "must be a compile-time constant"):
        sess.run(
            output, feed_dict={
                x: [0., 1., 2.],
            })

      xla_context.Exit()

  def testSwitchCaseAndTensorArrayInDefun(self):
    self.skipTest("b/127846988")
    with self.session(), self.test_scope():
      xla_context = control_flow_ops.XLAControlFlowContext()
      xla_context.Enter()

      @def_function.function
      def f():
        ta = tensor_array_ops.TensorArray(dtype=dtypes.float32, size=1)
        output = control_flow_switch_case.switch_case(
            constant_op.constant(1), {
                0: lambda: ta.write(0, 5.),
                1: lambda: ta.write(0, 10.),
                2: lambda: ta.write(0, 15.),
            })

        return output.stack()

      output_t = f()
      self.assertAllEqual([10.], self.evaluate(output_t))

      xla_context.Exit()

  def testSwitchCaseAndTensorArray_xlaCompile(self):
    self.skipTest("b/127846988")
    with self.session(), self.test_scope():
      xla_context = control_flow_ops.XLAControlFlowContext()
      xla_context.Enter()

      def f():
        ta = tensor_array_ops.TensorArray(dtype=dtypes.float32, size=1)
        output = control_flow_switch_case.switch_case(
            constant_op.constant(1), {
                0: lambda: ta.write(0, 5.),
                1: lambda: ta.write(0, 10.),
                2: lambda: ta.write(0, 15.),
            })

        return output.stack()

      output_t, = xla.compile(f)
      self.assertAllEqual([10.], self.evaluate(output_t))

      xla_context.Exit()

  def testSwitchCaseConstPropagation(self):
    self.skipTest("b/127846988")
    with self.session() as sess, self.test_scope():
      xla_context = control_flow_ops.XLAControlFlowContext()
      xla_context.Enter()

      x = array_ops.placeholder(dtypes.float32)
      p = array_ops.placeholder(dtypes.int32)

      def branch0():
        return 5.

      def branch1():
        return 15.

      # TODO(b/129021699): Wrapping this in a tf.function does not work.
      def branch2():
        # This emits a StridedSlice op which expects the index to be a
        # compile-time const.
        return x[p]

      output = control_flow_switch_case.switch_case(
          constant_op.constant(2), {
              0: branch0,
              1: branch1,
              2: branch2,
          })

      self.assertAllEqual(7.,
                          sess.run(output, feed_dict={
                              x: [0., 1., 7.],
                              p: 2,
                          }))

      xla_context.Exit()

  def testCondNoInputs(self):
    """Verifies against `Failed precondition: Expected one input shape`."""

    with self.session(), self.test_scope():
      xla_context = control_flow_ops.XLAControlFlowContext()
      xla_context.Enter()

      for pred in True, False:
        cond_out = cond.cond(
            array_ops.placeholder_with_default(pred, []),
            lambda: constant_op.constant(2.),
            lambda: constant_op.constant(1.))
        self.assertEqual(int(pred) + 1., self.evaluate(cond_out))

      xla_context.Exit()

  # The tests below compile whole functions with jit_compile=True, which is
  # what makes the predicate a compile-time constant; compiling the If on its
  # own under test_scope() would pass the predicate in as a parameter.

  def testIfBranchIsPruned(self):
    # With a constant predicate only the taken branch is compiled, so an
    # untaken branch that XLA cannot compile does not fail compilation. The
    # branch returns the unsupported op's result so that it is not pruned as
    # dead code before it reaches XLA.
    with context.eager_mode():

      @def_function.function(jit_compile=True)
      def f():

        def uncompilable():
          return array_ops.size(
              image_ops.decode_image(io_ops.read_file('/tmp/bmp')))

        return cond.cond(
            constant_op.constant(True), lambda: constant_op.constant(17),
            uncompilable)

      self.assertEqual(f().numpy(), 17)

  def testConstantPredicateKeepsTakenBranchShape(self):
    # rot90 builds nested conds over k whose branches have different static
    # shapes. Compiling every branch pads the result to a dynamic shape, which
    # the Roll in fftshift cannot slice.
    with context.eager_mode():
      x = random_ops.random_uniform([2, 16, 3], dtype=dtypes.float64)
      for k in range(4):

        def f(x, k=k):
          return fft_ops.fftshift(image_ops.rot90(x, k))

        self.assertAllClose(def_function.function(f, jit_compile=True)(x), f(x))


if __name__ == '__main__':
  test.main()
