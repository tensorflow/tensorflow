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
# ==============================================================================
"""Tests for while loops in XLA."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.compiler.tf2xla.python import xla
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import function
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import map_fn
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class WhileTest(xla_test.XLATestCase):

  def testSingletonLoopHandrolled(self):
    # Define a function for the loop body
    @function.Defun(dtypes.int32)
    def loop_body(step):
      step_out = step + constant_op.constant(1, dtype=dtypes.int32)
      return step_out

    # Define a function for the loop condition
    @function.Defun(dtypes.int32)
    def loop_cond(step):
      return step < 10

    with self.cached_session() as sess:
      init_index = array_ops.placeholder(dtypes.int32, [])
      with self.test_scope():
        loop_outputs = xla.while_loop([init_index], loop_cond, loop_body)

      result = sess.run(loop_outputs, {init_index: 0})
      self.assertAllClose(result, [10], rtol=1e-3)

  def testCountingLoopHandrolled(self):
    # Define a function for the loop body
    @function.Defun(dtypes.int32, dtypes.float32)
    def loop_body(step, rsum):
      step_out = step + constant_op.constant(1, dtype=dtypes.int32)
      sum_out = rsum + constant_op.constant(1.5, dtype=dtypes.float32)
      return step_out, sum_out

    # Define a function for the loop condition
    @function.Defun(dtypes.int32, dtypes.float32)
    def loop_cond(step, rsum):
      del rsum
      return step < 10

    with self.cached_session() as sess:
      init_index = array_ops.placeholder(dtypes.int32, [])
      init_sum = array_ops.placeholder(dtypes.float32, [])
      with self.test_scope():
        loop_outputs = xla.while_loop([init_index, init_sum], loop_cond,
                                      loop_body)

      result = sess.run(loop_outputs, {init_index: 0, init_sum: 0.0})
      self.assertAllClose(result, [10, 15.0], rtol=1e-3)
      no_iters_result = sess.run(loop_outputs, {init_index: 10, init_sum: 0.0})
      self.assertAllClose(no_iters_result, [10, 0.0], rtol=1e-3)

  def testCountingLoopHandrolledC64(self):
    # Define a function for the loop body
    @function.Defun(dtypes.int32, dtypes.complex64)
    def loop_body(step, rsum):
      step_out = step + constant_op.constant(1, dtype=dtypes.int32)
      sum_out = rsum + constant_op.constant(1.5 + 2j, dtype=dtypes.complex64)
      return step_out, sum_out

    # Define a function for the loop condition
    @function.Defun(dtypes.int32, dtypes.complex64)
    def loop_cond(step, rsum):
      del rsum
      return step < 10

    with self.cached_session() as sess:
      init_index = array_ops.placeholder(dtypes.int32, [])
      init_sum = array_ops.placeholder(dtypes.complex64, [])
      with self.test_scope():
        loop_outputs = xla.while_loop([init_index, init_sum], loop_cond,
                                      loop_body)

      result = sess.run(loop_outputs, {init_index: 0, init_sum: 0.0})
      self.assertAllClose(result[1], np.complex64(15 + 20j), rtol=1e-3)
      no_iters_result = sess.run(loop_outputs, {init_index: 10, init_sum: 0.0})
      self.assertAllClose(no_iters_result[1], np.complex64(0), rtol=1e-3)

  def testLoopWithConstantOutput(self):
    # Define a function for the loop body
    @function.Defun(dtypes.int32, dtypes.int32)
    def loop_body(step, x):
      del x
      step_out = step + constant_op.constant(1, dtype=dtypes.int32)
      return (step_out, 7)

    # Define a function for the loop condition
    @function.Defun(dtypes.int32, dtypes.int32)
    def loop_cond(step, x):
      del x
      return step < 10

    with self.cached_session() as sess:
      init_index = array_ops.placeholder(dtypes.int32, [])
      with self.test_scope():
        loop_outputs = xla.while_loop([init_index, 42], loop_cond, loop_body)

      result = sess.run(loop_outputs, {init_index: 0})
      self.assertAllClose(result, [10, 7], rtol=1e-3)

  def _testMaxItersSimple(self):
    if is_compile_on_demand():
      self.skipTest("list_ops are not supported in cpu_ondemand")
    with self.cached_session() as sess, self.test_scope():
      xla_context = control_flow_ops.XLAControlFlowContext()
      xla_context.Enter()
      v = constant_op.constant(1.0)
      p = array_ops.placeholder(dtype=dtypes.int32)

      def create_while_loop():
        iterations = array_ops.size(p, name="iterations")
        r = control_flow_ops.while_loop(
            lambda *_: True,
            lambda i, x: (i + 1, v * x), (0, 1.0),
            maximum_iterations=iterations,
            name="outer")
        return array_ops.identity(r[1])

      output = create_while_loop()
      output = gradients_impl.gradients(output, v)[0]

      result = sess.run(output, feed_dict={p: [0, 0, 0]})
      print(result)
      xla_context.Exit()

  def testMaxItersSimple(self):
    self.skipTest("Fails with v1 control flow")
    # This fails with old control.
    # self._testMaxItersSimple()

  @test_util.enable_control_flow_v2
  def testMaxItersSimpleV2(self):
    self._testMaxItersSimple()

  def _testNestedWhileLoopWithMaxItersFromOuterContext(self):
    if is_compile_on_demand():
      self.skipTest("list_ops are not supported in cpu_ondemand")
    with self.cached_session() as sess, self.test_scope():
      xla_context = control_flow_ops.XLAControlFlowContext()
      xla_context.Enter()
      v = constant_op.constant(1.0)
      p = array_ops.placeholder(dtype=dtypes.int32)

      def mid_body_builder(iterations):

        def mid_body(i, x):
          r = control_flow_ops.while_loop(
              lambda *_: True,
              lambda i, x: (i + 1, v * x), (0, x),
              maximum_iterations=iterations,
              name="inner")
          return (i + 1, gradients_impl.gradients(x + r[1], v)[0])

        return mid_body

      def outer_body(i, x):
        iterations = array_ops.size(p, name="iterations")
        return (i + 1, x + control_flow_ops.while_loop(
            lambda *_: True,
            mid_body_builder(iterations), (0, x),
            maximum_iterations=iterations,
            name="mid")[1])

      def create_while_loop():
        r = control_flow_ops.while_loop(
            lambda *_: True,
            outer_body, (0, 1.0),
            maximum_iterations=5,
            name="outer")
        return array_ops.identity(r[1])

      # p:placeholder
      # j = 0
      # i, x = 0, 1.
      # while j++ < 5:
      #   i1, x1 = 0, x
      #   while i1++ < len(p):
      #     i2, x2 = 0, x1
      #     while i2++ < len(p):
      #       x2 = v * x2
      #     x1 = grad(x1 + x2, v)
      #   x = x1
      # output = x
      output = create_while_loop()
      sess.run(output, feed_dict={p: [0, 0, 0]})
      xla_context.Exit()

  def testNestedWhileLoopWithMaxItersFromOuterContext(self):
    self._testNestedWhileLoopWithMaxItersFromOuterContext()

  @test_util.enable_control_flow_v2
  def testNestedWhileLoopWithMaxItersFromOuterContextV2(self):
    self._testNestedWhileLoopWithMaxItersFromOuterContext()

  @test_util.enable_control_flow_v2
  def testMap(self):
    if is_compile_on_demand():
      self.skipTest("list_ops are not supported in cpu_ondemand")
    with self.cached_session(), self.test_scope():
      xla_context = control_flow_ops.XLAControlFlowContext()
      xla_context.Enter()
      nums = [1, 2, 3, 4, 5, 6]
      elems = constant_op.constant(nums, name="data")
      r = map_fn.map_fn(lambda x: math_ops.multiply(math_ops.add(x, 3), 2),
                        elems)
      self.assertAllEqual(r, np.array([(x + 3) * 2 for x in nums]))
      xla_context.Exit()


def is_compile_on_demand():
  return ("TF_XLA_FLAGS" in os.environ and
          "tf_xla_compile_on_demand" in os.environ["TF_XLA_FLAGS"])


if __name__ == "__main__":
  os.environ["TF_XLA_FLAGS"] = ("--tf_xla_min_cluster_size=2 " +
                                os.environ.get("TF_XLA_FLAGS", ""))
  test.main()
