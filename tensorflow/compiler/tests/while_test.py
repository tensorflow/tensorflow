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

import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.compiler.tf2xla.python import xla
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import function
from tensorflow.python.ops import array_ops
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


if __name__ == '__main__':
  test.main()
