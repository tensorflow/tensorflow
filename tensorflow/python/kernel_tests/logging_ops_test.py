# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tensorflow.kernels.logging_ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class LoggingOpsTest(test.TestCase):

  def testAssertDivideByZero(self):
    with self.test_session() as sess:
      epsilon = ops.convert_to_tensor(1e-20)
      x = ops.convert_to_tensor(0.0)
      y = ops.convert_to_tensor(1.0)
      z = ops.convert_to_tensor(2.0)
      # assert(epsilon < y)
      # z / y
      with sess.graph.control_dependencies([
          control_flow_ops.Assert(
              math_ops.less(epsilon, y), ["Divide-by-zero"])
      ]):
        out = math_ops.div(z, y)
      self.assertAllEqual(2.0, out.eval())
      # assert(epsilon < x)
      # z / x
      #
      # This tests printing out multiple tensors
      with sess.graph.control_dependencies([
          control_flow_ops.Assert(
              math_ops.less(epsilon, x), ["Divide-by-zero", "less than x"])
      ]):
        out = math_ops.div(z, x)
      with self.assertRaisesOpError("less than x"):
        out.eval()


class PrintGradientTest(test.TestCase):

  @test_util.run_in_graph_and_eager_modes()
  def testPrintShape(self):
    inp = constant_op.constant(2.0, shape=[100, 32])
    inp_printed = logging_ops.Print(inp, [inp])
    self.assertEqual(inp.get_shape(), inp_printed.get_shape())

  def testPrintGradient(self):
    with self.test_session():
      inp = constant_op.constant(2.0, shape=[100, 32], name="in")
      w = constant_op.constant(4.0, shape=[10, 100], name="w")
      wx = math_ops.matmul(w, inp, name="wx")
      wx_print = logging_ops.Print(wx, [w, w, w])
      wx_grad = gradients_impl.gradients(wx, w)[0]
      wx_print_grad = gradients_impl.gradients(wx_print, w)[0]
      wxg = wx_grad.eval()
      wxpg = wx_print_grad.eval()
    self.assertAllEqual(wxg, wxpg)


if __name__ == "__main__":
  test.main()
