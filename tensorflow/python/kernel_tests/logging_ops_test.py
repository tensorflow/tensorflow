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

import os
import sys
import tempfile

from tensorflow.python.eager import context
from tensorflow.python.eager import function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test

class LoggingOpsTest(test.TestCase):

  def testAssertDivideByZero(self):
    with self.cached_session() as sess:
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
      self.assertAllEqual(2.0, self.evaluate(out))
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
        self.evaluate(out)


class PrintV2Test(test.TestCase):

  @test_util.run_in_graph_and_eager_modes()
  def testPrintOneTensor(self):
    with self.cached_session():
      tensor = math_ops.range(10)
      with self.captureWritesToStream(sys.stderr) as printed:
        print_op = logging_ops.print_v2(tensor)
        self.evaluate(print_op)

      expected = "[0 1 2 ... 7 8 9]"
      self.assertTrue((expected + "\n") in printed.contents())

  @test_util.run_in_graph_and_eager_modes()
  def testPrintOneTensorVarySummarize(self):
    with self.cached_session():
      tensor = math_ops.range(10)
      with self.captureWritesToStream(sys.stderr) as printed:
        print_op = logging_ops.print_v2(tensor, summarize=1)
        self.evaluate(print_op)

      expected = "[0 ... 9]"
      self.assertTrue((expected + "\n") in printed.contents())

    with self.cached_session():
      tensor = math_ops.range(10)
      with self.captureWritesToStream(sys.stderr) as printed:
        print_op = logging_ops.print_v2(tensor, summarize=2)
        self.evaluate(print_op)

      expected = "[0 1 ... 8 9]"
      self.assertTrue((expected + "\n") in printed.contents())

    with self.cached_session():
      tensor = math_ops.range(10)
      with self.captureWritesToStream(sys.stderr) as printed:
        print_op = logging_ops.print_v2(tensor, summarize=3)
        self.evaluate(print_op)

      expected = "[0 1 2 ... 7 8 9]"
      self.assertTrue((expected + "\n") in printed.contents())

    with self.cached_session():
      tensor = math_ops.range(10)
      with self.captureWritesToStream(sys.stderr) as printed:
        print_op = logging_ops.print_v2(tensor, summarize=-1)
        self.evaluate(print_op)

      expected = "[0 1 2 3 4 5 6 7 8 9]"
      self.assertTrue((expected + "\n") in printed.contents())

  @test_util.run_in_graph_and_eager_modes()
  def testPrintOneVariable(self):
    with self.cached_session():
      var = variables.Variable(math_ops.range(10))
      if not context.executing_eagerly():
        variables.global_variables_initializer().run()
      with self.captureWritesToStream(sys.stderr) as printed:
        print_op = logging_ops.print_v2(var)
        self.evaluate(print_op)
      expected = "[0 1 2 ... 7 8 9]"
      self.assertTrue((expected + "\n") in printed.contents())

  @test_util.run_in_graph_and_eager_modes()
  def testPrintTwoVariablesInStructWithAssignAdd(self):
    with self.cached_session():
      var_one = variables.Variable(2.14)
      plus_one = var_one.assign_add(1.0)
      var_two = variables.Variable(math_ops.range(10))
      if not context.executing_eagerly():
        variables.global_variables_initializer().run()
      with self.captureWritesToStream(sys.stderr) as printed:
        self.evaluate(plus_one)
        print_op = logging_ops.print_v2(var_one, {"second": var_two})
        self.evaluate(print_op)
      expected = "3.14 {'second': [0 1 2 ... 7 8 9]}"
      self.assertTrue((expected + "\n") in printed.contents())

  @test_util.run_in_graph_and_eager_modes()
  def testPrintTwoTensors(self):
    with self.cached_session():
      tensor = math_ops.range(10)
      with self.captureWritesToStream(sys.stderr) as printed:
        print_op = logging_ops.print_v2(tensor, tensor * 10)
        self.evaluate(print_op)
      expected = "[0 1 2 ... 7 8 9] [0 10 20 ... 70 80 90]"
      self.assertTrue((expected + "\n") in printed.contents())

  @test_util.run_in_graph_and_eager_modes()
  def testPrintPlaceholderGeneration(self):
    with self.cached_session():
      tensor = math_ops.range(10)
      with self.captureWritesToStream(sys.stderr) as printed:
        print_op = logging_ops.print_v2("{}6", {"{}": tensor * 10})
        self.evaluate(print_op)
      expected = "{}6 {'{}': [0 10 20 ... 70 80 90]}"
      self.assertTrue((expected + "\n") in printed.contents())

  @test_util.run_in_graph_and_eager_modes()
  def testPrintNoTensors(self):
    with self.cached_session():
      with self.captureWritesToStream(sys.stderr) as printed:
        print_op = logging_ops.print_v2(23, [23, 5], {"6": 12})
        self.evaluate(print_op)
      expected = "23 [23, 5] {'6': 12}"
      self.assertTrue((expected + "\n") in printed.contents())

  @test_util.run_in_graph_and_eager_modes()
  def testPrintFloatScalar(self):
    with self.cached_session():
      tensor = ops.convert_to_tensor(434.43)
      with self.captureWritesToStream(sys.stderr) as printed:
        print_op = logging_ops.print_v2(tensor)
        self.evaluate(print_op)
      expected = "434.43"
      self.assertTrue((expected + "\n") in printed.contents())

  @test_util.run_in_graph_and_eager_modes()
  def testPrintStringScalar(self):
    with self.cached_session():
      tensor = ops.convert_to_tensor("scalar")
      with self.captureWritesToStream(sys.stderr) as printed:
        print_op = logging_ops.print_v2(tensor)
        self.evaluate(print_op)
      expected = "scalar"
      self.assertTrue((expected + "\n") in printed.contents())

  @test_util.run_in_graph_and_eager_modes()
  def testPrintComplexTensorStruct(self):
    with self.cached_session():
      tensor = math_ops.range(10)
      small_tensor = constant_op.constant([0.3, 12.4, -16.1])
      big_tensor = math_ops.mul(tensor, 10)
      with self.captureWritesToStream(sys.stderr) as printed:
        print_op = logging_ops.print_v2(
            "first:", tensor, "middle:",
            {"small": small_tensor, "Big": big_tensor}, 10,
            [tensor * 2, tensor])
        self.evaluate(print_op)
      # Note that the keys in the dict will always be sorted,
      # so 'Big' comes before 'small'
      expected = ("first: [0 1 2 ... 7 8 9] "
                  "middle: {'Big': [0 10 20 ... 70 80 90], "
                  "'small': [0.3 12.4 -16.1]} "
                  "10 [[0 2 4 ... 14 16 18], [0 1 2 ... 7 8 9]]")
      self.assertTrue((expected + "\n") in printed.contents())

  @test_util.run_in_graph_and_eager_modes()
  def testPrintSparseTensor(self):
    with self.cached_session():
      ind = [[0, 0], [1, 0], [1, 3], [4, 1], [1, 4], [3, 2], [3, 3]]
      val = [0, 10, 13, 4, 14, 32, 33]
      shape = [5, 6]

      sparse = sparse_tensor.SparseTensor(
          constant_op.constant(ind, dtypes.int64),
          constant_op.constant(val, dtypes.int64),
          constant_op.constant(shape, dtypes.int64))

      with self.captureWritesToStream(sys.stderr) as printed:
        print_op = logging_ops.print_v2(sparse)
        self.evaluate(print_op)
      expected = ("'SparseTensor(indices=[[0 0]\n"
                  " [1 0]\n"
                  " [1 3]\n"
                  " ...\n"
                  " [1 4]\n"
                  " [3 2]\n"
                  " [3 3]], values=[0 10 13 ... 14 32 33], shape=[5 6])'")
      self.assertTrue((expected + "\n") in printed.contents())

  @test_util.run_in_graph_and_eager_modes()
  def testPrintSparseTensorInDataStruct(self):
    with self.cached_session():
      ind = [[0, 0], [1, 0], [1, 3], [4, 1], [1, 4], [3, 2], [3, 3]]
      val = [0, 10, 13, 4, 14, 32, 33]
      shape = [5, 6]

      sparse = sparse_tensor.SparseTensor(
          constant_op.constant(ind, dtypes.int64),
          constant_op.constant(val, dtypes.int64),
          constant_op.constant(shape, dtypes.int64))

      with self.captureWritesToStream(sys.stderr) as printed:
        print_op = logging_ops.print_v2([sparse])
        self.evaluate(print_op)
      expected = ("['SparseTensor(indices=[[0 0]\n"
                  " [1 0]\n"
                  " [1 3]\n"
                  " ...\n"
                  " [1 4]\n"
                  " [3 2]\n"
                  " [3 3]], values=[0 10 13 ... 14 32 33], shape=[5 6])']")
      self.assertTrue((expected + "\n") in printed.contents())

  @test_util.run_in_graph_and_eager_modes()
  def testPrintOneTensorStdout(self):
    with self.cached_session():
      tensor = math_ops.range(10)
      with self.captureWritesToStream(sys.stdout) as printed:
        print_op = logging_ops.print_v2(
            tensor, output_stream=sys.stdout)
        self.evaluate(print_op)
      expected = "[0 1 2 ... 7 8 9]"
      self.assertTrue((expected + "\n") in printed.contents())

  @test_util.run_in_graph_and_eager_modes()
  def testPrintTensorsToFile(self):
    tmpfile_name = tempfile.mktemp(".printv2_test")
    tensor_0 = math_ops.range(0, 10)
    print_op_0 = logging_ops.print_v2(tensor_0,
                                      output_stream="file://"+tmpfile_name)
    self.evaluate(print_op_0)
    tensor_1 = math_ops.range(11, 20)
    print_op_1 = logging_ops.print_v2(tensor_1,
                                      output_stream="file://"+tmpfile_name)
    self.evaluate(print_op_1)
    try:
      f = open(tmpfile_name, "r")
      line_0 = f.readline()
      expected_0 = "[0 1 2 ... 7 8 9]"
      self.assertTrue(expected_0 in line_0)
      line_1 = f.readline()
      expected_1 = "[11 12 13 ... 17 18 19]"
      self.assertTrue(expected_1 in line_1)
      f.close()
      os.remove(tmpfile_name)
    except IOError as e:
      self.fail(e)

  @test_util.run_in_graph_and_eager_modes()
  def testInvalidOutputStreamRaisesError(self):
    with self.cached_session():
      tensor = math_ops.range(10)
      with self.assertRaises(ValueError):
        print_op = logging_ops.print_v2(
            tensor, output_stream="unknown")
        self.evaluate(print_op)

  def testPrintOpName(self):
    with self.cached_session():
      tensor = math_ops.range(10)
      print_op = logging_ops.print_v2(tensor, name="print_name")
      self.assertEqual(print_op.name, "print_name")

  def testNoDuplicateFormatOpGraphModeAfterExplicitFormat(self):
    with self.cached_session():
      tensor = math_ops.range(10)
      formatted_string = string_ops.string_format("{}", tensor)
      print_op = logging_ops.print_v2(formatted_string)
      self.evaluate(print_op)
      graph_ops = ops.get_default_graph().get_operations()
      format_ops = [op for op in graph_ops if op.type == "StringFormat"]
      # Should be only 1 format_op for graph mode.
      self.assertEqual(len(format_ops), 1)

  def testPrintOneTensorEagerOnOpCreate(self):
    with self.cached_session():
      with context.eager_mode():
        tensor = math_ops.range(10)
        expected = "[0 1 2 ... 7 8 9]"
        with self.captureWritesToStream(sys.stderr) as printed:
          logging_ops.print_v2(tensor)
        self.assertTrue((expected + "\n") in printed.contents())

  def testPrintsOrderedInDefun(self):
    with context.eager_mode():

      @function.defun
      def prints():
        logging_ops.print_v2("A")
        logging_ops.print_v2("B")
        logging_ops.print_v2("C")

      with self.captureWritesToStream(sys.stderr) as printed:
        prints()
      self.assertTrue(("A\nB\nC\n") in printed.contents())

  @test_util.run_in_graph_and_eager_modes()
  def testPrintInDefunWithoutExplicitEvalOfPrint(self):
    @function.defun
    def f():
      tensor = math_ops.range(10)
      logging_ops.print_v2(tensor)
      return tensor

    expected = "[0 1 2 ... 7 8 9]"
    with self.captureWritesToStream(sys.stderr) as printed_one:
      x = f()
      self.evaluate(x)
    self.assertTrue((expected + "\n") in printed_one.contents())

    # We execute the function again to make sure it doesn't only print on the
    # first call.
    with self.captureWritesToStream(sys.stderr) as printed_two:
      y = f()
      self.evaluate(y)
    self.assertTrue((expected + "\n") in printed_two.contents())


class PrintGradientTest(test.TestCase):

  @test_util.run_in_graph_and_eager_modes
  def testPrintShape(self):
    inp = constant_op.constant(2.0, shape=[100, 32])
    inp_printed = logging_ops.Print(inp, [inp])
    self.assertEqual(inp.get_shape(), inp_printed.get_shape())

  def testPrintString(self):
    inp = constant_op.constant(2.0, shape=[100, 32])
    inp_printed = logging_ops.Print(inp, ["hello"])
    self.assertEqual(inp.get_shape(), inp_printed.get_shape())

  def testPrintGradient(self):
    with self.cached_session():
      inp = constant_op.constant(2.0, shape=[100, 32], name="in")
      w = constant_op.constant(4.0, shape=[10, 100], name="w")
      wx = math_ops.matmul(w, inp, name="wx")
      wx_print = logging_ops.Print(wx, [w, w, w])
      wx_grad = gradients_impl.gradients(wx, w)[0]
      wx_print_grad = gradients_impl.gradients(wx_print, w)[0]
      wxg = self.evaluate(wx_grad)
      wxpg = self.evaluate(wx_print_grad)
    self.assertAllEqual(wxg, wxpg)


if __name__ == "__main__":
  test.main()
