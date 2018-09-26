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


from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.util import compat


class StringFormatOpTest(test.TestCase):

  @test_util.run_in_graph_and_eager_modes()
  def testFormatOneTensorOneDim(self):
    with self.cached_session():
      tensor = math_ops.range(10)
      format_output = string_ops.string_format("{}", tensor)
      out = self.evaluate(format_output)
      expected = "[0 1 2 ... 7 8 9]"
      self.assertEqual(compat.as_text(out), expected)

    with self.cached_session():
      tensor = math_ops.range(10)
      format_output = string_ops.string_format("{}", [tensor])
      out = self.evaluate(format_output)
      expected = "[0 1 2 ... 7 8 9]"
      self.assertEqual(compat.as_text(out), expected)

  @test_util.run_in_graph_and_eager_modes()
  def testFormatOneVariableScalar(self):
    with self.cached_session():
      var = variables.Variable(3.34)
      format_output = string_ops.string_format("{}", [var])
      if not context.executing_eagerly():
        variables.global_variables_initializer().run()
      out = self.evaluate(format_output)
      expected = "3.34"
      self.assertEqual(compat.as_text(out), expected)

  @test_util.run_in_graph_and_eager_modes()
  def testFormatOneVariableOneDim(self):
    with self.cached_session():
      var = variables.Variable(math_ops.range(10))
      format_output = string_ops.string_format("{}", [var])
      if not context.executing_eagerly():
        variables.global_variables_initializer().run()
      out = self.evaluate(format_output)
      expected = "[0 1 2 ... 7 8 9]"
      self.assertEqual(compat.as_text(out), expected)

  @test_util.run_in_graph_and_eager_modes()
  def testFormatTwoVariablesWithAssignAdd(self):
    with self.cached_session():
      var_one = variables.Variable(2.14)
      plus_one = var_one.assign_add(1.0)
      var_two = variables.Variable(math_ops.range(10))
      format_output = string_ops.string_format("{}, {}", [var_one, var_two])
      if not context.executing_eagerly():
        variables.global_variables_initializer().run()
      self.evaluate(plus_one)
      out = self.evaluate(format_output)
      expected = "3.14, [0 1 2 ... 7 8 9]"
      self.assertEqual(compat.as_text(out), expected)

  @test_util.run_in_graph_and_eager_modes()
  def testFormatOneTensorOneDimFloat(self):
    with self.cached_session():
      tensor = constant_op.constant([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
      format_output = string_ops.string_format("{}", tensor)
      out = self.evaluate(format_output)
      expected = "[0 0.1 0.2 ... 0.5 0.6 0.7]"
      self.assertEqual(compat.as_text(out), expected)

  @test_util.run_in_graph_and_eager_modes()
  def testFormatOneTensorOneDimMatchesSummarize(self):
    with self.cached_session():
      tensor = math_ops.range(6)
      format_output = string_ops.string_format("{}", tensor, summarize=3)
      out = self.evaluate(format_output)
      expected = "[0 1 2 3 4 5]"
      self.assertEqual(compat.as_text(out), expected)

  @test_util.run_in_graph_and_eager_modes()
  def testFormatOneTensorOneDimVarySummarize(self):
    with self.cached_session():
      tensor = math_ops.range(6)
      format_output = string_ops.string_format("{}", tensor, summarize=-1)
      out = self.evaluate(format_output)
      expected = "[0 1 2 3 4 5]"
      self.assertEqual(compat.as_text(out), expected)

    with self.cached_session():
      tensor = math_ops.range(6)
      format_output = string_ops.string_format("{}", tensor, summarize=1)
      out = self.evaluate(format_output)
      expected = "[0 ... 5]"
      self.assertEqual(compat.as_text(out), expected)

    with self.cached_session():
      tensor = math_ops.range(6)
      format_output = string_ops.string_format("{}", tensor, summarize=2)
      out = self.evaluate(format_output)
      expected = "[0 1 ... 4 5]"
      self.assertEqual(compat.as_text(out), expected)

    with self.cached_session():
      tensor = math_ops.range(6)
      format_output = string_ops.string_format("{}", tensor, summarize=10)
      out = self.evaluate(format_output)
      expected = "[0 1 2 3 4 5]"
      self.assertEqual(compat.as_text(out), expected)

  @test_util.run_in_graph_and_eager_modes()
  def testFormatOneTensorOneDimAlmostSummarize(self):
    with self.cached_session():
      tensor = math_ops.range(5)
      format_output = string_ops.string_format("{}", tensor, summarize=3)
      out = self.evaluate(format_output)
      expected = "[0 1 2 3 4]"
      self.assertEqual(compat.as_text(out), expected)

  @test_util.run_in_graph_and_eager_modes()
  def testFormatOneTensorTwoDimLessThanSummarize(self):
    with self.cached_session():
      tensor = array_ops.reshape(math_ops.range(4), [2, 2])
      format_output = string_ops.string_format("{}", tensor, summarize=3)
      out = self.evaluate(format_output)
      expected = ("[[0 1]\n"
                  " [2 3]]")
      self.assertEqual(compat.as_text(out), expected)

  @test_util.run_in_graph_and_eager_modes()
  def testFormatOneTensorTwoDim(self):
    with self.cached_session():
      tensor = array_ops.reshape(math_ops.range(100), [10, 10])
      format_output = string_ops.string_format("{}", tensor)
      out = self.evaluate(format_output)
      expected = ("[[0 1 2 ... 7 8 9]\n"
                  " [10 11 12 ... 17 18 19]\n"
                  " [20 21 22 ... 27 28 29]\n"
                  " ...\n"
                  " [70 71 72 ... 77 78 79]\n"
                  " [80 81 82 ... 87 88 89]\n"
                  " [90 91 92 ... 97 98 99]]")
      self.assertEqual(compat.as_text(out), expected)

  @test_util.run_in_graph_and_eager_modes()
  def testFormatOneTensorTwoDimSummarizeTwo(self):
    with self.cached_session():
      tensor = array_ops.reshape(math_ops.range(100), [10, 10])
      format_output = string_ops.string_format("{}", tensor, summarize=2)
      out = self.evaluate(format_output)
      expected = ("[[0 1 ... 8 9]\n"
                  " [10 11 ... 18 19]\n"
                  " ...\n"
                  " [80 81 ... 88 89]\n"
                  " [90 91 ... 98 99]]")
      self.assertEqual(compat.as_text(out), expected)

  @test_util.run_in_graph_and_eager_modes()
  def testFormatOneTensorThreeDim(self):
    with self.cached_session():
      tensor = array_ops.reshape(math_ops.range(1000), [10, 10, 10])
      format_output = string_ops.string_format("{}", tensor)
      out = self.evaluate(format_output)
      expected = ("[[[0 1 2 ... 7 8 9]\n"
                  "  [10 11 12 ... 17 18 19]\n"
                  "  [20 21 22 ... 27 28 29]\n"
                  "  ...\n"
                  "  [70 71 72 ... 77 78 79]\n"
                  "  [80 81 82 ... 87 88 89]\n"
                  "  [90 91 92 ... 97 98 99]]\n"
                  "\n"
                  " [[100 101 102 ... 107 108 109]\n"
                  "  [110 111 112 ... 117 118 119]\n"
                  "  [120 121 122 ... 127 128 129]\n"
                  "  ...\n  [170 171 172 ... 177 178 179]\n"
                  "  [180 181 182 ... 187 188 189]\n"
                  "  [190 191 192 ... 197 198 199]]\n"
                  "\n"
                  " [[200 201 202 ... 207 208 209]\n"
                  "  [210 211 212 ... 217 218 219]\n"
                  "  [220 221 222 ... 227 228 229]\n"
                  "  ...\n"
                  "  [270 271 272 ... 277 278 279]\n"
                  "  [280 281 282 ... 287 288 289]\n"
                  "  [290 291 292 ... 297 298 299]]\n"
                  "\n"
                  " ...\n"
                  "\n"
                  " [[700 701 702 ... 707 708 709]\n"
                  "  [710 711 712 ... 717 718 719]\n"
                  "  [720 721 722 ... 727 728 729]\n"
                  "  ...\n"
                  "  [770 771 772 ... 777 778 779]\n"
                  "  [780 781 782 ... 787 788 789]\n"
                  "  [790 791 792 ... 797 798 799]]\n"
                  "\n"
                  " [[800 801 802 ... 807 808 809]\n"
                  "  [810 811 812 ... 817 818 819]\n"
                  "  [820 821 822 ... 827 828 829]\n"
                  "  ...\n"
                  "  [870 871 872 ... 877 878 879]\n"
                  "  [880 881 882 ... 887 888 889]\n"
                  "  [890 891 892 ... 897 898 899]]\n"
                  "\n"
                  " [[900 901 902 ... 907 908 909]\n"
                  "  [910 911 912 ... 917 918 919]\n"
                  "  [920 921 922 ... 927 928 929]\n"
                  "  ...\n"
                  "  [970 971 972 ... 977 978 979]\n"
                  "  [980 981 982 ... 987 988 989]\n"
                  "  [990 991 992 ... 997 998 999]]]")
      self.assertEqual(compat.as_text(out), expected)

  @test_util.run_in_graph_and_eager_modes()
  def testFormatOneTensorTemplatePrefix(self):
    with self.cached_session():
      tensor = array_ops.reshape(math_ops.range(100), [10, 10])
      format_output = string_ops.string_format("tensor summary: {}", tensor)
      out = self.evaluate(format_output)
      expected = ("tensor summary: [[0 1 2 ... 7 8 9]\n"
                  " [10 11 12 ... 17 18 19]\n"
                  " [20 21 22 ... 27 28 29]\n"
                  " ...\n"
                  " [70 71 72 ... 77 78 79]\n"
                  " [80 81 82 ... 87 88 89]\n"
                  " [90 91 92 ... 97 98 99]]")
      self.assertEqual(compat.as_text(out), expected)

  @test_util.run_in_graph_and_eager_modes()
  def testFormatOneTensorTemplatePrefixAndSuffix(self):
    with self.cached_session():
      tensor = array_ops.reshape(math_ops.range(100), [10, 10])
      format_output = string_ops.string_format("tensor summary: {}, suffix",
                                               tensor)
      out = self.evaluate(format_output)
      expected = ("tensor summary: [[0 1 2 ... 7 8 9]\n"
                  " [10 11 12 ... 17 18 19]\n"
                  " [20 21 22 ... 27 28 29]\n"
                  " ...\n"
                  " [70 71 72 ... 77 78 79]\n"
                  " [80 81 82 ... 87 88 89]\n"
                  " [90 91 92 ... 97 98 99]], suffix")
      self.assertEqual(compat.as_text(out), expected)

  @test_util.run_in_graph_and_eager_modes()
  def testFormatOneTensorTemplateSuffix(self):
    with self.cached_session():
      tensor = array_ops.reshape(math_ops.range(100), [10, 10])
      format_output = string_ops.string_format("{}, suffix", tensor)
      out = self.evaluate(format_output)
      expected = ("[[0 1 2 ... 7 8 9]\n"
                  " [10 11 12 ... 17 18 19]\n"
                  " [20 21 22 ... 27 28 29]\n"
                  " ...\n"
                  " [70 71 72 ... 77 78 79]\n"
                  " [80 81 82 ... 87 88 89]\n"
                  " [90 91 92 ... 97 98 99]], suffix")
      self.assertEqual(compat.as_text(out), expected)

  @test_util.run_in_graph_and_eager_modes()
  def testFormatNoTensor(self):
    with self.cached_session():
      format_output = string_ops.string_format("No tensor.", ())
      out = self.evaluate(format_output)
      expected = "No tensor."
      self.assertEqual(compat.as_text(out), expected)

  @test_util.run_in_graph_and_eager_modes()
  def testFormatMultiTensor(self):
    with self.cached_session():
      tensor_one = array_ops.reshape(math_ops.range(100), [10, 10])
      tensor_two = tensor_one * 10
      format_output = string_ops.string_format("One: {},\nTwo: {}",
                                               (tensor_one, tensor_two))
      out = self.evaluate(format_output)
      expected = ("One: [[0 1 2 ... 7 8 9]\n"
                  " [10 11 12 ... 17 18 19]\n"
                  " [20 21 22 ... 27 28 29]\n"
                  " ...\n"
                  " [70 71 72 ... 77 78 79]\n"
                  " [80 81 82 ... 87 88 89]\n"
                  " [90 91 92 ... 97 98 99]],\n"
                  "Two: [[0 10 20 ... 70 80 90]\n"
                  " [100 110 120 ... 170 180 190]\n"
                  " [200 210 220 ... 270 280 290]\n"
                  " ...\n"
                  " [700 710 720 ... 770 780 790]\n"
                  " [800 810 820 ... 870 880 890]\n"
                  " [900 910 920 ... 970 980 990]]")
      self.assertEqual(compat.as_text(out), expected)

  @test_util.run_in_graph_and_eager_modes()
  def testFormatSummarizeOne(self):
    with self.cached_session():
      tensor = array_ops.reshape(math_ops.range(100), [10, 10])
      format_output = string_ops.string_format("tensor summary: {}", tensor,
                                               summarize=1)
      out = self.evaluate(format_output)
      expected = ("tensor summary: [[0 ... 9]\n"
                  " ...\n"
                  " [90 ... 99]]")
      self.assertEqual(compat.as_text(out), expected)

  @test_util.run_in_graph_and_eager_modes()
  def testFormatSummarizeTwo(self):
    with self.cached_session():
      tensor = array_ops.reshape(math_ops.range(100), [10, 10])
      format_output = string_ops.string_format("tensor summary: {}", tensor,
                                               summarize=2)
      out = self.evaluate(format_output)
      expected = ("tensor summary: [[0 1 ... 8 9]\n"
                  " [10 11 ... 18 19]\n"
                  " ...\n"
                  " [80 81 ... 88 89]\n"
                  " [90 91 ... 98 99]]")
      self.assertEqual(compat.as_text(out), expected)

  @test_util.run_in_graph_and_eager_modes()
  def testFormatPlaceholder(self):
    with self.cached_session():
      tensor = array_ops.reshape(math_ops.range(100), [10, 10])
      format_output = string_ops.string_format("tensor summary: %t%", tensor,
                                               placeholder="%t%")
      out = self.evaluate(format_output)
      expected = ("tensor summary: [[0 1 2 ... 7 8 9]\n"
                  " [10 11 12 ... 17 18 19]\n"
                  " [20 21 22 ... 27 28 29]\n"
                  " ...\n"
                  " [70 71 72 ... 77 78 79]\n"
                  " [80 81 82 ... 87 88 89]\n"
                  " [90 91 92 ... 97 98 99]]")
      self.assertEqual(compat.as_text(out), expected)

  @test_util.run_in_graph_and_eager_modes()
  def testTensorCountMustMatchPlaceholderCount(self):
    with self.cached_session():
      with self.assertRaisesRegexp(
          ValueError, r"2 placeholder\(s\) in template does not match 1 "
                      r"tensor\(s\) provided as input"):
        tensor = math_ops.range(10)
        format_output = string_ops.string_format("{} {}", tensor)
        self.evaluate(format_output)
    with self.cached_session():
      with self.assertRaisesRegexp(
          ValueError, r"2 placeholder\(s\) in template does not match 1 "
                      r"tensor\(s\) provided as input"):
        tensor = math_ops.range(10)
        format_output = string_ops.string_format("{} {}", [tensor])
        self.evaluate(format_output)
    with self.cached_session():
      with self.assertRaisesRegexp(
          ValueError, r"1 placeholder\(s\) in template does not match 2 "
                      r"tensor\(s\) provided as input"):
        tensor = math_ops.range(10)
        format_output = string_ops.string_format("{}", (tensor, tensor))
        self.evaluate(format_output)


if __name__ == "__main__":
  test.main()
