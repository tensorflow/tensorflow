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
"""Tests for Rate."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.rate import rate
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


class RateTest(test.TestCase):

  @test_util.run_in_graph_and_eager_modes()
  def testBuildRate(self):
    m = rate.Rate()
    m.build(
        constant_op.constant([1], dtype=dtypes.float32),
        constant_op.constant([2], dtype=dtypes.float32))
    old_numer = m.numer
    m(
        constant_op.constant([2], dtype=dtypes.float32),
        constant_op.constant([2], dtype=dtypes.float32))
    self.assertTrue(old_numer is m.numer)

  @test_util.run_in_graph_and_eager_modes()
  def testBasic(self):
    with self.cached_session():
      r_ = rate.Rate()
      a = r_(array_ops.ones([1]), denominator=array_ops.ones([1]))
      self.evaluate(variables.global_variables_initializer())
      self.evaluate(variables.local_variables_initializer())
      self.assertEqual([[1]], self.evaluate(a))
      b = r_(constant_op.constant([2]), denominator=constant_op.constant([2]))
      self.assertEqual([[1]], self.evaluate(b))
      c = r_(constant_op.constant([4]), denominator=constant_op.constant([3]))
      self.assertEqual([[2]], self.evaluate(c))
      d = r_(constant_op.constant([16]), denominator=constant_op.constant([3]))
      self.assertEqual([[0]], self.evaluate(d))  # divide by 0

  def testNamesWithSpaces(self):
    m1 = rate.Rate(name="has space")
    m1(array_ops.ones([1]), array_ops.ones([1]))
    self.assertEqual(m1.name, "has space")
    self.assertEqual(m1.prev_values.name, "has_space_1/prev_values:0")

  @test_util.run_in_graph_and_eager_modes()
  def testWhileLoop(self):
    with self.cached_session():
      r_ = rate.Rate()

      def body(value, denom, i, ret_rate):
        i += 1
        ret_rate = r_(value, denom)
        with ops.control_dependencies([ret_rate]):
          value = math_ops.add(value, 2)
          denom = math_ops.add(denom, 1)
        return [value, denom, i, ret_rate]

      def condition(v, d, i, r):
        del v, d, r  # unused vars by condition
        return math_ops.less(i, 100)

      i = constant_op.constant(0)
      value = constant_op.constant([1], dtype=dtypes.float64)
      denom = constant_op.constant([1], dtype=dtypes.float64)
      ret_rate = r_(value, denom)
      self.evaluate(variables.global_variables_initializer())
      self.evaluate(variables.local_variables_initializer())
      loop = control_flow_ops.while_loop(condition, body,
                                         [value, denom, i, ret_rate])
      self.assertEqual([[2]], self.evaluate(loop[3]))


if __name__ == "__main__":
  test.main()
