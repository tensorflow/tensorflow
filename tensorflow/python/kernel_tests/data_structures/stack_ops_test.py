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
"""Tests for tensorflow.ops.stack_ops."""

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_util
from tensorflow.python.ops import gen_data_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import while_loop
from tensorflow.python.platform import test


class StackOpTest(test.TestCase):

  def _testStackPushPop(self, use_gpu):
    with self.cached_session(use_gpu=use_gpu):
      h = gen_data_flow_ops.stack_v2(
          -1, elem_type=dtypes.float32, stack_name="foo")
      c = gen_data_flow_ops.stack_push_v2(h, [[4.0, 5.0]])
      with ops.control_dependencies([c]):
        c1 = gen_data_flow_ops.stack_pop_v2(h, dtypes.float32)
      self.assertAllClose([[4.0, 5.0]], self.evaluate(c1))

  @test_util.run_deprecated_v1
  def testStackPushPop(self):
    self._testStackPushPop(use_gpu=False)
    self._testStackPushPop(use_gpu=True)

  def _testStackPushPopSwap(self, use_gpu):
    with self.cached_session(use_gpu=use_gpu):
      a = np.arange(2000)
      x = constant_op.constant(a, dtype=dtypes.float32)
      h = gen_data_flow_ops.stack_v2(
          -1, elem_type=dtypes.float32, stack_name="foo")
      c = gen_data_flow_ops.stack_push_v2(h, x, swap_memory=True)
      with ops.control_dependencies([c]):
        c1 = gen_data_flow_ops.stack_pop_v2(h, dtypes.float32)
      self.assertAllClose(a, self.evaluate(c1))

  @test_util.run_deprecated_v1
  def testStackPushPopSwap(self):
    self._testStackPushPopSwap(use_gpu=False)
    self._testStackPushPopSwap(use_gpu=True)

  def _testStackWhileSwap(self, use_gpu):
    with self.cached_session(use_gpu=use_gpu):
      n = constant_op.constant(0)
      h = gen_data_flow_ops.stack_v2(
          -1, elem_type=dtypes.float32, stack_name="foo")

      def c(x):
        return math_ops.less(x, 10)

      def b(x):
        with ops.control_dependencies([x]):
          a = constant_op.constant(np.ones(2000), dtype=dtypes.float32)
          v = gen_data_flow_ops.stack_push_v2(h, a, swap_memory=True)
        with ops.control_dependencies([v]):
          return math_ops.add(x, 1)

      r = while_loop.while_loop(c, b, [n])

      v = constant_op.constant(np.zeros(2000), dtype=dtypes.float32)

      def c1(x, y):
        del y
        return math_ops.greater(x, 0)

      def b1(x, y):
        nx = math_ops.subtract(x, 1)
        ny = y + gen_data_flow_ops.stack_pop_v2(h, dtypes.float32)
        return [nx, ny]

      _, ry = while_loop.while_loop(
          c1, b1, [r, v],
          [r.get_shape(), tensor_shape.unknown_shape()])
      self.assertAllClose(np.ones(2000) * 10.0, self.evaluate(ry))

  @test_util.run_v1_only("b/120545219")
  def testStackWhileSwap(self):
    self._testStackWhileSwap(use_gpu=False)
    self._testStackWhileSwap(use_gpu=True)

  def _testMultiStack(self, use_gpu):
    with self.cached_session(use_gpu=use_gpu):
      h1 = gen_data_flow_ops.stack_v2(
          -1, elem_type=dtypes.float32, stack_name="foo")
      c1 = gen_data_flow_ops.stack_push_v2(h1, 4.0)
      with ops.control_dependencies([c1]):
        c1 = gen_data_flow_ops.stack_pop_v2(h1, dtypes.float32)
      h2 = gen_data_flow_ops.stack_v2(
          -1, elem_type=dtypes.float32, stack_name="bar")
      c2 = gen_data_flow_ops.stack_push_v2(h2, 5.0)
      with ops.control_dependencies([c2]):
        c2 = gen_data_flow_ops.stack_pop_v2(h2, dtypes.float32)
      r = c1 + c2
      self.assertAllClose(9.0, self.evaluate(r))

  @test_util.run_deprecated_v1
  def testMultiStack(self):
    self._testMultiStack(use_gpu=False)
    self._testMultiStack(use_gpu=True)

  def _testSameNameStacks(self, use_gpu):
    """Different stacks with the same name do not interfere."""
    with self.cached_session(use_gpu=use_gpu) as sess:
      h1 = gen_data_flow_ops.stack_v2(
          -1, elem_type=dtypes.float32, stack_name="foo")
      h2 = gen_data_flow_ops.stack_v2(
          -1, elem_type=dtypes.float32, stack_name="foo")

      c1 = gen_data_flow_ops.stack_push_v2(h1, 4.0)
      with ops.control_dependencies([c1]):
        c2 = gen_data_flow_ops.stack_push_v2(h2, 5.0)
      with ops.control_dependencies([c2]):
        pop1 = gen_data_flow_ops.stack_pop_v2(h1, dtypes.float32)
        pop2 = gen_data_flow_ops.stack_pop_v2(h2, dtypes.float32)

      out1, out2 = self.evaluate([pop1, pop2])
      self.assertAllClose(out1, 4.0)
      self.assertAllClose(out2, 5.0)

  @test_util.run_deprecated_v1
  def testSameNameStacks(self):
    self._testSameNameStacks(use_gpu=False)
    self._testSameNameStacks(use_gpu=True)

  def _testCloseStack(self, use_gpu):
    with self.cached_session(use_gpu=use_gpu) as sess:
      h = gen_data_flow_ops.stack_v2(
          -1, elem_type=dtypes.float32, stack_name="foo")
      c1 = gen_data_flow_ops.stack_close_v2(h)
      self.evaluate(c1)

  @test_util.run_deprecated_v1
  def testCloseStack(self):
    self._testCloseStack(use_gpu=False)
    self._testCloseStack(use_gpu=True)

  def _testPushCloseStack(self, use_gpu):
    with self.cached_session(use_gpu=use_gpu) as sess:
      h = gen_data_flow_ops.stack_v2(
          -1, elem_type=dtypes.float32, stack_name="foo")
      c = gen_data_flow_ops.stack_push_v2(h, [[4.0, 5.0]])
      with ops.control_dependencies([c]):
        c1 = gen_data_flow_ops.stack_close_v2(h)
      self.evaluate(c1)

  @test_util.run_deprecated_v1
  def testPushCloseStack(self):
    self._testPushCloseStack(use_gpu=False)
    self._testPushCloseStack(use_gpu=True)


class StackOpRefTest(test.TestCase):
  """Tests for deprecated non-resource variant of stack ops."""

  def _testStackPushPop(self, use_gpu):
    with self.cached_session(use_gpu=use_gpu):
      h = gen_data_flow_ops._stack(dtypes.float32, stack_name="foo")
      c = gen_data_flow_ops.stack_push(h, [[4.0, 5.0]])
      with ops.control_dependencies([c]):
        c1 = gen_data_flow_ops.stack_pop(h, dtypes.float32)
      self.assertAllClose([[4.0, 5.0]], self.evaluate(c1))

  @test_util.run_deprecated_v1
  def testStackPushPop(self):
    self._testStackPushPop(use_gpu=False)
    self._testStackPushPop(use_gpu=True)

  def _testStackPushPopSwap(self, use_gpu):
    with self.cached_session(use_gpu=use_gpu):
      a = np.arange(2000)
      x = constant_op.constant(a, dtype=dtypes.float32)
      h = gen_data_flow_ops._stack(dtypes.float32, stack_name="foo")
      c = gen_data_flow_ops.stack_push(h, x, swap_memory=True)
      with ops.control_dependencies([c]):
        c1 = gen_data_flow_ops.stack_pop(h, dtypes.float32)
      self.assertAllClose(a, self.evaluate(c1))

  @test_util.run_deprecated_v1
  def testStackPushPopSwap(self):
    self._testStackPushPopSwap(use_gpu=False)
    self._testStackPushPopSwap(use_gpu=True)

  def _testMultiStack(self, use_gpu):
    with self.cached_session(use_gpu=use_gpu):
      h1 = gen_data_flow_ops._stack(dtypes.float32, stack_name="foo")
      c1 = gen_data_flow_ops.stack_push(h1, 4.0)
      with ops.control_dependencies([c1]):
        c1 = gen_data_flow_ops.stack_pop(h1, dtypes.float32)
      h2 = gen_data_flow_ops._stack(dtypes.float32, stack_name="bar")
      c2 = gen_data_flow_ops.stack_push(h2, 5.0)
      with ops.control_dependencies([c2]):
        c2 = gen_data_flow_ops.stack_pop(h2, dtypes.float32)
      r = c1 + c2
      self.assertAllClose(9.0, self.evaluate(r))

  def _testStackWhileSwap(self, use_gpu):
    with self.cached_session(use_gpu=use_gpu):
      n = constant_op.constant(0)
      h = gen_data_flow_ops._stack(dtypes.float32, stack_name="foo")

      def c(x):
        return math_ops.less(x, 10)

      def b(x):
        with ops.control_dependencies([x]):
          a = constant_op.constant(np.ones(2000), dtype=dtypes.float32)
          v = gen_data_flow_ops.stack_push(h, a, swap_memory=True)
        with ops.control_dependencies([v]):
          return math_ops.add(x, 1)

      r = while_loop.while_loop(c, b, [n])

      v = constant_op.constant(np.zeros(2000), dtype=dtypes.float32)

      def c1(x, y):
        del y
        return math_ops.greater(x, 0)

      def b1(x, y):
        nx = math_ops.subtract(x, 1)
        ny = y + gen_data_flow_ops.stack_pop(h, dtypes.float32)
        return [nx, ny]

      _, ry = while_loop.while_loop(
          c1, b1, [r, v],
          [r.get_shape(), tensor_shape.unknown_shape()])
      self.assertAllClose(np.ones(2000) * 10.0, self.evaluate(ry))

  @test_util.run_v1_only("b/120545219")
  def testStackWhileSwap(self):
    self._testStackWhileSwap(use_gpu=False)
    self._testStackWhileSwap(use_gpu=True)

  @test_util.run_deprecated_v1
  def testMultiStack(self):
    self._testMultiStack(use_gpu=False)
    self._testMultiStack(use_gpu=True)

  def _testSameNameStacks(self, use_gpu):
    with self.cached_session(use_gpu=use_gpu):
      h1 = gen_data_flow_ops._stack(dtypes.float32, stack_name="foo")
      c1 = gen_data_flow_ops.stack_push(h1, 4.0)
      h2 = gen_data_flow_ops._stack(dtypes.float32, stack_name="foo")
      c2 = gen_data_flow_ops.stack_push(h2, 5.0)
      _ = c1 + c2
      self.assertNotEqual(self.evaluate(h1)[1], self.evaluate(h2)[1])

  @test_util.run_deprecated_v1
  def testSameNameStacks(self):
    self._testSameNameStacks(use_gpu=False)
    self._testSameNameStacks(use_gpu=True)

  def _testCloseStack(self, use_gpu):
    with self.cached_session(use_gpu=use_gpu) as sess:
      h = gen_data_flow_ops._stack(dtypes.float32, stack_name="foo")
      c1 = gen_data_flow_ops.stack_close(h)
      self.evaluate(c1)

  @test_util.run_deprecated_v1
  def testCloseStack(self):
    self._testCloseStack(use_gpu=False)
    self._testCloseStack(use_gpu=True)

  def _testPushCloseStack(self, use_gpu):
    with self.cached_session(use_gpu=use_gpu) as sess:
      h = gen_data_flow_ops._stack(dtypes.float32, stack_name="foo")
      c = gen_data_flow_ops.stack_push(h, [[4.0, 5.0]])
      with ops.control_dependencies([c]):
        c1 = gen_data_flow_ops.stack_close(h)
      self.evaluate(c1)

  @test_util.run_deprecated_v1
  def testPushCloseStack(self):
    self._testPushCloseStack(use_gpu=False)
    self._testPushCloseStack(use_gpu=True)


if __name__ == "__main__":
  test.main()
