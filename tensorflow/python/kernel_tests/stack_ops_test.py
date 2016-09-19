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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow.python.framework import errors
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import gen_data_flow_ops


class StackOpTest(tf.test.TestCase):

  def _testStackPushPop(self, use_gpu):
    with self.test_session(use_gpu=use_gpu):
      h = gen_data_flow_ops._stack(tf.float32, stack_name="foo")
      c = gen_data_flow_ops._stack_push(h, [[4.0, 5.0]])
      with tf.control_dependencies([c]):
        c1 = gen_data_flow_ops._stack_pop(h, tf.float32)
      self.assertAllClose([[4.0, 5.0]], c1.eval())

  def testStackPushPop(self):
    self._testStackPushPop(use_gpu=False)
    self._testStackPushPop(use_gpu=True)

  def _testStackPushPopSwap(self, use_gpu):
    with self.test_session(use_gpu=use_gpu):
      a = np.arange(2000)
      x = tf.constant(a, dtype=tf.float32)
      h = gen_data_flow_ops._stack(tf.float32, stack_name="foo")
      c = gen_data_flow_ops._stack_push(h, x, swap_memory=True)
      with tf.control_dependencies([c]):
        c1 = gen_data_flow_ops._stack_pop(h, tf.float32)
      self.assertAllClose(a, c1.eval())

  def testStackPushPopSwap(self):
    self._testStackPushPopSwap(use_gpu=False)
    self._testStackPushPopSwap(use_gpu=True)

  def _testStackWhileSwap(self, use_gpu):
    with self.test_session(use_gpu=use_gpu):
      n = tf.constant(0)
      h = gen_data_flow_ops._stack(tf.float32, stack_name="foo")

      def c(x):
        return tf.less(x, 10)
      def b(x):
        with tf.control_dependencies([x]):
          a = tf.constant(np.ones(2000), dtype=tf.float32)
          v = gen_data_flow_ops._stack_push(h, a, swap_memory=True)
        with tf.control_dependencies([v]):
          return tf.add(x, 1)
      r = tf.while_loop(c, b, [n])

      v = tf.constant(np.zeros(2000), dtype=tf.float32)
      def c1(x, y):
        return tf.greater(x, 0)
      def b1(x, y):
        nx = tf.sub(x, 1)
        ny = y + gen_data_flow_ops._stack_pop(h, tf.float32)
        return [nx, ny]
      rx, ry = tf.while_loop(c1, b1, [r, v],
                             [r.get_shape(), tensor_shape.unknown_shape()])
      self.assertAllClose(np.ones(2000) * 10.0, ry.eval())

  def testStackWhileSwap(self):
    self._testStackWhileSwap(use_gpu=False)
    self._testStackWhileSwap(use_gpu=True)

  def _testMultiStack(self, use_gpu):
    with self.test_session(use_gpu=use_gpu):
      h1 = gen_data_flow_ops._stack(tf.float32, stack_name="foo")
      c1 = gen_data_flow_ops._stack_push(h1, 4.0)
      with tf.control_dependencies([c1]):
        c1 = gen_data_flow_ops._stack_pop(h1, tf.float32)
      h2 = gen_data_flow_ops._stack(tf.float32, stack_name="bar")
      c2 = gen_data_flow_ops._stack_push(h2, 5.0)
      with tf.control_dependencies([c2]):
        c2 = gen_data_flow_ops._stack_pop(h2, tf.float32)
      r = c1 + c2
      self.assertAllClose(9.0, r.eval())

  def testMultiStack(self):
    self._testMultiStack(use_gpu=False)
    self._testMultiStack(use_gpu=True)

  def _testSameNameStacks(self, use_gpu):
    with self.test_session(use_gpu=use_gpu):
      h1 = gen_data_flow_ops._stack(tf.float32, stack_name="foo")
      c1 = gen_data_flow_ops._stack_push(h1, 4.0)
      h2 = gen_data_flow_ops._stack(tf.float32, stack_name="foo")
      c2 = gen_data_flow_ops._stack_push(h2, 5.0)
      r = c1 + c2
      self.assertNotEqual(h1.eval()[1], h2.eval()[1])

  def testSameNameStacks(self):
    self._testSameNameStacks(use_gpu=False)
    self._testSameNameStacks(use_gpu=True)

  def _testCloseStack(self, use_gpu):
    with self.test_session(use_gpu=use_gpu) as sess:
      h = gen_data_flow_ops._stack(tf.float32, stack_name="foo")
      c1 = gen_data_flow_ops._stack_close(h)
      sess.run(c1)

  def testCloseStack(self):
    self._testCloseStack(use_gpu=False)
    self._testCloseStack(use_gpu=True)

  def _testPushCloseStack(self, use_gpu):
    with self.test_session(use_gpu=use_gpu) as sess:
      h = gen_data_flow_ops._stack(tf.float32, stack_name="foo")
      c = gen_data_flow_ops._stack_push(h, [[4.0, 5.0]])
      with tf.control_dependencies([c]):
        c1 = gen_data_flow_ops._stack_close(h)
      sess.run(c1)

  def testPushCloseStack(self):
    self._testPushCloseStack(use_gpu=False)
    self._testPushCloseStack(use_gpu=True)

if __name__ == "__main__":
  tf.test.main()
