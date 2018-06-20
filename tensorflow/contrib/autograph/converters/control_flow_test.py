# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for control_flow module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.autograph.converters import control_flow
from tensorflow.contrib.autograph.converters import converter_test_base
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.platform import test


class ControlFlowTest(converter_test_base.TestCase):

  def test_simple_while(self):

    def test_fn(n):
      i = 0
      s = 0
      while i < n:
        s += i
        i += 1
      return s, i, n

    node = self.parse_and_analyze(test_fn, {})
    node = control_flow.transform(node, self.ctx)

    with self.compiled(node) as result:
      with self.test_session() as sess:
        self.assertEqual((10, 5, 5),
                         sess.run(result.test_fn(constant_op.constant(5))))

  def test_while_single_var(self):

    def test_fn(n):
      while n > 0:
        n -= 1
      return n

    node = self.parse_and_analyze(test_fn, {})
    node = control_flow.transform(node, self.ctx)

    with self.compiled(node) as result:
      with self.test_session() as sess:
        self.assertEqual(0, sess.run(result.test_fn(constant_op.constant(5))))

  def test_simple_if(self):

    def test_fn(n):
      a = 0
      b = 0
      if n > 0:
        a = -n
      else:
        b = 2 * n
      return a, b

    node = self.parse_and_analyze(test_fn, {})
    node = control_flow.transform(node, self.ctx)

    with self.compiled(node) as result:
      with self.test_session() as sess:
        self.assertEqual((-1, 0),
                         sess.run(result.test_fn(constant_op.constant(1))))
        self.assertEqual((0, -2),
                         sess.run(result.test_fn(constant_op.constant(-1))))

  def test_if_single_var(self):

    def test_fn(n):
      if n > 0:
        n = -n
      return n

    node = self.parse_and_analyze(test_fn, {})
    node = control_flow.transform(node, self.ctx)

    with self.compiled(node) as result:
      with self.test_session() as sess:
        self.assertEqual(-1, sess.run(result.test_fn(constant_op.constant(1))))

  def test_imbalanced_aliasing(self):

    def test_fn(n):
      if n > 0:
        n = 3
      return n

    node = self.parse_and_analyze(test_fn, {})
    node = control_flow.transform(node, self.ctx)

    with self.compiled(node, control_flow_ops.cond) as result:
      with self.test_session() as sess:
        self.assertEqual(3, sess.run(result.test_fn(constant_op.constant(2))))
        self.assertEqual(-3, sess.run(result.test_fn(constant_op.constant(-3))))

  def test_ignore_unread_variable(self):

    def test_fn(n):
      b = 3  # pylint: disable=unused-variable
      if n > 0:
        b = 4
      return n

    node = self.parse_and_analyze(test_fn, {})
    node = control_flow.transform(node, self.ctx)

    with self.compiled(node, control_flow_ops.cond, array_ops.ones) as result:
      with self.test_session() as sess:
        self.assertEqual(3, sess.run(result.test_fn(constant_op.constant(3))))
        self.assertEqual(-3, sess.run(result.test_fn(constant_op.constant(-3))))

  def test_handle_temp_variable(self):

    def test_fn_using_temp(x, y, w):
      if x < y:
        z = x + y
      else:
        w = 2
        tmp = w
        z = x - tmp
      return z, w

    node = self.parse_and_analyze(test_fn_using_temp, {})
    node = control_flow.transform(node, self.ctx)

    with self.compiled(node, control_flow_ops.cond, array_ops.ones) as result:
      with self.test_session() as sess:
        z, w = sess.run(
            result.test_fn_using_temp(
                constant_op.constant(-3), constant_op.constant(3),
                constant_op.constant(3)))
        self.assertEqual(0, z)
        self.assertEqual(3, w)
        z, w = sess.run(
            result.test_fn_using_temp(
                constant_op.constant(3), constant_op.constant(-3),
                constant_op.constant(3)))
        self.assertEqual(1, z)
        self.assertEqual(2, w)

    def test_fn_ignoring_temp(x, y, w):
      if x < y:
        z = x + y
      else:
        w = 2
        tmp = w
        z = x - tmp
      return z

    node = self.parse_and_analyze(test_fn_ignoring_temp, {})
    node = control_flow.transform(node, self.ctx)

    with self.compiled(node, control_flow_ops.cond, array_ops.ones) as result:
      with self.test_session() as sess:
        z = sess.run(
            result.test_fn_ignoring_temp(
                constant_op.constant(-3), constant_op.constant(3),
                constant_op.constant(3)))
        self.assertEqual(0, z)
        z = sess.run(
            result.test_fn_ignoring_temp(
                constant_op.constant(3), constant_op.constant(-3),
                constant_op.constant(3)))
        self.assertEqual(1, z)

  def test_simple_for(self):

    def test_fn(l):
      s1 = 0
      s2 = 0
      for e in l:
        s1 += e
        s2 += e * e
      return s1, s2

    node = self.parse_and_analyze(test_fn, {})
    node = control_flow.transform(node, self.ctx)

    with self.compiled(node) as result:
      with self.test_session() as sess:
        l = [1, 2, 3]
        self.assertEqual(
            test_fn(l), sess.run(result.test_fn(constant_op.constant(l))))
        l = []
        self.assertEqual(
            test_fn(l),
            sess.run(
                result.test_fn(
                    constant_op.constant(l, shape=(0,), dtype=dtypes.int32))))

  def test_for_single_var(self):

    def test_fn(l):
      s = 0
      for e in l:
        s += e
      return s

    node = self.parse_and_analyze(test_fn, {})
    node = control_flow.transform(node, self.ctx)

    with self.compiled(node) as result:
      with self.test_session() as sess:
        l = [1, 2, 3]
        self.assertEqual(
            test_fn(l), sess.run(result.test_fn(constant_op.constant(l))))
        l = []
        self.assertEqual(
            test_fn(l),
            sess.run(
                result.test_fn(
                    constant_op.constant(l, shape=(0,), dtype=dtypes.int32))))

  def test_for_with_iterated_expression(self):

    eval_count = [0]

    def count_evals(x):
      eval_count[0] += 1
      return x

    def test_fn(n):
      s = 0
      for e in count_evals(range(n)):
        s += e
      return s

    node = self.parse_and_analyze(test_fn, {'count_evals': count_evals})
    node = control_flow.transform(node, self.ctx)

    with self.compiled(node) as result:
      result.count_evals = count_evals
      self.assertEqual(test_fn(5), result.test_fn(5))
      # count_evals ran twice, once for test_fn and another for result.test_fn
      self.assertEqual(eval_count[0], 2)


if __name__ == '__main__':
  test.main()
