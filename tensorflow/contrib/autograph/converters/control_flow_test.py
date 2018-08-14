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
from tensorflow.contrib.autograph.core import converter_testing
from tensorflow.contrib.autograph.pyct import transformer
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.platform import test


class ControlFlowTest(converter_testing.TestCase):

  def assertTransformedResult(self, test_fn, inputs, expected):
    if not isinstance(inputs, tuple):
      inputs = (inputs,)
    with self.converted(test_fn, control_flow, {},
                        constant_op.constant) as result:
      with self.test_session() as sess:
        self.assertEqual(sess.run(result.test_fn(*inputs)), expected)

  def test_while_basic(self):

    def test_fn(n):
      i = 0
      s = 0
      while i < n:
        s += i
        i += 1
      return s, i, n

    self.assertTransformedResult(test_fn, constant_op.constant(5), (10, 5, 5))

  def test_while_single_output(self):

    def test_fn(n):
      while n > 0:
        n -= 1
      return n

    self.assertTransformedResult(test_fn, constant_op.constant(5), 0)

  def test_if_basic(self):

    def test_fn(n):
      a = 0
      b = 0
      if n > 0:
        a = -n
      else:
        b = 2 * n
      return a, b

    self.assertTransformedResult(test_fn, constant_op.constant(1), (-1, 0))
    self.assertTransformedResult(test_fn, constant_op.constant(-1), (0, -2))

  def test_if_complex_outputs(self):

    class TestClass(object):

      def __init__(self, a, b):
        self.a = a
        self.b = b

    def test_fn(n, obj):
      obj.a = 0
      obj.b = 0
      if n > 0:
        obj.a = -n
      else:
        obj.b = 2 * n
      return obj

    with self.converted(test_fn, control_flow, {}) as result:
      with self.test_session() as sess:
        res_obj = result.test_fn(constant_op.constant(1), TestClass(0, 0))
        self.assertEqual(sess.run((res_obj.a, res_obj.b)), (-1, 0))
        res_obj = result.test_fn(constant_op.constant(-1), TestClass(0, 0))
        self.assertEqual(sess.run((res_obj.a, res_obj.b)), (0, -2))

  def test_if_single_output(self):

    def test_fn(n):
      if n > 0:
        n = -n
      return n

    self.assertTransformedResult(test_fn, constant_op.constant(1), -1)

  def test_if_semi(self):

    def test_fn(n):
      if n > 0:
        n = 3
      return n

    self.assertTransformedResult(test_fn, constant_op.constant(2), 3)
    self.assertTransformedResult(test_fn, constant_op.constant(-3), -3)

  def test_if_local_var(self):

    def test_fn(n):
      if n > 0:
        b = 4
        n = b + 1
      return n

    self.assertTransformedResult(test_fn, constant_op.constant(1), 5)
    self.assertTransformedResult(test_fn, constant_op.constant(-1), -1)

  def test_if_no_outputs(self):

    def test_fn(n):
      if n > 0:
        b = 4  # pylint:disable=unused-variable
      return n

    # Without side effect guards, the if statement will stage a cond,
    # but that will be pruned at execution.
    self.assertTransformedResult(test_fn, constant_op.constant(1), 1)
    self.assertTransformedResult(test_fn, constant_op.constant(-1), -1)

  def test_if_imbalanced_outputs(self):

    def test_fn(n):
      if n > 0:
        b = 4
      return b

    node, ctx = self.prepare(test_fn, {})
    with self.assertRaises(transformer.AutographParseError):
      control_flow.transform(node, ctx)

  def test_simple_for(self):

    def test_fn(l):
      s1 = 0
      s2 = 0
      for e in l:
        s1 += e
        s2 += e * e
      return s1, s2

    self.assertTransformedResult(test_fn, constant_op.constant([1, 3]), (4, 10))
    empty_vector = constant_op.constant([], shape=(0,), dtype=dtypes.int32)
    self.assertTransformedResult(test_fn, empty_vector, (0, 0))

  def test_for_single_output(self):

    def test_fn(l):
      s = 0
      for e in l:
        s += e
      return s

    self.assertTransformedResult(test_fn, constant_op.constant([1, 3]), 4)
    empty_vector = constant_op.constant([], shape=(0,), dtype=dtypes.int32)
    self.assertTransformedResult(test_fn, empty_vector, 0)

  def test_for_iterated_expression(self):

    eval_count = [0]

    def count_evals(x):
      eval_count[0] += 1
      return x

    def test_fn(n):
      s = 0
      for e in count_evals(range(n)):
        s += e
      return s

    ns = {'count_evals': count_evals}
    node, ctx = self.prepare(test_fn, ns)
    node = control_flow.transform(node, ctx)

    with self.compiled(node, ns) as result:
      self.assertEqual(result.test_fn(5), 10)
      self.assertEqual(eval_count[0], 1)


if __name__ == '__main__':
  test.main()
