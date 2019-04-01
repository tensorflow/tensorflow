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

import collections

from tensorflow.python.autograph.converters import control_flow
from tensorflow.python.autograph.core import converter_testing
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test


class ControlFlowTest(converter_testing.TestCase):

  def assertTransformedResult(self, test_fn, inputs, expected, symbols=None):
    if not isinstance(inputs, tuple):
      inputs = (inputs,)
    if not symbols:
      symbols = {}
    with self.converted(test_fn, control_flow, symbols,
                        constant_op.constant) as result:
      self.assertEqual(self.evaluate(result.test_fn(*inputs)), expected)

  @test_util.run_deprecated_v1
  def test_while_basic(self):

    def test_fn(n):
      i = 0
      s = 0
      while i < n:
        s += i
        i += 1
      return s, i, n

    self.assertTransformedResult(test_fn, constant_op.constant(5), (10, 5, 5))

  @test_util.run_deprecated_v1
  def test_while_nested(self):

    def test_fn(n):
      i = 0
      j = 0
      s = 0
      while i < n:
        while j < i:
          j += 3
        u = i + j  # 'u' is not defined within the inner loop
        s += u
        i += 1
        j = 0
      return s, i, j, n

    self.assertTransformedResult(test_fn, constant_op.constant(5),
                                 (25, 5, 0, 5))

  @test_util.run_deprecated_v1
  def test_while_single_output(self):

    def test_fn(n):
      while n > 0:
        n -= 1
      return n

    self.assertTransformedResult(test_fn, constant_op.constant(5), 0)

  def test_while_local_composite(self):

    class TestClass(object):

      def __init__(self):
        self.x = constant_op.constant(3)

    def test_fn(n):
      while n > 0:
        tc = TestClass()
        tc.x = tc.x
        n -= 1
      return n

    self.assertTransformedResult(
        test_fn, constant_op.constant(5), 0, symbols={'TestClass': TestClass})

  # TODO(b/127642077): Add tests for x.y.z = 2*x.y.z and x.y[z] = 2*x.y[z].
  def test_while_local_composite_complex_nestable(self):

    # This class is ok to be in a tf.while_loop's state.
    class TestClass(collections.namedtuple('TestClass', ('x'))):
      pass

    def test_fn(n):
      tc = TestClass([constant_op.constant(0)])
      while n > 0:
        tc = TestClass([constant_op.constant(3)])
        tc.x[0] = tc.x[0] + 1
        n -= 1
      return tc.x[0]

    ns = {'TestClass': TestClass, 'constant_op': constant_op}
    self.assertTransformedResult(
        test_fn, constant_op.constant(5), 4, symbols=ns)

  def test_while_local_composite_complex_illegal(self):

    class TestClass(object):

      def __init__(self):
        self.x = [constant_op.constant(3)]

    def test_fn(n):
      while n > 0:
        tc = TestClass()
        tc.x[0] = tc.x[0] + 1
        n -= 1
      return tc.x[0]

    with self.converted(
        test_fn, control_flow, {'TestClass': TestClass}) as result:
      # The tested function would require `tc` to become part of the while loop
      # state, but TensorFlow doesn't support classes at the moment.
      with self.assertRaisesRegexp(ValueError, 'must.*initialize.*Tensor.*tc'):
        result.test_fn(constant_op.constant(5))

  @test_util.run_deprecated_v1
  def test_while_dispatches_by_cond_only(self):

    class TensorIncompatibleNumeric(object):
      """Works in arithmetic expression, but errors out with TF ops."""

      def __init__(self, val):
        self.val = val

      def __add__(self, other):
        return TensorIncompatibleNumeric(self.val + other)

    def test_fn(n, s):
      while n > 0:
        n -= 1
        s += n
      return s

    self.assertTransformedResult(test_fn, (constant_op.constant(5), 0), 10)
    with self.converted(test_fn, control_flow, {}) as result:
      # n alone controls the staging. When the loop is not staged, Python
      # knows how to add the two objects. But when staged, tf.while_loop will
      # not know how to deal with the TensorIncompatibleNumeric object.
      self.assertEqual(result.test_fn(5, TensorIncompatibleNumeric(0)).val, 10)
      with self.assertRaises(TypeError):
        result.test_fn(constant_op.constant(5), TensorIncompatibleNumeric(0))

  @test_util.run_deprecated_v1
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

  @test_util.run_deprecated_v1
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
      res_obj = result.test_fn(constant_op.constant(1), TestClass(0, 0))
      self.assertEqual(self.evaluate((res_obj.a, res_obj.b)), (-1, 0))
      res_obj = result.test_fn(constant_op.constant(-1), TestClass(0, 0))
      self.assertEqual(self.evaluate((res_obj.a, res_obj.b)), (0, -2))

  @test_util.run_deprecated_v1
  def test_if_single_output(self):

    def test_fn(n):
      if n > 0:
        n = -n
      return n

    self.assertTransformedResult(test_fn, constant_op.constant(1), -1)

  @test_util.run_deprecated_v1
  def test_if_semi(self):

    def test_fn(n):
      if n > 0:
        n = 3
      return n

    self.assertTransformedResult(test_fn, constant_op.constant(2), 3)
    self.assertTransformedResult(test_fn, constant_op.constant(-3), -3)

  @test_util.run_deprecated_v1
  def test_if_local_var(self):

    def test_fn(n):
      if n > 0:
        b = 4
        n = b + 1
      return n

    self.assertTransformedResult(test_fn, constant_op.constant(1), 5)
    self.assertTransformedResult(test_fn, constant_op.constant(-1), -1)

  @test_util.run_deprecated_v1
  def test_if_no_outputs(self):

    def test_fn(n):
      if n > 0:
        b = 4  # pylint:disable=unused-variable
      return n

    # Without side effect guards, the if statement will stage a cond,
    # but that will be pruned at execution.
    self.assertTransformedResult(test_fn, constant_op.constant(1), 1)
    self.assertTransformedResult(test_fn, constant_op.constant(-1), -1)

  @test_util.run_deprecated_v1
  def test_if_unbalanced_multiple_composites(self):

    class Foo(object):

      def __init__(self):
        self.b = 2
        self.c = 3

    def test_fn(x, condition):

      z = 5
      if condition:
        x.b = 7
        x.c = 11
        z = 13

      return x.b, x.c, z

    self.assertTransformedResult(test_fn, (Foo(), constant_op.constant(True)),
                                 (7, 11, 13))
    self.assertTransformedResult(test_fn, (Foo(), constant_op.constant(False)),
                                 (2, 3, 5))

  @test_util.run_deprecated_v1
  def test_if_unbalanced_composite(self):

    class Foo(object):

      def __init__(self):
        self.b = 2

    def test_fn(x, condition):

      z = 5
      if condition:
        x.b = 7
        z = 13

      return x.b, z

    self.assertTransformedResult(test_fn, (Foo(), constant_op.constant(True)),
                                 (7, 13))
    self.assertTransformedResult(test_fn, (Foo(), constant_op.constant(False)),
                                 (2, 5))

  @test_util.run_deprecated_v1
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

  @test_util.run_deprecated_v1
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

  @test_util.run_deprecated_v1
  def test_for_tuple_unpacking(self):
    def test_fn(x_list):
      z = tf.constant(0)  # pylint:disable=undefined-variable
      for i, x in enumerate(x_list):
        z = z + x + i
      return z

    self.assertTransformedResult(test_fn, [3, 3], 7)


if __name__ == '__main__':
  test.main()
