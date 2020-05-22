# Lint as: python3
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

import numpy as np

from tensorflow.python.autograph.converters import control_flow
from tensorflow.python.autograph.core import converter_testing
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_util
from tensorflow.python.platform import test
from tensorflow.python.util import nest

# TODO(mdan): These tests are not isolated - they also test the operators.


class ControlFlowTestBase(converter_testing.TestCase):

  def assertValuesEqual(self, actual, expected):
    values = nest.map_structure(
        lambda x: self.evaluate(x) if tensor_util.is_tensor(x) else x,
        actual)
    self.assertAllEqual(values, expected)

  def assertTransformedResult(self, test_fn, inputs, expected, symbols=None):
    if not isinstance(inputs, tuple):
      inputs = (inputs,)
    if not symbols:
      symbols = {}
    with self.converted(test_fn, control_flow, symbols,
                        (constant_op.constant,)) as result:
      returns = result.test_fn(*inputs)
      self.assertValuesEqual(returns, expected)


class NestedControlFlowTest(ControlFlowTestBase):

  def test_basic(self):

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

  def test_composite_state_complex(self):

    class TestClassX(object):

      def __init__(self, x):
        self.x = x

    class TestClassY(object):

      def __init__(self, y):
        self.y = y

    def test_fn(n):
      tc = TestClassX(TestClassY({'z': TestClassX(n)}))
      if n > 0:
        while n > 0:
          if n < 2:
            tc.x.y['z'].x += 1
          n -= 1
      return n, tc

    with self.converted(test_fn, control_flow, {
        'TestClassX': TestClassX,
        'TestClassY': TestClassY,
    }) as result:
      n, tc = result.test_fn(constant_op.constant(5))
      self.assertValuesEqual((n, tc.x.y['z'].x), (0, 6))


class WhileStatementTest(ControlFlowTestBase):

  def test_basic(self):

    def test_fn(n):
      i = 0
      s = 0
      while i < n:
        s += i
        i += 1
      return s, i, n

    self.assertTransformedResult(test_fn, constant_op.constant(5), (10, 5, 5))

  def test_single_output(self):

    def test_fn(n):
      while n > 0:
        n -= 1
      return n

    self.assertTransformedResult(test_fn, constant_op.constant(5), 0)

  def test_composite_state_attr(self):

    class TestClass(object):

      def __init__(self):
        self.x = constant_op.constant(3)

    def test_fn(n):
      tc = TestClass()
      while n > 0:
        tc.x += 1
        n -= 1
      return n

    self.assertTransformedResult(
        test_fn, constant_op.constant(5), 0, symbols={'TestClass': TestClass})

  def test_composite_state_slice(self):

    def test_fn(n):
      d = {'a': n}
      k = 'a'
      while n > 0:
        d[k] += 1
        n -= 1
      return d[k], n

    self.assertTransformedResult(test_fn, constant_op.constant(5), (10, 0))

  def test_composite_state_literal_slice(self):

    def test_fn(n):
      d = {'a': n}
      while n > 0:
        d['a'] += 1
        n -= 1
      return d['a'], n

    self.assertTransformedResult(test_fn, constant_op.constant(5), (10, 0))

  def test_composite_state_attr_initialized_in_loop(self):

    class TestClass(object):
      pass

    def test_fn(n, x):
      tc = TestClass()
      while n < 5:
        if n == 0:
          tc.subattr = x
        else:
          tc.subattr = tc.subattr + 1
        n += 1
      return tc.subattr

    self.assertTransformedResult(
        test_fn, (0, constant_op.constant(10)),
        14,
        symbols={'TestClass': TestClass})
    with self.converted(
        test_fn, control_flow, {'TestClass': TestClass}) as result:
      # TODO(b/128519776): Better error message.
      with self.assertRaisesRegex(AttributeError, 'subattr'):
        result.test_fn(constant_op.constant(0), constant_op.constant(5))

  def test_composite_state_slice_initialized_in_loop(self):

    def test_fn(n, x):
      d = {}
      k = 'subkey'
      while n < 5:
        if n == 0:
          d[k] = x
        else:
          d[k] = d[k] + 1
        n += 1
      return d

    self.assertTransformedResult(test_fn, (0, constant_op.constant(10)),
                                 {'subkey': 14})
    with self.converted(test_fn, control_flow, {}) as result:
      # TODO(b/128519776): Better error message.
      with self.assertRaisesRegex(KeyError, 'subkey'):
        result.test_fn(constant_op.constant(0), constant_op.constant(5))

  def test_composite_state_literal_slice_initialized_in_loop(self):

    def test_fn(n, x):
      d = {}
      while n < 5:
        if n == 0:
          d['subkey'] = x
        else:
          d['subkey'] = d['subkey'] + 1
        n += 1
      return d

    self.assertTransformedResult(test_fn, (0, constant_op.constant(10)),
                                 {'subkey': 14})
    with self.converted(test_fn, control_flow, {}) as result:
      # TODO(b/128519776): Better error message.
      with self.assertRaisesRegex(KeyError, 'subkey'):
        result.test_fn(constant_op.constant(0), constant_op.constant(5))

  def test_composite_state_slice_aliased_to_local(self):

    def test_fn(n, x):
      d = {}
      while n < 5:
        k = 'subkey'
        d[k] = x + 1
        n += 1
      return d

    self.assertTransformedResult(test_fn, (0, constant_op.constant(10)),
                                 {'subkey': 11})
    with self.converted(test_fn, control_flow, {}) as result:
      # TODO(b/128519776): Better error message.
      # Note that this error happens at execution time.
      with self.assertRaises(errors.InaccessibleTensorError):
        graph_fn = def_function.function(result.test_fn, autograph=False)
        self.evaluate(
            graph_fn(constant_op.constant(0), constant_op.constant(5)))

  def test_local_composite_attr(self):

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

  def test_local_composite_slice(self):

    def test_fn(n):
      while n > 0:
        d = {'x': n}
        k = 'x'
        d[k] = d[k]
        n -= 1
      return n

    self.assertTransformedResult(test_fn, constant_op.constant(5), 0, {})

  def test_local_composite_literal_slice(self):

    def test_fn(n):
      while n > 0:
        d = {'x': n}
        d['x'] = d['x']
        n -= 1
      return n

    self.assertTransformedResult(test_fn, constant_op.constant(5), 0, {})

  def test_non_tensor_state(self):

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

  def test_non_tensor_state_illegal_type(self):

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
      with self.assertRaisesRegexp(
          ValueError, 'tc.*must be defined before the loop'):
        result.test_fn(constant_op.constant(5))

  def test_dispatches_by_cond_only(self):

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


class IfStatementTest(ControlFlowTestBase):

  def test_basic(self):

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

  def test_sparse_tensor(self):

    def test_fn(cond, a):
      if cond:
        a = -a
      return a

    st = sparse_tensor.SparseTensor(
        indices=((0,),), values=(0,), dense_shape=(1,))
    self.assertTransformedResult(test_fn, (st, constant_op.constant(1)), -1)
    self.assertTransformedResult(test_fn, (None, constant_op.constant(1)), 1)

  def test_complex_outputs(self):

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
      self.assertValuesEqual((res_obj.a, res_obj.b), (-1, 0))
      res_obj = result.test_fn(constant_op.constant(-1), TestClass(0, 0))
      self.assertValuesEqual((res_obj.a, res_obj.b), (0, -2))

  def test_single_output(self):

    def test_fn(n):
      if n > 0:
        n = -n
      return n

    self.assertTransformedResult(test_fn, constant_op.constant(1), -1)

  def test_unbalanced(self):

    def test_fn(n):
      if n > 0:
        n = 3
      return n

    self.assertTransformedResult(test_fn, constant_op.constant(2), 3)
    self.assertTransformedResult(test_fn, constant_op.constant(-3), -3)

  def test_unbalanced_raising(self):

    def test_fn(n):
      if n > 0:
        n = n + 1
        raise ValueError()
      return n

    self.assertTransformedResult(test_fn, -3, -3)

    with self.converted(test_fn, control_flow, {}) as result:
      with self.assertRaises(ValueError):
        result.test_fn(1)

  def test_local_var(self):

    def test_fn(n):
      if n > 0:
        b = 4
        n = b + 1
      return n

    self.assertTransformedResult(test_fn, constant_op.constant(1), 5)
    self.assertTransformedResult(test_fn, constant_op.constant(-1), -1)

  def test_local_remains_local(self):

    def test_fn(n):
      if n > 0:
        b = 4
        n = b + 1
      return n

    self.assertTransformedResult(test_fn, constant_op.constant(1), 5)
    self.assertTransformedResult(test_fn, constant_op.constant(-1), -1)

  def test_no_outputs(self):

    def test_fn(n):
      if n > 0:
        b = 4  # pylint:disable=unused-variable
      return n

    # Without side effect guards, the if statement will stage a cond,
    # but that will be pruned at execution.
    self.assertTransformedResult(test_fn, constant_op.constant(1), 1)
    self.assertTransformedResult(test_fn, constant_op.constant(-1), -1)

  def test_created_outputs(self):

    def test_fn(i):
      if i == 0:
        result = i - 1
      else:
        result = i + 1
      return result

    self.assertTransformedResult(test_fn, 0, -1)
    self.assertTransformedResult(test_fn, 1, 2)

  def test_created_loop_local_outputs(self):

    def test_fn(n, x):
      for i in n:
        if i == 0:
          result = i - 1
        else:
          result = i + 1
        if result > 0:
          x += 1
      return x

    self.assertTransformedResult(test_fn, (range(5), 10), 14)

  def test_created_loop_variable(self):

    def test_fn(n, x):
      for i in n:
        if i == 0:
          result = i - 1
        if i > 0:  # Using the result from previous iteration.
          if result < 0:
            x += 1
      return x

    self.assertTransformedResult(test_fn, (range(5), 10), 14)

  def test_unaffected_global(self):

    def test_fn(i):
      global g  # pylint:disable=global-variable-undefined
      if i == 0:
        g = i - 1
      return g

    self.assertTransformedResult(test_fn, 1, 3, symbols={'g': 3})
    self.assertTransformedResult(test_fn, 0, -1, symbols={'g': 3})

  def test_unaffected_nonlocal(self):

    def test_fn(i):
      def inner_fn():
        nonlocal n
        if i == 0:
          n = i - 1

      n = 3
      inner_fn()
      return n

    self.assertTransformedResult(test_fn, 1, 3)
    self.assertTransformedResult(test_fn, 0, -1)

  def test_output_defined_in_prior_except(self):

    def test_fn(i):
      try:
        raise ValueError()
      except ValueError:
        x = 1
      if i == 0:
        x = i - 1
      return x

    self.assertTransformedResult(test_fn, 1, 1)
    self.assertTransformedResult(test_fn, 0, -1)

  def test_unbalanced_multiple_composites(self):

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

  def test_unbalanced_composite(self):

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


class ForStatementTest(ControlFlowTestBase):

  def test_basic(self):

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

  def test_single_output(self):

    def test_fn(l):
      s = 0
      for e in l:
        s += e
      return s

    self.assertTransformedResult(test_fn, constant_op.constant([1, 3]), 4)
    empty_vector = constant_op.constant([], shape=(0,), dtype=dtypes.int32)
    self.assertTransformedResult(test_fn, empty_vector, 0)

  def test_iterated_expression(self):

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

  def test_composite_state_initialized_in_loop(self):

    class TestClass(object):
      pass

    def test_fn(n, x):
      tc = TestClass()
      for i in n:
        if i == 0:
          tc.x = x
        else:
          tc.x = tc.x + i
      return tc.x

    self.assertTransformedResult(
        test_fn, (range(5), constant_op.constant(10)),
        20,
        symbols={'TestClass': TestClass})
    with self.converted(
        test_fn, control_flow, {'TestClass': TestClass}) as result:
      # TODO(b/128519776): Better error message.
      with self.assertRaisesRegex(
          AttributeError, '\'TestClass\' object has no attribute \'x\''):
        result.test_fn(
            constant_op.constant(list(range(5))), constant_op.constant(5))

  def test_tuple_unpacking(self):
    def test_fn(x_list):
      z = tf.constant(0)  # pylint:disable=undefined-variable
      for i, x in enumerate(x_list):
        z = z + x + i
      return z

    self.assertTransformedResult(test_fn, [3, 3], 7)

  def test_with_comprehension_in_body(self):

    def test_fn(l, n):
      s = constant_op.constant(list(range(n)))
      for _ in l:
        s += constant_op.constant([a for a in range(n)])
      return s

    self.assertTransformedResult(
        test_fn, (constant_op.constant([1, 2, 3]), 5),
        np.array(range(5)) * 4,
        symbols={'constant_op': constant_op})


if __name__ == '__main__':
  test.main()
