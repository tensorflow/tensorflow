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
"""Nested loops and conditional statements (e.g. while, for, if).

Meant to verify that arbitrarily nested statements are processed correctly.
"""

import itertools

from absl.testing import parameterized
import tensorflow as tf

from tensorflow.python.autograph.tests import reference_test_base


def independent_ifs(x, y):
  z = 0
  if x > 0:
    if y > 0:
      z = x + y
  return z


def dependent_inner_if(x):
  y = 0
  if x > 0:
    y = -2 * x
    if y > 0:
      x = -3 * x
  else:
    y = 4 * x
  return x, y


def dependent_imbalanced_inner_if(x):
  y = 0
  if x > 0:
    if x < 3:
      y = -2 * x
      x = -3 * x
  return x, y


def _hidden_raise():
  raise ValueError('exception used for control flow')


def if_with_local_modification_masked_by_exception(x):
  y = 0
  if x > 0:
    try:
      if x > 1:
        _hidden_raise()
      y = 1
    except ValueError:
      pass
    if y == 0:
      y = 2
  return y


_test_global = None


def if_nested_with_modification_of_global(x):
  y = 0
  if x > 0:
    if x > 0:
      global _test_global
      if _test_global is None:
        _test_global = 1
      else:
        _test_global += 1
      y += _test_global
  return y


def independent_inner_for(a, b):
  p = 0
  for _ in a:
    tmp = b
    for j in tmp:
      p += j
  return p


def independent_inner_while(a, b):
  p = 0
  while a > 0:
    tmp = b
    while tmp > 0:
      p += 1
      tmp -= 1
    a -= 1
  return p


def dependent_inner_for(a, b):
  r = 1
  s = 0
  for _ in a:
    r += s
    tmp = b
    for j in tmp:
      s += j
  return r


def dependent_inner_while(a, b):
  r = 1
  while a > 0:
    r += 1
    tmp = b
    while tmp > 0:
      a -= 1
      tmp -= 1
    a -= 1
  return r


def if_in_for(a):
  k = 0
  for i in a:
    if i % 2 > 0:
      j = i // 2
      k += j
  return k


def while_with_continue_in_context_manager(x):
  z = 0
  while x > 0:
    with tf.name_scope(''):
      x = x - 1
      if x < 5:
        continue
      z = z + 1
  return z


def while_continue_in_try(x):
  z = 0
  while x > 0:
    x = x - 1
    try:
      if x < 5:
        continue
      z = z + 1
    finally:
      z = z + 10
  return z


def while_break_in_context_manager(x):
  z = 0
  while x > 0:
    with tf.name_scope(''):
      x = x - 1
      if x < 5:
        break
      z = z + 1
  return z


def while_break_in_try(x):
  z = 0
  while x > 0:
    x = x - 1
    try:
      if x < 5:
        break
      z = z + 1
    finally:
      z = z + 10
  return z


def loop_initializing_invariant_variable(n):
  for i in range(n):
    if i == 0:
      a = 1
    else:
      a = 2
  return a


def loop_initializing_variant_variable(n):
  for i in range(n):
    if i == 0:
      a = 1
    else:
      a = a + 1
  return a


def _int_tensor(x):
  return tf.constant(x, dtype=tf.int32)


class NestedControlFlowTest(
    reference_test_base.TestCase, parameterized.TestCase):

  @parameterized.parameters(*itertools.product(
      (
          -1, 1,
      ),
      (
          -1, 1,
      ),
      (
          int,
          _int_tensor,
      ),
      (
          int,
          _int_tensor,
      ),
  ))
  def test_independent_ifs(self, x, y, type_x, type_y):
    x = type_x(x)
    y = type_x(y)
    self.assertFunctionMatchesEager(independent_ifs, x, y)

  @parameterized.parameters(*itertools.product(
      (
          -1, 1,
      ),
      (
          int,
          _int_tensor,
      ),
  ))
  def test_dependent_inner_if(self, x, type_):
    x = type_(x)
    self.assertFunctionMatchesEager(dependent_inner_if, x)

  @parameterized.parameters(*itertools.product(
      (
          -1, 1,
      ),
      (
          int,
          _int_tensor,
      ),
  ))
  def test_dependent_imbalanced_inner_if(self, x, type_):
    x = type_(x)
    self.assertFunctionMatchesEager(dependent_imbalanced_inner_if, x)

  @parameterized.parameters(
      (-1,),
      (0,),
      (1,),
      (2,),
  )
  def test_if_with_local_modification_masked_by_exception(self, x):
    # Note: If the input is a Tensor, the behavior is undefined.
    self.assertFunctionMatchesEager(
        if_with_local_modification_masked_by_exception, x)

  def test_if_nested_with_modification_of_global(self):
    global _test_global
    _test_global = None
    self.assertEqual(tf.function(if_nested_with_modification_of_global)(1), 1)
    self.assertEqual(_test_global, 1)

  def test_if_nested_with_modification_of_global_not_executed(self):
    global _test_global
    _test_global = None
    self.assertEqual(tf.function(if_nested_with_modification_of_global)(0), 0)
    self.assertIsNone(_test_global)

  @parameterized.parameters(*itertools.product(
      (
          0, 1, 2,
      ),
      (
          0, 1, 2,
      ),
      (
          range,
          tf.range,
      ),
      (
          range,
          tf.range,
      ),
  ))
  def test_independent_inner_for(self, a, b, type_a, type_b):
    a = type_a(a)
    b = type_b(b)
    self.assertFunctionMatchesEager(independent_inner_for, a, b)

  @parameterized.parameters(*itertools.product(
      (
          0, 1, 2,
      ),
      (
          0, 1, 2,
      ),
      (
          int,
          _int_tensor,
      ),
      (
          int,
          _int_tensor,
      ),
  ))
  def test_independent_inner_while(self, a, b, type_a, type_b):
    a = type_a(a)
    b = type_b(b)
    self.assertFunctionMatchesEager(independent_inner_while, a, b)

  @parameterized.parameters(*itertools.product(
      (
          0, 1, 2,
      ),
      (
          0, 1, 2,
      ),
      (
          range,
          tf.range,
      ),
      (
          range,
          tf.range,
      ),
  ))
  def test_dependent_inner_for(self, a, b, type_a, type_b):
    a = type_a(a)
    b = type_b(b)
    self.assertFunctionMatchesEager(dependent_inner_for, a, b)

  @parameterized.parameters(*itertools.product(
      (
          0, 1, 2, 3, 4,
      ),
      (
          0, 1, 2, 3, 4,
      ),
      (
          int,
          _int_tensor,
      ),
      (
          int,
          _int_tensor,
      ),
  ))
  def test_dependent_inner_while(self, a, b, type_a, type_b):
    if (type_a is int) and (type_b is _int_tensor):
      self.skipTest('b/124378596')
    a = type_a(a)
    b = type_b(b)
    self.assertFunctionMatchesEager(dependent_inner_while, a, b)

  @parameterized.parameters(*itertools.product(
      (
          0, 1, 2,
      ),
      (
          range,
          tf.range,
      ),
  ))
  def test_if_in_for(self, a, type_):
    a = type_(a)
    self.assertFunctionMatchesEager(if_in_for, a)

  @parameterized.parameters(*itertools.product(
      (
          0, 4, 10,
      ),
      (
          int,
          _int_tensor,
      ),
  ))
  def test_while_continue_in_context_manager(self, x, type_):
    x = type_(x)
    self.assertFunctionMatchesEager(while_with_continue_in_context_manager, x)

  @parameterized.parameters(*itertools.product(
      (
          0, 4, 10,
      ),
      (
          int,
          _int_tensor,
      ),
  ))
  def test_while_continue_in_try(self, x, type_):
    x = type_(x)
    self.assertFunctionMatchesEager(while_continue_in_try, x)

  @parameterized.parameters(*itertools.product(
      (
          0, 4, 10,
      ),
      (
          int,
          _int_tensor,
      ),
  ))
  def test_while_break_in_context_manager(self, x, type_):
    x = type_(x)
    self.assertFunctionMatchesEager(while_break_in_context_manager, x)

  @parameterized.parameters(*itertools.product(
      (
          0, 4, 10,
      ),
      (
          int,
          _int_tensor,
      ),
  ))
  def test_while_break_in_try(self, x, type_):
    x = type_(x)
    self.assertFunctionMatchesEager(while_break_in_try, x)

  @parameterized.parameters(*itertools.product(
      (
          1, 2,
      ),
      (
          int,
          _int_tensor,
      ),
  ))
  def test_loop_initializing_invariant_variable_legal(self, n, type_):
    n = type_(n)
    self.assertFunctionMatchesEager(loop_initializing_invariant_variable, n)

  def test_loop_initializing_invariant_variable_illegal(self):
    with self.assertRaises(UnboundLocalError):
      tf.function(loop_initializing_invariant_variable)(0)
    with self.assertRaisesRegex(
        tf.errors.InvalidArgumentError, 'loop must iterate at least once'):
      tf.function(loop_initializing_invariant_variable)(tf.constant(0))

  @parameterized.parameters(
      (1,),
      (2,),
  )
  def test_loop_initializing_variant_variable_legal(self, n):
    tf.function(loop_initializing_variant_variable)(n)

  @parameterized.parameters(
      (0,),
      (1,),
      (2,),
  )
  def test_loop_initializing_variant_variable_illegal(self, n):
    with self.assertRaisesRegex(ValueError, 'must be defined before the loop'):
      tf.function(loop_initializing_variant_variable)(tf.constant(n))


if __name__ == '__main__':
  tf.test.main()
