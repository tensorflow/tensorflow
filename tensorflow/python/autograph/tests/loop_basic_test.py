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
"""Basic loops iterating over various types."""

import functools
import itertools

from absl.testing import parameterized
import tensorflow as tf

from tensorflow.python.autograph.tests import reference_test_base


def while_no_vars(n, v):
  v.assign(0)
  while v.read_value() < n:
    v.assign_add(1)
  return v.read_value()


def for_no_vars(l, v):
  v.assign(0)
  for _ in l:
    v.assign_add(1)
  return v.read_value()


def while_one_var(n):
  i = 0
  while i < n:
    i = i * 10 + 1
  return i


def for_one_var(l):
  i = 0
  for i in l:
    pass
  return i


def while_two_vars(n):
  s = 0
  i = 0
  while i < n:
    s = s * 10 + i
    i += 1
  return s


def for_two_vars(l):
  s = 0
  for i in l:
    s = s * 10 + i
  return s


def while_else(x, y):
  s = 0
  while x > 0:
    x -= 1
    if x > y:
      break
    s += x
  else:
    s = -100
  return s


def for_else(l1, l2):
  s = 0
  for c in l1:
    if c in l2:
      break
    s = s * 10 + c
  else:
    s = -1000
  return s


def while_creates_var(n):
  i = 0
  while i < n:
    i = n + 1
    j = i + 1
  return i, j


def for_creates_var(l):
  for i in l:
    pass
  return i  # pylint:disable=undefined-loop-variable


def successive_while_loops(n1, n2):
  s = 0
  i = 0
  while i < n1:
    s = s * 10 + i
    i += 1
  i = 0
  while i < n2:
    s = s * 10 + i
    i += 1
  return s


def successive_for_loops(l1, l2):
  s = 0
  for i in l1:
    s = s * 10 + i
  for i in l2:
    s = s * 10 + i
  return s


def nested_while_loops(n1, n2):
  i = 0
  l = tf.TensorArray(tf.int32, size=0, dynamic_size=True, element_shape=())
  while i < n1:
    j = 0
    s = 0
    while j < n2:
      s = s * 10 + i * j
      j += 1
    l = l.write(i, s)
    i += 1
  return l.stack()


def nested_for_loops(m):
  l = tf.TensorArray(tf.int32, size=0, dynamic_size=True, element_shape=())
  for i in m:
    s = 0
    for j in i:
      s = s * 10 + j
    l = l.write(l.size(), s)
  return l.stack()


def _int_tensor(x):
  return tf.constant(x, dtype=tf.int32)


def _int_dataset(l):
  return tf.data.Dataset.from_tensor_slices(tf.constant(l, dtype=tf.int32))


def double_product(l1, l2):
  for i in l1:
    for j in l2:
      for k in l1:
        for l in l2:
          yield i, j, k, l


class ReferenceTest(reference_test_base.TestCase, parameterized.TestCase):

  @parameterized.parameters(*itertools.product(
      (
          0,
          1,
          2,
      ),
      (
          int,
          _int_tensor,
      ),
  ))
  def test_while_no_vars(self, n, type_):
    n = type_(n)
    self.assertFunctionMatchesEager(while_no_vars, n, tf.Variable(0))

  @parameterized.parameters(*itertools.product(
      (
          [],
          [1],
          [1, 2],
      ),
      (
          list,
          _int_tensor,
          # TODO(mdan): Enable this once #35335 is fixed.
          # _int_dataset,
      ),
      (
          True,
          False,
      ),
  ))
  def test_for_no_vars(self, l, type_, xla):
    if type_ is _int_tensor and xla and not l:
      self.skipTest('Empty loops not supported in XLA')

    l = type_(l)
    self.assertFunctionMatchesEager(for_no_vars, l, tf.Variable(0), xla=xla)

  @parameterized.parameters(
      ([],),
      ([1],),
      ([1, 2],),
  )
  def test_for_no_vars_ds_iterator(self, l):
    inputs_ = lambda: (iter(_int_dataset(l)), tf.Variable(0))
    self.assertFunctionMatchesEagerStatefulInput(for_no_vars, inputs_)

  @parameterized.parameters(*itertools.product(
      (
          0,
          1,
          2,
      ),
      (
          int,
          _int_tensor,
      ),
  ))
  def test_while_one_var(self, n, type_):
    n = type_(n)
    self.assertFunctionMatchesEager(while_one_var, n)

  @parameterized.parameters(*itertools.product(
      (
          [],
          [1],
          [1, 2],
      ),
      (
          list,
          _int_tensor,
          _int_dataset,
      ),
      (
          True,
          False,
      ),
  ))
  def test_for_one_var(self, l, type_, xla):
    if type_ is _int_dataset and xla:
      self.skipTest('Datasets not supported in XLA')
    if type_ is _int_tensor and xla and not l:
      self.skipTest('Empty loops not supported in XLA')

    l = type_(l)
    self.assertFunctionMatchesEager(for_one_var, l, xla=xla)

  @parameterized.parameters(
      ([],),
      ([1],),
      ([1, 2],),
  )
  def test_for_one_var_ds_iterator(self, l):
    inputs_ = lambda: (iter(_int_dataset(l)), tf.Variable(0))
    self.assertFunctionMatchesEagerStatefulInput(for_one_var, inputs_)

  @parameterized.parameters(*itertools.product(
      (
          0,
          1,
          2,
      ),
      (
          int,
          _int_tensor,
      ),
  ))
  def test_while_two_vars(self, n, type_):
    n = type_(n)
    self.assertFunctionMatchesEager(while_two_vars, n)

  @parameterized.parameters(*itertools.product(
      (
          [],
          [1],
          [1, 2],
      ),
      (
          list,
          _int_tensor,
          _int_dataset,
      ),
      (
          True,
          False,
      ),
  ))
  def test_for_two_vars(self, l, type_, xla):
    if type_ is _int_dataset and xla:
      self.skipTest('Datasets not supported in XLA')
    if type_ is _int_tensor and xla and not l:
      self.skipTest('Empty loops not supported in XLA')

    l = type_(l)
    self.assertFunctionMatchesEager(for_two_vars, l, xla=xla)

  @parameterized.parameters(
      ([],),
      ([1],),
      ([1, 2],),
  )
  def test_for_two_vars_ds_iterator(self, l):
    inputs_ = lambda: (iter(_int_dataset(l)), tf.Variable(0))
    self.assertFunctionMatchesEagerStatefulInput(for_two_vars, inputs_)

  @parameterized.parameters(*itertools.product(
      (
          [],
          [1, 2],
      ),
      (
          [],
          [1, 2],
      ),
      (list, _int_tensor),
  ))
  def test_for_else(self, l1, l2, type_):
    l1 = type_(l1)
    l2 = type_(l2)
    with self.assertRaisesRegex(NotImplementedError, 'for/else'):
      tf.function(for_else)(l1, l2)

  @parameterized.parameters(*itertools.product(
      (0, 2),
      (0, 2),
      (int, _int_tensor),
  ))
  def test_while_else(self, n1, n2, type_):
    n1 = type_(n1)
    n2 = type_(n2)
    with self.assertRaisesRegex(NotImplementedError, 'while/else'):
      tf.function(while_else)(n1, n2)

  @parameterized.parameters(*itertools.product(
      (
          1,
          2,
      ),
      (
          int,
          _int_tensor,
      ),
  ))
  def test_while_creates_var_legal(self, n, type_):
    n = type_(n)
    self.assertFunctionMatchesEager(while_creates_var, n)

  def test_while_creates_var_illegal(self):
    with self.assertRaisesRegex(UnboundLocalError, 'used before assignment'):
      tf.function(while_creates_var)(0)
    with self.assertRaisesRegex(
        tf.errors.InvalidArgumentError, 'loop must iterate at least once'):
      tf.function(while_creates_var)(tf.constant(0))

  @parameterized.parameters(*itertools.product(
      (
          [1],
          [1, 2],
      ),
      (
          list,
          _int_tensor,
      ),
  ))
  def test_for_creates_var_legal(self, l, type_):
    l = type_(l)
    self.assertFunctionMatchesEager(for_creates_var, l)

  def test_for_creates_var_illegal(self):
    with self.assertRaisesRegex(UnboundLocalError, 'used before assignment'):
      tf.function(for_creates_var)([])
    with self.assertRaisesRegex(
        tf.errors.InvalidArgumentError, 'loop must iterate at least once'):
      tf.function(for_creates_var)(tf.constant([]))

  @parameterized.parameters(*double_product(
      (
          0,
          1,
          2,
      ),
      (
          int,
          _int_tensor,
      ),
  ))
  def test_successive_while_loops(self, n1, type1, n2, type2):
    n1 = type1(n1)
    n2 = type1(n2)
    self.assertFunctionMatchesEager(successive_while_loops, n1, n2)

  @parameterized.parameters(*double_product(
      (
          [],
          [1],
          [1, 2],
      ),
      (
          list,
          _int_tensor,
          _int_dataset,
      ),
  ))
  def test_successive_for_loops(self, l1, type1, l2, type2):
    l1 = type1(l1)
    l2 = type1(l2)
    self.assertFunctionMatchesEager(successive_for_loops, l1, l2)

  @parameterized.parameters(*double_product(
      (
          [],
          [1],
          [1, 2],
      ),
      (
          list,
          _int_dataset,
      ),
  ))
  def test_successive_for_loops_iterators(self, l1, type1, l2, type2):
    inputs_ = lambda: (iter(type1(l1)), iter(type2(l2)))
    self.assertFunctionMatchesEagerStatefulInput(successive_for_loops, inputs_)

  @parameterized.parameters(*double_product(
      (
          0,
          1,
          2,
      ),
      (
          int,
          _int_tensor,
      ),
  ))
  def test_nested_while_loops(self, n1, type1, n2, type2):
    n1 = type1(n1)
    n2 = type1(n2)
    self.assertFunctionMatchesEager(nested_while_loops, n1, n2)

  @parameterized.parameters(*itertools.product(
      (
          [[]],
          [[], []],
          [[1]],
          [[1], [2]],
          [[1, 2]],
          [[1, 2], [3, 4]],
      ),
      (
          _int_tensor,
          _int_dataset,
      ),
  ))
  def test_nested_for_loops_dense(self, m, type_):
    m = type_(m)
    self.assertFunctionMatchesEager(nested_for_loops, m)

  @parameterized.parameters(*itertools.product(
      (
          [[]],
          [[], [1]],
          [[], [1], [1, 2]],
      ),
      (
          list,
          functools.partial(tf.ragged.constant, dtype=tf.int32),
      ),
  ))
  def test_nested_for_loops_ragged(self, m, type_):
    m = type_(m)
    self.assertFunctionMatchesEager(nested_for_loops, m)

  def test_nested_for_loops_mixed_list(self):
    m = [[], _int_tensor([]), [1], _int_tensor([1]), [1, 2]]
    self.assertFunctionMatchesEager(nested_for_loops, m)


if __name__ == '__main__':
  tf.test.main()
