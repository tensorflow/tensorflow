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
"""Tests that verify scoping around loops."""

import itertools

from absl.testing import parameterized
import tensorflow as tf

from tensorflow.python.autograph.tests import reference_test_base

#tf.autograph.set_verbosity(10, True)
def for_with_local_var(l):
  s = 0
  for i in l:
    x = i + 2
    s = s * 10 + x
  return s


def while_with_local_var(x):
  s = 0
  while x > 0:
    y = x + 2
    s = s * 10 + y
    x -= 1
  return s

def for_with_lambda_iter(l):
  fns = []
  results = []
  for i in l:
    fns.append(lambda: i)
  for f in fns:
    results.append(f())
  return results

def for_with_lambda_iter_local_var(l):
  fns = []
  results = []
  for i in l:
    fns.append(lambda i=i: i)
  for f in fns:
    results.append(f())
  return results

def for_initializes_local_var(l):
  s = 0
  for i in l:
    if i == l[0]:
      x = 0
    else:
      x += 1
    s = s * 10 + x
  return s


def while_initializes_local_var(x):
  s = 0
  while x > 0:
    if x > 0:
      y = 0
    else:
      y += 1
    s = s * 10 + y
    x -= 1
  return s


def for_defines_var(l):
  for i in l:
    x = i + 2
  return x


def while_defines_var(x):
  while x > 0:
    y = x + 2
    x -= 1
  return y


def for_defines_iterate(n, fn):
  s = 0
  for i in fn(n):
    s = s * 10 + i
  return i, s  # pylint:disable=undefined-loop-variable


def for_reuses_iterate(n, fn):
  i = 7
  s = 0
  for i in fn(n):
    s = s * 10 + i
  return i, s


def for_alters_iterate(n, fn):
  i = 7
  s = 0
  for i in fn(n):
    i = 3 * i + 1
    s = s * 10 + i
  return i, s


def _int_tensor(x):
  return tf.constant(x, dtype=tf.int32)


class LoopScopingTest(reference_test_base.TestCase, parameterized.TestCase):

  @parameterized.parameters(*itertools.product(
      ([], [1], [1, 2]),
      (list, _int_tensor),
  ))
  def test_for_with_local_var(self, l, type_):
    l = type_(l)
    self.assertFunctionMatchesEager(for_with_local_var, l)

  @parameterized.parameters(*itertools.product(
      (0, 1, 2),
      (range, tf.range),
  ))
  def test_for_with_local_var_range(self, l, type_):
    l = type_(l)
    self.assertFunctionMatchesEager(for_with_local_var, l)

  @parameterized.parameters(*itertools.product(
      ([], [1], [1, 2], [(1,2),(3,4)]),
      (list, list),
  ))
  def test_for_with_lambda_iter(self, l, type_):
    l = type_(l)
    self.assertFunctionMatchesEager(for_with_lambda_iter, l)

  @parameterized.parameters(*itertools.product(
      ([], [1], [1, 2], [(1,2),(3,4)]),
      (list, list),
  ))

  def test_for_with_lambda_iter_local_var(self, l, type_):
    l = type_(l)
    self.assertFunctionMatchesEager(for_with_lambda_iter_local_var, l)

  @parameterized.parameters(*itertools.product(
      (0, 1, 2),
      (int, _int_tensor),
  ))
  def test_while_with_local_var(self, x, type_):
    x = type_(x)
    self.assertFunctionMatchesEager(while_with_local_var, x)

  @parameterized.parameters(
      ([],),
      ([1],),
      ([1, 2],),
  )
  def test_for_initializes_local_var_legal_cases(self, l):
    self.assertFunctionMatchesEager(for_initializes_local_var, l)

  @parameterized.parameters(
      ([],),
      ([1],),
      ([1, 2],),
  )
  def test_for_initializes_local_var_illegal_cases(self, l):
    self.skipTest("TODO(mdanatg): Check")
    l = tf.constant(l)
    with self.assertRaisesRegex(ValueError, '"x" must be defined'):
      tf.function(for_initializes_local_var)(l)

  @parameterized.parameters(
      0,
      1,
      2,
  )
  def test_while_initializes_local_var_legal_cases(self, x):
    self.assertFunctionMatchesEager(while_initializes_local_var, x)

  @parameterized.parameters(
      0,
      1,
      2,
  )
  def test_while_initializes_local_var_illegal_cases(self, x):
    self.skipTest("TODO(mdanatg): check")
    x = tf.constant(x)
    with self.assertRaisesRegex(ValueError, '"y" must be defined'):
      tf.function(while_initializes_local_var)(x)

  @parameterized.parameters(
      # TODO(b/155171694): Enable once the error message here is corrected.
      # ([],),
      ([1],),
      ([1, 2],),
  )
  def test_for_defines_var_legal_cases(self, l):
    self.assertFunctionMatchesEager(for_defines_var, l)

  @parameterized.parameters(
      ([],),
      ([1],),
      ([1, 2],),
  )
  def test_for_defines_var_illegal_cases(self, l):
    self.skipTest("TODO(mdanatg): check")
    l = tf.constant(l)
    with self.assertRaisesRegex(ValueError, '"x" must be defined'):
      tf.function(for_defines_var)(l)

  @parameterized.parameters(
      # TODO(b/155171694): Enable once the error message here is corrected.
      # (0,),
      (1,),
      (2,),
  )
  def test_while_defines_var_legal_cases(self, x):
    self.assertFunctionMatchesEager(while_defines_var, x)

  @parameterized.parameters(
      (0,),
      (1,),
      (2,),
  )
  def test_while_defines_var_illegal_cases(self, x):
    self.skipTest("TODO(mdanatg): check")
    x = tf.constant(x)
    with self.assertRaisesRegex(ValueError, '"y" must be defined'):
      tf.function(while_defines_var)(x)

  @parameterized.parameters(*itertools.product(
      (1, 2),
      (range, tf.range),
  ))
  def test_for_defines_iterate_legal_cases(self, n, fn):
    self.assertFunctionMatchesEager(for_defines_iterate, n, fn)

  def test_for_defines_iterate_range(self):
    self.skipTest('b/155171694')

  def test_for_defines_iterate_tf_range(self):
    # Deviating from the normal Python semantics here to avoid inserting
    # an extra assert op. If needed, we can insert it and raise an error
    # to mimic the eager behavior, but this is an exceptionally uncummon
    # use case.
    self.assertAllEqual(tf.function(for_defines_iterate)(0, tf.range), (0, 0))

  @parameterized.parameters(*itertools.product(
      ([], [1], [1, 2]),
      (list, _int_tensor),
  ))
  def test_for_reuses_iterate(self, l, fn):
    self.assertFunctionMatchesEager(for_reuses_iterate, l, fn)

  @parameterized.parameters(*itertools.product(
      (0, 1, 2),
      (range, tf.range),
  ))
  def test_for_reuses_iterate_range(self, n, fn):
    self.assertFunctionMatchesEager(for_reuses_iterate, n, fn)

  @parameterized.parameters(*itertools.product(
      ([], [1], [1, 2]),
      (list, _int_tensor),
  ))
  def test_for_alters_iterate(self, l, fn):
    self.assertFunctionMatchesEager(for_alters_iterate, l, fn)

  @parameterized.parameters(*itertools.product(
      (0, 1, 2),
      (range, tf.range),
  ))
  def test_for_alters_iterate_range(self, n, fn):
    self.assertFunctionMatchesEager(for_alters_iterate, n, fn)


if __name__ == '__main__':
  tf.test.main()
