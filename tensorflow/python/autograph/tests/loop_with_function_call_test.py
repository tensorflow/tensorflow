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
"""Function calls inside the while loop body."""

import itertools

from absl.testing import parameterized
import tensorflow as tf

from tensorflow.python.autograph.tests import reference_test_base


def while_with_call_in_cond(n, fn):
  i = 0
  s = 0
  while i < fn(n):
    s = s * 10 + i
    i += 1
  return s


def for_with_call_in_target(l, fn):
  s = 0
  for i in fn(l):
    s = s * 10 + i
  return s


def while_with_local_call_in_cond(n):

  def local_fn(x):
    return x * 3

  i = 0
  s = 0
  while i < local_fn(n):
    s = s * 10 + i
    i += 1
  return s


def for_with_local_call_in_target(l):

  def local_fn(l):
    return l * 1

  s = 0
  for i in local_fn(l):
    s = s * 10 + i
  return s


def while_with_call(n, fn):
  i = 0
  s = 0
  while i < n:
    s = s * 10 + fn(i)
    i += 1
  return s


def for_with_call(l, fn):
  s = 0
  for i in l:
    s = s * 10 + fn(i)
  return s


def while_with_local_call(n):

  def local_fn(x):
    return x * 3

  i = 0
  s = 0
  while i < n:
    s = s * 10 + local_fn(i)
    i += 1
  return s


def for_with_local_call(l):

  def local_fn(x):
    return x * 3

  s = 0
  for i in l:
    s = s * 10 + local_fn(i)
  return s


def while_with_closure_call(n):
  i = 0

  def i_via_closure():
    return i + 2

  i = 0
  s = 0
  while i < n:
    s = s * 10 + i_via_closure()
    i += 1
  return s


def for_with_closure_call(l):
  i = 0

  def i_via_closure():
    return i + 2

  s = 0
  for i in l:
    s = s * 10 + i_via_closure()
  # TODO(b/134822197): Remove i from return values.
  return s, i


def while_with_lambda_closure_call(n):
  i = 0
  s = 0
  i_via_closure = lambda: i + 2
  while i < n:
    s = s * 10 + i_via_closure()
    i += 1
  return s


def for_with_lambda_closure_call(l):
  i = 0
  s = 0
  i_via_closure = lambda: i + 2
  for i in l:
    s = s * 10 + i_via_closure()
  # TODO(b/134822197): Remove i from return values.
  return s, i


def while_with_method_closure_call(n):
  i = 0

  class Callable(object):

    def __call__(self):
      return i

  i_via_closure = Callable()
  i = 0
  s = 0
  while i < n:
    s = s * 10 + i_via_closure()
    i += 1
  return s


def for_with_method_closure_call(l):
  i = 0

  class Callable(object):

    def __call__(self):
      return i

  i_via_closure = Callable()
  i = 0
  s = 0
  for i in l:
    s = s * 10 + i_via_closure()
  # TODO(b/134822197): Remove i from return values.
  return s, i


def global_fn(x):
  return x * 2


class TestClass(object):

  def method(self, x):
    return x * 4


def _int_tensor(x):
  return tf.constant(x, dtype=tf.int32)


class ReferenceTest(reference_test_base.TestCase, parameterized.TestCase):

  @parameterized.parameters(*itertools.product(
      (0, 1, 2),
      (int, tf.constant),
      (global_fn, lambda x: x * 1, TestClass().method, abs),
  ))
  def test_while_with_call_in_cond(self, n, type_, fn):
    n = type_(n)
    self.assertFunctionMatchesEager(while_with_call_in_cond, n, fn)

  @parameterized.parameters(*itertools.product(
      ([], [1], [1, 2]),
      (list, _int_tensor),
      (global_fn, lambda x: x * 1, TestClass().method, tf.abs),
  ))
  def test_for_with_call_in_target(self, l, type_, fn):
    if fn is tf.abs and type_ is list:
      self.skipTest('tf.abs([]) defaults to float32')
    l = type_(l)
    self.assertFunctionMatchesEager(for_with_call_in_target, l, fn)

  @parameterized.parameters(*itertools.product(
      (0, 1, 2),
      (int, _int_tensor),
      (range, tf.range),
  ))
  def test_for_with_range_call_in_target(self, l, type_, fn):
    l = type_(l)
    self.assertFunctionMatchesEager(for_with_call_in_target, l, fn)

  @parameterized.parameters(*itertools.product(
      (0, 1, 2),
      (int, tf.constant),
      (global_fn, lambda x: x * 1, TestClass().method, abs),
  ))
  def test_while_with_call(self, n, type_, fn):
    n = type_(n)
    self.assertFunctionMatchesEager(while_with_call, n, fn)

  @parameterized.parameters(*itertools.product(
      ([], [1], [1, 2]),
      (list, _int_tensor),
      (global_fn, lambda x: x * 1, TestClass().method, abs),
  ))
  def test_for_with_call(self, l, type_, fn):
    l = type_(l)
    self.assertFunctionMatchesEager(for_with_call, l, fn)

  @parameterized.parameters(*itertools.product(
      (0, 1, 2),
      (int, tf.constant),
  ))
  def test_while_with_local_call(self, n, type_):
    n = type_(n)
    self.assertFunctionMatchesEager(while_with_local_call, n)

  @parameterized.parameters(*itertools.product(
      ([], [1], [1, 2]),
      (list, _int_tensor),
  ))
  def test_for_with_local_call(self, l, type_):
    l = type_(l)
    self.assertFunctionMatchesEager(for_with_local_call, l)

  @parameterized.parameters(*itertools.product(
      (0, 1, 2),
      (int, tf.constant),
  ))
  def test_while_with_closure_call(self, n, type_):
    n = type_(n)
    self.assertFunctionMatchesEager(while_with_closure_call, n)

  @parameterized.parameters(*itertools.product(
      ([], [1], [1, 2]),
      (list, _int_tensor),
  ))
  def test_for_with_closure_call(self, l, type_):
    l = type_(l)
    self.assertFunctionMatchesEager(for_with_closure_call, l)

  @parameterized.parameters(*itertools.product(
      (0, 1, 2),
      (int, tf.constant),
  ))
  def test_while_with_lambda_closure_call(self, n, type_):
    n = type_(n)
    self.assertFunctionMatchesEager(while_with_lambda_closure_call, n)

  @parameterized.parameters(*itertools.product(
      ([], [1], [1, 2]),
      (list, _int_tensor),
  ))
  def test_for_with_lambda_closure_call(self, l, type_):
    l = type_(l)
    self.assertFunctionMatchesEager(for_with_lambda_closure_call, l)

  @parameterized.parameters(*itertools.product(
      (0, 1, 2),
      (int, tf.constant),
  ))
  def test_while_with_method_closure_call(self, n, type_):
    self.skipTest('fix static analysis for nested classes')
    n = type_(n)
    self.assertFunctionMatchesEager(while_with_method_closure_call, n)

  @parameterized.parameters(*itertools.product(
      ([], [1], [1, 2]),
      (list, _int_tensor),
  ))
  def test_for_with_method_closure_call(self, l, type_):
    self.skipTest('fix static analysis for nested classes')
    l = type_(l)
    self.assertFunctionMatchesEager(for_with_method_closure_call, l)


if __name__ == '__main__':
  tf.test.main()
