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
"""Multiple returns, some in conditionals."""

import itertools

from absl.testing import parameterized
import tensorflow as tf

from tensorflow.python.autograph.tests import reference_test_base


def return_with_default(x):
  if x > 0:
    tf.print('x', x)
    return x
  return x * x


def return_dependent_on_local(c):
  t = tf.constant(1)
  if c:
    return t
  t = tf.stack([t, t])
  return tf.reduce_sum(t)


def return_possibly_undefined(x):
  if x > 0:
    if x < 5:
      return x
  else:
    return x * x * x


def nested_ifs(x):
  if x > 0:
    if x < 5:
      return x
    else:
      return x * x
  else:
    return x * x * x


def possible_return_before_loop(c1, c2, n):
  if c1:
    if c2:
      return 1
  for _ in range(n):
    pass
  return 2


def nested_ifs_and_context_managers(x):
  with tf.name_scope(''):
    if x > 0:
      if x < 5:
        with tf.name_scope(''):
          return x
      else:
        return x * x
    else:
      return x * x * x


def unreachable_return(x):
  with tf.name_scope(''):
    if x > 0:
      if x < 5:
        with tf.name_scope(''):
          return x
      else:
        return x * x
    else:
      return x * x * x
  return x * x * x * x


def return_with_default_in_contexmanager(x):
  with tf.name_scope(''):
    if x > 0:
      return 1
    return 0


def return_in_try_with_finally(x):
  try:
    if x > 0:
      return 1
    else:
      return 0
  finally:
    x = x + 1


def return_with_default_in_try_with_finally(x):
  try:
    if x > 0:
      return 1
    return 0
  finally:
    x = x + 1


def return_in_finally(x):
  try:
    return 2
  finally:
    if x > 0:
      return 1  # pylint: disable=lost-exception
    else:
      return 0  # pylint: disable=lost-exception


def return_with_default_in_finally(x):
  try:
    return 2
  finally:
    if x > 0:
      return 1  # pylint: disable=lost-exception
    return 0  # pylint: disable=lost-exception


def return_in_finally_default_in_try(x):
  try:
    if x > 0:
      return 0
  finally:
    return 1  # pylint: disable=lost-exception


def _raising_helper():
  raise ValueError()


def raise_during_return_caught():
  try:
    return _raising_helper()
  except ValueError:
    pass
  return 1


def raise_during_return_caught_in_tail_branch(c):
  if c:
    return 2
  try:
    return _raising_helper()
  except ValueError:
    pass
  return 1


class ReferenceTest(reference_test_base.TestCase, parameterized.TestCase):
  """Base class for the reference tests."""

  @parameterized.parameters(*itertools.product(
      (0, 1),
      (int, tf.constant),
  ))
  def test_return_with_default(self, n, type_):
    self.assertFunctionMatchesEager(return_with_default, type_(n))

  @parameterized.parameters(*itertools.product(
      (True, False),
      (int, tf.constant),
  ))
  def test_return_dependent_on_local(self, c, type_):
    self.assertFunctionMatchesEager(return_dependent_on_local, type_(c))

  @parameterized.parameters((0,), (3,), (5,))
  def test_return_possibly_undefined_legal(self, n):
    self.assertFunctionMatchesEager(return_possibly_undefined, n)

  @parameterized.parameters((0,), (3,), (5,))
  def test_return_possibly_undefined_illegal(self, n):
    with self.assertRaisesRegex(
        ValueError, 'else branch must also have a return'):
      tf.function(return_possibly_undefined)(tf.constant(n))

  @parameterized.parameters(*itertools.product(
      (-1, 3, 6),
      (int, tf.constant),
  ))
  def test_nested_ifs(self, n, type_):
    self.assertFunctionMatchesEager(nested_ifs, type_(n))

  @parameterized.parameters(*itertools.product(
      (True, False),
      (True, False),
      (0, 1, 2),
  ))
  def test_possible_return_before_loop(self, c1, c2, n):
    self.assertFunctionMatchesEager(possible_return_before_loop, c1, c2, n)

  @parameterized.parameters(*itertools.product(
      (0, 3, 5),
      (int, tf.constant),
  ))
  def test_nested_ifs_and_context_managers(self, x, type_):
    self.assertFunctionMatchesEager(nested_ifs_and_context_managers, type_(x))

  @parameterized.parameters(*itertools.product(
      (0, 3, 5),
      (int, tf.constant),
  ))
  def test_unreachable_return(self, x, type_):
    self.assertFunctionMatchesEager(unreachable_return, type_(x))

  @parameterized.parameters(*itertools.product(
      (0, 1),
      (int, tf.constant),
  ))
  def test_return_with_default_in_contexmanager(self, x, type_):
    self.assertFunctionMatchesEager(
        return_with_default_in_contexmanager, type_(x))

  @parameterized.parameters(*itertools.product(
      (0, 1),
      (int, tf.constant),
  ))
  def test_return_in_try_finally(self, x, type_):
    self.assertFunctionMatchesEager(return_in_try_with_finally, type_(x))

  @parameterized.parameters(*itertools.product(
      (0, 1),
      (int, tf.constant),
  ))
  def test_return_with_default_try_finally(self, x, type_):
    self.assertFunctionMatchesEager(
        return_with_default_in_try_with_finally, type_(x))

  @parameterized.parameters(*itertools.product(
      (0, 1),
      (int, tf.constant),
  ))
  def test_return_in_finally(self, x, type_):
    self.assertFunctionMatchesEager(return_in_finally, type_(x))

  @parameterized.parameters(*itertools.product(
      (0, 1),
      (int, tf.constant),
  ))
  def test_return_with_default_in_finally(self, x, type_):
    self.assertFunctionMatchesEager(return_with_default_in_finally, type_(x))

  @parameterized.parameters(*itertools.product(
      (0, 1),
      (int, tf.constant),
  ))
  def test_return_in_finally_default_in_try(self, x, type_):
    self.assertFunctionMatchesEager(return_in_finally_default_in_try, type_(x))

  def test_raise_during_return_caught(self):
    self.assertFunctionMatchesEager(raise_during_return_caught)

  @parameterized.parameters(*itertools.product(
      (True, False),
      (int, tf.constant),
  ))
  def test_raise_during_return_caught_in_tail_branch(self, c, type_):
    self.assertFunctionMatchesEager(
        raise_during_return_caught_in_tail_branch, type_(c))


if __name__ == '__main__':
  tf.test.main()
