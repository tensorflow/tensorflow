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
"""Tests involving the tf.data.Datasets API."""

import tensorflow as tf

from tensorflow.python.autograph.tests import reference_test_base


def dataset_no_vars_loop(ds):
  for e in ds:
    tf.print(e)


def iterator_no_vars_loop(ds):
  for e in iter(ds):
    tf.print(e)


def dataset_single_var_loop(ds):
  s = tf.constant(0, dtype=tf.int64)
  for e in ds:
    s = s * 10 + e
  return s


def iterator_single_var_loop(ds):
  s = tf.constant(0, dtype=tf.int64)
  for e in iter(ds):
    s = s * 10 + e
  return s


def dataset_two_vars_loop(ds):
  s = tf.constant(0, dtype=tf.int64)
  p = tf.constant(1, dtype=tf.int64)
  for e in ds:
    s += e
    p *= e
  return s, p


def iterator_two_vars_loop(ds):
  s = tf.constant(0, dtype=tf.int64)
  p = tf.constant(1, dtype=tf.int64)
  for e in iter(ds):
    s += e
    p *= e
  return s, p


def dataset_loop_with_break(ds):
  s = tf.constant(0, dtype=tf.int64)
  for e in ds:
    s = s * 10 + e
    if s > 100:
      break
  return s


def iterator_loop_with_break(ds):
  s = tf.constant(0, dtype=tf.int64)
  for e in iter(ds):
    s = s + e
    if s > 100:
      break
  return s


def iterator_resuming_loop(ds):
  s = tf.constant(0, dtype=tf.int64)
  itr = iter(ds)
  for e in itr:
    s = s * 10 + e
    break
  for e in itr:
    s = s * 10 + e
    break
  for e in itr:
    s = s * 10 + e
  return s


def dataset_loop_with_return(ds):
  y = tf.constant(0, dtype=tf.int64)
  for e in ds:
    y = e
    return y
  return y


def iterator_loop_with_return(ds):
  y = tf.constant(0, dtype=tf.int64)
  for e in iter(ds):
    y = e
    return y
  return y


def iterator_next(ds):
  itr = iter(ds)
  return next(itr)


def iterator_next_multiple_calls(ds):
  itr = iter(ds)
  return 10 * next(itr) + next(itr)


def iterator_next_in_loop(ds, n):
  itr = iter(ds)
  s = tf.constant(0, dtype=tf.int64)
  for _ in range(n):
    s = s * 10 + next(itr)
  return s


def iterator_next_stopping(ds, cond):
  # This case will raise, but not the expected StopIteration error.
  itr = iter(ds)
  while cond:
    next(itr)


def iterator_next_with_catching_stop_iteration(ds, cond):
  # This is the only instance when the use of TF iterators does not work as
  # intended. In graph mode, the `except` below will never catch, and the
  # tf.function will raise the error instead.
  # TODO(b/132311724): The error should be friendlier here.
  # Note: b/132298783 covers actually supporting this pattern.
  itr = iter(ds)
  try:
    while cond:
      next(itr)
  except StopIteration:
    pass


class ReferenceTest(reference_test_base.TestCase):

  def setUp(self):
    super(ReferenceTest, self).setUp()
    self.ds = tf.data.Dataset.range(7)

  def test_dataset_no_vars_loop(self):
    self.assertFunctionMatchesEager(dataset_no_vars_loop, self.ds)

  def test_iterator_no_vars_loop(self):
    self.assertFunctionMatchesEager(iterator_no_vars_loop, self.ds)

  def test_dataset_single_var_loop(self):
    self.assertFunctionMatchesEager(dataset_single_var_loop, self.ds)

  def test_iterator_single_var_loop(self):
    self.assertFunctionMatchesEager(iterator_single_var_loop, self.ds)

  def test_dataset_two_vars_loop(self):
    self.assertFunctionMatchesEager(dataset_two_vars_loop, self.ds)

  def test_iterator_two_vars_loop(self):
    self.assertFunctionMatchesEager(iterator_two_vars_loop, self.ds)

  def test_dataset_loop_with_break(self):
    self.assertFunctionMatchesEager(dataset_loop_with_break, self.ds)

  def test_iterator_loop_with_break(self):
    self.assertFunctionMatchesEager(iterator_loop_with_break, self.ds)

  def test_dataset_loop_with_return_raises(self):
    # This is for the same reason why returns in loops aren't allowed.
    # TODO(mdan): This might be resolved by unrolling the loop once.
    with self.assertRaisesRegex(
        NotImplementedError,
        'a return statement cannot be placed inside this TensorFlow loop'):
      tf.function(dataset_loop_with_return)(self.ds)

  def test_iterator_loop_with_return_raises(self):
    # This is for the same reason why returns in loops aren't allowed.
    # TODO(mdan): This might be resolved by unrolling the loop once.
    with self.assertRaisesRegex(
        NotImplementedError,
        'a return statement cannot be placed inside this TensorFlow loop'):
      tf.function(iterator_loop_with_return)(self.ds)

  def test_iterator_next(self):
    self.assertFunctionMatchesEager(iterator_next, self.ds)

  def test_iterator_next_multiple_calls(self):
    self.assertFunctionMatchesEager(iterator_next_multiple_calls, self.ds)

  def test_iterator_next_in_loop(self):
    self.assertFunctionMatchesEager(iterator_next_in_loop, self.ds, 7)

  def test_iterator_next_stopping(self):
    # Graph ops raise OutOfRangeError, but eager ops raise StopIteration
    with self.assertRaises(tf.errors.OutOfRangeError):
      tf.function(iterator_next_stopping)(self.ds, tf.constant(True))

  def test_iterator_next_with_catching_stop_iteration(self):
    # Graph ops raise OutOfRangeError, but eager ops raise StopIteration
    with self.assertRaises(tf.errors.OutOfRangeError):
      tf.function(iterator_next_with_catching_stop_iteration)(
          self.ds, tf.constant(True))


if __name__ == '__main__':
  tf.test.main()
