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
"""Tests involving the tf.distributed datasets."""

import itertools

from absl.testing import parameterized
import tensorflow as tf

from tensorflow.python.autograph.tests import reference_test_base


def no_vars_loop(strat, iterable):
  for pr in iterable:
    tf.print(strat.reduce('SUM', pr, axis=0))


def single_var_loop(strat, iterable):
  s = 0
  for pr in iterable:
    # TODO(mdan): It would be nice to be able to write s = s * 10 + pr.
    s = s * 10 + strat.reduce('SUM', pr, axis=0)
  return s


def loop_with_break(strat, iterable):
  s = 0
  for pr in iterable:
    if strat.reduce('SUM', pr, axis=0) % 5 == 0:
      break
    s = s * 10 + strat.reduce('SUM', pr, axis=0)
  return s


def loop_with_continue(strat, iterable):
  s = 0
  for pr in iterable:
    if strat.reduce('SUM', pr, axis=0) % 2 == 0:
      continue
    s = s * 10 + strat.reduce('SUM', pr, axis=0)
  return s


def two_vars_loop(strat, iterable):
  s = 0
  p = 1
  for pr in iterable:
    e = strat.reduce('SUM', pr, axis=0)
    s += e
    p *= e
  return s, p


def enumeration(strat, iterable):
  s = 0
  p = 1
  for i, pr in enumerate(iterable):
    e = strat.reduce('SUM', pr, axis=0)
    s = s * 10 + e
    p *= i
  return s, p


def iterator_next(strat, iterable):
  itr = iter(iterable)
  return strat.reduce('SUM', next(itr), axis=0)


def iterator_next_multiple_calls(strat, iterable):
  itr = iter(iterable)
  a = strat.reduce('SUM', next(itr), axis=0)
  b = strat.reduce('SUM', next(itr), axis=0)
  return a * 10 + b


def iterator_next_in_limited_loop(strat, iterable, l):
  itr = iter(iterable)
  s = 0
  for _ in l:
    s = s * 10 + strat.reduce('SUM', next(itr), axis=0)
  return s


def iterator_next_stopping(strat, iterable, cond):
  # This case will raise, but not the expected StopIteration error.
  itr = iter(iterable)
  while cond:
    strat.reduce('SUM', next(itr), axis=0)


def iterator_next_with_catching_stop_iteration(strat, iterable, cond):
  # This is the one instance when the use of TF iterators does not work as
  # intended. In graph mode, the `except` below will never catch, and the
  # tf.function will raise the error instead.
  # TODO(b/132311724): The error should be friendlier here.
  # Note: b/132298783 covers actually supporting this pattern.
  itr = iter(iterable)
  try:
    while cond:
      strat.reduce('SUM', next(itr), axis=0)
  except StopIteration:
    pass


def _distributed_dataset():
  cpus = tf.config.experimental.list_physical_devices('CPU')
  tf.config.experimental.set_virtual_device_configuration(
      cpus[0], [tf.config.experimental.VirtualDeviceConfiguration()] * 2)

  strat = tf.distribute.MirroredStrategy()
  ds = tf.data.Dataset.from_tensor_slices(tf.reshape(tf.range(12), (3, 4)))

  return strat, strat.experimental_distribute_dataset(ds)


def _distributed_iterator():
  strat, ds = _distributed_dataset()
  return strat, iter(ds)


class ReferenceTest(reference_test_base.TestCase, parameterized.TestCase):

  @parameterized.parameters(*itertools.product(
      (
          no_vars_loop,
          single_var_loop,
          two_vars_loop,
          loop_with_break,
          loop_with_continue,
      ),
      (
          _distributed_dataset,
          _distributed_iterator,
      ),
  ))
  def test_basic(self, test_fn, target):
    if (test_fn in (loop_with_break, loop_with_continue) and
        target is _distributed_dataset):
      self.skipTest('b/162250181')
    self.assertFunctionMatchesEagerStatefulInput(test_fn, target)

  def test_iterator_next(self):
    strat, ds = _distributed_dataset()
    self.assertFunctionMatchesEager(iterator_next, strat, ds)

  def test_iterator_next_multiple_calls(self):
    strat, ds = _distributed_dataset()
    self.assertFunctionMatchesEager(iterator_next_multiple_calls, strat, ds)

  @parameterized.parameters(*itertools.product(
      (
          0,
          1,
          2,
      ),
      (
          range,
          tf.range,
      ),
  ))
  def test_iterator_next_in_limited_loop(self, n, type_):
    n = type_(n)
    strat, ds = _distributed_dataset()
    self.assertFunctionMatchesEager(iterator_next_in_limited_loop, strat, ds, n)

  @parameterized.parameters(
      (iterator_next_stopping,),
      # Note that `except` has no effect in graph mode.
      (iterator_next_with_catching_stop_iteration,),
  )
  def test_iterator_next_stopping(self, test_fn):
    strat, ds = _distributed_dataset()
    with self.assertRaises(tf.errors.OutOfRangeError):
      tf.function(test_fn)(strat, ds, tf.constant(True))


if __name__ == '__main__':
  tf.test.main()
