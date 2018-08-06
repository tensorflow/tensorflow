# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for the experimental input pipeline ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

from tensorflow.contrib.data.python.kernel_tests import stats_dataset_test_base
from tensorflow.contrib.data.python.ops import optimization
from tensorflow.contrib.data.python.ops import stats_ops
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class OptimizeDatasetTest(test.TestCase, parameterized.TestCase):

  def testAssertSuffix(self):
    dataset = dataset_ops.Dataset.from_tensors(0).apply(
        optimization.assert_next(["Map"])).map(lambda x: x)
    iterator = dataset.make_one_shot_iterator()
    get_next = iterator.get_next()

    with self.test_session() as sess:
      self.assertEqual(0, sess.run(get_next))

  def testAssertSuffixInvalid(self):
    dataset = dataset_ops.Dataset.from_tensors(0).apply(
        optimization.assert_next(["Whoops"])).map(lambda x: x)
    iterator = dataset.make_one_shot_iterator()
    get_next = iterator.get_next()

    with self.test_session() as sess:
      with self.assertRaisesRegexp(
          errors.InvalidArgumentError,
          "Asserted Whoops transformation at offset 0 but encountered "
          "Map transformation instead."):
        sess.run(get_next)

  def testAssertSuffixShort(self):
    dataset = dataset_ops.Dataset.from_tensors(0).apply(
        optimization.assert_next(["Map", "Whoops"])).map(lambda x: x)
    iterator = dataset.make_one_shot_iterator()
    get_next = iterator.get_next()

    with self.test_session() as sess:
      with self.assertRaisesRegexp(
          errors.InvalidArgumentError,
          "Asserted next 2 transformations but encountered only 1."):
        sess.run(get_next)

  def testDefaultOptimizations(self):
    dataset = dataset_ops.Dataset.range(10).apply(
        optimization.assert_next(
            ["Map", "Batch"])).map(lambda x: x * x).batch(10).apply(
                optimization.optimize())
    iterator = dataset.make_one_shot_iterator()
    get_next = iterator.get_next()

    with self.test_session() as sess:
      self.assertAllEqual([x * x for x in range(10)], sess.run(get_next))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  def testEmptyOptimizations(self):
    dataset = dataset_ops.Dataset.range(10).apply(
        optimization.assert_next(
            ["Map", "Batch"])).map(lambda x: x * x).batch(10).apply(
                optimization.optimize([]))
    iterator = dataset.make_one_shot_iterator()
    get_next = iterator.get_next()

    with self.test_session() as sess:
      self.assertAllEqual([x * x for x in range(10)], sess.run(get_next))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  def testOptimization(self):
    dataset = dataset_ops.Dataset.range(10).apply(
        optimization.assert_next(
            ["MapAndBatch"])).map(lambda x: x * x).batch(10).apply(
                optimization.optimize(["map_and_batch_fusion"]))
    iterator = dataset.make_one_shot_iterator()
    get_next = iterator.get_next()

    with self.test_session() as sess:
      self.assertAllEqual([x * x for x in range(10)], sess.run(get_next))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  def testFunctionLibraryDefinitionModification(self):
    dataset = dataset_ops.Dataset.from_tensors(0).map(lambda x: x).apply(
        optimization.optimize(["_test_only_function_rename"]))
    iterator = dataset.make_one_shot_iterator()
    get_next = iterator.get_next()

    with self.test_session() as sess:
      with self.assertRaisesRegexp(errors.NotFoundError,
                                   "Function .* is not defined."):
        sess.run(get_next)

  @staticmethod
  def map_functions():
    identity = lambda x: x
    increment = lambda x: x + 1

    def increment_and_square(x):
      y = x + 1
      return y * y

    functions = [identity, increment, increment_and_square]
    tests = []
    for i, fun1 in enumerate(functions):
      for j, fun2 in enumerate(functions):
        tests.append((
            "test_{}_{}".format(i, j),
            [fun1, fun2],
        ))
        for k, fun3 in enumerate(functions):
          tests.append((
              "test_{}_{}_{}".format(i, j, k),
              [fun1, fun2, fun3],
          ))

    swap = lambda x, n: (n, x)
    tests.append((
        "swap1",
        [lambda x: (x, 42), swap],
    ))
    tests.append((
        "swap2",
        [lambda x: (x, 42), swap, swap],
    ))
    return tuple(tests)

  @parameterized.named_parameters(*map_functions.__func__())
  def testMapFusion(self, functions):
    dataset = dataset_ops.Dataset.range(5).apply(
        optimization.assert_next(["Map", "Prefetch"]))
    for function in functions:
      dataset = dataset.map(function)

    dataset = dataset.prefetch(0).apply(optimization.optimize(["map_fusion"]))
    iterator = dataset.make_one_shot_iterator()
    get_next = iterator.get_next()
    with self.test_session() as sess:
      for x in range(5):
        result = sess.run(get_next)
        r = x
        for function in functions:
          if isinstance(r, tuple):
            r = function(*r)  # Pass tuple as multiple arguments.
          else:
            r = function(r)
        self.assertAllEqual(r, result)

      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  @staticmethod
  def map_and_filter_functions():
    identity = lambda x: x
    increment = lambda x: x + 1
    minus_five = lambda x: x - 5

    def increment_and_square(x):
      y = x + 1
      return y * y

    take_all = lambda x: constant_op.constant(True)
    is_zero = lambda x: math_ops.equal(x, 0)
    is_odd = lambda x: math_ops.equal(x % 2, 0)
    greater = lambda x: math_ops.greater(x + 5, 0)

    functions = [identity, increment, minus_five, increment_and_square]
    filters = [take_all, is_zero, is_odd, greater]
    tests = []

    for x, fun in enumerate(functions):
      for y, predicate in enumerate(filters):
        tests.append(("mixed_{}_{}".format(x, y), fun, predicate))

    # Multi output
    tests.append(("multiOne", lambda x: (x, x),
                  lambda x, y: constant_op.constant(True)))
    tests.append(
        ("multiTwo", lambda x: (x, 2),
         lambda x, y: math_ops.equal(x * math_ops.cast(y, dtypes.int64), 0)))
    return tuple(tests)

  @parameterized.named_parameters(*map_and_filter_functions.__func__())
  def testMapFilterFusion(self, function, predicate):
    dataset = dataset_ops.Dataset.range(10).apply(
        optimization.assert_next(
            ["Map",
             "FilterByLastComponent"])).map(function).filter(predicate).apply(
                 optimization.optimize(["map_and_filter_fusion"]))
    self._testMapAndFilter(dataset, function, predicate)

  def _testMapAndFilter(self, dataset, function, predicate):
    iterator = dataset.make_one_shot_iterator()
    get_next = iterator.get_next()
    with self.test_session() as sess:
      for x in range(10):
        r = function(x)
        if isinstance(r, tuple):
          b = predicate(*r)  # Pass tuple as multiple arguments.
        else:
          b = predicate(r)
        if sess.run(b):
          result = sess.run(get_next)
          self.assertAllEqual(r, result)
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  def testAdditionalInputs(self):
    a = constant_op.constant(3, dtype=dtypes.int64)
    b = constant_op.constant(4, dtype=dtypes.int64)
    some_tensor = math_ops.mul(a, b)
    function = lambda x: x * x

    def predicate(y):
      return math_ops.less(math_ops.cast(y, dtypes.int64), some_tensor)

    # We are currently not supporting functions with additional inputs.
    dataset = dataset_ops.Dataset.range(10).apply(
        optimization.assert_next(
            ["Map", "Filter"])).map(function).filter(predicate).apply(
                optimization.optimize(["map_and_filter_fusion"]))

    self._testMapAndFilter(dataset, function, predicate)


class OptimizeStatsDatasetTest(stats_dataset_test_base.StatsDatasetTestBase):

  def testLatencyStatsOptimization(self):

    stats_aggregator = stats_ops.StatsAggregator()
    dataset = dataset_ops.Dataset.from_tensors(1).apply(
        optimization.assert_next(
            ["LatencyStats", "Map", "LatencyStats", "Prefetch",
             "LatencyStats"])).map(lambda x: x * x).prefetch(1).apply(
                 optimization.optimize(["latency_all_edges"])).apply(
                     stats_ops.set_stats_aggregator(stats_aggregator))
    iterator = dataset.make_initializable_iterator()
    get_next = iterator.get_next()
    summary_t = stats_aggregator.get_summary()

    with self.test_session() as sess:
      sess.run(iterator.initializer)
      self.assertEqual(1 * 1, sess.run(get_next))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)
      summary_str = sess.run(summary_t)
      self._assertSummaryHasCount(summary_str,
                                  "record_latency_TensorDataset/_1", 1)
      self._assertSummaryHasCount(summary_str, "record_latency_MapDataset/_4",
                                  1)
      self._assertSummaryHasCount(summary_str,
                                  "record_latency_PrefetchDataset/_6", 1)


if __name__ == "__main__":
  test.main()
