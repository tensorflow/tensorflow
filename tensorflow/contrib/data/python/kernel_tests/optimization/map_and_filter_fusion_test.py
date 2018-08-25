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
"""Tests for the MapAndFilterFusion optimization."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

from tensorflow.contrib.data.python.ops import optimization
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class MapAndFilterFusionTest(test.TestCase, parameterized.TestCase):

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

  @staticmethod
  def filter_functions():
    take_all = lambda x: constant_op.constant(True)
    is_zero = lambda x: math_ops.equal(x, 0)
    greater = lambda x: math_ops.greater(x + 5, 0)

    tests = []
    filters = [take_all, is_zero, greater]
    identity = lambda x: x
    for x, predicate_1 in enumerate(filters):
      for y, predicate_2 in enumerate(filters):
        tests.append(("mixed_{}_{}".format(x, y), identity,
                      [predicate_1, predicate_2]))
        for z, predicate_3 in enumerate(filters):
          tests.append(("mixed_{}_{}_{}".format(x, y, z), identity,
                        [predicate_1, predicate_2, predicate_3]))

    take_all_multiple = lambda x, y: constant_op.constant(True)
    # Multi output
    tests.append(("multiOne", lambda x: (x, x),
                  [take_all_multiple, take_all_multiple]))
    tests.append(("multiTwo", lambda x: (x, 2), [
        take_all_multiple,
        lambda x, y: math_ops.equal(x * math_ops.cast(y, dtypes.int64), 0)
    ]))
    return tuple(tests)

  @parameterized.named_parameters(*filter_functions.__func__())
  def testFilterFusion(self, map_function, predicates):
    dataset = dataset_ops.Dataset.range(5).apply(
        optimization.assert_next(["Map", "Filter",
                                  "Prefetch"])).map(map_function)
    for predicate in predicates:
      dataset = dataset.filter(predicate)

    dataset = dataset.prefetch(0).apply(
        optimization.optimize(["filter_fusion"]))
    iterator = dataset.make_one_shot_iterator()
    get_next = iterator.get_next()
    with self.test_session() as sess:
      for x in range(5):
        r = map_function(x)
        filtered = False
        for predicate in predicates:
          if isinstance(r, tuple):
            b = predicate(*r)  # Pass tuple as multiple arguments.
          else:
            b = predicate(r)
          if not sess.run(b):
            filtered = True
            break

        if not filtered:
          result = sess.run(get_next)
          self.assertAllEqual(r, result)
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)


if __name__ == "__main__":
  test.main()
