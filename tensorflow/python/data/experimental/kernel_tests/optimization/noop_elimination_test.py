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
"""Tests for the `NoopElimination` optimization."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

from absl.testing import parameterized

from tensorflow.python.data.experimental.ops import testing
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import combinations
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import logging_ops
from tensorflow.python.platform import test


def _test_combinations():

  def make_range():
    return dataset_ops.Dataset.range(10)

  def fn_with_side_effect(arg):
    logging_ops.print_v2(arg)
    return arg

  # Test case for map function with capture args
  def apply_map_with_capture(ds):
    const = constant_op.constant(-1, dtype=dtypes.int64)
    return ds.map(lambda x: (x, const))

  # Test case for map functions with multiple components
  def apply_map_with_multiple_components(ds):
    ds = ds.map(lambda x: (x, x), num_parallel_calls=2)  # Not eliminated
    return ds.map(lambda x, y: (x, y))  # Eliminated

  parallel_map_name = "ParallelMap"

  cases = [
      ("Skip0", lambda ds: ds.skip(0), None),
      ("SkipN", lambda ds: ds.skip(5), "FiniteSkip"),
      ("Repeat1", lambda ds: ds.repeat(1), None),
      ("RepeatN", lambda ds: ds.repeat(10), "FiniteRepeat"),
      ("Prefetch0", lambda ds: ds.prefetch(0), None),
      ("PrefetchN", lambda ds: ds.prefetch(1), "Prefetch"),
      ("Take-1", lambda ds: ds.take(-1), None),
      ("TakeN", lambda ds: ds.take(2), "FiniteTake"),
      ("MapIdentity", lambda ds: ds.map(lambda x: x), None),
      ("MapNonIdentity", lambda ds: ds.map(lambda x: x * 2), "Map"),
      ("MapWithSideEffect", lambda ds: ds.map(fn_with_side_effect), "Map"),
      ("MapWithCapture", apply_map_with_capture, "Map"),
      ("MapWithMultipleComponents", apply_map_with_multiple_components,
       parallel_map_name),
      ("MapRestructure", lambda ds: ds.map(lambda x: {"value": x}), ""),
      ("PMapIdentity", lambda ds: ds.map(lambda x: x, num_parallel_calls=2),
       None),
      ("PMapNonIdentity",
       lambda ds: ds.map(lambda x: x * 2, num_parallel_calls=2),
       parallel_map_name),
      ("Shard1", lambda ds: ds.shard(1, 0), None),
      ("ShardN", lambda ds: ds.shard(2, 0), "Shard"),
  ]

  def reduce_fn(result, case):
    name, transformation, expected = case
    return result + combinations.combine(
        init_dataset_fn=make_range,
        transformation=combinations.NamedObject(name, transformation),
        expected_name=expected)

  test_combinations = functools.reduce(reduce_fn, cases, [])

  return test_combinations


class NoopEliminationTest(test_base.DatasetTestBase, parameterized.TestCase):

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         _test_combinations()))
  def testNoopElimination(self, init_dataset_fn, transformation, expected_name):
    """Runs a noop elimination test case.

    Args:
      init_dataset_fn: Function to create the initial dataset
      transformation: Transformation to apply
      expected_name: Name of the transformation if it is not eliminated
    """
    dataset = init_dataset_fn()

    if expected_name:
      dataset = dataset.apply(
          testing.assert_next([expected_name, "FiniteTake"]))
    else:
      dataset = dataset.apply(testing.assert_next(["FiniteTake"]))

    dataset = dataset.apply(transformation)
    dataset = dataset.take(1)
    options = dataset_ops.Options()
    options.experimental_optimization.apply_default_optimizations = False
    options.experimental_optimization.noop_elimination = True
    dataset = dataset.with_options(options)

    # Run the first iteration for the side effect of checking the assertion.
    get_next = self.getNext(dataset)
    self.evaluate(get_next())


if __name__ == "__main__":
  test.main()
