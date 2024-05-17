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
"""Tests for the static tf.data optimizations."""
import functools
import time

from absl.testing import parameterized
import numpy as np

from tensorflow.python.data.experimental.ops import batching
from tensorflow.python.data.experimental.ops import grouping
from tensorflow.python.data.experimental.ops import scan_ops
from tensorflow.python.data.experimental.ops import testing
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.framework import combinations
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


def _captured_refvar_test_combinations():

  def make_map_dataset(var):
    return dataset_ops.Dataset.from_tensors(0).map(lambda x: x + var)

  def make_flat_map_dataset(var):
    return dataset_ops.Dataset.from_tensors(
        0).flat_map(lambda _: dataset_ops.Dataset.from_tensors(var))

  def make_filter_dataset(var):
    return dataset_ops.Dataset.from_tensors(0).filter(lambda x: x < var)

  def make_map_and_batch_dataset(var):

    def map_fn(x):
      return x + var

    return dataset_ops.Dataset.from_tensors(0).apply(
        batching.map_and_batch(map_fn, 1))

  def make_group_by_reducer_dataset(var):
    reducer = grouping.Reducer(
        init_func=lambda _: 0,
        reduce_func=lambda x, y: x,
        finalize_func=lambda _: var)
    return dataset_ops.Dataset.range(5).apply(
        grouping.group_by_reducer(lambda x: x % 2, reducer))

  def make_group_by_window_dataset(var):

    def reduce_fn(key, bucket):
      del key, bucket
      return dataset_ops.Dataset.from_tensors(var)

    return dataset_ops.Dataset.from_tensors(0).repeat(10).apply(
        grouping.group_by_window(lambda _: 0, reduce_fn, 10))

  def make_scan_dataset(var):
    return dataset_ops.Dataset.from_tensors(0).apply(
        scan_ops.scan(
            0, lambda old_state, elem: (old_state + 1, elem + old_state + var)))

  cases = [
      # Core datasets
      ("Map", make_map_dataset),
      ("FlatMap", make_flat_map_dataset),
      ("Filter", make_filter_dataset),
      # Experimental datasets
      ("MapAndBatch", make_map_and_batch_dataset),
      ("GroupByReducer", make_group_by_reducer_dataset),
      ("GroupByWindow", make_group_by_window_dataset),
      ("Scan", make_scan_dataset)
  ]

  def reduce_fn(x, y):
    name, dataset_fn = y
    return x + combinations.combine(
        dataset_fn=combinations.NamedObject(name, dataset_fn))

  return functools.reduce(reduce_fn, cases, [])


class OptimizationTest(test_base.DatasetTestBase, parameterized.TestCase):

  @combinations.generate(test_base.default_test_combinations())
  def testOptimizationStatefulFunction(self):
    dataset = dataset_ops.Dataset.range(
        10).map(lambda _: random_ops.random_uniform([])).batch(10)
    options = options_lib.Options()
    options.experimental_optimization.apply_default_optimizations = False
    dataset = dataset.with_options(options)
    get_next = self.getNext(dataset)
    self.evaluate(get_next())

  # TODO(b/123354468)
  @combinations.generate(test_base.graph_only_combinations())
  def testOptimizationLargeInputFromTensor(self):
    input_t = array_ops.placeholder(dtypes.int32, (None, None, None))
    dataset = dataset_ops.Dataset.from_tensors(input_t)
    options = options_lib.Options()
    options.experimental_optimization.apply_default_optimizations = False
    dataset = dataset.with_options(options)
    iterator = dataset_ops.make_initializable_iterator(dataset)
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.cached_session() as sess:
      sess.run(init_op, {input_t: np.ones([512, 1024, 1025], np.int32)})
      self.evaluate(get_next)

  # TODO(b/123354468)
  @combinations.generate(test_base.graph_only_combinations())
  def testOptimizationLargeInputFromTensorSlices(self):
    input_t = array_ops.placeholder(dtypes.int32, (None, None, None, None))
    dataset = dataset_ops.Dataset.from_tensor_slices(input_t)
    options = options_lib.Options()
    options.experimental_optimization.apply_default_optimizations = False
    dataset = dataset.with_options(options)
    iterator = dataset_ops.make_initializable_iterator(dataset)
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.cached_session() as sess:
      sess.run(init_op, {input_t: np.ones([1, 512, 1024, 1025], np.int32)})
      self.evaluate(get_next)

  @combinations.generate(test_base.default_test_combinations())
  def testOptimizationNestedDataset(self):

    def flat_map_fn(_):
      dataset = dataset_ops.Dataset.from_tensors(0)
      dataset = dataset.apply(testing.assert_next(["MemoryCacheImpl"]))
      dataset = dataset.skip(0)  # Should be removed by noop elimination
      dataset = dataset.cache()
      return dataset

    dataset = dataset_ops.Dataset.range(1)
    dataset = dataset.flat_map(flat_map_fn)
    options = options_lib.Options()
    options.experimental_optimization.apply_default_optimizations = False
    options.experimental_optimization.noop_elimination = True
    dataset = dataset.with_options(options)
    self.assertDatasetProduces(dataset, expected_output=[0])

  @combinations.generate(test_base.default_test_combinations())
  def testOptimizationNestedDatasetWithModifiedRetval(self):

    def flat_map_fn(_):
      dataset = dataset_ops.Dataset.from_tensors(0)
      dataset = dataset.apply(testing.assert_next(["MapAndBatch"]))
      # Should be fused by map and batch fusion
      dataset = dataset.map(lambda x: x)
      dataset = dataset.batch(1)
      return dataset

    dataset = dataset_ops.Dataset.range(1)
    dataset = dataset.flat_map(flat_map_fn)

    options = options_lib.Options()
    options.experimental_optimization.apply_default_optimizations = False
    options.experimental_optimization.map_and_batch_fusion = True
    dataset = dataset.with_options(options)
    self.assertDatasetProduces(dataset, expected_output=[[0]])

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(autotune=[True, False, None]),
          combinations.combine(map_parallelization=[True, False, None])))
  def testOptimizationMapParallelization(self, autotune, map_parallelization):
    dataset = dataset_ops.Dataset.range(5)
    if autotune is not False and map_parallelization is not False:  # pylint: disable=g-bool-id-comparison
      dataset = dataset.apply(testing.assert_next(["ParallelMap"]))
    else:
      dataset = dataset.apply(testing.assert_next(["Map"]))
    dataset = dataset.map(lambda x: x + 1)

    options = options_lib.Options()
    if autotune is not None:
      options.autotune.enabled = autotune
    if map_parallelization is not None:
      options.experimental_optimization.map_parallelization = (
          map_parallelization)
    dataset = dataset.with_options(options)

    self.assertDatasetProduces(dataset, expected_output=list(range(1, 6)))

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(existing_prefetch=[True, False]),
                         combinations.combine(autotune=[True, False]),
                         combinations.combine(inject_prefetch=[True, False])))
  def testOptimizationInjectPrefetch(self, existing_prefetch, autotune,
                                     inject_prefetch):
    dataset = dataset_ops.Dataset.range(5)
    dataset = dataset.map(
        lambda x: x + 1, num_parallel_calls=dataset_ops.AUTOTUNE)
    dataset = dataset.batch(1)
    if existing_prefetch:
      dataset = dataset.prefetch(1)
    if autotune and inject_prefetch and not existing_prefetch:
      dataset = dataset.apply(testing.assert_next(["Prefetch", "Root"]))
    else:
      dataset = dataset.apply(testing.assert_next(["Root"]))

    options = options_lib.Options()
    options.autotune.enabled = autotune
    options.experimental_optimization.map_and_batch_fusion = False
    if not inject_prefetch:
      options.experimental_optimization.inject_prefetch = False
    dataset = dataset.with_options(options)

    self.assertDatasetProduces(dataset, expected_output=[np.array([x]) for x in
                                                         range(1, 6)])

  # Reference variables are not supported in eager mode.
  @combinations.generate(
      combinations.times(test_base.graph_only_combinations(),
                         _captured_refvar_test_combinations()))
  def testOptimizationWithCapturedRefVar(self, dataset_fn):
    """Tests that default optimizations are disabled with ref variables."""
    variable = variable_scope.get_variable(
        "v", initializer=0, use_resource=False)
    assign_op = variable.assign_add(1)
    unoptimized_dataset = dataset_fn(variable)

    options = options_lib.Options()
    options.experimental_optimization.apply_default_optimizations = False
    options.experimental_optimization.noop_elimination = True
    options.experimental_optimization.map_and_batch_fusion = True
    options.experimental_warm_start = False
    optimized_dataset = unoptimized_dataset.with_options(options)
    optimized_it = dataset_ops.make_initializable_iterator(optimized_dataset)

    # Check that outputs are the same in the optimized and unoptimized cases,
    # when the variable value is changing.
    unoptimized_it = dataset_ops.make_initializable_iterator(
        unoptimized_dataset)
    with ops.control_dependencies([assign_op]):
      unoptimized_output = unoptimized_it.get_next()
      optimized_output = optimized_it.get_next()

    self.evaluate(variable.initializer)
    self.evaluate((unoptimized_it.initializer, optimized_it.initializer))
    while True:
      try:
        unoptimized, optimized = self.evaluate((unoptimized_output,
                                                optimized_output))
        self.assertEqual(unoptimized, optimized)
      except errors.OutOfRangeError:
        break

  @combinations.generate(
      combinations.times(
          test_base.eager_only_combinations(),
          combinations.combine(warm_start=[True, False]),
      )
  )
  def testOptimizationWarmStart(self, warm_start):
    dataset = dataset_ops.Dataset.range(10)
    counter = variables.Variable(0)

    def update_counter(x):
      counter.assign_add(1)
      return x

    options = options_lib.Options()
    options.experimental_optimization.apply_default_optimizations = False
    if warm_start:
      options.experimental_warm_start = True
    else:
      options.experimental_warm_start = False
    dataset = dataset.with_options(options)
    dataset = dataset.map(update_counter).prefetch(10)
    unused_iter = iter(dataset)

    if warm_start:
      for sleep_time_secs in [0.1, 0.2, 0.5, 2, 5, 10]:
        if counter.numpy() == 0:
          time.sleep(sleep_time_secs)
        else:
          break
      self.assertGreater(counter.numpy(), 0)
    else:
      self.assertEqual(counter.numpy(), 0)


if __name__ == "__main__":
  test.main()
