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
"""Tests for the private `_OptimizeDataset` transformation."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings

from absl.testing import parameterized
import numpy as np

from tensorflow.python.data.experimental.ops import batching
from tensorflow.python.data.experimental.ops import grouping
from tensorflow.python.data.experimental.ops import optimization
from tensorflow.python.data.experimental.ops import optimization_options
from tensorflow.python.data.experimental.ops import scan_ops
from tensorflow.python.data.experimental.ops import threadpool
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import test


def _generate_captured_refvar_test_cases():
  """Generates testcases.

  Returns:
    A list of tuples of (testcase_name, make_dataset_fn). make_dataset_fn takes
    a tf.Variable as input and creates a test dataset that uses that variable.
  """

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

  return [
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


@test_util.run_all_in_graph_and_eager_modes
class OptimizeDatasetTest(test_base.DatasetTestBase, parameterized.TestCase):

  def testOptimizationStatefulFunction(self):
    dataset = dataset_ops.Dataset.range(
        10).map(lambda _: random_ops.random_uniform([])).batch(10)
    dataset = dataset_ops._OptimizeDataset(dataset, [])
    get_next = self.getNext(dataset)
    self.evaluate(get_next())

  # TODO(b/117581999): Add eager coverage for the following tests.
  def testSkipEagerOptimizationLargeInputFromTensor(self):
    input_t = array_ops.placeholder(dtypes.int32, (None, None, None))
    dataset = dataset_ops.Dataset.from_tensors(input_t)
    dataset = dataset_ops._OptimizeDataset(dataset, [])
    iterator = dataset_ops.make_initializable_iterator(dataset)
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.cached_session() as sess:
      sess.run(init_op, {input_t: np.ones([512, 1024, 1025], np.int32)})
      self.evaluate(get_next)

  # TODO(b/117581999): Add eager coverage for the following tests.
  def testSkipEagerOptimizationLargeInputFromTensorSlices(self):
    input_t = array_ops.placeholder(dtypes.int32, (None, None, None, None))
    dataset = dataset_ops.Dataset.from_tensor_slices(input_t)
    dataset = dataset_ops._OptimizeDataset(dataset, [])
    iterator = dataset_ops.make_initializable_iterator(dataset)
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.cached_session() as sess:
      sess.run(init_op, {input_t: np.ones([1, 512, 1024, 1025], np.int32)})
      self.evaluate(get_next)

  def testOptimizationNestedDataset(self):

    def flat_map_fn(_):
      dataset = dataset_ops.Dataset.from_tensors(0)
      dataset = dataset.apply(optimization.assert_next(["MemoryCacheImpl"]))
      dataset = dataset.skip(0)  # Should be removed by noop elimination
      dataset = dataset.cache()
      return dataset

    dataset = dataset_ops.Dataset.range(1)
    dataset = dataset.flat_map(flat_map_fn)
    dataset = dataset_ops._OptimizeDataset(dataset, ["noop_elimination"])
    self.assertDatasetProduces(dataset, expected_output=[0])

  def testOptimizationNestedDatasetWithModifiedRetval(self):

    def flat_map_fn(_):
      dataset = dataset_ops.Dataset.from_tensors(0)
      dataset = dataset.apply(optimization.assert_next(["MapAndBatch"]))
      # Should be fused by map and batch fusion
      dataset = dataset.map(lambda x: x)
      dataset = dataset.batch(1)
      return dataset

    dataset = dataset_ops.Dataset.range(1)
    dataset = dataset.flat_map(flat_map_fn)

    # TODO(b/120558523): We use Options instead of _OptimizeDataset directly
    # here because of a bug with chaining _OptimizeDatasets when there are
    # nested dataset functions
    options = dataset_ops.Options()
    opt_options = optimization_options.OptimizationOptions()
    opt_options.map_and_batch_fusion = True
    options.experimental_optimization = opt_options
    dataset = dataset.with_options(options)
    self.assertDatasetProduces(dataset, expected_output=[[0]])

  def testOptimizationThreadPoolDataset(self):
    dataset = dataset_ops.Dataset.range(10).batch(10)

    dataset = threadpool.override_threadpool(
        dataset,
        threadpool.PrivateThreadPool(
            2, display_name="private_thread_pool_%d" % 2))

    dataset = dataset_ops._OptimizeDataset(dataset, [])
    self.assertDatasetProduces(
        dataset,
        expected_output=[list(range(10))],
        requires_initialization=True)

  def testOptimizationNonSerializable(self):
    dataset = dataset_ops.Dataset.from_tensors(0)
    dataset = dataset.apply(optimization.assert_next(["FiniteSkip"]))
    dataset = dataset.skip(0)  # Should not be removed by noop elimination
    dataset = dataset.apply(optimization.non_serializable())
    dataset = dataset.apply(optimization.assert_next(["MemoryCacheImpl"]))
    dataset = dataset.skip(0)  # Should be removed by noop elimination
    dataset = dataset.cache()
    dataset = dataset_ops._OptimizeDataset(dataset, ["noop_elimination"])
    self.assertDatasetProduces(dataset, expected_output=[0])

  def testOptimizationNonSerializableAsDirectInput(self):
    """Tests that non-serializable dataset can be OptimizeDataset's input."""
    dataset = dataset_ops.Dataset.from_tensors(0)
    dataset = dataset.apply(optimization.non_serializable())
    dataset = dataset_ops._OptimizeDataset(dataset, ["noop_elimination"])
    self.assertDatasetProduces(dataset, expected_output=[0])

  @parameterized.named_parameters(_generate_captured_refvar_test_cases())
  # Skip eager because RefVariables are not supported in eager mode.
  def testSkipEagerOptimizationWithCapturedRefVar(self, dataset_fn):
    """Tests that default optimizations are disabled with ref variables."""
    variable = variable_scope.get_variable(
        "v", initializer=0, use_resource=False)
    assign_op = variable.assign_add(1)

    unoptimized_dataset = dataset_fn(variable)

    options = dataset_ops.Options()
    opt_options = optimization_options.OptimizationOptions()
    opt_options.noop_elimination = True
    opt_options.map_and_batch_fusion = True
    options.experimental_optimization = opt_options
    optimized_dataset = unoptimized_dataset.with_options(options)

    # Check that warning is logged.
    warnings.simplefilter("always")
    with warnings.catch_warnings(record=True) as w:
      optimized_it = optimized_dataset.make_initializable_iterator()

    self.assertGreaterEqual(len(w), 1)
    expected = ("tf.data static optimizations are not compatible with "
                "tf.Variable. The following optimizations will be disabled: "
                "map_and_batch_fusion, noop_elimination. To enable "
                "optimizations, use resource variables instead by calling "
                "`tf.enable_resource_variables()` at the start of the program.")
    self.assertTrue(any([expected in str(warning) for warning in w]))

    # Check that outputs are the same in the optimized and unoptimized cases,
    # when the variable value is changing.
    unoptimized_it = unoptimized_dataset.make_initializable_iterator()
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

  def testOptimizationEnabledByDefault(self):
    """Tests that some optimizations are applied to datasets by default."""
    options = dataset_ops.Options()
    expected_optimizations = ["noop_elimination", "map_and_batch_fusion"]
    self.assertEqual(
        set(options._static_optimizations()), set(expected_optimizations))

  def testOptimizationDisableDefault(self):
    """Tests that we can disable all static optimizations enabled by default.

    If the `apply_default_optimizations` optimization options flag is False,
    only explicitly enabled optimizations will be applied.
    """
    options = dataset_ops.Options()
    opt_options = optimization_options.OptimizationOptions()
    opt_options.hoist_random_uniform = True
    opt_options.apply_default_optimizations = False
    options.experimental_optimization = opt_options
    expected_optimizations = ["hoist_random_uniform"]
    self.assertEqual(options._static_optimizations(), expected_optimizations)


if __name__ == "__main__":
  test.main()
