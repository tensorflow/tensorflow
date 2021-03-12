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

import functools
import os
import warnings

from absl.testing import parameterized
import numpy as np

from tensorflow.python.data.experimental.ops import batching
from tensorflow.python.data.experimental.ops import grouping
from tensorflow.python.data.experimental.ops import optimization
from tensorflow.python.data.experimental.ops import optimization_options
from tensorflow.python.data.experimental.ops import scan_ops
from tensorflow.python.data.experimental.ops import testing
from tensorflow.python.data.experimental.ops import threadpool
from tensorflow.python.data.kernel_tests import checkpoint_test_base
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import combinations
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variable_scope
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


def _disable_intra_op_parallelism_test_combinations():

  def make_tensor_dataset():
    return dataset_ops.Dataset.from_tensors(42)

  def make_map_dataset():
    return dataset_ops.Dataset.from_tensors(42).map(lambda x: x + 1)

  cases = [
      ("FromTensors", make_tensor_dataset, [42]),
      ("Map", make_map_dataset, [43]),
  ]

  def reduce_fn(x, y):
    name, dataset_fn, expected_output = y
    return x + combinations.combine(
        dataset_fn=combinations.NamedObject(name, dataset_fn),
        expected_output=[expected_output])

  return functools.reduce(reduce_fn, cases, [])


class OptimizeDatasetTest(test_base.DatasetTestBase, parameterized.TestCase):

  @combinations.generate(test_base.default_test_combinations())
  def testOptimizationStatefulFunction(self):
    dataset = dataset_ops.Dataset.range(
        10).map(lambda _: random_ops.random_uniform([])).batch(10)
    options = dataset_ops.Options()
    options.experimental_optimization.apply_default_optimizations = False
    dataset = dataset.with_options(options)
    get_next = self.getNext(dataset)
    self.evaluate(get_next())

  # TODO(b/123902160)
  @combinations.generate(test_base.graph_only_combinations())
  def testOptimizationLargeInputFromTensor(self):
    input_t = array_ops.placeholder(dtypes.int32, (None, None, None))
    dataset = dataset_ops.Dataset.from_tensors(input_t)
    options = dataset_ops.Options()
    options.experimental_optimization.apply_default_optimizations = False
    dataset = dataset.with_options(options)
    iterator = dataset_ops.make_initializable_iterator(dataset)
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.cached_session() as sess:
      sess.run(init_op, {input_t: np.ones([512, 1024, 1025], np.int32)})
      self.evaluate(get_next)

  # TODO(b/123902160)
  @combinations.generate(test_base.graph_only_combinations())
  def testOptimizationLargeInputFromTensorSlices(self):
    input_t = array_ops.placeholder(dtypes.int32, (None, None, None, None))
    dataset = dataset_ops.Dataset.from_tensor_slices(input_t)
    options = dataset_ops.Options()
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
    options = dataset_ops.Options()
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

    options = dataset_ops.Options()
    options.experimental_optimization.apply_default_optimizations = False
    options.experimental_optimization.map_and_batch_fusion = True
    dataset = dataset.with_options(options)
    self.assertDatasetProduces(dataset, expected_output=[[0]])

  @combinations.generate(test_base.default_test_combinations())
  def testOptimizationDoubleOptimizeDatasetNested(self):
    def flat_map_fn(_):
      dataset = dataset_ops.Dataset.from_tensors(0)
      dataset = dataset.apply(testing.assert_next(["MapAndBatch"]))
      dataset = dataset.skip(0)
      # Should be fused by map and batch fusion
      dataset = dataset.map(lambda x: x)
      dataset = dataset.batch(1)
      return dataset

    dataset = dataset_ops.Dataset.from_tensors(0)
    dataset = dataset.flat_map(flat_map_fn)
    dataset = dataset_ops._OptimizeDataset(dataset, ["map_and_batch_fusion"],
                                           [], [])
    dataset = dataset_ops._OptimizeDataset(dataset, ["noop_elimination"], [],
                                           [])

    self.assertDatasetProduces(dataset, expected_output=[[0]])

  @combinations.generate(test_base.default_test_combinations())
  def testOptimizationDifferentOrderOptionsCompareEqual(self):
    with ops.Graph().as_default() as first_graph:
      dataset = dataset_ops.Dataset.from_tensors(0)
      dataset_ops._OptimizeDataset(dataset,
                                   ["map_and_batch_fusion", "noop_elimination"],
                                   [], [])

    with ops.Graph().as_default() as second_graph:
      dataset = dataset_ops.Dataset.from_tensors(0)
      dataset_ops._OptimizeDataset(dataset,
                                   ["noop_elimination", "map_and_batch_fusion"],
                                   [], [])

    self.assertEqual(first_graph.as_graph_def(), second_graph.as_graph_def())

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          _disable_intra_op_parallelism_test_combinations(),
          combinations.combine(apply_autotune=[None, True, False])))
  def testOptimizationDisableIntraOpParallelism(self, dataset_fn,
                                                expected_output,
                                                apply_autotune):
    dataset = dataset_fn()
    dataset = dataset.apply(testing.assert_next(["MaxIntraOpParallelism"]))
    if apply_autotune is not None:
      options = dataset_ops.Options()
      options.experimental_optimization.autotune = apply_autotune
      dataset = dataset.with_options(options)

    self.assertDatasetProduces(dataset, expected_output=expected_output)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(autotune=False, autotune_buffers=False) +
          combinations.combine(autotune=True, autotune_buffers=False) +
          combinations.combine(autotune=True, autotune_buffers=True),
          combinations.combine(set_env=[False, True])))
  def testOptimizationEnableGradientDescent(self, autotune, autotune_buffers,
                                            set_env):
    if set_env:
      os.environ["TF_DATA_EXPERIMENT_OPT_IN"] = "enable_gradient_descent"
      os.environ["TF_JOB_NAME"] = "test_job"

    dataset = dataset_ops.Dataset.range(5)
    dataset = dataset.prefetch(buffer_size=-1)
    dataset = dataset.map(lambda x: x + 1, num_parallel_calls=2)
    dataset = dataset.map(lambda x: x + 1, num_parallel_calls=-1)
    dataset = dataset.prefetch(buffer_size=3)
    dataset = dataset.map(lambda x: x + 1, num_parallel_calls=-1)
    dataset = dataset.prefetch(buffer_size=1)

    options = dataset_ops.Options()
    options.experimental_optimization.autotune = autotune
    options.experimental_optimization.autotune_buffers = autotune_buffers
    dataset = dataset.with_options(options)

    self.assertDatasetProduces(dataset, expected_output=list(range(3, 8)))

    if set_env:
      del os.environ["TF_DATA_EXPERIMENT_OPT_IN"]
      del os.environ["TF_JOB_NAME"]

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

    options = dataset_ops.Options()
    if autotune is not None:
      options.experimental_optimization.autotune = autotune
    if map_parallelization is not None:
      options.experimental_optimization.map_parallelization = (
          map_parallelization)
    dataset = dataset.with_options(options)

    self.assertDatasetProduces(dataset, expected_output=list(range(1, 6)))

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(set_env=[True, False])))
  def testOptimizationUsePrivateThreadPool(self, set_env):
    if set_env:
      os.environ["TF_DATA_EXPERIMENT_OPT_IN"] = "use_private_thread_pool"
      os.environ["TF_JOB_NAME"] = "test_job"

    dataset = dataset_ops.Dataset.range(6)
    if set_env:
      dataset = dataset.apply(
          testing.assert_next(
              ["MaxIntraOpParallelism", "PrivateThreadPool", "Model"]))
    else:
      dataset = dataset.apply(
          testing.assert_next(["MaxIntraOpParallelism", "Model"]))

    self.assertDatasetProduces(dataset, expected_output=list(range(6)))

    if set_env:
      del os.environ["TF_DATA_EXPERIMENT_OPT_IN"]
      del os.environ["TF_JOB_NAME"]

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(autotune=False, autotune_buffers=False) +
          combinations.combine(autotune=True, autotune_buffers=False) +
          combinations.combine(autotune=True, autotune_buffers=True),
          combinations.combine(first_buffer_sizes=[(1, -1, -1, 4),
                                                   (2, -1, 3, -1),
                                                   (2, 1, -1, -1)]),
          combinations.combine(second_buffer_sizes=[(1, -1, -1, 4),
                                                    (2, -1, 3, -1),
                                                    (2, 1, -1, -1)]))
  )
  def testOptimizationAutotuneBuffers(self, autotune, autotune_buffers,
                                      first_buffer_sizes, second_buffer_sizes):
    dataset = dataset_ops.Dataset.range(10)
    for buffer_size in first_buffer_sizes:
      dataset = dataset.prefetch(buffer_size=buffer_size)
    dataset = dataset.map(lambda x: x + 1)
    for buffer_size in second_buffer_sizes:
      dataset = dataset.prefetch(buffer_size=buffer_size)
    options = dataset_ops.Options()
    options.experimental_optimization.autotune = autotune
    options.experimental_optimization.autotune_buffers = autotune_buffers
    dataset = dataset.with_options(options)
    self.assertDatasetProduces(dataset, expected_output=list(range(1, 11)))

  @combinations.generate(test_base.default_test_combinations())
  def testOptimizationThreadPoolDataset(self):
    dataset = dataset_ops.Dataset.range(10).batch(10)

    dataset = threadpool.override_threadpool(
        dataset,
        threadpool.PrivateThreadPool(
            2, display_name="private_thread_pool_%d" % 2))

    options = dataset_ops.Options()
    options.experimental_optimization.apply_default_optimizations = False
    dataset = dataset.with_options(options)
    self.assertDatasetProduces(
        dataset,
        expected_output=[list(range(10))],
        requires_initialization=True)

  # Reference variables are not supported in eager mode.
  @combinations.generate(
      combinations.times(test_base.graph_only_combinations(),
                         _captured_refvar_test_combinations()))
  def testOptimizationWithCapturedRefVar(self, dataset_fn):
    """Tests that default optimizations are disabled with ref variables."""
    variable = variable_scope.get_variable(
        "v", initializer=0, use_resource=False)
    assign_op = variable.assign_add(1)

    # Check that warning is logged.
    warnings.simplefilter("always")
    with warnings.catch_warnings(record=True) as w:
      unoptimized_dataset = dataset_fn(variable)

      options = dataset_ops.Options()
      options.experimental_optimization.apply_default_optimizations = False
      options.experimental_optimization.noop_elimination = True
      options.experimental_optimization.map_and_batch_fusion = True
      optimized_dataset = unoptimized_dataset.with_options(options)
      optimized_it = dataset_ops.make_initializable_iterator(optimized_dataset)

    self.assertGreaterEqual(len(w), 1)
    graph_rewrites = options._graph_rewrites()
    expected = (
        "tf.data graph rewrites are not compatible with "
        "tf.Variable. The following rewrites will be disabled: %s."
        " To enable rewrites, use resource variables instead by "
        "calling `tf.enable_resource_variables()` at the start of the "
        "program." %
        (", ".join(graph_rewrites.enabled + graph_rewrites.default)))
    self.assertTrue(any(expected in str(warning) for warning in w))

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

  @combinations.generate(test_base.default_test_combinations())
  def testOptimizationDefault(self):
    """Tests the optimization settings by default."""
    options = dataset_ops.Options()
    expected_optimizations_enabled = []
    expected_optimizations_disabled = []
    expected_optimizations_default = [
        "map_and_batch_fusion",
        "map_parallelization",
        "noop_elimination",
        "shuffle_and_repeat_fusion",
    ]
    graph_rewrites = options._graph_rewrites()
    self.assertEqual(set(graph_rewrites.enabled),
                     set(expected_optimizations_enabled))
    self.assertEqual(set(graph_rewrites.disabled),
                     set(expected_optimizations_disabled))
    self.assertEqual(set(graph_rewrites.default),
                     set(expected_optimizations_default))

    options.experimental_optimization.apply_default_optimizations = True
    graph_rewrites = options._graph_rewrites()
    self.assertEqual(set(graph_rewrites.enabled),
                     set(expected_optimizations_enabled))
    self.assertEqual(set(graph_rewrites.disabled),
                     set(expected_optimizations_disabled))
    self.assertEqual(set(graph_rewrites.default),
                     set(expected_optimizations_default))

    options.experimental_optimization.apply_default_optimizations = False
    expected_optimizations_default = []
    graph_rewrites = options._graph_rewrites()
    self.assertEqual(set(graph_rewrites.enabled),
                     set(expected_optimizations_enabled))
    self.assertEqual(set(graph_rewrites.disabled),
                     set(expected_optimizations_disabled))
    self.assertEqual(set(graph_rewrites.default),
                     set(expected_optimizations_default))

  @combinations.generate(test_base.default_test_combinations())
  def testOptimizationEnabled(self):
    """Tests the optimization settings by enabling all."""
    options = dataset_ops.Options()
    options.experimental_optimization.filter_fusion = True
    options.experimental_optimization.filter_with_random_uniform_fusion = True
    options.experimental_optimization.hoist_random_uniform = True
    options.experimental_optimization.map_and_batch_fusion = True
    options.experimental_optimization.map_and_filter_fusion = True
    options.experimental_optimization.map_parallelization = True
    options.experimental_optimization.map_fusion = True
    options.experimental_optimization.noop_elimination = True
    options.experimental_optimization.parallel_batch = True
    options.experimental_optimization.shuffle_and_repeat_fusion = True
    options.experimental_optimization.map_vectorization.enabled = True
    options.experimental_optimization.autotune_buffers = True
    options.experimental_deterministic = False
    options.experimental_stats.latency_all_edges = True
    options.experimental_slack = True

    expected_optimizations_enabled = [
        "filter_fusion",
        "filter_with_random_uniform_fusion",
        "hoist_random_uniform",
        "map_and_batch_fusion",
        "map_and_filter_fusion",
        "map_parallelization",
        "map_fusion",
        "noop_elimination",
        "parallel_batch",
        "shuffle_and_repeat_fusion",
        "map_vectorization",
        "autotune_buffer_sizes",
        "make_sloppy",
        "latency_all_edges",
        "slack",
        "disable_prefetch_legacy_autotune",
    ]
    expected_optimizations_disabled = []
    expected_optimizations_default = []
    graph_rewrites = options._graph_rewrites()
    self.assertEqual(set(graph_rewrites.enabled),
                     set(expected_optimizations_enabled))
    self.assertEqual(set(graph_rewrites.disabled),
                     set(expected_optimizations_disabled))
    self.assertEqual(set(graph_rewrites.default),
                     set(expected_optimizations_default))

  @combinations.generate(test_base.default_test_combinations())
  def testOptimizationDisabled(self):
    """Tests the optimization settings by disabling all."""
    options = dataset_ops.Options()
    options.experimental_optimization.filter_fusion = False
    options.experimental_optimization.filter_with_random_uniform_fusion = False
    options.experimental_optimization.hoist_random_uniform = False
    options.experimental_optimization.map_and_batch_fusion = False
    options.experimental_optimization.map_and_filter_fusion = False
    options.experimental_optimization.map_parallelization = False
    options.experimental_optimization.map_fusion = False
    options.experimental_optimization.noop_elimination = False
    options.experimental_optimization.parallel_batch = False
    options.experimental_optimization.shuffle_and_repeat_fusion = False
    options.experimental_optimization.map_vectorization.enabled = False
    options.experimental_optimization.autotune = False
    options.experimental_deterministic = True
    options.experimental_stats.latency_all_edges = False
    options.experimental_slack = False

    expected_optimizations_enabled = []
    expected_optimizations_disabled = [
        "filter_fusion",
        "filter_with_random_uniform_fusion",
        "hoist_random_uniform",
        "map_and_batch_fusion",
        "map_and_filter_fusion",
        "map_parallelization",
        "map_fusion",
        "noop_elimination",
        "parallel_batch",
        "shuffle_and_repeat_fusion",
        "map_vectorization",
        "autotune_buffer_sizes",
        "make_sloppy",
        "latency_all_edges",
        "slack",
        "disable_prefetch_legacy_autotune",
    ]
    expected_optimizations_default = []
    graph_rewrites = options._graph_rewrites()
    self.assertEqual(set(graph_rewrites.enabled),
                     set(expected_optimizations_enabled))
    self.assertEqual(set(graph_rewrites.disabled),
                     set(expected_optimizations_disabled))
    self.assertEqual(set(graph_rewrites.default),
                     set(expected_optimizations_default))

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(autotune=[True, False, None]),
          combinations.combine(autotune_buffers=[True, False, None])))
  def testAutotuningSettings(self, autotune, autotune_buffers):
    options = dataset_ops.Options()
    if autotune is not None:
      options.experimental_optimization.autotune = autotune
    if autotune_buffers is not None:
      options.experimental_optimization.autotune_buffers = autotune_buffers

    # Check defaults
    autotune_settings = options._autotune_settings()
    autotune_val = autotune_settings[0]
    autotune_buffers_val = options.experimental_optimization._autotune_buffers()

    if autotune is not False:  # pylint: disable=g-bool-id-comparison
      self.assertTrue(autotune_val)
    else:
      self.assertFalse(autotune_val)
    if autotune_buffers is True:  # pylint: disable=g-bool-id-comparison
      self.assertTrue(autotune_buffers_val)
    else:
      self.assertFalse(autotune_buffers_val)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(autotune_buffers=[True, False, None])))
  def testAutotuneBuffersSettings(self, autotune_buffers):
    options = dataset_ops.Options()
    if autotune_buffers is not None:
      options.experimental_optimization.autotune_buffers = autotune_buffers

    graph_rewrites = options._graph_rewrites()
    autotune_settings = options._autotune_settings()
    algorithm = autotune_settings[1]

    if autotune_buffers is True:  # pylint: disable=g-bool-id-comparison
      self.assertIn("autotune_buffer_sizes", graph_rewrites.enabled)
      self.assertIn("disable_prefetch_legacy_autotune", graph_rewrites.enabled)
      self.assertEqual(algorithm,
                       optimization_options._AutotuneAlgorithm.GRADIENT_DESCENT)
    else:
      self.assertNotIn("autotune_buffer_sizes", graph_rewrites.enabled)
      self.assertNotIn("disable_prefetch_legacy_autotune",
                       graph_rewrites.enabled)
      self.assertEqual(algorithm,
                       optimization_options._AutotuneAlgorithm.HILL_CLIMB)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(set_budget=[True, False]),
      ))
  def testResourceBudgets(self, set_budget):
    options = dataset_ops.Options()
    if set_budget:
      options.experimental_optimization.autotune_cpu_budget = 1000
      options.experimental_optimization.autotune_ram_budget = 999999999

    autotune_settings = options._autotune_settings()
    cpu_budget = autotune_settings[2]
    ram_budget = autotune_settings[3]

    if set_budget:
      self.assertEqual(cpu_budget, 1000)
      self.assertEqual(ram_budget, 999999999)
    else:
      self.assertEqual(cpu_budget, 0)
      self.assertEqual(ram_budget, 0)


class OptimizeDatasetCheckpointTest(checkpoint_test_base.CheckpointTestBase,
                                    parameterized.TestCase):

  @combinations.generate(test_base.default_test_combinations())
  def testCore(self):

    def build_dataset(num_elements, batch_size):
      return dataset_ops.Dataset.range(num_elements).map(lambda x: x * x).batch(
          batch_size).apply(
              optimization.optimize(["map_and_batch_fusion"], None, None))

    self.run_core_tests(lambda: build_dataset(200, 10), 20)

  @combinations.generate(test_base.default_test_combinations())
  def testWithNewFunction(self):
    """Tests that optimized datasets with new functions work."""

    def build_dataset():
      dataset = dataset_ops.Dataset.range(100)
      dataset = dataset.map(lambda x: x)
      dataset = dataset.batch(5)
      # map_vectorization adds a new vectorized function to the function
      # library.
      dataset = dataset.apply(
          optimization.optimize(["map_vectorization"], None, None))
      return dataset

    self.run_core_tests(build_dataset, 20)


if __name__ == "__main__":
  test.main()
