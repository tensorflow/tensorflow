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
"""Tests for the MapVectorization optimization."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

from absl.testing import parameterized
import numpy as np

from tensorflow.contrib.data.python.kernel_tests import test_utils
from tensorflow.contrib.data.python.ops import optimization
from tensorflow.python.client import session
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class MapVectorizationTest(test_utils.DatasetTestBase, parameterized.TestCase):

  def _get_test_datasets(self,
                         base_dataset,
                         map_fn,
                         num_parallel_calls=None,
                         expect_optimized=True):
    """Given base dataset and map fn, creates test datasets.

    Returns a tuple of (unoptimized, dataset, optimized dataset). The
    unoptimized dataset has the assertion that Batch follows Map. The optimized
    dataset has the assertion that Map follows Batch, and has the
    "map_vectorization" optimization applied.

    Args:
      base_dataset: Input dataset to map->batch
      map_fn: Map function to use
      num_parallel_calls: (Optional.) num_parallel_calls argument for map
      expect_optimized: (Optional.) Whether we expect the optimization to take
        place, in which case we will assert that Batch is followed by Map,
        otherwise Map followed by Batch. Defaults to True.

    Returns:
      Tuple of (unoptimized dataset, optimized dataset).
    """
    map_node_name = "Map" if num_parallel_calls is None else "ParallelMap"
    batch_size = 100

    def _make_dataset(node_names):
      return base_dataset.apply(optimization.assert_next(node_names)).map(
          map_fn, num_parallel_calls=num_parallel_calls).batch(batch_size)

    unoptimized = _make_dataset([map_node_name, "Batch"])
    optimized = _make_dataset(["Batch", map_node_name] if expect_optimized else
                              [map_node_name, "Batch"]).apply(
                                  optimization.optimize(["map_vectorization"]))

    return unoptimized, optimized

  @parameterized.named_parameters(
      ("Basic", lambda x: (x, x + 1), None),
      ("Parallel", lambda x: (x, x + 1), 12),
      ("Gather", lambda x: array_ops.gather(x, 0), 12),
  )
  def testOptimization(self, map_fn, num_parallel_calls):
    base_dataset = dataset_ops.Dataset.from_tensor_slices([[1, 2],
                                                           [3, 4]]).repeat(5)
    unoptimized, optimized = self._get_test_datasets(base_dataset, map_fn,
                                                     num_parallel_calls)
    self._assert_datasets_equal(unoptimized, optimized)

  def testOptimizationBadMapFn(self):
    # Test map functions that give an error
    def map_fn(x):
      # x has leading dimension 5, this will raise an error
      return array_ops.gather(x, 10)

    base_dataset = dataset_ops.Dataset.range(5).repeat(5).batch(
        5, drop_remainder=True)
    _, optimized = self._get_test_datasets(base_dataset, map_fn)
    nxt = optimized.make_one_shot_iterator().get_next()
    with self.assertRaisesRegexp(errors.InvalidArgumentError,
                                 r"indices = 10 is not in \[0, 5\)"):
      self.evaluate(nxt)

  def testOptimizationWithCapturedInputs(self):
    # Tests that vectorization works with captured inputs
    def map_fn(x):
      return x + y

    y = constant_op.constant(1, shape=(2,))
    base_dataset = dataset_ops.Dataset.from_tensor_slices([[1, 2],
                                                           [3, 4]]).repeat(5)
    # TODO(rachelim): when this optimization works, turn on expect_optimized
    unoptimized, optimized = self._get_test_datasets(
        base_dataset, map_fn, expect_optimized=False)
    self._assert_datasets_equal(optimized, unoptimized)

  def testOptimizationIgnoreStateful(self):

    def map_fn(x):
      with ops.control_dependencies([check_ops.assert_equal(x, 0)]):
        return array_ops.identity(x)

    base_dataset = dataset_ops.Dataset.from_tensor_slices([[1, 2],
                                                           [3, 4]]).repeat(5)
    unoptimized, optimized = self._get_test_datasets(
        base_dataset, map_fn, expect_optimized=False)
    self._assert_datasets_raise_same_error(
        unoptimized, optimized, errors.InvalidArgumentError,
        [("OneShotIterator", "OneShotIterator_1", 1),
         ("IteratorGetNext", "IteratorGetNext_1", 1)])

  def testOptimizationIgnoreRagged(self):
    # Make sure we ignore inputs that might not be uniformly sized
    def map_fn(x):
      return array_ops.gather(x, 0)

    # output_shape = (?,)
    base_dataset = dataset_ops.Dataset.range(20).batch(3, drop_remainder=False)
    unoptimized, optimized = self._get_test_datasets(
        base_dataset, map_fn, expect_optimized=False)
    self._assert_datasets_equal(unoptimized, optimized)

  def testOptimizationIgnoreRaggedMap(self):
    # Don't optimize when the output of the map fn shapes are unknown.
    def map_fn(x):
      return array_ops.tile(x, x)

    base_dataset = dataset_ops.Dataset.range(20).batch(1, drop_remainder=True)
    unoptimized, optimized = self._get_test_datasets(
        base_dataset, map_fn, expect_optimized=False)
    self._assert_datasets_raise_same_error(
        unoptimized, optimized, errors.InvalidArgumentError,
        [("OneShotIterator", "OneShotIterator_1", 1),
         ("IteratorGetNext", "IteratorGetNext_1", 1)])


class MapVectorizationBenchmark(test.Benchmark):
  # TODO(rachelim): Add a benchmark for more expensive transformations, such as
  # vgg_preprocessing.

  def _run(self, x, num_iters=100, name=None):
    deltas = []
    with session.Session() as sess:
      for _ in range(5):
        # Warm up session...
        sess.run(x)
      for _ in range(num_iters):
        start = time.time()
        sess.run(x)
        end = time.time()
        deltas.append(end - start)
    median_time = np.median(deltas)
    self.report_benchmark(iters=num_iters, wall_time=median_time, name=name)
    return median_time

  def benchmark_CheapFns(self):

    input_sizes = [(10, 10, 3), (10, 100, 300)]
    batch_size = 1000
    for input_size in input_sizes:
      input_dataset = dataset_ops.Dataset.from_tensor_slices(
          (np.random.rand(*input_size), np.random.rand(*input_size))).repeat()
      for map_fn, str_id in self._get_known_cheap_fns():
        self._compare(input_dataset, map_fn, batch_size, input_size, str_id)

  def _compare(self, input_dataset, map_fn, batch_size, input_size, str_id):
    num_elems = np.prod(input_size)
    name_template = "{}__batch_size_{}_input_size_{}_{}"
    unoptimized = input_dataset.map(map_fn).batch(batch_size)
    unoptimized_op = unoptimized.make_one_shot_iterator().get_next()

    optimized = unoptimized.apply(optimization.optimize(["map_vectorization"]))
    optimized_op = optimized.make_one_shot_iterator().get_next()

    unoptimized_time = self._run(
        unoptimized_op,
        name=name_template.format(str_id, batch_size, num_elems, "unoptimized"))
    optimized_time = self._run(
        optimized_op,
        name=name_template.format(str_id, batch_size, num_elems, "optimized"))

    print("Batch size: {}\n"
          "Input size: {}\n"
          "Transformation: {}\n"
          "Speedup: {}\n".format(batch_size, input_size, str_id,
                                 (unoptimized_time / optimized_time)))

  def _get_known_cheap_fns(self):
    return [
        (lambda *args: [array_ops.identity(x) for x in args], "identity"),
        (lambda *args: [x + 1 for x in args], "add_const"),
        (lambda *args: args[0], "select"),
        (lambda *args: [math_ops.cast(x, dtypes.float64) for x in args],
         "cast"),
    ]


if __name__ == "__main__":
  test.main()
