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
"""Benchmarks for static optimizations."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np

from tensorflow.python.client import session
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


# TODO(b/119837791): Add eager benchmarks too.
class OptimizationBenchmark(test.Benchmark):
  """Benchmarks for static optimizations."""

  def benchmarkMapFusion(self):
    """Evaluates performance map of fusion."""

    chain_lengths = [0, 1, 2, 5, 10, 20, 50]
    for chain_length in chain_lengths:
      self._benchmarkMapFusion(chain_length, False)
      self._benchmarkMapFusion(chain_length, True)

  def _benchmarkMapFusion(self, chain_length, optimize_dataset):
    with ops.Graph().as_default():
      dataset = dataset_ops.Dataset.from_tensors(0).repeat(None)
      for _ in range(chain_length):
        dataset = dataset.map(lambda x: x)
      if optimize_dataset:
        options = dataset_ops.Options()
        options.experimental_map_fusion = True
        dataset = dataset.with_options(options)

      iterator = dataset_ops.make_one_shot_iterator(dataset)
      next_element = iterator.get_next()

      with session.Session() as sess:
        for _ in range(5):
          sess.run(next_element.op)
        deltas = []
        for _ in range(100):
          start = time.time()
          for _ in range(100):
            sess.run(next_element.op)
          end = time.time()
          deltas.append(end - start)

        median_wall_time = np.median(deltas) / 100
        opt_mark = "opt" if optimize_dataset else "noopt"
        print("Map dataset {} chain length: {} Median wall time: {}".format(
            opt_mark, chain_length, median_wall_time))
        self.report_benchmark(
            iters=100,
            wall_time=median_wall_time,
            name="map_fusion_{}_chain_length_{}".format(
                opt_mark, chain_length))

  def benchmarkMapAndFilterFusion(self):
    """Evaluates performance map of fusion."""

    chain_lengths = [0, 1, 2, 5, 10, 20, 50]
    for chain_length in chain_lengths:
      self._benchmarkMapAndFilterFusion(chain_length, False)
      self._benchmarkMapAndFilterFusion(chain_length, True)

  def _benchmarkMapAndFilterFusion(self, chain_length, optimize_dataset):
    with ops.Graph().as_default():
      dataset = dataset_ops.Dataset.from_tensors(0).repeat(None)
      for _ in range(chain_length):
        dataset = dataset.map(lambda x: x + 5).filter(
            lambda x: math_ops.greater_equal(x - 5, 0))
      if optimize_dataset:
        options = dataset_ops.Options()
        options.experimental_map_and_filter_fusion = True
        dataset = dataset.with_options(options)
      iterator = dataset_ops.make_one_shot_iterator(dataset)
      next_element = iterator.get_next()

      with session.Session() as sess:
        for _ in range(10):
          sess.run(next_element.op)
        deltas = []
        for _ in range(100):
          start = time.time()
          for _ in range(100):
            sess.run(next_element.op)
          end = time.time()
          deltas.append(end - start)

        median_wall_time = np.median(deltas) / 100
        opt_mark = "opt" if optimize_dataset else "noopt"
        print("Map and filter dataset {} chain length: {} Median wall time: {}"
              .format(opt_mark, chain_length, median_wall_time))
        self.report_benchmark(
            iters=100,
            wall_time=median_wall_time,
            name="map_and_filter_fusion_{}_chain_length_{}".format(
                opt_mark, chain_length))

  # This benchmark compares the performance of pipeline with multiple chained
  # filter with and without filter fusion.
  def benchmarkFilterFusion(self):
    chain_lengths = [0, 1, 2, 5, 10, 20, 50]
    for chain_length in chain_lengths:
      self._benchmarkFilters(chain_length, False)
      self._benchmarkFilters(chain_length, True)

  def _benchmarkFilterFusion(self, chain_length, optimize_dataset):
    with ops.Graph().as_default():
      dataset = dataset_ops.Dataset.from_tensors(5).repeat(None)
      for _ in range(chain_length):
        dataset = dataset.filter(lambda x: math_ops.greater_equal(x - 5, 0))
      if optimize_dataset:
        options = dataset_ops.Options()
        options.experimental_filter_fusion = True
        dataset = dataset.with_options(options)

      iterator = dataset_ops.make_one_shot_iterator(dataset)
      next_element = iterator.get_next()

      for _ in range(10):
        self.evaluate(next_element.op)
      deltas = []
      for _ in range(100):
        start = time.time()
        for _ in range(100):
          self.evaluate(next_element.op)
        end = time.time()
        deltas.append(end - start)

      median_wall_time = np.median(deltas) / 100
      opt_mark = "opt" if optimize_dataset else "no-opt"
      print("Filter dataset {} chain length: {} Median wall time: {}".format(
          opt_mark, chain_length, median_wall_time))
      self.report_benchmark(
          iters=1000,
          wall_time=median_wall_time,
          name="chain_length_{}_{}".format(opt_mark, chain_length))


if __name__ == "__main__":
  test.main()
