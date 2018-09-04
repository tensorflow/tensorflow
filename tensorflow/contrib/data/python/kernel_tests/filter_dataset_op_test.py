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
"""Benchmarks FilterDataset input pipeline op."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np

from tensorflow.contrib.data.python.ops import optimization
from tensorflow.python.client import session
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class FilterBenchmark(test.Benchmark):

  # This benchmark compares the performance of pipeline with multiple chained
  # filter with and without filter fusion.
  def benchmarkFilters(self):
    chain_lengths = [0, 1, 2, 5, 10, 20, 50]
    for chain_length in chain_lengths:
      self._benchmarkFilters(chain_length, False)
      self._benchmarkFilters(chain_length, True)

  def _benchmarkFilters(self, chain_length, optimize_dataset):
    with ops.Graph().as_default():
      dataset = dataset_ops.Dataset.from_tensors(5).repeat(None)
      for _ in range(chain_length):
        dataset = dataset.filter(lambda x: math_ops.greater_equal(x - 5, 0))
      if optimize_dataset:
        dataset = dataset.apply(optimization.optimize(["filter_fusion"]))

      iterator = dataset.make_one_shot_iterator()
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
        opt_mark = "opt" if optimize_dataset else "no-opt"
        print("Filter dataset {} chain length: {} Median wall time: {}".format(
            opt_mark, chain_length, median_wall_time))
        self.report_benchmark(
            iters=1000,
            wall_time=median_wall_time,
            name="benchmark_filter_dataset_chain_latency_{}_{}".format(
                opt_mark, chain_length))


if __name__ == "__main__":
  test.main()
