# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
from tensorflow.python.data.experimental.ops import optimization
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.platform import test


# TODO(b/119837791): Add eager benchmarks too.
class ChooseFastestBenchmark(test.Benchmark):
  """Benchmarks for static optimizations."""

  def benchmarkChooseFastest(self):

    dataset = dataset_ops.Dataset.range(1000**2).repeat()
    options = dataset_ops.Options()
    options.experimental_optimization.apply_default_optimizations = False
    dataset = dataset.with_options(options)
    map_batch_dataset = dataset.map(lambda x: x + 1).batch(100)
    batch_map_dataset = dataset.batch(100).map(lambda x: x + 1)

    merge_dataset = optimization._ChooseFastestDataset(  # pylint: disable=protected-access
        [batch_map_dataset, map_batch_dataset])
    self._benchmark(map_batch_dataset, "map_batch_dataset")
    self._benchmark(batch_map_dataset, "batch_map_dataset")
    self._benchmark(merge_dataset, "merge_dataset")

  def benchmarkChooseFastestFirstNIterations(self):

    dataset = dataset_ops.Dataset.range(1000**2).repeat()
    options = dataset_ops.Options()
    options.experimental_optimization.apply_default_optimizations = False
    dataset = dataset.with_options(options)
    map_batch_dataset = dataset.map(lambda x: x + 1).batch(100)
    batch_map_dataset = dataset.batch(100).map(lambda x: x + 1)

    merge_dataset = optimization._ChooseFastestDataset(  # pylint: disable=protected-access
        [batch_map_dataset, map_batch_dataset])

    self._benchmarkFirstN(map_batch_dataset, "map_batch_dataset")
    self._benchmarkFirstN(batch_map_dataset, "batch_map_dataset")
    self._benchmarkFirstN(merge_dataset, "merge_dataset")

  def _benchmarkFirstN(self, dataset, name):
    n = 10  # The default num_experiments for ChooseFastestDataset
    iterator = dataset_ops.make_one_shot_iterator(dataset)
    next_element = iterator.get_next()

    deltas = []
    for _ in range(100):
      with session.Session() as sess:
        start = time.time()
        for _ in range(n):
          sess.run(next_element.op)
        end = time.time()
        deltas.append(end - start)
    median_wall_time = np.median(deltas) / n
    self.report_benchmark(
        iters=n, wall_time=median_wall_time, name=name + "_first_%d" % n)

  def _benchmark(self, dataset, name):
    iterator = dataset_ops.make_one_shot_iterator(dataset)
    next_element = iterator.get_next()

    with session.Session() as sess:
      # Run 10 steps to warm up the session caches before taking the first
      # measurement. Additionally, 10 is the default num_experiments for
      # ChooseFastestDataset.
      for _ in range(10):
        sess.run(next_element.op)
      deltas = []
      for _ in range(50):
        start = time.time()
        for _ in range(50):
          sess.run(next_element.op)
        end = time.time()
        deltas.append(end - start)

      median_wall_time = np.median(deltas) / 100
      self.report_benchmark(iters=100, wall_time=median_wall_time, name=name)


if __name__ == "__main__":
  test.main()
