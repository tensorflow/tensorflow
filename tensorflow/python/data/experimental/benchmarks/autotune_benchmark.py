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
"""Benchmarks for autotuning performance knobs."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np

from tensorflow.python.client import session
from tensorflow.python.data.experimental.ops import batching
from tensorflow.python.data.experimental.ops import optimization
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class AutotuneBenchmark(test.Benchmark):
  """Benchmarks for autotuning performance knobs."""

  def benchmarkMap(self):
    k = 1024 * 1024
    dataset = dataset_ops.Dataset.from_tensors((np.random.rand(1, 4 * k),
                                                np.random.rand(4 * k,
                                                               1))).repeat()
    dataset = dataset.map(
        math_ops.matmul, num_parallel_calls=optimization.AUTOTUNE)
    iterator = dataset_ops.make_one_shot_iterator(dataset)
    get_next = iterator.get_next()

    deltas = []
    with session.Session() as sess:
      for _ in range(5):
        sess.run(get_next.op)
      for _ in range(1000):
        start = time.time()
        sess.run(get_next.op)
        end = time.time()
        deltas.append(end - start)

    print("%f (median), %f (mean), %f (stddev), %f (min), %f (max)\n" %
          (np.median(deltas), np.mean(deltas), np.std(deltas), np.min(deltas),
           np.max(deltas)))
    self.report_benchmark(
        iters=1000, wall_time=np.median(deltas), name="map_autotune")

  def benchmarkMapAndBatch(self):
    self._benchmarkMapAndBatch(numa_aware=False)
    self._benchmarkMapAndBatch(numa_aware=True)

  def _benchmarkMapAndBatch(self, numa_aware):
    batch_size = 16
    k = 1024 * 1024
    dataset = dataset_ops.Dataset.from_tensors((np.random.rand(1, 4 * k),
                                                np.random.rand(4 * k,
                                                               1))).repeat()
    dataset = dataset.apply(
        batching.map_and_batch(
            math_ops.matmul,
            num_parallel_calls=optimization.AUTOTUNE,
            batch_size=batch_size))
    options = dataset_ops.Options()
    options.experimental_numa_aware = numa_aware
    dataset = dataset.with_options(options)
    iterator = dataset_ops.make_one_shot_iterator(dataset)
    get_next = iterator.get_next()

    deltas = []
    with session.Session() as sess:
      for _ in range(5):
        sess.run(get_next.op)
      for _ in range(100):
        start = time.time()
        sess.run(get_next.op)
        end = time.time()
        deltas.append(end - start)

    print("%f (median), %f (mean), %f (stddev), %f (min), %f (max)\n" %
          (np.median(deltas), np.mean(deltas), np.std(deltas), np.min(deltas),
           np.max(deltas)))

    self.report_benchmark(
        iters=100,
        wall_time=np.median(deltas),
        name=("numa_" if numa_aware else "") + "map_and_batch_autotune")

  def benchmarkInterleave(self):
    k = 1024 * 1024
    dataset = dataset_ops.Dataset.from_tensors((np.random.rand(1, 4 * k),
                                                np.random.rand(4 * k,
                                                               1))).repeat()
    dataset = dataset.map(math_ops.matmul)
    dataset = dataset_ops.Dataset.range(1).repeat().interleave(
        lambda _: dataset,
        cycle_length=10,
        num_parallel_calls=optimization.AUTOTUNE)
    iterator = dataset_ops.make_one_shot_iterator(dataset)
    get_next = iterator.get_next()

    deltas = []
    with session.Session() as sess:
      for _ in range(5):
        sess.run(get_next.op)
      for _ in range(1000):
        start = time.time()
        sess.run(get_next.op)
        end = time.time()
        deltas.append(end - start)

    print("%f (median), %f (mean), %f (stddev), %f (min), %f (max)\n" %
          (np.median(deltas), np.mean(deltas), np.std(deltas), np.min(deltas),
           np.max(deltas)))
    self.report_benchmark(
        iters=1000,
        wall_time=np.median(deltas),
        name="interleave_autotune")

  def benchmarkMapAndInterleave(self):
    k = 1024 * 1024
    a = (np.random.rand(1, 8 * k), np.random.rand(8 * k, 1))
    b = (np.random.rand(1, 4 * k), np.random.rand(4 * k, 1))
    c = (np.random.rand(1, 2 * k), np.random.rand(2 * k, 1))
    dataset = dataset_ops.Dataset.from_tensors((a, b, c)).repeat()

    def f1(a, b, c):
      x, y = a
      return math_ops.matmul(x, y), b, c

    def f2(a, b, c):
      x, y = b
      return a, math_ops.matmul(x, y), c

    def f3(a, b, c):
      x, y = c
      return a, b, math_ops.matmul(x, y)

    dataset = dataset.map(f1, num_parallel_calls=optimization.AUTOTUNE)
    dataset = dataset_ops.Dataset.range(1).repeat().interleave(
        lambda _: dataset,
        num_parallel_calls=optimization.AUTOTUNE,
        cycle_length=2)

    dataset = dataset.map(f2, num_parallel_calls=optimization.AUTOTUNE)
    dataset = dataset_ops.Dataset.range(1).repeat().interleave(
        lambda _: dataset,
        num_parallel_calls=optimization.AUTOTUNE,
        cycle_length=2)

    dataset = dataset.map(f3, num_parallel_calls=optimization.AUTOTUNE)
    iterator = dataset_ops.make_one_shot_iterator(dataset)
    get_next = iterator.get_next()

    deltas = []
    with session.Session() as sess:
      for _ in range(5):
        sess.run(get_next)
      for _ in range(100):
        start = time.time()
        sess.run(get_next)
        end = time.time()
        deltas.append(end - start)

    print("%f (median), %f (mean), %f (stddev), %f (min), %f (max)\n" %
          (np.median(deltas), np.mean(deltas), np.std(deltas), np.min(deltas),
           np.max(deltas)))
    self.report_benchmark(
        iters=100,
        wall_time=np.median(deltas),
        name="map_and_interleave_autotune")


if __name__ == "__main__":
  test.main()
