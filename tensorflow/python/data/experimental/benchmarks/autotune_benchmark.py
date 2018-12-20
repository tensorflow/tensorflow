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
    a = self._benchmarkMap(autotune=False)
    b = self._benchmarkMap(autotune=True)
    print("speedup: %f" % (a / b))

  def _benchmarkMap(self, autotune):
    k = 1024 * 1024
    dataset = dataset_ops.Dataset.from_tensors((np.random.rand(1, 4 * k),
                                                np.random.rand(4 * k,
                                                               1))).repeat()
    dataset = dataset.map(
        math_ops.matmul, num_parallel_calls=optimization.AUTOTUNE)
    options = dataset_ops.Options()
    options.experimental_autotune = autotune
    dataset = dataset.with_options(options)
    iterator = dataset_ops.make_one_shot_iterator(dataset)
    get_next = iterator.get_next()

    deltas = []
    with session.Session() as sess:
      for _ in range(5):
        sess.run(get_next.op)
      for _ in range(10000):
        start = time.time()
        sess.run(get_next.op)
        end = time.time()
        deltas.append(end - start)

    print("%f (median), %f (mean), %f (stddev), %f (min), %f (max)\n" %
          (np.median(deltas), np.mean(deltas), np.std(deltas), np.min(deltas),
           np.max(deltas)))
    self.report_benchmark(
        iters=10000,
        wall_time=np.median(deltas),
        name="map" + ("_autotune" if autotune else ""))
    return np.median(deltas)

  def benchmarkMapAndBatch(self):
    a = self._benchmarkMapAndBatch(autotune=False)
    b = self._benchmarkMapAndBatch(autotune=True)
    print("speedup: %f" % (a / b))

  def _benchmarkMapAndBatch(self, autotune):
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
    options.experimental_autotune = autotune
    dataset = dataset.with_options(options)
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
        name="map_and_batch" + ("_autotune" if autotune else ""))
    return np.median(deltas)

  def benchmarkInterleave(self):
    a = self._benchmarkInterleave(autotune=False)
    b = self._benchmarkInterleave(autotune=True)
    print("speedup: %f" % (a / b))

  def _benchmarkInterleave(self, autotune):
    k = 1024 * 1024
    dataset = dataset_ops.Dataset.from_tensors((np.random.rand(1, 4 * k),
                                                np.random.rand(4 * k,
                                                               1))).repeat()
    dataset = dataset.map(math_ops.matmul)
    dataset = dataset_ops.Dataset.range(1).repeat().interleave(
        lambda _: dataset,
        cycle_length=10,
        num_parallel_calls=optimization.AUTOTUNE)
    options = dataset_ops.Options()
    options.experimental_autotune = autotune
    dataset = dataset.with_options(options)
    iterator = dataset_ops.make_one_shot_iterator(dataset)
    get_next = iterator.get_next()

    deltas = []
    with session.Session() as sess:
      for _ in range(5):
        sess.run(get_next.op)
      for _ in range(10000):
        start = time.time()
        sess.run(get_next.op)
        end = time.time()
        deltas.append(end - start)

    print("%f (median), %f (mean), %f (stddev), %f (min), %f (max)\n" %
          (np.median(deltas), np.mean(deltas), np.std(deltas), np.min(deltas),
           np.max(deltas)))
    self.report_benchmark(
        iters=10000,
        wall_time=np.median(deltas),
        name="interleave" + ("_autotune" if autotune else ""))
    return np.median(deltas)

  def benchmarkMapAndInterleave(self):
    a = self._benchmarkMapAndInterleave(autotune=False)
    b = self._benchmarkMapAndInterleave(autotune=True)
    print("speedup: %f" % (a / b))

  def _benchmarkMapAndInterleave(self, autotune):
    k = 1024 * 1024
    a = (np.random.rand(1, 8 * k), np.random.rand(8 * k, 1))
    b = (np.random.rand(1, 4 * k), np.random.rand(4 * k, 1))
    c = (np.random.rand(1, 2 * k), np.random.rand(2 * k, 1))
    dataset_a = dataset_ops.Dataset.from_tensors(a).repeat()
    dataset_b = dataset_ops.Dataset.from_tensors(b).repeat()
    dataset_c = dataset_ops.Dataset.from_tensors(c).repeat()

    def f1(x, y):
      return math_ops.matmul(x, y)

    def f2(a, b):
      x, y = b
      return a, math_ops.matmul(x, y)

    dataset = dataset_a
    dataset = dataset.map(f1, num_parallel_calls=optimization.AUTOTUNE)
    dataset = dataset_ops.Dataset.range(1).repeat().interleave(
        lambda _: dataset,
        num_parallel_calls=optimization.AUTOTUNE,
        cycle_length=2)

    dataset = dataset_ops.Dataset.zip((dataset, dataset_b))
    dataset = dataset.map(f2, num_parallel_calls=optimization.AUTOTUNE)
    dataset = dataset_ops.Dataset.range(1).repeat().interleave(
        lambda _: dataset,
        num_parallel_calls=optimization.AUTOTUNE,
        cycle_length=2)

    dataset = dataset_ops.Dataset.zip((dataset, dataset_c))
    dataset = dataset.map(f2, num_parallel_calls=optimization.AUTOTUNE)
    options = dataset_ops.Options()
    options.experimental_autotune = autotune
    dataset = dataset.with_options(options)
    iterator = dataset_ops.make_one_shot_iterator(dataset)
    get_next = iterator.get_next()

    deltas = []
    with session.Session() as sess:
      for _ in range(5):
        sess.run(get_next)
      for _ in range(1000):
        start = time.time()
        sess.run(get_next)
        end = time.time()
        deltas.append(end - start)

    print("%f (median), %f (mean), %f (stddev), %f (min), %f (max)\n" %
          (np.median(deltas), np.mean(deltas), np.std(deltas), np.min(deltas),
           np.max(deltas)))
    self.report_benchmark(
        iters=1000,
        wall_time=np.median(deltas),
        name="map_and_interleave" + ("_autotune" if autotune else ""))
    return np.median(deltas)


if __name__ == "__main__":
  test.main()
