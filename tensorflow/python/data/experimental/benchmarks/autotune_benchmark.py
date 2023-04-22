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


import numpy as np

from tensorflow.python.data.benchmarks import benchmark_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.ops import math_ops


class AutotuneBenchmark(benchmark_base.DatasetBenchmarkBase):
  """Benchmarks for autotuning performance knobs."""

  def _run_benchmark(self, dataset, autotune, autotune_buffers,
                     benchmark_iters, benchmark_label, benchmark_id):
    options = dataset_ops.Options()
    options.experimental_optimization.apply_default_optimizations = False
    options.experimental_optimization.autotune = autotune
    options.experimental_optimization.autotune_buffers = autotune_buffers
    dataset = dataset.with_options(options)

    autotune_string = "_autotune_{}".format(
        "parallelism_and_buffer_sizes"
        if autotune_buffers else "parallelism_only")
    wall_time = self.run_and_report_benchmark(
        dataset=dataset,
        num_elements=benchmark_iters,
        warmup=True,
        iters=1,
        extras={
            "model_name":
                "autotune.benchmark.%s.%d" % (benchmark_label, benchmark_id),
            "parameters":
                "%s.%s" % (autotune, autotune_buffers),
        },
        name=benchmark_label + (autotune_string if autotune else ""))
    return wall_time

  def benchmark_batch(self):
    a = self._benchmark_batch(
        autotune=False, autotune_buffers=False, benchmark_id=1)
    b = self._benchmark_batch(
        autotune=True, autotune_buffers=False, benchmark_id=2)
    c = self._benchmark_batch(
        autotune=True, autotune_buffers=True, benchmark_id=3)
    print("autotune parallelism vs no autotuning speedup: {}".format(a / b))
    print("autotune parallelism and buffer sizes vs no autotuning speedup: {}"
          .format(a / c))

  def _benchmark_batch(self, autotune, autotune_buffers, benchmark_id):
    batch_size = 128
    k = 1024
    dataset = dataset_ops.Dataset.from_tensors(
        (np.random.rand(1, 4 * k), np.random.rand(4 * k, 1))).repeat()
    dataset = dataset.map(math_ops.matmul)
    dataset = dataset.batch(
        batch_size=batch_size, num_parallel_calls=dataset_ops.AUTOTUNE)
    return self._run_benchmark(
        dataset=dataset,
        autotune=autotune,
        autotune_buffers=autotune_buffers,
        benchmark_iters=10000,
        benchmark_label="batch",
        benchmark_id=benchmark_id)

  def benchmark_map(self):
    a = self._benchmark_map(
        autotune=False, autotune_buffers=False, benchmark_id=1)
    b = self._benchmark_map(
        autotune=True, autotune_buffers=False, benchmark_id=2)
    c = self._benchmark_map(
        autotune=True, autotune_buffers=True, benchmark_id=3)
    print("autotune parallelism vs no autotuning speedup: {}".format(a / b))
    print("autotune parallelism and buffer sizes vs no autotuning speedup: {}"
          .format(a / c))

  def _benchmark_map(self, autotune, autotune_buffers, benchmark_id):
    k = 1024 * 1024
    dataset = dataset_ops.Dataset.from_tensors(
        (np.random.rand(1, 4 * k), np.random.rand(4 * k, 1))).repeat()
    dataset = dataset.map(
        math_ops.matmul, num_parallel_calls=dataset_ops.AUTOTUNE)
    return self._run_benchmark(
        dataset=dataset,
        autotune=autotune,
        autotune_buffers=autotune_buffers,
        benchmark_iters=10000,
        benchmark_label="map",
        benchmark_id=benchmark_id)

  def benchmark_map_and_batch(self):
    a = self._benchmark_map_and_batch(
        autotune=False, autotune_buffers=False, benchmark_id=1)
    b = self._benchmark_map_and_batch(
        autotune=True, autotune_buffers=False, benchmark_id=2)
    c = self._benchmark_map_and_batch(
        autotune=True, autotune_buffers=True, benchmark_id=3)
    print("autotune parallelism vs no autotuning speedup: {}".format(a / b))
    print("autotune parallelism and buffer sizes vs no autotuning speedup: {}"
          .format(a / c))

  def _benchmark_map_and_batch(self, autotune, autotune_buffers, benchmark_id):
    batch_size = 16
    k = 1024 * 1024
    dataset = dataset_ops.Dataset.from_tensors(
        (np.random.rand(1, 4 * k), np.random.rand(4 * k, 1))).repeat()
    dataset = dataset.map(
        math_ops.matmul, num_parallel_calls=dataset_ops.AUTOTUNE)
    dataset = dataset.batch(batch_size=batch_size)
    return self._run_benchmark(
        dataset=dataset,
        autotune=autotune,
        autotune_buffers=autotune_buffers,
        benchmark_iters=1000,
        benchmark_label="map_and_batch",
        benchmark_id=benchmark_id)

  def benchmark_interleave(self):
    a = self._benchmark_interleave(
        autotune=False, autotune_buffers=False, benchmark_id=1)
    b = self._benchmark_interleave(
        autotune=True, autotune_buffers=False, benchmark_id=2)
    c = self._benchmark_interleave(
        autotune=True, autotune_buffers=True, benchmark_id=3)
    print("autotune parallelism vs no autotuning speedup: {}".format(a / b))
    print("autotune parallelism and buffer sizes vs no autotuning speedup: {}"
          .format(a / c))

  def _benchmark_interleave(self, autotune, autotune_buffers, benchmark_id):
    k = 1024 * 1024
    dataset = dataset_ops.Dataset.from_tensors(
        (np.random.rand(1, 4 * k), np.random.rand(4 * k, 1))).repeat()
    dataset = dataset.map(math_ops.matmul)
    dataset = dataset_ops.Dataset.range(1).repeat().interleave(
        lambda _: dataset,
        cycle_length=10,
        num_parallel_calls=dataset_ops.AUTOTUNE)
    return self._run_benchmark(
        dataset=dataset,
        autotune=autotune,
        autotune_buffers=autotune_buffers,
        benchmark_iters=10000,
        benchmark_label="interleave",
        benchmark_id=benchmark_id)

  def benchmark_map_and_interleave(self):
    a = self._benchmark_map_and_interleave(
        autotune=False, autotune_buffers=False, benchmark_id=1)
    b = self._benchmark_map_and_interleave(
        autotune=True, autotune_buffers=False, benchmark_id=2)
    c = self._benchmark_map_and_interleave(
        autotune=True, autotune_buffers=True, benchmark_id=3)
    print("autotune parallelism vs no autotuning speedup: {}".format(a / b))
    print("autotune parallelism and buffer sizes vs no autotuning speedup: {}"
          .format(a / c))

  def _benchmark_map_and_interleave(self, autotune, autotune_buffers,
                                    benchmark_id):
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
    dataset = dataset.map(f1, num_parallel_calls=dataset_ops.AUTOTUNE)
    dataset = dataset_ops.Dataset.range(1).repeat().interleave(
        lambda _: dataset,
        num_parallel_calls=dataset_ops.AUTOTUNE,
        cycle_length=2)

    dataset = dataset_ops.Dataset.zip((dataset, dataset_b))
    dataset = dataset.map(f2, num_parallel_calls=dataset_ops.AUTOTUNE)
    dataset = dataset_ops.Dataset.range(1).repeat().interleave(
        lambda _: dataset,
        num_parallel_calls=dataset_ops.AUTOTUNE,
        cycle_length=2)

    dataset = dataset_ops.Dataset.zip((dataset, dataset_c))
    dataset = dataset.map(f2, num_parallel_calls=dataset_ops.AUTOTUNE)
    return self._run_benchmark(
        dataset=dataset,
        autotune=autotune,
        autotune_buffers=autotune_buffers,
        benchmark_iters=10000,
        benchmark_label="map_and_interleave",
        benchmark_id=benchmark_id)

  def benchmark_map_batch_and_interleave(self):
    a = self._benchmark_map_batch_and_interleave(
        autotune=False, autotune_buffers=False, benchmark_id=1)
    b = self._benchmark_map_batch_and_interleave(
        autotune=True, autotune_buffers=False, benchmark_id=2)
    c = self._benchmark_map_batch_and_interleave(
        autotune=True, autotune_buffers=True, benchmark_id=3)
    print("autotune parallelism vs no autotuning speedup: {}".format(a / b))
    print("autotune parallelism and buffer sizes vs no autotuning speedup: {}"
          .format(a / c))

  def _benchmark_map_batch_and_interleave(self, autotune, autotune_buffers,
                                          benchmark_id):
    batch_size = 16
    k = 1024 * 1024
    a = (np.random.rand(1, 8 * k), np.random.rand(8 * k, 1))
    b = (np.random.rand(1, 4 * k), np.random.rand(4 * k, 1))
    c = (np.random.rand(1, 2 * k), np.random.rand(2 * k, 1))
    dataset_a = dataset_ops.Dataset.from_tensors(a).repeat()
    dataset_b = dataset_ops.Dataset.from_tensors(b).repeat()
    dataset_c = dataset_ops.Dataset.from_tensors(c).repeat()

    dataset = dataset_a
    dataset = dataset.map(
        math_ops.matmul, num_parallel_calls=dataset_ops.AUTOTUNE)
    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset_ops.Dataset.range(1).repeat().interleave(
        lambda _: dataset,
        num_parallel_calls=dataset_ops.AUTOTUNE,
        cycle_length=2)

    dataset = dataset_ops.Dataset.zip((dataset, dataset_b))
    dataset = dataset_ops.Dataset.range(1).repeat().interleave(
        lambda _: dataset,
        num_parallel_calls=dataset_ops.AUTOTUNE,
        cycle_length=2)

    dataset_c = dataset_c.map(
        math_ops.matmul, num_parallel_calls=dataset_ops.AUTOTUNE)
    dataset_c = dataset_c.batch(batch_size=batch_size)
    dataset = dataset_ops.Dataset.zip((dataset, dataset_c))
    return self._run_benchmark(
        dataset=dataset,
        autotune=autotune,
        autotune_buffers=autotune_buffers,
        benchmark_iters=1000,
        benchmark_label="map_and_batch_and_interleave",
        benchmark_id=benchmark_id)


if __name__ == "__main__":
  benchmark_base.test.main()
