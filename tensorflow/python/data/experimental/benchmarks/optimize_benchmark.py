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


from tensorflow.python.data.benchmarks import benchmark_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.ops import math_ops


class OptimizationBenchmark(benchmark_base.DatasetBenchmarkBase):
  """Benchmarks for static optimizations."""

  def benchmark_map_fusion(self):
    """Evaluates performance map of fusion."""

    chain_lengths = [0, 1, 2, 5, 10, 20, 50]
    for chain_length in chain_lengths:
      self._benchmark_map_fusion(
          chain_length=chain_length, optimize_dataset=False)
      self._benchmark_map_fusion(
          chain_length=chain_length, optimize_dataset=True)

  def _benchmark_map_fusion(self, chain_length, optimize_dataset):

    dataset = dataset_ops.Dataset.from_tensors(0).repeat(None)
    for _ in range(chain_length):
      dataset = dataset.map(lambda x: x)
    if optimize_dataset:
      options = dataset_ops.Options()
      options.experimental_optimization.apply_default_optimizations = False
      options.experimental_optimization.map_fusion = True
      dataset = dataset.with_options(options)

    opt_mark = "opt" if optimize_dataset else "noopt"
    self.run_and_report_benchmark(
        dataset=dataset,
        num_elements=100,
        iters=10,
        warmup=True,
        extras={
            "model_name": "optimize.benchmark.1",
            "parameters": "%d.%s" % (chain_length, optimize_dataset),
        },
        name="map_fusion_{}_chain_length_{}".format(opt_mark, chain_length))

  def benchmark_map_and_filter_fusion(self):
    """Evaluates performance map of fusion."""

    chain_lengths = [0, 1, 2, 5, 10, 20, 50]
    for chain_length in chain_lengths:
      self._benchmark_map_and_filter_fusion(
          chain_length=chain_length, optimize_dataset=False)
      self._benchmark_map_and_filter_fusion(
          chain_length=chain_length, optimize_dataset=True)

  def _benchmark_map_and_filter_fusion(self, chain_length, optimize_dataset):

    dataset = dataset_ops.Dataset.from_tensors(0).repeat(None)
    for _ in range(chain_length):
      dataset = dataset.map(lambda x: x + 5).filter(
          lambda x: math_ops.greater_equal(x - 5, 0))
    if optimize_dataset:
      options = dataset_ops.Options()
      options.experimental_optimization.apply_default_optimizations = False
      options.experimental_optimization.map_and_filter_fusion = True
      dataset = dataset.with_options(options)

    opt_mark = "opt" if optimize_dataset else "noopt"
    self.run_and_report_benchmark(
        dataset=dataset,
        num_elements=100,
        iters=10,
        warmup=True,
        extras={
            "model_name": "optimize.benchmark.2",
            "parameters": "%d.%s" % (chain_length, optimize_dataset),
        },
        name="map_and_filter_fusion_{}_chain_length_{}".format(
            opt_mark, chain_length))

  # This benchmark compares the performance of pipeline with multiple chained
  # filter with and without filter fusion.

  def benchmark_filter_fusion(self):
    chain_lengths = [0, 1, 2, 5, 10, 20, 50]
    for chain_length in chain_lengths:
      self._benchmark_filter_fusion(
          chain_length=chain_length, optimize_dataset=False)
      self._benchmark_filter_fusion(
          chain_length=chain_length, optimize_dataset=True)

  def _benchmark_filter_fusion(self, chain_length, optimize_dataset):

    dataset = dataset_ops.Dataset.from_tensors(5).repeat(None)
    for _ in range(chain_length):
      dataset = dataset.filter(lambda x: math_ops.greater_equal(x - 5, 0))
    if optimize_dataset:
      options = dataset_ops.Options()
      options.experimental_optimization.apply_default_optimizations = False
      options.experimental_optimization.filter_fusion = True
      dataset = dataset.with_options(options)

    opt_mark = "opt" if optimize_dataset else "noopt"
    self.run_and_report_benchmark(
        dataset=dataset,
        num_elements=100,
        iters=10,
        warmup=True,
        extras={
            "model_name": "optimize.benchmark.3",
            "parameters": "%d.%s" % (chain_length, optimize_dataset),
        },
        name="filter_fusion_{}_chain_length_{}".format(opt_mark, chain_length))


if __name__ == "__main__":
  benchmark_base.test.main()
