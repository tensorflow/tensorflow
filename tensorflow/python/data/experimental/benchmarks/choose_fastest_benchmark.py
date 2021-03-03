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

from tensorflow.python.data.experimental.ops import optimization
from tensorflow.python.data.benchmarks import benchmark_base
from tensorflow.python.data.ops import dataset_ops


class ChooseFastestBenchmark(benchmark_base.DatasetBenchmarkBase):
  """Benchmarks for static optimizations."""

  def benchmark_choose_fastest(self):

    dataset = dataset_ops.Dataset.range(1000**2).repeat()
    options = dataset_ops.Options()
    options.experimental_optimization.apply_default_optimizations = False
    dataset = dataset.with_options(options)
    map_batch_dataset = dataset.map(lambda x: x + 1).batch(100)
    batch_map_dataset = dataset.batch(100).map(lambda x: x + 1)

    merge_dataset = optimization._ChooseFastestDataset(  # pylint: disable=protected-access
        [batch_map_dataset, map_batch_dataset])
    self._benchmark(dataset=map_batch_dataset, name="map_batch_dataset")
    self._benchmark(dataset=batch_map_dataset, name="batch_map_dataset")
    self._benchmark(dataset=merge_dataset, name="merge_dataset")

  def benchmark_choose_fastest_first_n_iterations(self):

    dataset = dataset_ops.Dataset.range(1000**2).repeat()
    options = dataset_ops.Options()
    options.experimental_optimization.apply_default_optimizations = False
    dataset = dataset.with_options(options)
    map_batch_dataset = dataset.map(lambda x: x + 1).batch(100)
    batch_map_dataset = dataset.batch(100).map(lambda x: x + 1)

    merge_dataset = optimization._ChooseFastestDataset(  # pylint: disable=protected-access
        [batch_map_dataset, map_batch_dataset])

    self._benchmark_first_n(dataset=map_batch_dataset, name="map_batch_dataset")
    self._benchmark_first_n(dataset=batch_map_dataset, name="batch_map_dataset")
    self._benchmark_first_n(dataset=merge_dataset, name="merge_dataset")

  def _benchmark_first_n(self, dataset, name):
    n = 10  # The default num_experiments for ChooseFastestDataset
    self.run_and_report_benchmark(
        dataset=dataset,
        num_elements=n,
        iters=100,
        warmup=True,
        name=name + "_first_%d" % n)

  def _benchmark(self, dataset, name):

    self.run_and_report_benchmark(
        dataset=dataset, num_elements=100, iters=100, warmup=True, name=name)


if __name__ == "__main__":
  benchmark_base.test.main()
