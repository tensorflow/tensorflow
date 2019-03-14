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
"""Benchmarks for ChooseFastestBranchDataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.data.benchmarks import benchmark_base
from tensorflow.python.data.experimental.ops import optimization
from tensorflow.python.data.ops import dataset_ops


class ChooseFastestBranchBenchmark(benchmark_base.DatasetBenchmarkBase):
  """Benchmarks for ChooseFastestBranchDatast."""

  def make_benchmark_datasets(self):

    dataset = dataset_ops.Dataset.range(1000**2).repeat()

    def branch_0(dataset):
      return dataset.map(lambda x: x + 1).batch(100)

    def branch_1(dataset):
      return dataset.batch(100).map(lambda x: x + 1)

    map_batch_dataset = branch_0(dataset)
    batch_map_dataset = branch_1(dataset)
    choose_fastest_dataset = optimization._ChooseFastestBranchDataset(  # pylint: disable=protected-access
        dataset, [branch_0, branch_1],
        ratio_numerator=100)
    return map_batch_dataset, batch_map_dataset, choose_fastest_dataset

  def benchmarkChooseFastest(self):
    map_batch, batch_map, choose_fastest = self.make_benchmark_datasets()

    def benchmark(dataset, name):
      self.run_and_report_benchmark(dataset, 5000, name, iters=1)

    benchmark(map_batch, "map_batch_dataset")
    benchmark(batch_map, "batch_map_dataset")
    benchmark(choose_fastest, "choose_fastest_dataset")

  def benchmarkChooseFastestFirstNIterations(self):

    map_batch, batch_map, choose_fastest = self.make_benchmark_datasets()

    def benchmark(dataset, name):
      self.run_and_report_benchmark(
          dataset, num_elements=10, name="%s_first_10" % name, iters=5)

    benchmark(map_batch, "map_batch_dataset")
    benchmark(batch_map, "batch_map_dataset")
    benchmark(choose_fastest, "choose_fastest_dataset")


if __name__ == "__main__":
  benchmark_base.test.main()
