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
from tensorflow.python.data.experimental.ops import sleep
from tensorflow.python.data.ops import dataset_ops


class ChooseFastestBranchBenchmark(benchmark_base.DatasetBenchmarkBase):
  """Benchmarks for ChooseFastestBranchDatast."""

  def make_benchmark_datasets(self,
                              input_dataset,
                              branch_0,
                              branch_1,
                              ratio_numerator,
                              num_elements_per_branch=None):
    ds_0 = branch_0(input_dataset)
    ds_1 = branch_1(input_dataset)
    choose_fastest_dataset = optimization._ChooseFastestBranchDataset(  # pylint: disable=protected-access
        input_dataset, [branch_0, branch_1],
        ratio_numerator=ratio_numerator,
        num_elements_per_branch=num_elements_per_branch)

    return ds_0, ds_1, choose_fastest_dataset

  def make_simple_benchmark_datasets(self):

    dataset = dataset_ops.Dataset.range(1000**2).repeat()

    def branch_0(dataset):
      return dataset.map(lambda x: x + 1).batch(100)

    def branch_1(dataset):
      return dataset.batch(100).map(lambda x: x + 1)

    return self.make_benchmark_datasets(dataset, branch_0, branch_1, 100)

  def benchmarkChooseFastest(self):
    map_batch, batch_map, choose_fastest = self.make_simple_benchmark_datasets()

    def benchmark(dataset, name):
      self.run_and_report_benchmark(dataset, 5000, name, iters=1)

    benchmark(map_batch, "map_batch_dataset")
    benchmark(batch_map, "batch_map_dataset")
    benchmark(choose_fastest, "choose_fastest_dataset")

  def benchmarkChooseFastestFirstNIterations(self):

    map_batch, batch_map, choose_fastest = self.make_simple_benchmark_datasets()

    def benchmark(dataset, name):
      self.run_and_report_benchmark(
          dataset, num_elements=10, name="%s_first_10" % name, iters=5)

    benchmark(map_batch, "map_batch_dataset")
    benchmark(batch_map, "batch_map_dataset")
    benchmark(choose_fastest, "choose_fastest_dataset")

  def benchmarkWithInputSkew(self):

    def make_dataset(time_us, num_elements):
      return dataset_ops.Dataset.range(num_elements).apply(sleep.sleep(time_us))

    # Dataset with 100 elements that emulates performance characteristics of a
    # file-based dataset stored in remote storage, where the first element
    # takes significantly longer to produce than the remaining elements.
    input_dataset = make_dataset(1000 * 1000,
                                 0).concatenate(make_dataset(1, 100)).take(100)

    def slow_branch(dataset):
      return dataset.apply(sleep.sleep(10000))

    def fast_branch(dataset):
      return dataset.apply(sleep.sleep(10))

    def benchmark(dataset, name):
      self.run_and_report_benchmark(
          dataset, num_elements=100, name="%s_with_skew" % name, iters=1)

    # ChooseFastestBranch dataset should choose the same branch regardless
    # of the order of the branches, so we expect the iteration speed to be
    # comparable for both versions.
    slow_ds, fast_ds, choose_fastest_0 = self.make_benchmark_datasets(
        input_dataset, slow_branch, fast_branch, 1, num_elements_per_branch=2)
    _, _, choose_fastest_1 = self.make_benchmark_datasets(
        input_dataset, fast_branch, slow_branch, 1, num_elements_per_branch=2)

    benchmark(slow_ds, "slow_dataset")
    benchmark(fast_ds, "fast_dataset")
    benchmark(choose_fastest_0, "choose_fastest_dataset_0")
    benchmark(choose_fastest_1, "choose_fastest_dataset_1")


if __name__ == "__main__":
  benchmark_base.test.main()
