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
"""Bechmarks for `tf.data.Dataset.map()`."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.data.benchmarks import benchmark_base
from tensorflow.python.data.ops import dataset_ops


# TODO(b/119837791): Add eager benchmarks.
class MapBenchmark(benchmark_base.DatasetBenchmarkBase):
  """Benchmarks for `tf.data.Dataset.map()`."""

  def benchmark_chain_of_maps(self):

    def benchmark_helper(chain_length, map_fn, use_inter_op_parallelism, label):
      dataset = dataset_ops.Dataset.from_tensors(0).repeat(None)
      for _ in range(chain_length):
        dataset = dataset_ops.MapDataset(
            dataset, map_fn, use_inter_op_parallelism=use_inter_op_parallelism)
      self.run_and_report_benchmark(
          dataset,
          num_elements=10000,
          name="chain_length_%d%s" % (chain_length, label))

    chain_lengths = [0, 1, 2, 5, 10, 20, 50]
    for chain_length in chain_lengths:
      benchmark_helper(chain_length, lambda x: x + 1, True, "")
      benchmark_helper(chain_length, lambda x: x + 1, False, "_single_threaded")
      benchmark_helper(chain_length, lambda x: x, True, "_short_circuit")

  def benchmark_map_fan_out(self):
    fan_outs = [1, 2, 5, 10, 20, 50, 100]

    def benchmark_helper(fan_out, map_fn, use_inter_op_parallelism, label):
      dataset = dataset_ops.Dataset.from_tensors(
          tuple(0 for _ in range(fan_out))).repeat(None)
      dataset = dataset_ops.MapDataset(
          dataset, map_fn, use_inter_op_parallelism=use_inter_op_parallelism)
      self.run_and_report_benchmark(
          dataset,
          num_elements=10000,
          name="fan_out_%d%s" % (fan_out, label))

    for fan_out in fan_outs:
      benchmark_helper(fan_out, lambda *xs: [x + 1 for x in xs], True, "")
      benchmark_helper(fan_out, lambda *xs: [x + 1 for x in xs], False,
                       "_single_threaded")
      benchmark_helper(fan_out, lambda *xs: xs, True, "_short_circuit")


if __name__ == "__main__":
  benchmark_base.test.main()
