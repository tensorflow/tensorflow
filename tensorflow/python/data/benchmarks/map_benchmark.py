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
"""Benchmarks for `tf.data.Dataset.map()`."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.data.benchmarks import benchmark_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import map_fn
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops


class MapBenchmark(benchmark_base.DatasetBenchmarkBase):
  """Benchmarks for `tf.data.Dataset.map()`."""

  def benchmark_chain_of_maps(self):

    def benchmark_helper(chain_length, fn, use_inter_op_parallelism, label,
                         benchmark_id):
      dataset = dataset_ops.Dataset.range(10000)
      for _ in range(chain_length):
        dataset = dataset_ops.MapDataset(
            dataset, fn, use_inter_op_parallelism=use_inter_op_parallelism)
      self.run_and_report_benchmark(
          dataset,
          num_elements=10000,
          extras={
              "model_name": "map.benchmark.%d" % benchmark_id,
              "parameters": "%d" % chain_length,
          },
          name="chain_length_%d%s" % (chain_length, label))

    chain_lengths = [0, 1, 2, 5, 10, 20, 50]
    for chain_length in chain_lengths:
      benchmark_helper(
          chain_length=chain_length,
          fn=lambda x: x + 1,
          use_inter_op_parallelism=True,
          label="",
          benchmark_id=1)
      benchmark_helper(
          chain_length=chain_length,
          fn=lambda x: x + 1,
          use_inter_op_parallelism=False,
          label="_single_threaded",
          benchmark_id=2)
      benchmark_helper(
          chain_length=chain_length,
          fn=lambda x: x,
          use_inter_op_parallelism=True,
          label="_short_circuit",
          benchmark_id=3)

  def benchmark_map_fan_out(self):
    fan_outs = [1, 2, 5, 10, 20, 50, 100]

    def benchmark_helper(fan_out, fn, use_inter_op_parallelism, label,
                         benchmark_id):
      dataset = dataset_ops.Dataset.from_tensors(
          tuple(0 for _ in range(fan_out))).repeat(None)
      dataset = dataset_ops.MapDataset(
          dataset, fn, use_inter_op_parallelism=use_inter_op_parallelism)
      self.run_and_report_benchmark(
          dataset,
          num_elements=10000,
          extras={
              "model_name": "map.benchmark.%d" % benchmark_id,
              "parameters": "%d" % fan_out,
          },
          name="fan_out_%d%s" % (fan_out, label))

    for fan_out in fan_outs:
      benchmark_helper(
          fan_out=fan_out,
          fn=lambda *xs: [x + 1 for x in xs],
          use_inter_op_parallelism=True,
          label="",
          benchmark_id=4)
      benchmark_helper(
          fan_out=fan_out,
          fn=lambda *xs: [x + 1 for x in xs],
          use_inter_op_parallelism=False,
          label="_single_threaded",
          benchmark_id=5)
      benchmark_helper(
          fan_out=fan_out,
          fn=lambda *xs: xs,
          use_inter_op_parallelism=True,
          label="_short_circuit",
          benchmark_id=6)

  def benchmark_sequential_control_flow(self):
    dataset = dataset_ops.Dataset.from_tensors(100000)

    def fn(x):
      i = constant_op.constant(0)

      def body(i, x):
        return math_ops.add(i, 1), x

      return control_flow_ops.while_loop(math_ops.less, body, [i, x])

    num_elements = 1
    dataset = dataset.map(fn)
    self.run_and_report_benchmark(
        dataset,
        num_elements=num_elements,
        extras={
            "model_name": "map.benchmark.8",
            "parameters": "%d" % num_elements,
        },
        name="sequential_control_flow",
        apply_default_optimizations=True)

  def benchmark_parallel_control_flow(self):
    dataset = dataset_ops.Dataset.from_tensors(
        random_ops.random_uniform([100, 10000000]))

    def fn(x):
      return map_fn.map_fn(
          lambda y: y * array_ops.transpose(y), x, parallel_iterations=10)

    num_elements = 1
    dataset = dataset.map(fn)
    self.run_and_report_benchmark(
        dataset,
        num_elements=1,
        extras={
            "model_name": "map.benchmark.9",
            "parameters": "%d" % num_elements,
        },
        name="parallel_control_flow",
        apply_default_optimizations=True)

  def _benchmark_nested_parallel_map(self, cycle_length, num_parallel_calls):
    k = 1024 * 1024
    num_map_elements = 10
    num_range_elements = 2000

    def g(_):
      return np.random.rand(50 * k).sum()

    def f(_):
      return dataset_ops.Dataset.range(num_map_elements).map(
          g, num_parallel_calls=num_parallel_calls)

    dataset = dataset_ops.Dataset.range(num_range_elements)
    dataset = dataset.interleave(
        f, cycle_length=cycle_length, num_parallel_calls=dataset_ops.AUTOTUNE)

    cycle_length_str = ("default"
                        if cycle_length is None else str(cycle_length))
    num_parallel_calls_str = ("autotune"
                              if num_parallel_calls == dataset_ops.AUTOTUNE else
                              str(num_parallel_calls))
    map_dataset_str = ("map" if num_parallel_calls is None else
                       "parallel_map_num_parallel_calls_%s" %
                       num_parallel_calls_str)

    self.run_and_report_benchmark(
        dataset,
        num_elements=num_map_elements * num_range_elements,
        extras={
            "model_name": "map.benchmark.10",
            "parameters": "%s_%s" % (cycle_length_str, num_parallel_calls_str),
        },
        name=("%s_cycle_length_%s" % (map_dataset_str, cycle_length_str)))

  def benchmark_nested_parallel_map(self):
    cycle_lengths = [None, 100]
    nums_parallel_calls = [None, 1, 10, 100, dataset_ops.AUTOTUNE]
    for cycle_length in cycle_lengths:
      for num_parallel_calls in nums_parallel_calls:
        self._benchmark_nested_parallel_map(cycle_length, num_parallel_calls)


if __name__ == "__main__":
  benchmark_base.test.main()
