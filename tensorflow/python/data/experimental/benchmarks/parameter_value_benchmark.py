# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Benchmarks to compare effect of different paramemter values on the performance."""
import time
import numpy as np

from tensorflow.python.data.benchmarks import benchmark_base
from tensorflow.python.data.experimental.ops import testing
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import script_ops

# The maximum sleeping time for each output step and the input sleeping time in
# milliseconds.
max_output_sleep_ms = 0.5
input_sleep_ms = 1.5


def sleep_function(x):
  time.sleep(np.random.uniform(max_output_sleep_ms) / 1000)
  return x


def map_function(x):
  return script_ops.py_func(sleep_function, [x], x.dtype)


class ParameterValueBenchmark(benchmark_base.DatasetBenchmarkBase):
  """Benchmarks to compare effect of different paramemter values on the performance."""

  def _benchmark_map(self, num_parallel_calls, buffer_size):
    k = 1024 * 1024
    dataset = dataset_ops.Dataset.from_tensors(
        (np.random.rand(1, 4 * k), np.random.rand(4 * k, 1))).repeat()
    dataset = dataset.map(
        math_ops.matmul, num_parallel_calls=num_parallel_calls)
    dataset = dataset.map(map_function)
    dataset = dataset.prefetch(buffer_size=buffer_size)
    dataset = dataset.apply(testing.sleep(int(input_sleep_ms * 1000)))

    name_str = ("map_max_output_sleep_ms_%.2f_input_sleep_ms_%.2f_"
                "num_parallel_calls_%d_buffer_size_%d")
    return self.run_and_report_benchmark(
        dataset=dataset,
        num_elements=10000,
        name=name_str %
        (max_output_sleep_ms, input_sleep_ms, num_parallel_calls, buffer_size))

  def benchmark_map(self):
    nums_parallel_calls = [4, 8, 12]
    buffer_sizes = [10, 50, 100, 150, 200, 250, 300]

    parameters_list = []
    wall_time_map = {}

    for num_parallel_calls in nums_parallel_calls:
      for buffer_size in buffer_sizes:
        parameters = (num_parallel_calls, buffer_size)
        parameters_list.append(parameters)
        wall_time = self._benchmark_map(num_parallel_calls, buffer_size)
        wall_time_map[parameters] = wall_time

    parameters_list.sort(key=lambda x: wall_time_map[x])
    for parameters in parameters_list:
      print("num_parallel_calls_%d_buffer_size_%d_wall_time:" % parameters,
            wall_time_map[parameters])

  def _benchmark_map_and_batch(self, num_parallel_calls, buffer_size):
    batch_size = 16
    k = 1024 * 1024

    dataset = dataset_ops.Dataset.from_tensors(
        (np.random.rand(1, 4 * k), np.random.rand(4 * k, 1))).repeat()
    dataset = dataset.map(
        math_ops.matmul, num_parallel_calls=num_parallel_calls)
    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.map(map_function)
    dataset = dataset.prefetch(buffer_size=buffer_size)
    dataset = dataset.apply(testing.sleep(int(input_sleep_ms * 1000)))

    name_str = ("map_and_batch_max_output_sleep_ms_%.2f_input_sleep_ms_%.2f"
                "_num_parallel_calls_%d_buffer_size_%d")
    return self.run_and_report_benchmark(
        dataset=dataset,
        num_elements=1000,
        name=name_str %
        (max_output_sleep_ms, input_sleep_ms, num_parallel_calls, buffer_size))

  def benchmark_map_and_batch(self):
    nums_parallel_calls = [4, 8, 12]
    buffer_sizes = [10, 50, 100, 150, 200, 250, 300]

    parameters_list = []
    wall_time_map = {}

    for num_parallel_calls in nums_parallel_calls:
      for buffer_size in buffer_sizes:
        parameters = (num_parallel_calls, buffer_size)
        parameters_list.append(parameters)
        wall_time = self._benchmark_map_and_batch(num_parallel_calls,
                                                  buffer_size)
        wall_time_map[parameters] = wall_time

    parameters_list.sort(key=lambda x: wall_time_map[x])
    for parameters in parameters_list:
      print("num_parallel_calls_%d_buffer_size_%d_wall_time:" % parameters,
            wall_time_map[parameters])

  def _benchmark_interleave(self, num_parallel_calls, buffer_size):
    k = 1024 * 1024
    dataset = dataset_ops.Dataset.from_tensors(
        (np.random.rand(1, 4 * k), np.random.rand(4 * k, 1))).repeat()
    dataset = dataset.map(math_ops.matmul)
    dataset = dataset.map(map_function)
    dataset = dataset_ops.Dataset.range(1).repeat().interleave(
        lambda _: dataset,  # pylint: disable=cell-var-from-loop
        cycle_length=10,
        num_parallel_calls=num_parallel_calls)
    dataset = dataset.prefetch(buffer_size=buffer_size)
    dataset = dataset.apply(testing.sleep(int(input_sleep_ms * 1000)))

    name_str = ("interleave_max_output_sleep_ms_%.2f_input_sleep_ms_%.2f"
                "_num_parallel_calls_%d_buffer_size_%d")
    return self.run_and_report_benchmark(
        dataset=dataset,
        num_elements=10000,
        name=name_str %
        (max_output_sleep_ms, input_sleep_ms, num_parallel_calls, buffer_size))

  def benchmark_interleave(self):
    nums_parallel_calls = [4, 8, 10]
    buffer_sizes = [10, 50, 100, 150, 200, 250, 300]

    parameters_list = []
    wall_time_map = {}

    for num_parallel_calls in nums_parallel_calls:
      for buffer_size in buffer_sizes:
        parameters = (num_parallel_calls, buffer_size)
        parameters_list.append(parameters)
        wall_time = self._benchmark_interleave(num_parallel_calls, buffer_size)
        wall_time_map[parameters] = wall_time

    parameters_list.sort(key=lambda x: wall_time_map[x])
    for parameters in parameters_list:
      print("num_parallel_calls_%d_buffer_size_%d_wall_time:" % parameters,
            wall_time_map[parameters])


if __name__ == "__main__":
  benchmark_base.test.main()
