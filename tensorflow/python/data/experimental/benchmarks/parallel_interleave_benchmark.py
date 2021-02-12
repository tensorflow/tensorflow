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
"""Benchmarks for `tf.data.experimental.parallel_interleave()`."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from tensorflow.python.data.experimental.ops import interleave_ops
from tensorflow.python.data.experimental.ops import stats_aggregator
from tensorflow.python.data.experimental.ops import testing
from tensorflow.python.data.benchmarks import benchmark_base
from tensorflow.python.data.ops import dataset_ops

NON_PARALLEL = "non_parallel"
EXPERIMENTAL_PARALLEL = "experimental_parallel"
CORE_PARALLEL = "core_parallel"


def _make_fake_dataset_fn(initial_delay_us, remainder_delay_us):
  """Returns a dataset that emulates a remote storage data source.

  Returns a dataset factory which creates a dataset with 100 elements that
  emulates the performance characteristic of a file-based dataset stored in a
  remote storage. In particular, the first element will take an order of
  magnitude longer to produce than the remaining elements (100ms vs. 1ms).

  Args:
    initial_delay_us: How long to wait before producing the first element.
    remainder_delay_us: How long to wait before producing subsequent elements.
  """

  def fake_dataset_fn(unused):
    """Returns a function that creates a dataset with the specified delays."""
    del unused

    def make_dataset(time_us, num_elements):
      dataset = dataset_ops.Dataset.range(num_elements)
      if time_us > 0:
        dataset = dataset.apply(testing.sleep(time_us))
      return dataset

    if not initial_delay_us:
      return make_dataset(remainder_delay_us, 100)

    return make_dataset(initial_delay_us,
                        0).concatenate(make_dataset(remainder_delay_us, 100))

  return fake_dataset_fn


class ParallelInterleaveBenchmark(benchmark_base.DatasetBenchmarkBase):
  """Benchmarks for `tf.data.experimental.parallel_interleave()`."""

  def apply_interleave(self, interleave_version, dataset, interleave_fn,
                       cycle_length, num_parallel_calls):
    if interleave_version == NON_PARALLEL:
      return dataset.interleave(interleave_fn, cycle_length=cycle_length)
    elif interleave_version == EXPERIMENTAL_PARALLEL:
      return dataset.apply(
          interleave_ops.parallel_interleave(
              interleave_fn, cycle_length=cycle_length))
    elif interleave_version == CORE_PARALLEL:
      if not num_parallel_calls:
        num_parallel_calls = cycle_length
      return dataset.interleave(
          interleave_fn,
          cycle_length=cycle_length,
          num_parallel_calls=num_parallel_calls)
    else:
      raise ValueError("Unknown version: " + interleave_version)

  def make_dataset(self,
                   interleave_version,
                   initial_delay,
                   remainder_delay,
                   cycle_length,
                   num_parallel_calls=None):
    dataset = dataset_ops.Dataset.range(1).repeat()
    interleave_fn = _make_fake_dataset_fn(initial_delay, remainder_delay)
    return self.apply_interleave(
        interleave_version=interleave_version,
        dataset=dataset,
        interleave_fn=interleave_fn,
        cycle_length=cycle_length,
        num_parallel_calls=num_parallel_calls)

  def _benchmark(self,
                 interleave_version,
                 num_elements,
                 initial_delay_us=0,
                 remainder_delay_us=0,
                 cycle_length=10,
                 iters=100,
                 num_parallel_calls=None,
                 attach_stats_aggregator=False,
                 name=None):
    dataset = self.make_dataset(
        interleave_version=interleave_version,
        initial_delay=initial_delay_us,
        remainder_delay=remainder_delay_us,
        cycle_length=cycle_length,
        num_parallel_calls=num_parallel_calls)
    if attach_stats_aggregator:
      aggregator = stats_aggregator.StatsAggregator()
      opts = dataset_ops.Options()
      opts.experimental_stats.aggregator = aggregator
      dataset = dataset.with_options(opts)

    self.run_and_report_benchmark(
        dataset=dataset,
        num_elements=num_elements,
        iters=iters,
        warmup=True,
        name=name)

  def benchmark_remote_file_simulation(self):
    for version in [EXPERIMENTAL_PARALLEL, CORE_PARALLEL]:
      self._benchmark(
          interleave_version=version,
          initial_delay_us=100 * 1000,
          remainder_delay_us=1000,
          num_elements=5000,
          name="remote_file_simulation_" + version)

  def benchmark_fast_input(self):
    for version in [EXPERIMENTAL_PARALLEL, CORE_PARALLEL]:
      self._benchmark(
          interleave_version=version,
          num_elements=200000,
          name="fast_input_" + version)

  # Measure the overhead of parallel interleaves compared to non-parallel
  # interleave.
  def benchmark_single_cycle(self):
    for version in [NON_PARALLEL, EXPERIMENTAL_PARALLEL, CORE_PARALLEL]:
      self._benchmark(
          interleave_version=version,
          cycle_length=1,
          num_elements=200000,
          name="single_cycle_" + version)

  # Compare with a more reasonable cycle length. Experimental interleave
  # cannot be compared here because it sets num_parallel_calls = cycle_length.
  def benchmark_single_parallel_call(self):
    self._benchmark(
        interleave_version=CORE_PARALLEL,
        num_elements=200000,
        num_parallel_calls=1,
        name="single_parallel_call_" + CORE_PARALLEL)

  def benchmark_long_cycle(self):
    for version in [EXPERIMENTAL_PARALLEL, CORE_PARALLEL]:
      self._benchmark(
          interleave_version=version,
          cycle_length=1000,
          num_elements=100000,
          name="long_cycle_" + version)

  def benchmark_stats(self):
    self._benchmark(
        interleave_version=CORE_PARALLEL,
        cycle_length=50,
        num_elements=1000,
        name="stats",
        attach_stats_aggregator=True)


if __name__ == "__main__":
  benchmark_base.test.main()
