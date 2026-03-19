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
"""Test utilities for tf.data benchmarking functionality."""
import timeit

import numpy as np

from tensorflow.python.client import session
from tensorflow.python.data.experimental.ops import testing
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.eager import context
from tensorflow.python.platform import test


class MetaBenchmark(test.Benchmark):
  """Benchmark that compares various ways of running tf.data benchmarks."""

  # Note that each of these benchmarks is a separate method so that we can
  # run them independently and collect a performance profile.

  def setup_fast_dataset(self):
    self.num_reps = 15
    self.iters = 100000
    options = options_lib.Options()
    options.experimental_optimization.apply_default_optimizations = False
    return dataset_ops.Dataset.range(10000**2).with_options(options)

  def benchmark_fast_dataset_with_only_cpp_iterations(self):
    dataset = self.setup_fast_dataset()
    self.run_benchmark_with_only_cpp_iterations(dataset)

  def benchmark_fast_dataset_with_session_run(self):
    dataset = self.setup_fast_dataset()
    self.run_benchmark_with_session_run(dataset)

  def benchmark_fast_dataset_with_session_callable(self):
    dataset = self.setup_fast_dataset()
    self.run_benchmark_with_session_run(dataset, make_callable=True)

  def benchmark_fast_dataset_in_eager(self):
    with context.eager_mode():
      dataset = self.setup_fast_dataset()
      self.run_benchmark_in_eager(dataset)

  def setup_slow_dataset(self):
    dataset = self.setup_fast_dataset()
    self.iters = 1000
    # sleep for 1e-3s per iteration
    return dataset.apply(testing.sleep(1000))

  def benchmark_slow_dataset_with_only_cpp_iterations(self):
    dataset = self.setup_slow_dataset()
    self.run_benchmark_with_only_cpp_iterations(dataset)

  def benchmark_slow_dataset_with_session_run(self):
    dataset = self.setup_slow_dataset()
    self.run_benchmark_with_session_run(dataset)

  def benchmark_slow_dataset_with_session_callable(self):
    dataset = self.setup_slow_dataset()
    self.run_benchmark_with_session_run(dataset, make_callable=True)

  def benchmark_slow_dataset_in_eager(self):
    with context.eager_mode():
      dataset = self.setup_slow_dataset()
      self.run_benchmark_in_eager(dataset)

  def report(self, deltas):
    # Each `delta` is the time taken for `self.iters` iterations. Divide by the
    # number of iterations here to get per-element iteration time.
    deltas = np.array(deltas) / self.iters
    # Discard the first 5 results from "warming up" the session.
    deltas = deltas[5:]

    median = np.median(deltas)
    mean = np.mean(deltas)
    min_val = np.min(deltas)
    max_val = np.max(deltas)
    extras = {
        "iters_per_second": 1 / median,
        "median": median,
        "mean": mean,
        "min": min_val,
        "max": max_val,
        "num_reps": self.num_reps - 5,
    }
    self.report_benchmark(wall_time=median, iters=self.iters, extras=extras)

  def run_benchmark_in_eager(self, dataset):
    deltas = []
    for _ in range(self.num_reps):
      iterator = iter(dataset)
      deltas.append(timeit.timeit(lambda: next(iterator), number=self.iters))  # pylint: disable=cell-var-from-loop

    self.report(deltas)

  def run_benchmark_with_session_run(self, dataset, make_callable=False):
    iterator = dataset_ops.make_initializable_iterator(dataset)
    next_element = iterator.get_next()

    with session.Session() as sess:
      deltas = []
      for _ in range(self.num_reps):
        if make_callable:
          get_next_element = sess.make_callable(next_element)
        else:
          # Note: session.run(next_element.op) is more performant than
          # session.run(next_element) because we avoid the cost of copying the
          # tensor from C++ to python.
          get_next_element = lambda: sess.run(next_element.op)

        sess.run(iterator.initializer)
        deltas.append(timeit.timeit(get_next_element, number=self.iters))
    self.report(deltas)

  def run_benchmark_with_only_cpp_iterations(self, dataset):
    """Benchmarks the dataset with the iterations performed in C++."""
    # NOTE: We use `dataset.skip()` to perform the iterations in C++, avoiding
    # the overhead of multiple `session.run()` calls. Note that this relies on
    # the underlying implementation of `skip`: if it is optimized in the future,
    # we will have to change this code.
    dataset = dataset.skip(self.iters - 1)
    iterator = dataset_ops.make_initializable_iterator(dataset)
    next_element = iterator.get_next()

    with session.Session() as sess:
      deltas = []
      for _ in range(self.num_reps):
        sess.run(iterator.initializer)
        deltas.append(
            timeit.timeit(lambda: sess.run(next_element.op), number=1))
    self.report(deltas)


if __name__ == "__main__":
  test.main()
