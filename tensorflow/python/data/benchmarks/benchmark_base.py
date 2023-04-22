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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np

from tensorflow.python.client import session
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import nest
from tensorflow.python.eager import context
from tensorflow.python.platform import test


class DatasetBenchmarkBase(test.Benchmark):
  """Base class for dataset benchmarks."""

  def _run_eager_benchmark(self, iterable, iters, warmup):
    """Benchmark the iterable in eager mode.

    Runs the iterable `iters` times. In each iteration, the benchmark measures
    the time it takes to go execute the iterable.

    Args:
      iterable: The tf op or tf.data Dataset to benchmark.
      iters: Number of times to repeat the timing.
      warmup: If true, warms up the session caches by running an untimed run.

    Returns:
      A float, representing the median time (with respect to `iters`)
      it takes for the iterable to be executed `iters` num of times.

    Raises:
      RuntimeError: When executed in graph mode.
    """

    deltas = []
    if not context.executing_eagerly():
      raise RuntimeError(
          "Eager mode benchmarking is not supported in graph mode.")

    for _ in range(iters):
      if warmup:
        iterator = iter(iterable)
        next(iterator)

      iterator = iter(iterable)
      start = time.time()
      next(iterator)
      end = time.time()
      deltas.append(end - start)
    return np.median(deltas)

  def _run_graph_benchmark(self,
                           iterable,
                           iters,
                           warmup,
                           session_config,
                           initializer=None):
    """Benchmarks the iterable in graph mode.

    Runs the iterable `iters` times. In each iteration, the benchmark measures
    the time it takes to go execute the iterable.

    Args:
      iterable: The tf op or tf.data Dataset to benchmark.
      iters: Number of times to repeat the timing.
      warmup: If true, warms up the session caches by running an untimed run.
      session_config: A ConfigProto protocol buffer with configuration options
        for the session. Applicable only for benchmarking in graph mode.
      initializer: The initializer op required to initialize the iterable.

    Returns:
      A float, representing the median time (with respect to `iters`)
      it takes for the iterable to be executed `iters` num of times.

    Raises:
      RuntimeError: When executed in eager mode.
    """

    deltas = []
    if context.executing_eagerly():
      raise RuntimeError(
          "Graph mode benchmarking is not supported in eager mode.")

    for _ in range(iters):
      with session.Session(config=session_config) as sess:
        if warmup:
          # Run once to warm up the session caches.
          if initializer:
            sess.run(initializer)
          sess.run(iterable)

        if initializer:
          sess.run(initializer)
        start = time.time()
        sess.run(iterable)
        end = time.time()
      deltas.append(end - start)
    return np.median(deltas)

  def run_op_benchmark(self, op, iters=1, warmup=True, session_config=None):
    """Benchmarks the op.

    Runs the op `iters` times. In each iteration, the benchmark measures
    the time it takes to go execute the op.

    Args:
      op: The tf op to benchmark.
      iters: Number of times to repeat the timing.
      warmup: If true, warms up the session caches by running an untimed run.
      session_config: A ConfigProto protocol buffer with configuration options
        for the session. Applicable only for benchmarking in graph mode.

    Returns:
      A float, representing the per-execution wall time of the op in seconds.
      This is the median time (with respect to `iters`) it takes for the op
      to be executed `iters` num of times.
    """

    if context.executing_eagerly():
      return self._run_eager_benchmark(iterable=op, iters=iters, warmup=warmup)

    return self._run_graph_benchmark(
        iterable=op, iters=iters, warmup=warmup, session_config=session_config)

  def run_benchmark(self,
                    dataset,
                    num_elements,
                    iters=1,
                    warmup=True,
                    apply_default_optimizations=False,
                    session_config=None):
    """Benchmarks the dataset.

    Runs the dataset `iters` times. In each iteration, the benchmark measures
    the time it takes to go through `num_elements` elements of the dataset.

    Args:
      dataset: Dataset to benchmark.
      num_elements: Number of dataset elements to iterate through each benchmark
        iteration.
      iters: Number of times to repeat the timing.
      warmup: If true, warms up the session caches by running an untimed run.
      apply_default_optimizations: Determines whether default optimizations
        should be applied.
      session_config: A ConfigProto protocol buffer with configuration options
        for the session. Applicable only for benchmarking in graph mode.

    Returns:
      A float, representing the per-element wall time of the dataset in seconds.
      This is the median time (with respect to `iters`) it takes for the dataset
      to go through `num_elements` elements, divided by `num_elements.`
    """

    # The options that have been applied to the dataset are preserved so that
    # they are not overwritten while benchmarking.
    options = dataset_ops.Options()
    options.experimental_optimization.apply_default_optimizations = (
        apply_default_optimizations)
    dataset = dataset.with_options(options)

    # NOTE: We use `dataset.skip()` to perform the iterations in C++, avoiding
    # the overhead of having to execute a TensorFlow op for each step of the
    # input pipeline. Note that this relies on the underlying implementation of
    # `skip` to execute upstream computation. If it is optimized in the future,
    # we will have to change this code.
    dataset = dataset.skip(num_elements - 1)

    if context.executing_eagerly():
      median_duration = self._run_eager_benchmark(
          iterable=dataset, iters=iters, warmup=warmup)
      return median_duration / float(num_elements)

    iterator = dataset_ops.make_initializable_iterator(dataset)
    next_element = iterator.get_next()
    op = nest.flatten(next_element)[0].op
    median_duration = self._run_graph_benchmark(
        iterable=op,
        iters=iters,
        warmup=warmup,
        session_config=session_config,
        initializer=iterator.initializer)
    return median_duration / float(num_elements)

  def run_and_report_benchmark(self,
                               dataset,
                               num_elements,
                               name,
                               iters=5,
                               extras=None,
                               warmup=True,
                               apply_default_optimizations=False,
                               session_config=None):
    """Benchmarks the dataset and reports the stats.

    Runs the dataset `iters` times. In each iteration, the benchmark measures
    the time it takes to go through `num_elements` elements of the dataset.
    This is followed by logging/printing the benchmark stats.

    Args:
      dataset: Dataset to benchmark.
      num_elements: Number of dataset elements to iterate through each benchmark
        iteration.
      name: Name of the benchmark.
      iters: Number of times to repeat the timing.
      extras: A dict which maps string keys to additional benchmark info.
      warmup: If true, warms up the session caches by running an untimed run.
      apply_default_optimizations: Determines whether default optimizations
        should be applied.
      session_config: A ConfigProto protocol buffer with configuration options
        for the session. Applicable only for benchmarking in graph mode.

    Returns:
      A float, representing the per-element wall time of the dataset in seconds.
      This is the median time (with respect to `iters`) it takes for the dataset
      to go through `num_elements` elements, divided by `num_elements.`
    """
    wall_time = self.run_benchmark(
        dataset=dataset,
        num_elements=num_elements,
        iters=iters,
        warmup=warmup,
        apply_default_optimizations=apply_default_optimizations,
        session_config=session_config)
    if extras is None:
      extras = {}
    if context.executing_eagerly():
      name = "{}.eager".format(name)
      extras["implementation"] = "eager"
    else:
      name = "{}.graph".format(name)
      extras["implementation"] = "graph"
    extras["num_elements"] = num_elements
    self.report_benchmark(
        wall_time=wall_time, iters=iters, name=name, extras=extras)
    return wall_time
