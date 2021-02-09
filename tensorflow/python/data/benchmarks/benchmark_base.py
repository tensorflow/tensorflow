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

from tensorflow.python.eager import context
from tensorflow.python.client import session
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import nest
from tensorflow.python.platform import test


class DatasetBenchmarkBase(test.Benchmark):
  """Base class for dataset benchmarks."""

  def run_benchmark(self,
                    dataset,
                    num_elements,
                    iters=1,
                    warmup=True,
                    apply_default_optimizations=False):
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

    Returns:
      A float, representing the per-element wall time of the dataset in seconds.
      This is the median time (with respect to `iters`) it takes for the dataset
      to go through `num_elements` elements, divided by `num_elements.`
    """

    # The options that have been applied to the dataset are preserved so that
    # they are not overwritten while benchmarking.
    options = dataset.options()
    options.experimental_optimization.apply_default_optimizations = (
        apply_default_optimizations)
    dataset = dataset.with_options(options)

    # NOTE: We use `dataset.skip()` to perform the iterations in C++, avoiding
    # the overhead of having to execute a TensorFlow op for each step of the input
    # pipeline. Note that this relies on the underlying implementation of `skip`
    # to execute upstream computation. If it is optimized in the future,
    # we will have to change this code.
    dataset = dataset.skip(num_elements - 1)
    deltas = []

    if context.executing_eagerly():
      for _ in range(iters):
        if warmup:
          iterator = iter(dataset)
          next(iterator)

        iterator = iter(dataset)
        start = time.time()
        next(iterator)
        end = time.time()
        deltas.append(end - start)
      return np.median(deltas) / float(num_elements)

    iterator = dataset_ops.make_initializable_iterator(dataset)
    next_element = iterator.get_next()
    next_element = nest.flatten(next_element)[0]

    for _ in range(iters):
      with session.Session() as sess:
        if warmup:
          # Run once to warm up the session caches.
          sess.run(iterator.initializer)
          sess.run(next_element.op)

        sess.run(iterator.initializer)
        start = time.time()
        sess.run(next_element.op)
        end = time.time()
      deltas.append(end - start)
    return np.median(deltas) / float(num_elements)

  def run_and_report_benchmark(self,
                               dataset,
                               num_elements,
                               name,
                               iters=5,
                               extras=None,
                               warmup=True,
                               apply_default_optimizations=False):
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
        apply_default_optimizations=apply_default_optimizations)
    if context.executing_eagerly():
      name = "{}.eager".format(name)
    else:
      name = "{}.graph".format(name)
    if extras is None:
      extras = {}
    extras["num_elements"] = num_elements
    self.report_benchmark(
        wall_time=wall_time, iters=iters, name=name, extras=extras)
    return wall_time
