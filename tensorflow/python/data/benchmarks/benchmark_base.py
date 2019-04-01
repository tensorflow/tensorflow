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
from tensorflow.python.platform import test


# TODO(b/119837791): Add eager benchmarks.
class DatasetBenchmarkBase(test.Benchmark):
  """Base class for dataset benchmarks."""

  def run_benchmark(self, dataset, num_elements, iters=1):
    """Benchmarks the dataset.

    Runs the dataset `iters` times. In each iteration, the benchmark measures
    the time it takes to go through `num_elements` elements of the dataset.

    Args:
      dataset: Dataset to benchmark.
      num_elements: Number of dataset elements to iterate through each benchmark
        iteration.
      iters: Number of times to repeat the timing.

    Returns:
      A float, representing the per-element wall time of the dataset in seconds.
      This is the median time (with respect to `iters`) it takes for the dataset
      to go through `num_elements` elements, divided by `num_elements.`
    """
    options = dataset_ops.Options()
    options.experimental_optimization.apply_default_optimizations = False
    dataset = dataset.with_options(options)
    # NOTE: We use `dataset.skip()` to perform the iterations in C++, avoiding
    # the overhead of multiple `session.run()` calls. Note that this relies on
    # the underlying implementation of `skip`: if it is optimized in the future,
    # we will have to change this code.
    dataset = dataset.skip(num_elements - 1)
    iterator = dataset_ops.make_initializable_iterator(dataset)
    next_element = iterator.get_next()
    next_element = nest.flatten(next_element)[0]

    deltas = []
    for _ in range(iters):
      with session.Session() as sess:
        # Run once to warm up the session caches.
        sess.run(iterator.initializer)
        sess.run(next_element)

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
                               extras=None):
    # Measure the per-element wall time.
    wall_time = self.run_benchmark(dataset, num_elements, iters)

    if extras is None:
      extras = {}
    extras["num_elements"] = num_elements
    self.report_benchmark(
        wall_time=wall_time, iters=iters, name=name, extras=extras)
