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

import time

import numpy as np

from tensorflow.python.client import session
from tensorflow.python.data.experimental.ops import interleave_ops
from tensorflow.python.data.experimental.ops import optimization
from tensorflow.python.data.experimental.ops import sleep
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import ops
from tensorflow.python.platform import test


def _make_fake_dataset_fn():
  """Returns a dataset that emulates a remote storage data source.

  Returns a dataset factory which creates a dataset with 100 elements that
  emulates the performance characteristic of a file-based dataset stored in a
  remote storage. In particular, the first element will take an order of
  magnitude longer to produce than the remaining elements (1s vs. 1ms).
  """

  def fake_dataset_fn(unused):
    del unused

    def make_dataset(time_us, num_elements):
      return dataset_ops.Dataset.range(num_elements).apply(sleep.sleep(time_us))

    return make_dataset(1000 * 1000, 0).concatenate(make_dataset(1000,
                                                                 100)).take(100)

  return fake_dataset_fn


class ParallelInterleaveBenchmark(test.Benchmark):
  """Benchmarks for `tf.data.experimental.parallel_interleave()`."""

  def _benchmark(self, dataset_fn, iters, num_elements):
    with ops.Graph().as_default():
      iterator = dataset_ops.make_one_shot_iterator(dataset_fn())
      next_element = iterator.get_next()
      with session.Session() as sess:
        deltas = []
        for _ in range(iters):
          start = time.time()
          for _ in range(num_elements):
            sess.run(next_element.op)
          end = time.time()
          deltas.append(end - start)

    mean_wall_time = np.mean(deltas) / num_elements
    self.report_benchmark(iters=iters, wall_time=mean_wall_time)

  def benchmark_sequential_interleave(self):

    def dataset_fn():
      return dataset_ops.Dataset.range(1).repeat().interleave(
          _make_fake_dataset_fn(), cycle_length=10)

    self._benchmark(dataset_fn=dataset_fn, iters=10, num_elements=100)

  def benchmark_parallel_interleave_v1(self):
    """Benchmark for parallel interleave that does not support autotuning."""

    def dataset_fn():
      return dataset_ops.Dataset.range(1).repeat().apply(
          interleave_ops.parallel_interleave(
              _make_fake_dataset_fn(), cycle_length=10))

    self._benchmark(dataset_fn=dataset_fn, iters=100, num_elements=1000)

  def benchmark_parallel_interleave_v2(self):
    """Benchmark for parallel interleave that supports autotuning."""

    def dataset_fn():
      return dataset_ops.Dataset.range(1).repeat().interleave(
          _make_fake_dataset_fn(),
          cycle_length=10, num_parallel_calls=optimization.AUTOTUNE)

    self._benchmark(dataset_fn=dataset_fn, iters=100, num_elements=1000)


if __name__ == "__main__":
  test.main()
