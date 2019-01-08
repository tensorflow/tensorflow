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
"""Benchmarks for `tf.data.experimental.unbatch()`."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np

from tensorflow.python.client import session
from tensorflow.python.data.experimental.ops import batching
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


class UnbatchBenchmark(test.Benchmark):
  """Benchmarks for `tf.data.experimental.unbatch()`."""

  def benchmarkNativeUnbatch(self):
    batch_sizes = [1, 2, 5, 10, 20, 50]
    elems_per_trial = 10000
    with ops.Graph().as_default():
      dataset = dataset_ops.Dataset.from_tensors("element").repeat(None)
      batch_size_placeholder = array_ops.placeholder(dtypes.int64, shape=[])
      dataset = dataset.batch(batch_size_placeholder)
      dataset = dataset.apply(batching.unbatch())
      dataset = dataset.skip(elems_per_trial)
      iterator = dataset_ops.make_initializable_iterator(dataset)
      next_element = iterator.get_next()

      with session.Session() as sess:
        for batch_size in batch_sizes:
          deltas = []
          for _ in range(5):
            sess.run(
                iterator.initializer,
                feed_dict={batch_size_placeholder: batch_size})
            start = time.time()
            sess.run(next_element.op)
            end = time.time()
            deltas.append((end - start) / elems_per_trial)

          median_wall_time = np.median(deltas)
          print("Unbatch (native) batch size: %d Median wall time per element:"
                " %f microseconds" % (batch_size, median_wall_time * 1e6))
          self.report_benchmark(
              iters=10000,
              wall_time=median_wall_time,
              name="native_batch_size_%d" %
              batch_size)

  # Include a benchmark of the previous `unbatch()` implementation that uses
  # a composition of more primitive ops. Eventually we'd hope to generate code
  # that is as good in both cases.
  def benchmarkOldUnbatchImplementation(self):
    batch_sizes = [1, 2, 5, 10, 20, 50]
    elems_per_trial = 10000
    with ops.Graph().as_default():
      dataset = dataset_ops.Dataset.from_tensors("element").repeat(None)
      batch_size_placeholder = array_ops.placeholder(dtypes.int64, shape=[])
      dataset = dataset.batch(batch_size_placeholder)
      dataset = dataset.flat_map(dataset_ops.Dataset.from_tensor_slices)
      dataset = dataset.skip(elems_per_trial)
      iterator = dataset_ops.make_initializable_iterator(dataset)
      next_element = iterator.get_next()

      with session.Session() as sess:
        for batch_size in batch_sizes:
          deltas = []
          for _ in range(5):
            sess.run(
                iterator.initializer,
                feed_dict={batch_size_placeholder: batch_size})
            start = time.time()
            sess.run(next_element.op)
            end = time.time()
            deltas.append((end - start) / elems_per_trial)

          median_wall_time = np.median(deltas)
          print("Unbatch (unfused) batch size: %d Median wall time per element:"
                " %f microseconds" % (batch_size, median_wall_time * 1e6))
          self.report_benchmark(
              iters=10000,
              wall_time=median_wall_time,
              name="unfused_batch_size_%d" %
              batch_size)


if __name__ == "__main__":
  test.main()
