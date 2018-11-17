# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Benchmarks for `tf.data.experimental.map_and_batch()`."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.data.experimental.ops import batching
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.platform import test


class MapAndBatchBenchmark(test.Benchmark):
  """Benchmarks for `tf.data.experimental.map_and_batch()`."""

  def benchmarkMapAndBatchDense(self):
    """Measures the performance of parallelized batching."""
    shapes = [(), (10,), (10, 10), (10, 10, 10), (224, 224, 3)]
    batch_size_values = [1, 32, 64, 128, 1024]

    shape_placeholder = array_ops.placeholder(dtypes.int64, shape=[None])
    batch_size_placeholder = array_ops.placeholder(dtypes.int64, shape=[])

    dataset = dataset_ops.Dataset.range(1000000000)

    dense_value = random_ops.random_normal(shape=shape_placeholder)

    dataset = dataset.apply(batching.map_and_batch(
        lambda _: dense_value, batch_size_placeholder))
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()

    for shape in shapes:
      for batch_size in batch_size_values:

        with session.Session() as sess:
          sess.run(iterator.initializer, feed_dict={
              shape_placeholder: shape, batch_size_placeholder: batch_size})

          # Use a C++ callable to minimize the Python overhead in the benchmark.
          callable_opts = config_pb2.CallableOptions()
          callable_opts.target.append(next_element.op.name)
          op_callable = sess._make_callable_from_options(callable_opts)  # pylint: disable=protected-access

          # Run five steps to warm up the session caches before taking the
          # first measurement.
          for _ in range(5):
            op_callable()
          deltas = []
          overall_start = time.time()
          # Run at least five repetitions and for at least five seconds.
          while len(deltas) < 5 or time.time() - overall_start < 5.0:
            start = time.time()
            for _ in range(100):
              op_callable()
            end = time.time()
            deltas.append(end - start)
          del op_callable

        median_wall_time = np.median(deltas) / 100.0
        iters = len(deltas) * 100

        print("Map and batch dense dataset shape: %r batch_size: %d "
              "wall time: %f (%d iters)"
              % (shape, batch_size, median_wall_time, iters))
        self.report_benchmark(
            iters=iters, wall_time=median_wall_time,
            name="benchmark_batch_dense_dataset_nnz_%d_batch_size_%d" % (
                np.prod(shape), batch_size))


if __name__ == "__main__":
  test.main()
