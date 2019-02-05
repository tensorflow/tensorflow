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
"""Benchmarks for `tf.data.Dataset.batch()`."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np

from tensorflow.python.client import session
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


# TODO(b/119837791): Add eager benchmarks.
class BatchBenchmark(test.Benchmark):
  """Benchmarks for `tf.data.Dataset.batch()`."""

  def benchmarkBatchSparse(self):
    non_zeros_per_row_values = [0, 1, 5, 10, 100]
    batch_size_values = [1, 32, 64, 128, 1024]

    sparse_placeholder = array_ops.sparse_placeholder(dtype=dtypes.int64)
    batch_size_placeholder = array_ops.placeholder(dtype=dtypes.int64, shape=[])

    dataset = dataset_ops.Dataset.from_tensors(sparse_placeholder).repeat(
        ).batch(batch_size_placeholder)
    options = dataset_ops.Options()
    options.experimental_optimization.apply_default_optimizations = False
    dataset = dataset.with_options(options)
    iterator = dataset_ops.make_initializable_iterator(dataset)
    next_element = iterator.get_next()

    for non_zeros_per_row in non_zeros_per_row_values:

      sparse_value = sparse_tensor.SparseTensorValue(
          indices=np.arange(non_zeros_per_row, dtype=np.int64)[:, np.newaxis],
          values=np.arange(non_zeros_per_row, dtype=np.int64),
          dense_shape=[1000])

      for batch_size in batch_size_values:

        with session.Session() as sess:
          sess.run(iterator.initializer, feed_dict={
              sparse_placeholder: sparse_value,
              batch_size_placeholder: batch_size})
          # Run five steps to warm up the session caches before taking the
          # first measurement.
          for _ in range(5):
            sess.run(next_element.indices.op)
          deltas = []
          for _ in range(100):
            start = time.time()
            for _ in range(100):
              sess.run(next_element.indices.op)
            end = time.time()
            deltas.append(end - start)

        median_wall_time = np.median(deltas) / 100.0

        self.report_benchmark(
            iters=10000,
            wall_time=median_wall_time,
            name="sparse_num_elements_%d_batch_size_%d" %
            (non_zeros_per_row, batch_size))


if __name__ == "__main__":
  test.main()
