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

import hashlib
import itertools
import time

import numpy as np

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.data.experimental.ops import batching
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.platform import test

_NUMPY_RANDOM_SEED = 42


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

  def benchmarkMapAndBatchChainingVersusFusing(self):
    """Compares the performance of chaining and fusing map and batch.

    NOTE: It is recommended to build the benchmark with
    `-c opt --copt=-mavx --copt=-mavx2 --copt=-mfma --copt=-gmlt`
    and execute it on a machine with at least 32 CPU cores.
    """

    # Sequential pipeline configurations.
    seq_elem_size_series = itertools.product([1], [1], [1, 2, 4, 8], [16])
    seq_batch_size_series = itertools.product([1], [1], [1], [8, 16, 32, 64])

    # Parallel pipeline configuration.
    par_elem_size_series = itertools.product([32], [32], [1, 2, 4, 8], [256])
    par_batch_size_series = itertools.product([32], [32], [1],
                                              [128, 256, 512, 1024])
    par_num_calls_series = itertools.product([8, 16, 32, 64], [32], [1], [512])
    par_inter_op_series = itertools.product([32], [8, 16, 32, 64], [1], [512])

    def name(method, label, num_calls, inter_op, element_size, batch_size):
      return ("%s_id_%s_num_calls_%d_inter_op_%d_elem_size_%d_batch_size_%d" % (
          method,
          hashlib.sha1(label).hexdigest()[:8],
          num_calls,
          inter_op,
          element_size,
          batch_size,
      ))

    def benchmark(label, series):
      """Runs benchmark the given series."""

      print("%s:" % label)

      def make_base_dataset(element_size):
        k = 1024 * 1024
        x = constant_op.constant(np.random.rand(element_size, 4 * k))
        y = constant_op.constant(np.random.rand(4 * k, 1))
        return dataset_ops.Dataset.range(1000000000000).map(lambda _: (x, y))

      for num_calls, inter_op, element_size, batch_size in series:

        num_iters = 1024 // (
            (element_size * batch_size) // min(num_calls, inter_op))
        dataset = make_base_dataset(element_size)
        chained_dataset = dataset.map(
            math_ops.matmul,
            num_parallel_calls=num_calls).batch(batch_size=batch_size)
        chained_iterator = chained_dataset.make_one_shot_iterator()
        chained_get_next = chained_iterator.get_next()

        chained_deltas = []
        with session.Session(
            config=config_pb2.ConfigProto(
                inter_op_parallelism_threads=inter_op,
                use_per_session_threads=True)) as sess:
          for _ in range(5):
            sess.run(chained_get_next.op)
          for _ in range(num_iters):
            start = time.time()
            sess.run(chained_get_next.op)
            end = time.time()
            chained_deltas.append(end - start)

        fused_dataset = dataset.apply(
            batching.map_and_batch(
                math_ops.matmul,
                num_parallel_calls=num_calls,
                batch_size=batch_size))
        fused_iterator = fused_dataset.make_one_shot_iterator()
        fused_get_next = fused_iterator.get_next()

        fused_deltas = []
        with session.Session(
            config=config_pb2.ConfigProto(
                inter_op_parallelism_threads=inter_op,
                use_per_session_threads=True)) as sess:

          for _ in range(5):
            sess.run(fused_get_next.op)
          for _ in range(num_iters):
            start = time.time()
            sess.run(fused_get_next.op)
            end = time.time()
            fused_deltas.append(end - start)

        print(
            "batch size: %d, num parallel calls: %d, inter-op parallelism: %d, "
            "element size: %d, num iters: %d\nchained wall time: %f (median), "
            "%f (mean), %f (stddev), %f (min), %f (max)\n  fused wall time: "
            "%f (median), %f (mean), %f (stddev), %f (min), %f (max)\n    "
            "chained/fused:    %.2fx (median),    %.2fx (mean)" %
            (batch_size, num_calls, inter_op, element_size, num_iters,
             np.median(chained_deltas), np.mean(chained_deltas),
             np.std(chained_deltas), np.min(chained_deltas),
             np.max(chained_deltas), np.median(fused_deltas),
             np.mean(fused_deltas), np.std(fused_deltas), np.min(fused_deltas),
             np.max(fused_deltas),
             np.median(chained_deltas) / np.median(fused_deltas),
             np.mean(chained_deltas) / np.mean(fused_deltas)))

        self.report_benchmark(
            iters=num_iters,
            wall_time=np.median(chained_deltas),
            name=name("chained", label, num_calls, inter_op, element_size,
                      batch_size))

        self.report_benchmark(
            iters=num_iters,
            wall_time=np.median(fused_deltas),
            name=name("fused", label, num_calls, inter_op, element_size,
                      batch_size))

      print()

    np.random.seed(_NUMPY_RANDOM_SEED)
    benchmark("Sequential element size evaluation", seq_elem_size_series)
    benchmark("Sequential batch size evaluation", seq_batch_size_series)
    benchmark("Parallel element size evaluation", par_elem_size_series)
    benchmark("Parallel batch size evaluation", par_batch_size_series)
    benchmark("Transformation parallelism evaluation", par_num_calls_series)
    benchmark("Threadpool size evaluation", par_inter_op_series)


if __name__ == "__main__":
  test.main()
