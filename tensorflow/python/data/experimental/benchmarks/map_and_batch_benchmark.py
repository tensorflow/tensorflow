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
import hashlib
import itertools

import numpy as np

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.data.benchmarks import benchmark_base
from tensorflow.python.data.experimental.ops import batching
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops

_NUMPY_RANDOM_SEED = 42


class MapAndBatchBenchmark(benchmark_base.DatasetBenchmarkBase):
  """Benchmarks for `tf.data.experimental.map_and_batch()`."""

  def benchmark_map_and_batch(self):
    """Measures the performance of parallelized batching."""
    shapes = [(), (10,), (10, 10), (10, 10, 10), (224, 224, 3)]
    batch_size_values = [1, 32, 64, 128, 1024]

    for shape in shapes:
      for batch_size in batch_size_values:

        dataset = dataset_ops.Dataset.range(1000000000)
        dense_value = random_ops.random_normal(shape=shape)

        dataset = dataset.apply(
            batching.map_and_batch(lambda _: dense_value, batch_size))  # pylint: disable=cell-var-from-loop
        options = options_lib.Options()
        options.experimental_optimization.apply_default_optimizations = False
        dataset = dataset.with_options(options)

        self.run_and_report_benchmark(
            dataset=dataset,
            num_elements=batch_size,
            iters=100,
            warmup=True,
            extras={
                "model_name": "map_and_batch.benchmark.1",
                "parameters": "%d.%s" % (batch_size, str(shape))
            },
            name="num_elements_%d_batch_size_%d" % (np.prod(shape), batch_size))

  def _benchmark_series(self, label, series, benchmark_id):
    """Runs benchmark the given series."""

    # Decides a proper number of iterations according to the inputs.
    def compute_num_iters(map_num_calls, inter_op, element_size, batch_size):
      return 1024 // (
          (element_size * batch_size) //
          min(12 if map_num_calls == dataset_ops.AUTOTUNE else map_num_calls,
              inter_op))

    # Makes the dataset based on the inputs.
    def make_dataset(map_num_calls, element_size, batch_size, batch_num_calls,
                     apply_fusion):
      k = 1024 * 1024
      x = constant_op.constant(np.random.rand(element_size, 4 * k))
      y = constant_op.constant(np.random.rand(4 * k, 1))
      dataset = dataset_ops.Dataset.range(1000000000000).map(lambda _: (x, y))
      dataset = dataset.map(math_ops.matmul, num_parallel_calls=map_num_calls)
      dataset = dataset.batch(
          batch_size=batch_size, num_parallel_calls=batch_num_calls)
      options = options_lib.Options()
      options.experimental_optimization.apply_default_optimizations = False
      options.experimental_optimization.map_and_batch_fusion = apply_fusion
      dataset = dataset.with_options(options)
      return dataset

    # Makes the name of the dataset based on the inputs.
    def make_name(label, map_num_calls, inter_op, element_size, batch_size,
                  batch_num_calls, apply_fusion):
      map_num_calls_str = ("autotuned" if map_num_calls == dataset_ops.AUTOTUNE
                           else str(map_num_calls))
      batch_num_calls_str = (
          "autotuned" if batch_num_calls == dataset_ops.AUTOTUNE else
          str(1 if batch_num_calls is None else batch_num_calls))
      name_str = ("%s_id_%s_map_num_calls_%s_batch_num_calls_%s_inter_op_%d"
                  "_elem_size_%d_batch_size_%d")
      name = (
          name_str % (
              "fused" if apply_fusion else "chained",
              hashlib.sha1((label).encode("utf-8")).hexdigest()[:8],
              map_num_calls_str,
              batch_num_calls_str,
              inter_op,
              element_size,
              batch_size,
          ))
      return name

    for (map_num_calls, inter_op, element_size, batch_size, batch_num_calls,
         apply_fusion) in series:
      num_iters = compute_num_iters(map_num_calls, inter_op, element_size,
                                    batch_size)
      dataset = make_dataset(map_num_calls, element_size, batch_size,
                             batch_num_calls, apply_fusion)
      name = make_name(label, map_num_calls, inter_op, element_size, batch_size,
                       batch_num_calls, apply_fusion)

      session_config = config_pb2.ConfigProto(
          inter_op_parallelism_threads=inter_op, use_per_session_threads=True)

      self.run_and_report_benchmark(
          dataset=dataset,
          iters=num_iters,
          num_elements=batch_size,
          warmup=True,
          extras={
              "model_name":
                  "map_and_batch.benchmark.%d" % benchmark_id,
              "parameters":
                  "%d.%d.%d.%d.%d.%s" %
                  (map_num_calls, inter_op, element_size, batch_size,
                   batch_num_calls, apply_fusion),
          },
          session_config=session_config,
          name=name)

  def benchmark_map_and_batch_chaining_versus_fusing(self):
    """Compares the performance of chaining and fusing map and batch.

    NOTE: It is recommended to build the benchmark with
    `-c opt --copt=-mavx --copt=-mavx2 --copt=-mfma --copt=-gmlt`
    and execute it on a machine with at least 32 CPU cores.
    """

    # Sequential pipeline configurations.
    seq_elem_size_series = itertools.product([1], [1], [1, 2, 4, 8], [16],
                                             [None], [False, True])
    seq_batch_size_series = itertools.product([1], [1], [1], [8, 16, 32, 64],
                                              [None], [False, True])

    # Parallel pipeline configuration.
    par_elem_size_series = itertools.product([32], [32], [1, 2, 4, 8], [256],
                                             [None], [False, True])
    par_batch_size_series = itertools.product([32], [32], [1],
                                              [128, 256, 512, 1024], [None],
                                              [False, True])
    par_map_num_calls_series = itertools.product([8, 16, 32, 64], [32], [1],
                                                 [512], [None], [False, True])
    par_inter_op_series = itertools.product([32], [8, 16, 32, 64], [1], [512],
                                            [None], [False, True])

    # Autotuned pipeline configuration.
    fused_versus_chained_series = [
        (dataset_ops.AUTOTUNE, 32, 1, 16, dataset_ops.AUTOTUNE, False),
        (dataset_ops.AUTOTUNE, 32, 1, 16, None, True)
    ]

    np.random.seed(_NUMPY_RANDOM_SEED)
    self._benchmark_series(
        "Sequential element size evaluation",
        seq_elem_size_series,
        benchmark_id=2)
    self._benchmark_series(
        "Sequential batch size evaluation",
        seq_batch_size_series,
        benchmark_id=3)
    self._benchmark_series(
        "Parallel element size evaluation",
        par_elem_size_series,
        benchmark_id=4)
    self._benchmark_series(
        "Parallel batch size evaluation", par_batch_size_series, benchmark_id=5)
    self._benchmark_series(
        "Transformation parallelism evaluation",
        par_map_num_calls_series,
        benchmark_id=6)
    self._benchmark_series(
        "Threadpool size evaluation", par_inter_op_series, benchmark_id=7)
    self._benchmark_series(
        "Autotune chained versus fused evaluation",
        fused_versus_chained_series,
        benchmark_id=8)


if __name__ == "__main__":
  benchmark_base.test.main()
