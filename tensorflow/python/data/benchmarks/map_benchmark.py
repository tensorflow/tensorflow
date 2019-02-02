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
"""Bechmarks for `tf.data.Dataset.map()`."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np

from tensorflow.python.client import session
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import ops
from tensorflow.python.platform import test


# TODO(b/119837791): Add eager benchmarks.
class MapBenchmark(test.Benchmark):
  """Bechmarks for `tf.data.Dataset.map()`."""

  def benchmarkChainOfMaps(self):
    chain_lengths = [0, 1, 2, 5, 10, 20, 50]
    for chain_length in chain_lengths:
      for mode in ["general", "single-threaded", "short-circuit"]:
        if mode == "general":
          map_fn = lambda x: x + 1
          use_inter_op_parallelism = True
          benchmark_label = ""
        if mode == "single-threaded":
          map_fn = lambda x: x + 1
          use_inter_op_parallelism = False
          benchmark_label = "_single_threaded"
        if mode == "short-circuit":
          map_fn = lambda x: x
          use_inter_op_parallelism = True  # should not have any significance
          benchmark_label = "_short_circuit"

        with ops.Graph().as_default():
          dataset = dataset_ops.Dataset.from_tensors(0).repeat(None)
          for _ in range(chain_length):
            dataset = dataset_ops.MapDataset(
                dataset,
                map_fn,
                use_inter_op_parallelism=use_inter_op_parallelism)
          options = dataset_ops.Options()
          options.experimental_optimization.apply_default_optimizations = False
          dataset = dataset.with_options(options)
          iterator = dataset_ops.make_one_shot_iterator(dataset)
          next_element = iterator.get_next()

          with session.Session() as sess:
            for _ in range(5):
              sess.run(next_element.op)
            deltas = []
            for _ in range(100):
              start = time.time()
              for _ in range(100):
                sess.run(next_element.op)
              end = time.time()
              deltas.append(end - start)

            median_wall_time = np.median(deltas) / 100
            self.report_benchmark(
                iters=1000,
                wall_time=median_wall_time,
                name="chain_length_%d%s" % (chain_length, benchmark_label))

  def benchmarkMapFanOut(self):
    fan_outs = [1, 2, 5, 10, 20, 50, 100]
    for fan_out in fan_outs:
      for mode in ["general", "single-threaded", "short-circuit"]:
        if mode == "general":
          map_fn = lambda *xs: [x + 1 for x in xs]
          use_inter_op_parallelism = True
          benchmark_label = ""
        if mode == "single-threaded":
          map_fn = lambda *xs: [x + 1 for x in xs]
          use_inter_op_parallelism = False
          benchmark_label = "_single_threaded"
        if mode == "short-circuit":
          map_fn = lambda *xs: xs
          use_inter_op_parallelism = True  # should not have any significance
          benchmark_label = "_short_circuit"

        with ops.Graph().as_default():
          dataset = dataset_ops.Dataset.from_tensors(
              tuple(0 for _ in range(fan_out))).repeat(None)
          dataset = dataset_ops.MapDataset(
              dataset,
              map_fn,
              use_inter_op_parallelism=use_inter_op_parallelism)
          options = dataset_ops.Options()
          options.experimental_optimization.apply_default_optimizations = False
          dataset = dataset.with_options(options)
          iterator = dataset_ops.make_one_shot_iterator(dataset)
          next_element = iterator.get_next()

          with session.Session() as sess:
            for _ in range(5):
              sess.run(next_element[0].op)
            deltas = []
            for _ in range(100):
              start = time.time()
              for _ in range(100):
                sess.run(next_element[0].op)
              end = time.time()
              deltas.append(end - start)

            median_wall_time = np.median(deltas) / 100
            self.report_benchmark(
                iters=1000,
                wall_time=median_wall_time,
                name="fan_out_%d%s" % (fan_out, benchmark_label))


if __name__ == "__main__":
  test.main()
