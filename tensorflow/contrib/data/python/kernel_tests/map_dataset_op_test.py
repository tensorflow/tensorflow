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
"""Tests for the experimental input pipeline ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import itertools
import os
import time

import numpy as np

from tensorflow.contrib.data.python.ops import batching
from tensorflow.contrib.data.python.ops import error_ops
from tensorflow.contrib.data.python.ops import optimization
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import io_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test
from tensorflow.python.util import compat

_NUMPY_RANDOM_SEED = 42


class MapDatasetTest(test.TestCase):

  def testMapIgnoreError(self):
    components = np.array([1., 2., 3., np.nan, 5.]).astype(np.float32)

    dataset = (
        dataset_ops.Dataset.from_tensor_slices(components)
        .map(lambda x: array_ops.check_numerics(x, "message")).apply(
            error_ops.ignore_errors()))
    iterator = dataset.make_initializable_iterator()
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.test_session() as sess:
      sess.run(init_op)
      for x in [1., 2., 3., 5.]:
        self.assertEqual(x, sess.run(get_next))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  def testParallelMapIgnoreError(self):
    components = np.array([1., 2., 3., np.nan, 5.]).astype(np.float32)

    dataset = (
        dataset_ops.Dataset.from_tensor_slices(components).map(
            lambda x: array_ops.check_numerics(x, "message"),
            num_parallel_calls=2).prefetch(2).apply(error_ops.ignore_errors()))
    iterator = dataset.make_initializable_iterator()
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.test_session() as sess:
      sess.run(init_op)
      for x in [1., 2., 3., 5.]:
        self.assertEqual(x, sess.run(get_next))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  def testReadFileIgnoreError(self):
    def write_string_to_file(value, filename):
      with open(filename, "w") as f:
        f.write(value)

    filenames = [
        os.path.join(self.get_temp_dir(), "file_%d.txt" % i) for i in range(5)
    ]
    for filename in filenames:
      write_string_to_file(filename, filename)

    dataset = (
        dataset_ops.Dataset.from_tensor_slices(filenames).map(
            io_ops.read_file,
            num_parallel_calls=2).prefetch(2).apply(error_ops.ignore_errors()))
    iterator = dataset.make_initializable_iterator()
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.test_session() as sess:
      # All of the files are present.
      sess.run(init_op)
      for filename in filenames:
        self.assertEqual(compat.as_bytes(filename), sess.run(get_next))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

      # Delete one of the files.
      os.remove(filenames[0])

      # Attempting to read filenames[0] will fail, but ignore_errors()
      # will catch the error.
      sess.run(init_op)
      for filename in filenames[1:]:
        self.assertEqual(compat.as_bytes(filename), sess.run(get_next))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  def testCaptureResourceInMapFn(self):

    def _build_ds(iterator):

      def _map_fn(x):
        get_next = iterator.get_next()
        return x * get_next

      return dataset_ops.Dataset.range(10).map(_map_fn)

    def _build_graph():
      captured_iterator = dataset_ops.Dataset.range(
          10).make_initializable_iterator()
      ds = _build_ds(captured_iterator)
      iterator = ds.make_initializable_iterator()
      init_op = iterator.initializer
      get_next = iterator.get_next()
      return captured_iterator.initializer, init_op, get_next

    with ops.Graph().as_default() as g:
      captured_init_op, init_op, get_next = _build_graph()
      with self.test_session(graph=g) as sess:
        sess.run(captured_init_op)
        sess.run(init_op)
        for i in range(10):
          self.assertEquals(i * i, sess.run(get_next))
        with self.assertRaises(errors.OutOfRangeError):
          sess.run(get_next)


class MapDatasetBenchmark(test.Benchmark):

  # The purpose of this benchmark is to compare the performance of chaining vs
  # fusing of the map and batch transformations across various configurations.
  #
  # NOTE: It is recommended to build the benchmark with
  # `-c opt --copt=-mavx --copt=-mavx2 --copt=-mfma --copt=-gmlt`
  # and execute it on a machine with at least 32 CPU cores.
  def benchmarkMapAndBatch(self):

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
          hashlib.sha1(label).hexdigest(),
          num_calls,
          inter_op,
          element_size,
          batch_size,
      ))

    def benchmark(label, series):

      print("%s:" % label)
      for num_calls, inter_op, element_size, batch_size in series:

        num_iters = 1024 // (
            (element_size * batch_size) // min(num_calls, inter_op))
        k = 1024 * 1024
        dataset = dataset_ops.Dataset.from_tensors((np.random.rand(
            element_size, 4 * k), np.random.rand(4 * k, 1))).repeat()

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

        fused_dataset = dataset = dataset.apply(
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

      print("")

    np.random.seed(_NUMPY_RANDOM_SEED)
    benchmark("Sequential element size evaluation", seq_elem_size_series)
    benchmark("Sequential batch size evaluation", seq_batch_size_series)
    benchmark("Parallel element size evaluation", par_elem_size_series)
    benchmark("Parallel batch size evaluation", par_batch_size_series)
    benchmark("Transformation parallelism evaluation", par_num_calls_series)
    benchmark("Threadpool size evaluation", par_inter_op_series)

  # This benchmark compares the performance of pipeline with multiple chained
  # maps with and without map fusion.
  def benchmarkChainOfMaps(self):
    chain_lengths = [0, 1, 2, 5, 10, 20, 50]
    for chain_length in chain_lengths:
      self._benchmarkChainOfMaps(chain_length, False)
      self._benchmarkChainOfMaps(chain_length, True)

  def _benchmarkChainOfMaps(self, chain_length, optimize_dataset):
    with ops.Graph().as_default():
      dataset = dataset_ops.Dataset.from_tensors(0).repeat(None)
      for _ in range(chain_length):
        dataset = dataset.map(lambda x: x)
      if optimize_dataset:
        dataset = dataset.apply(optimization.optimize(["map_fusion"]))

      iterator = dataset.make_one_shot_iterator()
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
        opt_mark = "opt" if optimize_dataset else "no-opt"
        print("Map dataset {} chain length: {} Median wall time: {}".format(
            opt_mark, chain_length, median_wall_time))
        self.report_benchmark(
            iters=1000,
            wall_time=median_wall_time,
            name="benchmark_map_dataset_chain_latency_{}_{}".format(
                opt_mark, chain_length))


if __name__ == "__main__":
  test.main()
