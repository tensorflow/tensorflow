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

import itertools
import os
import time

import numpy as np

from tensorflow.contrib.data.python.ops import batching
from tensorflow.contrib.data.python.ops import error_ops
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import io_ops
from tensorflow.python.platform import test
from tensorflow.python.util import compat


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
    filenames = [os.path.join(self.get_temp_dir(), "file_%d.txt" % i)
                 for i in range(5)]
    for filename in filenames:
      write_string_to_file(filename, filename)

    dataset = (
        dataset_ops.Dataset.from_tensor_slices(filenames).map(
            io_ops.read_file, num_parallel_calls=2).prefetch(2).apply(
                error_ops.ignore_errors()))
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

  def benchmarkMapAndBatch(self):
    small = itertools.product([1, 4], [1, 4], [1, 4], [16, 64], [100])
    large = itertools.product([16, 64], [16, 64], [16, 64], [256, 1024], [10])

    num_iters = 100

    def benchmark(series):

      for num_calls, inter_op, element_size, batch_size, num_steps in series:
        dataset = dataset_ops.Dataset.from_tensors(
            np.random.randint(100, size=element_size)).repeat().map(
                lambda x: x,
                num_parallel_calls=num_calls).batch(batch_size=batch_size)
        iterator = dataset.make_one_shot_iterator()
        get_next = iterator.get_next()

        fused_dataset = dataset_ops.Dataset.from_tensors(
            np.random.randint(100, size=element_size)).repeat(None).apply(
                batching.map_and_batch(
                    lambda x: x,
                    num_parallel_calls=num_calls,
                    batch_size=batch_size))
        fused_iterator = fused_dataset.make_one_shot_iterator()
        fused_get_next = fused_iterator.get_next()

        fused_deltas = []
        with session.Session(
            config=config_pb2.ConfigProto(
                inter_op_parallelism_threads=inter_op)) as sess:

          for _ in range(5):
            sess.run(fused_get_next)
          for _ in range(num_iters):
            start = time.time()
            for _ in range(num_steps):
              sess.run(fused_get_next)
            end = time.time()
            fused_deltas.append(end - start)

        chained_deltas = []
        with session.Session(
            config=config_pb2.ConfigProto(
                inter_op_parallelism_threads=inter_op)) as sess:
          for _ in range(5):
            sess.run(get_next)
          for _ in range(num_iters):
            start = time.time()
            for _ in range(num_steps):
              sess.run(get_next)
            end = time.time()
            chained_deltas.append(end - start)

        chained_wall_time = np.median(chained_deltas) / num_iters
        fused_wall_time = np.median(fused_deltas) / num_iters
        print(
            "batch size: %d, num parallel calls: %d, inter-op parallelism: %d, "
            "element size: %d, chained wall time: %f, fused wall time: %f" %
            (batch_size, num_calls, inter_op, element_size, chained_wall_time,
             fused_wall_time))

        self.report_benchmark(
            iters=num_iters,
            wall_time=chained_wall_time,
            name="chained_batch_size_%d_num_calls_%d_inter_op_%d_elem_size_%d"
            % (batch_size, num_calls, inter_op, element_size))

        self.report_benchmark(
            iters=num_iters,
            wall_time=fused_wall_time,
            name="fused_batch_size_%d_num_calls_%d_inter_op_%d_elem_size_%d"
            % (batch_size, num_calls, inter_op, element_size))

    benchmark(small)
    benchmark(large)

if __name__ == "__main__":
  test.main()
