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
"""Tests for the experimental input pipeline ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np

from tensorflow.contrib.data.python.ops import batching
from tensorflow.contrib.data.python.ops import optimization
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class ModelDatasetTest(test_base.DatasetTestBase):

  def testModelMap(self):
    k = 1024 * 1024
    dataset = dataset_ops.Dataset.from_tensors((np.random.rand(1, 4 * k),
                                                np.random.rand(4 * k,
                                                               1))).repeat()
    dataset = dataset.map(math_ops.matmul)
    iterator = dataset.apply(optimization.model()).make_one_shot_iterator()
    get_next = iterator.get_next()

    deltas = []
    with self.cached_session() as sess:
      for _ in range(5):
        sess.run(get_next.op)
      for _ in range(100):
        start = time.time()
        sess.run(get_next.op)
        end = time.time()
        deltas.append(end - start)

    print("%f (median), %f (mean), %f (stddev), %f (min), %f (max)\n" %
          (np.median(deltas), np.mean(deltas), np.std(deltas), np.min(deltas),
           np.max(deltas)))

  def testModelParallelMap(self):
    k = 1024 * 1024
    dataset = dataset_ops.Dataset.from_tensors((np.random.rand(1, 4 * k),
                                                np.random.rand(4 * k,
                                                               1))).repeat()
    dataset = dataset.map(
        math_ops.matmul, num_parallel_calls=optimization.AUTOTUNE)
    iterator = dataset.apply(optimization.model()).make_one_shot_iterator()
    get_next = iterator.get_next()

    deltas = []
    with self.cached_session() as sess:
      for _ in range(5):
        sess.run(get_next.op)
      for _ in range(1000):
        start = time.time()
        sess.run(get_next.op)
        end = time.time()
        deltas.append(end - start)

    print("%f (median), %f (mean), %f (stddev), %f (min), %f (max)\n" %
          (np.median(deltas), np.mean(deltas), np.std(deltas), np.min(deltas),
           np.max(deltas)))

  def testModelMapAndBatch(self):
    batch_size = 16
    k = 1024 * 1024
    dataset = dataset_ops.Dataset.from_tensors((np.random.rand(1, 4 * k),
                                                np.random.rand(4 * k,
                                                               1))).repeat()
    dataset = dataset.apply(
        batching.map_and_batch(
            math_ops.matmul,
            num_parallel_calls=optimization.AUTOTUNE,
            batch_size=batch_size))
    iterator = dataset.apply(optimization.model()).make_one_shot_iterator()
    get_next = iterator.get_next()

    deltas = []
    with self.cached_session() as sess:
      for _ in range(5):
        sess.run(get_next.op)
      for _ in range(10):
        start = time.time()
        sess.run(get_next.op)
        end = time.time()
        deltas.append(end - start)

    print("%f (median), %f (mean), %f (stddev), %f (min), %f (max)\n" %
          (np.median(deltas), np.mean(deltas), np.std(deltas), np.min(deltas),
           np.max(deltas)))

  def testModelParallelInterleave(self):
    k = 1024 * 1024
    dataset = dataset_ops.Dataset.from_tensors((np.random.rand(1, 4 * k),
                                                np.random.rand(4 * k,
                                                               1))).repeat()
    dataset = dataset.map(math_ops.matmul)
    dataset = dataset_ops.Dataset.range(1).repeat().interleave(
        lambda _: dataset,
        cycle_length=10,
        num_parallel_calls=optimization.AUTOTUNE)
    iterator = dataset.apply(optimization.model()).make_one_shot_iterator()
    get_next = iterator.get_next()

    deltas = []
    with self.cached_session() as sess:
      for _ in range(5):
        sess.run(get_next.op)
      for _ in range(1000):
        start = time.time()
        sess.run(get_next.op)
        end = time.time()
        deltas.append(end - start)

    print("%f (median), %f (mean), %f (stddev), %f (min), %f (max)\n" %
          (np.median(deltas), np.mean(deltas), np.std(deltas), np.min(deltas),
           np.max(deltas)))

  def testModelNested(self):
    k = 1024 * 1024
    a = (np.random.rand(1, 8 * k), np.random.rand(8 * k, 1))
    b = (np.random.rand(1, 4 * k), np.random.rand(4 * k, 1))
    c = (np.random.rand(1, 2 * k), np.random.rand(2 * k, 1))
    dataset = dataset_ops.Dataset.from_tensors((a, b, c)).repeat()

    def f1(a, b, c):
      x, y = a
      return math_ops.matmul(x, y), b, c

    def f2(a, b, c):
      x, y = b
      return a, math_ops.matmul(x, y), c

    def f3(a, b, c):
      x, y = c
      return a, b, math_ops.matmul(x, y)

    dataset = dataset.map(f1, num_parallel_calls=optimization.AUTOTUNE)
    dataset = dataset_ops.Dataset.range(1).repeat().interleave(
        lambda _: dataset, cycle_length=2)

    dataset = dataset.map(f2, num_parallel_calls=optimization.AUTOTUNE)
    dataset = dataset_ops.Dataset.range(1).repeat().interleave(
        lambda _: dataset, cycle_length=2)

    dataset = dataset.map(f3, num_parallel_calls=optimization.AUTOTUNE)
    iterator = dataset.apply(optimization.model()).make_one_shot_iterator()
    get_next = iterator.get_next()

    deltas = []
    with self.cached_session() as sess:
      for _ in range(5):
        sess.run(get_next)
      for _ in range(100):
        start = time.time()
        sess.run(get_next)
        end = time.time()
        deltas.append(end - start)

    print("%f (median), %f (mean), %f (stddev), %f (min), %f (max)\n" %
          (np.median(deltas), np.mean(deltas), np.std(deltas), np.min(deltas),
           np.max(deltas)))


if __name__ == "__main__":
  test.main()
