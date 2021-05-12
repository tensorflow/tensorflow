# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for SoftmaxCrossEntropyWithLogits op."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import sys

from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.kernel_tests import xent_op_test_base
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.platform import test


XentOpTest = xent_op_test_base.XentOpTestBase


class XentBenchmark(test.Benchmark):

  def benchmarkZeroDimension(self):
    for (m, n, p, use_gpu) in itertools.product(
        [128],
        [10, 100, 1000, 10000, 100000],
        [0.001, 0.01, 0.5, 0.99, 1.0],
        [False]):
      k = int(p * n)
      if k == 0:
        continue
      name = "zero_dimension_m_%d_n_%d_k_%g_use_gpu_%s" % (m, n, k, use_gpu)
      device = "/%s:0" % ("gpu" if use_gpu else "cpu")
      with ops.Graph().as_default():
        with ops.device(device):
          labels = array_ops.zeros([0, 2, 4], dtype=dtypes.float32)
          logits = array_ops.zeros([0, 2, 4], dtype=dtypes.float32)
          op = nn_ops.softmax_cross_entropy_with_logits(
              labels=labels, logits=logits)
        with session.Session() as sess:
          r = self.run_op_benchmark(sess, op, min_iters=100, name=name)
          gb_processed_input = m * n / 1.0e9
          throughput = gb_processed_input / r["wall_time"]
          print("Benchmark: %s \t wall_time: %0.03g s \t "
                "Throughput: %0.03g GB/s" % (name, r["wall_time"], throughput))
          sys.stdout.flush()

  def benchmarkSingleClass(self):
    for (m, n, p, use_gpu) in itertools.product(
        [128],
        [10, 100, 1000, 10000, 100000],
        [0.001, 0.01, 0.5, 0.99, 1.0],
        [False]):
      k = int(p * n)
      if k == 0:
        continue
      name = "single_class_m_%d_n_%d_k_%g_use_gpu_%s" % (m, n, k, use_gpu)
      device = "/%s:0" % ("gpu" if use_gpu else "cpu")
      with ops.Graph().as_default():
        with ops.device(device):
          labels = constant_op.constant([[1.], [-1.], [0.]],
                                        dtype=dtypes.float32)
          logits = constant_op.constant([[-1.], [0.], [1.]],
                                        dtype=dtypes.float32)
          op = nn_ops.softmax_cross_entropy_with_logits(
              labels=labels, logits=logits)
        with session.Session() as sess:
          r = self.run_op_benchmark(sess, op, min_iters=100, name=name)
          gb_processed_input = m * n / 1.0e9
          throughput = gb_processed_input / r["wall_time"]
          print("Benchmark: %s \t wall_time: %0.03g s \t "
                "Throughput: %0.03g GB/s" % (name, r["wall_time"], throughput))
          sys.stdout.flush()


if __name__ == "__main__":
  test.main()
