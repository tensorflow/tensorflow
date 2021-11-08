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

import itertools
import sys

import numpy as np

from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.kernel_tests.nn_ops import xent_op_test_base
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.platform import test


class XentOpTest(xent_op_test_base.XentOpTestBase):

  @test_util.run_deprecated_v1
  def testRankTooLarge(self):
    for dtype in np.float16, np.float32:
      np_features = np.array([[[1., 1., 1., 1.]], [[1., 2., 3.,
                                                    4.]]]).astype(dtype)
      np_labels = np.array([[[0., 0., 0., 1.]], [[0., .5, .5,
                                                  0.]]]).astype(dtype)
      self.assertRaisesRegex(ValueError, "rank 2, but is rank 3",
                             gen_nn_ops.softmax_cross_entropy_with_logits,
                             np_features, np_labels)

  def testFeaturesBroadcast(self):
    np_f = np.array([[1., 2., 3., 4.],
                     [1., 2., 3., 4.]]).astype(np.float32)
    np_l = np.array([[0., 0., 0., 1.],
                     [0., .5, .5, 0.]]).astype(np.float32)
    np_loss, np_gradient = self._npXent(labels=np_l, logits=np_f)
    tf_f = constant_op.constant(
        np.array([[1., 2., 3., 4.]]).astype(np.float32))
    tf_l = constant_op.constant(
        np.array([[0., 0., 0., 1.], [0., .5, .5, 0.]]).astype(np.float32))
    tf_loss, tf_gradient = gen_nn_ops.softmax_cross_entropy_with_logits(
        tf_f, tf_l)
    self.assertAllCloseAccordingToType(np_loss, tf_loss)
    self.assertAllCloseAccordingToType(np_gradient, tf_gradient)

    tf_f = constant_op.constant(np.array([[1.]]).astype(np.float32))
    tf_l = constant_op.constant(np.array([[1.], [1.]]).astype(np.float32))
    tf_loss, tf_gradient = gen_nn_ops.softmax_cross_entropy_with_logits(
        tf_f, tf_l)
    self.assertAllClose([0, 0], tf_loss)
    self.assertAllCloseAccordingToType([[0], [0]], tf_gradient)

  @test_util.run_deprecated_v1
  def testNotMatrix(self):
    with self.cached_session():
      with self.assertRaises(ValueError):
        gen_nn_ops.softmax_cross_entropy_with_logits([0., 1., 2., 3.],
                                                     [0., 1., 0., 1.])


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
