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

import numpy as np

from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
import tensorflow.python.ops.nn_grad  # pylint: disable=unused-import
from tensorflow.python.platform import test


class XentTest(test.TestCase):

  def _npXent(self, features, labels, dim=-1):
    if dim == -1:
      dim = len(features.shape) - 1
    one_only_on_dim = list(features.shape)
    one_only_on_dim[dim] = 1
    e = np.exp(
        features - np.reshape(np.amax(features, axis=dim), one_only_on_dim))
    probs = e / np.reshape(np.sum(e, axis=dim), one_only_on_dim)
    bp = (probs - labels)
    l = -np.sum(labels * np.log(probs + 1.0e-20), axis=dim)
    return l, bp

  # TODO(b/123860949): The values are constant folded for XLA, so placeholders
  # are needed.
  def _testXent(self,
                np_features,
                np_labels,
                use_gpu=False,
                with_placeholders=False):
    np_loss, np_backprop = self._npXent(np_features, np_labels)
    with self.cached_session(use_gpu=use_gpu) as sess:
      if with_placeholders:
        features_placeholder = array_ops.placeholder(np_features.dtype)
        labels_placeholder = array_ops.placeholder(np_labels.dtype)
        loss, backprop = gen_nn_ops.softmax_cross_entropy_with_logits(
            labels=labels_placeholder, features=features_placeholder)
        tf_loss, tf_backprop = sess.run([loss, backprop],
                                        feed_dict={
                                            labels_placeholder: np_labels,
                                            features_placeholder: np_features
                                        })
      else:
        loss, backprop = gen_nn_ops.softmax_cross_entropy_with_logits(
            np_features, np_labels)
        tf_loss, tf_backprop = self.evaluate([loss, backprop])
    self.assertAllCloseAccordingToType(np_loss, tf_loss, half_rtol=1e-2)
    self.assertAllCloseAccordingToType(np_backprop, tf_backprop)

  def _testXentWrapper(self, np_features, np_labels, dim=-1, use_gpu=False):
    np_loss, _ = self._npXent(np_features, np_labels, dim=dim)
    with self.cached_session(use_gpu=use_gpu) as sess:
      loss = nn_ops.softmax_cross_entropy_with_logits(
          labels=np_labels, logits=np_features, dim=dim)
      tf_loss = self.evaluate(loss)
    print("np_loss:", np_loss)
    print("tf_loss:", tf_loss)
    self.assertAllCloseAccordingToType(np_loss, tf_loss)

  # TODO(b/123860949): The values are constant folded for XLA, so placeholders
  # are needed.
  def _testAll(self, features, labels, with_placeholders=False):
    self._testXent(
        features, labels, use_gpu=False, with_placeholders=with_placeholders)
    self._testXent(
        features, labels, use_gpu=True, with_placeholders=with_placeholders)

  def _testSingleClass(self, use_gpu=False):
    for dtype in np.float16, np.float32:
      with self.cached_session(use_gpu=use_gpu) as sess:
        loss, backprop = gen_nn_ops.softmax_cross_entropy_with_logits(
            np.array([[1.], [-1.], [0.]]).astype(dtype),
            np.array([[-1.], [0.], [1.]]).astype(dtype))
        tf_loss, tf_backprop = self.evaluate([loss, backprop])
      self.assertAllClose([0.0, 0.0, 0.0], tf_loss)
      self.assertAllClose([[2.0], [1.0], [0.0]], tf_backprop)

  def testSingleClass(self):
    self._testSingleClass(True)
    self._testSingleClass(False)

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

  def testNpXent(self):
    # We create 2 batches of logits for testing.
    # batch 0 is the boring uniform distribution: 1, 1, 1, 1, with target 3.
    # batch 1 has a bit of difference: 1, 2, 3, 4, with soft targets (1, 2).
    features = [[1., 1., 1., 1.], [1., 2., 3., 4.]]
    labels = [[0., 0., 0., 1.], [0., .5, .5, 0.]]

    # For batch 0, we expect the uniform distribution: 0.25, 0.25, 0.25, 0.25
    # With a hard target 3, the backprop is [0.25, 0.25, 0.25, -0.75]
    # The loss for this batch is -log(0.25) = 1.386
    #
    # For batch 1, we have:
    # exp(0) = 1
    # exp(1) = 2.718
    # exp(2) = 7.389
    # exp(3) = 20.085
    # SUM = 31.192
    # So we have as probabilities:
    # exp(0) / SUM = 0.032
    # exp(1) / SUM = 0.087
    # exp(2) / SUM = 0.237
    # exp(3) / SUM = 0.644
    # With a soft target (1, 2), the backprop is
    # [0.032, 0.087 - 0.5 = -0.413, 0.237 - 0.5 = -0.263, 0.644]
    # The loss for this batch is [0.5 * -log(0.087), 0.5 * -log(0.237)]
    # = [1.3862, 1.9401]
    np_loss, np_backprop = self._npXent(np.array(features), np.array(labels))
    self.assertAllClose(
        np.array([[0.25, 0.25, 0.25, -0.75], [0.0321, -0.4129, -0.2632,
                                              0.6439]]),
        np_backprop,
        rtol=1.e-3,
        atol=1.e-3)
    self.assertAllClose(
        np.array([1.3862, 1.9401]), np_loss, rtol=1.e-3, atol=1.e-3)

  def testShapeBroadcast(self):
    np_f = np.array([[1., 2., 3., 4.],
                     [1., 2., 3., 4.]]).astype(np.float32)
    np_l = np.array([[0., 0., 0., 1.],
                     [0., .5, .5, 0.]]).astype(np.float32)
    np_loss, np_backprop = self._npXent(np_f, np_l)
    tf_f = constant_op.constant(
        np.array([[1., 2., 3., 4.]]).astype(np.float32))
    tf_l = constant_op.constant(
        np.array([[0., 0., 0., 1.], [0., .5, .5, 0.]]).astype(np.float32))
    for use_gpu in [False, True]:
      with self.cached_session(use_gpu=use_gpu) as sess:
        loss, backprop = gen_nn_ops.softmax_cross_entropy_with_logits(
            tf_f, tf_l)
        tf_loss, tf_backprop = self.evaluate([loss, backprop])
      self.assertAllCloseAccordingToType(np_loss, tf_loss)
      self.assertAllCloseAccordingToType(np_backprop, tf_backprop)

  # TODO(b/123860949): The values are constant folded for XLA, so placeholders
  # are needed.
  @test_util.run_deprecated_v1
  def testFeatureBroadcast(self):
    self._testAll(
        np.array([[1., 1., 1., 1.], [1., 2., 3., 4.]]).astype(np.float16),
        np.array([[0., 0., 0., 1.]]).astype(np.float16),
        with_placeholders=True)
    self._testAll(
        np.array([[1., 1., 1., 1.], [1., 2., 3., 4.]]).astype(np.float16),
        np.array([[0.], [2.]]).astype(np.float16),
        with_placeholders=True)

  @test_util.run_deprecated_v1
  def testShapeMismatch(self):
    with self.cached_session():
      with self.assertRaises(ValueError):
        gen_nn_ops.softmax_cross_entropy_with_logits(
            [[0., 1.], [2., 3.]], [[0., 1., 0.], [1., 0., 0.]])

  @test_util.run_deprecated_v1
  def testNotMatrix(self):
    with self.cached_session():
      with self.assertRaises(ValueError):
        gen_nn_ops.softmax_cross_entropy_with_logits([0., 1., 2., 3.],
                                                     [0., 1., 0., 1.])

  def testHalf(self):
    self._testAll(
        np.array([[1., 1., 1., 1.], [1., 2., 3., 4.]]).astype(np.float16),
        np.array([[0., 0., 0., 1.], [0., .5, .5, 0.]]).astype(np.float16))

  def testFloat(self):
    self._testAll(
        np.array([[1., 1., 1., 1.], [1., 2., 3., 4.]]).astype(np.float32),
        np.array([[0., 0., 0., 1.], [0., .5, .5, 0.]]).astype(np.float32))

  def testDouble(self):
    self._testAll(
        np.array([[1., 1., 1., 1.], [1., 2., 3., 4.]]).astype(np.float64),
        np.array([[0., 0., 0., 1.], [0., .5, .5, 0.]]).astype(np.float64))

  @test_util.run_deprecated_v1
  def testGradient(self):
    with self.cached_session() as sess:
      l = constant_op.constant(
          [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.5],
          shape=[3, 4],
          dtype=dtypes.float64,
          name="l")
      f = constant_op.constant(
          [0.1, 0.2, 0.3, 0.4, 0.1, 0.4, 0.9, 1.6, 0.1, 0.8, 2.7, 6.4],
          shape=[3, 4],
          dtype=dtypes.float64,
          name="f")
      x = nn_ops.softmax_cross_entropy_with_logits(
          labels=l, logits=f, name="xent")
      err = gradient_checker.compute_gradient_error(f, [3, 4], x, [3])

      # Check that no extra computation performed. When only first derivative is requested,
      # second derivative must not be computed. So when there is no second derivative,
      # there is no `BatchMatMul` op in the graph.
      op_names = [
          op.op_def.name for op in sess.graph.get_operations() if op.op_def
      ]
      self.assertNotIn("BatchMatMul", op_names)
      self.assertNotIn("BatchMatMulV2", op_names)

    print("cross entropy gradient err = ", err)
    self.assertLess(err, 5e-8)

  @test_util.run_deprecated_v1
  def testGradientLabelWithV2(self):
    with self.cached_session():
      l = constant_op.constant(
          [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.5],
          shape=[3, 4],
          dtype=dtypes.float64,
          name="l")
      f = constant_op.constant(
          [0.1, 0.2, 0.3, 0.4, 0.1, 0.4, 0.9, 1.6, 0.1, 0.8, 2.7, 6.4],
          shape=[3, 4],
          dtype=dtypes.float64,
          name="f")
      x = nn_ops.softmax_cross_entropy_with_logits_v2(
          labels=l, logits=f, name="xent")
      err = gradient_checker.compute_gradient_error(l, [3, 4], x, [3])

    self.assertLess(err, 5e-8)

  @test_util.run_deprecated_v1
  def testSecondGradient(self):
    with self.cached_session() as sess:
      l = constant_op.constant(
          [
              0.0, 0.0, 1.0 / 3, 0.0, 1.0 / 3, 0.0, 0.0, 0.0, 0.0, 0.5 / 3, 0.0,
              0.5 / 3
          ],
          shape=[12],
          dtype=dtypes.float64,
          name="l")
      f = constant_op.constant(
          [0.1, 0.2, 0.3, 0.4, 0.1, 0.4, 0.9, 1.6, 0.1, 0.8, 2.7, 6.4],
          shape=[12],
          dtype=dtypes.float64,
          name="f")
      x = nn_ops.softmax_cross_entropy_with_logits(
          labels=l, logits=f, name="xent")
      loss = math_ops.reduce_sum(x)

      gradients = gradients_impl.gradients(loss, [f])[0]

      err = gradient_checker.compute_gradient_error(f, [12], gradients, [12])

      # Check that second derivative is calculated.
      # (it is equivalent to being `BatchMatMul` op in the graph because of implementation of xentropy grad)
      op_names = [
          op.op_def.name for op in sess.graph.get_operations() if op.op_def
      ]
      self.assertIn("BatchMatMulV2", op_names)

    print("cross entropy hessian err = ", err)
    self.assertLess(err, 5e-8)

  def testWrapper(self):
    features = np.array([[[1., 1., 1., 1.], [1., 2., 3., 4.]],
                         [[2., 3., 4., 5.], [6., 7., 8., 9.]],
                         [[5., 4., 3., 2.], [1., 2., 3., 4.]]]).astype(
                             np.float32)
    labels = np.array([[[0., 0., 0., 1.], [0., 1., 0., 0.]],
                       [[0., 0.5, 0.5, 0.], [0.5, 0.5, 0., 0.]],
                       [[0., 1., 0., 0.], [0., 0., 1., 0.]]]).astype(
                           np.float32)
    self._testXentWrapper(features, labels, dim=0, use_gpu=False)
    self._testXentWrapper(features, labels, dim=0, use_gpu=True)
    self._testXentWrapper(features, labels, dim=1, use_gpu=False)
    self._testXentWrapper(features, labels, dim=1, use_gpu=True)
    self._testXentWrapper(features, labels, dim=-1, use_gpu=False)
    self._testXentWrapper(features, labels, dim=-1, use_gpu=True)

  def testZeroDimension(self):
    features = np.zeros([0, 2, 4]).astype(np.float32)
    labels = np.zeros([0, 2, 4]).astype(np.float32)
    np_loss, _ = self._npXent(features, labels)
    with self.session(use_gpu=True) as sess:
      loss = nn_ops.softmax_cross_entropy_with_logits(
          labels=labels, logits=features)
      tf_loss = self.evaluate(loss)
    self.assertAllEqual(np_loss, tf_loss)


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
