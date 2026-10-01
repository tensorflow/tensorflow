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
from tensorflow.python.eager import backprop
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.kernel_tests.nn_ops import xent_op_test_base
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.platform import test


class XentOpTest(xent_op_test_base.XentOpTestBase):

  @test_util.run_in_graph_and_eager_modes
  def testSmallGradientAcrossDtypes(self):
    for dtype in (np.float32, np.float64):
      tail_probability = np.exp(-dtype(37.42994775023705))
      dominant_gradient = -tail_probability if dtype == np.float64 else 0.0
      rtol = 1e-14 if dtype == np.float64 else 1e-6
      for batch_size in (0, 1, 4096):
        for target_class in (0, 1):
          for broadcast_labels in (False, True):
            with self.subTest(dtype=dtype, batch_size=batch_size,
                              target_class=target_class,
                              broadcast_labels=broadcast_labels):
              logits = np.zeros((batch_size, 2), dtype=dtype)
              logits[:, target_class] = 37.42994775023705
              labels = np.zeros(
                  (1 if broadcast_labels else batch_size, 2), dtype=dtype)
              labels[:, target_class] = 1.0
              expected = np.full_like(logits, tail_probability)
              expected[:, target_class] = dominant_gradient

              _, gradient = gen_nn_ops.softmax_cross_entropy_with_logits(
                  features=logits, labels=labels)
              gradient = self.evaluate(gradient)

              self.assertAllClose(expected, gradient, rtol=rtol, atol=0.0)
              self.assertTrue(np.all(gradient[:, 1 - target_class] > 0.0))
              if dtype == np.float64:
                self.assertTrue(np.all(gradient[:, target_class] < 0.0))
                self.assertAllClose(gradient[:, 0], -gradient[:, 1], rtol=1e-14,
                                    atol=1e-15)
              else:
                self.assertAllEqual(gradient[:, target_class],
                                    np.zeros(batch_size, dtype=dtype))

  @test_util.run_in_graph_and_eager_modes
  def testSmallGradientThroughPublicApi(self):
    with ops.device("/CPU:0"):
      for dtype in (dtypes.float32, dtypes.float64):
        with self.subTest(dtype=dtype):
          logits = constant_op.constant([[37.42994775023705, 0.0]], dtype)
          labels = constant_op.constant([[1.0, 0.0]], dtype)
          with backprop.GradientTape() as tape:
            tape.watch(logits)
            loss = nn_ops.softmax_cross_entropy_with_logits_v2(
                labels=labels, logits=logits)
          gradient = self.evaluate(tape.gradient(loss, logits))
          tail = np.exp(-dtype.as_numpy_dtype(37.42994775023705))
          dominant = -tail if dtype == dtypes.float64 else 0.0
          self.assertAllClose(
              [[dominant, tail]], gradient,
              rtol=1e-14 if dtype == dtypes.float64 else 1e-6, atol=0.0)

  @test_util.run_in_graph_and_eager_modes
  def testRejectsZeroClasses(self):
    for batch_size in (0, 1):
      for dtype in (dtypes.float32, dtypes.float64):
        with self.subTest(batch_size=batch_size, dtype=dtype):
          empty = constant_op.constant([], shape=[batch_size, 0], dtype=dtype)
          with self.assertRaisesRegex(
              (ValueError, errors.InvalidArgumentError),
              "Must have at least one class, but got 0 classes"):
            result = gen_nn_ops.softmax_cross_entropy_with_logits(
                features=empty, labels=empty)
            self.evaluate(result)

  @test_util.run_deprecated_v1
  def testRejectsDynamicallyZeroClasses(self):
    with self.cached_session() as sess:
      for dtype in (dtypes.float32, dtypes.float64):
        features = array_ops.placeholder(dtype, shape=[None, None])
        result = gen_nn_ops.softmax_cross_entropy_with_logits(
            features=features, labels=features)
        for batch_size in (0, 1):
          with self.subTest(batch_size=batch_size, dtype=dtype):
            with self.assertRaisesRegex(
                errors.InvalidArgumentError,
                "Must have at least one class, but got 0 classes"):
              sess.run(result, feed_dict={
                  features: np.zeros((batch_size, 0),
                                     dtype=dtype.as_numpy_dtype)
              })

  @test_util.run_in_graph_and_eager_modes
  def testDoublePreservesMultiClassTailGradient(self):
    batch_size = 3
    logits = np.zeros((batch_size, 10), dtype=np.float64)
    logits[0, 0] = 40.0
    logits[1, 9] = 40.0
    logits[2, 4] = 3.0  # Confident, but its denominator does not round to one.
    labels = np.zeros_like(logits)
    labels[np.arange(batch_size), [0, 9, 4]] = 1.0

    _, gradient = gen_nn_ops.softmax_cross_entropy_with_logits(
        features=logits, labels=labels)
    gradient = self.evaluate(gradient)
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= np.sum(probabilities, axis=1, keepdims=True)
    expected = probabilities - labels
    tail = np.exp(-40.0)
    expected[0, 0] = -9.0 * tail
    expected[1, 9] = -9.0 * tail

    self.assertAllClose(expected, gradient, rtol=1e-13, atol=1e-15)
    self.assertLess(gradient[0, 0], 0.0)
    self.assertLess(gradient[1, 9], 0.0)
    self.assertAllClose(np.sum(gradient, axis=-1), np.zeros(batch_size),
                        atol=1e-15)

  @test_util.run_in_graph_and_eager_modes
  def testDoublePreservesSoftLabelsAndPositiveZero(self):
    tail_logit = 37.42994775023705
    logits = np.array([[tail_logit, 0.0]], dtype=np.float64)
    labels = np.array([[0.5, 0.5]], dtype=np.float64)
    _, gradient = gen_nn_ops.softmax_cross_entropy_with_logits(
        features=logits, labels=labels)
    gradient = self.evaluate(gradient)
    self.assertLess(gradient[0, 0], 0.5)
    self.assertAllClose(np.sum(gradient, axis=-1), [0.0], atol=1e-15)

    _, single_gradient = gen_nn_ops.softmax_cross_entropy_with_logits(
        labels=np.array([[1.0]], dtype=np.float64),
        features=np.array([[0.0]], dtype=np.float64))
    single_gradient = self.evaluate(single_gradient)
    self.assertEqual(single_gradient[0, 0], 0.0)
    self.assertFalse(np.signbit(single_gradient[0, 0]))

  @test_util.run_in_graph_and_eager_modes
  def testDoublePreservesGeneralLabelGradients(self):
    logits = np.array([[0., 0.], [0., 0.], [0., 0.], [np.inf, 0.],
                       [np.nan, 0.], [-np.inf, -np.inf]],
                      dtype=np.float64)
    labels = np.array([[0., 0.], [2., 0.], [np.nan, 0.], [1., 0.],
                       [1., 0.], [1., 0.]],
                      dtype=np.float64)
    _, gradient = gen_nn_ops.softmax_cross_entropy_with_logits(
        features=logits, labels=labels)
    gradient = self.evaluate(gradient)

    self.assertAllClose([[0.5, 0.5], [-1.5, 0.5]], gradient[:2])
    self.assertTrue(np.isnan(gradient[2, 0]))
    self.assertAllClose(0.5, gradient[2, 1])
    self.assertTrue(np.all(np.isnan(gradient[3:])))

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
