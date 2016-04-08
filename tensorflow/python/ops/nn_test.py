# Copyright 2015 Google Inc. All Rights Reserved.
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

"""Tests for tensorflow.ops.nn."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.python.ops import gen_nn_ops

exp = math.exp
log = math.log


class SigmoidCrossEntropyWithLogitsTest(tf.test.TestCase):

  def _SigmoidCrossEntropyWithLogits(self, logits, targets):
    assert len(logits) == len(targets)
    pred = [1 / (1 + exp(-x)) for x in logits]
    eps = 0.0001
    pred = [min(max(p, eps), 1 - eps) for p in pred]
    return [-z * log(y) - (1 - z) * log(1 - y) for y, z in zip(pred, targets)]

  def _Inputs(self, x=None, y=None, dtype=tf.float64, sizes=None):
    x = [-100, -2, -2, 0, 2, 2, 2, 100] if x is None else x
    y = [0, 0, 1, 0, 0, 1, 0.5, 1] if y is None else y
    assert len(x) == len(y)
    sizes = sizes if sizes else [len(x)]
    logits = tf.constant(x, shape=sizes, dtype=dtype, name="logits")
    targets = tf.constant(y, shape=sizes, dtype=dtype, name="targets")
    losses = np.array(self._SigmoidCrossEntropyWithLogits(x, y)).reshape(*sizes)
    return logits, targets, losses

  def testConstructionNamed(self):
    with self.test_session():
      logits, targets, _ = self._Inputs()
      loss = tf.nn.sigmoid_cross_entropy_with_logits(logits,
                                                     targets,
                                                     name="mylogistic")
    self.assertEqual("mylogistic", loss.op.name)

  def testLogisticOutput(self):
    for use_gpu in [True, False]:
      with self.test_session(use_gpu=use_gpu):
        logits, targets, losses = self._Inputs(dtype=tf.float32)
        loss = tf.nn.sigmoid_cross_entropy_with_logits(logits, targets)
        np_loss = np.array(losses).astype(np.float32)
        tf_loss = loss.eval()
      self.assertAllClose(np_loss, tf_loss, atol=0.001)

  def testLogisticOutputMultiDim(self):
    for use_gpu in [True, False]:
      with self.test_session(use_gpu=use_gpu):
        logits, targets, losses = self._Inputs(dtype=tf.float32,
                                               sizes=[2, 2, 2])
        loss = tf.nn.sigmoid_cross_entropy_with_logits(logits, targets)
        np_loss = np.array(losses).astype(np.float32)
        tf_loss = loss.eval()
      self.assertAllClose(np_loss, tf_loss, atol=0.001)

  def testGradient(self):
    sizes = [4, 2]
    with self.test_session():
      logits, targets, _ = self._Inputs(sizes=sizes)
      loss = tf.nn.sigmoid_cross_entropy_with_logits(logits, targets)
      err = tf.test.compute_gradient_error(logits, sizes, loss, sizes)
    print("logistic loss gradient err = ", err)
    self.assertLess(err, 1e-7)

  def testShapeError(self):
    with self.assertRaisesRegexp(ValueError, "must have the same shape"):
      tf.nn.sigmoid_cross_entropy_with_logits([[2, 1]], [1, 2, 3])


class WeightedCrossEntropyTest(tf.test.TestCase):

  def _WeightedCrossEntropy(self, logits, targets, pos_coeff):
    assert len(logits) == len(targets)
    pred = [1 / (1 + exp(-x)) for x in logits]
    eps = 0.0001
    pred = [min(max(p, eps), 1 - eps) for p in pred]
    return [-z * pos_coeff * log(y) - (1 - z) * log(1 - y)
            for y, z in zip(pred, targets)]

  def _Inputs(self, x=None, y=None, q=3.0, dtype=tf.float64, sizes=None):
    x = [-100, -2, -2, 0, 2, 2, 2, 100] if x is None else x
    y = [0, 0, 1, 0, 0, 1, 0.5, 1] if y is None else y
    assert len(x) == len(y)
    sizes = sizes if sizes else [len(x)]
    logits = tf.constant(x, shape=sizes, dtype=dtype, name="logits")
    targets = tf.constant(y, shape=sizes, dtype=dtype, name="targets")
    losses = np.array(self._WeightedCrossEntropy(x, y, q)).reshape(*sizes)
    return logits, targets, q, losses

  def testConstructionNamed(self):
    with self.test_session():
      logits, targets, pos_weight, _ = self._Inputs()
      loss = tf.nn.weighted_cross_entropy_with_logits(logits, targets,
                                                      pos_weight, name="mybce")
    self.assertEqual("mybce", loss.op.name)

  def testOutput(self):
    for use_gpu in [True, False]:
      with self.test_session(use_gpu=use_gpu):
        logits, targets, pos_weight, losses = self._Inputs(dtype=tf.float32)
        loss = tf.nn.weighted_cross_entropy_with_logits(logits, targets,
                                                        pos_weight)
        np_loss = np.array(losses).astype(np.float32)
        tf_loss = loss.eval()
      self.assertAllClose(np_loss, tf_loss, atol=0.001)

  def testOutputMultiDim(self):
    for use_gpu in [True, False]:
      with self.test_session(use_gpu=use_gpu):
        logits, targets, pos_weight, losses = self._Inputs(dtype=tf.float32,
                                                           sizes=[2, 2, 2])
        loss = tf.nn.weighted_cross_entropy_with_logits(logits, targets,
                                                        pos_weight)
        np_loss = np.array(losses).astype(np.float32)
        tf_loss = loss.eval()
      self.assertAllClose(np_loss, tf_loss, atol=0.001)

  def testGradient(self):
    sizes = [4, 2]
    with self.test_session():
      logits, targets, pos_weight, _ = self._Inputs(sizes=sizes)
      loss = tf.nn.weighted_cross_entropy_with_logits(logits, targets,
                                                      pos_weight)
      err = tf.test.compute_gradient_error(logits, sizes, loss, sizes)
    print("logistic loss gradient err = ", err)
    self.assertLess(err, 1e-7)

  def testShapeError(self):
    with self.assertRaisesRegexp(ValueError, "must have the same shape"):
      tf.nn.weighted_cross_entropy_with_logits([[2, 1]], [1, 2, 3], 2.0)


class ZeroFractionTest(tf.test.TestCase):

  def _ZeroFraction(self, x):
    assert x.shape
    total_elements = np.prod(x.shape)
    nonzeros = np.count_nonzero(x.flatten())
    return 1.0 - nonzeros / total_elements

  def testZeroFraction(self):
    x_shape = [5, 17]
    x_np = np.random.randint(0, 2, size=x_shape).astype(np.float32)
    y_np = self._ZeroFraction(x_np)
    with self.test_session():
      x_tf = tf.constant(x_np)
      x_tf.set_shape(x_shape)
      y_tf = tf.nn.zero_fraction(x_tf)
      y_tf_np = y_tf.eval()
    eps = 1e-8
    self.assertAllClose(y_tf_np, y_np, eps)

  def testZeroFractionEmpty(self):
    with self.test_session():
      x = np.zeros(0)
      y = tf.nn.zero_fraction(x).eval()
      self.assertTrue(np.isnan(y))


class SoftmaxTest(tf.test.TestCase):

  def _softmax(self, x):
    assert len(x.shape) == 2
    m = x.max(1)[:, np.newaxis]
    u = np.exp(x - m)
    z = u.sum(1)[:, np.newaxis]
    return u / z

  def testSoftmax(self):
    x_shape = [5, 10]
    x_np = np.random.randn(*x_shape).astype(np.float32)
    y_np = self._softmax(x_np)
    with self.test_session():
      x_tf = tf.constant(x_np)
      y_tf = tf.nn.softmax(x_tf)
      y_tf_np = y_tf.eval()
    eps = 1e-3
    self.assertAllClose(y_tf_np, y_np, eps)

  def testGradient(self):
    x_shape = [5, 10]
    x_np = np.random.randn(*x_shape).astype(np.float64)
    with self.test_session():
      x_tf = tf.constant(x_np)
      y_tf = tf.nn.softmax(x_tf)
      err = tf.test.compute_gradient_error(x_tf, x_shape, y_tf, x_shape)
    eps = 1e-8
    self.assertLess(err, eps)


class Conv2DTransposeTest(tf.test.TestCase):

  def testConv2DTransposeSingleStride(self):
    with self.test_session():
      strides = [1, 1, 1, 1]

      # Input, output: [batch, height, width, depth]
      x_shape = [2, 6, 4, 3]
      y_shape = [2, 6, 4, 2]

      # Filter: [kernel_height, kernel_width, output_depth, input_depth]
      f_shape = [3, 3, 2, 3]

      x = tf.constant(1.0, shape=x_shape, name="x", dtype=tf.float32)
      f = tf.constant(1.0, shape=f_shape, name="filter", dtype=tf.float32)
      output = tf.nn.conv2d_transpose(x, f, y_shape, strides=strides,
                                      padding="SAME")
      value = output.eval()

      # We count the number of cells being added at the locations in the output.
      # At the center, #cells=kernel_height * kernel_width
      # At the corners, #cells=ceil(kernel_height/2) * ceil(kernel_width/2)
      # At the borders, #cells=ceil(kernel_height/2)*kernel_width or
      #                        kernel_height * ceil(kernel_width/2)

      for n in xrange(x_shape[0]):
        for k in xrange(f_shape[2]):
          for w in xrange(y_shape[2]):
            for h in xrange(y_shape[1]):
              target = 4 * 3.0
              h_in = h > 0 and h < y_shape[1] - 1
              w_in = w > 0 and w < y_shape[2] - 1
              if h_in and w_in:
                target += 5 * 3.0
              elif h_in or w_in:
                target += 2 * 3.0
              self.assertAllClose(target, value[n, h, w, k])

  def testConv2DTransposeSame(self):
    with self.test_session():
      strides = [1, 2, 2, 1]

      # Input, output: [batch, height, width, depth]
      x_shape = [2, 6, 4, 3]
      y_shape = [2, 12, 8, 2]

      # Filter: [kernel_height, kernel_width, output_depth, input_depth]
      f_shape = [3, 3, 2, 3]

      x = tf.constant(1.0, shape=x_shape, name="x", dtype=tf.float32)
      f = tf.constant(1.0, shape=f_shape, name="filter", dtype=tf.float32)
      output = tf.nn.conv2d_transpose(x, f, y_shape, strides=strides,
                                      padding="SAME")
      value = output.eval()

      for n in xrange(x_shape[0]):
        for k in xrange(f_shape[2]):
          for w in xrange(y_shape[2]):
            for h in xrange(y_shape[1]):
              target = 3.0
              # We add a case for locations divisible by the stride.
              h_in = h % strides[1] == 0 and h > 0 and h < y_shape[1] - 1
              w_in = w % strides[2] == 0 and w > 0 and w < y_shape[2] - 1
              if h_in and w_in:
                target += 9.0
              elif h_in or w_in:
                target += 3.0
              self.assertAllClose(target, value[n, h, w, k])

  def testConv2DTransposeValid(self):
    with self.test_session():
      strides = [1, 2, 2, 1]

      # Input, output: [batch, height, width, depth]
      x_shape = [2, 6, 4, 3]
      y_shape = [2, 13, 9, 2]

      # Filter: [kernel_height, kernel_width, output_depth, input_depth]
      f_shape = [3, 3, 2, 3]

      x = tf.constant(1.0, shape=x_shape, name="x", dtype=tf.float32)
      f = tf.constant(1.0, shape=f_shape, name="filter", dtype=tf.float32)
      output = tf.nn.conv2d_transpose(x, f, y_shape, strides=strides,
                                      padding="VALID")
      value = output.eval()

      cache_values = np.zeros(y_shape, dtype=np.float32)

      # The amount of padding added
      pad = 1

      for n in xrange(x_shape[0]):
        for k in xrange(f_shape[2]):
          for w in xrange(pad, y_shape[2] - pad):
            for h in xrange(pad, y_shape[1] - pad):
              target = 3.0
              # We add a case for locations divisible by the stride.
              h_in = h % strides[
                  1] == 0 and h > pad and h < y_shape[1] - 1 - pad
              w_in = w % strides[
                  2] == 0 and w > pad and w < y_shape[2] - 1 - pad
              if h_in and w_in:
                target += 9.0
              elif h_in or w_in:
                target += 3.0
              cache_values[n, h, w, k] = target

          # copy values in the border
          cache_values[n, :, 0, k] = cache_values[n, :, 1, k]
          cache_values[n, :, -1, k] = cache_values[n, :, -2, k]
          cache_values[n, 0, :, k] = cache_values[n, 1, :, k]
          cache_values[n, -1, :, k] = cache_values[n, -2, :, k]

    self.assertAllClose(cache_values, value)

  def testGradient(self):
    x_shape = [2, 6, 4, 3]
    f_shape = [3, 3, 2, 3]
    y_shape = [2, 12, 8, 2]
    strides = [1, 2, 2, 1]
    np.random.seed(1)  # Make it reproducible.
    x_val = np.random.random_sample(x_shape).astype(np.float64)
    f_val = np.random.random_sample(f_shape).astype(np.float64)
    with self.test_session():
      x = tf.constant(x_val, name="x", dtype=tf.float32)
      f = tf.constant(f_val, name="f", dtype=tf.float32)
      output = tf.nn.conv2d_transpose(x, f, y_shape, strides=strides,
                                      padding="SAME")
      err = tf.test.compute_gradient_error(
          [x, f], [x_shape, f_shape], output, y_shape)
    print("DeConv gradient err = %g " % err)
    err_tolerance = 0.0005
    self.assertLess(err, err_tolerance)


class L2LossTest(tf.test.TestCase):

  def testL2Loss(self):
    with self.test_session():
      x = tf.constant([1.0, 0.0, 3.0, 2.0], shape=[2, 2], name="x")
      l2loss = tf.nn.l2_loss(x)
      value = l2loss.eval()
    self.assertAllClose(7.0, value)

  def testGradient(self):
    x_shape = [20, 7, 3]
    np.random.seed(1)  # Make it reproducible.
    x_val = np.random.random_sample(x_shape).astype(np.float64)
    with self.test_session():
      x = tf.constant(x_val, name="x")
      output = tf.nn.l2_loss(x)
      err = tf.test.compute_gradient_error(x, x_shape, output, [1])
    print("L2Loss gradient err = %g " % err)
    err_tolerance = 1e-11
    self.assertLess(err, err_tolerance)


class L2NormalizeTest(tf.test.TestCase):

  def _l2Normalize(self, x, dim):
    norm = np.apply_along_axis(np.linalg.norm, dim, x)
    return x / np.expand_dims(norm, dim)

  def testL2Normalize(self):
    x_shape = [20, 7, 3]
    np.random.seed(1)
    x_np = np.random.random_sample(x_shape).astype(np.float32)
    for dim in range(len(x_shape)):
      y_np = self._l2Normalize(x_np, dim)
      with self.test_session():
        x_tf = tf.constant(x_np, name="x")
        y_tf = tf.nn.l2_normalize(x_tf, dim)
        self.assertAllClose(y_np, y_tf.eval())

  def testL2NormalizeGradient(self):
    x_shape = [20, 7, 3]
    np.random.seed(1)
    x_np = np.random.random_sample(x_shape).astype(np.float64)
    for dim in range(len(x_shape)):
      with self.test_session():
        x_tf = tf.constant(x_np, name="x")
        y_tf = tf.nn.l2_normalize(x_tf, dim)
        err = tf.test.compute_gradient_error(x_tf, x_shape, y_tf, x_shape)
      print("L2Normalize gradient err = %g " % err)
      self.assertLess(err, 1e-4)


class DropoutTest(tf.test.TestCase):

  def testDropout(self):
    # Runs dropout with 0-1 tensor 10 times, sum the number of ones and validate
    # that it is producing approximately the right number of ones over a large
    # number of samples, based on the keep probability.
    x_dim = 40
    y_dim = 30
    num_iter = 10
    for keep_prob in [0.1, 0.5, 0.8]:
      with self.test_session():
        t = tf.constant(1.0, shape=[x_dim, y_dim], dtype=tf.float32)
        dropout = tf.nn.dropout(t, keep_prob)
        final_count = 0
        self.assertEqual([x_dim, y_dim], dropout.get_shape())
        for _ in xrange(0, num_iter):
          value = dropout.eval()
          final_count += np.count_nonzero(value)
          # Verifies that there are only two values: 0 and 1/keep_prob.
          sorted_value = np.unique(np.sort(value))
          self.assertEqual(0, sorted_value[0])
          self.assertAllClose(1 / keep_prob, sorted_value[1])
      # Check that we are in the 15% error range
      expected_count = x_dim * y_dim * keep_prob * num_iter
      rel_error = math.fabs(final_count - expected_count) / expected_count
      print(rel_error)
      self.assertTrue(rel_error < 0.15)

  def testShapedDropout(self):
    # Runs dropout with 0-1 tensor 10 times, sum the number of ones and validate
    # that it is producing approximately the right number of ones over a large
    # number of samples, based on the keep probability. This time with shaped
    # noise.
    x_dim = 40 * 30
    y_dim = 3
    num_iter = 10
    for keep_prob in [0.1, 0.5, 0.8]:
      with self.test_session():
        t = tf.constant(1.0, shape=[x_dim, y_dim], dtype=tf.float32)
        dropout = tf.nn.dropout(t, keep_prob, noise_shape=[x_dim, 1])
        self.assertEqual([x_dim, y_dim], dropout.get_shape())
        final_count = 0
        for _ in xrange(0, num_iter):
          value = dropout.eval()
          final_count += np.count_nonzero(value)
          # Verifies that there are only two values: 0 and 1/keep_prob.
          sorted_value = np.unique(np.sort(value))
          self.assertEqual(0, sorted_value[0])
          self.assertAllClose(1 / keep_prob, sorted_value[1])
      # Check that we are in the 15% error range
      expected_count = x_dim * y_dim * keep_prob * num_iter
      rel_error = math.fabs(final_count - expected_count) / expected_count
      print(rel_error)
      self.assertTrue(rel_error < 0.15)

  def testShapedDropoutCorrelation(self):
    # Runs a shaped dropout and tests that the correlations are correct.
    x_dim = 40
    y_dim = 30
    num_iter = 10
    for keep_prob in [0.1, 0.5, 0.8]:
      with self.test_session():
        t = tf.constant(1.0, shape=[x_dim, y_dim], dtype=tf.float32)
        dropout = tf.nn.dropout(t, keep_prob, noise_shape=[x_dim, 1])
        self.assertEqual([x_dim, y_dim], dropout.get_shape())
        for _ in xrange(0, num_iter):
          value = dropout.eval()
          # Verifies that each y column as only one type of activation.
          for i in xrange(x_dim):
            sorted_value = np.unique(np.sort(value[i, :]))
            self.assertEqual(sorted_value.size, 1)

  def testDropoutPlaceholderKeepProb(self):
    # Runs dropout with 0-1 tensor 10 times, sum the number of ones and validate
    # that it is producing approximately the right number of ones over a large
    # number of samples, based on the keep probability.
    x_dim = 40
    y_dim = 30
    num_iter = 10
    for keep_prob in [0.1, 0.5, 0.8]:
      with self.test_session():
        t = tf.constant(1.0, shape=[x_dim, y_dim], dtype=tf.float32)
        keep_prob_placeholder = tf.placeholder(tf.float32)
        dropout = tf.nn.dropout(t, keep_prob_placeholder)
        final_count = 0
        self.assertEqual([x_dim, y_dim], dropout.get_shape())
        for _ in xrange(0, num_iter):
          value = dropout.eval(feed_dict={keep_prob_placeholder: keep_prob})
          final_count += np.count_nonzero(value)
          # Verifies that there are only two values: 0 and 1/keep_prob.
          sorted_value = np.unique(np.sort(value))
          self.assertEqual(0, sorted_value[0])
          self.assertAllClose(1 / keep_prob, sorted_value[1])
      # Check that we are in the 15% error range
      expected_count = x_dim * y_dim * keep_prob * num_iter
      rel_error = math.fabs(final_count - expected_count) / expected_count
      print(rel_error)
      self.assertTrue(rel_error < 0.15)

  def testShapedDropoutUnknownShape(self):
    x_dim = 40
    y_dim = 30
    keep_prob = 0.5
    x = tf.constant(1.0, shape=[x_dim, y_dim], dtype=tf.float32)
    dropout_x = tf.nn.dropout(x,
                              keep_prob,
                              noise_shape=tf.placeholder(tf.int32))
    self.assertEqual(x.get_shape(), dropout_x.get_shape())

  def testInvalidKeepProb(self):
    x_dim = 40
    y_dim = 30
    t = tf.constant(1.0, shape=[x_dim, y_dim], dtype=tf.float32)
    with self.assertRaises(ValueError):
      tf.nn.dropout(t, -1.0)
    with self.assertRaises(ValueError):
      tf.nn.dropout(t, 1.1)
    with self.assertRaises(ValueError):
      tf.nn.dropout(t, [0.0, 1.0])
    with self.assertRaises(ValueError):
      tf.nn.dropout(t, tf.placeholder(tf.float64))
    with self.assertRaises(ValueError):
      tf.nn.dropout(t, tf.placeholder(tf.float32, shape=[2]))

  def testShapedDropoutShapeError(self):
    # Runs shaped dropout and verifies an error is thrown on misshapen noise.
    x_dim = 40
    y_dim = 30
    keep_prob = 0.5
    t = tf.constant(1.0, shape=[x_dim, y_dim], dtype=tf.float32)
    with self.assertRaises(ValueError):
      _ = tf.nn.dropout(t, keep_prob, noise_shape=[x_dim, y_dim + 10])
    with self.assertRaises(ValueError):
      _ = tf.nn.dropout(t, keep_prob, noise_shape=[x_dim, y_dim, 5])
    with self.assertRaises(ValueError):
      _ = tf.nn.dropout(t, keep_prob, noise_shape=[x_dim + 3])
    with self.assertRaises(ValueError):
      _ = tf.nn.dropout(t, keep_prob, noise_shape=[x_dim])
    # test that broadcasting proceeds
    _ = tf.nn.dropout(t, keep_prob, noise_shape=[y_dim])
    _ = tf.nn.dropout(t, keep_prob, noise_shape=[1, y_dim])
    _ = tf.nn.dropout(t, keep_prob, noise_shape=[x_dim, 1])
    _ = tf.nn.dropout(t, keep_prob, noise_shape=[1, 1])


class BatchNormalizationTest(tf.test.TestCase):

  def _npBatchNorm(self, x, m, v, beta, gamma, epsilon,
                   scale_after_normalization, shift_after_normalization):
    y = (x - m) / np.sqrt(v + epsilon)
    y = y * gamma if scale_after_normalization else y
    return y + beta if shift_after_normalization else y

  def _opsBatchNorm(self, x, m, v, beta, gamma, epsilon,
                    scale_after_normalization, shift_after_normalization):
    y = (x - m) * tf.rsqrt(v + epsilon)
    if scale_after_normalization:
      y = gamma * y
    return y + beta if shift_after_normalization else y

  def _tfBatchNormV1(self, x, m, v, beta, gamma, epsilon,
                     scale_after_normalization):
    """Original implementation."""
    # _batch_norm_with_global_normalization is deprecated in v9
    tf.get_default_graph().graph_def_versions.producer = 8
    # pylint: disable=protected-access
    return gen_nn_ops._batch_norm_with_global_normalization(
        x, m, v, beta, gamma, epsilon, scale_after_normalization)
    # pylint: enable=protected-access

  def _tfBatchNormV1BW(self, x, m, v, beta, gamma, epsilon,
                       scale_after_normalization):
    """Re-implementation of the original kernel for backward compatibility."""
    return tf.nn.batch_norm_with_global_normalization(
        x, m, v, beta, gamma, epsilon, scale_after_normalization)

  def _tfBatchNormV2(self, x, m, v, beta, gamma, epsilon,
                     scale_after_normalization, shift_after_normalization):
    """New implementation."""
    return tf.nn.batch_normalization(
        x, m, v, beta if shift_after_normalization else None,
        gamma if scale_after_normalization else None, epsilon)

  def testBatchNorm(self):
    x_shape = [3, 5, 4, 2]
    param_shape = [2]
    x_val = np.random.random_sample(x_shape).astype(np.float32)
    m_val = np.random.random_sample(param_shape).astype(np.float32)
    v_val = np.random.random_sample(param_shape).astype(np.float32)
    beta_val = np.random.random_sample(param_shape).astype(np.float32)
    gamma_val = np.random.random_sample(param_shape).astype(np.float32)
    for use_gpu in [True, False]:
      with self.test_session(use_gpu=use_gpu) as sess:
        x = tf.constant(x_val, name="x")
        m = tf.constant(m_val, name="m")
        v = tf.constant(v_val, name="v")
        beta = tf.constant(beta_val, name="beta")
        gamma = tf.constant(gamma_val, name="gamma")
        epsilon = 0.001
        for scale_after_normalization in [True, False]:
          for shift_after_normalization in [True, False]:
            bn2 = self._tfBatchNormV2(
                x, m, v, beta, gamma, epsilon, scale_after_normalization,
                shift_after_normalization)
            bn1bw = self._tfBatchNormV1BW(
                x, m, v, beta, gamma, epsilon, scale_after_normalization)
            bn1 = self._tfBatchNormV1(
                x, m, v, beta, gamma, epsilon, scale_after_normalization)
            on = self._opsBatchNorm(
                x, m, v, beta, gamma, epsilon, scale_after_normalization,
                shift_after_normalization)
            np_bn = self._npBatchNorm(
                x_val, m_val, v_val, beta_val, gamma_val, epsilon,
                scale_after_normalization, shift_after_normalization)
            tf_bn_v2, tf_bn_v1bw, tf_bn_v1, ops_bn = sess.run(
                [bn2, bn1bw, bn1, on])
            self.assertAllClose(np_bn, ops_bn, atol=0.00001)
            self.assertAllClose(np_bn, tf_bn_v2, atol=0.00001)
            self.assertAllClose(tf_bn_v2, ops_bn, atol=0.00001)
            # shift_after_normalization=False is not supported in v1.
            if shift_after_normalization:
              self.assertAllClose(np_bn, tf_bn_v1bw, atol=0.00001)
              self.assertAllClose(np_bn, tf_bn_v1, atol=0.00001)
              self.assertAllClose(tf_bn_v1, ops_bn, atol=0.00001)
              self.assertAllClose(tf_bn_v1bw, ops_bn, atol=0.00001)

  def _testBatchNormGradient(self, param_index, tag, scale_after_normalization,
                             shift_after_normalization, version,
                             err_tolerance=1e-11):
    x_shape = [3, 5, 4, 5]
    param_shape = [5]
    np.random.seed(1)  # Make it reproducible.
    x_val = np.random.random_sample(x_shape).astype(np.float64)
    m_val = np.random.random_sample(param_shape).astype(np.float64)
    v_val = np.random.random_sample(param_shape).astype(np.float64)
    beta_val = np.random.random_sample(param_shape).astype(np.float64)
    gamma_val = np.random.random_sample(param_shape).astype(np.float64)
    with self.test_session():
      x = tf.constant(x_val, name="x")
      m = tf.constant(m_val, name="m")
      v = tf.constant(v_val, name="v")
      beta = tf.constant(beta_val, name="beta")
      gamma = tf.constant(gamma_val, name="gamma")
      epsilon = 0.001
      if version == 1:
        output = self._tfBatchNormV1(
            x, m, v, beta, gamma, epsilon, scale_after_normalization)
      elif version == 2:
        output = self._tfBatchNormV2(
            x, m, v, beta, gamma, epsilon, scale_after_normalization,
            shift_after_normalization)
      else:
        print("Invalid version", version)
        raise ValueError()
      all_params = [x, m, v, beta, gamma]
      all_shapes = [x_shape, param_shape, param_shape, param_shape, param_shape]
      err = tf.test.compute_gradient_error(
          all_params[param_index], all_shapes[param_index], output, x_shape)
    print("Batch normalization v%d %s gradient %s scale and %s shift err = " %
          (version, tag, "with" if scale_after_normalization else "without",
           "with" if shift_after_normalization else "without"),
          err)
    self.assertLess(err, err_tolerance)

  def _testBatchNormGradientInAllNeedConfigs(
      self, param_index, tag, err_tolerance=1e-11):
    for scale_after_normalization in [True, False]:
      for shift_after_normalization in [True, False]:
        # shift_after_normalization=False is not supported in version 1.
        for v in ([1, 2] if shift_after_normalization else [2]):
          self._testBatchNormGradient(
              param_index, tag, scale_after_normalization,
              shift_after_normalization, v, err_tolerance)

  def testBatchNormInputGradient(self):
    self._testBatchNormGradientInAllNeedConfigs(0, "x")

  def testBatchNormMeanGradient(self):
    self._testBatchNormGradientInAllNeedConfigs(1, "mean")

  def testBatchNormVarianceGradient(self):
    self._testBatchNormGradientInAllNeedConfigs(2, "variance",
                                                err_tolerance=1e-03)

  def testBatchNormBetaGradient(self):
    # Since beta does not exist when scale_after_normalization=False, we only
    # test for scale_after_normalization=True.
    for scale_after_normalization in [True, False]:
      for v in [1, 2]:
        self._testBatchNormGradient(3, "beta", scale_after_normalization, True,
                                    v)

  def testBatchNormGammaGradient(self):
    # If scale_after_normalization is False, backprop for gamma in v1
    # will be 0. In version 2 of the API, if scale_after_normalization is False,
    # gamma is not used at all, and the gradient is None, which displeases the
    # gradient checker.
    for scale_after_normalization in [True, False]:
      self._testBatchNormGradient(4, "gamma", scale_after_normalization, True,
                                  1)
    for shift_after_normalization in [True, False]:
      self._testBatchNormGradient(4, "gamma", True, shift_after_normalization,
                                  2)

  def testBatchNormGradImpl(self):
    x_shape = [7, 5, 4, 6]
    param_shape = [6]
    np.random.seed(1)  # Make it reproducible.
    x_val = np.random.random_sample(x_shape).astype(np.float32)
    m_val = np.random.random_sample(param_shape).astype(np.float32)
    v_val = np.random.random_sample(param_shape).astype(np.float32)
    beta_val = np.random.random_sample(param_shape).astype(np.float32)
    gamma_val = np.random.random_sample(param_shape).astype(np.float32)
    backprop_val = np.random.random_sample(x_shape).astype(np.float32)
    for use_gpu in [False, True]:
      with self.test_session(use_gpu=use_gpu) as sess:
        x = tf.constant(x_val, name="x")
        m = tf.constant(m_val, name="m")
        v = tf.constant(v_val, name="v")
        beta = tf.constant(beta_val, name="beta")
        gamma = tf.constant(gamma_val, name="gamma")
        backprop = tf.constant(backprop_val, name="backprop")
        epsilon = 0.001
        for scale_after_normalization in [True, False]:
          # _batch_norm_with_global_normalization_grad is deprecated in v9
          tf.get_default_graph().graph_def_versions.producer = 8
          dx, dm, dv, db, dg = (
              gen_nn_ops._batch_norm_with_global_normalization_grad(
              x, m, v, gamma, backprop, epsilon, scale_after_normalization))
          on = self._opsBatchNorm(
              x, m, v, beta, gamma, epsilon, scale_after_normalization, True)
          odx, odm, odv, odb, odg = tf.gradients(
              [on], [x, m, v, beta, gamma], [backprop])
          if scale_after_normalization:
            all_grads = sess.run([dx, dm, dv, db, dg, odx, odm, odv, odb, odg])
            to_check = ["dx", "dm", "dv", "db", "dg"]
          else:
            all_grads = sess.run([dx, dm, dv, db, odx, odm, odv, odb])
            to_check = ["dx", "dm", "dv", "db"]
          for i, _ in enumerate(to_check):
            self.assertAllClose(
                all_grads[i + len(to_check)], all_grads[i], atol=0.000001)

  def testBatchNormKeepDims(self):
    """Test for tf.nn.moments(..., keep_dims=True / False).

    Make sure that parameters with shape (1, 1, 1, depth) yield the same
    result as parameters with shape (depth)
    """
    x_shape = (3, 5, 4, 2)
    param_shape = (2)
    keep_dims_param_shape = (1, 1, 1, 2)
    x_val = np.random.random_sample(x_shape).astype(np.float32)
    m_val = np.random.random_sample(param_shape).astype(np.float32)
    v_val = np.random.random_sample(param_shape).astype(np.float32)
    beta_val = np.random.random_sample(param_shape).astype(np.float32)
    gamma_val = np.random.random_sample(param_shape).astype(np.float32)
    for use_gpu in [True, False]:
      with self.test_session(use_gpu=use_gpu) as sess:
        x = tf.constant(x_val, name="x")
        m = tf.constant(m_val, name="m")
        v = tf.constant(v_val, name="v")
        beta = tf.constant(beta_val, name="beta")
        gamma = tf.constant(gamma_val, name="gamma")
        keep_dims_m = tf.reshape(m, keep_dims_param_shape, name="keep_dims_m")
        keep_dims_v = tf.reshape(v, keep_dims_param_shape, name="keep_dims_v")
        keep_dims_beta = tf.reshape(
            beta, keep_dims_param_shape, name="keep_dims_beta")
        keep_dims_gamma = tf.reshape(
            gamma, keep_dims_param_shape, name="keep_dims_gamma")
        epsilon = 0.001
        for scale_after_normalization in [True, False]:
          for shift_after_normalization in [True, False]:
            bn = self._tfBatchNormV2(
                x, m, v, beta, gamma, epsilon, scale_after_normalization,
                shift_after_normalization)
            keep_dims_bn = self._tfBatchNormV2(
                x, keep_dims_m, keep_dims_v, keep_dims_beta,
                keep_dims_gamma, epsilon, scale_after_normalization,
                shift_after_normalization)
            tf_batch_norm, keep_dims_tf_batch_norm = sess.run(
                [bn, keep_dims_bn])
            self.assertEquals(x_shape, tf_batch_norm.shape)
            self.assertEquals(x_shape, keep_dims_tf_batch_norm.shape)
            self.assertAllClose(
                tf_batch_norm, keep_dims_tf_batch_norm, atol=0.000001)

  def _testBatchNormArbitraryShapes(self, x_shape, param_shape, atol=0.0001):
    x_val = np.random.random_sample(x_shape).astype(np.float32)
    m_val = np.random.random_sample(param_shape).astype(np.float32)
    v_val = np.random.random_sample(param_shape).astype(np.float32)
    beta_val = np.random.random_sample(param_shape).astype(np.float32)
    gamma_val = np.random.random_sample(param_shape).astype(np.float32)
    for use_gpu in [True, False]:
      with self.test_session(use_gpu=use_gpu) as sess:
        x = tf.constant(x_val, name="x")
        m = tf.constant(m_val, name="m")
        v = tf.constant(v_val, name="v")
        beta = tf.constant(beta_val, name="beta")
        gamma = tf.constant(gamma_val, name="gamma")
        epsilon = 0.001
        for scale_after_normalization in [True, False]:
          for shift_after_normalization in [True, False]:
            bn = self._tfBatchNormV2(
                x, m, v, beta, gamma, epsilon, scale_after_normalization,
                shift_after_normalization)
            np_batch_norm = self._npBatchNorm(
                x_val, m_val, v_val, beta_val, gamma_val, epsilon,
                scale_after_normalization, shift_after_normalization)
            [tf_batch_norm] = sess.run([bn])
            self.assertEquals(x_shape, np_batch_norm.shape)
            self.assertEquals(x_shape, tf_batch_norm.shape)
            self.assertAllClose(np_batch_norm, tf_batch_norm, atol=atol)

  def testBatchNormArbitraryShapes(self):
    """Test for a variety of shapes and moments.

    Batch normalization is expected to work regardless of the position and
    dimensionality of the 'depth' axis/axes.
    """
    self._testBatchNormArbitraryShapes((3, 3), (1, 3))
    self._testBatchNormArbitraryShapes((3, 3), (3, 1))
    self._testBatchNormArbitraryShapes((3, 2, 4, 5), (1, 2, 1, 1))
    self._testBatchNormArbitraryShapes((2, 3, 2, 4, 5), (1, 1, 1, 4, 5),
                                       atol=0.005)


class SufficientStatisticsTest(tf.test.TestCase):

  def _npSuffStats(self, x, axes, shift, keep_dims):
    axis = tuple(axes)
    if shift:
      shift_value = x[[slice(None) if i not in set(axis) else slice(0, 1)
                       for i in xrange(x.ndim)]]
      m_ss = np.sum(x - shift_value, axis=axis, keepdims=keep_dims)
      v_ss = np.sum(
          (x - shift_value) * (x - shift_value),
          axis=axis,
          keepdims=keep_dims)
    else:
      shift_value = None
      m_ss = np.sum(x, axis=axis, keepdims=keep_dims)
      v_ss = np.sum(x * x, axis=axis, keepdims=keep_dims)
    count = 1.0
    for d in xrange(x.ndim):
      if d in set(axes):
        count *= x.shape[d]
    if not keep_dims:
      shift_value = np.squeeze(shift_value, axis=axis)
    return count, m_ss, v_ss, shift_value

  def _opSuffStats(self, x, axes, shift, keep_dims):
    return tf.nn.sufficient_statistics(x, axes, shift, keep_dims)

  def _testSuffStats(self, x_shape, axes, shift, keep_dims, has_shape):
    x_val = np.random.random_sample(x_shape).astype(np.float32)
    np_c, np_m, np_v, np_s = self._npSuffStats(x_val, axes, shift, keep_dims)
    for use_gpu in [True, False]:
      with self.test_session(use_gpu=use_gpu) as sess:
        if has_shape:
          x = tf.constant(x_val, name="x")
          x.set_shape(x_shape)
          op_c, op_m, op_v, op_s = self._opSuffStats(x, axes, shift, keep_dims)
          if shift:
            tf_c, tf_m, tf_v, tf_s = sess.run([op_c, op_m, op_v, op_s])
          else:
            tf_c, tf_m, tf_v = sess.run([op_c, op_m, op_v])
        else:
          x = tf.placeholder(dtype=tf.float32,
                             shape=[None] * len(x_shape),
                             name="x")
          op_c, op_m, op_v, op_s = self._opSuffStats(x, axes, shift, keep_dims)
          if shift:
            tf_c, tf_m, tf_v, tf_s = sess.run(
                [op_c, op_m, op_v, op_s],
                feed_dict={x: x_val})
          else:
            tf_c, tf_m, tf_v = sess.run(
                [op_c, op_m, op_v],
                feed_dict={x: x_val})
        self.assertAllClose(np_c, tf_c, atol=0.000001)
        self.assertAllClose(np_m, tf_m, atol=0.000001)
        self.assertAllClose(np_v, tf_v, atol=0.000001)
        if shift:
          self.assertAllClose(np_s, tf_s, atol=0.000001)

  def testSuffStats(self):
    for has_shape in [True, False]:
      for keep_dims in [True, False]:
        for shift in [True, False]:
          self._testSuffStats([2, 3], [1], shift, keep_dims, has_shape)
          self._testSuffStats([2, 3], [0], shift, keep_dims, has_shape)
          self._testSuffStats([1, 2, 3], [0, 2], shift, keep_dims, has_shape)


class NormalizeMomentsTest(tf.test.TestCase):

  def _npNormalizeMoments(self, counts, mean_ss, variance_ss, shift):
    mean = mean_ss / counts
    variance = variance_ss / counts - mean * mean
    if shift is not None:
      mean += shift
    return mean, variance

  def _opNormalizeMoments(self, counts, mean_ss, variance_ss, shift):
    return tf.nn.normalize_moments(counts, mean_ss, variance_ss, shift)

  def _testNormalizeMoments(self, shape, shift):
    counts = np.ones([1]).astype(np.float32)
    mean_ss = np.random.random_sample(shape).astype(np.float32)
    variance_ss = np.random.random_sample(shape).astype(np.float32)
    variance_ss *= variance_ss
    if shift:
      shift_v = np.random.random_sample(shape).astype(np.float32)
    else:
      shift_v = None
    npm, npv = self._npNormalizeMoments(counts, mean_ss, variance_ss, shift_v)
    for use_gpu in [True, False]:
      with self.test_session(use_gpu=use_gpu) as sess:
        tf_counts = tf.constant(counts, name="counts")
        tf_mean_ss = tf.constant(mean_ss, name="mean_ss")
        tf_variance_ss = tf.constant(variance_ss, name="variance_ss")
        if shift:
          tf_shift_v = tf.constant(shift_v, name="shift")
        else:
          tf_shift_v = None
        opm, opv = self._opNormalizeMoments(tf_counts, tf_mean_ss,
                                            tf_variance_ss, tf_shift_v)
        tfm, tfv = sess.run([opm, opv])
        self.assertAllClose(npm, tfm, atol=0.000001)
        self.assertAllClose(npv, tfv, atol=0.000001)

  def testNormalizeMoments(self):
    for shift in [True, False]:
      self._testNormalizeMoments([3], shift)
      self._testNormalizeMoments([2, 3], shift)


class MomentsTest(tf.test.TestCase):

  def RunMomentTestWithDynamicShape(self, shape, axes, keep_dims):
    with self.test_session():
      # shape = [batch, width, height, depth]
      assert len(shape) == 4

      x_numpy = np.random.normal(size=shape).astype(np.float32)
      x = tf.placeholder(tf.float32, shape=[None] * len(shape))

      mean, var = tf.nn.moments(x, axes, keep_dims=keep_dims)

      num_elements = np.prod([shape[i] for i in axes])

      ax = tuple(axes)
      expected_mean = np.sum(
          x_numpy, axis=ax, keepdims=keep_dims) / num_elements
      expected_mean_squared = np.multiply(expected_mean, expected_mean)
      expected_x_squared = np.sum(
          np.multiply(x_numpy, x_numpy),
          axis=ax,
          keepdims=keep_dims) / num_elements
      expected_variance = expected_x_squared - expected_mean_squared

      # Check that the moments are correct.
      self.assertAllClose(expected_mean, mean.eval(feed_dict={x: x_numpy}))
      self.assertAllClose(expected_variance, var.eval(feed_dict={x: x_numpy}))

  def RunMomentTest(self, shape, axes, keep_dims):
    with self.test_session():
      # shape = [batch, width, height, depth]
      assert len(shape) == 4

      x_numpy = np.random.normal(size=shape).astype(np.float32)
      x = tf.constant(x_numpy)

      mean, var = tf.nn.moments(x, axes, keep_dims=keep_dims)

      num_elements = np.prod([shape[i] for i in axes])

      ax = tuple(axes)
      expected_mean = np.sum(
          x_numpy, axis=ax, keepdims=keep_dims) / num_elements
      expected_mean_squared = np.multiply(expected_mean, expected_mean)
      expected_x_squared = np.sum(
          np.multiply(x_numpy, x_numpy),
          axis=ax,
          keepdims=keep_dims) / num_elements
      expected_variance = expected_x_squared - expected_mean_squared

      # Check that the moments are correct.
      self.assertAllClose(expected_mean, mean.eval())
      self.assertAllClose(expected_variance, var.eval())

  def testBasic(self):
    for keep_dims in [False, True]:
      self.RunMomentTest(shape=[2, 3, 5, 4], axes=[0], keep_dims=keep_dims)
      self.RunMomentTestWithDynamicShape(
          shape=[2, 3, 5, 4], axes=[0], keep_dims=keep_dims)

  def testGlobalNormalization(self):
    for keep_dims in [False, True]:
      self.RunMomentTest(
          shape=[2, 3, 5, 4], axes=[0, 1, 2], keep_dims=keep_dims)
      self.RunMomentTestWithDynamicShape(
          shape=[2, 3, 5, 4], axes=[0, 1, 2], keep_dims=keep_dims)

  def testAxes(self):
    for keep_dims in [False, True]:
      self.RunMomentTest(
          shape=[2, 3, 5, 4], axes=[1, 2, 3], keep_dims=keep_dims)
      self.RunMomentTestWithDynamicShape(
          shape=[2, 3, 5, 4], axes=[1, 2, 3], keep_dims=keep_dims)

  def _testGlobalGradient(self, from_y="mean"):
    with self.test_session():
      x_shape = [3, 5, 4, 2]
      x_val = np.random.random_sample(x_shape).astype(np.float64)
      x = tf.constant(x_val)
      x.set_shape(x_shape)

      axes = [0, 1, 2]
      y_shape = [2]  # Depth of x
      out_mean, out_var = tf.nn.moments(x, axes)
      if from_y == "mean":
        y = out_mean
      elif from_y == "var":
        y = out_var
      err = tf.test.compute_gradient_error(x, x_shape, y, y_shape)
      print("Moments %s gradient err = %g" % (from_y, err))
      self.assertLess(err, 1e-11)

  def testMeanGlobalGradient(self):
    self._testGlobalGradient(from_y="mean")

  def testVarGlobalGradient(self):
    self._testGlobalGradient(from_y="var")

  def testOutputNamesNoKeep(self):
    """Make sure the output names are stable."""
    with self.test_session():
      mean, var = tf.nn.moments(tf.constant([1]), [0], keep_dims=False)
      self.assertEquals(mean.op.name, "moments/normalize/mean")
      self.assertEquals(var.op.name, "moments/normalize/variance")

  def testOutputNamesKeep(self):
    """Make sure the output names are stable."""
    with self.test_session():
      mean, var = tf.nn.moments(tf.constant([1]), [0], keep_dims=True)
      self.assertEquals(mean.op.name, "moments/normalize/mean")
      self.assertEquals(var.op.name, "moments/normalize/variance")


class ComputeSampledLogitsTest(tf.test.TestCase):

  def setUp(self):
    self._num_classes = 5
    self._dim = 10
    self._batch_size = 3
    self._num_shards = 3

  def _GenerateTestInputs(self):
    np.random.seed(0)
    weights = np.random.randn(self._num_classes, self._dim).astype(np.float32)
    biases = np.random.randn(self._num_classes).astype(np.float32)
    hidden_acts = np.random.randn(self._batch_size, self._dim).astype(
        np.float32)
    sharded_weights = [
        weights[[row for row in range(self._num_classes)
                 if row % self._num_shards == shard]]
        for shard in range(self._num_shards)]
    return weights, biases, hidden_acts, sharded_weights

  def _ComputeSampledLogitsNP(self, true_w, true_b, sampled_w, sampled_b,
                              hidden_acts,
                              num_true=1,
                              true_expected=None,
                              sampled_expected=None):

    batch_size, dim = hidden_acts.shape
    true_logits = np.sum(
        hidden_acts.reshape((batch_size, 1, dim)) * true_w.reshape(
            (batch_size, num_true, dim)),
        axis=2)
    true_b = true_b.reshape((batch_size, num_true))
    true_logits += true_b
    sampled_logits = np.dot(hidden_acts, sampled_w.T) + sampled_b

    if true_expected is not None:
      true_logits -= np.log(true_expected)
    if sampled_expected is not None:
      sampled_logits -= np.log(sampled_expected[np.newaxis, :])

    out_logits = np.concatenate([true_logits, sampled_logits], axis=1)
    out_labels = np.hstack((np.ones_like(true_logits) / num_true,
                            np.zeros_like(sampled_logits)))

    return out_logits, out_labels

  def _ComputeSampledLogitsTF(self, weights, biases, hidden_acts, labels,
                              num_sampled, num_classes, num_true, sampled_vals,
                              subtract_log_q, remove_accidental_hits,
                              name="sampled_loss_TF"):
    # Should be called from within a `with test_session():` block
    if isinstance(weights, list):
      weights_tf = [tf.constant(shard) for shard in weights]
    else:
      weights_tf = tf.constant(weights)
    biases_tf = tf.constant(biases)
    hidden_acts_tf = tf.constant(hidden_acts,
                                 shape=(self._batch_size, self._dim))
    labels_tf = tf.constant(labels,
                            dtype=tf.int64,
                            shape=(self._batch_size, num_true))

    pred_logits_tf, pred_labels_tf = tf.nn._compute_sampled_logits(
        weights_tf,
        biases_tf,
        hidden_acts_tf,
        labels_tf,
        num_sampled,
        num_classes,
        num_true,
        sampled_vals,
        subtract_log_q=subtract_log_q,
        remove_accidental_hits=remove_accidental_hits,
        name=name)
    return pred_logits_tf, pred_labels_tf

  def testComputeSampledLogitsShapes(self):
    # We just check that the shapes of the returned values are correct.
    weights, biases, hidden_acts, _ = self._GenerateTestInputs()
    sampled = [1, 0, 2, 3]
    num_sampled = len(sampled)
    true_exp = sampled_exp = [1., 1., 1., 1.]
    test_sampled_vals = (sampled, true_exp, sampled_exp)
    sampled_w, sampled_b = weights[sampled], biases[sampled]

    with self.test_session() as sess:
      for num_true_test in range(1, 5):
        labels = np.random.randint(low=0, high=self._num_classes,
                                   size=self._batch_size * num_true_test)
        true_w, true_b = weights[labels], biases[labels]

        logits_np, labels_np = self._ComputeSampledLogitsNP(
            true_w, true_b, sampled_w, sampled_b, hidden_acts,
            num_true=num_true_test)

        logits_tf, labels_tf = self._ComputeSampledLogitsTF(
            weights, biases, hidden_acts, labels, num_sampled,
            self._num_classes,
            num_true=num_true_test,
            sampled_vals=test_sampled_vals,
            remove_accidental_hits=True,
            subtract_log_q=False)

      logits_tf_val, labels_tf_val = sess.run([logits_tf, labels_tf])
      self.assertEqual(logits_np.shape, logits_tf_val.shape)
      self.assertEqual(labels_np.shape, labels_tf_val.shape)

  def testComputeSampledLogitsValues(self):
    # Here we check the actual numerics.
    weights, biases, hidden_acts, sharded_weights = self._GenerateTestInputs()
    eps = 1e-3
    sampled = [1, 0, 2, 3]
    num_sampled = len(sampled)
    true_exp = np.empty([self._batch_size, 1], dtype=np.float32)
    true_exp.fill(0.5)
    sampled_exp = np.empty([num_sampled], dtype=np.float32)
    sampled_exp.fill(0.5)
    sampled_w, sampled_b = weights[sampled], biases[sampled]
    test_sampled_vals = (sampled, true_exp, sampled_exp)

    with self.test_session() as sess:
      for num_true_test in range(1, 5):
        # Generate test data for this run
        labels = np.random.randint(low=0, high=self._num_classes,
                                   size=self._batch_size * num_true_test)
        true_w, true_b = weights[labels], biases[labels]

        # Test 1: Without accidental hit removal or subtract_log_q
        logits_np, labels_np = self._ComputeSampledLogitsNP(
            true_w, true_b, sampled_w, sampled_b, hidden_acts,
            num_true=num_true_test)
        logits_tf, labels_tf = self._ComputeSampledLogitsTF(
            weights, biases, hidden_acts, labels, num_sampled,
            self._num_classes,
            num_true=num_true_test,
            sampled_vals=test_sampled_vals,
            subtract_log_q=False,
            remove_accidental_hits=False,
            name="sampled_loss_test1_num_true%d" % num_true_test)

        logits_tf_val, labels_tf_val = sess.run([logits_tf, labels_tf])
        self.assertAllClose(logits_np, logits_tf_val, eps)
        self.assertAllClose(labels_np, labels_tf_val, eps)

        # Test 2: With accidental hit removal, no subtract_log_q
        logits_tf, labels_tf = self._ComputeSampledLogitsTF(
            weights, biases, hidden_acts, labels, num_sampled,
            self._num_classes,
            num_true=num_true_test,
            sampled_vals=test_sampled_vals,
            subtract_log_q=False,
            remove_accidental_hits=True,
            name="sampled_loss_test2_num_true%d" % num_true_test)

        # Test that the exponentiated logits of accidental hits are near 0.
        # First we need to find the hits in this random test run:
        labels_reshape = labels.reshape((self._batch_size, num_true_test))
        logits_tf_np = logits_tf.eval()
        for row in xrange(self._batch_size):
          row_labels = labels_reshape[row, :]
          for col in xrange(num_sampled):
            if sampled[col] in row_labels:
              # We need to add the num_true_test offset into logits_*
              self.assertNear(
                  np.exp(logits_tf_np[row, col + num_true_test]), 0., eps)

        # Test 3: With subtract_log_q, no accidental hit removal
        logits_np, labels_np = self._ComputeSampledLogitsNP(
            true_w, true_b, sampled_w, sampled_b, hidden_acts,
            num_true=num_true_test,
            true_expected=true_exp,
            sampled_expected=sampled_exp)
        logits_tf, labels_tf = self._ComputeSampledLogitsTF(
            weights, biases, hidden_acts, labels, num_sampled,
            self._num_classes,
            num_true=num_true_test,
            sampled_vals=test_sampled_vals,
            subtract_log_q=True,
            remove_accidental_hits=False,
            name="sampled_loss_test3_num_true%d" % num_true_test)

        logits_tf_val, labels_tf_val = sess.run([logits_tf, labels_tf])
        self.assertAllClose(logits_np, logits_tf_val, eps)
        self.assertAllClose(labels_np, labels_tf_val, eps)

        # Test 4: Test 1, with sharded weights
        logits_np, labels_np = self._ComputeSampledLogitsNP(
            true_w, true_b, sampled_w, sampled_b, hidden_acts,
            num_true=num_true_test)
        logits_tf, labels_tf = self._ComputeSampledLogitsTF(
            sharded_weights, biases, hidden_acts, labels, num_sampled,
            self._num_classes,
            num_true=num_true_test,
            sampled_vals=test_sampled_vals,
            subtract_log_q=False,
            remove_accidental_hits=False,
            name="sampled_loss_test1_num_true%d" % num_true_test)

        logits_tf_val, labels_tf_val = sess.run([logits_tf, labels_tf])
        self.assertAllClose(logits_np, logits_tf_val, eps)
        self.assertAllClose(labels_np, labels_tf_val, eps)

  def testNCELoss(self):
    # A simple test to verify the numerics.

    def _SigmoidCrossEntropyWithLogits(logits, targets):
      # logits, targets: float arrays of the same shape.
      assert logits.shape == targets.shape
      pred = 1. / (1. + np.exp(-logits))
      eps = 0.0001
      pred = np.minimum(np.maximum(pred, eps), 1 - eps)
      return -targets * np.log(pred) - (1. - targets) * np.log(1. - pred)

    weights, biases, hidden_acts, sharded_weights = self._GenerateTestInputs()
    labels = [0, 1, 2]
    true_w, true_b = weights[labels], biases[labels]
    sampled = [1, 0, 2, 3]
    num_sampled = len(sampled)
    true_exp = np.empty([self._batch_size, 1], dtype=np.float32)
    true_exp.fill(0.5)
    sampled_exp = np.empty([num_sampled], dtype=np.float32)
    sampled_exp.fill(0.5)
    sampled_w, sampled_b = weights[sampled], biases[sampled]
    test_sampled_vals = (sampled, true_exp, sampled_exp)

    with self.test_session():
      logits_np, labels_np = self._ComputeSampledLogitsNP(
          true_w, true_b, sampled_w, sampled_b, hidden_acts,
          true_expected=true_exp,
          sampled_expected=sampled_exp)
      nce_loss_np = np.sum(
          _SigmoidCrossEntropyWithLogits(logits_np, labels_np), 1)

      labels_tf = tf.constant(labels, shape=(self._batch_size, 1))
      weights_tf = tf.constant(weights)
      biases_tf = tf.constant(biases)
      inputs_tf = tf.constant(hidden_acts)

      nce_loss_tf = tf.nn.nce_loss(weights_tf,
                                   biases_tf,
                                   inputs_tf,
                                   labels_tf,
                                   num_sampled=1,
                                   num_classes=self._num_classes,
                                   num_true=1,
                                   sampled_values=test_sampled_vals)

      self.assertAllClose(nce_loss_np, nce_loss_tf.eval(), 1e-4)

      # Test with sharded weights
      nce_loss_tf = tf.nn.nce_loss(
          [tf.constant(shard) for shard in sharded_weights],
          biases_tf,
          inputs_tf,
          labels_tf,
          num_sampled=1,
          num_classes=self._num_classes,
          num_true=1,
          sampled_values=test_sampled_vals)

      self.assertAllClose(nce_loss_np, nce_loss_tf.eval(), 1e-4)

  def testSampledSoftmaxLoss(self):
    # A simple test to verify the numerics.

    def _SoftmaxCrossEntropyWithLogits(logits, targets):
      # logits, targets: float arrays of the same shape.
      assert logits.shape == targets.shape
      stable_exp_logits = np.exp(logits - np.amax(
          logits, axis=1, keepdims=True))
      pred = stable_exp_logits / np.sum(stable_exp_logits, 1, keepdims=True)
      return -np.sum(targets * np.log(pred + 1.0e-20), axis=1)

    weights, biases, hidden_acts, sharded_weights = self._GenerateTestInputs()
    labels = [0, 1, 2]
    true_w, true_b = weights[labels], biases[labels]
    sampled = [1, 0, 2, 3]
    num_sampled = len(sampled)
    true_exp = np.full([self._batch_size, 1], fill_value=0.5, dtype=np.float32)
    sampled_exp = np.full([num_sampled], fill_value=0.5, dtype=np.float32)
    sampled_w, sampled_b = weights[sampled], biases[sampled]
    test_sampled_vals = (sampled, true_exp, sampled_exp)

    with self.test_session():
      logits_np, labels_np = self._ComputeSampledLogitsNP(
          true_w, true_b, sampled_w, sampled_b, hidden_acts,
          true_expected=true_exp,
          sampled_expected=sampled_exp)
      sampled_softmax_loss_np = _SoftmaxCrossEntropyWithLogits(logits_np,
                                                               labels_np)

      labels_tf = tf.constant(labels, shape=(self._batch_size, 1))
      weights_tf = tf.constant(weights)
      biases_tf = tf.constant(biases)
      inputs_tf = tf.constant(hidden_acts)

      sampled_softmax_loss_tf = tf.nn.sampled_softmax_loss(
          weights_tf,
          biases_tf,
          inputs_tf,
          labels_tf,
          num_sampled=1,
          num_classes=self._num_classes,
          num_true=1,
          sampled_values=test_sampled_vals,
          remove_accidental_hits=False)

      self.assertAllClose(
          sampled_softmax_loss_np, sampled_softmax_loss_tf.eval(), 1e-4)

      # Test with sharded weights
      sampled_softmax_loss_tf = tf.nn.sampled_softmax_loss(
          [tf.constant(shard) for shard in sharded_weights],
          biases_tf,
          inputs_tf,
          labels_tf,
          num_sampled=1,
          num_classes=self._num_classes,
          num_true=1,
          sampled_values=test_sampled_vals,
          remove_accidental_hits=False)

      self.assertAllClose(
          sampled_softmax_loss_np, sampled_softmax_loss_tf.eval(), 1e-4)


if __name__ == "__main__":
  tf.test.main()
