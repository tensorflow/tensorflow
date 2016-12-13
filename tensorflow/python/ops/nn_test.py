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
"""Tests for miscellaneous functionality in tensorflow.ops.nn."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.python.ops.nn_impl import _compute_sampled_logits


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


class LogPoissonLossTest(tf.test.TestCase):

  def _log_poisson_loss(self, x, z, compute_full_loss=False):
    lpl = np.exp(x) - z * x
    if compute_full_loss:
      stirling_approx = z * np.log(z) - z + 0.5 * np.log(2. * np.pi * z)
      lpl += np.ma.masked_array(stirling_approx, mask=(z <= 1)).filled(0.)
    return lpl

  def testLogPoissonLoss(self):
    x_shape = [5, 10]
    x_np = np.random.randn(*x_shape).astype(np.float32)
    z_np = np.random.randint(0, 5, size=x_shape).astype(np.float32)
    y_np = self._log_poisson_loss(x_np, z_np, compute_full_loss=False)
    y_np_stirling = self._log_poisson_loss(x_np, z_np, compute_full_loss=True)
    with self.test_session():
      y_tf = tf.nn.log_poisson_loss(z_np, x_np, compute_full_loss=False)
      y_tf_stirling = tf.nn.log_poisson_loss(z_np, x_np, compute_full_loss=True)
      y_tf_np = y_tf.eval()
      y_tf_np_stirling = y_tf_stirling.eval()
    eps = 1e-3
    self.assertAllClose(y_tf_np, y_np, eps)
    self.assertAllClose(y_tf_np_stirling, y_np_stirling, eps)

  def testGradient(self):
    x_shape = [5, 10]
    x_np = np.random.randn(*x_shape).astype(np.float64)
    z_np = np.random.randint(0, 5, size=x_shape).astype(np.float64)
    with self.test_session():
      x_tf = tf.constant(x_np)
      y_tf = tf.nn.log_poisson_loss(z_np, x_tf, compute_full_loss=False)
      y_tf_stirling = tf.nn.log_poisson_loss(z_np, x_tf, compute_full_loss=True)
      err = tf.test.compute_gradient_error(x_tf, x_shape, y_tf, x_shape)
      err_stirling = tf.test.compute_gradient_error(x_tf, x_shape,
                                                    y_tf_stirling, x_shape)
    eps = 1e-6
    self.assertLess(err, eps)
    self.assertLess(err_stirling, eps)


class LogSoftmaxTest(tf.test.TestCase):

  def _log_softmax(self, x):
    assert len(x.shape) == 2
    m = x.max(1)[:, np.newaxis]
    u = x - m
    return u - np.log(np.sum(np.exp(u), 1, keepdims=True))

  def testLogSoftmax(self):
    x_shape = [5, 10]
    x_np = np.random.randn(*x_shape).astype(np.float32)
    y_np = self._log_softmax(x_np)
    with self.test_session():
      x_tf = tf.constant(x_np)
      y_tf = tf.nn.log_softmax(x_tf)
      y_tf_np = y_tf.eval()
    eps = 1e-3
    self.assertAllClose(y_tf_np, y_np, eps)

  def testGradient(self):
    x_shape = [5, 10]
    x_np = np.random.randn(*x_shape).astype(np.float64)
    with self.test_session():
      x_tf = tf.constant(x_np)
      y_tf = tf.nn.log_softmax(x_tf)
      err = tf.test.compute_gradient_error(x_tf, x_shape, y_tf, x_shape)
    eps = 1e-7
    self.assertLess(err, eps)


class L2LossTest(tf.test.TestCase):

  def testL2Loss(self):
    for dtype in [tf.float32, tf.float64]:
      with self.test_session():
        x = tf.constant(
            [1.0, 0.0, 3.0, 2.0], shape=[2, 2], name="x", dtype=dtype)
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
    if isinstance(dim, list):
      norm = np.linalg.norm(x, axis=tuple(dim))
      for d in dim:
        norm = np.expand_dims(norm, d)
      return x / norm
    else:
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

  def testL2NormalizeDimArray(self):
    x_shape = [20, 7, 3]
    np.random.seed(1)
    x_np = np.random.random_sample(x_shape).astype(np.float32)
    dim = [1, 2]
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
    dropout_x = tf.nn.dropout(
        x, keep_prob, noise_shape=tf.placeholder(tf.int32))
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

  def testNoDropoutFast(self):
    x = tf.zeros((5,))
    for p in 1, tf.constant(1.0):
      y = tf.nn.dropout(x, keep_prob=p)
      self.assertTrue(x is y)


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
    hidden_acts = np.random.randn(self._batch_size,
                                  self._dim).astype(np.float32)
    sharded_weights = [
        weights[[
            row for row in range(self._num_classes)
            if row % self._num_shards == shard
        ]] for shard in range(self._num_shards)
    ]
    return weights, biases, hidden_acts, sharded_weights

  def _ComputeSampledLogitsNP(self,
                              true_w,
                              true_b,
                              sampled_w,
                              sampled_b,
                              hidden_acts,
                              num_true=1,
                              true_expected=None,
                              sampled_expected=None):

    batch_size, dim = hidden_acts.shape
    true_logits = np.sum(hidden_acts.reshape(
        (batch_size, 1, dim)) * true_w.reshape((batch_size, num_true, dim)),
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

  def _ComputeSampledLogitsTF(self,
                              weights,
                              biases,
                              hidden_acts,
                              labels,
                              num_sampled,
                              num_classes,
                              num_true,
                              sampled_vals,
                              subtract_log_q,
                              remove_accidental_hits,
                              name="sampled_loss_TF"):
    # Should be called from within a `with test_session():` block
    if isinstance(weights, list):
      weights_tf = [tf.constant(shard) for shard in weights]
    else:
      weights_tf = tf.constant(weights)
    biases_tf = tf.constant(biases)
    hidden_acts_tf = tf.constant(
        hidden_acts, shape=(self._batch_size, self._dim))
    labels_tf = tf.constant(
        labels, dtype=tf.int64, shape=(self._batch_size, num_true))

    pred_logits_tf, pred_labels_tf = _compute_sampled_logits(
        weights_tf,
        biases_tf,
        labels_tf,
        hidden_acts_tf,
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
        labels = np.random.randint(
            low=0,
            high=self._num_classes,
            size=self._batch_size * num_true_test)
        true_w, true_b = weights[labels], biases[labels]

        logits_np, labels_np = self._ComputeSampledLogitsNP(
            true_w,
            true_b,
            sampled_w,
            sampled_b,
            hidden_acts,
            num_true=num_true_test)

        logits_tf, labels_tf = self._ComputeSampledLogitsTF(
            weights,
            biases,
            hidden_acts,
            labels,
            num_sampled,
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
        labels = np.random.randint(
            low=0,
            high=self._num_classes,
            size=self._batch_size * num_true_test)
        true_w, true_b = weights[labels], biases[labels]

        # Test 1: Without accidental hit removal or subtract_log_q
        logits_np, labels_np = self._ComputeSampledLogitsNP(
            true_w,
            true_b,
            sampled_w,
            sampled_b,
            hidden_acts,
            num_true=num_true_test)
        logits_tf, labels_tf = self._ComputeSampledLogitsTF(
            weights,
            biases,
            hidden_acts,
            labels,
            num_sampled,
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
            weights,
            biases,
            hidden_acts,
            labels,
            num_sampled,
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
            true_w,
            true_b,
            sampled_w,
            sampled_b,
            hidden_acts,
            num_true=num_true_test,
            true_expected=true_exp,
            sampled_expected=sampled_exp)
        logits_tf, labels_tf = self._ComputeSampledLogitsTF(
            weights,
            biases,
            hidden_acts,
            labels,
            num_sampled,
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
            true_w,
            true_b,
            sampled_w,
            sampled_b,
            hidden_acts,
            num_true=num_true_test)
        logits_tf, labels_tf = self._ComputeSampledLogitsTF(
            sharded_weights,
            biases,
            hidden_acts,
            labels,
            num_sampled,
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
          true_w,
          true_b,
          sampled_w,
          sampled_b,
          hidden_acts,
          true_expected=true_exp,
          sampled_expected=sampled_exp)
      nce_loss_np = np.sum(
          _SigmoidCrossEntropyWithLogits(logits_np, labels_np), 1)

      labels_tf = tf.constant(labels, shape=(self._batch_size, 1))
      weights_tf = tf.constant(weights)
      biases_tf = tf.constant(biases)
      inputs_tf = tf.constant(hidden_acts)

      nce_loss_tf = tf.nn.nce_loss(
          weights_tf,
          biases_tf,
          labels_tf,
          inputs_tf,
          num_sampled=1,
          num_classes=self._num_classes,
          num_true=1,
          sampled_values=test_sampled_vals)

      self.assertAllClose(nce_loss_np, nce_loss_tf.eval(), 1e-4)

      # Test with sharded weights
      nce_loss_tf = tf.nn.nce_loss(
          [tf.constant(shard) for shard in sharded_weights],
          biases_tf,
          labels_tf,
          inputs_tf,
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
          true_w,
          true_b,
          sampled_w,
          sampled_b,
          hidden_acts,
          true_expected=true_exp,
          sampled_expected=sampled_exp)
      sampled_softmax_loss_np = _SoftmaxCrossEntropyWithLogits(logits_np,
                                                               labels_np)

      labels_tf = tf.constant(labels, shape=(self._batch_size, 1))
      weights_tf = tf.constant(weights)
      biases_tf = tf.constant(biases)
      inputs_tf = tf.constant(hidden_acts)

      sampled_softmax_loss_tf = tf.nn.sampled_softmax_loss(
          weights=weights_tf,
          biases=biases_tf,
          labels=labels_tf,
          inputs=inputs_tf,
          num_sampled=1,
          num_classes=self._num_classes,
          num_true=1,
          sampled_values=test_sampled_vals,
          remove_accidental_hits=False)

      self.assertAllClose(sampled_softmax_loss_np,
                          sampled_softmax_loss_tf.eval(), 1e-4)

      # Test with sharded weights
      sampled_softmax_loss_tf = tf.nn.sampled_softmax_loss(
          weights=[tf.constant(shard) for shard in sharded_weights],
          biases=biases_tf,
          labels=labels_tf,
          inputs=inputs_tf,
          num_sampled=1,
          num_classes=self._num_classes,
          num_true=1,
          sampled_values=test_sampled_vals,
          remove_accidental_hits=False)

      self.assertAllClose(sampled_softmax_loss_np,
                          sampled_softmax_loss_tf.eval(), 1e-4)


class CReluTest(tf.test.TestCase):

  def test(self):
    x = np.random.rand(3, 4).astype(np.float32)
    y = np.concatenate([x * (x > 0), -x * (x < 0)], axis=1)
    with self.test_session():
      z = tf.nn.crelu(tf.constant(x)).eval()
      self.assertAllClose(y, z, 1e-4)


if __name__ == "__main__":
  tf.test.main()
