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

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_impl
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
import tensorflow.python.ops.nn_grad  # pylint: disable=unused-import
from tensorflow.python.ops.nn_impl import _compute_sampled_logits
from tensorflow.python.platform import test as test_lib


class ZeroFractionTest(test_lib.TestCase):

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
      x_tf = constant_op.constant(x_np)
      x_tf.set_shape(x_shape)
      y_tf = nn_impl.zero_fraction(x_tf)
      y_tf_np = y_tf.eval()
    eps = 1e-8
    self.assertAllClose(y_tf_np, y_np, eps)

  def testZeroFractionEmpty(self):
    with self.test_session():
      x = np.zeros(0)
      y = nn_impl.zero_fraction(x).eval()
      self.assertTrue(np.isnan(y))


class SoftmaxTest(test_lib.TestCase):

  def _softmax(self, x):
    assert len(x.shape) == 2
    m = x.max(1)[:, np.newaxis]
    u = np.exp(x - m)
    z = u.sum(1)[:, np.newaxis]
    return u / z

  @test_util.run_in_graph_and_eager_modes()
  def testSoftmax(self):
    x_shape = [5, 10]
    x_np = np.random.randn(*x_shape).astype(np.float32)
    y_np = self._softmax(x_np)
    x_tf = constant_op.constant(x_np)
    y_tf = nn_ops.softmax(x_tf)
    y_tf_last_dim = nn_ops.softmax(x_tf, 1)
    y_tf_np = self.evaluate(y_tf)
    y_tf_last_dim_np = self.evaluate(y_tf_last_dim)
    eps = 1e-3
    self.assertAllClose(y_tf_np, y_np, eps)
    self.assertAllClose(y_tf_last_dim_np, y_np, eps)

  def testGradient(self):
    x_shape = [5, 10]
    x_np = np.random.randn(*x_shape).astype(np.float64)
    with self.test_session():
      x_tf = constant_op.constant(x_np)
      y_tf = nn_ops.softmax(x_tf)
      err = gradient_checker.compute_gradient_error(x_tf, x_shape, y_tf,
                                                    x_shape)
    eps = 1e-8
    self.assertLess(err, eps)


class LogPoissonLossTest(test_lib.TestCase):

  def _log_poisson_loss(self, x, z, compute_full_loss=False):
    lpl = np.exp(x) - z * x
    if compute_full_loss:
      stirling_approx = z * np.log(z) - z + 0.5 * np.log(2. * np.pi * z)
      lpl += np.ma.masked_array(stirling_approx, mask=(z <= 1)).filled(0.)
    return lpl

  @test_util.run_in_graph_and_eager_modes()
  def testLogPoissonLoss(self):
    x_shape = [5, 10]
    x_np = np.random.randn(*x_shape).astype(np.float32)
    z_np = np.random.randint(0, 5, size=x_shape).astype(np.float32)
    y_np = self._log_poisson_loss(x_np, z_np, compute_full_loss=False)
    y_np_stirling = self._log_poisson_loss(x_np, z_np, compute_full_loss=True)
    y_tf = nn_impl.log_poisson_loss(z_np, x_np, compute_full_loss=False)
    y_tf_stirling = nn_impl.log_poisson_loss(
        z_np, x_np, compute_full_loss=True)
    y_tf_np = self.evaluate(y_tf)
    y_tf_np_stirling = self.evaluate(y_tf_stirling)
    eps = 1e-3
    self.assertAllClose(y_tf_np, y_np, eps)
    self.assertAllClose(y_tf_np_stirling, y_np_stirling, eps)

  def testGradient(self):
    x_shape = [5, 10]
    x_np = np.random.randn(*x_shape).astype(np.float64)
    z_np = np.random.randint(0, 5, size=x_shape).astype(np.float64)
    with self.test_session():
      x_tf = constant_op.constant(x_np)
      y_tf = nn_impl.log_poisson_loss(z_np, x_tf, compute_full_loss=False)
      y_tf_stirling = nn_impl.log_poisson_loss(
          z_np, x_tf, compute_full_loss=True)
      err = gradient_checker.compute_gradient_error(x_tf, x_shape, y_tf,
                                                    x_shape)
      err_stirling = gradient_checker.compute_gradient_error(
          x_tf, x_shape, y_tf_stirling, x_shape)
    eps = 1e-6
    self.assertLess(err, eps)
    self.assertLess(err_stirling, eps)


class LogSoftmaxTest(test_lib.TestCase):

  def _log_softmax(self, x):
    assert len(x.shape) == 2
    m = x.max(1)[:, np.newaxis]
    u = x - m
    return u - np.log(np.sum(np.exp(u), 1, keepdims=True))

  @test_util.run_in_graph_and_eager_modes()
  def testLogSoftmax(self):
    x_shape = [5, 10]
    x_np = np.random.randn(*x_shape).astype(np.float32)
    y_np = self._log_softmax(x_np)
    x_tf = constant_op.constant(x_np)
    y_tf = nn_ops.log_softmax(x_tf)
    y_tf_np = self.evaluate(y_tf)
    eps = 1e-3
    self.assertAllClose(y_tf_np, y_np, eps)

  def testGradient(self):
    x_shape = [5, 10]
    x_np = np.random.randn(*x_shape).astype(np.float64)
    with self.test_session():
      x_tf = constant_op.constant(x_np)
      y_tf = nn_ops.log_softmax(x_tf)
      err = gradient_checker.compute_gradient_error(x_tf, x_shape, y_tf,
                                                    x_shape)
    eps = 1e-7
    self.assertLess(err, eps)


class L2LossTest(test_lib.TestCase):

  @test_util.run_in_graph_and_eager_modes()
  def testL2Loss(self):
    for dtype in [dtypes.float32, dtypes.float64]:
      x = constant_op.constant(
          [1.0, 0.0, 3.0, 2.0], shape=[2, 2], name="x", dtype=dtype)
      l2loss = nn_ops.l2_loss(x)
      value = self.evaluate(l2loss)
      self.assertAllClose(7.0, value)

  def testGradient(self):
    x_shape = [20, 7, 3]
    np.random.seed(1)  # Make it reproducible.
    x_val = np.random.random_sample(x_shape).astype(np.float64)
    with self.test_session():
      x = constant_op.constant(x_val, name="x")
      output = nn_ops.l2_loss(x)
      err = gradient_checker.compute_gradient_error(x, x_shape, output, [1])
    print("L2Loss gradient err = %g " % err)
    err_tolerance = 1e-11
    self.assertLess(err, err_tolerance)


class L2NormalizeTest(test_lib.TestCase):

  def _l2Normalize(self, x, dim):
    if isinstance(dim, list):
      norm = np.linalg.norm(x, axis=tuple(dim))
      for d in dim:
        norm = np.expand_dims(norm, d)
      return x / norm
    else:
      norm = np.apply_along_axis(np.linalg.norm, dim, x)
      return x / np.expand_dims(norm, dim)

  @test_util.run_in_graph_and_eager_modes()
  def testL2Normalize(self):
    x_shape = [20, 7, 3]
    np.random.seed(1)
    x_np = np.random.random_sample(x_shape).astype(np.float32)
    for dim in range(len(x_shape)):
      y_np = self._l2Normalize(x_np, dim)
      x_tf = constant_op.constant(x_np, name="x")
      y_tf = nn_impl.l2_normalize(x_tf, dim)
      self.assertAllClose(y_np, self.evaluate(y_tf))

  @test_util.run_in_graph_and_eager_modes()
  def testL2NormalizeDimArray(self):
    x_shape = [20, 7, 3]
    np.random.seed(1)
    x_np = np.random.random_sample(x_shape).astype(np.float32)
    dim = [1, 2]
    y_np = self._l2Normalize(x_np, dim)
    x_tf = constant_op.constant(x_np, name="x")
    y_tf = nn_impl.l2_normalize(x_tf, dim)
    self.assertAllClose(y_np, self.evaluate(y_tf))

  def testL2NormalizeGradient(self):
    x_shape = [20, 7, 3]
    np.random.seed(1)
    x_np = np.random.random_sample(x_shape).astype(np.float64)
    for dim in range(len(x_shape)):
      with self.test_session():
        x_tf = constant_op.constant(x_np, name="x")
        y_tf = nn_impl.l2_normalize(x_tf, dim)
        err = gradient_checker.compute_gradient_error(x_tf, x_shape, y_tf,
                                                      x_shape)
      print("L2Normalize gradient err = %g " % err)
      self.assertLess(err, 1e-4)


class DropoutTest(test_lib.TestCase):

  def testDropout(self):
    # Runs dropout with 0-1 tensor 10 times, sum the number of ones and validate
    # that it is producing approximately the right number of ones over a large
    # number of samples, based on the keep probability.
    x_dim = 40
    y_dim = 30
    num_iter = 10
    for keep_prob in [0.1, 0.5, 0.8]:
      with self.test_session():
        t = constant_op.constant(
            1.0, shape=[x_dim, y_dim], dtype=dtypes.float32)
        dropout = nn_ops.dropout(t, keep_prob)
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
        t = constant_op.constant(
            1.0, shape=[x_dim, y_dim], dtype=dtypes.float32)
        dropout = nn_ops.dropout(t, keep_prob, noise_shape=[x_dim, 1])
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
        t = constant_op.constant(
            1.0, shape=[x_dim, y_dim], dtype=dtypes.float32)
        dropout = nn_ops.dropout(t, keep_prob, noise_shape=[x_dim, 1])
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
        t = constant_op.constant(
            1.0, shape=[x_dim, y_dim], dtype=dtypes.float32)
        keep_prob_placeholder = array_ops.placeholder(dtypes.float32)
        dropout = nn_ops.dropout(t, keep_prob_placeholder)
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
    x = constant_op.constant(1.0, shape=[x_dim, y_dim], dtype=dtypes.float32)
    dropout_x = nn_ops.dropout(
        x, keep_prob, noise_shape=array_ops.placeholder(dtypes.int32))
    self.assertEqual(x.get_shape(), dropout_x.get_shape())

  def testInvalidKeepProb(self):
    x_dim = 40
    y_dim = 30
    t = constant_op.constant(1.0, shape=[x_dim, y_dim], dtype=dtypes.float32)
    with self.assertRaises(ValueError):
      nn_ops.dropout(t, -1.0)
    with self.assertRaises(ValueError):
      nn_ops.dropout(t, 1.1)
    with self.assertRaises(ValueError):
      nn_ops.dropout(t, [0.0, 1.0])
    with self.assertRaises(ValueError):
      nn_ops.dropout(t, array_ops.placeholder(dtypes.float64))
    with self.assertRaises(ValueError):
      nn_ops.dropout(t, array_ops.placeholder(dtypes.float32, shape=[2]))

  def testShapedDropoutShapeError(self):
    # Runs shaped dropout and verifies an error is thrown on misshapen noise.
    x_dim = 40
    y_dim = 30
    keep_prob = 0.5
    t = constant_op.constant(1.0, shape=[x_dim, y_dim], dtype=dtypes.float32)
    with self.assertRaises(ValueError):
      _ = nn_ops.dropout(t, keep_prob, noise_shape=[x_dim, y_dim + 10])
    with self.assertRaises(ValueError):
      _ = nn_ops.dropout(t, keep_prob, noise_shape=[x_dim, y_dim, 5])
    with self.assertRaises(ValueError):
      _ = nn_ops.dropout(t, keep_prob, noise_shape=[x_dim + 3])
    with self.assertRaises(ValueError):
      _ = nn_ops.dropout(t, keep_prob, noise_shape=[x_dim])
    # test that broadcasting proceeds
    _ = nn_ops.dropout(t, keep_prob, noise_shape=[y_dim])
    _ = nn_ops.dropout(t, keep_prob, noise_shape=[1, y_dim])
    _ = nn_ops.dropout(t, keep_prob, noise_shape=[x_dim, 1])
    _ = nn_ops.dropout(t, keep_prob, noise_shape=[1, 1])

  def testNoDropoutFast(self):
    x = array_ops.zeros((5,))
    for p in 1, constant_op.constant(1.0):
      y = nn_ops.dropout(x, keep_prob=p)
      self.assertTrue(x is y)

  def testDropoutWithIntegerInputs(self):
    x = constant_op.constant([1, 1, 1, 1, 1])
    with self.assertRaises(ValueError):
      _ = nn_ops.dropout(x, 0.5)


class ComputeSampledLogitsTest(test_lib.TestCase):

  def setUp(self):
    self._eps = 1e-3

  def _GenerateTestData(self, num_classes, dim, batch_size, num_true, labels,
                        sampled, subtract_log_q):
    """Randomly generates input/output data for a single test case.

    This function returns numpy constants for use in a test case.

    Args:
      num_classes: An int. The number of embedding classes in the test case.
      dim: An int. The dimension of the embedding.
      batch_size: An int. The batch size.
      num_true: An int. The number of target classes per training example.
      labels: A list of batch_size * num_true ints. The target classes.
      sampled: A list of indices in [0, num_classes).
      subtract_log_q: A bool corresponding to the parameter in
          _compute_sampled_logits().

    Returns:
      weights: Embedding weights to use as test input. It is a numpy array
          of shape [num_classes, dim]
      biases: Embedding biases to use as test input. It is a numpy array
          of shape [num_classes].
      hidden_acts: Forward activations of the network to use as test input.
          It is a numpy array of shape [batch_size, dim].
      sampled_vals: A tuple based on `sampled` to use as test input in the
          format returned by a *_candidate_sampler function.
      exp_logits: The output logits expected from _compute_sampled_logits().
          It is a numpy array of shape [batch_size, num_true + len(sampled)].
      exp_labels: The output labels expected from _compute_sampled_logits().
          It is a numpy array of shape [batch_size, num_true + len(sampled)].
    """
    weights = np.random.randn(num_classes, dim).astype(np.float32)
    biases = np.random.randn(num_classes).astype(np.float32)
    hidden_acts = np.random.randn(batch_size, dim).astype(np.float32)

    true_exp = np.full([batch_size, 1], fill_value=0.5, dtype=np.float32)
    sampled_exp = np.full([len(sampled)], fill_value=0.5, dtype=np.float32)
    sampled_vals = (sampled, true_exp, sampled_exp)

    sampled_w, sampled_b = weights[sampled], biases[sampled]
    true_w, true_b = weights[labels], biases[labels]

    true_logits = np.sum(
        hidden_acts.reshape((batch_size, 1, dim)) * true_w.reshape(
            (batch_size, num_true, dim)),
        axis=2)
    true_b = true_b.reshape((batch_size, num_true))
    true_logits += true_b
    sampled_logits = np.dot(hidden_acts, sampled_w.T) + sampled_b

    if subtract_log_q:
      true_logits -= np.log(true_exp)
      sampled_logits -= np.log(sampled_exp[np.newaxis, :])

    exp_logits = np.concatenate([true_logits, sampled_logits], axis=1)
    exp_labels = np.hstack((np.ones_like(true_logits) / num_true,
                            np.zeros_like(sampled_logits)))

    return weights, biases, hidden_acts, sampled_vals, exp_logits, exp_labels

  def _ShardTestEmbeddings(self, weights, biases, num_shards):
    """Shards the weights and biases returned by _GenerateTestData.

    Args:
      weights: The weights returned by _GenerateTestData.
      biases: The biases returned by _GenerateTestData.
      num_shards: The number of shards to create.

    Returns:
      sharded_weights: A list of size `num_shards` containing all the weights.
      sharded_biases: A list of size `num_shards` containing all the biases.
    """
    with ops.Graph().as_default() as g:
      sharded_weights = variable_scope.get_variable(
          "w",
          partitioner=partitioned_variables.fixed_size_partitioner(num_shards),
          initializer=constant_op.constant(weights))
      sharded_biases = variable_scope.get_variable(
          "b",
          partitioner=partitioned_variables.fixed_size_partitioner(num_shards),
          initializer=constant_op.constant(biases))
      with self.test_session(graph=g) as sess:
        variables.global_variables_initializer().run()
        return sess.run([list(sharded_weights), list(sharded_biases)])

  def testShapes(self):
    np.random.seed(0)
    num_classes = 5
    batch_size = 3
    with self.test_session() as sess:
      for num_true in range(1, 5):
        labels = np.random.randint(
            low=0, high=num_classes, size=batch_size * num_true)
        (weights, biases, hidden_acts, sampled_vals, exp_logits,
         exp_labels) = self._GenerateTestData(
             num_classes=num_classes,
             dim=10,
             batch_size=batch_size,
             num_true=num_true,
             labels=labels,
             sampled=[1, 0, 2, 3],
             subtract_log_q=False)
        logits_tensor, labels_tensor = _compute_sampled_logits(
            weights=constant_op.constant(weights),
            biases=constant_op.constant(biases),
            labels=constant_op.constant(
                labels, dtype=dtypes.int64, shape=(batch_size, num_true)),
            inputs=constant_op.constant(hidden_acts),
            num_sampled=4,
            num_classes=num_classes,
            num_true=num_true,
            sampled_values=sampled_vals,
            subtract_log_q=False,
            remove_accidental_hits=False,
            partition_strategy="div",
            name="sampled_logits_basic_num_true_%d" % num_true)
        got_logits, got_labels = sess.run([logits_tensor, labels_tensor])
        self.assertEqual(exp_logits.shape, got_logits.shape, self._eps)
        self.assertEqual(exp_labels.shape, got_labels.shape, self._eps)

  def testBasic(self):
    """Without accidental hit removal or subtract_log_q."""
    np.random.seed(0)
    num_classes = 5
    batch_size = 3
    with self.test_session() as sess:
      for num_true in range(1, 5):
        labels = np.random.randint(
            low=0, high=num_classes, size=batch_size * num_true)
        (weights, biases, hidden_acts, sampled_vals, exp_logits,
         exp_labels) = self._GenerateTestData(
             num_classes=num_classes,
             dim=10,
             batch_size=batch_size,
             num_true=num_true,
             labels=labels,
             sampled=[1, 0, 2, 3],
             subtract_log_q=False)
        logits_tensor, labels_tensor = _compute_sampled_logits(
            weights=constant_op.constant(weights),
            biases=constant_op.constant(biases),
            labels=constant_op.constant(
                labels, dtype=dtypes.int64, shape=(batch_size, num_true)),
            inputs=constant_op.constant(hidden_acts),
            num_sampled=4,
            num_classes=num_classes,
            num_true=num_true,
            sampled_values=sampled_vals,
            subtract_log_q=False,
            remove_accidental_hits=False,
            partition_strategy="div",
            name="sampled_logits_basic_num_true_%d" % num_true)
        got_logits, got_labels = sess.run([logits_tensor, labels_tensor])
        self.assertAllClose(exp_logits, got_logits, self._eps)
        self.assertAllClose(exp_labels, got_labels, self._eps)

  def testAccidentalHitRemoval(self):
    """With accidental hit removal, no subtract_log_q."""
    np.random.seed(0)
    num_classes = 5
    batch_size = 3
    sampled = [1, 0, 2, 3]
    with self.test_session():
      for num_true in range(1, 5):
        labels = np.random.randint(
            low=0, high=num_classes, size=batch_size * num_true)
        (weights, biases, hidden_acts, sampled_vals, _,
         _) = self._GenerateTestData(
             num_classes=num_classes,
             dim=10,
             batch_size=batch_size,
             num_true=num_true,
             labels=labels,
             sampled=sampled,
             subtract_log_q=False)
        logits_tensor, _ = _compute_sampled_logits(
            weights=constant_op.constant(weights),
            biases=constant_op.constant(biases),
            labels=constant_op.constant(
                labels, dtype=dtypes.int64, shape=(batch_size, num_true)),
            inputs=constant_op.constant(hidden_acts),
            num_sampled=len(sampled),
            num_classes=num_classes,
            num_true=num_true,
            sampled_values=sampled_vals,
            subtract_log_q=False,
            remove_accidental_hits=True,
            partition_strategy="div",
            name="sampled_logits_accidental_hit_removal_num_true_%d" % num_true)
        # Test that the exponentiated logits of accidental hits are near 0.
        # First we need to find the hits in this random test run:
        labels_reshape = labels.reshape((batch_size, num_true))
        got_logits = logits_tensor.eval()
        for row in xrange(batch_size):
          row_labels = labels_reshape[row, :]
          for col in xrange(len(sampled)):
            if sampled[col] in row_labels:
              # We need to add the num_true_test offset into logits_*
              self.assertNear(
                  np.exp(got_logits[row, col + num_true]), 0., self._eps)

  def testSubtractLogQ(self):
    """With subtract_log_q, no accidental hit removal."""
    np.random.seed(0)
    num_classes = 5
    batch_size = 3
    with self.test_session() as sess:
      for num_true in range(1, 5):
        labels = np.random.randint(
            low=0, high=num_classes, size=batch_size * num_true)
        (weights, biases, hidden_acts, sampled_vals, exp_logits,
         exp_labels) = self._GenerateTestData(
             num_classes=num_classes,
             dim=10,
             batch_size=batch_size,
             num_true=num_true,
             labels=labels,
             sampled=[1, 0, 2, 3],
             subtract_log_q=True)
        logits_tensor, labels_tensor = _compute_sampled_logits(
            weights=constant_op.constant(weights),
            biases=constant_op.constant(biases),
            labels=constant_op.constant(
                labels, dtype=dtypes.int64, shape=(batch_size, num_true)),
            inputs=constant_op.constant(hidden_acts),
            num_sampled=4,
            num_classes=num_classes,
            num_true=num_true,
            sampled_values=sampled_vals,
            subtract_log_q=True,
            remove_accidental_hits=False,
            partition_strategy="div",
            name="sampled_logits_subtract_log_q_num_true_%d" % num_true)
        got_logits, got_labels = sess.run([logits_tensor, labels_tensor])
        self.assertAllClose(exp_logits, got_logits, self._eps)
        self.assertAllClose(exp_labels, got_labels, self._eps)

  def testSharded(self):
    """With sharded weights and sharded biases."""
    np.random.seed(0)
    num_classes = 5
    batch_size = 3
    with self.test_session() as sess:
      for num_true in range(1, 5):
        labels = np.random.randint(
            low=0, high=num_classes, size=batch_size * num_true)
        (weights, biases, hidden_acts, sampled_vals, exp_logits,
         exp_labels) = self._GenerateTestData(
             num_classes=num_classes,
             dim=10,
             batch_size=batch_size,
             num_true=num_true,
             labels=labels,
             sampled=[1, 0, 2, 3],
             subtract_log_q=False)
        weight_shards, bias_shards = self._ShardTestEmbeddings(
            weights, biases, num_shards=3)
        logits_tensor, labels_tensor = _compute_sampled_logits(
            weights=[constant_op.constant(shard) for shard in weight_shards],
            biases=[constant_op.constant(shard) for shard in bias_shards],
            labels=constant_op.constant(
                labels, dtype=dtypes.int64, shape=(batch_size, num_true)),
            inputs=constant_op.constant(hidden_acts),
            num_sampled=4,
            num_classes=num_classes,
            num_true=num_true,
            sampled_values=sampled_vals,
            subtract_log_q=False,
            remove_accidental_hits=False,
            partition_strategy="div",
            name="sampled_logits_sharded_num_true_%d" % num_true)
        got_logits, got_labels = sess.run([logits_tensor, labels_tensor])
        self.assertAllClose(exp_logits, got_logits, self._eps)
        self.assertAllClose(exp_labels, got_labels, self._eps)

  def testNCELoss(self):
    # A simple test to verify the numerics.

    def _SigmoidCrossEntropyWithLogits(logits, targets):
      # logits, targets: float arrays of the same shape.
      assert logits.shape == targets.shape
      pred = 1. / (1. + np.exp(-logits))
      eps = 0.0001
      pred = np.minimum(np.maximum(pred, eps), 1 - eps)
      return -targets * np.log(pred) - (1. - targets) * np.log(1. - pred)

    np.random.seed(0)
    num_classes = 5
    batch_size = 3
    labels = [0, 1, 2]
    (weights, biases, hidden_acts, sampled_vals, exp_logits,
     exp_labels) = self._GenerateTestData(
         num_classes=num_classes,
         dim=10,
         batch_size=batch_size,
         num_true=1,
         labels=labels,
         sampled=[1, 0, 2, 3],
         subtract_log_q=True)
    exp_nce_loss = np.sum(
        _SigmoidCrossEntropyWithLogits(exp_logits, exp_labels), 1)

    with self.test_session():
      got_nce_loss = nn_impl.nce_loss(
          weights=constant_op.constant(weights),
          biases=constant_op.constant(biases),
          labels=constant_op.constant(labels, shape=(batch_size, 1)),
          inputs=constant_op.constant(hidden_acts),
          num_sampled=4,
          num_classes=num_classes,
          num_true=1,
          sampled_values=sampled_vals,
          partition_strategy="div")

      self.assertAllClose(exp_nce_loss, got_nce_loss.eval(), 1e-4)

      # Test with sharded weights and sharded biases.
      weight_shards, bias_shards = self._ShardTestEmbeddings(
          weights, biases, num_shards=3)
      got_nce_loss = nn_impl.nce_loss(
          weights=[constant_op.constant(shard) for shard in weight_shards],
          biases=[constant_op.constant(shard) for shard in bias_shards],
          labels=constant_op.constant(labels, shape=(batch_size, 1)),
          inputs=constant_op.constant(hidden_acts),
          num_sampled=4,
          num_classes=num_classes,
          num_true=1,
          sampled_values=sampled_vals,
          partition_strategy="div")

      self.assertAllClose(exp_nce_loss, got_nce_loss.eval(), 1e-4)

  def testSampledSoftmaxLoss(self):
    # A simple test to verify the numerics.

    def _SoftmaxCrossEntropyWithLogits(logits, targets):
      # logits, targets: float arrays of the same shape.
      assert logits.shape == targets.shape
      stable_exp_logits = np.exp(logits - np.amax(
          logits, axis=1, keepdims=True))
      pred = stable_exp_logits / np.sum(stable_exp_logits, 1, keepdims=True)
      return -np.sum(targets * np.log(pred + 1.0e-20), axis=1)

    np.random.seed(0)
    num_classes = 5
    batch_size = 3
    labels = [0, 1, 2]
    (weights, biases, hidden_acts, sampled_vals, exp_logits,
     exp_labels) = self._GenerateTestData(
         num_classes=num_classes,
         dim=10,
         batch_size=batch_size,
         num_true=1,
         labels=labels,
         sampled=[1, 0, 2, 3],
         subtract_log_q=True)
    exp_sampled_softmax_loss = _SoftmaxCrossEntropyWithLogits(
        exp_logits, exp_labels)

    with self.test_session():
      got_sampled_softmax_loss = nn_impl.sampled_softmax_loss(
          weights=constant_op.constant(weights),
          biases=constant_op.constant(biases),
          labels=constant_op.constant(labels, shape=(batch_size, 1)),
          inputs=constant_op.constant(hidden_acts),
          num_sampled=4,
          num_classes=num_classes,
          num_true=1,
          sampled_values=sampled_vals,
          remove_accidental_hits=False,
          partition_strategy="div")

      self.assertAllClose(exp_sampled_softmax_loss,
                          got_sampled_softmax_loss.eval(), 1e-4)

      # Test with sharded weights and sharded biases.
      weight_shards, bias_shards = self._ShardTestEmbeddings(
          weights, biases, num_shards=3)
      got_sampled_softmax_loss = nn_impl.sampled_softmax_loss(
          weights=[constant_op.constant(shard) for shard in weight_shards],
          biases=[constant_op.constant(shard) for shard in bias_shards],
          labels=constant_op.constant(labels, shape=(batch_size, 1)),
          inputs=constant_op.constant(hidden_acts),
          num_sampled=4,
          num_classes=num_classes,
          num_true=1,
          sampled_values=sampled_vals,
          remove_accidental_hits=False,
          partition_strategy="div")

      self.assertAllClose(exp_sampled_softmax_loss,
                          got_sampled_softmax_loss.eval(), 1e-4)


class CReluTest(test_lib.TestCase):

  def test(self):
    np.random.seed(1)  # Make it reproducible.
    x = np.random.randn(3, 4).astype(np.float32)
    y = np.concatenate([x * (x > 0), -x * (x < 0)], axis=1)
    with self.test_session():
      z = nn_ops.crelu(constant_op.constant(x)).eval()
      self.assertAllClose(y, z, 1e-4)


class ReluTest(test_lib.TestCase):

  def test(self):
    np.random.seed(1)  # Make it reproducible.
    x = np.random.randn(3, 4).astype(np.float32)
    y = np.maximum(x, 0.0)
    with self.test_session():
      z = nn_ops.relu(constant_op.constant(x)).eval()
      self.assertAllEqual(y, z)

  def testNaNs(self):
    # Test that relu(nan) = nan for various sizes.
    for i in range(18):
      x = np.zeros(i) + np.nan
      with self.test_session():
        z = nn_ops.relu(constant_op.constant(x)).eval()
        self.assertTrue(np.isnan(z).all())


class LeakyReluTest(test_lib.TestCase):

  def testRange(self):
    batch_size = 3
    height, width = 4, 4
    np.random.seed(1)  # Make it reproducible.
    inputs = np.random.uniform(
        size=(batch_size, height, width, 3)).astype(np.float32)
    inputs = constant_op.constant(inputs)

    outputs = nn_ops.leaky_relu(inputs)
    self.assertEquals(inputs.shape, outputs.shape)
    with self.test_session() as sess:
      inputs, outputs = sess.run([inputs, outputs])
    self.assertGreaterEqual(outputs.min(), 0.0)
    self.assertLessEqual(outputs.max(), 1.0)
    self.assertAllClose(inputs, outputs)

  def testValues(self):
    np_values = np.array([-1.0, 0.0, 0.5, 1.0, 2.0], dtype=np.float32)
    outputs = nn_ops.leaky_relu(constant_op.constant(np_values))
    with self.test_session() as sess:
      outputs = sess.run(outputs)
    self.assertAllClose(outputs, [-0.2, 0.0, 0.5, 1.0, 2.0])


class SwishTest(test_lib.TestCase):

  def testValues(self):
    np_values = np.array(
        [np.linspace(-10.0, 0.0, 100),
         np.linspace(0.0, 10.0, 100)],
        dtype=np.float32)
    tf_values = constant_op.constant(np_values)
    actual_tf_outputs = nn_impl.swish(tf_values)
    expected_tf_outputs = tf_values * math_ops.sigmoid(tf_values)
    with self.test_session() as sess:
      actual_outputs, expected_outputs = sess.run(
          [actual_tf_outputs, expected_tf_outputs])
    self.assertAllClose(actual_outputs, expected_outputs)

  def testGradients(self):
    shape = [5, 3, 4]
    sigma = 5
    input_values = np.random.randn(*shape) * sigma
    x_tf = constant_op.constant(input_values)
    y_tf = nn_impl.swish(x_tf)
    with self.test_session():
      err = gradient_checker.compute_gradient_error(x_tf, shape, y_tf, shape)
    self.assertLess(err, 1e-4)


class MomentsTest(test_lib.TestCase):

  def doOutputTest(self, input_shape, moments_axes, tol=1e-4,
                   check_gradients=False):
    for mu in [0.0, 1.0, 1e3]:
      for sigma in [1.0, 0.1]:
        for keep_dims in [True, False]:
          input_values = np.random.rand(*input_shape) * sigma + mu
          expected_mean = np.mean(
              input_values, axis=moments_axes, keepdims=keep_dims)
          expected_var = np.var(
              input_values, axis=moments_axes, keepdims=keep_dims)
          with ops.Graph().as_default() as g:
            with self.test_session(graph=g) as sess:
              inputs = constant_op.constant(
                  input_values, shape=input_shape, dtype=dtypes.float32)
              mean, variance = nn_impl.moments(
                  inputs, moments_axes, keep_dims=keep_dims)

              if check_gradients:
                err = gradient_checker.compute_gradient_error(
                    inputs, input_shape, mean, mean.shape.as_list())
                self.assertLess(err, 1e-3)
                err = gradient_checker.compute_gradient_error(
                    inputs, input_shape, variance, variance.shape.as_list())
                self.assertLess(err, 1e-3)

              # Evaluate.
              [mean, variance] = sess.run([mean, variance])
              # Make sure that there are no NaNs
              self.assertFalse(np.isnan(mean).any())
              self.assertFalse(np.isnan(variance).any())
              self.assertAllClose(mean, expected_mean, rtol=tol, atol=tol)
              self.assertAllClose(variance, expected_var, rtol=tol, atol=tol)

  def testOutputAndGradient2DInput0(self):
    self.doOutputTest((10, 10), (0,), check_gradients=True)

  def testOutputAndGradient2DInput01(self):
    self.doOutputTest((10, 10), (0, 1), check_gradients=True)

  def testOutput2DInput0(self):
    self.doOutputTest((10, 300), (0,))

  def testOutput2DInput1(self):
    self.doOutputTest((10, 300), (1,))

  def testOutput2DInput01(self):
    self.doOutputTest((10, 300), (0, 1))

  def testOutput4DInput0(self):
    self.doOutputTest((10, 10, 10, 30), (0,))

  def testOutput4DInput1(self):
    self.doOutputTest((10, 10, 10, 30), (1,))

  def testOutput4DInput3(self):
    self.doOutputTest((10, 10, 10, 30), (3,))

  def testOutput4DInput012(self):
    self.doOutputTest((10, 10, 10, 30), (0, 1, 2))

  def testOutput4DInput123(self):
    self.doOutputTest((10, 10, 10, 30), (1, 2, 3))


if __name__ == "__main__":
  test_lib.main()
