# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from tensorflow.contrib import distributions
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test

ds = distributions


class DirichletMultinomialTest(test.TestCase):

  def setUp(self):
    self._rng = np.random.RandomState(42)

  def testSimpleShapes(self):
    with self.test_session():
      alpha = np.random.rand(3)
      dist = ds.DirichletMultinomial(1., alpha)
      self.assertEqual(3, dist.event_shape_tensor().eval())
      self.assertAllEqual([], dist.batch_shape_tensor().eval())
      self.assertEqual(tensor_shape.TensorShape([3]), dist.event_shape)
      self.assertEqual(tensor_shape.TensorShape([]), dist.batch_shape)

  def testComplexShapes(self):
    with self.test_session():
      alpha = np.random.rand(3, 2, 2)
      n = [[3., 2], [4, 5], [6, 7]]
      dist = ds.DirichletMultinomial(n, alpha)
      self.assertEqual(2, dist.event_shape_tensor().eval())
      self.assertAllEqual([3, 2], dist.batch_shape_tensor().eval())
      self.assertEqual(tensor_shape.TensorShape([2]), dist.event_shape)
      self.assertEqual(tensor_shape.TensorShape([3, 2]), dist.batch_shape)

  def testNproperty(self):
    alpha = [[1., 2, 3]]
    n = [[5.]]
    with self.test_session():
      dist = ds.DirichletMultinomial(n, alpha)
      self.assertEqual([1, 1], dist.total_count.get_shape())
      self.assertAllClose(n, dist.total_count.eval())

  def testAlphaProperty(self):
    alpha = [[1., 2, 3]]
    with self.test_session():
      dist = ds.DirichletMultinomial(1, alpha)
      self.assertEqual([1, 3], dist.concentration.get_shape())
      self.assertAllClose(alpha, dist.concentration.eval())

  def testPmfNandCountsAgree(self):
    alpha = [[1., 2, 3]]
    n = [[5.]]
    with self.test_session():
      dist = ds.DirichletMultinomial(n, alpha, validate_args=True)
      dist.prob([2., 3, 0]).eval()
      dist.prob([3., 0, 2]).eval()
      with self.assertRaisesOpError("counts must be non-negative"):
        dist.prob([-1., 4, 2]).eval()
      with self.assertRaisesOpError(
          "counts last-dimension must sum to `self.total_count`"):
        dist.prob([3., 3, 0]).eval()

  def testPmfNonIntegerCounts(self):
    alpha = [[1., 2, 3]]
    n = [[5.]]
    with self.test_session():
      dist = ds.DirichletMultinomial(n, alpha, validate_args=True)
      dist.prob([2., 3, 0]).eval()
      dist.prob([3., 0, 2]).eval()
      dist.prob([3.0, 0, 2.0]).eval()
      # Both equality and integer checking fail.
      with self.assertRaisesOpError(
          "counts cannot contain fractional components"):
        dist.prob([1.0, 2.5, 1.5]).eval()
      dist = ds.DirichletMultinomial(n, alpha, validate_args=False)
      dist.prob([1., 2., 3.]).eval()
      # Non-integer arguments work.
      dist.prob([1.0, 2.5, 1.5]).eval()

  def testPmfBothZeroBatches(self):
    # The probabilities of one vote falling into class k is the mean for class
    # k.
    with self.test_session():
      # Both zero-batches.  No broadcast
      alpha = [1., 2]
      counts = [1., 0]
      dist = ds.DirichletMultinomial(1., alpha)
      pmf = dist.prob(counts)
      self.assertAllClose(1 / 3., pmf.eval())
      self.assertEqual((), pmf.get_shape())

  def testPmfBothZeroBatchesNontrivialN(self):
    # The probabilities of one vote falling into class k is the mean for class
    # k.
    with self.test_session():
      # Both zero-batches.  No broadcast
      alpha = [1., 2]
      counts = [3., 2]
      dist = ds.DirichletMultinomial(5., alpha)
      pmf = dist.prob(counts)
      self.assertAllClose(1 / 7., pmf.eval())
      self.assertEqual((), pmf.get_shape())

  def testPmfBothZeroBatchesMultidimensionalN(self):
    # The probabilities of one vote falling into class k is the mean for class
    # k.
    with self.test_session():
      alpha = [1., 2]
      counts = [3., 2]
      n = np.full([4, 3], 5., dtype=np.float32)
      dist = ds.DirichletMultinomial(n, alpha)
      pmf = dist.prob(counts)
      self.assertAllClose([[1 / 7., 1 / 7., 1 / 7.]] * 4, pmf.eval())
      self.assertEqual((4, 3), pmf.get_shape())

  def testPmfAlphaStretchedInBroadcastWhenSameRank(self):
    # The probabilities of one vote falling into class k is the mean for class
    # k.
    with self.test_session():
      alpha = [[1., 2]]
      counts = [[1., 0], [0., 1]]
      dist = ds.DirichletMultinomial([1.], alpha)
      pmf = dist.prob(counts)
      self.assertAllClose([1 / 3., 2 / 3.], pmf.eval())
      self.assertAllEqual([2], pmf.get_shape())

  def testPmfAlphaStretchedInBroadcastWhenLowerRank(self):
    # The probabilities of one vote falling into class k is the mean for class
    # k.
    with self.test_session():
      alpha = [1., 2]
      counts = [[1., 0], [0., 1]]
      pmf = ds.DirichletMultinomial(1., alpha).prob(counts)
      self.assertAllClose([1 / 3., 2 / 3.], pmf.eval())
      self.assertAllEqual([2], pmf.get_shape())

  def testPmfCountsStretchedInBroadcastWhenSameRank(self):
    # The probabilities of one vote falling into class k is the mean for class
    # k.
    with self.test_session():
      alpha = [[1., 2], [2., 3]]
      counts = [[1., 0]]
      pmf = ds.DirichletMultinomial([1., 1.], alpha).prob(counts)
      self.assertAllClose([1 / 3., 2 / 5.], pmf.eval())
      self.assertAllEqual([2], pmf.get_shape())

  def testPmfCountsStretchedInBroadcastWhenLowerRank(self):
    # The probabilities of one vote falling into class k is the mean for class
    # k.
    with self.test_session():
      alpha = [[1., 2], [2., 3]]
      counts = [1., 0]
      pmf = ds.DirichletMultinomial(1., alpha).prob(counts)
      self.assertAllClose([1 / 3., 2 / 5.], pmf.eval())
      self.assertAllEqual([2], pmf.get_shape())

  def testPmfForOneVoteIsTheMeanWithOneRecordInput(self):
    # The probabilities of one vote falling into class k is the mean for class
    # k.
    alpha = [1., 2, 3]
    with self.test_session():
      for class_num in range(3):
        counts = np.zeros([3], dtype=np.float32)
        counts[class_num] = 1
        dist = ds.DirichletMultinomial(1., alpha)
        mean = dist.mean().eval()
        pmf = dist.prob(counts).eval()

        self.assertAllClose(mean[class_num], pmf)
        self.assertAllEqual([3], mean.shape)
        self.assertAllEqual([], pmf.shape)

  def testMeanDoubleTwoVotes(self):
    # The probabilities of two votes falling into class k for
    # DirichletMultinomial(2, alpha) is twice as much as the probability of one
    # vote falling into class k for DirichletMultinomial(1, alpha)
    alpha = [1., 2, 3]
    with self.test_session():
      for class_num in range(3):
        counts_one = np.zeros([3], dtype=np.float32)
        counts_one[class_num] = 1.
        counts_two = np.zeros([3], dtype=np.float32)
        counts_two[class_num] = 2

        dist1 = ds.DirichletMultinomial(1., alpha)
        dist2 = ds.DirichletMultinomial(2., alpha)

        mean1 = dist1.mean().eval()
        mean2 = dist2.mean().eval()

        self.assertAllClose(mean2[class_num], 2 * mean1[class_num])
        self.assertAllEqual([3], mean1.shape)

  def testCovarianceFromSampling(self):
    # We will test mean, cov, var, stddev on a DirichletMultinomial constructed
    # via broadcast between alpha, n.
    alpha = np.array([[1., 2, 3],
                      [2.5, 4, 0.01]], dtype=np.float32)
    # Ideally we'd be able to test broadcasting but, the multinomial sampler
    # doesn't support different total counts.
    n = np.float32(5)
    with self.test_session() as sess:
      # batch_shape=[2], event_shape=[3]
      dist = ds.DirichletMultinomial(n, alpha)
      x = dist.sample(int(250e3), seed=1)
      sample_mean = math_ops.reduce_mean(x, 0)
      x_centered = x - sample_mean[array_ops.newaxis, ...]
      sample_cov = math_ops.reduce_mean(math_ops.matmul(
          x_centered[..., array_ops.newaxis],
          x_centered[..., array_ops.newaxis, :]), 0)
      sample_var = array_ops.matrix_diag_part(sample_cov)
      sample_stddev = math_ops.sqrt(sample_var)
      [
          sample_mean_,
          sample_cov_,
          sample_var_,
          sample_stddev_,
          analytic_mean,
          analytic_cov,
          analytic_var,
          analytic_stddev,
      ] = sess.run([
          sample_mean,
          sample_cov,
          sample_var,
          sample_stddev,
          dist.mean(),
          dist.covariance(),
          dist.variance(),
          dist.stddev(),
      ])
      self.assertAllClose(sample_mean_, analytic_mean, atol=0., rtol=0.04)
      self.assertAllClose(sample_cov_, analytic_cov, atol=0., rtol=0.05)
      self.assertAllClose(sample_var_, analytic_var, atol=0., rtol=0.03)
      self.assertAllClose(sample_stddev_, analytic_stddev, atol=0., rtol=0.02)

  def testCovariance(self):
    # Shape [2]
    alpha = [1., 2]
    ns = [2., 3., 4., 5.]
    alpha_0 = np.sum(alpha)

    # Diagonal entries are of the form:
    # Var(X_i) = n * alpha_i / alpha_sum * (1 - alpha_i / alpha_sum) *
    # (alpha_sum + n) / (alpha_sum + 1)
    variance_entry = lambda a, a_sum: a / a_sum * (1 - a / a_sum)
    # Off diagonal entries are of the form:
    # Cov(X_i, X_j) = -n * alpha_i * alpha_j / (alpha_sum ** 2) *
    # (alpha_sum + n) / (alpha_sum + 1)
    covariance_entry = lambda a, b, a_sum: -a * b / a_sum**2
    # Shape [2, 2].
    shared_matrix = np.array([[
        variance_entry(alpha[0], alpha_0),
        covariance_entry(alpha[0], alpha[1], alpha_0)
    ], [
        covariance_entry(alpha[1], alpha[0], alpha_0),
        variance_entry(alpha[1], alpha_0)
    ]])

    with self.test_session():
      for n in ns:
        # n is shape [] and alpha is shape [2].
        dist = ds.DirichletMultinomial(n, alpha)
        covariance = dist.covariance()
        expected_covariance = n * (n + alpha_0) / (1 + alpha_0) * shared_matrix

        self.assertEqual([2, 2], covariance.get_shape())
        self.assertAllClose(expected_covariance, covariance.eval())

  def testCovarianceNAlphaBroadcast(self):
    alpha_v = [1., 2, 3]
    alpha_0 = 6.

    # Shape [4, 3]
    alpha = np.array(4 * [alpha_v], dtype=np.float32)
    # Shape [4, 1]
    ns = np.array([[2.], [3.], [4.], [5.]], dtype=np.float32)

    variance_entry = lambda a, a_sum: a / a_sum * (1 - a / a_sum)
    covariance_entry = lambda a, b, a_sum: -a * b / a_sum**2
    # Shape [4, 3, 3]
    shared_matrix = np.array(
        4 * [[[
            variance_entry(alpha_v[0], alpha_0),
            covariance_entry(alpha_v[0], alpha_v[1], alpha_0),
            covariance_entry(alpha_v[0], alpha_v[2], alpha_0)
        ], [
            covariance_entry(alpha_v[1], alpha_v[0], alpha_0),
            variance_entry(alpha_v[1], alpha_0),
            covariance_entry(alpha_v[1], alpha_v[2], alpha_0)
        ], [
            covariance_entry(alpha_v[2], alpha_v[0], alpha_0),
            covariance_entry(alpha_v[2], alpha_v[1], alpha_0),
            variance_entry(alpha_v[2], alpha_0)
        ]]],
        dtype=np.float32)

    with self.test_session():
      # ns is shape [4, 1], and alpha is shape [4, 3].
      dist = ds.DirichletMultinomial(ns, alpha)
      covariance = dist.covariance()
      expected_covariance = shared_matrix * (
          ns * (ns + alpha_0) / (1 + alpha_0))[..., array_ops.newaxis]

      self.assertEqual([4, 3, 3], covariance.get_shape())
      self.assertAllClose(expected_covariance, covariance.eval())

  def testCovarianceMultidimensional(self):
    alpha = np.random.rand(3, 5, 4).astype(np.float32)
    alpha2 = np.random.rand(6, 3, 3).astype(np.float32)

    ns = np.random.randint(low=1, high=11, size=[3, 5, 1]).astype(np.float32)
    ns2 = np.random.randint(low=1, high=11, size=[6, 1, 1]).astype(np.float32)

    with self.test_session():
      dist = ds.DirichletMultinomial(ns, alpha)
      dist2 = ds.DirichletMultinomial(ns2, alpha2)

      covariance = dist.covariance()
      covariance2 = dist2.covariance()
      self.assertEqual([3, 5, 4, 4], covariance.get_shape())
      self.assertEqual([6, 3, 3, 3], covariance2.get_shape())

  def testZeroCountsResultsInPmfEqualToOne(self):
    # There is only one way for zero items to be selected, and this happens with
    # probability 1.
    alpha = [5, 0.5]
    counts = [0., 0]
    with self.test_session():
      dist = ds.DirichletMultinomial(0., alpha)
      pmf = dist.prob(counts)
      self.assertAllClose(1.0, pmf.eval())
      self.assertEqual((), pmf.get_shape())

  def testLargeTauGivesPreciseProbabilities(self):
    # If tau is large, we are doing coin flips with probability mu.
    mu = np.array([0.1, 0.1, 0.8], dtype=np.float32)
    tau = np.array([100.], dtype=np.float32)
    alpha = tau * mu

    # One (three sided) coin flip.  Prob[coin 3] = 0.8.
    # Note that since it was one flip, value of tau didn't matter.
    counts = [0., 0, 1]
    with self.test_session():
      dist = ds.DirichletMultinomial(1., alpha)
      pmf = dist.prob(counts)
      self.assertAllClose(0.8, pmf.eval(), atol=1e-4)
      self.assertEqual((), pmf.get_shape())

    # Two (three sided) coin flips.  Prob[coin 3] = 0.8.
    counts = [0., 0, 2]
    with self.test_session():
      dist = ds.DirichletMultinomial(2., alpha)
      pmf = dist.prob(counts)
      self.assertAllClose(0.8**2, pmf.eval(), atol=1e-2)
      self.assertEqual((), pmf.get_shape())

    # Three (three sided) coin flips.
    counts = [1., 0, 2]
    with self.test_session():
      dist = ds.DirichletMultinomial(3., alpha)
      pmf = dist.prob(counts)
      self.assertAllClose(3 * 0.1 * 0.8 * 0.8, pmf.eval(), atol=1e-2)
      self.assertEqual((), pmf.get_shape())

  def testSmallTauPrefersCorrelatedResults(self):
    # If tau is small, then correlation between draws is large, so draws that
    # are both of the same class are more likely.
    mu = np.array([0.5, 0.5], dtype=np.float32)
    tau = np.array([0.1], dtype=np.float32)
    alpha = tau * mu

    # If there is only one draw, it is still a coin flip, even with small tau.
    counts = [1., 0]
    with self.test_session():
      dist = ds.DirichletMultinomial(1., alpha)
      pmf = dist.prob(counts)
      self.assertAllClose(0.5, pmf.eval())
      self.assertEqual((), pmf.get_shape())

    # If there are two draws, it is much more likely that they are the same.
    counts_same = [2., 0]
    counts_different = [1, 1.]
    with self.test_session():
      dist = ds.DirichletMultinomial(2., alpha)
      pmf_same = dist.prob(counts_same)
      pmf_different = dist.prob(counts_different)
      self.assertLess(5 * pmf_different.eval(), pmf_same.eval())
      self.assertEqual((), pmf_same.get_shape())

  def testNonStrictTurnsOffAllChecks(self):
    # Make totally invalid input.
    with self.test_session():
      alpha = [[-1., 2]]  # alpha should be positive.
      counts = [[1., 0], [0., -1]]  # counts should be non-negative.
      n = [-5.3]  # n should be a non negative integer equal to counts.sum.
      dist = ds.DirichletMultinomial(n, alpha, validate_args=False)
      dist.prob(counts).eval()  # Should not raise.

  def testSampleUnbiasedNonScalarBatch(self):
    with self.test_session() as sess:
      dist = ds.DirichletMultinomial(
          total_count=5.,
          concentration=1. + 2. * self._rng.rand(4, 3, 2).astype(np.float32))
      n = int(3e3)
      x = dist.sample(n, seed=0)
      sample_mean = math_ops.reduce_mean(x, 0)
      # Cyclically rotate event dims left.
      x_centered = array_ops.transpose(x - sample_mean, [1, 2, 3, 0])
      sample_covariance = math_ops.matmul(
          x_centered, x_centered, adjoint_b=True) / n
      [
          sample_mean_,
          sample_covariance_,
          actual_mean_,
          actual_covariance_,
      ] = sess.run([
          sample_mean,
          sample_covariance,
          dist.mean(),
          dist.covariance(),
      ])
      self.assertAllEqual([4, 3, 2], sample_mean.get_shape())
      self.assertAllClose(actual_mean_, sample_mean_, atol=0., rtol=0.15)
      self.assertAllEqual([4, 3, 2, 2], sample_covariance.get_shape())
      self.assertAllClose(
          actual_covariance_, sample_covariance_, atol=0., rtol=0.20)

  def testSampleUnbiasedScalarBatch(self):
    with self.test_session() as sess:
      dist = ds.DirichletMultinomial(
          total_count=5.,
          concentration=1. + 2. * self._rng.rand(4).astype(np.float32))
      n = int(5e3)
      x = dist.sample(n, seed=0)
      sample_mean = math_ops.reduce_mean(x, 0)
      x_centered = x - sample_mean  # Already transposed to [n, 2].
      sample_covariance = math_ops.matmul(
          x_centered, x_centered, adjoint_a=True) / n
      [
          sample_mean_,
          sample_covariance_,
          actual_mean_,
          actual_covariance_,
      ] = sess.run([
          sample_mean,
          sample_covariance,
          dist.mean(),
          dist.covariance(),
      ])
      self.assertAllEqual([4], sample_mean.get_shape())
      self.assertAllClose(actual_mean_, sample_mean_, atol=0., rtol=0.05)
      self.assertAllEqual([4, 4], sample_covariance.get_shape())
      self.assertAllClose(
          actual_covariance_, sample_covariance_, atol=0., rtol=0.15)


if __name__ == "__main__":
  test.main()
