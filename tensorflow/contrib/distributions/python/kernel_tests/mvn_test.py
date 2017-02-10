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
"""Tests for MultivariateNormal."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy import stats
from tensorflow.contrib import distributions
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.platform import test


ds = distributions


class MultivariateNormalDiagTest(test.TestCase):
  """Well tested because this is a simple override of the base class."""

  def setUp(self):
    self._rng = np.random.RandomState(42)

  def testScalarParams(self):
    mu = -1.
    diag = -5.
    with self.test_session():
      # TODO(b/35244539): Choose better exception, once LinOp throws it.
      with self.assertRaises(IndexError):
        ds.MultivariateNormalDiag(mu, diag)

  def testVectorParams(self):
    mu = [-1.]
    diag = [-5.]
    with self.test_session():
      dist = ds.MultivariateNormalDiag(mu, diag, validate_args=True)
      self.assertAllEqual([3, 1], dist.sample(3).get_shape())

  def testMean(self):
    mu = [-1., 1]
    diag = [1., -5]
    with self.test_session():
      dist = ds.MultivariateNormalDiag(mu, diag, validate_args=True)
      self.assertAllEqual(mu, dist.mean().eval())

  def testEntropy(self):
    mu = [-1., 1]
    diag = [-1., 5]
    diag_mat = np.diag(diag)
    scipy_mvn = stats.multivariate_normal(mean=mu, cov=diag_mat**2)
    with self.test_session():
      dist = ds.MultivariateNormalDiag(mu, diag, validate_args=True)
      self.assertAllClose(scipy_mvn.entropy(), dist.entropy().eval(), atol=1e-4)

  def testSample(self):
    mu = [-1., 1]
    diag = [1., -2]
    with self.test_session():
      dist = ds.MultivariateNormalDiag(mu, diag, validate_args=True)
      samps = dist.sample(int(1e3), seed=0).eval()
      cov_mat = array_ops.matrix_diag(diag).eval()**2

      self.assertAllClose(mu, samps.mean(axis=0),
                          atol=0., rtol=0.05)
      self.assertAllClose(cov_mat, np.cov(samps.T),
                          atol=0.05, rtol=0.05)

  def testCovariance(self):
    with self.test_session():
      mvn = ds.MultivariateNormalDiag(
          loc=array_ops.zeros([2, 3], dtype=dtypes.float32))
      self.assertAllClose(
          np.diag(np.ones([3], dtype=np.float32)),
          mvn.covariance().eval())

      mvn = ds.MultivariateNormalDiag(
          loc=array_ops.zeros([3], dtype=dtypes.float32),
          scale_identity_multiplier=[3., 2.])
      self.assertAllEqual([2], mvn.batch_shape)
      self.assertAllEqual([3], mvn.event_shape)
      self.assertAllClose(
          np.array([[[3., 0, 0],
                     [0, 3, 0],
                     [0, 0, 3]],
                    [[2, 0, 0],
                     [0, 2, 0],
                     [0, 0, 2]]])**2.,
          mvn.covariance().eval())

      mvn = ds.MultivariateNormalDiag(
          loc=array_ops.zeros([3], dtype=dtypes.float32),
          scale_diag=[[3., 2, 1], [4, 5, 6]])
      self.assertAllEqual([2], mvn.batch_shape)
      self.assertAllEqual([3], mvn.event_shape)
      self.assertAllClose(
          np.array([[[3., 0, 0],
                     [0, 2, 0],
                     [0, 0, 1]],
                    [[4, 0, 0],
                     [0, 5, 0],
                     [0, 0, 6]]])**2.,
          mvn.covariance().eval())

  def testVariance(self):
    with self.test_session():
      mvn = ds.MultivariateNormalDiag(
          loc=array_ops.zeros([2, 3], dtype=dtypes.float32))
      self.assertAllClose(
          np.ones([3], dtype=np.float32),
          mvn.variance().eval())

      mvn = ds.MultivariateNormalDiag(
          loc=array_ops.zeros([3], dtype=dtypes.float32),
          scale_identity_multiplier=[3., 2.])
      self.assertAllClose(
          np.array([[3., 3, 3],
                    [2, 2, 2]])**2.,
          mvn.variance().eval())

      mvn = ds.MultivariateNormalDiag(
          loc=array_ops.zeros([3], dtype=dtypes.float32),
          scale_diag=[[3., 2, 1],
                      [4, 5, 6]])
      self.assertAllClose(
          np.array([[3., 2, 1],
                    [4, 5, 6]])**2.,
          mvn.variance().eval())

  def testStddev(self):
    with self.test_session():
      mvn = ds.MultivariateNormalDiag(
          loc=array_ops.zeros([2, 3], dtype=dtypes.float32))
      self.assertAllClose(
          np.ones([3], dtype=np.float32),
          mvn.stddev().eval())

      mvn = ds.MultivariateNormalDiag(
          loc=array_ops.zeros([3], dtype=dtypes.float32),
          scale_identity_multiplier=[3., 2.])
      self.assertAllClose(
          np.array([[3., 3, 3],
                    [2, 2, 2]]),
          mvn.stddev().eval())

      mvn = ds.MultivariateNormalDiag(
          loc=array_ops.zeros([3], dtype=dtypes.float32),
          scale_diag=[[3., 2, 1], [4, 5, 6]])
      self.assertAllClose(
          np.array([[3., 2, 1],
                    [4, 5, 6]]),
          mvn.stddev().eval())

  def testMultivariateNormalDiagWithSoftplusScale(self):
    mu = [-1.0, 1.0]
    diag = [-1.0, -2.0]
    with self.test_session():
      dist = ds.MultivariateNormalDiagWithSoftplusScale(
          mu, diag, validate_args=True)
      samps = dist.sample(1000, seed=0).eval()
      cov_mat = array_ops.matrix_diag(nn_ops.softplus(diag)).eval()**2

      self.assertAllClose(mu, samps.mean(axis=0), atol=0.1)
      self.assertAllClose(cov_mat, np.cov(samps.T), atol=0.1)


class MultivariateNormalDiagPlusLowRankTest(test.TestCase):
  """Well tested because this is a simple override of the base class."""

  def setUp(self):
    self._rng = np.random.RandomState(42)

  def testDiagBroadcastBothBatchAndEvent(self):
    # batch_shape: [3], event_shape: [2]
    diag = np.array([[1., 2], [3, 4], [5, 6]])
    # batch_shape: [1], event_shape: []
    identity_multiplier = np.array([5.])
    with self.test_session():
      dist = ds.MultivariateNormalDiagPlusLowRank(
          scale_diag=diag,
          scale_identity_multiplier=identity_multiplier,
          validate_args=True)
      self.assertAllClose(
          np.array([[[1. + 5, 0],
                     [0, 2 + 5]],
                    [[3 + 5, 0],
                     [0, 4 + 5]],
                    [[5 + 5, 0],
                     [0, 6 + 5]]]),
          dist.scale.to_dense().eval())

  def testDiagBroadcastBothBatchAndEvent2(self):
    # This test differs from `testDiagBroadcastBothBatchAndEvent` in that it
    # broadcasts batch_shape's from both the `scale_diag` and
    # `scale_identity_multiplier` args.
    # batch_shape: [3], event_shape: [2]
    diag = np.array([[1., 2], [3, 4], [5, 6]])
    # batch_shape: [3, 1], event_shape: []
    identity_multiplier = np.array([[5.], [4], [3]])
    with self.test_session():
      dist = ds.MultivariateNormalDiagPlusLowRank(
          scale_diag=diag,
          scale_identity_multiplier=identity_multiplier,
          validate_args=True)
      self.assertAllEqual(
          [3, 3, 2, 2],
          dist.scale.to_dense().get_shape())

  def testDiagBroadcastOnlyEvent(self):
    # batch_shape: [3], event_shape: [2]
    diag = np.array([[1., 2], [3, 4], [5, 6]])
    # batch_shape: [3], event_shape: []
    identity_multiplier = np.array([5., 4, 3])
    with self.test_session():
      dist = ds.MultivariateNormalDiagPlusLowRank(
          scale_diag=diag,
          scale_identity_multiplier=identity_multiplier,
          validate_args=True)
      self.assertAllClose(
          np.array([[[1. + 5, 0],
                     [0, 2 + 5]],
                    [[3 + 4, 0],
                     [0, 4 + 4]],
                    [[5 + 3, 0],
                     [0, 6 + 3]]]),   # shape: [3, 2, 2]
          dist.scale.to_dense().eval())

  def testDiagBroadcastMultiplierAndLoc(self):
    # batch_shape: [], event_shape: [3]
    loc = np.array([1., 0, -1])
    # batch_shape: [3], event_shape: []
    identity_multiplier = np.array([5., 4, 3])
    with self.test_session():
      dist = ds.MultivariateNormalDiagPlusLowRank(
          loc=loc,
          scale_identity_multiplier=identity_multiplier,
          validate_args=True)
      self.assertAllClose(
          np.array([[[5, 0, 0],
                     [0, 5, 0],
                     [0, 0, 5]],
                    [[4, 0, 0],
                     [0, 4, 0],
                     [0, 0, 4]],
                    [[3, 0, 0],
                     [0, 3, 0],
                     [0, 0, 3]]]),
          dist.scale.to_dense().eval())

  def testMean(self):
    mu = [-1.0, 1.0]
    diag_large = [1.0, 5.0]
    v = [[2.0], [3.0]]
    diag_small = [3.0]
    with self.test_session():
      dist = ds.MultivariateNormalDiagPlusLowRank(
          loc=mu,
          scale_diag=diag_large,
          scale_perturb_factor=v,
          scale_perturb_diag=diag_small,
          validate_args=True)
      self.assertAllEqual(mu, dist.mean().eval())

  def testSample(self):
    # TODO(jvdillon): This test should be the basis of a new test fixture which
    # is applied to every distribution. When we make this fixture, we'll also
    # separate the analytical- and sample-based tests as well as for each
    # function tested. For now, we group things so we can recycle one batch of
    # samples (thus saving resources).

    mu = np.array([-1., 1, 0.5], dtype=np.float32)
    diag_large = np.array([1., 0.5, 0.75], dtype=np.float32)
    diag_small = np.array([-1.1, 1.2], dtype=np.float32)
    v = np.array([[0.7, 0.8],
                  [0.9, 1],
                  [0.5, 0.6]], dtype=np.float32)  # shape: [k, r] = [3, 2]

    true_mean = mu
    true_scale = np.diag(diag_large) + np.matmul(np.matmul(
        v, np.diag(diag_small)), v.T)
    true_covariance = np.matmul(true_scale, true_scale.T)
    true_variance = np.diag(true_covariance)
    true_stddev = np.sqrt(true_variance)
    true_det_covariance = np.linalg.det(true_covariance)
    true_log_det_covariance = np.log(true_det_covariance)

    with self.test_session() as sess:
      dist = ds.MultivariateNormalDiagPlusLowRank(
          loc=mu,
          scale_diag=diag_large,
          scale_perturb_factor=v,
          scale_perturb_diag=diag_small,
          validate_args=True)

      # The following distributions will test the KL divergence calculation.
      mvn_identity = ds.MultivariateNormalDiag(
          loc=np.array([1., 2, 0.25], dtype=np.float32),
          validate_args=True)
      mvn_scaled = ds.MultivariateNormalDiag(
          loc=mvn_identity.loc,
          scale_identity_multiplier=2.2,
          validate_args=True)
      mvn_diag = ds.MultivariateNormalDiag(
          loc=mvn_identity.loc,
          scale_diag=np.array([0.5, 1.5, 1.], dtype=np.float32),
          validate_args=True)
      mvn_chol = ds.MultivariateNormalTriL(
          loc=np.array([1., 2, -1], dtype=np.float32),
          scale_tril=np.array([[6., 0, 0],
                               [2, 5, 0],
                               [1, 3, 4]], dtype=np.float32) / 10.,
          validate_args=True)

      n = int(30e3)
      samps = dist.sample(n, seed=0)
      sample_mean = math_ops.reduce_mean(samps, 0)
      x = samps - sample_mean
      sample_covariance = math_ops.matmul(x, x, transpose_a=True) / n

      sample_kl_identity = math_ops.reduce_mean(
          dist.log_prob(samps) - mvn_identity.log_prob(samps), 0)
      analytical_kl_identity = ds.kl(dist, mvn_identity)

      sample_kl_scaled = math_ops.reduce_mean(
          dist.log_prob(samps) - mvn_scaled.log_prob(samps), 0)
      analytical_kl_scaled = ds.kl(dist, mvn_scaled)

      sample_kl_diag = math_ops.reduce_mean(
          dist.log_prob(samps) - mvn_diag.log_prob(samps), 0)
      analytical_kl_diag = ds.kl(dist, mvn_diag)

      sample_kl_chol = math_ops.reduce_mean(
          dist.log_prob(samps) - mvn_chol.log_prob(samps), 0)
      analytical_kl_chol = ds.kl(dist, mvn_chol)

      scale = dist.scale.to_dense()

      [
          sample_mean_,
          analytical_mean_,
          sample_covariance_,
          analytical_covariance_,
          analytical_variance_,
          analytical_stddev_,
          analytical_log_det_covariance_,
          analytical_det_covariance_,
          sample_kl_identity_, analytical_kl_identity_,
          sample_kl_scaled_, analytical_kl_scaled_,
          sample_kl_diag_, analytical_kl_diag_,
          sample_kl_chol_, analytical_kl_chol_,
          scale_,
      ] = sess.run([
          sample_mean,
          dist.mean(),
          sample_covariance,
          dist.covariance(),
          dist.variance(),
          dist.stddev(),
          dist.log_det_covariance(),
          dist.det_covariance(),
          sample_kl_identity, analytical_kl_identity,
          sample_kl_scaled, analytical_kl_scaled,
          sample_kl_diag, analytical_kl_diag,
          sample_kl_chol, analytical_kl_chol,
          scale,
      ])

      sample_variance_ = np.diag(sample_covariance_)
      sample_stddev_ = np.sqrt(sample_variance_)
      sample_det_covariance_ = np.linalg.det(sample_covariance_)
      sample_log_det_covariance_ = np.log(sample_det_covariance_)

      print("true_mean:\n{}  ".format(true_mean))
      print("sample_mean:\n{}".format(sample_mean_))
      print("analytical_mean:\n{}".format(analytical_mean_))

      print("true_covariance:\n{}".format(true_covariance))
      print("sample_covariance:\n{}".format(sample_covariance_))
      print("analytical_covariance:\n{}".format(
          analytical_covariance_))

      print("true_variance:\n{}".format(true_variance))
      print("sample_variance:\n{}".format(sample_variance_))
      print("analytical_variance:\n{}".format(analytical_variance_))

      print("true_stddev:\n{}".format(true_stddev))
      print("sample_stddev:\n{}".format(sample_stddev_))
      print("analytical_stddev:\n{}".format(analytical_stddev_))

      print("true_log_det_covariance:\n{}".format(
          true_log_det_covariance))
      print("sample_log_det_covariance:\n{}".format(
          sample_log_det_covariance_))
      print("analytical_log_det_covariance:\n{}".format(
          analytical_log_det_covariance_))

      print("true_det_covariance:\n{}".format(
          true_det_covariance))
      print("sample_det_covariance:\n{}".format(
          sample_det_covariance_))
      print("analytical_det_covariance:\n{}".format(
          analytical_det_covariance_))

      print("true_scale:\n{}".format(true_scale))
      print("scale:\n{}".format(scale_))

      print("kl_identity:  analytical:{}  sample:{}".format(
          analytical_kl_identity_, sample_kl_identity_))

      print("kl_scaled:    analytical:{}  sample:{}".format(
          analytical_kl_scaled_, sample_kl_scaled_))

      print("kl_diag:      analytical:{}  sample:{}".format(
          analytical_kl_diag_, sample_kl_diag_))

      print("kl_chol:      analytical:{}  sample:{}".format(
          analytical_kl_chol_, sample_kl_chol_))

      self.assertAllClose(true_mean, sample_mean_,
                          atol=0., rtol=0.02)
      self.assertAllClose(true_mean, analytical_mean_,
                          atol=0., rtol=1e-6)

      self.assertAllClose(true_covariance, sample_covariance_,
                          atol=0., rtol=0.02)
      self.assertAllClose(true_covariance, analytical_covariance_,
                          atol=0., rtol=1e-6)

      self.assertAllClose(true_variance, sample_variance_,
                          atol=0., rtol=0.02)
      self.assertAllClose(true_variance, analytical_variance_,
                          atol=0., rtol=1e-6)

      self.assertAllClose(true_stddev, sample_stddev_,
                          atol=0., rtol=0.02)
      self.assertAllClose(true_stddev, analytical_stddev_,
                          atol=0., rtol=1e-6)

      self.assertAllClose(true_log_det_covariance, sample_log_det_covariance_,
                          atol=0., rtol=0.02)
      self.assertAllClose(true_log_det_covariance,
                          analytical_log_det_covariance_,
                          atol=0., rtol=1e-6)

      self.assertAllClose(true_det_covariance, sample_det_covariance_,
                          atol=0., rtol=0.02)
      self.assertAllClose(true_det_covariance, analytical_det_covariance_,
                          atol=0., rtol=1e-5)

      self.assertAllClose(true_scale, scale_,
                          atol=0., rtol=1e-6)

      self.assertAllClose(sample_kl_identity_, analytical_kl_identity_,
                          atol=0., rtol=0.02)
      self.assertAllClose(sample_kl_scaled_, analytical_kl_scaled_,
                          atol=0., rtol=0.02)
      self.assertAllClose(sample_kl_diag_, analytical_kl_diag_,
                          atol=0., rtol=0.02)
      self.assertAllClose(sample_kl_chol_, analytical_kl_chol_,
                          atol=0., rtol=0.02)

  def testImplicitLargeDiag(self):
    mu = np.array([[1., 2, 3],
                   [11, 22, 33]])      # shape: [b, k] = [2, 3]
    u = np.array([[[1., 2],
                   [3, 4],
                   [5, 6]],
                  [[0.5, 0.75],
                   [1, 0.25],
                   [1.5, 1.25]]])      # shape: [b, k, r] = [2, 3, 2]
    m = np.array([[0.1, 0.2],
                  [0.4, 0.5]])         # shape: [b, r] = [2, 2]
    scale = np.stack([
        np.eye(3) + np.matmul(np.matmul(u[0], np.diag(m[0])),
                              np.transpose(u[0])),
        np.eye(3) + np.matmul(np.matmul(u[1], np.diag(m[1])),
                              np.transpose(u[1])),
    ])
    cov = np.stack([np.matmul(scale[0], scale[0].T),
                    np.matmul(scale[1], scale[1].T)])
    print("expected_cov:\n{}".format(cov))
    with self.test_session():
      mvn = ds.MultivariateNormalDiagPlusLowRank(
          loc=mu,
          scale_perturb_factor=u,
          scale_perturb_diag=m)
      self.assertAllClose(cov, mvn.covariance().eval(), atol=0., rtol=1e-6)


class MultivariateNormalTriLTest(test.TestCase):

  def setUp(self):
    self._rng = np.random.RandomState(42)

  def _random_chol(self, *shape):
    mat = self._rng.rand(*shape)
    chol = ds.matrix_diag_transform(mat, transform=nn_ops.softplus)
    chol = array_ops.matrix_band_part(chol, -1, 0)
    sigma = math_ops.matmul(chol, chol, adjoint_b=True)
    return chol.eval(), sigma.eval()

  def testLogPDFScalarBatch(self):
    with self.test_session():
      mu = self._rng.rand(2)
      chol, sigma = self._random_chol(2, 2)
      chol[1, 1] = -chol[1, 1]
      mvn = ds.MultivariateNormalTriL(mu, chol, validate_args=True)
      x = self._rng.rand(2)

      log_pdf = mvn.log_prob(x)
      pdf = mvn.prob(x)

      scipy_mvn = stats.multivariate_normal(mean=mu, cov=sigma)

      expected_log_pdf = scipy_mvn.logpdf(x)
      expected_pdf = scipy_mvn.pdf(x)
      self.assertEqual((), log_pdf.get_shape())
      self.assertEqual((), pdf.get_shape())
      self.assertAllClose(expected_log_pdf, log_pdf.eval())
      self.assertAllClose(expected_pdf, pdf.eval())

  def testLogPDFXIsHigherRank(self):
    with self.test_session():
      mu = self._rng.rand(2)
      chol, sigma = self._random_chol(2, 2)
      chol[0, 0] = -chol[0, 0]
      mvn = ds.MultivariateNormalTriL(mu, chol, validate_args=True)
      x = self._rng.rand(3, 2)

      log_pdf = mvn.log_prob(x)
      pdf = mvn.prob(x)

      scipy_mvn = stats.multivariate_normal(mean=mu, cov=sigma)

      expected_log_pdf = scipy_mvn.logpdf(x)
      expected_pdf = scipy_mvn.pdf(x)
      self.assertEqual((3,), log_pdf.get_shape())
      self.assertEqual((3,), pdf.get_shape())
      self.assertAllClose(expected_log_pdf, log_pdf.eval(), atol=0., rtol=0.02)
      self.assertAllClose(expected_pdf, pdf.eval(), atol=0., rtol=0.03)

  def testLogPDFXLowerDimension(self):
    with self.test_session():
      mu = self._rng.rand(3, 2)
      chol, sigma = self._random_chol(3, 2, 2)
      chol[0, 0, 0] = -chol[0, 0, 0]
      chol[2, 1, 1] = -chol[2, 1, 1]
      mvn = ds.MultivariateNormalTriL(mu, chol, validate_args=True)
      x = self._rng.rand(2)

      log_pdf = mvn.log_prob(x)
      pdf = mvn.prob(x)

      self.assertEqual((3,), log_pdf.get_shape())
      self.assertEqual((3,), pdf.get_shape())

      # scipy can't do batches, so just test one of them.
      scipy_mvn = stats.multivariate_normal(mean=mu[1, :], cov=sigma[1, :, :])
      expected_log_pdf = scipy_mvn.logpdf(x)
      expected_pdf = scipy_mvn.pdf(x)

      self.assertAllClose(expected_log_pdf, log_pdf.eval()[1])
      self.assertAllClose(expected_pdf, pdf.eval()[1])

  def testEntropy(self):
    with self.test_session():
      mu = self._rng.rand(2)
      chol, sigma = self._random_chol(2, 2)
      chol[0, 0] = -chol[0, 0]
      mvn = ds.MultivariateNormalTriL(mu, chol, validate_args=True)
      entropy = mvn.entropy()

      scipy_mvn = stats.multivariate_normal(mean=mu, cov=sigma)
      expected_entropy = scipy_mvn.entropy()
      self.assertEqual(entropy.get_shape(), ())
      self.assertAllClose(expected_entropy, entropy.eval())

  def testEntropyMultidimensional(self):
    with self.test_session():
      mu = self._rng.rand(3, 5, 2)
      chol, sigma = self._random_chol(3, 5, 2, 2)
      chol[1, 0, 0, 0] = -chol[1, 0, 0, 0]
      chol[2, 3, 1, 1] = -chol[2, 3, 1, 1]
      mvn = ds.MultivariateNormalTriL(mu, chol, validate_args=True)
      entropy = mvn.entropy()

      # Scipy doesn't do batches, so test one of them.
      expected_entropy = stats.multivariate_normal(
          mean=mu[1, 1, :], cov=sigma[1, 1, :, :]).entropy()
      self.assertEqual(entropy.get_shape(), (3, 5))
      self.assertAllClose(expected_entropy, entropy.eval()[1, 1])

  def testSample(self):
    with self.test_session():
      mu = self._rng.rand(2)
      chol, sigma = self._random_chol(2, 2)
      chol[0, 0] = -chol[0, 0]
      sigma[0, 1] = -sigma[0, 1]
      sigma[1, 0] = -sigma[1, 0]

      n = constant_op.constant(100000)
      mvn = ds.MultivariateNormalTriL(mu, chol, validate_args=True)
      samples = mvn.sample(n, seed=137)
      sample_values = samples.eval()
      self.assertEqual(samples.get_shape(), [int(100e3), 2])
      self.assertAllClose(sample_values.mean(axis=0), mu, atol=1e-2)
      self.assertAllClose(np.cov(sample_values, rowvar=0), sigma, atol=0.06)

  def testSampleWithSampleShape(self):
    with self.test_session():
      mu = self._rng.rand(3, 5, 2)
      chol, sigma = self._random_chol(3, 5, 2, 2)
      chol[1, 0, 0, 0] = -chol[1, 0, 0, 0]
      chol[2, 3, 1, 1] = -chol[2, 3, 1, 1]

      mvn = ds.MultivariateNormalTriL(mu, chol, validate_args=True)
      samples_val = mvn.sample((10, 11, 12), seed=137).eval()

      # Check sample shape
      self.assertEqual((10, 11, 12, 3, 5, 2), samples_val.shape)

      # Check sample means
      x = samples_val[:, :, :, 1, 1, :]
      self.assertAllClose(
          x.reshape(10 * 11 * 12, 2).mean(axis=0), mu[1, 1], atol=0.05)

      # Check that log_prob(samples) works
      log_prob_val = mvn.log_prob(samples_val).eval()
      x_log_pdf = log_prob_val[:, :, :, 1, 1]
      expected_log_pdf = stats.multivariate_normal(
          mean=mu[1, 1, :], cov=sigma[1, 1, :, :]).logpdf(x)
      self.assertAllClose(expected_log_pdf, x_log_pdf)

  def testSampleMultiDimensional(self):
    with self.test_session():
      mu = self._rng.rand(3, 5, 2)
      chol, sigma = self._random_chol(3, 5, 2, 2)
      chol[1, 0, 0, 0] = -chol[1, 0, 0, 0]
      chol[2, 3, 1, 1] = -chol[2, 3, 1, 1]

      mvn = ds.MultivariateNormalTriL(mu, chol, validate_args=True)
      n = constant_op.constant(100000)
      samples = mvn.sample(n, seed=137)
      sample_values = samples.eval()

      self.assertEqual(samples.get_shape(), (100000, 3, 5, 2))
      self.assertAllClose(
          sample_values[:, 1, 1, :].mean(axis=0), mu[1, 1, :], atol=0.05)
      self.assertAllClose(
          np.cov(sample_values[:, 1, 1, :], rowvar=0),
          sigma[1, 1, :, :],
          atol=1e-1)

  def testShapes(self):
    with self.test_session():
      mu = self._rng.rand(3, 5, 2)
      chol, _ = self._random_chol(3, 5, 2, 2)
      chol[1, 0, 0, 0] = -chol[1, 0, 0, 0]
      chol[2, 3, 1, 1] = -chol[2, 3, 1, 1]

      mvn = ds.MultivariateNormalTriL(mu, chol, validate_args=True)

      # Shapes known at graph construction time.
      self.assertEqual((2,), tuple(mvn.event_shape.as_list()))
      self.assertEqual((3, 5), tuple(mvn.batch_shape.as_list()))

      # Shapes known at runtime.
      self.assertEqual((2,), tuple(mvn.event_shape_tensor().eval()))
      self.assertEqual((3, 5), tuple(mvn.batch_shape_tensor().eval()))

  def _random_mu_and_sigma(self, batch_shape, event_shape):
    # This ensures sigma is positive def.
    mat_shape = batch_shape + event_shape + event_shape
    mat = self._rng.randn(*mat_shape)
    perm = np.arange(mat.ndim)
    perm[-2:] = [perm[-1], perm[-2]]
    sigma = np.matmul(mat, np.transpose(mat, perm))

    mu_shape = batch_shape + event_shape
    mu = self._rng.randn(*mu_shape)

    return mu, sigma

  def testKLNonBatch(self):
    batch_shape = ()
    event_shape = (2,)
    with self.test_session():
      mu_a, sigma_a = self._random_mu_and_sigma(batch_shape, event_shape)
      mu_b, sigma_b = self._random_mu_and_sigma(batch_shape, event_shape)
      mvn_a = ds.MultivariateNormalTriL(
          loc=mu_a,
          scale_tril=np.linalg.cholesky(sigma_a),
          validate_args=True)
      mvn_b = ds.MultivariateNormalTriL(
          loc=mu_b,
          scale_tril=np.linalg.cholesky(sigma_b),
          validate_args=True)

      kl = ds.kl(mvn_a, mvn_b)
      self.assertEqual(batch_shape, kl.get_shape())

      kl_v = kl.eval()
      expected_kl = _compute_non_batch_kl(mu_a, sigma_a, mu_b, sigma_b)
      self.assertAllClose(expected_kl, kl_v)

  def testKLBatch(self):
    batch_shape = (2,)
    event_shape = (3,)
    with self.test_session():
      mu_a, sigma_a = self._random_mu_and_sigma(batch_shape, event_shape)
      mu_b, sigma_b = self._random_mu_and_sigma(batch_shape, event_shape)
      mvn_a = ds.MultivariateNormalTriL(
          loc=mu_a,
          scale_tril=np.linalg.cholesky(sigma_a),
          validate_args=True)
      mvn_b = ds.MultivariateNormalTriL(
          loc=mu_b,
          scale_tril=np.linalg.cholesky(sigma_b),
          validate_args=True)

      kl = ds.kl(mvn_a, mvn_b)
      self.assertEqual(batch_shape, kl.get_shape())

      kl_v = kl.eval()
      expected_kl_0 = _compute_non_batch_kl(mu_a[0, :], sigma_a[0, :, :],
                                            mu_b[0, :], sigma_b[0, :])
      expected_kl_1 = _compute_non_batch_kl(mu_a[1, :], sigma_a[1, :, :],
                                            mu_b[1, :], sigma_b[1, :])
      self.assertAllClose(expected_kl_0, kl_v[0])
      self.assertAllClose(expected_kl_1, kl_v[1])

  def testKLTwoIdenticalDistributionsIsZero(self):
    batch_shape = (2,)
    event_shape = (3,)
    with self.test_session():
      mu_a, sigma_a = self._random_mu_and_sigma(batch_shape, event_shape)
      mvn_a = ds.MultivariateNormalTriL(
          loc=mu_a,
          scale_tril=np.linalg.cholesky(sigma_a),
          validate_args=True)

      # Should be zero since KL(p || p) = =.
      kl = ds.kl(mvn_a, mvn_a)
      self.assertEqual(batch_shape, kl.get_shape())

      kl_v = kl.eval()
      self.assertAllClose(np.zeros(*batch_shape), kl_v)

  def testSampleLarge(self):
    mu = np.array([-1., 1], dtype=np.float32)
    scale_tril = np.array([[3., 0], [1, -2]], dtype=np.float32) / 3.

    true_mean = mu
    true_scale = scale_tril
    true_covariance = np.matmul(true_scale, true_scale.T)
    true_variance = np.diag(true_covariance)
    true_stddev = np.sqrt(true_variance)
    true_det_covariance = np.linalg.det(true_covariance)
    true_log_det_covariance = np.log(true_det_covariance)

    with self.test_session() as sess:
      dist = ds.MultivariateNormalTriL(
          loc=mu,
          scale_tril=scale_tril,
          validate_args=True)

      # The following distributions will test the KL divergence calculation.
      mvn_chol = ds.MultivariateNormalTriL(
          loc=np.array([0.5, 1.2], dtype=np.float32),
          scale_tril=np.array([[3., 0], [1, 2]], dtype=np.float32),
          validate_args=True)

      n = int(10e3)
      samps = dist.sample(n, seed=0)
      sample_mean = math_ops.reduce_mean(samps, 0)
      x = samps - sample_mean
      sample_covariance = math_ops.matmul(x, x, transpose_a=True) / n

      sample_kl_chol = math_ops.reduce_mean(
          dist.log_prob(samps) - mvn_chol.log_prob(samps), 0)
      analytical_kl_chol = ds.kl(dist, mvn_chol)

      scale = dist.scale.to_dense()

      [
          sample_mean_,
          analytical_mean_,
          sample_covariance_,
          analytical_covariance_,
          analytical_variance_,
          analytical_stddev_,
          analytical_log_det_covariance_,
          analytical_det_covariance_,
          sample_kl_chol_, analytical_kl_chol_,
          scale_,
      ] = sess.run([
          sample_mean,
          dist.mean(),
          sample_covariance,
          dist.covariance(),
          dist.variance(),
          dist.stddev(),
          dist.log_det_covariance(),
          dist.det_covariance(),
          sample_kl_chol, analytical_kl_chol,
          scale,
      ])

      sample_variance_ = np.diag(sample_covariance_)
      sample_stddev_ = np.sqrt(sample_variance_)
      sample_det_covariance_ = np.linalg.det(sample_covariance_)
      sample_log_det_covariance_ = np.log(sample_det_covariance_)

      print("true_mean:\n{}  ".format(true_mean))
      print("sample_mean:\n{}".format(sample_mean_))
      print("analytical_mean:\n{}".format(analytical_mean_))

      print("true_covariance:\n{}".format(true_covariance))
      print("sample_covariance:\n{}".format(sample_covariance_))
      print("analytical_covariance:\n{}".format(analytical_covariance_))

      print("true_variance:\n{}".format(true_variance))
      print("sample_variance:\n{}".format(sample_variance_))
      print("analytical_variance:\n{}".format(analytical_variance_))

      print("true_stddev:\n{}".format(true_stddev))
      print("sample_stddev:\n{}".format(sample_stddev_))
      print("analytical_stddev:\n{}".format(analytical_stddev_))

      print("true_log_det_covariance:\n{}".format(true_log_det_covariance))
      print("sample_log_det_covariance:\n{}".format(sample_log_det_covariance_))
      print("analytical_log_det_covariance:\n{}".format(
          analytical_log_det_covariance_))

      print("true_det_covariance:\n{}".format(true_det_covariance))
      print("sample_det_covariance:\n{}".format(sample_det_covariance_))
      print("analytical_det_covariance:\n{}".format(analytical_det_covariance_))

      print("true_scale:\n{}".format(true_scale))
      print("scale:\n{}".format(scale_))

      print("kl_chol:      analytical:{}  sample:{}".format(
          analytical_kl_chol_, sample_kl_chol_))

      self.assertAllClose(true_mean, sample_mean_,
                          atol=0., rtol=0.03)
      self.assertAllClose(true_mean, analytical_mean_,
                          atol=0., rtol=1e-6)

      self.assertAllClose(true_covariance, sample_covariance_,
                          atol=0., rtol=0.03)
      self.assertAllClose(true_covariance, analytical_covariance_,
                          atol=0., rtol=1e-6)

      self.assertAllClose(true_variance, sample_variance_,
                          atol=0., rtol=0.02)
      self.assertAllClose(true_variance, analytical_variance_,
                          atol=0., rtol=1e-6)

      self.assertAllClose(true_stddev, sample_stddev_,
                          atol=0., rtol=0.01)
      self.assertAllClose(true_stddev, analytical_stddev_,
                          atol=0., rtol=1e-6)

      self.assertAllClose(true_log_det_covariance, sample_log_det_covariance_,
                          atol=0., rtol=0.04)
      self.assertAllClose(true_log_det_covariance,
                          analytical_log_det_covariance_,
                          atol=0., rtol=1e-6)

      self.assertAllClose(true_det_covariance, sample_det_covariance_,
                          atol=0., rtol=0.03)
      self.assertAllClose(true_det_covariance, analytical_det_covariance_,
                          atol=0., rtol=1e-6)

      self.assertAllClose(true_scale, scale_,
                          atol=0., rtol=1e-6)

      self.assertAllClose(sample_kl_chol_, analytical_kl_chol_,
                          atol=0., rtol=0.02)


def _compute_non_batch_kl(mu_a, sigma_a, mu_b, sigma_b):
  """Non-batch KL for N(mu_a, sigma_a), N(mu_b, sigma_b)."""
  # Check using numpy operations
  # This mostly repeats the tensorflow code _kl_mvn_mvn(), but in numpy.
  # So it is important to also check that KL(mvn, mvn) = 0.
  sigma_b_inv = np.linalg.inv(sigma_b)

  t = np.trace(sigma_b_inv.dot(sigma_a))
  q = (mu_b - mu_a).dot(sigma_b_inv).dot(mu_b - mu_a)
  k = mu_a.shape[0]
  l = np.log(np.linalg.det(sigma_b) / np.linalg.det(sigma_a))

  return 0.5 * (t + q - k + l)


if __name__ == "__main__":
  test.main()
