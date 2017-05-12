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
from tensorflow.contrib import distributions
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging


ds = distributions


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

      scale = dist.scale.to_dense()

      n = int(30e3)
      samps = dist.sample(n, seed=0)
      sample_mean = math_ops.reduce_mean(samps, 0)
      x = samps - sample_mean
      sample_covariance = math_ops.matmul(x, x, transpose_a=True) / n

      sample_kl_identity = math_ops.reduce_mean(
          dist.log_prob(samps) - mvn_identity.log_prob(samps), 0)
      analytical_kl_identity = ds.kl_divergence(dist, mvn_identity)

      sample_kl_scaled = math_ops.reduce_mean(
          dist.log_prob(samps) - mvn_scaled.log_prob(samps), 0)
      analytical_kl_scaled = ds.kl_divergence(dist, mvn_scaled)

      sample_kl_diag = math_ops.reduce_mean(
          dist.log_prob(samps) - mvn_diag.log_prob(samps), 0)
      analytical_kl_diag = ds.kl_divergence(dist, mvn_diag)

      sample_kl_chol = math_ops.reduce_mean(
          dist.log_prob(samps) - mvn_chol.log_prob(samps), 0)
      analytical_kl_chol = ds.kl_divergence(dist, mvn_chol)

      n = int(10e3)
      baseline = ds.MultivariateNormalDiag(
          loc=np.array([-1., 0.25, 1.25], dtype=np.float32),
          scale_diag=np.array([1.5, 0.5, 1.], dtype=np.float32),
          validate_args=True)
      samps = baseline.sample(n, seed=0)

      sample_kl_identity_diag_baseline = math_ops.reduce_mean(
          baseline.log_prob(samps) - mvn_identity.log_prob(samps), 0)
      analytical_kl_identity_diag_baseline = ds.kl_divergence(
          baseline, mvn_identity)

      sample_kl_scaled_diag_baseline = math_ops.reduce_mean(
          baseline.log_prob(samps) - mvn_scaled.log_prob(samps), 0)
      analytical_kl_scaled_diag_baseline = ds.kl_divergence(
          baseline, mvn_scaled)

      sample_kl_diag_diag_baseline = math_ops.reduce_mean(
          baseline.log_prob(samps) - mvn_diag.log_prob(samps), 0)
      analytical_kl_diag_diag_baseline = ds.kl_divergence(baseline, mvn_diag)

      sample_kl_chol_diag_baseline = math_ops.reduce_mean(
          baseline.log_prob(samps) - mvn_chol.log_prob(samps), 0)
      analytical_kl_chol_diag_baseline = ds.kl_divergence(baseline, mvn_chol)

      [
          sample_mean_,
          analytical_mean_,
          sample_covariance_,
          analytical_covariance_,
          analytical_variance_,
          analytical_stddev_,
          scale_,
          sample_kl_identity_, analytical_kl_identity_,
          sample_kl_scaled_, analytical_kl_scaled_,
          sample_kl_diag_, analytical_kl_diag_,
          sample_kl_chol_, analytical_kl_chol_,
          sample_kl_identity_diag_baseline_,
          analytical_kl_identity_diag_baseline_,
          sample_kl_scaled_diag_baseline_, analytical_kl_scaled_diag_baseline_,
          sample_kl_diag_diag_baseline_, analytical_kl_diag_diag_baseline_,
          sample_kl_chol_diag_baseline_, analytical_kl_chol_diag_baseline_,
      ] = sess.run([
          sample_mean,
          dist.mean(),
          sample_covariance,
          dist.covariance(),
          dist.variance(),
          dist.stddev(),
          scale,
          sample_kl_identity, analytical_kl_identity,
          sample_kl_scaled, analytical_kl_scaled,
          sample_kl_diag, analytical_kl_diag,
          sample_kl_chol, analytical_kl_chol,
          sample_kl_identity_diag_baseline,
          analytical_kl_identity_diag_baseline,
          sample_kl_scaled_diag_baseline, analytical_kl_scaled_diag_baseline,
          sample_kl_diag_diag_baseline, analytical_kl_diag_diag_baseline,
          sample_kl_chol_diag_baseline, analytical_kl_chol_diag_baseline,
      ])

      sample_variance_ = np.diag(sample_covariance_)
      sample_stddev_ = np.sqrt(sample_variance_)

      logging.vlog(2, "true_mean:\n{}  ".format(true_mean))
      logging.vlog(2, "sample_mean:\n{}".format(sample_mean_))
      logging.vlog(2, "analytical_mean:\n{}".format(analytical_mean_))

      logging.vlog(2, "true_covariance:\n{}".format(true_covariance))
      logging.vlog(2, "sample_covariance:\n{}".format(sample_covariance_))
      logging.vlog(2, "analytical_covariance:\n{}".format(
          analytical_covariance_))

      logging.vlog(2, "true_variance:\n{}".format(true_variance))
      logging.vlog(2, "sample_variance:\n{}".format(sample_variance_))
      logging.vlog(2, "analytical_variance:\n{}".format(analytical_variance_))

      logging.vlog(2, "true_stddev:\n{}".format(true_stddev))
      logging.vlog(2, "sample_stddev:\n{}".format(sample_stddev_))
      logging.vlog(2, "analytical_stddev:\n{}".format(analytical_stddev_))

      logging.vlog(2, "true_scale:\n{}".format(true_scale))
      logging.vlog(2, "scale:\n{}".format(scale_))

      logging.vlog(2, "kl_identity:  analytical:{}  sample:{}".format(
          analytical_kl_identity_, sample_kl_identity_))

      logging.vlog(2, "kl_scaled:    analytical:{}  sample:{}".format(
          analytical_kl_scaled_, sample_kl_scaled_))

      logging.vlog(2, "kl_diag:      analytical:{}  sample:{}".format(
          analytical_kl_diag_, sample_kl_diag_))

      logging.vlog(2, "kl_chol:      analytical:{}  sample:{}".format(
          analytical_kl_chol_, sample_kl_chol_))

      logging.vlog(
          2, "kl_identity_diag_baseline:  analytical:{}  sample:{}".format(
              analytical_kl_identity_diag_baseline_,
              sample_kl_identity_diag_baseline_))

      logging.vlog(
          2, "kl_scaled_diag_baseline:  analytical:{}  sample:{}".format(
              analytical_kl_scaled_diag_baseline_,
              sample_kl_scaled_diag_baseline_))

      logging.vlog(2, "kl_diag_diag_baseline:  analytical:{}  sample:{}".format(
          analytical_kl_diag_diag_baseline_,
          sample_kl_diag_diag_baseline_))

      logging.vlog(2, "kl_chol_diag_baseline:  analytical:{}  sample:{}".format(
          analytical_kl_chol_diag_baseline_,
          sample_kl_chol_diag_baseline_))

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

      self.assertAllClose(
          sample_kl_identity_diag_baseline_,
          analytical_kl_identity_diag_baseline_,
          atol=0., rtol=0.02)
      self.assertAllClose(
          sample_kl_scaled_diag_baseline_,
          analytical_kl_scaled_diag_baseline_,
          atol=0., rtol=0.02)
      self.assertAllClose(
          sample_kl_diag_diag_baseline_,
          analytical_kl_diag_diag_baseline_,
          atol=0., rtol=0.04)
      self.assertAllClose(
          sample_kl_chol_diag_baseline_,
          analytical_kl_chol_diag_baseline_,
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
    logging.vlog(2, "expected_cov:\n{}".format(cov))
    with self.test_session():
      mvn = ds.MultivariateNormalDiagPlusLowRank(
          loc=mu,
          scale_perturb_factor=u,
          scale_perturb_diag=m)
      self.assertAllClose(cov, mvn.covariance().eval(), atol=0., rtol=1e-6)


if __name__ == "__main__":
  test.main()
