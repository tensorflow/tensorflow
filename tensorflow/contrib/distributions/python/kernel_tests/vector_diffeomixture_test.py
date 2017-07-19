# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for VectorDiffeomixture."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.distributions.python.ops import vector_diffeomixture as vector_diffeomixture_lib
from tensorflow.contrib.linalg.python.ops import linear_operator_diag as linop_diag_lib
from tensorflow.contrib.linalg.python.ops import linear_operator_identity as linop_identity_lib
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.distributions import normal as normal_lib
from tensorflow.python.platform import test


class VectorDistributionTestHelpers(object):
  """VectorDistributionTestHelpers helps test vector-event distributions."""

  def linop(self, num_rows=None, multiplier=None, diag=None):
    """Helper to create non-singular, symmetric, positive definite matrices."""
    if num_rows is not None and multiplier is not None:
      if any(p is not None for p in [diag]):
        raise ValueError("Found extra args for scaled identity.")
      return linop_identity_lib.LinearOperatorScaledIdentity(
          num_rows=num_rows,
          multiplier=multiplier,
          is_positive_definite=True)
    elif num_rows is not None:
      if any(p is not None for p in [multiplier, diag]):
        raise ValueError("Found extra args for identity.")
      return linop_identity_lib.LinearOperatorIdentity(
          num_rows=num_rows,
          is_positive_definite=True)
    elif diag is not None:
      if any(p is not None for p in [num_rows, multiplier]):
        raise ValueError("Found extra args for diag.")
      return linop_diag_lib.LinearOperatorDiag(
          diag=diag,
          is_positive_definite=True)
    else:
      raise ValueError("Must specify at least one arg.")

  def run_test_sample_consistent_log_prob(
      self,
      sess,
      dist,
      num_samples=int(1e5),
      radius=1.,
      center=0.,
      seed=42,
      rtol=1e-2,
      atol=0.):
    """Tests that sample/log_prob are mutually consistent.

    "Consistency" means that `sample` and `log_prob` correspond to the same
    distribution.

    The idea of this test is to compute the Monte-Carlo estimate of the volume
    enclosed by a hypersphere, i.e., the volume of an `n`-ball. While we could
    choose an arbitrary function to integrate, the hypersphere's volume is nice
    because it is intuitive, has an easy analytical expression, and works for
    `dimensions > 1`.

    Technical Details:

    Observe that:

    ```none
    int_{R**d} dx [x in Ball(radius=r, center=c)]
    = E_{p(X)}[ [X in Ball(r, c)] / p(X) ]
    = lim_{m->infty} m**-1 sum_j^m [x[j] in Ball(r, c)] / p(x[j]),
        where x[j] ~iid p(X)
    ```

    Thus, for fixed `m`, the above is approximately true when `sample` and
    `log_prob` are mutually consistent.

    Furthermore, the above calculation has the analytical result:
    `pi**(d/2) r**d / Gamma(1 + d/2)`.

    Note: this test only verifies a necessary condition for consistency--it does
    does not verify sufficiency hence does not prove `sample`, `log_prob` truly
    are consistent. For this reason we recommend testing several different
    hyperspheres (assuming the hypersphere is supported by the distribution).
    Furthermore, we gain additional trust in this test when also tested `sample`
    against the first, second moments
    (`run_test_sample_consistent_mean_covariance`); it is probably unlikely that
    a "best-effort" implementation of `log_prob` would incorrectly pass both
    tests and for different hyperspheres.

    For a discussion on the analytical result (second-line) see:
      https://en.wikipedia.org/wiki/Volume_of_an_n-ball.

    For a discussion of importance sampling (fourth-line) see:
      https://en.wikipedia.org/wiki/Importance_sampling.

    Args:
      sess: Tensorflow session.
      dist: Distribution instance or object which implements `sample`,
        `log_prob`, `event_shape_tensor` and `batch_shape_tensor`. The
        distribution must have non-zero probability of sampling every point
        enclosed by the hypersphere.
      num_samples: Python `int` scalar indicating the number of Monte-Carlo
        samples to draw from `dist`.
      radius: Python `float`-type indicating the radius of the `n`-ball which
        we're computing the volume.
      center: Python floating-type vector (or scalar) indicating the center of
        the `n`-ball which we're computing the volume. When scalar, the value is
        broadcast to all event dims.
      seed: Python `int` indicating the seed to use when sampling from `dist`.
        In general it is not recommended to use `None` during a test as this
        increases the likelihood of spurious test failure.
      rtol: Python `float`-type indicating the admissible relative error between
        actual- and approximate-volumes.
      atol: Python `float`-type indicating the admissible absolute error between
        actual- and approximate-volumes. In general this should be zero since
        a typical radius implies a non-zero volume.
    """

    def actual_hypersphere_volume(dims, radius):
      # https://en.wikipedia.org/wiki/Volume_of_an_n-ball
      # Using tf.lgamma because we'd have to otherwise use SciPy which is not
      # a required dependency of core.
      radius = np.asarray(radius)
      dims = math_ops.cast(dims, dtype=radius.dtype)
      return math_ops.exp(
          (dims / 2.) * np.log(np.pi)
          - math_ops.lgamma(1. + dims / 2.)
          + dims * math_ops.log(radius))

    def is_in_ball(x, radius, center):
      return math_ops.cast(linalg_ops.norm(x - center, axis=-1) <= radius,
                           dtype=x.dtype)

    def monte_carlo_hypersphere_volume(dist, num_samples, radius, center):
      # https://en.wikipedia.org/wiki/Importance_sampling
      x = dist.sample(num_samples, seed=seed)
      return math_ops.reduce_mean(
          math_ops.exp(-dist.log_prob(x)) * is_in_ball(x, radius, center),
          axis=0)

    [
        batch_shape_,
        actual_volume_,
        sample_volume_,
    ] = sess.run([
        dist.batch_shape_tensor(),
        actual_hypersphere_volume(
            dims=dist.event_shape_tensor()[0],
            radius=radius),
        monte_carlo_hypersphere_volume(
            dist,
            num_samples=num_samples,
            radius=radius,
            center=center),
    ])

    self.assertAllClose(np.tile(actual_volume_, reps=batch_shape_),
                        sample_volume_,
                        rtol=rtol, atol=atol)

  def run_test_sample_consistent_mean_covariance(
      self,
      sess,
      dist,
      num_samples=int(1e5),
      seed=24,
      rtol=1e-2,
      atol=0.,
      cov_rtol=None,
      cov_atol=None):
    """Tests that sample/mean/covariance are consistent with each other.

    "Consistency" means that `sample`, `mean`, `covariance`, etc all correspond
    to the same distribution.

    Args:
      sess: Tensorflow session.
      dist: Distribution instance or object which implements `sample`,
        `log_prob`, `event_shape_tensor` and `batch_shape_tensor`.
      num_samples: Python `int` scalar indicating the number of Monte-Carlo
        samples to draw from `dist`.
      seed: Python `int` indicating the seed to use when sampling from `dist`.
        In general it is not recommended to use `None` during a test as this
        increases the likelihood of spurious test failure.
      rtol: Python `float`-type indicating the admissible relative error between
        analytical and sample statistics.
      atol: Python `float`-type indicating the admissible absolute error between
        analytical and sample statistics.
      cov_rtol: Python `float`-type indicating the admissible relative error
        between analytical and sample covariance. Default: rtol.
      cov_atol: Python `float`-type indicating the admissible absolute error
        between analytical and sample covariance. Default: atol.
    """

    def vec_osquare(x):
      """Computes the outer-product of a vector, i.e., x.T x."""
      return x[..., :, array_ops.newaxis] * x[..., array_ops.newaxis, :]

    x = dist.sample(num_samples, seed=seed)
    sample_mean = math_ops.reduce_mean(x, axis=0)
    sample_covariance = math_ops.reduce_mean(
        vec_osquare(x - sample_mean), axis=0)
    sample_variance = array_ops.matrix_diag_part(sample_covariance)
    sample_stddev = math_ops.sqrt(sample_variance)

    [
        sample_mean_,
        sample_covariance_,
        sample_variance_,
        sample_stddev_,
        mean_,
        covariance_,
        variance_,
        stddev_
    ] = sess.run([
        sample_mean,
        sample_covariance,
        sample_variance,
        sample_stddev,
        dist.mean(),
        dist.covariance(),
        dist.variance(),
        dist.stddev(),
    ])

    self.assertAllClose(mean_, sample_mean_, rtol=rtol, atol=atol)
    self.assertAllClose(covariance_, sample_covariance_,
                        rtol=cov_rtol or rtol,
                        atol=cov_atol or atol)
    self.assertAllClose(variance_, sample_variance_, rtol=rtol, atol=atol)
    self.assertAllClose(stddev_, sample_stddev_, rtol=rtol, atol=atol)


class VectorDiffeomixtureTest(VectorDistributionTestHelpers, test.TestCase):
  """Tests the VectorDiffeomixture distribution."""

  def testSampleProbConsistentBroadcastMix(self):
    with self.test_session() as sess:
      dims = 4
      vdm = vector_diffeomixture_lib.VectorDiffeomixture(
          mix_loc=[[0.], [1.]],
          mix_scale=[1.],
          distribution=normal_lib.Normal(0., 1.),
          loc=[
              None,
              np.float32([2.]*dims),
          ],
          scale=[
              linop_identity_lib.LinearOperatorScaledIdentity(
                  num_rows=dims,
                  multiplier=np.float32(1.1),
                  is_positive_definite=True),
              linop_diag_lib.LinearOperatorDiag(
                  diag=np.linspace(2.5, 3.5, dims, dtype=np.float32),
                  is_positive_definite=True),
          ],
          validate_args=True)
      # Ball centered at component0's mean.
      self.run_test_sample_consistent_log_prob(
          sess, vdm, radius=2., center=0., rtol=0.005)
      # Larger ball centered at component1's mean.
      self.run_test_sample_consistent_log_prob(
          sess, vdm, radius=4., center=2., rtol=0.005)

  def testSampleProbConsistentBroadcastMixNonStandardBase(self):
    with self.test_session() as sess:
      dims = 4
      vdm = vector_diffeomixture_lib.VectorDiffeomixture(
          mix_loc=[[0.], [1.]],
          mix_scale=[1.],
          distribution=normal_lib.Normal(1., 1.5),
          loc=[
              None,
              np.float32([2.]*dims),
          ],
          scale=[
              linop_identity_lib.LinearOperatorScaledIdentity(
                  num_rows=dims,
                  multiplier=np.float32(1.1),
                  is_positive_definite=True),
              linop_diag_lib.LinearOperatorDiag(
                  diag=np.linspace(2.5, 3.5, dims, dtype=np.float32),
                  is_positive_definite=True),
          ],
          validate_args=True)
      # Ball centered at component0's mean.
      self.run_test_sample_consistent_log_prob(
          sess, vdm, radius=2., center=1., rtol=0.006)
      # Larger ball centered at component1's mean.
      self.run_test_sample_consistent_log_prob(
          sess, vdm, radius=4., center=3., rtol=0.009)

  def testMeanCovariance(self):
    with self.test_session() as sess:
      dims = 3
      vdm = vector_diffeomixture_lib.VectorDiffeomixture(
          mix_loc=[[0.], [4.]],
          mix_scale=[10.],
          distribution=normal_lib.Normal(0., 1.),
          loc=[
              np.float32([-2.]),
              None,
          ],
          scale=[
              linop_identity_lib.LinearOperatorScaledIdentity(
                  num_rows=dims,
                  multiplier=np.float32(1.5),
                  is_positive_definite=True),
              linop_diag_lib.LinearOperatorDiag(
                  diag=np.linspace(2.5, 3.5, dims, dtype=np.float32),
                  is_positive_definite=True),
          ],
          validate_args=True)
      self.run_test_sample_consistent_mean_covariance(
          sess, vdm, rtol=0.02, cov_rtol=0.06)

  def testMeanCovarianceUncenteredNonStandardBase(self):
    with self.test_session() as sess:
      dims = 3
      vdm = vector_diffeomixture_lib.VectorDiffeomixture(
          mix_loc=[[0.], [4.]],
          mix_scale=[10.],
          distribution=normal_lib.Normal(-1., 1.5),
          loc=[
              np.float32([-2.]),
              np.float32([0.]),
          ],
          scale=[
              linop_identity_lib.LinearOperatorScaledIdentity(
                  num_rows=dims,
                  multiplier=np.float32(1.5),
                  is_positive_definite=True),
              linop_diag_lib.LinearOperatorDiag(
                  diag=np.linspace(2.5, 3.5, dims, dtype=np.float32),
                  is_positive_definite=True),
          ],
          validate_args=True)
      self.run_test_sample_consistent_mean_covariance(
          sess, vdm, num_samples=int(1e6), rtol=0.01, cov_atol=0.025)

  # TODO(jvdillon): We've tested that (i) .sample and .log_prob are consistent,
  # (ii) .mean, .stddev etc... and .sample are consistent. However, we haven't
  # tested that the quadrature approach well-approximates the integral.
  #
  # To that end, consider adding these tests:
  #
  # Test1: In the limit of high mix_scale, this approximates a discrete mixture,
  # and there are many discrete mixtures where we can explicitly compute
  # mean/var, etc... So test1 would choose one of those discrete mixtures and
  # show our mean/var/etc... is close to that.
  #
  # Test2:  In the limit of low mix_scale, the a diffeomixture of Normal(-5, 1),
  # Normal(5, 1) should (I believe...must check) should look almost like
  # Uniform(-5, 5), and thus (i) .prob(x) should be about 1/10 for x in (-5, 5),
  # and (ii) the first few moments should approximately match that of
  # Uniform(-5, 5)
  #
  # Test3:  If mix_loc is symmetric, then for any mix_scale, our
  # quadrature-based diffeomixture of Normal(-1, 1), Normal(1, 1) should have
  # mean zero, exactly.

  # TODO(jvdillon): Add more tests which verify broadcasting.


if __name__ == "__main__":
  test.main()
