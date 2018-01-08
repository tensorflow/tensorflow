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

from tensorflow.contrib.distributions.python.ops import test_util
from tensorflow.contrib.distributions.python.ops import vector_diffeomixture as vdm_lib
from tensorflow.python.ops.distributions import normal as normal_lib
from tensorflow.python.ops.linalg import linear_operator_diag as linop_diag_lib
from tensorflow.python.ops.linalg import linear_operator_identity as linop_identity_lib
from tensorflow.python.platform import test


class VectorDiffeomixtureTest(
    test_util.VectorDistributionTestHelpers, test.TestCase):
  """Tests the VectorDiffeomixture distribution."""

  def testSampleProbConsistentBroadcastMixNoBatch(self):
    with self.test_session() as sess:
      dims = 4
      vdm = vdm_lib.VectorDiffeomixture(
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
          quadrature_size=8,
          validate_args=True)
      # Ball centered at component0's mean.
      self.run_test_sample_consistent_log_prob(
          sess.run, vdm, radius=2., center=0., rtol=0.015)
      # Larger ball centered at component1's mean.
      self.run_test_sample_consistent_log_prob(
          sess.run, vdm, radius=4., center=2., rtol=0.015)

  def testSampleProbConsistentBroadcastMixNonStandardBase(self):
    with self.test_session() as sess:
      dims = 4
      vdm = vdm_lib.VectorDiffeomixture(
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
          quadrature_size=8,
          validate_args=True)
      # Ball centered at component0's mean.
      self.run_test_sample_consistent_log_prob(
          sess.run, vdm, radius=2., center=1., rtol=0.015)
      # Larger ball centered at component1's mean.
      self.run_test_sample_consistent_log_prob(
          sess.run, vdm, radius=4., center=3., rtol=0.01)

  def testSampleProbConsistentBroadcastMixBatch(self):
    with self.test_session() as sess:
      dims = 4
      vdm = vdm_lib.VectorDiffeomixture(
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
                  multiplier=[np.float32(1.1)],
                  is_positive_definite=True),
              linop_diag_lib.LinearOperatorDiag(
                  diag=np.stack([
                      np.linspace(2.5, 3.5, dims, dtype=np.float32),
                      np.linspace(2.75, 3.25, dims, dtype=np.float32),
                  ]),
                  is_positive_definite=True),
          ],
          quadrature_size=8,
          validate_args=True)
      # Ball centered at component0's mean.
      self.run_test_sample_consistent_log_prob(
          sess.run, vdm, radius=2., center=0., rtol=0.01)
      # Larger ball centered at component1's mean.
      self.run_test_sample_consistent_log_prob(
          sess.run, vdm, radius=4., center=2., rtol=0.01)

  def testMeanCovarianceNoBatch(self):
    with self.test_session() as sess:
      dims = 3
      vdm = vdm_lib.VectorDiffeomixture(
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
          quadrature_size=8,
          validate_args=True)
      self.run_test_sample_consistent_mean_covariance(
          sess.run, vdm, rtol=0.02, cov_rtol=0.08)

  def testMeanCovarianceNoBatchUncenteredNonStandardBase(self):
    with self.test_session() as sess:
      dims = 3
      vdm = vdm_lib.VectorDiffeomixture(
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
          quadrature_size=8,
          validate_args=True)
      self.run_test_sample_consistent_mean_covariance(
          sess.run, vdm, num_samples=int(1e6), rtol=0.01, cov_atol=0.025)

  def testMeanCovarianceBatch(self):
    with self.test_session() as sess:
      dims = 3
      vdm = vdm_lib.VectorDiffeomixture(
          mix_loc=[[0.], [4.]],
          mix_scale=[10.],
          distribution=normal_lib.Normal(0., 1.),
          loc=[
              np.float32([[-2.]]),
              None,
          ],
          scale=[
              linop_identity_lib.LinearOperatorScaledIdentity(
                  num_rows=dims,
                  multiplier=[np.float32(1.5)],
                  is_positive_definite=True),
              linop_diag_lib.LinearOperatorDiag(
                  diag=np.stack([
                      np.linspace(2.5, 3.5, dims, dtype=np.float32),
                      np.linspace(0.5, 1.5, dims, dtype=np.float32),
                  ]),
                  is_positive_definite=True),
          ],
          quadrature_size=8,
          validate_args=True)
      self.run_test_sample_consistent_mean_covariance(
          sess.run, vdm, rtol=0.02, cov_rtol=0.07)

  def testSampleProbConsistentQuadrature(self):
    with self.test_session() as sess:
      dims = 4
      vdm = vdm_lib.VectorDiffeomixture(
          mix_loc=[0.],
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
          quadrature_size=3,
          validate_args=True)
      # Ball centered at component0's mean.
      self.run_test_sample_consistent_log_prob(
          sess.run, vdm, radius=2., center=0., rtol=0.015)
      # Larger ball centered at component1's mean.
      self.run_test_sample_consistent_log_prob(
          sess.run, vdm, radius=4., center=2., rtol=0.005)

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
