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

rng = np.random.RandomState(0)


class VectorDiffeomixtureTest(
    test_util.VectorDistributionTestHelpers, test.TestCase):
  """Tests the VectorDiffeomixture distribution."""

  def testSampleProbConsistentBroadcastMixNoBatch(self):
    with self.cached_session() as sess:
      dims = 4
      vdm = vdm_lib.VectorDiffeomixture(
          mix_loc=[[0.], [1.]],
          temperature=[1.],
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
    with self.cached_session() as sess:
      dims = 4
      vdm = vdm_lib.VectorDiffeomixture(
          mix_loc=[[0.], [1.]],
          temperature=[1.],
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
    with self.cached_session() as sess:
      dims = 4
      vdm = vdm_lib.VectorDiffeomixture(
          mix_loc=[[0.], [1.]],
          temperature=[1.],
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

  def testSampleProbConsistentBroadcastMixTwoBatchDims(self):
    dims = 4
    loc_1 = rng.randn(2, 3, dims).astype(np.float32)

    with self.cached_session() as sess:
      vdm = vdm_lib.VectorDiffeomixture(
          mix_loc=(rng.rand(2, 3, 1) - 0.5).astype(np.float32),
          temperature=[1.],
          distribution=normal_lib.Normal(0., 1.),
          loc=[
              None,
              loc_1,
          ],
          scale=[
              linop_identity_lib.LinearOperatorScaledIdentity(
                  num_rows=dims,
                  multiplier=[np.float32(1.1)],
                  is_positive_definite=True),
          ] * 2,
          validate_args=True)
      # Ball centered at component0's mean.
      self.run_test_sample_consistent_log_prob(
          sess.run, vdm, radius=2., center=0., rtol=0.01)
      # Larger ball centered at component1's mean.
      self.run_test_sample_consistent_log_prob(
          sess.run, vdm, radius=3., center=loc_1, rtol=0.02)

  def testMeanCovarianceNoBatch(self):
    with self.cached_session() as sess:
      dims = 3
      vdm = vdm_lib.VectorDiffeomixture(
          mix_loc=[[0.], [4.]],
          temperature=[1 / 10.],
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

  def testTemperatureControlsHowMuchThisLooksLikeDiscreteMixture(self):
    # As temperature decreases, this should approach a mixture of normals, with
    # components at -2, 2.
    with self.cached_session() as sess:
      dims = 1
      vdm = vdm_lib.VectorDiffeomixture(
          mix_loc=[0.],
          temperature=[[2.], [1.], [0.2]],
          distribution=normal_lib.Normal(0., 1.),
          loc=[
              np.float32([-2.]),
              np.float32([2.]),
          ],
          scale=[
              linop_identity_lib.LinearOperatorScaledIdentity(
                  num_rows=dims,
                  multiplier=np.float32(0.5),
                  is_positive_definite=True),
          ] * 2,  # Use the same scale for each component.
          quadrature_size=8,
          validate_args=True)

      samps = vdm.sample(10000)
      self.assertAllEqual((10000, 3, 1), samps.shape)
      samps_ = sess.run(samps).reshape(10000, 3)  # Make scalar event shape.

      # One characteristic of a discrete mixture (as opposed to a "smear") is
      # that more weight is put near the component centers at -2, 2, and thus
      # less weight is put near the origin.
      prob_of_being_near_origin = (np.abs(samps_) < 1).mean(axis=0)
      self.assertGreater(
          prob_of_being_near_origin[0], prob_of_being_near_origin[1])
      self.assertGreater(
          prob_of_being_near_origin[1], prob_of_being_near_origin[2])

      # Run this test as well, just because we can.
      self.run_test_sample_consistent_mean_covariance(
          sess.run, vdm, rtol=0.02, cov_rtol=0.08)

  def testConcentrationLocControlsHowMuchWeightIsOnEachComponent(self):
    with self.cached_session() as sess:
      dims = 1
      vdm = vdm_lib.VectorDiffeomixture(
          mix_loc=[[-1.], [0.], [1.]],
          temperature=[0.5],
          distribution=normal_lib.Normal(0., 1.),
          loc=[
              np.float32([-2.]),
              np.float32([2.]),
          ],
          scale=[
              linop_identity_lib.LinearOperatorScaledIdentity(
                  num_rows=dims,
                  multiplier=np.float32(0.5),
                  is_positive_definite=True),
          ] * 2,  # Use the same scale for each component.
          quadrature_size=8,
          validate_args=True)

      samps = vdm.sample(10000)
      self.assertAllEqual((10000, 3, 1), samps.shape)
      samps_ = sess.run(samps).reshape(10000, 3)  # Make scalar event shape.

      # One characteristic of putting more weight on a component is that the
      # mean is closer to that component's mean.
      # Get the mean for each batch member, the names signify the value of
      # concentration for that batch member.
      mean_neg1, mean_0, mean_1 = samps_.mean(axis=0)

      # Since concentration is the concentration for component 0,
      # concentration = -1 ==> more weight on component 1, which has mean = 2
      # concentration = 0 ==> equal weight
      # concentration = 1 ==> more weight on component 0, which has mean = -2
      self.assertLess(-2, mean_1)
      self.assertLess(mean_1, mean_0)
      self.assertLess(mean_0, mean_neg1)
      self.assertLess(mean_neg1, 2)

      # Run this test as well, just because we can.
      self.run_test_sample_consistent_mean_covariance(
          sess.run, vdm, rtol=0.02, cov_rtol=0.08)

  def testMeanCovarianceNoBatchUncenteredNonStandardBase(self):
    with self.cached_session() as sess:
      dims = 3
      vdm = vdm_lib.VectorDiffeomixture(
          mix_loc=[[0.], [4.]],
          temperature=[0.1],
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
    with self.cached_session() as sess:
      dims = 3
      vdm = vdm_lib.VectorDiffeomixture(
          mix_loc=[[0.], [4.]],
          temperature=[0.1],
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
    with self.cached_session() as sess:
      dims = 4
      vdm = vdm_lib.VectorDiffeomixture(
          mix_loc=[0.],
          temperature=[0.1],
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


if __name__ == "__main__":
  test.main()
