# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for correlation_matrix_volumes_lib.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.distributions.python.kernel_tests.util import correlation_matrix_volumes_lib as corr
from tensorflow.contrib.distributions.python.ops import statistical_testing as st
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.platform import test


# NxN correlation matrices are determined by the N*(N-1)/2
# lower-triangular entries.  In addition to being between -1 and 1,
# they must also obey the constraint that the determinant of the
# resulting symmetric matrix is non-negative.  In 2x2, we can even
# analytically compute the volume when the determinant is bounded to >
# epsilon, as that boils down to the one lower-triangular entry being
# less than 1 - epsilon in absolute value.
def two_by_two_volume(det_bound):
  return 2 * np.sqrt(1.0 - det_bound)


# The post
# https://psychometroscar.com/the-volume-of-a-3-x-3-correlation-matrix/
# derives (with elementary calculus) that the volume (with respect to
# Lebesgue^3 measure) of the set of 3x3 correlation matrices is
# pi^2/2.  The same result is also obtained by [1].
def three_by_three_volume():
  return np.pi**2 / 2.


# The volume of the unconstrained set of correlation matrices is also
# the normalization constant of the LKJ distribution from [2].  As
# part of defining the distribution, that reference a derives general
# formula for this volume for all dimensions.  A TensorFlow
# computation thereof gave the below result for 4x4:
def four_by_four_volume():
  # This constant computed as math_ops.exp(lkj.log_norm_const(4, [1.0]))
  return 11.6973076

# [1] Rousseeuw, P. J., & Molenberghs, G. (1994). "The shape of
# correlation matrices." The American Statistician, 48(4), 276-279.

# [2] Daniel Lewandowski, Dorota Kurowicka, and Harry Joe, "Generating
# random correlation matrices based on vines and extended onion
# method," Journal of Multivariate Analysis 100 (2009), pp 1989-2001.


class CorrelationMatrixVolumesTest(test.TestCase):

  def testRejection2D(self):
    num_samples = int(1e5)  # Chosen for a small min detectable discrepancy
    det_bounds = np.array(
        [0.01, 0.02, 0.03, 0.04, 0.05, 0.3, 0.35, 0.4, 0.5], dtype=np.float32)
    exact_volumes = two_by_two_volume(det_bounds)
    (rej_weights,
     rej_proposal_volume) = corr.correlation_matrix_volume_rejection_samples(
         det_bounds, 2, [num_samples, 9], dtype=np.float32, seed=43)
    # shape of rej_weights: [num_samples, 9, 2, 2]
    chk1 = st.assert_true_mean_equal_by_dkwm(
        rej_weights, low=0., high=rej_proposal_volume, expected=exact_volumes,
        false_fail_rate=1e-6)
    chk2 = check_ops.assert_less(
        st.min_discrepancy_of_true_means_detectable_by_dkwm(
            num_samples, low=0., high=rej_proposal_volume,
            # Correct the false fail rate due to different broadcasting
            false_fail_rate=1.1e-7, false_pass_rate=1e-6),
        0.036)
    with ops.control_dependencies([chk1, chk2]):
      rej_weights = array_ops.identity(rej_weights)
    self.evaluate(rej_weights)

  def testRejection3D(self):
    num_samples = int(1e5)  # Chosen for a small min detectable discrepancy
    det_bounds = np.array([0.0], dtype=np.float32)
    exact_volumes = np.array([three_by_three_volume()], dtype=np.float32)
    (rej_weights,
     rej_proposal_volume) = corr.correlation_matrix_volume_rejection_samples(
         det_bounds, 3, [num_samples, 1], dtype=np.float32, seed=44)
    # shape of rej_weights: [num_samples, 1, 3, 3]
    chk1 = st.assert_true_mean_equal_by_dkwm(
        rej_weights, low=0., high=rej_proposal_volume, expected=exact_volumes,
        false_fail_rate=1e-6)
    chk2 = check_ops.assert_less(
        st.min_discrepancy_of_true_means_detectable_by_dkwm(
            num_samples, low=0., high=rej_proposal_volume,
            false_fail_rate=1e-6, false_pass_rate=1e-6),
        # Going for about a 3% relative error
        0.15)
    with ops.control_dependencies([chk1, chk2]):
      rej_weights = array_ops.identity(rej_weights)
    self.evaluate(rej_weights)

  def testRejection4D(self):
    num_samples = int(1e5)  # Chosen for a small min detectable discrepancy
    det_bounds = np.array([0.0], dtype=np.float32)
    exact_volumes = [four_by_four_volume()]
    (rej_weights,
     rej_proposal_volume) = corr.correlation_matrix_volume_rejection_samples(
         det_bounds, 4, [num_samples, 1], dtype=np.float32, seed=45)
    # shape of rej_weights: [num_samples, 1, 4, 4]
    chk1 = st.assert_true_mean_equal_by_dkwm(
        rej_weights, low=0., high=rej_proposal_volume, expected=exact_volumes,
        false_fail_rate=1e-6)
    chk2 = check_ops.assert_less(
        st.min_discrepancy_of_true_means_detectable_by_dkwm(
            num_samples, low=0., high=rej_proposal_volume,
            false_fail_rate=1e-6, false_pass_rate=1e-6),
        # Going for about a 10% relative error
        1.1)
    with ops.control_dependencies([chk1, chk2]):
      rej_weights = array_ops.identity(rej_weights)
    self.evaluate(rej_weights)

  def testVolumeEstimation2D(self):
    # Test that the confidence intervals produced by
    # corr.compte_true_volumes are sound, in the sense of containing
    # the exact volume.
    num_samples = int(1e5)  # Chosen by symmetry with testRejection2D
    det_bounds = np.array(
        [0.01, 0.02, 0.03, 0.04, 0.05, 0.3, 0.35, 0.4, 0.5], dtype=np.float32)
    volume_bounds = corr.compute_true_volumes(
        det_bounds, 2, num_samples, error_rate=1e-6, seed=47)
    exact_volumes = two_by_two_volume(det_bounds)
    for det, volume in zip(det_bounds, exact_volumes):
      computed_low, computed_high = volume_bounds[det]
      self.assertLess(computed_low, volume)
      self.assertGreater(computed_high, volume)

if __name__ == "__main__":
  test.main()
