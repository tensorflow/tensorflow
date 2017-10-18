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
"""Tests for PoissonLogNormalQuadratureCompoundTest."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.distributions.python.ops import poisson_lognormal
from tensorflow.contrib.distributions.python.ops import test_util
from tensorflow.python.platform import test


class PoissonLogNormalQuadratureCompoundTest(
    test_util.DiscreteScalarDistributionTestHelpers, test.TestCase):
  """Tests the PoissonLogNormalQuadratureCompoundTest distribution."""

  def testSampleProbConsistent(self):
    with self.test_session() as sess:
      pln = poisson_lognormal.PoissonLogNormalQuadratureCompound(
          loc=-2.,
          scale=1.1,
          quadrature_grid_and_probs=(
              np.polynomial.hermite.hermgauss(deg=10)),
          validate_args=True)
      self.run_test_sample_consistent_log_prob(
          sess, pln, rtol=0.1)

  def testMeanVariance(self):
    with self.test_session() as sess:
      pln = poisson_lognormal.PoissonLogNormalQuadratureCompound(
          loc=0.,
          scale=1.,
          quadrature_grid_and_probs=(
              np.polynomial.hermite.hermgauss(deg=10)),
          validate_args=True)
      self.run_test_sample_consistent_mean_variance(
          sess, pln, rtol=0.02)

  def testSampleProbConsistentBroadcastScalar(self):
    with self.test_session() as sess:
      pln = poisson_lognormal.PoissonLogNormalQuadratureCompound(
          loc=[0., -0.5],
          scale=1.,
          quadrature_grid_and_probs=(
              np.polynomial.hermite.hermgauss(deg=10)),
          validate_args=True)
      self.run_test_sample_consistent_log_prob(
          sess, pln, rtol=0.1, atol=0.01)

  def testMeanVarianceBroadcastScalar(self):
    with self.test_session() as sess:
      pln = poisson_lognormal.PoissonLogNormalQuadratureCompound(
          loc=[0., -0.5],
          scale=1.,
          quadrature_grid_and_probs=(
              np.polynomial.hermite.hermgauss(deg=10)),
          validate_args=True)
      self.run_test_sample_consistent_mean_variance(
          sess, pln, rtol=0.1, atol=0.01)

  def testSampleProbConsistentBroadcastBoth(self):
    with self.test_session() as sess:
      pln = poisson_lognormal.PoissonLogNormalQuadratureCompound(
          loc=[[0.], [-0.5]],
          scale=[[1., 0.9]],
          quadrature_grid_and_probs=(
              np.polynomial.hermite.hermgauss(deg=10)),
          validate_args=True)
      self.run_test_sample_consistent_log_prob(
          sess, pln, rtol=0.1, atol=0.08)

  def testMeanVarianceBroadcastBoth(self):
    with self.test_session() as sess:
      pln = poisson_lognormal.PoissonLogNormalQuadratureCompound(
          loc=[[0.], [-0.5]],
          scale=[[1., 0.9]],
          quadrature_grid_and_probs=(
              np.polynomial.hermite.hermgauss(deg=10)),
          validate_args=True)
      self.run_test_sample_consistent_mean_variance(
          sess, pln, rtol=0.1, atol=0.01)


if __name__ == "__main__":
  test.main()
