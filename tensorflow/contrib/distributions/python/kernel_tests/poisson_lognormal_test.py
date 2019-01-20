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

from tensorflow.contrib.distributions.python.ops import poisson_lognormal
from tensorflow.contrib.distributions.python.ops import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


class _PoissonLogNormalQuadratureCompoundTest(
    test_util.DiscreteScalarDistributionTestHelpers):
  """Tests the PoissonLogNormalQuadratureCompoundTest distribution."""

  def testSampleProbConsistent(self):
    with self.cached_session() as sess:
      pln = poisson_lognormal.PoissonLogNormalQuadratureCompound(
          loc=array_ops.placeholder_with_default(
              -2.,
              shape=[] if self.static_shape else None),
          scale=array_ops.placeholder_with_default(
              1.1,
              shape=[] if self.static_shape else None),
          quadrature_size=10,
          validate_args=True)
      self.run_test_sample_consistent_log_prob(
          sess.run, pln, batch_size=1, rtol=0.1)

  def testMeanVariance(self):
    with self.cached_session() as sess:
      pln = poisson_lognormal.PoissonLogNormalQuadratureCompound(
          loc=array_ops.placeholder_with_default(
              0.,
              shape=[] if self.static_shape else None),
          scale=array_ops.placeholder_with_default(
              1.,
              shape=[] if self.static_shape else None),
          quadrature_size=10,
          validate_args=True)
      self.run_test_sample_consistent_mean_variance(
          sess.run, pln, rtol=0.02)

  def testSampleProbConsistentBroadcastScalar(self):
    with self.cached_session() as sess:
      pln = poisson_lognormal.PoissonLogNormalQuadratureCompound(
          loc=array_ops.placeholder_with_default(
              [0., -0.5],
              shape=[2] if self.static_shape else None),
          scale=array_ops.placeholder_with_default(
              1.,
              shape=[] if self.static_shape else None),
          quadrature_size=10,
          validate_args=True)
      self.run_test_sample_consistent_log_prob(
          sess.run, pln, batch_size=2, rtol=0.1, atol=0.01)

  def testMeanVarianceBroadcastScalar(self):
    with self.cached_session() as sess:
      pln = poisson_lognormal.PoissonLogNormalQuadratureCompound(
          loc=array_ops.placeholder_with_default(
              [0., -0.5],
              shape=[2] if self.static_shape else None),
          scale=array_ops.placeholder_with_default(
              1.,
              shape=[] if self.static_shape else None),
          quadrature_size=10,
          validate_args=True)
      self.run_test_sample_consistent_mean_variance(
          sess.run, pln, rtol=0.1, atol=0.01)

  def testSampleProbConsistentBroadcastBoth(self):
    with self.cached_session() as sess:
      pln = poisson_lognormal.PoissonLogNormalQuadratureCompound(
          loc=array_ops.placeholder_with_default(
              [[0.], [-0.5]],
              shape=[2, 1] if self.static_shape else None),
          scale=array_ops.placeholder_with_default(
              [[1., 0.9]],
              shape=[1, 2] if self.static_shape else None),
          quadrature_size=10,
          validate_args=True)
      self.run_test_sample_consistent_log_prob(
          sess.run, pln, batch_size=4, rtol=0.1, atol=0.08)

  def testMeanVarianceBroadcastBoth(self):
    with self.cached_session() as sess:
      pln = poisson_lognormal.PoissonLogNormalQuadratureCompound(
          loc=array_ops.placeholder_with_default(
              [[0.], [-0.5]],
              shape=[2, 1] if self.static_shape else None),
          scale=array_ops.placeholder_with_default(
              [[1., 0.9]],
              shape=[1, 2] if self.static_shape else None),
          quadrature_size=10,
          validate_args=True)
      self.run_test_sample_consistent_mean_variance(
          sess.run, pln, rtol=0.1, atol=0.01)


class PoissonLogNormalQuadratureCompoundStaticShapeTest(
    _PoissonLogNormalQuadratureCompoundTest, test.TestCase):

  @property
  def static_shape(self):
    return True


class PoissonLogNormalQuadratureCompoundDynamicShapeTest(
    _PoissonLogNormalQuadratureCompoundTest, test.TestCase):

  @property
  def static_shape(self):
    return False


if __name__ == "__main__":
  test.main()
