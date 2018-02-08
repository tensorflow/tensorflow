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
"""Tests for MCMC diagnostic utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.bayesflow.python.ops import mcmc_diagnostics_impl as mcmc_diagnostics
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test

rng = np.random.RandomState(42)


class _PotentialScaleReductionTest(object):

  @property
  def use_static_shape(self):
    raise NotImplementedError(
        "Subclass failed to impliment `use_static_shape`.")

  def testListOfStatesWhereFirstPassesSecondFails(self):
    """Simple test showing API with two states.  Read first!."""
    n_samples = 1000

    # state_0 is two scalar chains taken from iid Normal(0, 1).  Will pass.
    state_0 = rng.randn(n_samples, 2)

    # state_1 is three 4-variate chains taken from Normal(0, 1) that have been
    # shifted.  Since every chain is shifted, they are not the same, and the
    # test should fail.
    offset = np.array([1., -1., 2.]).reshape(3, 1)
    state_1 = rng.randn(n_samples, 3, 4) + offset

    rhat = mcmc_diagnostics.potential_scale_reduction(
        state=[state_0, state_1], independent_chain_ndims=1)

    self.assertIsInstance(rhat, list)
    with self.test_session() as sess:
      rhat_0_, rhat_1_ = sess.run(rhat)

    # r_hat_0 should be close to 1, meaning test is passed.
    self.assertAllEqual((), rhat_0_.shape)
    self.assertAllClose(1., rhat_0_, rtol=0.02)

    # r_hat_1 should be greater than 1.2, meaning test has failed.
    self.assertAllEqual((4,), rhat_1_.shape)
    self.assertAllEqual(np.ones_like(rhat_1_).astype(bool), rhat_1_ > 1.2)

  def check_results(self, state_, independent_chain_shape, should_pass):
    sample_ndims = 1
    independent_chain_ndims = len(independent_chain_shape)
    with self.test_session():
      state = array_ops.placeholder_with_default(
          input=state_, shape=state_.shape if self.use_static_shape else None)

      rhat = mcmc_diagnostics.potential_scale_reduction(
          state, independent_chain_ndims=independent_chain_ndims)

      if self.use_static_shape:
        self.assertAllEqual(
            state_.shape[sample_ndims + independent_chain_ndims:], rhat.shape)

      rhat_ = rhat.eval()
      if should_pass:
        self.assertAllClose(np.ones_like(rhat_), rhat_, atol=0, rtol=0.02)
      else:
        self.assertAllEqual(np.ones_like(rhat_).astype(bool), rhat_ > 1.2)

  def iid_normal_chains_should_pass_wrapper(self,
                                            sample_shape,
                                            independent_chain_shape,
                                            other_shape,
                                            dtype=np.float32):
    """Check results with iid normal chains."""

    state_shape = sample_shape + independent_chain_shape + other_shape
    state_ = rng.randn(*state_shape).astype(dtype)

    # The "other" dimensions do not have to be identical, just independent, so
    # force them to not be identical.
    if other_shape:
      state_ *= rng.rand(*other_shape).astype(dtype)

    self.check_results(state_, independent_chain_shape, should_pass=True)

  def testPassingIIDNdimsAreIndependentOneOtherZero(self):
    self.iid_normal_chains_should_pass_wrapper(
        sample_shape=[10000], independent_chain_shape=[4], other_shape=[])

  def testPassingIIDNdimsAreIndependentOneOtherOne(self):
    self.iid_normal_chains_should_pass_wrapper(
        sample_shape=[10000], independent_chain_shape=[3], other_shape=[7])

  def testPassingIIDNdimsAreIndependentOneOtherTwo(self):
    self.iid_normal_chains_should_pass_wrapper(
        sample_shape=[10000], independent_chain_shape=[2], other_shape=[5, 7])

  def testPassingIIDNdimsAreIndependentTwoOtherTwo64Bit(self):
    self.iid_normal_chains_should_pass_wrapper(
        sample_shape=[10000],
        independent_chain_shape=[2, 3],
        other_shape=[5, 7],
        dtype=np.float64)

  def offset_normal_chains_should_fail_wrapper(
      self, sample_shape, independent_chain_shape, other_shape):
    """Check results with normal chains that are offset from each other."""

    state_shape = sample_shape + independent_chain_shape + other_shape
    state_ = rng.randn(*state_shape)

    # Add a significant offset to the different (formerly iid) chains.
    offset = np.linspace(
        0, 2, num=np.prod(independent_chain_shape)).reshape([1] * len(
            sample_shape) + independent_chain_shape + [1] * len(other_shape))
    state_ += offset

    self.check_results(state_, independent_chain_shape, should_pass=False)

  def testFailingOffsetNdimsAreSampleOneIndependentOneOtherOne(self):
    self.offset_normal_chains_should_fail_wrapper(
        sample_shape=[10000], independent_chain_shape=[2], other_shape=[5])


class PotentialScaleReductionStaticTest(test.TestCase,
                                        _PotentialScaleReductionTest):

  @property
  def use_static_shape(self):
    return True

  def testIndependentNdimsLessThanOneRaises(self):
    with self.assertRaisesRegexp(ValueError, "independent_chain_ndims"):
      mcmc_diagnostics.potential_scale_reduction(
          rng.rand(2, 3, 4), independent_chain_ndims=0)


class PotentialScaleReductionDynamicTest(test.TestCase,
                                         _PotentialScaleReductionTest):

  @property
  def use_static_shape(self):
    return False


class _ReduceVarianceTest(object):

  @property
  def use_static_shape(self):
    raise NotImplementedError(
        "Subclass failed to impliment `use_static_shape`.")

  def check_versus_numpy(self, x_, axis, biased, keepdims):
    with self.test_session():
      x_ = np.asarray(x_)
      x = array_ops.placeholder_with_default(
          input=x_, shape=x_.shape if self.use_static_shape else None)
      var = mcmc_diagnostics._reduce_variance(
          x, axis=axis, biased=biased, keepdims=keepdims)
      np_var = np.var(x_, axis=axis, ddof=0 if biased else 1, keepdims=keepdims)

      if self.use_static_shape:
        self.assertAllEqual(np_var.shape, var.shape)

      var_ = var.eval()
      # We will mask below, which changes shape, so check shape explicitly here.
      self.assertAllEqual(np_var.shape, var_.shape)

      # We get NaN when we divide by zero due to the size being the same as ddof
      nan_mask = np.isnan(np_var)
      if nan_mask.any():
        self.assertTrue(np.isnan(var_[nan_mask]).all())
      self.assertAllClose(np_var[~nan_mask], var_[~nan_mask], atol=0, rtol=0.02)

  def testScalarBiasedTrue(self):
    self.check_versus_numpy(x_=-1.234, axis=None, biased=True, keepdims=False)

  def testScalarBiasedFalse(self):
    # This should result in NaN.
    self.check_versus_numpy(x_=-1.234, axis=None, biased=False, keepdims=False)

  def testShape2x3x4AxisNoneBiasedFalseKeepdimsFalse(self):
    self.check_versus_numpy(
        x_=rng.randn(2, 3, 4), axis=None, biased=True, keepdims=False)

  def testShape2x3x4Axis1BiasedFalseKeepdimsTrue(self):
    self.check_versus_numpy(
        x_=rng.randn(2, 3, 4), axis=1, biased=True, keepdims=True)

  def testShape2x3x4x5Axis13BiasedFalseKeepdimsTrue(self):
    self.check_versus_numpy(
        x_=rng.randn(2, 3, 4, 5), axis=1, biased=True, keepdims=True)

  def testShape2x3x4x5Axis13BiasedFalseKeepdimsFalse(self):
    self.check_versus_numpy(
        x_=rng.randn(2, 3, 4, 5), axis=1, biased=False, keepdims=False)


class ReduceVarianceTestStaticShape(test.TestCase, _ReduceVarianceTest):

  @property
  def use_static_shape(self):
    return True


class ReduceVarianceTestDynamicShape(test.TestCase, _ReduceVarianceTest):

  @property
  def use_static_shape(self):
    return False


if __name__ == "__main__":
  test.main()
