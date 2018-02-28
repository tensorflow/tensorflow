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
from tensorflow.python.ops import spectral_ops_test_util
from tensorflow.python.platform import test

rng = np.random.RandomState(42)


class _EffectiveSampleSizeTest(object):

  @property
  def use_static_shape(self):
    raise NotImplementedError(
        "Subclass failed to implement `use_static_shape`.")

  def _check_versus_expected_effective_sample_size(self,
                                                   x_,
                                                   expected_ess,
                                                   sess,
                                                   atol=1e-2,
                                                   rtol=1e-2,
                                                   filter_threshold=None,
                                                   filter_beyond_lag=None):
    x = array_ops.placeholder_with_default(
        input=x_, shape=x_.shape if self.use_static_shape else None)
    ess = mcmc_diagnostics.effective_sample_size(
        x,
        filter_threshold=filter_threshold,
        filter_beyond_lag=filter_beyond_lag)
    if self.use_static_shape:
      self.assertAllEqual(x.shape[1:], ess.shape)

    ess_ = sess.run(ess)

    self.assertAllClose(
        np.ones_like(ess_) * expected_ess, ess_, atol=atol, rtol=rtol)

  def testIidRank1NormalHasFullEssMaxLags10(self):
    # With a length 5000 iid normal sequence, and filter_beyond_lag = 10, we
    # should have a good estimate of ESS, and it should be close to the full
    # sequence length of 5000.
    # The choice of filter_beyond_lag = 10 is a short cutoff, reasonable only
    # since we know the correlation length should be zero right away.
    with self.test_session() as sess:
      with spectral_ops_test_util.fft_kernel_label_map():
        self._check_versus_expected_effective_sample_size(
            x_=rng.randn(5000).astype(np.float32),
            expected_ess=5000,
            sess=sess,
            filter_beyond_lag=10,
            filter_threshold=None,
            rtol=0.3)

  def testIidRank2NormalHasFullEssMaxLags10(self):
    # See similar test for Rank1Normal for reasoning.
    with self.test_session() as sess:
      with spectral_ops_test_util.fft_kernel_label_map():
        self._check_versus_expected_effective_sample_size(
            x_=rng.randn(5000, 2).astype(np.float32),
            expected_ess=5000,
            sess=sess,
            filter_beyond_lag=10,
            filter_threshold=None,
            rtol=0.3)

  def testIidRank1NormalHasFullEssMaxLagThresholdZero(self):
    # With a length 5000 iid normal sequence, and filter_threshold = 0,
    # we should have a super-duper estimate of ESS, and it should be very close
    # to the full sequence length of 5000.
    # The choice of filter_beyond_lag = 0 means we cutoff as soon as the
    # auto-corris below zero.  This should happen very quickly, due to the fact
    # that the theoretical auto-corr is [1, 0, 0,...]
    with self.test_session() as sess:
      with spectral_ops_test_util.fft_kernel_label_map():
        self._check_versus_expected_effective_sample_size(
            x_=rng.randn(5000).astype(np.float32),
            expected_ess=5000,
            sess=sess,
            filter_beyond_lag=None,
            filter_threshold=0.,
            rtol=0.1)

  def testIidRank2NormalHasFullEssMaxLagThresholdZero(self):
    # See similar test for Rank1Normal for reasoning.
    with self.test_session() as sess:
      with spectral_ops_test_util.fft_kernel_label_map():
        self._check_versus_expected_effective_sample_size(
            x_=rng.randn(5000, 2).astype(np.float32),
            expected_ess=5000,
            sess=sess,
            filter_beyond_lag=None,
            filter_threshold=0.,
            rtol=0.1)

  def testLength10CorrelationHasEssOneTenthTotalLengthUsingMaxLags50(self):
    # Create x_, such that
    #   x_[i] = iid_x_[0], i = 0,...,9
    #   x_[i] = iid_x_[1], i = 10,..., 19,
    #   and so on.
    iid_x_ = rng.randn(5000, 1).astype(np.float32)
    x_ = (iid_x_ * np.ones((5000, 10)).astype(np.float32)).reshape((50000,))
    with self.test_session() as sess:
      with spectral_ops_test_util.fft_kernel_label_map():
        self._check_versus_expected_effective_sample_size(
            x_=x_,
            expected_ess=50000 // 10,
            sess=sess,
            filter_beyond_lag=50,
            filter_threshold=None,
            rtol=0.2)

  def testLength10CorrelationHasEssOneTenthTotalLengthUsingMaxLagsThresholdZero(
      self):
    # Create x_, such that
    #   x_[i] = iid_x_[0], i = 0,...,9
    #   x_[i] = iid_x_[1], i = 10,..., 19,
    #   and so on.
    iid_x_ = rng.randn(5000, 1).astype(np.float32)
    x_ = (iid_x_ * np.ones((5000, 10)).astype(np.float32)).reshape((50000,))
    with self.test_session() as sess:
      with spectral_ops_test_util.fft_kernel_label_map():
        self._check_versus_expected_effective_sample_size(
            x_=x_,
            expected_ess=50000 // 10,
            sess=sess,
            filter_beyond_lag=None,
            filter_threshold=0.,
            rtol=0.1)

  def testListArgs(self):
    # x_ has correlation length 10 ==> ESS = N / 10
    # y_ has correlation length 1  ==> ESS = N
    iid_x_ = rng.randn(5000, 1).astype(np.float32)
    x_ = (iid_x_ * np.ones((5000, 10)).astype(np.float32)).reshape((50000,))
    y_ = rng.randn(50000).astype(np.float32)
    states = [x_, x_, y_, y_]
    filter_threshold = [0., None, 0., None]
    filter_beyond_lag = [None, 5, None, 5]

    # See other tests for reasoning on tolerance.
    with self.test_session() as sess:
      with spectral_ops_test_util.fft_kernel_label_map():
        ess = mcmc_diagnostics.effective_sample_size(
            states,
            filter_threshold=filter_threshold,
            filter_beyond_lag=filter_beyond_lag)
        ess_ = sess.run(ess)
    self.assertAllEqual(4, len(ess_))

    self.assertAllClose(50000 // 10, ess_[0], rtol=0.3)
    self.assertAllClose(50000 // 10, ess_[1], rtol=0.3)
    self.assertAllClose(50000, ess_[2], rtol=0.1)
    self.assertAllClose(50000, ess_[3], rtol=0.1)

  def testMaxLagsThresholdLessThanNeg1SameAsNone(self):
    # Setting both means we filter out items R_k from the auto-correlation
    # sequence if k > filter_beyond_lag OR k >= j where R_j < filter_threshold.

    # x_ has correlation length 10.
    iid_x_ = rng.randn(500, 1).astype(np.float32)
    x_ = (iid_x_ * np.ones((500, 10)).astype(np.float32)).reshape((5000,))
    with self.test_session() as sess:
      with spectral_ops_test_util.fft_kernel_label_map():
        x = array_ops.placeholder_with_default(
            input=x_, shape=x_.shape if self.use_static_shape else None)

        ess_none_none = mcmc_diagnostics.effective_sample_size(
            x, filter_threshold=None, filter_beyond_lag=None)
        ess_none_200 = mcmc_diagnostics.effective_sample_size(
            x, filter_threshold=None, filter_beyond_lag=200)
        ess_neg2_200 = mcmc_diagnostics.effective_sample_size(
            x, filter_threshold=-2., filter_beyond_lag=200)
        ess_neg2_none = mcmc_diagnostics.effective_sample_size(
            x, filter_threshold=-2., filter_beyond_lag=None)
        ess_none_none_, ess_none_200_, ess_neg2_200_, ess_neg2_none_ = sess.run(
            [ess_none_none, ess_none_200, ess_neg2_200, ess_neg2_none])

        # filter_threshold=-2 <==> filter_threshold=None.
        self.assertAllClose(ess_none_none_, ess_neg2_none_)
        self.assertAllClose(ess_none_200_, ess_neg2_200_)

  def testMaxLagsArgsAddInAnOrManner(self):
    # Setting both means we filter out items R_k from the auto-correlation
    # sequence if k > filter_beyond_lag OR k >= j where R_j < filter_threshold.

    # x_ has correlation length 10.
    iid_x_ = rng.randn(500, 1).astype(np.float32)
    x_ = (iid_x_ * np.ones((500, 10)).astype(np.float32)).reshape((5000,))
    with self.test_session() as sess:
      with spectral_ops_test_util.fft_kernel_label_map():
        x = array_ops.placeholder_with_default(
            input=x_, shape=x_.shape if self.use_static_shape else None)

        ess_1_9 = mcmc_diagnostics.effective_sample_size(
            x, filter_threshold=1., filter_beyond_lag=9)
        ess_1_none = mcmc_diagnostics.effective_sample_size(
            x, filter_threshold=1., filter_beyond_lag=None)
        ess_none_9 = mcmc_diagnostics.effective_sample_size(
            x, filter_threshold=1., filter_beyond_lag=9)
        ess_1_9_, ess_1_none_, ess_none_9_ = sess.run(
            [ess_1_9, ess_1_none, ess_none_9])

        # Since R_k = 1 for k < 10, and R_k < 1 for k >= 10,
        # filter_threshold = 1 <==> filter_beyond_lag = 9.
        self.assertAllClose(ess_1_9_, ess_1_none_)
        self.assertAllClose(ess_1_9_, ess_none_9_)


class EffectiveSampleSizeStaticTest(test.TestCase, _EffectiveSampleSizeTest):

  @property
  def use_static_shape(self):
    return True


class EffectiveSampleSizeDynamicTest(test.TestCase, _EffectiveSampleSizeTest):

  @property
  def use_static_shape(self):
    return False


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
        chains_states=[state_0, state_1], independent_chain_ndims=1)

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
