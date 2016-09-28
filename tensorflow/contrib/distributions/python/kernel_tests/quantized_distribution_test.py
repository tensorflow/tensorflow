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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy import stats
import tensorflow as tf

distributions = tf.contrib.distributions


class QuantizedDistributionTest(tf.test.TestCase):

  def setUp(self):
    self._rng = np.random.RandomState(0)

  def _assert_all_finite(self, array):
    self.assertTrue(np.isfinite(array).all())

  def test_quantization_of_uniform_with_cutoffs_having_no_effect(self):
    with self.test_session() as sess:
      # The Quantized uniform with cutoffs == None divides the real line into:
      # R = ...(-1, 0](0, 1](1, 2](2, 3](3, 4]...
      # j = ...     0     1     2     3     4 ...
      # Since this uniform (below) is supported on [0, 3],
      # it places 1/3 of its mass in the intervals j = 1, 2, 3.
      # Adding a cutoff at y = 0 changes the picture to
      # R = ...(-inf, 0](0, 1](1, 2](2, 3](3, 4]...
      # j = ...     0     1     2     3     4 ...
      # So the QUniform still places 1/3 of its mass in the intervals
      # j = 1, 2, 3.
      # Adding a cutoff at y = 3 changes the picture to
      # R = ...(-1, 0](0, 1](1, 2](2, inf)
      # j = ...     0     1     2     3
      # and the QUniform still places 1/3 of its mass in the intervals
      # j = 1, 2, 3.
      for lcut, ucut in [
          (None, None), (0.0, None), (None, 3.0), (0.0, 3.0), (-10., 10.)
      ]:
        qdist = distributions.QuantizedDistribution(
            base_dist_cls=distributions.Uniform,
            lower_cutoff=lcut,
            upper_cutoff=ucut,
            a=0.0,
            b=3.0)

        # pmf
        pmf_n1, pmf_0, pmf_1, pmf_2, pmf_3, pmf_4, pmf_5 = sess.run(
            qdist.pmf([-1., 0., 1., 2., 3., 4., 5.]))
        # uniform had no mass below -1.
        self.assertAllClose(0., pmf_n1)
        # uniform had no mass below 0.
        self.assertAllClose(0., pmf_0)
        # uniform put 1/3 of its mass in each of (0, 1], (1, 2], (2, 3],
        # which are the intervals j = 1, 2, 3.
        self.assertAllClose(1 / 3, pmf_1)
        self.assertAllClose(1 / 3, pmf_2)
        self.assertAllClose(1 / 3, pmf_3)
        # uniform had no mass in (3, 4] or (4, 5], which are j = 4, 5.
        self.assertAllClose(0 / 3, pmf_4)
        self.assertAllClose(0 / 3, pmf_5)

        # cdf
        cdf_n1, cdf_0, cdf_1, cdf_2, cdf_2p5, cdf_3, cdf_4, cdf_5 = sess.run(
            qdist.cdf([-1., 0., 1., 2., 2.5, 3., 4., 5.]))
        self.assertAllClose(0., cdf_n1)
        self.assertAllClose(0., cdf_0)
        self.assertAllClose(1 / 3, cdf_1)
        self.assertAllClose(2 / 3, cdf_2)
        # Note fractional values allowed for cdfs of discrete distributions.
        # And adding 0.5 makes no difference because the quantized dist has
        # mass only on the integers, never in between.
        self.assertAllClose(2 / 3, cdf_2p5)
        self.assertAllClose(3 / 3, cdf_3)
        self.assertAllClose(3 / 3, cdf_4)
        self.assertAllClose(3 / 3, cdf_5)

  def test_quantization_of_uniform_with_cutoffs_in_the_middle(self):
    with self.test_session() as sess:
      # The uniform is supported on [-3, 3]
      # Consider partitions the real line in intervals
      # ...(-3, -2](-2, -1](-1, 0](0, 1](1, 2](2, 3] ...
      # Before cutoffs, the uniform puts a mass of 1/6 in each interval written
      # above.  Because of cutoffs, the qdist considers intervals and indices
      # ...(-infty, -1](-1, 0](0, infty) ...
      #             -1      0     1
      qdist = distributions.QuantizedDistribution(
          base_dist_cls=distributions.Uniform,
          lower_cutoff=-1.0,
          upper_cutoff=1.0,
          a=-3.0,
          b=3.0)

      # pmf
      cdf_n3, cdf_n2, cdf_n1, cdf_0, cdf_0p5, cdf_1, cdf_10 = sess.run(
          qdist.cdf([-3., -2., -1., 0., 0.5, 1.0, 10.0]))
      # Uniform had no mass on (-4, -3] or (-3, -2]
      self.assertAllClose(0., cdf_n3)
      self.assertAllClose(0., cdf_n2)
      # Uniform had 1/6 of its mass in each of (-3, -2], and (-2, -1], which
      # were collapsed into (-infty, -1], which is now the "-1" interval.
      self.assertAllClose(1 / 3, cdf_n1)
      # The j=0 interval contained mass from (-3, 0], which is 1/2 of the
      # uniform's mass.
      self.assertAllClose(1 / 2, cdf_0)
      # Adding 0.5 makes no difference because the quantized dist has mass on
      # the integers, not in between them.
      self.assertAllClose(1 / 2, cdf_0p5)
      # After applying the cutoff, all mass was either in the interval
      # (0, infty), or below.  (0, infty) is the interval indexed by j=1,
      # so pmf(1) should equal 1.
      self.assertAllClose(1., cdf_1)
      # Since no mass of qdist is above 1,
      # pmf(10) = P[Y <= 10] = P[Y <= 1] = pmf(1).
      self.assertAllClose(1., cdf_10)

  def test_quantization_of_batch_of_uniforms(self):
    batch_shape = (5, 5)
    with self.test_session():
      # The uniforms are supported on [0, 10].  The qdist considers the
      # intervals
      # ... (0, 1](1, 2]...(9, 10]...
      # with the intervals displayed above each holding 1 / 10 of the mass.
      # The qdist will be defined with no cutoffs,
      qdist = distributions.QuantizedDistribution(
          base_dist_cls=distributions.Uniform,
          lower_cutoff=None,
          upper_cutoff=None,
          a=tf.zeros(
              batch_shape, dtype=tf.float32),
          b=10 * tf.ones(
              batch_shape, dtype=tf.float32))

      # x is random integers in {-3,...,12}.
      x = self._rng.randint(-3, 13, size=batch_shape).astype(np.float32)

      # pmf
      # qdist.pmf(j) = 1 / 10 for j in {1,...,10}, and 0 otherwise,
      expected_pmf = (1 / 10) * np.ones(batch_shape)
      expected_pmf[x < 1] = 0.
      expected_pmf[x > 10] = 0.
      self.assertAllClose(expected_pmf, qdist.pmf(x).eval())

      # cdf
      # qdist.cdf(j)
      #    = 0 for j < 1
      #    = j / 10, for j in {1,...,10},
      #    = 1, for j > 10.
      expected_cdf = x.copy() / 10
      expected_cdf[x < 1] = 0.
      expected_cdf[x > 10] = 1.
      self.assertAllClose(expected_cdf, qdist.cdf(x).eval())

  def test_sampling_from_batch_of_normals(self):
    batch_shape = (2,)
    with self.test_session():
      qdist = distributions.QuantizedDistribution(
          base_dist_cls=distributions.Normal,
          lower_cutoff=0.,
          upper_cutoff=None,
          mu=tf.zeros(
              batch_shape, dtype=tf.float32),
          sigma=tf.ones(
              batch_shape, dtype=tf.float32))

      samps = qdist.sample_n(n=5000, seed=42)
      samps_v = samps.eval()

      # With lower_cutoff = 0, the interval j=0 is (-infty, 0], which holds 1/2
      # of the mass of the normals.
      # rtol chosen to be 2x as large as necessary to pass.
      self.assertAllClose([0.5, 0.5], (samps_v == 0).mean(axis=0), rtol=0.03)

      # The interval j=1 is (0, 1], which is from the mean to one standard
      # deviation out.  This should contain 0.6827 / 2 of the mass.
      self.assertAllClose(
          [0.6827 / 2, 0.6827 / 2], (samps_v == 1).mean(axis=0), rtol=0.03)

  def test_samples_agree_with_cdf_for_samples_over_large_range(self):
    # Consider the cdf for distribution X, F(x).
    # If U ~ Uniform[0, 1], then Y := F^{-1}(U) is distributed like X since
    # P[Y <= y] = P[F^{-1}(U) <= y] = P[U <= F(y)] = F(y).
    # If F is a bijection, we also have Z = F(X) is Uniform.
    #
    # Make an exponential with large mean (= 100).  This ensures we will get
    # quantized values over a large range.  This large range allows us to
    # pretend that the cdf F is a bijection, and hence F(X) is uniform.
    # Note that F cannot be bijection since it is constant between the
    # integers.  Hence, F(X) (see below) will not be uniform exactly.
    with self.test_session():
      qdist = distributions.QuantizedDistribution(
          base_dist_cls=distributions.Exponential,
          lam=0.01)
      # X ~ QuantizedExponential
      x = qdist.sample_n(n=10000, seed=42)
      # Z = F(X), should be Uniform.
      z = qdist.cdf(x)
      # Compare the CDF of Z to that of a Uniform.
      # dist = maximum distance between P[Z <= a] and P[U <= a].
      # We ignore pvalue, since of course this distribution is not exactly, and
      # with so many sample points we would get a false fail.
      dist, _ = stats.kstest(z.eval(), "uniform")

      # Since the distribution take values (approximately) in [0, 100], the
      # cdf should have jumps (approximately) every 1/100 of the way up.
      # Assert that the jumps are not more than 2/100.
      self.assertLess(dist, 0.02)

  def test_samples_agree_with_pdf_for_samples_over_small_range(self):
    # Testing that samples and pdf agree for a small range is important because
    # it makes sure the bin edges are consistent.

    # Make an exponential with mean 5.
    with self.test_session():
      qdist = distributions.QuantizedDistribution(
          base_dist_cls=distributions.Exponential,
          lam=0.2)
      # Standard error should be less than 1 / (2 * sqrt(n_samples))
      n_samples = 10000
      std_err_bound = 1 / (2 * np.sqrt(n_samples))
      samps = qdist.sample((n_samples,), seed=42).eval()
      # The smallest value the samples can take on is 1, which corresponds to
      # the interval (0, 1].  Recall we use ceiling in the sampling definition.
      self.assertLess(0.5, samps.min())
      x_vals = np.arange(1, 11).astype(np.float32)
      pmf_vals = qdist.pmf(x_vals).eval()
      for ii in range(10):
        self.assertAllClose(
            pmf_vals[ii],
            (samps == x_vals[ii]).mean(),
            atol=std_err_bound)

  def test_normal_cdf_and_survival_function(self):
    # At integer values, the result should be the same as the standard normal.
    batch_shape = (3, 3)
    mu = self._rng.randn(*batch_shape)
    sigma = self._rng.rand(*batch_shape) + 1.0
    with self.test_session():
      qdist = distributions.QuantizedDistribution(
          base_dist_cls=distributions.Normal,
          mu=mu,
          sigma=sigma)
      sp_normal = stats.norm(mu, sigma)

      x = self._rng.randint(-5, 5, size=batch_shape).astype(np.float64)

      self.assertAllClose(
          sp_normal.cdf(x),
          qdist.cdf(x).eval())

      self.assertAllClose(
          sp_normal.sf(x),
          qdist.survival_function(x).eval())

  def test_normal_log_cdf_and_log_survival_function(self):
    # At integer values, the result should be the same as the standard normal.
    batch_shape = (3, 3)
    mu = self._rng.randn(*batch_shape)
    sigma = self._rng.rand(*batch_shape) + 1.0
    with self.test_session():
      qdist = distributions.QuantizedDistribution(
          base_dist_cls=distributions.Normal,
          mu=mu,
          sigma=sigma)
      sp_normal = stats.norm(mu, sigma)

      x = self._rng.randint(-10, 10, size=batch_shape).astype(np.float64)

      self.assertAllClose(
          sp_normal.logcdf(x),
          qdist.log_cdf(x).eval())

      self.assertAllClose(
          sp_normal.logsf(x),
          qdist.log_survival_function(x).eval())

  def test_normal_prob_with_cutoffs(self):
    # At integer values, the result should be the same as the standard normal.
    with self.test_session():
      qdist = distributions.QuantizedDistribution(
          base_dist_cls=distributions.Normal,
          mu=0.,
          sigma=1.,
          lower_cutoff=-2.,
          upper_cutoff=2.)
      sm_normal = stats.norm(0., 1.)
      # These cutoffs create partitions of the real line, and indices:
      # (-inf, -2](-2, -1](-1, 0](0, 1](1, inf)
      #        -2      -1      0     1     2
      # Test interval (-inf, -2], <--> index -2.
      self.assertAllClose(
          sm_normal.cdf(-2),
          qdist.prob(-2.).eval(),
          atol=0)
      # Test interval (-2, -1], <--> index -1.
      self.assertAllClose(
          sm_normal.cdf(-1) - sm_normal.cdf(-2),
          qdist.prob(-1.).eval(),
          atol=0)
      # Test interval (-1, 0], <--> index 0.
      self.assertAllClose(
          sm_normal.cdf(0) - sm_normal.cdf(-1),
          qdist.prob(0.).eval(),
          atol=0)
      # Test interval (1, inf), <--> index 2.
      self.assertAllClose(
          1. - sm_normal.cdf(1),
          qdist.prob(2.).eval(),
          atol=0)

  def test_normal_log_prob_with_cutoffs(self):
    # At integer values, the result should be the same as the standard normal.
    with self.test_session():
      qdist = distributions.QuantizedDistribution(
          base_dist_cls=distributions.Normal,
          mu=0.,
          sigma=1.,
          lower_cutoff=-2.,
          upper_cutoff=2.)
      sm_normal = stats.norm(0., 1.)
      # These cutoffs create partitions of the real line, and indices:
      # (-inf, -2](-2, -1](-1, 0](0, 1](1, inf)
      #        -2      -1      0     1     2
      # Test interval (-inf, -2], <--> index -2.
      self.assertAllClose(
          np.log(sm_normal.cdf(-2)),
          qdist.log_prob(-2.).eval(),
          atol=0)
      # Test interval (-2, -1], <--> index -1.
      self.assertAllClose(
          np.log(sm_normal.cdf(-1) - sm_normal.cdf(-2)),
          qdist.log_prob(-1.).eval(),
          atol=0)
      # Test interval (-1, 0], <--> index 0.
      self.assertAllClose(
          np.log(sm_normal.cdf(0) - sm_normal.cdf(-1)),
          qdist.log_prob(0.).eval(),
          atol=0)
      # Test interval (1, inf), <--> index 2.
      self.assertAllClose(
          np.log(1. - sm_normal.cdf(1)),
          qdist.log_prob(2.).eval(),
          atol=0)

  def test_log_prob_and_grad_gives_finite_results(self):
    with self.test_session():
      for dtype in [np.float32, np.float64]:
        mu = tf.Variable(0., name="mu", dtype=dtype)
        sigma = tf.Variable(1., name="sigma", dtype=dtype)
        qdist = distributions.QuantizedDistribution(
            base_dist_cls=distributions.Normal,
            mu=mu,
            sigma=sigma)
        x = np.arange(-100, 100, 2).astype(dtype)

        tf.initialize_all_variables().run()

        proba = qdist.log_prob(x)
        grads = tf.gradients(proba, [mu, sigma])

        self._assert_all_finite(proba.eval())
        self._assert_all_finite(grads[0].eval())
        self._assert_all_finite(grads[1].eval())

  def test_prob_and_grad_gives_finite_results_for_common_events(self):
    with self.test_session():
      mu = tf.Variable(0.0, name="mu")
      sigma = tf.Variable(1.0, name="sigma")
      qdist = distributions.QuantizedDistribution(
          base_dist_cls=distributions.Normal,
          mu=mu,
          sigma=sigma)
      x = tf.ceil(4 * self._rng.rand(100).astype(np.float32) - 2)

      tf.initialize_all_variables().run()

      proba = qdist.prob(x)
      self._assert_all_finite(proba.eval())

      grads = tf.gradients(proba, [mu, sigma])
      self._assert_all_finite(grads[0].eval())
      self._assert_all_finite(grads[1].eval())

  def test_lower_cutoff_must_be_below_upper_cutoff_or_we_raise(self):
    with self.test_session():
      qdist = distributions.QuantizedDistribution(
          base_dist_cls=distributions.Normal,
          lower_cutoff=1.,  # not strictly less than upper_cutoff.
          upper_cutoff=1.,
          mu=0.,
          sigma=1.,
          validate_args=True)

      self.assertTrue(qdist.validate_args)  # Default is True.
      with self.assertRaisesOpError("must be strictly less"):
        qdist.sample().eval()

  def test_cutoffs_must_be_integer_valued_if_validate_args_true(self):
    with self.test_session():
      qdist = distributions.QuantizedDistribution(
          base_dist_cls=distributions.Normal,
          lower_cutoff=1.5,
          upper_cutoff=10.,
          mu=0.,
          sigma=1.,
          validate_args=True)

      self.assertTrue(qdist.validate_args)  # Default is True.
      with self.assertRaisesOpError("has non-integer components"):
        qdist.sample().eval()

  def test_cutoffs_can_be_float_valued_if_validate_args_false(self):
    with self.test_session():
      qdist = distributions.QuantizedDistribution(
          base_dist_cls=distributions.Normal,
          lower_cutoff=1.5,
          upper_cutoff=10.11,
          mu=0.,
          sigma=1.,
          validate_args=False)

      self.assertFalse(qdist.validate_args)  # Default is True.

      # Should not raise
      qdist.sample().eval()

  def test_dtype_and_shape_inherited_from_base_dist(self):
    batch_shape = (2, 3)
    with self.test_session():
      qdist = distributions.QuantizedDistribution(
          base_dist_cls=distributions.Normal,
          lower_cutoff=1.0,
          upper_cutoff=10.0,
          mu=tf.zeros(batch_shape),
          sigma=tf.ones(batch_shape))

      self.assertEqual(batch_shape, qdist.get_batch_shape())
      self.assertAllEqual(batch_shape, qdist.batch_shape().eval())
      self.assertEqual((), qdist.get_event_shape())
      self.assertAllEqual((), qdist.event_shape().eval())

      samps = qdist.sample_n(n=10)
      self.assertEqual((10,) + batch_shape, samps.get_shape())
      self.assertAllEqual((10,) + batch_shape, samps.eval().shape)

      y = self._rng.randint(0, 5, size=batch_shape).astype(np.float32)
      self.assertEqual(batch_shape, qdist.prob(y).get_shape())
      self.assertEqual(batch_shape, qdist.prob(y).eval().shape)


if __name__ == "__main__":
  tf.test.main()
