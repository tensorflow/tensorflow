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
"""Tests for Csiszar Divergence Ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.bayesflow.python.ops import csiszar_divergence_impl
from tensorflow.contrib.distributions.python.ops import mvn_diag as mvn_diag_lib
from tensorflow.contrib.distributions.python.ops import mvn_full_covariance as mvn_full_lib
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops.distributions import kullback_leibler
from tensorflow.python.ops.distributions import normal as normal_lib
from tensorflow.python.platform import test


cd = csiszar_divergence_impl


class AmariAlphaTest(test.TestCase):

  def setUp(self):
    self._logu = np.linspace(-10., 10, 100)
    self._u = np.exp(self._logu)

  def test_at_zero(self):
    for alpha in [-1., 0., 1., 2.]:
      for normalized in [True, False]:
        with self.test_session(graph=ops.Graph()):
          self.assertAllClose(
              cd.amari_alpha(0., alpha=alpha,
                             self_normalized=normalized).eval(),
              0.)

  def test_correct_when_alpha0(self):
    with self.test_session():
      self.assertAllClose(
          cd.amari_alpha(self._logu, alpha=0.).eval(),
          -self._logu)

      self.assertAllClose(
          cd.amari_alpha(self._logu, alpha=0., self_normalized=True).eval(),
          -self._logu + (self._u - 1.))

  def test_correct_when_alpha1(self):
    with self.test_session():
      self.assertAllClose(
          cd.amari_alpha(self._logu, alpha=1.).eval(),
          self._u * self._logu)

      self.assertAllClose(
          cd.amari_alpha(self._logu, alpha=1., self_normalized=True).eval(),
          self._u * self._logu - (self._u - 1.))

  def test_correct_when_alpha_not_01(self):
    for alpha in [-2, -1., -0.5, 0.5, 2.]:
      with self.test_session(graph=ops.Graph()):
        self.assertAllClose(
            cd.amari_alpha(self._logu,
                           alpha=alpha,
                           self_normalized=False).eval(),
            ((self._u**alpha - 1)) / (alpha * (alpha - 1.)))

        self.assertAllClose(
            cd.amari_alpha(self._logu,
                           alpha=alpha,
                           self_normalized=True).eval(),
            ((self._u**alpha - 1.)
             - alpha * (self._u - 1)) / (alpha * (alpha - 1.)))


class KLReverseTest(test.TestCase):

  def setUp(self):
    self._logu = np.linspace(-10., 10, 100)
    self._u = np.exp(self._logu)

  def test_at_zero(self):
    for normalized in [True, False]:
      with self.test_session(graph=ops.Graph()):
        self.assertAllClose(
            cd.kl_reverse(0., self_normalized=normalized).eval(),
            0.)

  def test_correct(self):
    with self.test_session():
      self.assertAllClose(
          cd.kl_reverse(self._logu).eval(),
          -self._logu)

      self.assertAllClose(
          cd.kl_reverse(self._logu, self_normalized=True).eval(),
          -self._logu + (self._u - 1.))


class KLForwardTest(test.TestCase):

  def setUp(self):
    self._logu = np.linspace(-10., 10, 100)
    self._u = np.exp(self._logu)

  def test_at_zero(self):
    for normalized in [True, False]:
      with self.test_session(graph=ops.Graph()):
        self.assertAllClose(
            cd.kl_forward(0., self_normalized=normalized).eval(),
            0.)

  def test_correct(self):
    with self.test_session():
      self.assertAllClose(
          cd.kl_forward(self._logu).eval(),
          self._u * self._logu)

      self.assertAllClose(
          cd.kl_forward(self._logu, self_normalized=True).eval(),
          self._u * self._logu - (self._u - 1.))


class JensenShannonTest(test.TestCase):

  def setUp(self):
    self._logu = np.linspace(-10., 10, 100)
    self._u = np.exp(self._logu)

  def test_at_zero(self):
    with self.test_session():
      self.assertAllClose(cd.jensen_shannon(0.).eval(), np.log(0.25))

  def test_symmetric(self):
    with self.test_session():
      self.assertAllClose(
          cd.jensen_shannon(self._logu).eval(),
          cd.symmetrized_csiszar_function(
              self._logu, cd.jensen_shannon).eval())

      self.assertAllClose(
          cd.jensen_shannon(self._logu, self_normalized=True).eval(),
          cd.symmetrized_csiszar_function(
              self._logu,
              lambda x: cd.jensen_shannon(x, self_normalized=True)).eval())

  def test_correct(self):
    with self.test_session():
      self.assertAllClose(
          cd.jensen_shannon(self._logu).eval(),
          (self._u * self._logu
           - (1 + self._u) * np.log1p(self._u)))

      self.assertAllClose(
          cd.jensen_shannon(self._logu, self_normalized=True).eval(),
          (self._u * self._logu
           - (1 + self._u) * np.log((1 + self._u) / 2)))


class ArithmeticGeometricMeanTest(test.TestCase):

  def setUp(self):
    self._logu = np.linspace(-10., 10, 100)
    self._u = np.exp(self._logu)

  def test_at_zero(self):
    with self.test_session():
      self.assertAllClose(cd.arithmetic_geometric(0.).eval(), np.log(4))
      self.assertAllClose(
          cd.arithmetic_geometric(0., self_normalized=True).eval(), 0.)

  def test_symmetric(self):
    with self.test_session():
      self.assertAllClose(
          cd.arithmetic_geometric(self._logu).eval(),
          cd.symmetrized_csiszar_function(
              self._logu, cd.arithmetic_geometric).eval())

  def test_correct(self):
    with self.test_session():
      self.assertAllClose(
          cd.arithmetic_geometric(self._logu).eval(),
          (1. + self._u) * np.log((1. + self._u) / np.sqrt(self._u)))

      self.assertAllClose(
          cd.arithmetic_geometric(self._logu, self_normalized=True).eval(),
          (1. + self._u) * np.log(0.5 * (1. + self._u) / np.sqrt(self._u)))


class TotalVariationTest(test.TestCase):

  def setUp(self):
    self._logu = np.linspace(-10., 10, 100)
    self._u = np.exp(self._logu)

  def test_at_zero(self):
    with self.test_session():
      self.assertAllClose(cd.total_variation(0.).eval(), 0.)

  def test_correct(self):
    with self.test_session():
      self.assertAllClose(
          cd.total_variation(self._logu).eval(),
          0.5 * np.abs(self._u - 1))


class PearsonTest(test.TestCase):

  def setUp(self):
    self._logu = np.linspace(-10., 10, 100)
    self._u = np.exp(self._logu)

  def test_at_zero(self):
    with self.test_session():
      self.assertAllClose(cd.pearson(0.).eval(), 0.)

  def test_correct(self):
    with self.test_session():
      self.assertAllClose(
          cd.pearson(self._logu).eval(),
          np.square(self._u - 1))


class SquaredHellingerTest(test.TestCase):

  def setUp(self):
    self._logu = np.linspace(-10., 10, 100)
    self._u = np.exp(self._logu)

  def test_at_zero(self):
    with self.test_session():
      self.assertAllClose(cd.squared_hellinger(0.).eval(), 0.)

  def test_symmetric(self):
    with self.test_session():
      self.assertAllClose(
          cd.squared_hellinger(self._logu).eval(),
          cd.symmetrized_csiszar_function(
              self._logu, cd.squared_hellinger).eval())

  def test_correct(self):
    with self.test_session():
      self.assertAllClose(
          cd.squared_hellinger(self._logu).eval(),
          np.square(np.sqrt(self._u) - 1))


class TriangularTest(test.TestCase):

  def setUp(self):
    self._logu = np.linspace(-10., 10, 100)
    self._u = np.exp(self._logu)

  def test_at_zero(self):
    with self.test_session():
      self.assertAllClose(cd.triangular(0.).eval(), 0.)

  def test_symmetric(self):
    with self.test_session():
      self.assertAllClose(
          cd.triangular(self._logu).eval(),
          cd.symmetrized_csiszar_function(
              self._logu, cd.triangular).eval())

  def test_correct(self):
    with self.test_session():
      self.assertAllClose(
          cd.triangular(self._logu).eval(),
          np.square(self._u - 1) / (1 + self._u))


class Log1pAbsTest(test.TestCase):

  def setUp(self):
    self._logu = np.linspace(-10., 10, 100)
    self._u = np.exp(self._logu)

  def test_at_zero(self):
    with self.test_session():
      self.assertAllClose(cd.log1p_abs(0.).eval(), 0.)

  def test_correct(self):
    with self.test_session():
      self.assertAllClose(
          cd.log1p_abs(self._logu).eval(),
          self._u**(np.sign(self._u - 1)) - 1)


class JeffreysTest(test.TestCase):

  def setUp(self):
    self._logu = np.linspace(-10., 10, 100)
    self._u = np.exp(self._logu)

  def test_at_zero(self):
    with self.test_session():
      self.assertAllClose(cd.jeffreys(0.).eval(), 0.)

  def test_symmetric(self):
    with self.test_session():
      self.assertAllClose(
          cd.jeffreys(self._logu).eval(),
          cd.symmetrized_csiszar_function(
              self._logu, cd.jeffreys).eval())

  def test_correct(self):
    with self.test_session():
      self.assertAllClose(
          cd.jeffreys(self._logu).eval(),
          0.5 * (self._u * self._logu - self._logu))


class ChiSquareTest(test.TestCase):

  def setUp(self):
    self._logu = np.linspace(-10., 10, 100)
    self._u = np.exp(self._logu)

  def test_at_zero(self):
    with self.test_session():
      self.assertAllClose(cd.chi_square(0.).eval(), 0.)

  def test_correct(self):
    with self.test_session():
      self.assertAllClose(
          cd.chi_square(self._logu).eval(),
          self._u**2 - 1)


class ModifiedGanTest(test.TestCase):

  def setUp(self):
    self._logu = np.linspace(-10., 10, 100)
    self._u = np.exp(self._logu)

  def test_at_zero(self):
    with self.test_session():
      self.assertAllClose(
          cd.modified_gan(0.).eval(), np.log(2))
      self.assertAllClose(
          cd.modified_gan(0., self_normalized=True).eval(), np.log(2))

  def test_correct(self):
    with self.test_session():
      self.assertAllClose(
          cd.modified_gan(self._logu).eval(),
          np.log1p(self._u) - self._logu)

      self.assertAllClose(
          cd.modified_gan(self._logu, self_normalized=True).eval(),
          np.log1p(self._u) - self._logu + 0.5 * (self._u - 1))


class SymmetrizedCsiszarFunctionTest(test.TestCase):

  def setUp(self):
    self._logu = np.linspace(-10., 10., 100)
    self._u = np.exp(self._logu)

  def test_jensen_shannon(self):
    with self.test_session():

      # The following functions come from the claim made in the
      # symmetrized_csiszar_function docstring.
      def js1(logu):
        return (-logu
                - (1. + math_ops.exp(logu)) * (
                    nn_ops.softplus(logu)))

      def js2(logu):
        return 2. * (math_ops.exp(logu) * (
            logu - nn_ops.softplus(logu)))

      self.assertAllClose(
          cd.symmetrized_csiszar_function(self._logu, js1).eval(),
          cd.jensen_shannon(self._logu).eval())

      self.assertAllClose(
          cd.symmetrized_csiszar_function(self._logu, js2).eval(),
          cd.jensen_shannon(self._logu).eval())

  def test_jeffreys(self):
    with self.test_session():
      self.assertAllClose(
          cd.symmetrized_csiszar_function(self._logu, cd.kl_reverse).eval(),
          cd.jeffreys(self._logu).eval())

      self.assertAllClose(
          cd.symmetrized_csiszar_function(self._logu, cd.kl_forward).eval(),
          cd.jeffreys(self._logu).eval())


class DualCsiszarFunctionTest(test.TestCase):

  def setUp(self):
    self._logu = np.linspace(-10., 10., 100)
    self._u = np.exp(self._logu)

  def test_kl_forward(self):
    with self.test_session():
      self.assertAllClose(
          cd.dual_csiszar_function(self._logu, cd.kl_forward).eval(),
          cd.kl_reverse(self._logu).eval())

  def test_kl_reverse(self):
    with self.test_session():
      self.assertAllClose(
          cd.dual_csiszar_function(self._logu, cd.kl_reverse).eval(),
          cd.kl_forward(self._logu).eval())


class MonteCarloCsiszarFDivergenceTest(test.TestCase):

  def test_kl_forward(self):
    with self.test_session() as sess:
      q = normal_lib.Normal(
          loc=np.ones(6),
          scale=np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0]))

      p = normal_lib.Normal(loc=q.loc + 0.1, scale=q.scale - 0.2)

      approx_kl = cd.monte_carlo_csiszar_f_divergence(
          f=cd.kl_forward,
          p=p,
          q=q,
          num_draws=int(1e5),
          seed=1)

      approx_kl_self_normalized = cd.monte_carlo_csiszar_f_divergence(
          f=lambda logu: cd.kl_forward(logu, self_normalized=True),
          p=p,
          q=q,
          num_draws=int(1e5),
          seed=1)

      exact_kl = kullback_leibler.kl_divergence(p, q)

      [approx_kl_, approx_kl_self_normalized_, exact_kl_] = sess.run([
          approx_kl, approx_kl_self_normalized, exact_kl])

      self.assertAllClose(approx_kl_, exact_kl_,
                          rtol=0.08, atol=0.)

      self.assertAllClose(approx_kl_self_normalized_, exact_kl_,
                          rtol=0.02, atol=0.)

  def test_kl_reverse(self):
    with self.test_session() as sess:

      q = normal_lib.Normal(
          loc=np.ones(6),
          scale=np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0]))

      p = normal_lib.Normal(loc=q.loc + 0.1, scale=q.scale - 0.2)

      approx_kl = cd.monte_carlo_csiszar_f_divergence(
          f=cd.kl_reverse,
          p=p,
          q=q,
          num_draws=int(1e5),
          seed=1)

      approx_kl_self_normalized = cd.monte_carlo_csiszar_f_divergence(
          f=lambda logu: cd.kl_reverse(logu, self_normalized=True),
          p=p,
          q=q,
          num_draws=int(1e5),
          seed=1)

      exact_kl = kullback_leibler.kl_divergence(q, p)

      [approx_kl_, approx_kl_self_normalized_, exact_kl_] = sess.run([
          approx_kl, approx_kl_self_normalized, exact_kl])

      self.assertAllClose(approx_kl_, exact_kl_,
                          rtol=0.07, atol=0.)

      self.assertAllClose(approx_kl_self_normalized_, exact_kl_,
                          rtol=0.02, atol=0.)

  def _tridiag(self, d, diag_value, offdiag_value):
    """d x d matrix with given value on diag, and one super/sub diag."""
    diag_mat = linalg_ops.eye(d) * (diag_value - offdiag_value)
    three_bands = array_ops.matrix_band_part(
        array_ops.fill([d, d], offdiag_value), 1, 1)
    return diag_mat + three_bands

  def test_kl_reverse_multidim(self):

    with self.test_session() as sess:
      d = 5  # Dimension

      p = mvn_full_lib.MultivariateNormalFullCovariance(
          covariance_matrix=self._tridiag(d, diag_value=1, offdiag_value=0.5))

      q = mvn_diag_lib.MultivariateNormalDiag(scale_diag=[0.5]*d)

      approx_kl = cd.monte_carlo_csiszar_f_divergence(
          f=cd.kl_reverse,
          p=p,
          q=q,
          num_draws=int(1e5),
          seed=1)

      approx_kl_self_normalized = cd.monte_carlo_csiszar_f_divergence(
          f=lambda logu: cd.kl_reverse(logu, self_normalized=True),
          p=p,
          q=q,
          num_draws=int(1e5),
          seed=1)

      exact_kl = kullback_leibler.kl_divergence(q, p)

      [approx_kl_, approx_kl_self_normalized_, exact_kl_] = sess.run([
          approx_kl, approx_kl_self_normalized, exact_kl])

      self.assertAllClose(approx_kl_, exact_kl_,
                          rtol=0.02, atol=0.)

      self.assertAllClose(approx_kl_self_normalized_, exact_kl_,
                          rtol=0.08, atol=0.)

  def test_kl_forward_multidim(self):

    with self.test_session() as sess:
      d = 5  # Dimension

      p = mvn_full_lib.MultivariateNormalFullCovariance(
          covariance_matrix=self._tridiag(d, diag_value=1, offdiag_value=0.5))

      # Variance is very high when approximating Forward KL, so we make
      # scale_diag larger than in test_kl_reverse_multidim. This ensures q
      # "covers" p and thus Var_q[p/q] is smaller.
      q = mvn_diag_lib.MultivariateNormalDiag(scale_diag=[1.]*d)

      approx_kl = cd.monte_carlo_csiszar_f_divergence(
          f=cd.kl_forward,
          p=p,
          q=q,
          num_draws=int(1e5),
          seed=1)

      approx_kl_self_normalized = cd.monte_carlo_csiszar_f_divergence(
          f=lambda logu: cd.kl_forward(logu, self_normalized=True),
          p=p,
          q=q,
          num_draws=int(1e5),
          seed=1)

      exact_kl = kullback_leibler.kl_divergence(p, q)

      [approx_kl_, approx_kl_self_normalized_, exact_kl_] = sess.run([
          approx_kl, approx_kl_self_normalized, exact_kl])

      self.assertAllClose(approx_kl_, exact_kl_,
                          rtol=0.06, atol=0.)

      self.assertAllClose(approx_kl_self_normalized_, exact_kl_,
                          rtol=0.05, atol=0.)

  def test_score_trick(self):

    with self.test_session() as sess:
      d = 5  # Dimension
      num_draws = int(1e5)
      seed = 1

      p = mvn_full_lib.MultivariateNormalFullCovariance(
          covariance_matrix=self._tridiag(d, diag_value=1, offdiag_value=0.5))

      # Variance is very high when approximating Forward KL, so we make
      # scale_diag larger than in test_kl_reverse_multidim. This ensures q
      # "covers" p and thus Var_q[p/q] is smaller.
      s = array_ops.constant(1.)
      q = mvn_diag_lib.MultivariateNormalDiag(
          scale_diag=array_ops.tile([s], [d]))

      approx_kl = cd.monte_carlo_csiszar_f_divergence(
          f=cd.kl_reverse,
          p=p,
          q=q,
          num_draws=num_draws,
          seed=seed)

      approx_kl_self_normalized = cd.monte_carlo_csiszar_f_divergence(
          f=lambda logu: cd.kl_reverse(logu, self_normalized=True),
          p=p,
          q=q,
          num_draws=num_draws,
          seed=seed)

      approx_kl_score_trick = cd.monte_carlo_csiszar_f_divergence(
          f=cd.kl_reverse,
          p=p,
          q=q,
          num_draws=num_draws,
          use_reparametrization=False,
          seed=seed)

      approx_kl_self_normalized_score_trick = (
          cd.monte_carlo_csiszar_f_divergence(
              f=lambda logu: cd.kl_reverse(logu, self_normalized=True),
              p=p,
              q=q,
              num_draws=num_draws,
              use_reparametrization=False,
              seed=seed))

      exact_kl = kullback_leibler.kl_divergence(q, p)

      grad = lambda fs: gradients_impl.gradients(fs, s)[0]

      [
          approx_kl_grad_,
          approx_kl_self_normalized_grad_,
          approx_kl_score_trick_grad_,
          approx_kl_self_normalized_score_trick_grad_,
          exact_kl_grad_,
          approx_kl_,
          approx_kl_self_normalized_,
          approx_kl_score_trick_,
          approx_kl_self_normalized_score_trick_,
          exact_kl_,
      ] = sess.run([
          grad(approx_kl),
          grad(approx_kl_self_normalized),
          grad(approx_kl_score_trick),
          grad(approx_kl_self_normalized_score_trick),
          grad(exact_kl),
          approx_kl,
          approx_kl_self_normalized,
          approx_kl_score_trick,
          approx_kl_self_normalized_score_trick,
          exact_kl,
      ])

      # Test average divergence.
      self.assertAllClose(approx_kl_, exact_kl_,
                          rtol=0.02, atol=0.)

      self.assertAllClose(approx_kl_self_normalized_, exact_kl_,
                          rtol=0.08, atol=0.)

      self.assertAllClose(approx_kl_score_trick_, exact_kl_,
                          rtol=0.02, atol=0.)

      self.assertAllClose(approx_kl_self_normalized_score_trick_, exact_kl_,
                          rtol=0.08, atol=0.)

      # Test average gradient-divergence.
      self.assertAllClose(approx_kl_grad_, exact_kl_grad_,
                          rtol=0.007, atol=0.)

      self.assertAllClose(approx_kl_self_normalized_grad_, exact_kl_grad_,
                          rtol=0.011, atol=0.)

      self.assertAllClose(approx_kl_score_trick_grad_, exact_kl_grad_,
                          rtol=0.018, atol=0.)

      self.assertAllClose(
          approx_kl_self_normalized_score_trick_grad_, exact_kl_grad_,
          rtol=0.017, atol=0.)


if __name__ == '__main__':
  test.main()
