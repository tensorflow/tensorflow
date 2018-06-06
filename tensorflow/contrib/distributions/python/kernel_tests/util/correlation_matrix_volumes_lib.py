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
"""Estimating the volume of the correlation matrices with bounded determinant.

Why?  Because lkj_test.py tests the sampler for the LKJ distribution
by estimating the same volume another way.

How?  Rejection sampling.  Or, more precisely, importance sampling,
proposing from the uniform distribution on symmetric matrices with
diagonal 1s and entries in [-1, 1].  Such a matrix is a correlation
matrix if and only if it is also positive semi-definite.

The samples can then be converted into a confidence interval on the
volume in question by the [Clopper-Pearson
method](https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval),
also implemented here.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import importlib
import sys

import numpy as np

from tensorflow.python.client import session
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.distributions import uniform
from tensorflow.python.ops.distributions import util
from tensorflow.python.platform import tf_logging

__all__ = [
    "correlation_matrix_volume_rejection_samples",
    "compute_true_volumes",
]


def try_import(name):  # pylint: disable=invalid-name
  module = None
  try:
    module = importlib.import_module(name)
  except ImportError as e:
    tf_logging.warning("Could not import %s: %s" % (name, str(e)))
  return module

optimize = try_import("scipy.optimize")
stats = try_import("scipy.stats")


def _psd_mask(x):
  """Computes whether each square matrix in the input is positive semi-definite.

  Args:
    x: A floating-point `Tensor` of shape `[B1, ..., Bn, M, M]`.

  Returns:
    mask: A floating-point `Tensor` of shape `[B1, ... Bn]`.  Each
      scalar is 1 if the corresponding matrix was PSD, otherwise 0.
  """
  # Allegedly
  # https://scicomp.stackexchange.com/questions/12979/testing-if-a-matrix-is-positive-semi-definite
  # it is more efficient to test for positive semi-definiteness by
  # trying to compute the Cholesky decomposition -- the matrix is PSD
  # if you succeed and not PSD if you fail.  However, TensorFlow's
  # Cholesky raises an exception if _any_ of the input matrices are
  # not PSD, from which I don't know how to extract _which ones_, so I
  # proceed by explicitly computing all the eigenvalues and checking
  # whether they are all positive or not.
  #
  # Also, as was discussed in the answer, it is somewhat dangerous to
  # treat SPD-ness as binary in floating-point arithmetic. Cholesky
  # factorization can complete and 'look' like everything is fine
  # (e.g., O(1) entries and a diagonal of all ones) but the matrix can
  # have an exponential condition number.
  eigenvalues, _ = linalg_ops.self_adjoint_eig(x)
  return math_ops.cast(
      math_ops.reduce_min(eigenvalues, axis=-1) >= 0, dtype=x.dtype)


def _det_large_enough_mask(x, det_bounds):
  """Returns whether the input matches the given determinant limit.

  Args:
    x: A floating-point `Tensor` of shape `[B1, ..., Bn, M, M]`.
    det_bounds: A floating-point `Tensor` that must broadcast to shape
      `[B1, ..., Bn]`, giving the desired lower bound on the
      determinants in `x`.

  Returns:
    mask: A floating-point `Tensor` of shape [B1, ..., Bn].  Each
      scalar is 1 if the corresponding matrix had determinant above
      the corresponding bound, otherwise 0.
  """
  # For the curious: I wonder whether it is possible and desirable to
  # use a Cholesky decomposition-based algorithm for this, since the
  # only matrices whose determinant this code cares about will be PSD.
  # Didn't figure out how to code that in TensorFlow.
  #
  # Expert opinion is that it would be about twice as fast since
  # Cholesky is roughly half the cost of Gaussian Elimination with
  # Partial Pivoting. But this is less of an impact than the switch in
  # _psd_mask.
  return math_ops.cast(
      linalg_ops.matrix_determinant(x) > det_bounds, dtype=x.dtype)


def _uniform_correlation_like_matrix(num_rows, batch_shape, dtype, seed):
  """Returns a uniformly random `Tensor` of "correlation-like" matrices.

  A "correlation-like" matrix is a symmetric square matrix with all entries
  between -1 and 1 (inclusive) and 1s on the main diagonal.  Of these,
  the ones that are positive semi-definite are exactly the correlation
  matrices.

  Args:
    num_rows: Python `int` dimension of the correlation-like matrices.
    batch_shape: `Tensor` or Python `tuple` of `int` shape of the
      batch to return.
    dtype: `dtype` of the `Tensor` to return.
    seed: Random seed.

  Returns:
    matrices: A `Tensor` of shape `batch_shape + [num_rows, num_rows]`
      and dtype `dtype`.  Each entry is in [-1, 1], and each matrix
      along the bottom two dimensions is symmetric and has 1s on the
      main diagonal.
  """
  num_entries = num_rows * (num_rows + 1) / 2
  ones = array_ops.ones(shape=[num_entries], dtype=dtype)
  # It seems wasteful to generate random values for the diagonal since
  # I am going to throw them away, but `fill_triangular` fills the
  # diagonal, so I probably need them.
  # It's not impossible that it would be more efficient to just fill
  # the whole matrix with random values instead of messing with
  # `fill_triangular`.  Then would need to filter almost half out with
  # `matrix_band_part`.
  unifs = uniform.Uniform(-ones, ones).sample(batch_shape, seed=seed)
  tril = util.fill_triangular(unifs)
  symmetric = tril + array_ops.matrix_transpose(tril)
  diagonal_ones = array_ops.ones(
      shape=util.pad(batch_shape, axis=0, back=True, value=num_rows),
      dtype=dtype)
  return array_ops.matrix_set_diag(symmetric, diagonal_ones)


def correlation_matrix_volume_rejection_samples(
    det_bounds, dim, sample_shape, dtype, seed):
  """Returns rejection samples from trying to get good correlation matrices.

  The proposal being rejected from is the uniform distribution on
  "correlation-like" matrices.  We say a matrix is "correlation-like"
  if it is a symmetric square matrix with all entries between -1 and 1
  (inclusive) and 1s on the main diagonal.  Of these, the ones that
  are positive semi-definite are exactly the correlation matrices.

  The rejection algorithm, then, is to sample a `Tensor` of
  `sample_shape` correlation-like matrices of dimensions `dim` by
  `dim`, and check each one for (i) being a correlation matrix (i.e.,
  PSD), and (ii) having determinant at least the corresponding entry
  of `det_bounds`.

  Args:
    det_bounds: A `Tensor` of lower bounds on the determinants of
      acceptable matrices.  The shape must broadcast with `sample_shape`.
    dim: A Python `int` dimension of correlation matrices to sample.
    sample_shape: Python `tuple` of `int` shape of the samples to
      compute, excluding the two matrix dimensions.
    dtype: The `dtype` in which to do the computation.
    seed: Random seed.

  Returns:
    weights: A `Tensor` of shape `sample_shape`.  Each entry is 0 if the
      corresponding matrix was not a correlation matrix, or had too
      small of a determinant.  Otherwise, the entry is the
      multiplicative inverse of the density of proposing that matrix
      uniformly, i.e., the volume of the set of `dim` by `dim`
      correlation-like matrices.
    volume: The volume of the set of `dim` by `dim` correlation-like
      matrices.
  """
  with ops.name_scope("rejection_sampler"):
    rej_proposals = _uniform_correlation_like_matrix(
        dim, sample_shape, dtype, seed=seed)
    rej_proposal_volume = 2. ** (dim * (dim - 1) / 2.)
    # The density of proposing any given point is 1 / rej_proposal_volume;
    # The weight of that point should be scaled by
    # 1 / density = rej_proposal_volume.
    rej_weights = rej_proposal_volume * _psd_mask(
        rej_proposals) * _det_large_enough_mask(rej_proposals, det_bounds)
    return rej_weights, rej_proposal_volume


def _clopper_pearson_confidence_interval(samples, error_rate):
  """Computes a confidence interval for the mean of the given 1-D distribution.

  Assumes (and checks) that the given distribution is Bernoulli, i.e.,
  takes only two values.  This licenses using the CDF of the binomial
  distribution for the confidence, which is tighter (for extreme
  probabilities) than the DKWM inequality.  The method is known as the
  [Clopper-Pearson method]
  (https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval).

  Assumes:

  - The given samples were drawn iid from the distribution of interest.

  - The given distribution is a Bernoulli, i.e., supported only on
    low and high.

  Guarantees:

  - The probability (over the randomness of drawing the given sample)
    that the true mean is outside the returned interval is no more
    than the given error_rate.

  Args:
    samples: `np.ndarray` of samples drawn iid from the distribution
      of interest.
    error_rate: Python `float` admissible rate of mistakes.

  Returns:
    low: Lower bound of confidence interval.
    high: Upper bound of confidence interval.

  Raises:
    ValueError: If `samples` has rank other than 1 (batch semantics
      are not implemented), or if `samples` contains values other than
      `low` or `high` (as that makes the distribution not Bernoulli).
  """
  # TODO(b/78025336) Migrate this confidence interval function
  # to statistical_testing.py.  In order to do that
  # - Get the binomial CDF from the Binomial distribution
  # - Implement scalar root finding in TF.  Batch bisection search
  #   shouldn't be too hard, and is definitely good enough for this
  #   problem.  Batching the Brent algorithm (from scipy) that is used
  #   here may be more involved, but may also not be necessary---it's
  #   only used here because scipy made it convenient.  In particular,
  #   robustness is more important than speed here, which may make
  #   bisection search actively better.
  # - The rest is just a matter of rewriting in the appropriate style.
  if optimize is None or stats is None:
    raise ValueError(
        "Scipy is required for computing Clopper-Pearson confidence intervals")
  if len(samples.shape) != 1:
    raise ValueError("Batch semantics not implemented")
  n = len(samples)
  low = np.amin(samples)
  high = np.amax(samples)
  successes = np.count_nonzero(samples - low)
  failures = np.count_nonzero(samples - high)
  if successes + failures != n:
    uniques = np.unique(samples)
    msg = ("Purportedly Bernoulli distribution had distinct samples"
           " {}, {}, and {}".format(uniques[0], uniques[1], uniques[2]))
    raise ValueError(msg)
  def p_small_enough(p):
    prob = stats.binom.logcdf(successes, n, p)
    return prob - np.log(error_rate / 2.)
  def p_big_enough(p):
    prob = stats.binom.logsf(successes, n, p)
    return prob - np.log(error_rate / 2.)
  high_p = optimize.brentq(
      p_small_enough, float(successes) / n, 1., rtol=1e-9)
  low_p = optimize.brentq(
      p_big_enough, 0., float(successes) / n, rtol=1e-9)
  low_interval = low + (high - low) * low_p
  high_interval = low + (high - low) * high_p
  return (low_interval, high_interval)


def compute_true_volumes(
    det_bounds, dim, num_samples, error_rate=1e-6, seed=42):
  """Returns confidence intervals for the desired correlation matrix volumes.

  The confidence intervals are computed by the [Clopper-Pearson method]
  (https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval).

  Args:
    det_bounds: A rank-1 numpy array of lower bounds on the
      determinants of acceptable matrices.  Entries must be unique.
    dim: A Python `int` dimension of correlation matrices to sample.
    num_samples: The number of samples to draw.
    error_rate: The statistical significance of the returned
      confidence intervals.  The significance is broadcast: Each
      returned interval separately may be incorrect with probability
      (under the sample of correlation-like matrices drawn internally)
      at most `error_rate`.
    seed: Random seed.

  Returns:
    bounds: A Python `dict` mapping each determinant bound to the low, high
      tuple giving the confidence interval.
  """
  bounds = {}
  with session.Session() as sess:
    rej_weights, _ = correlation_matrix_volume_rejection_samples(
        det_bounds, dim, [num_samples, len(det_bounds)], np.float32, seed=seed)
    rej_weights = sess.run(rej_weights)
    for rw, det in zip(np.rollaxis(rej_weights, 1), det_bounds):
      template = ("Estimating volume of {}x{} correlation "
                  "matrices with determinant >= {}.")
      print(template.format(dim, dim, det))
      sys.stdout.flush()
      bounds[det] = _clopper_pearson_confidence_interval(
          rw, error_rate=error_rate)
    return bounds
