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
"""Utilities for testing random variables."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np

from tensorflow.python.ops.distributions import special_math


def test_moment_matching(
    samples,
    number_moments,
    dist,
    stride=0):
  """Return z-test scores for sample moments to match analytic moments.

  Given `samples`, check that the first sample `number_moments` match
  the given  `dist` moments by doing a z-test.

  Args:
    samples: Samples from target distribution.
    number_moments: Python `int` describing how many sample moments to check.
    dist: SciPy distribution object that provides analytic moments.
    stride: Distance between samples to check for statistical properties.
      A stride of 0 means to use all samples, while other strides test for
      spatial correlation.
  Returns:
    Array of z_test scores.
  """

  sample_moments = []
  expected_moments = []
  variance_sample_moments = []
  for i in range(1, number_moments + 1):
    if len(samples.shape) == 2:
      strided_range = samples.flat[::(i - 1) * stride + 1]
    else:
      strided_range = samples[::(i - 1) * stride + 1, ...]
    sample_moments.append(np.mean(strided_range**i, axis=0))
    expected_moments.append(dist.moment(i))
    variance_sample_moments.append(
        (dist.moment(2 * i) - dist.moment(i) ** 2) / len(strided_range))

  z_test_scores = []
  for i in range(1, number_moments + 1):
    # Assume every operation has a small numerical error.
    # It takes i multiplications to calculate one i-th moment.
    total_variance = (
        variance_sample_moments[i - 1] +
        i * np.finfo(samples.dtype).eps)
    tiny = np.finfo(samples.dtype).tiny
    assert np.all(total_variance > 0)
    total_variance = np.where(total_variance < tiny, tiny, total_variance)
    # z_test is approximately a unit normal distribution.
    z_test_scores.append(abs(
        (sample_moments[i - 1] - expected_moments[i - 1]) / np.sqrt(
            total_variance)))
  return z_test_scores


def chi_squared(x, bins):
  """Pearson's Chi-squared test."""
  x = np.ravel(x)
  n = len(x)
  histogram, _ = np.histogram(x, bins=bins, range=(0, 1))
  expected = n / float(bins)
  return np.sum(np.square(histogram - expected) / expected)


def normal_cdf(x):
  """Cumulative distribution function for a standard normal distribution."""
  return 0.5 + 0.5 * np.vectorize(math.erf)(x / math.sqrt(2))


def anderson_darling(x):
  """Anderson-Darling test for a standard normal distribution."""
  x = np.sort(np.ravel(x))
  n = len(x)
  i = np.linspace(1, n, n)
  z = np.sum((2 * i - 1) * np.log(normal_cdf(x)) +
             (2 * (n - i) + 1) * np.log(1 - normal_cdf(x)))
  return -n - z / n


def test_truncated_normal(assert_equal, assert_all_close, n, y,
                          mean_atol=5e-4, median_atol=8e-4, variance_rtol=1e-3):
  """Tests truncated normal distribution's statistics."""
  def _normal_cdf(x):
    return .5 * math.erfc(-x / math.sqrt(2))

  def normal_pdf(x):
    return math.exp(-(x**2) / 2.) / math.sqrt(2 * math.pi)

  def probit(x):
    return special_math.ndtri(x)

  a = -2.
  b = 2.
  mu = 0.
  sigma = 1.

  alpha = (a - mu) / sigma
  beta = (b - mu) / sigma
  z = _normal_cdf(beta) - _normal_cdf(alpha)

  assert_equal((y >= a).sum(), n)
  assert_equal((y <= b).sum(), n)

  # For more information on these calculations, see:
  # Burkardt, John. "The Truncated Normal Distribution".
  # Department of Scientific Computing website. Florida State University.
  expected_mean = mu + (normal_pdf(alpha) - normal_pdf(beta)) / z * sigma
  y = y.astype(float)
  actual_mean = np.mean(y)
  assert_all_close(actual_mean, expected_mean, atol=mean_atol)

  expected_median = mu + probit(
      (_normal_cdf(alpha) + _normal_cdf(beta)) / 2.) * sigma
  actual_median = np.median(y)
  assert_all_close(actual_median, expected_median, atol=median_atol)

  expected_variance = sigma**2 * (1 + (
      (alpha * normal_pdf(alpha) - beta * normal_pdf(beta)) / z) - (
          (normal_pdf(alpha) - normal_pdf(beta)) / z)**2)
  actual_variance = np.var(y)
  assert_all_close(
      actual_variance,
      expected_variance,
      rtol=variance_rtol)
