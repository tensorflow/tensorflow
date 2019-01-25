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

import numpy as np


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
  x = samples.flat
  for i in range(1, number_moments + 1):
    strided_range = x[::(i - 1) * stride + 1]
    sample_moments.append(np.mean(strided_range ** i))
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
    if total_variance < tiny:
      total_variance = tiny
    # z_test is approximately a unit normal distribution.
    z_test_scores.append(abs(
        (sample_moments[i - 1] - expected_moments[i - 1]) / np.sqrt(
            total_variance)))
  return z_test_scores

