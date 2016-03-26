# Copyright 2016 Google Inc. All Rights Reserved.
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
"""The Normal (Gaussian) distribution class.

@@Gaussian
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import constant_op
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops


class Gaussian(object):
  """The Normal (Gaussian) distribution with mean mu and stddev sigma.

  The PDF of this distribution is:
    f(x) = sqrt(1/(2*pi*sigma^2)) exp(-(x-mu)^2/(2*sigma^2))
  """

  def __init__(self, mu, sigma):
    """Construct a new Gaussian distribution with mean mu and stddev sigma.

    Args:
      mu: Scalar tensor, the mean of the distribution.
      sigma: Scalar tensor, the precision of the distribution.

    Raises:
      TypeError: if mu and sigma are different dtypes.
    """
    self._mu = ops.convert_to_tensor(mu)
    self._sigma = ops.convert_to_tensor(sigma)
    if mu.dtype != sigma.dtype:
      raise TypeError("Expected same dtype for mu, sigma but got: %s vs. %s"
                      % (mu.dtype, sigma.dtype))

  @property
  def dtype(self):
    return self._mu.dtype

  @property
  def shape(self):
    return constant_op.constant([])  # Scalar

  @property
  def mu(self):
    return self._mu

  @property
  def sigma(self):
    return self._sigma

  def log_pdf(self, x):
    """Log likelihood of observations in x under Gaussian with mu and sigma.

    Args:
      x: 1-D, a vector of observations.

    Returns:
      log_lik: 1-D, a vector of log likelihoods of `x` under the model.
    """
    return (-0.5*math.log(2 * math.pi) - math_ops.log(self._sigma)
            -0.5*math_ops.square((x - self._mu) / self._sigma))

  def cdf(self, x):
    """CDF of observations in x under Gaussian with mu and sigma.

    Args:
      x: 1-D, a vector of observations.

    Returns:
      cdf: 1-D, a vector of CDFs of `x` under the model.
    """
    return (0.5 + 0.5*math_ops.erf(
        1.0/(math.sqrt(2.0) * self._sigma)*(x - self._mu)))

  def log_cdf(self, x):
    """Log of the CDF of observations x under Gaussian with mu and sigma."""
    return math_ops.log(self.cdf(x))

  def pdf(self, x):
    """The PDF for observations x.

    Args:
      x: 1-D, a vector of observations.

    Returns:
      pdf: 1-D, a vector of pdf values of `x` under the model.
    """
    return math_ops.exp(self.log_pdf(x))

  def sample(self, n, seed=None):
    """Sample `n` observations from this Distribution.

    Args:
      n: Scalar int `Tensor`, the number of observations to sample.
      seed: Python integer, the random seed.

    Returns:
      samples: A vector of samples with shape `[n]`.
    """
    return random_ops.random_normal(
        shape=array_ops.expand_dims(n, 0), mean=self._mu,
        stddev=self._sigma, dtype=self._mu.dtype, seed=seed)
