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
"""The Gaussian distribution: conjugate posterior closed form calculations.

@@known_sigma_posterior
@@known_sigma_predictive
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.distributions.python.ops.gaussian import Gaussian  # pylint: disable=line-too-long

from tensorflow.python.ops import math_ops


def known_sigma_posterior(prior, sigma, s, n):
  """Return the conjugate posterior distribution with known sigma.

  Accepts a prior Gaussian distribution, having parameters `mu0` and `sigma0`,
  a known `sigma` of the predictive distribution (also assumed Gaussian),
  and statistical estimates `s` (the sum of the observations) and
  `n` (the number of observations).

  Returns a posterior (also Gaussian) distribution object, with parameters
  `(mu', sigma'^2)`, where:
  ```
  sigma'^2 = 1/(1/sigma0^2 + n/sigma^2),
  mu' = (mu0/sigma0^2 + s/sigma^2) * sigma'^2.
  ```

  Args:
    prior: `Normal` object of type `dtype`, the prior distribution having
      parameters `(mu0, sigma0)`.
    sigma: Scalar of type `dtype`, `sigma > 0`.  The known stddev parameter.
    s: Scalar, of type `dtype`, the sum of observations.
    n: Scalar int, the number of observations.

  Returns:
    A new Gaussian posterior distribution.

  Raises:
    TypeError: if dtype of `s` does not match `dtype`, or `prior` is not a
      Gaussian object.
  """
  if not isinstance(prior, Gaussian):
    raise TypeError("Expected prior to be an instance of type Gaussian")

  if s.dtype != prior.dtype:
    raise TypeError(
        "Observation sum s.dtype does not match prior dtype: %s vs. %s"
        % (s.dtype, prior.dtype))

  n = math_ops.cast(n, prior.dtype)
  sigma0_2 = math_ops.square(prior.sigma)
  sigma_2 = math_ops.square(sigma)
  sigmap_2 = 1.0/(1/sigma0_2 + n/sigma_2)
  return Gaussian(
      mu=(prior.mu/sigma0_2 + s/sigma_2) * sigmap_2,
      sigma=math_ops.sqrt(sigmap_2))


def known_sigma_predictive(prior, sigma, s, n):
  """Return the posterior predictive distribution with known sigma.

  Accepts a prior Gaussian distribution, having parameters `mu0` and `sigma0`,
  a known `sigma` of the predictive distribution (also assumed Gaussian),
  and statistical estimates `s` (the sum of the observations) and
  `n` (the number of observations).

  Calculates the Gaussian distribution p(x | sigma):
  ```
    p(x | sigma) = int N(x | mu, sigma^2) N(mu | prior.mu, prior.sigma^2) dmu
                 = N(x | prior.mu, 1/(sigma^2 + prior.sigma^2))
  ```

  Returns the predictive posterior distribution object, with parameters
  `(mu', sigma'^2)`, where:
  ```
  sigma_n^2 = 1/(1/sigma0^2 + n/sigma^2),
  mu' = (mu0/sigma0^2 + s/sigma^2) * sigma_n^2.
  sigma'^2 = sigma_n^2 + sigma^2,
  ```

  Args:
    prior: `Normal` object of type `dtype`, the prior distribution having
      parameters `(mu0, sigma0)`.
    sigma: Scalar of type `dtype`, `sigma > 0`.  The known stddev parameter.
    s: Scalar, of type `dtype`, the sum of observations.
    n: Scalar int, the number of observations.

  Returns:
    A new Gaussian posterior distribution.

  Raises:
    TypeError: if dtype of `s` does not match `dtype`, or `prior` is not a
      Gaussian object.
  """
  if not isinstance(prior, Gaussian):
    raise TypeError("Expected prior to be an instance of type Gaussian")

  if s.dtype != prior.dtype:
    raise TypeError(
        "Observation sum s.dtype does not match prior dtype: %s vs. %s"
        % (s.dtype, prior.dtype))

  n = math_ops.cast(n, prior.dtype)
  sigma0_2 = math_ops.square(prior.sigma)
  sigma_2 = math_ops.square(sigma)
  sigmap_2 = 1.0/(1/sigma0_2 + n/sigma_2)
  return Gaussian(
      mu=(prior.mu/sigma0_2 + s/sigma_2) * sigmap_2,
      sigma=math_ops.sqrt(sigmap_2 + sigma_2))
