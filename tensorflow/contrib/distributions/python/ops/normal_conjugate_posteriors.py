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
"""The Normal distribution: conjugate posterior closed form calculations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.distributions.python.ops.normal import Normal  # pylint: disable=line-too-long

from tensorflow.python.ops import math_ops


def normal_conjugates_known_scale_posterior(prior, scale, s, n):
  """Posterior Normal distribution with conjugate prior on the mean.

  This model assumes that `n` observations (with sum `s`) come from a
  Normal with unknown mean `loc` (described by the Normal `prior`)
  and known variance `scale^2`.  The "known scale posterior" is
  the distribution of the unknown `loc`.

  Accepts a prior Normal distribution object, having parameters
  `loc0` and `scale0`, as well as known `scale` values of the predictive
  distribution(s) (also assumed Normal),
  and statistical estimates `s` (the sum(s) of the observations) and
  `n` (the number(s) of observations).

  Returns a posterior (also Normal) distribution object, with parameters
  `(loc', scale'^2)`, where:

  ```
  mu ~ N(mu', sigma'^2)
  sigma'^2 = 1/(1/sigma0^2 + n/sigma^2),
  mu' = (mu0/sigma0^2 + s/sigma^2) * sigma'^2.
  ```

  Distribution parameters from `prior`, as well as `scale`, `s`, and `n`.
  will broadcast in the case of multidimensional sets of parameters.

  Args:
    prior: `Normal` object of type `dtype`:
      the prior distribution having parameters `(loc0, scale0)`.
    scale: tensor of type `dtype`, taking values `scale > 0`.
      The known stddev parameter(s).
    s: Tensor of type `dtype`.  The sum(s) of observations.
    n: Tensor of type `int`.  The number(s) of observations.

  Returns:
    A new Normal posterior distribution object for the unknown observation
    mean `loc`.

  Raises:
    TypeError: if dtype of `s` does not match `dtype`, or `prior` is not a
      Normal object.
  """
  if not isinstance(prior, Normal):
    raise TypeError("Expected prior to be an instance of type Normal")

  if s.dtype != prior.dtype:
    raise TypeError(
        "Observation sum s.dtype does not match prior dtype: %s vs. %s"
        % (s.dtype, prior.dtype))

  n = math_ops.cast(n, prior.dtype)
  scale0_2 = math_ops.square(prior.scale)
  scale_2 = math_ops.square(scale)
  scalep_2 = 1.0/(1/scale0_2 + n/scale_2)
  return Normal(
      loc=(prior.loc/scale0_2 + s/scale_2) * scalep_2,
      scale=math_ops.sqrt(scalep_2))


def normal_conjugates_known_scale_predictive(prior, scale, s, n):
  """Posterior predictive Normal distribution w. conjugate prior on the mean.

  This model assumes that `n` observations (with sum `s`) come from a
  Normal with unknown mean `loc` (described by the Normal `prior`)
  and known variance `scale^2`.  The "known scale predictive"
  is the distribution of new observations, conditioned on the existing
  observations and our prior.

  Accepts a prior Normal distribution object, having parameters
  `loc0` and `scale0`, as well as known `scale` values of the predictive
  distribution(s) (also assumed Normal),
  and statistical estimates `s` (the sum(s) of the observations) and
  `n` (the number(s) of observations).

  Calculates the Normal distribution(s) `p(x | sigma^2)`:

  ```
  p(x | sigma^2) = int N(x | mu, sigma^2) N(mu | prior.loc, prior.scale**2) dmu
                 = N(x | prior.loc, 1/(sigma^2 + prior.scale**2))
  ```

  Returns the predictive posterior distribution object, with parameters
  `(loc', scale'^2)`, where:

  ```
  sigma_n^2 = 1/(1/sigma0^2 + n/sigma^2),
  mu' = (mu0/sigma0^2 + s/sigma^2) * sigma_n^2.
  sigma'^2 = sigma_n^2 + sigma^2,
  ```

  Distribution parameters from `prior`, as well as `scale`, `s`, and `n`.
  will broadcast in the case of multidimensional sets of parameters.

  Args:
    prior: `Normal` object of type `dtype`:
      the prior distribution having parameters `(loc0, scale0)`.
    scale: tensor of type `dtype`, taking values `scale > 0`.
      The known stddev parameter(s).
    s: Tensor of type `dtype`.  The sum(s) of observations.
    n: Tensor of type `int`.  The number(s) of observations.

  Returns:
    A new Normal predictive distribution object.

  Raises:
    TypeError: if dtype of `s` does not match `dtype`, or `prior` is not a
      Normal object.
  """
  if not isinstance(prior, Normal):
    raise TypeError("Expected prior to be an instance of type Normal")

  if s.dtype != prior.dtype:
    raise TypeError(
        "Observation sum s.dtype does not match prior dtype: %s vs. %s"
        % (s.dtype, prior.dtype))

  n = math_ops.cast(n, prior.dtype)
  scale0_2 = math_ops.square(prior.scale)
  scale_2 = math_ops.square(scale)
  scalep_2 = 1.0/(1/scale0_2 + n/scale_2)
  return Normal(
      loc=(prior.loc/scale0_2 + s/scale_2) * scalep_2,
      scale=math_ops.sqrt(scalep_2 + scale_2))
