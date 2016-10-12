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
"""A Transformed Distribution class."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.distributions.python.ops import distribution
from tensorflow.contrib.distributions.python.ops import distribution_util
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops


class TransformedDistribution(distribution.Distribution):
  """A Transformed Distribution.

  A Transformed Distribution models `p(y)` given a base distribution `p(x)`, and
  a deterministic, invertible, differentiable transform, `Y = g(X)`. The
  transform is typically an instance of the `Bijector` class and the base
  distribution is typically an instance of the `Distribution` class.

  Shapes, type, and reparameterization are taken from the base distribution.

  Write `P(Y=y)` for cumulative density function of random variable (rv) `Y` and
  `p` for its derivative wrt to `Y`.  Assume that `Y=g(X)` where `g` is
  continuous and `X=g^{-1}(Y)`. Write `J` for the Jacobian (of some function).

  A `TransformedDistribution` alters the input/outputs of a `Distribution`
  associated with rv `X` in the following ways:

    * `sample`:

      Mathematically:

      ```
      Y = g(X)
      ```

      Programmatically:

      ```python
      return bijector.forward(distribution.sample(...))
      ```

    * `log_prob`:

      Mathematically:

      ```
      (log o p o g^{-1})(y) + (log o det o J o g^{-1})(y)
      ```

      Programmatically:

      ```python
      return (bijector.inverse_log_det_jacobian(x) +
              distribution.log_prob(bijector.inverse(x))
      ```

    * `log_cdf`:

      Mathematically:

      ```
      (log o P o g^{-1})(y)
      ```

      Programmatically:

      ```python
      return distribution.log_prob(bijector.inverse(x))
      ```

    * and similarly for: `cdf`, `prob`, `log_survival_function`,
     `survival_function`.

  A simple example constructing a Log-Normal distribution from a Normal
  distribution:

  ```python
  ds = tf.contrib.distributions
  log_normal = ds.TransformedDistribution(
    base_distribution=ds.Normal(mu=mu, sigma=sigma),
    bijector=ds.bijector.Exp(),
    name="LogNormalTransformedDistribution")
  ```

  A `LogNormal` made from callables:

  ```python
  ds = tf.contrib.distributions
  log_normal = ds.TransformedDistribution(
    base_distribution=ds.Normal(mu=mu, sigma=sigma),
    bijector=ds.bijector.Inline(
      forward_fn=tf.exp,
      inverse_fn=tf.log,
      inverse_log_det_jacobian_fn=(
        lambda y: -tf.reduce_sum(tf.log(x), reduction_indices=-1)),
    name="LogNormalTransformedDistribution")
  ```

  Another example constructing a Normal from a StandardNormal:

  ```python
  ds = tf.contrib.distributions
  normal = ds.TransformedDistribution(
    base_distribution=ds.Normal(mu=0, sigma=1),
    bijector=ds.bijector.ScaleAndShift(loc=mu, scale=sigma, event_ndims=0),
    name="NormalTransformedDistribution")
  ```

  """

  def __init__(self,
               base_distribution,
               bijector,
               name="TransformedDistribution"):
    """Construct a Transformed Distribution.

    Args:
      base_distribution: The base distribution class to transform. Typically an
        instance of `Distribution`.
      bijector: The object responsible for calculating the transformation.
        Typically an instance of `Bijector`.
      name: The name for the distribution.
    """
    with ops.name_scope(name) as ns:
      self._base_distribution = base_distribution
      self._bijector = bijector
      self._inverse_cache = {}
      super(TransformedDistribution, self).__init__(
          dtype=self._base_distribution.dtype,
          parameters={"base_distribution": base_distribution,
                      "bijector": bijector},
          is_continuous=self._base_distribution.is_continuous,
          is_reparameterized=self._base_distribution.is_reparameterized,
          validate_args=self._base_distribution.validate_args,
          allow_nan_stats=self._base_distribution.allow_nan_stats,
          name=ns)

  @property
  def base_distribution(self):
    """Base distribution, p(x)."""
    return self._base_distribution

  @property
  def bijector(self):
    """Function transforming x => y."""
    return self._bijector

  def _batch_shape(self):
    return self.base_distribution.batch_shape()

  def _get_batch_shape(self):
    return self.base_distribution.get_batch_shape()

  def _event_shape(self):
    return self.base_distribution.event_shape()

  def _get_event_shape(self):
    return self.base_distribution.get_event_shape()

  @distribution_util.AppendDocstring(
      """Samples from the base distribution and then passes through
      the bijector's forward transform.""")
  def _sample_n(self, n, seed=None):
    raw_samples = self.base_distribution.sample_n(n=n, seed=seed)
    samples = self.bijector.forward(raw_samples)
    self._inverse_cache[samples] = raw_samples
    return samples

  @distribution_util.AppendDocstring(
      """Implements `(log o p o g^{-1})(y) + (log o det o J o g^{-1})(y)`,
      where `g^{-1}` is the inverse of `transform`.

      Also raises a `ValueError` if `inverse` was not provided to the
      distribution and `y` was not returned from `sample`.""")
  def _log_prob(self, y):
    x = self._inverse_possibly_from_cache(y)
    inverse_log_det_jacobian = self.bijector.inverse_log_det_jacobian(y)
    return self.base_distribution.log_prob(x) + inverse_log_det_jacobian

  @distribution_util.AppendDocstring(
      """Implements `p(g^{-1}(y)) det|J(g^{-1}(y))|`, where `g^{-1}` is the
      inverse of `transform`.

      Also raises a `ValueError` if `inverse` was not provided to the
      distribution and `y` was not returned from `sample`.""")
  def _prob(self, y):
    return math_ops.exp(self._log_prob(y))

  def _log_cdf(self, y):
    x = self._inverse_possibly_from_cache(y)
    return self.base_distribution.log_cdf(x)

  def _cdf(self, y):
    x = self._inverse_possibly_from_cache(y)
    return self.base_distribution.cdf(x)

  def _log_survival_function(self, y):
    x = self._inverse_possibly_from_cache(y)
    return self.base_distribution.log_survival_function(x)

  def _survival_function(self, y):
    x = self._inverse_possibly_from_cache(y)
    return self.base_distribution.survival_function(x)

  def _inverse_possibly_from_cache(self, y):
    """Return `self._inverse(y)`, possibly using cached value."""
    y = ops.convert_to_tensor(y, name="y")
    if y in self._inverse_cache:
      return self._inverse_cache[y]
    else:
      return self.bijector.inverse(y)
