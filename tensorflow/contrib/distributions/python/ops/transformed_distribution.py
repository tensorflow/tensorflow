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

from tensorflow.contrib.distributions.python.ops import distribution as distributions
from tensorflow.contrib.distributions.python.ops import distribution_util
from tensorflow.python.ops import math_ops


_condition_kwargs_dict = {
    "bijector_kwargs": ("Python dictionary of arg names/values "
                        "forwarded to the bijector."),
    "distribution_kwargs": ("Python dictionary of arg names/values "
                            "forwarded to the distribution."),
}


class TransformedDistribution(distributions.Distribution):
  """A Transformed Distribution.

  A `TransformedDistribution` models `p(y)` given a base distribution `p(x)`,
  and a deterministic, invertible, differentiable transform, `Y = g(X)`. The
  transform is typically an instance of the `Bijector` class and the base
  distribution is typically an instance of the `Distribution` class.

  A `Bijector` is expected to implement the following functions:
  - `forward`,
  - `inverse`,
  - `inverse_log_det_jacobian`.
  The semantics of these functions are outlined in the `Bijector` documentation.

  Shapes, type, and reparameterization are taken from the base distribution.

  Write `P(Y=y)` for cumulative density function of random variable (rv) `Y` and
  `p` for its derivative wrt to `Y`.  Assume that `Y=g(X)` where `g` is
  continuous and `X=g^{-1}(Y)`. Write `J` for the Jacobian (of some function).

  A `TransformedDistribution` alters the input/outputs of a `Distribution`
  associated with rv `X` in the following ways:

    * `sample`:

      Mathematically:

      ```none
      Y = g(X)
      ```

      Programmatically:

      ```python
      return bijector.forward(distribution.sample(...))
      ```

    * `log_prob`:

      Mathematically:

      ```none
      (log o p o g^{-1})(y) + (log o det o J o g^{-1})(y)
      ```

      Programmatically:

      ```python
      return (bijector.inverse_log_det_jacobian(x) +
              distribution.log_prob(bijector.inverse(x))
      ```

    * `log_cdf`:

      Mathematically:

      ```none
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
    distribution=ds.Normal(mu=mu, sigma=sigma),
    bijector=ds.bijector.Exp(),
    name="LogNormalTransformedDistribution")
  ```

  A `LogNormal` made from callables:

  ```python
  ds = tf.contrib.distributions
  log_normal = ds.TransformedDistribution(
    distribution=ds.Normal(mu=mu, sigma=sigma),
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
    distribution=ds.Normal(mu=0, sigma=1),
    bijector=ds.bijector.ScaleAndShift(loc=mu, scale=sigma, event_ndims=0),
    name="NormalTransformedDistribution")
  ```

  """

  def __init__(self,
               distribution,
               bijector,
               validate_args=False,
               name=None):
    """Construct a Transformed Distribution.

    Args:
      distribution: The base distribution class to transform. Typically an
        instance of `Distribution`.
      bijector: The object responsible for calculating the transformation.
        Typically an instance of `Bijector`.
      validate_args: Python boolean.  Whether to validate input with asserts.
        If `validate_args` is `False`, and the inputs are invalid,
        correct behavior is not guaranteed.
      name: The name for the distribution. Default:
        `bijector.name + distribution.name`.
    """
    parameters = locals()
    parameters.pop("self")
    name = name or bijector.name + distribution.name
    self._distribution = distribution
    self._bijector = bijector
    super(TransformedDistribution, self).__init__(
        dtype=self._distribution.dtype,
        is_continuous=self._distribution.is_continuous,
        is_reparameterized=self._distribution.is_reparameterized,
        validate_args=validate_args,
        allow_nan_stats=self._distribution.allow_nan_stats,
        parameters=parameters,
        # We let TransformedDistribution access _graph_parents since this class
        # is more like a baseclass than derived.
        graph_parents=(distribution._graph_parents +  # pylint: disable=protected-access
                       list(bijector.parameters.values())),
        name=name)

  @property
  def distribution(self):
    """Base distribution, p(x)."""
    return self._distribution

  @property
  def bijector(self):
    """Function transforming x => y."""
    return self._bijector

  def _event_shape(self):
    return self.bijector.forward_event_shape(
        self.distribution.event_shape())

  def _get_event_shape(self):
    return self.bijector.get_forward_event_shape(
        self.distribution.get_event_shape())

  def _batch_shape(self):
    return self.distribution.batch_shape()

  def _get_batch_shape(self):
    return self.distribution.get_batch_shape()

  @distribution_util.AppendDocstring(
      """Samples from the base distribution and then passes through
      the bijector's forward transform.""",
      condition_kwargs_dict=_condition_kwargs_dict)
  def _sample_n(self, n, seed=None,
                bijector_kwargs=None, distribution_kwargs=None):
    bijector_kwargs = bijector_kwargs or {}
    distribution_kwargs = distribution_kwargs or {}
    x = self.distribution.sample(sample_shape=n, seed=seed,
                                 **distribution_kwargs)
    # Recall that a bijector is named for its forward transform, i.e.,
    # `Y = g(X)`,
    return self.bijector.forward(x, **bijector_kwargs)

  @distribution_util.AppendDocstring(
      """Implements `(log o p o g^{-1})(y) + (log o det o J o g^{-1})(y)`,
      where `g^{-1}` is the inverse of `transform`.

      Also raises a `ValueError` if `inverse` was not provided to the
      distribution and `y` was not returned from `sample`.""",
      condition_kwargs_dict=_condition_kwargs_dict)
  def _log_prob(self, y, bijector_kwargs=None, distribution_kwargs=None):
    bijector_kwargs = bijector_kwargs or {}
    distribution_kwargs = distribution_kwargs or {}
    x, ildj = self.bijector.inverse_and_inverse_log_det_jacobian(
        y, **bijector_kwargs)
    return ildj + self.distribution.log_prob(x, **distribution_kwargs)

  @distribution_util.AppendDocstring(
      """Implements `p(g^{-1}(y)) det|J(g^{-1}(y))|`, where `g^{-1}` is the
      inverse of `transform`.

      Also raises a `ValueError` if `inverse` was not provided to the
      distribution and `y` was not returned from `sample`.""",
      condition_kwargs_dict=_condition_kwargs_dict)
  def _prob(self, y, bijector_kwargs=None, distribution_kwargs=None):
    bijector_kwargs = bijector_kwargs or {}
    distribution_kwargs = distribution_kwargs or {}
    x, ildj = self.bijector.inverse_and_inverse_log_det_jacobian(
        y, **bijector_kwargs)
    return math_ops.exp(ildj) * self.distribution.prob(x, **distribution_kwargs)

  @distribution_util.AppendDocstring(
      condition_kwargs_dict=_condition_kwargs_dict)
  def _log_cdf(self, y, bijector_kwargs=None, distribution_kwargs=None):
    bijector_kwargs = bijector_kwargs or {}
    distribution_kwargs = distribution_kwargs or {}
    x = self.bijector.inverse(y, **bijector_kwargs)
    return self.distribution.log_cdf(x, distribution_kwargs)

  @distribution_util.AppendDocstring(
      condition_kwargs_dict=_condition_kwargs_dict)
  def _cdf(self, y, bijector_kwargs=None, distribution_kwargs=None):
    bijector_kwargs = bijector_kwargs or {}
    distribution_kwargs = distribution_kwargs or {}
    x = self.bijector.inverse(y, **bijector_kwargs)
    return self.distribution.cdf(x, **distribution_kwargs)

  @distribution_util.AppendDocstring(
      condition_kwargs_dict=_condition_kwargs_dict)
  def _log_survival_function(self, y,
                             bijector_kwargs=None, distribution_kwargs=None):
    bijector_kwargs = bijector_kwargs or {}
    distribution_kwargs = distribution_kwargs or {}
    x = self.bijector.inverse(y, **bijector_kwargs)
    return self.distribution.log_survival_function(x, **distribution_kwargs)

  @distribution_util.AppendDocstring(
      condition_kwargs_dict=_condition_kwargs_dict)
  def _survival_function(self, y,
                         bijector_kwargs=None, distribution_kwargs=None):
    bijector_kwargs = bijector_kwargs or {}
    distribution_kwargs = distribution_kwargs or {}
    x = self.bijector.inverse(y, **bijector_kwargs)
    return self.distribution.survival_function(x, **distribution_kwargs)
