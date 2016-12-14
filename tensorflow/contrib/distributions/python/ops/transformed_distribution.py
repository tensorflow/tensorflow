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

import numpy as np

from tensorflow.contrib.distributions.python.ops import bijector as bijectors
from tensorflow.contrib.distributions.python.ops import distribution as distributions
from tensorflow.contrib.distributions.python.ops import distribution_util
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops

__all__ = [
    "TransformedDistribution",
]

_condition_kwargs_dict = {
    "bijector_kwargs": ("Python dictionary of arg names/values "
                        "forwarded to the bijector."),
    "distribution_kwargs": ("Python dictionary of arg names/values "
                            "forwarded to the distribution."),
}


# The following helper functions attempt to statically perform a TF operation.
# These functions make debugging easier since we can do more validation during
# graph construction.


def _logical_and(*args):
  """Convenience function which attempts to statically `reduce_all`."""
  args_static = [tensor_util.constant_value(x) for x in args]
  if any(x is None for x in args_static):
    return math_ops.reduce_all(args)
  return ops.convert_to_tensor(all(args_static), name="logical_and")


def _ones_like(x):
  """Convenience function attempts to statically construct `ones_like`."""
  # Should only be used for small vectors.
  if x.get_shape().is_fully_defined():
    return np.ones(x.get_shape().as_list(), dtype=x.dtype.as_numpy_dtype())
  return array_ops.ones_like(x)


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

  We now describe how a `TransformedDistribution` alters the input/outputs of a
  `Distribution` associated with a random variable (rv) `X`.

  Write `cdf(Y=y)` for an absolutely continuous cumulative distribution function
  of random variable `Y`; write the probability density function `pdf(Y=y) :=
  d^k / (dy_1,...,dy_k) cdf(Y=y)` for its derivative wrt to `Y` evaluated at
  `y`.  Assume that `Y = g(X)` where `g` is a deterministic diffeomorphism,
  i.e., a non-random, continuous, differentiable, and invertible function.
  Write the inverse of `g` as `X = g^{-1}(Y)` and `(J o g)(x)` for the Jacobian
  of `g` evaluated at `x`.

  A `TransformedDistribution` implements the following operations:

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
      (log o pdf)(Y=y) = (log o pdf o g^{-1})(y) + (log o det o J o g^{-1})(y)
      ```

      Programmatically:

      ```python
      return (bijector.inverse_log_det_jacobian(x) +
              distribution.log_prob(bijector.inverse(x))
      ```

    * `log_cdf`:

      Mathematically:

      ```none
      (log o cdf)(Y=y) = (log o cdf o g^{-1})(y)
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
        lambda y: -tf.reduce_sum(tf.log(y), reduction_indices=-1)),
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

  A `TransformedDistribution`'s batch- and event-shape are implied by the base
  distribution unless explicitly overridden by `batch_shape` or `event_shape`
  arguments.  Specifying an overriding `batch_shape` (`event_shape`) is
  permitted only if the base distribution has scalar batch-shape (event-shape).
  The bijector is applied to the distribution as if the distribution possessed
  the overridden shape(s). The following example demonstrates how to construct a
  multivariate Normal as a `TransformedDistribution`.

  ```python
  bs = tf.contrib.distributions.bijector
  ds = tf.contrib.distributions
  # We will create two MVNs with batch_shape = event_shape = 2.
  mean = [[-1., 0],      # batch:0
          [0., 1]]       # batch:1
  chol_cov = [[[1., 0],
               [0, 1]],  # batch:0
              [[1, 0],
               [2, 2]]]  # batch:1
  mvn1 = ds.TransformedDistribution(
      distribution=ds.Normal(mu=0., sigma=1.),
      bijector=bs.Affine(shift=mean, tril=chol_cov),
      batch_shape=[2],  # Valid because base_distribution.batch_shape == [].
      event_shape=[2])  # Valid because base_distribution.event_shape == [].
  mvn2 = ds.MultivariateNormalCholesky(mu=mean, chol=chol_cov)
  # mvn1.log_prob(x) == mvn2.log_prob(x)
  ```

  """

  def __init__(self,
               distribution,
               bijector=None,
               batch_shape=None,
               event_shape=None,
               validate_args=False,
               name=None):
    """Construct a Transformed Distribution.

    Args:
      distribution: The base distribution instance to transform. Typically an
        instance of `Distribution`.
      bijector: The object responsible for calculating the transformation.
        Typically an instance of `Bijector`. `None` means `Identity()`.
      batch_shape: `integer` vector `Tensor` which overrides `distribution`
        `batch_shape`; valid only if `distribution.is_scalar_batch` and
        `distribution.is_scalar_event`.
      event_shape: `integer` vector `Tensor` which overrides `distribution`
        `event_shape`; valid only if `distribution.is_scalar_batch` and
        `distribution.is_scalar_event`
      validate_args: Python `Boolean`.  Whether to validate input with asserts.
        If `validate_args` is `False`, and the inputs are invalid,
        correct behavior is not guaranteed.
      name: The name for the distribution. Default:
        `bijector.name + distribution.name`.
    """
    parameters = locals()
    parameters.pop("self")
    if bijector is None:
      bijector = bijectors.Identity(validate_args=validate_args)
    name = name or bijector.name + distribution.name
    with ops.name_scope(name, values=[event_shape, batch_shape]):
      if batch_shape is not None or event_shape is not None:
        is_scalar_batch_and_scalar_event = _logical_and(
            distribution.is_scalar_batch,
            distribution.is_scalar_event)
      if batch_shape is not None:
        batch_shape = self._maybe_validate_shape_override(
            ops.convert_to_tensor(batch_shape, name="batch_shape"),
            is_scalar_batch_and_scalar_event, validate_args)
      self._override_batch_shape = batch_shape

      if event_shape is not None:
        event_shape = self._maybe_validate_shape_override(
            ops.convert_to_tensor(event_shape, name="event_shape"),
            is_scalar_batch_and_scalar_event, validate_args)
        event_ndims = (event_shape.get_shape().ndims
                       if event_shape.get_shape().ndims is not None
                       else array_ops.rank(event_shape, "event_ndims"))
        self._reduce_event_indices = math_ops.range(-event_ndims, 0)
      self._override_event_shape = event_shape

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
                       bijector.graph_parents),
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
        self.distribution.event_shape()
        if self._override_event_shape is None
        else self._override_event_shape)

  def _get_event_shape(self):
    return self.bijector.get_forward_event_shape(
        self.distribution.get_event_shape()
        if self._override_event_shape is None
        else tensor_shape.TensorShape(
            tensor_util.constant_value(self._override_event_shape)))

  def _batch_shape(self):
    if self._override_batch_shape is None:
      return self.distribution.batch_shape()
    return self._override_batch_shape

  def _get_batch_shape(self):
    if self._override_batch_shape is None:
      return self.distribution.get_batch_shape()
    return tensor_shape.TensorShape(tensor_util.constant_value(
        self._override_batch_shape))

  @distribution_util.AppendDocstring(
      """Samples from the base distribution and then passes through
      the bijector's forward transform.""",
      condition_kwargs_dict=_condition_kwargs_dict)
  def _sample_n(self, n, seed=None,
                bijector_kwargs=None, distribution_kwargs=None):
    bijector_kwargs = bijector_kwargs or {}
    distribution_kwargs = distribution_kwargs or {}
    if (self._override_batch_shape is None and
        self._override_event_shape is None):
      sample_shape = [n]
    else:
      sample_shape = [[n]]
      if self._override_batch_shape is not None:
        sample_shape += [self._override_batch_shape]
      if self._override_event_shape is not None:
        sample_shape += [self._override_event_shape]
      sample_shape = array_ops.concat_v2(sample_shape, 0)
    x = self.distribution.sample(sample_shape=sample_shape, seed=seed,
                                 **distribution_kwargs)
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
    log_prob = self.distribution.log_prob(x, **distribution_kwargs)
    if self._override_event_shape is not None:
      log_prob = math_ops.reduce_sum(log_prob, self._reduce_event_indices)
    return ildj + log_prob

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
    prob = self.distribution.prob(x, **distribution_kwargs)
    if self._override_event_shape is not None:
      prob = math_ops.reduce_prod(prob, self._reduce_event_indices)
    return math_ops.exp(ildj) * prob

  @distribution_util.AppendDocstring(
      condition_kwargs_dict=_condition_kwargs_dict)
  def _log_cdf(self, y, bijector_kwargs=None, distribution_kwargs=None):
    if self._override_event_shape is not None:
      raise NotImplementedError("log_cdf is not implemented when overriding "
                                "event_shape")
    bijector_kwargs = bijector_kwargs or {}
    distribution_kwargs = distribution_kwargs or {}
    x = self.bijector.inverse(y, **bijector_kwargs)
    return self.distribution.log_cdf(x, **distribution_kwargs)

  @distribution_util.AppendDocstring(
      condition_kwargs_dict=_condition_kwargs_dict)
  def _cdf(self, y, bijector_kwargs=None, distribution_kwargs=None):
    if self._override_event_shape is not None:
      raise NotImplementedError("cdf is not implemented when overriding "
                                "event_shape")
    bijector_kwargs = bijector_kwargs or {}
    distribution_kwargs = distribution_kwargs or {}
    x = self.bijector.inverse(y, **bijector_kwargs)
    return self.distribution.cdf(x, **distribution_kwargs)

  @distribution_util.AppendDocstring(
      condition_kwargs_dict=_condition_kwargs_dict)
  def _log_survival_function(self, y,
                             bijector_kwargs=None, distribution_kwargs=None):
    if self._override_event_shape is not None:
      raise NotImplementedError("log_survival_function is not implemented when "
                                "overriding event_shape")
    bijector_kwargs = bijector_kwargs or {}
    distribution_kwargs = distribution_kwargs or {}
    x = self.bijector.inverse(y, **bijector_kwargs)
    return self.distribution.log_survival_function(x, **distribution_kwargs)

  @distribution_util.AppendDocstring(
      condition_kwargs_dict=_condition_kwargs_dict)
  def _survival_function(self, y,
                         bijector_kwargs=None, distribution_kwargs=None):
    if self._override_event_shape is not None:
      raise NotImplementedError("survival_function is not implemented when "
                                "overriding event_shape")
    bijector_kwargs = bijector_kwargs or {}
    distribution_kwargs = distribution_kwargs or {}
    x = self.bijector.inverse(y, **bijector_kwargs)
    return self.distribution.survival_function(x, **distribution_kwargs)

  def _entropy(self):
    if (not self.distribution.is_continuous or
        not self.bijector.is_constant_jacobian):
      raise NotImplementedError("entropy is not implemented")
    # Suppose Y = g(X) where g is a diffeomorphism and X is a continuous rv. It
    # can be shown that:
    #   H[Y] = H[X] + E_X[(log o det o Jacobian o g)(X)].
    # If is_constant_jacobian then:
    #   E_X[(log o det o Jacobian o g)(X)] = (log o det o Jacobian o g)(c)
    # where c can by anything.
    entropy = self.distribution.entropy()
    if self._override_event_shape is not None:
      # H[X] = sum_i H[X_i] if X_i are mutually independent.
      # This means that a reduce_sum is a simple rescaling.
      entropy *= math_ops.cast(math_ops.reduce_prod(self._override_event_shape),
                               dtype=entropy.dtype.base_dtype)
    if self._override_batch_shape is not None:
      entropy = array_ops.reshape(entropy,
                                  _ones_like(self._override_batch_shape))
      entropy = array_ops.tile(entropy, self._override_batch_shape)
    dummy = 0.
    return entropy - self.bijector.inverse_log_det_jacobian(dummy)

  def _maybe_validate_shape_override(self, override_shape, base_is_scalar,
                                     validate_args):
    """Helper to __init__ which ensures override batch/event_shape are valid."""
    if not override_shape.dtype.is_integer:
      raise TypeError("shape override must be an integer")

    if override_shape.get_shape().ndims is not None:
      if override_shape.get_shape().ndims != 1:
        raise ValueError("shape override must be a vector")
    elif validate_args:
      is_vector = check_ops.assert_rank(
          override_shape, 1,
          message="shape override must be a vector")
      override_shape = control_flow_ops.with_dependencies(
          [is_vector], override_shape)

    if override_shape.get_shape().is_fully_defined():
      if any(s <= 0 for s in override_shape.get_shape().as_list()):
        raise ValueError("shape override must have positive elements")
    elif validate_args:
      is_positive = check_ops.assert_positive(
          override_shape,
          message="shape override must have positive elements")
      override_shape = control_flow_ops.with_dependencies(
          [is_positive], override_shape)

    if tensor_util.constant_value(base_is_scalar) is not None:
      if not tensor_util.constant_value(base_is_scalar):
        raise ValueError("shape override requires scalar distribution.")
    elif validate_args:
      is_scalar = check_ops.assert_equal(
          base_is_scalar, True,
          message="shape override requires scalar distribution.")
      override_shape = control_flow_ops.with_dependencies(
          [is_scalar], override_shape)

    return override_shape
