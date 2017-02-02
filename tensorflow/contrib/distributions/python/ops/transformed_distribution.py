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
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
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


# The following helper functions attempt to statically perform a TF operation.
# These functions make debugging easier since we can do more validation during
# graph construction.


def _static_value(x):
  """Returns the static value of a `Tensor` or `None`."""
  return tensor_util.constant_value(ops.convert_to_tensor(x))


def _logical_and(*args):
  """Convenience function which attempts to statically `reduce_all`."""
  args_ = [_static_value(x) for x in args]
  if any(x is not None and not bool(x) for x in args_):
    return constant_op.constant(False)
  if all(x is not None and bool(x) for x in args_):
    return constant_op.constant(True)
  if len(args) == 2:
    return math_ops.logical_and(*args)
  return math_ops.reduce_all(args)


def _logical_equal(x, y):
  """Convenience function which attempts to statically compute `x == y`."""
  x_ = _static_value(x)
  y_ = _static_value(y)
  if x_ is None or y_ is None:
    return math_ops.equal(x, y)
  return constant_op.constant(np.array_equal(x_, y_))


def _logical_not(x):
  """Convenience function which attempts to statically apply `logical_not`."""
  x_ = _static_value(x)
  if x_ is None:
    return math_ops.logical_not(x)
  return constant_op.constant(np.logical_not(x_))


def _concat_vectors(*args):
  """Convenience function which concatenates input vectors."""
  args_ = [_static_value(x) for x in args]
  if any(x_ is None for x_ in args_):
    return array_ops.concat(args, 0)
  return constant_op.constant([x_ for vec_ in args_ for x_ in vec_])


def _pick_scalar_condition(pred, cond_true, cond_false):
  """Convenience function which chooses the condition based on the predicate."""
  # Note: This function is only valid if all of pred, cond_true, and cond_false
  # are scalars. This means its semantics are arguably more like tf.cond than
  # tf.select even though we use tf.select to implement it.
  pred_ = _static_value(pred)
  if pred_ is None:
    return array_ops.where(pred, cond_true, cond_false)
  return cond_true if pred_ else cond_false


def _ones_like(x):
  """Convenience function attempts to statically construct `ones_like`."""
  # Should only be used for small vectors.
  if x.get_shape().is_fully_defined():
    return array_ops.ones(x.get_shape().as_list(), dtype=x.dtype)
  return array_ops.ones_like(x)


def _ndims_from_shape(shape):
  """Returns `Tensor`'s `rank` implied by a `Tensor` shape."""
  if shape.get_shape().ndims not in (None, 1):
    raise ValueError("input is not a valid shape: not 1D")
  if not shape.dtype.is_integer:
    raise TypeError("input is not a valid shape: wrong dtype")
  if shape.get_shape().is_fully_defined():
    return constant_op.constant(shape.get_shape().as_list()[0])
  return array_ops.shape(shape)[0]


def _is_scalar_from_shape(shape):
  """Returns `True` `Tensor` if `Tensor` shape implies a scalar."""
  return _logical_equal(_ndims_from_shape(shape), 0)


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
      (log o pdf)(Y=y) = (log o pdf o g^{-1})(y) +
                           (log o abs o det o J o g^{-1})(y)
      ```

      Programmatically:

      ```python
      return (distribution.log_prob(bijector.inverse(x)) +
              bijector.inverse_log_det_jacobian(x))
      ```

    * `log_cdf`:

      Mathematically:

      ```none
      (log o cdf)(Y=y) = (log o cdf o g^{-1})(y)
      ```

      Programmatically:

      ```python
      return distribution.log_cdf(bijector.inverse(x))
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
        `batch_shape`; valid only if `distribution.is_scalar_batch()`.
      event_shape: `integer` vector `Tensor` which overrides `distribution`
        `event_shape`; valid only if `distribution.is_scalar_event()`.
      validate_args: Python `Boolean`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
      name: `String` name prefixed to Ops created by this class. Default:
        `bijector.name + distribution.name`.
    """
    parameters = locals()
    name = name or (("" if bijector is None else bijector.name) +
                    distribution.name)
    with ops.name_scope(name, values=[event_shape, batch_shape]):
      # For convenience we define some handy constants.
      self._zero = constant_op.constant(0, dtype=dtypes.int32, name="zero")
      self._empty = constant_op.constant([], dtype=dtypes.int32, name="empty")

      if bijector is None:
        bijector = bijectors.Identity(validate_args=validate_args)

      # We will keep track of a static and dynamic version of
      # self._is_{batch,event}_override. This way we can do more prior to graph
      # execution, including possibly raising Python exceptions.

      self._override_batch_shape = self._maybe_validate_shape_override(
          batch_shape, distribution.is_scalar_batch(), validate_args,
          "batch_shape")
      self._is_batch_override = _logical_not(_logical_equal(
          _ndims_from_shape(self._override_batch_shape), self._zero))
      self._is_maybe_batch_override = bool(
          tensor_util.constant_value(self._override_batch_shape) is None or
          tensor_util.constant_value(self._override_batch_shape))

      self._override_event_shape = self._maybe_validate_shape_override(
          event_shape, distribution.is_scalar_event(), validate_args,
          "event_shape")
      self._is_event_override = _logical_not(_logical_equal(
          _ndims_from_shape(self._override_event_shape), self._zero))
      self._is_maybe_event_override = bool(
          tensor_util.constant_value(self._override_event_shape) is None or
          tensor_util.constant_value(self._override_event_shape))

      # To convert a scalar distribution into a multivariate distribution we
      # will draw dims from the sample dims, which are otherwise iid. This is
      # easy to do except in the case that the base distribution has batch dims
      # and we're overriding event shape.  When that case happens the event dims
      # will incorrectly be to the left of the batch dims. In this case we'll
      # cyclically permute left the new dims.
      self._needs_rotation = _logical_and(
          self._is_event_override,
          _logical_not(self._is_batch_override),
          _logical_not(distribution.is_scalar_batch()))
      override_event_ndims = _ndims_from_shape(self._override_event_shape)
      self._rotate_ndims = _pick_scalar_condition(
          self._needs_rotation, override_event_ndims, 0)
      # We'll be reducing the head dims (if at all), i.e., this will be []
      # if we don't need to reduce.
      self._reduce_event_indices = math_ops.range(
          self._rotate_ndims - override_event_ndims, self._rotate_ndims)

    self._distribution = distribution
    self._bijector = bijector
    super(TransformedDistribution, self).__init__(
        dtype=self._distribution.dtype,
        is_continuous=self._distribution.is_continuous,
        reparameterization_type=self._distribution.reparameterization_type,
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

  def _event_shape_tensor(self):
    return self.bijector.forward_event_shape_tensor(
        distribution_util.pick_vector(
            self._is_event_override,
            self._override_event_shape,
            self.distribution.event_shape_tensor()))

  def _event_shape(self):
    static_override = tensor_util.constant_value(self._override_event_shape)
    return self.bijector.forward_event_shape(
        self.distribution.event_shape
        if static_override is not None and not static_override
        else tensor_shape.TensorShape(static_override))

  def _batch_shape_tensor(self):
    return distribution_util.pick_vector(
        self._is_batch_override,
        self._override_batch_shape,
        self.distribution.batch_shape_tensor())

  def _batch_shape(self):
    static_override = tensor_util.constant_value(self._override_batch_shape)
    if static_override is not None and not static_override:
      return self.distribution.batch_shape
    return tensor_shape.TensorShape(static_override)

  @distribution_util.AppendDocstring(
      """Samples from the base distribution and then passes through
      the bijector's forward transform.""")
  def _sample_n(self, n, seed=None):
    sample_shape = _concat_vectors(
        distribution_util.pick_vector(self._needs_rotation, self._empty, [n]),
        self._override_batch_shape,
        self._override_event_shape,
        distribution_util.pick_vector(self._needs_rotation, [n], self._empty))
    x = self.distribution.sample(sample_shape=sample_shape, seed=seed)
    x = self._maybe_rotate_dims(x)
    return self.bijector.forward(x)

  @distribution_util.AppendDocstring(
      """Implements `(log o p o g^{-1})(y) + (log o abs o det o J o g^{-1})(y)`,
      where `g^{-1}` is the inverse of `transform`.

      Also raises a `ValueError` if `inverse` was not provided to the
      distribution and `y` was not returned from `sample`.""")
  def _log_prob(self, y):
    x, ildj = self.bijector.inverse_and_inverse_log_det_jacobian(y)
    x = self._maybe_rotate_dims(x, rotate_right=True)
    log_prob = self.distribution.log_prob(x)
    if self._is_maybe_event_override:
      log_prob = math_ops.reduce_sum(log_prob, self._reduce_event_indices)
    return ildj + log_prob

  @distribution_util.AppendDocstring(
      """Implements `p(g^{-1}(y)) det|J(g^{-1}(y))|`, where `g^{-1}` is the
      inverse of `transform`.

      Also raises a `ValueError` if `inverse` was not provided to the
      distribution and `y` was not returned from `sample`.""")
  def _prob(self, y):
    x, ildj = self.bijector.inverse_and_inverse_log_det_jacobian(y)
    x = self._maybe_rotate_dims(x, rotate_right=True)
    prob = self.distribution.prob(x)
    if self._is_maybe_event_override:
      prob = math_ops.reduce_prod(prob, self._reduce_event_indices)
    return math_ops.exp(ildj) * prob

  def _log_cdf(self, y):
    if self._is_maybe_event_override:
      raise NotImplementedError("log_cdf is not implemented when overriding "
                                "event_shape")
    x = self.bijector.inverse(y)
    return self.distribution.log_cdf(x)

  def _cdf(self, y):
    if self._is_maybe_event_override:
      raise NotImplementedError("cdf is not implemented when overriding "
                                "event_shape")
    x = self.bijector.inverse(y)
    return self.distribution.cdf(x)

  def _log_survival_function(self, y):
    if self._is_maybe_event_override:
      raise NotImplementedError("log_survival_function is not implemented when "
                                "overriding event_shape")
    x = self.bijector.inverse(y)
    return self.distribution.log_survival_function(x)

  def _survival_function(self, y):
    if self._is_maybe_event_override:
      raise NotImplementedError("survival_function is not implemented when "
                                "overriding event_shape")
    x = self.bijector.inverse(y)
    return self.distribution.survival_function(x)

  def _entropy(self):
    if (not self.distribution.is_continuous or
        not self.bijector.is_constant_jacobian):
      raise NotImplementedError("entropy is not implemented")
    # Suppose Y = g(X) where g is a diffeomorphism and X is a continuous rv. It
    # can be shown that:
    #   H[Y] = H[X] + E_X[(log o abs o det o J o g)(X)].
    # If is_constant_jacobian then:
    #   E_X[(log o abs o det o J o g)(X)] = (log o abs o det o J o g)(c)
    # where c can by anything.
    entropy = self.distribution.entropy()
    if self._is_maybe_event_override:
      # H[X] = sum_i H[X_i] if X_i are mutually independent.
      # This means that a reduce_sum is a simple rescaling.
      entropy *= math_ops.cast(math_ops.reduce_prod(self._override_event_shape),
                               dtype=entropy.dtype.base_dtype)
    if self._is_maybe_batch_override:
      new_shape = array_ops.concat([
          _ones_like(self._override_batch_shape),
          self.distribution.batch_shape_tensor()
      ], 0)
      entropy = array_ops.reshape(entropy, new_shape)
      multiples = array_ops.concat([
          self._override_batch_shape,
          _ones_like(self.distribution.batch_shape_tensor())
      ], 0)
      entropy = array_ops.tile(entropy, multiples)
    dummy = 0.
    return entropy - self.bijector.inverse_log_det_jacobian(dummy)

  def _maybe_validate_shape_override(self, override_shape, base_is_scalar,
                                     validate_args, name):
    """Helper to __init__ which ensures override batch/event_shape are valid."""
    if override_shape is None:
      override_shape = []

    override_shape = ops.convert_to_tensor(override_shape, dtype=dtypes.int32,
                                           name=name)

    if not override_shape.dtype.is_integer:
      raise TypeError("shape override must be an integer")

    override_is_scalar = _is_scalar_from_shape(override_shape)
    if tensor_util.constant_value(override_is_scalar):
      return self._empty

    dynamic_assertions = []

    if override_shape.get_shape().ndims is not None:
      if override_shape.get_shape().ndims != 1:
        raise ValueError("shape override must be a vector")
    elif validate_args:
      dynamic_assertions += [check_ops.assert_rank(
          override_shape, 1,
          message="shape override must be a vector")]

    if tensor_util.constant_value(override_shape) is not None:
      if any(s <= 0 for s in tensor_util.constant_value(override_shape)):
        raise ValueError("shape override must have positive elements")
    elif validate_args:
      dynamic_assertions += [check_ops.assert_positive(
          override_shape,
          message="shape override must have positive elements")]

    is_both_nonscalar = _logical_and(_logical_not(base_is_scalar),
                                     _logical_not(override_is_scalar))
    if tensor_util.constant_value(is_both_nonscalar) is not None:
      if tensor_util.constant_value(is_both_nonscalar):
        raise ValueError("base distribution not scalar")
    elif validate_args:
      dynamic_assertions += [check_ops.assert_equal(
          is_both_nonscalar, False,
          message="base distribution not scalar")]

    if not dynamic_assertions:
      return override_shape
    return control_flow_ops.with_dependencies(
        dynamic_assertions, override_shape)

  def _maybe_rotate_dims(self, x, rotate_right=False):
    """Helper which rolls left event_dims left or right event_dims right."""
    if tensor_util.constant_value(self._needs_rotation) is False:
      return x
    ndims = array_ops.rank(x)
    n = (ndims - self._rotate_ndims) if rotate_right else self._rotate_ndims
    return array_ops.transpose(
        x, _concat_vectors(math_ops.range(n, ndims), math_ops.range(0, n)))
