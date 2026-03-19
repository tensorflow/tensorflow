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
import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.distributions import distribution as distribution_lib
from tensorflow.python.ops.distributions import identity_bijector
from tensorflow.python.ops.distributions import util as distribution_util
from tensorflow.python.util import deprecation

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
    return array_ops.where_v2(pred, cond_true, cond_false)
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


class TransformedDistribution(distribution_lib.Distribution):
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
  `y`. Assume that `Y = g(X)` where `g` is a deterministic diffeomorphism,
  i.e., a non-random, continuous, differentiable, and invertible function.
  Write the inverse of `g` as `X = g^{-1}(Y)` and `(J o g)(x)` for the Jacobian
  of `g` evaluated at `x`.

  A `TransformedDistribution` implements the following operations:

    * `sample`
      Mathematically:   `Y = g(X)`
      Programmatically: `bijector.forward(distribution.sample(...))`

    * `log_prob`
      Mathematically:   `(log o pdf)(Y=y) = (log o pdf o g^{-1})(y)
                         + (log o abs o det o J o g^{-1})(y)`
      Programmatically: `(distribution.log_prob(bijector.inverse(y))
                         + bijector.inverse_log_det_jacobian(y))`

    * `log_cdf`
      Mathematically:   `(log o cdf)(Y=y) = (log o cdf o g^{-1})(y)`
      Programmatically: `distribution.log_cdf(bijector.inverse(x))`

    * and similarly for: `cdf`, `prob`, `log_survival_function`,
     `survival_function`.

  A simple example constructing a Log-Normal distribution from a Normal
  distribution:

  ```python
  ds = tfp.distributions
  log_normal = ds.TransformedDistribution(
    distribution=ds.Normal(loc=0., scale=1.),
    bijector=ds.bijectors.Exp(),
    name="LogNormalTransformedDistribution")
  ```

  A `LogNormal` made from callables:

  ```python
  ds = tfp.distributions
  log_normal = ds.TransformedDistribution(
    distribution=ds.Normal(loc=0., scale=1.),
    bijector=ds.bijectors.Inline(
      forward_fn=tf.exp,
      inverse_fn=tf.math.log,
      inverse_log_det_jacobian_fn=(
        lambda y: -tf.reduce_sum(tf.math.log(y), axis=-1)),
    name="LogNormalTransformedDistribution")
  ```

  Another example constructing a Normal from a StandardNormal:

  ```python
  ds = tfp.distributions
  normal = ds.TransformedDistribution(
    distribution=ds.Normal(loc=0., scale=1.),
    bijector=ds.bijectors.Affine(
      shift=-1.,
      scale_identity_multiplier=2.)
    name="NormalTransformedDistribution")
  ```

  A `TransformedDistribution`'s batch- and event-shape are implied by the base
  distribution unless explicitly overridden by `batch_shape` or `event_shape`
  arguments. Specifying an overriding `batch_shape` (`event_shape`) is
  permitted only if the base distribution has scalar batch-shape (event-shape).
  The bijector is applied to the distribution as if the distribution possessed
  the overridden shape(s). The following example demonstrates how to construct a
  multivariate Normal as a `TransformedDistribution`.

  ```python
  ds = tfp.distributions
  # We will create two MVNs with batch_shape = event_shape = 2.
  mean = [[-1., 0],      # batch:0
          [0., 1]]       # batch:1
  chol_cov = [[[1., 0],
               [0, 1]],  # batch:0
              [[1, 0],
               [2, 2]]]  # batch:1
  mvn1 = ds.TransformedDistribution(
      distribution=ds.Normal(loc=0., scale=1.),
      bijector=ds.bijectors.Affine(shift=mean, scale_tril=chol_cov),
      batch_shape=[2],  # Valid because base_distribution.batch_shape == [].
      event_shape=[2])  # Valid because base_distribution.event_shape == [].
  mvn2 = ds.MultivariateNormalTriL(loc=mean, scale_tril=chol_cov)
  # mvn1.log_prob(x) == mvn2.log_prob(x)
  ```

  """

  @deprecation.deprecated(
      "2019-01-01",
      "The TensorFlow Distributions library has moved to "
      "TensorFlow Probability "
      "(https://github.com/tensorflow/probability). You "
      "should update all references to use `tfp.distributions` "
      "instead of `tf.distributions`.",
      warn_once=True)
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
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
      name: Python `str` name prefixed to Ops created by this class. Default:
        `bijector.name + distribution.name`.
    """
    parameters = dict(locals())
    name = name or (("" if bijector is None else bijector.name) +
                    distribution.name)
    with ops.name_scope(name, values=[event_shape, batch_shape]) as name:
      # For convenience we define some handy constants.
      self._zero = constant_op.constant(0, dtype=dtypes.int32, name="zero")
      self._empty = constant_op.constant([], dtype=dtypes.int32, name="empty")

      if bijector is None:
        bijector = identity_bijector.Identity(validate_args=validate_args)

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
          tensor_util.constant_value(self._override_batch_shape).size != 0)

      self._override_event_shape = self._maybe_validate_shape_override(
          event_shape, distribution.is_scalar_event(), validate_args,
          "event_shape")
      self._is_event_override = _logical_not(_logical_equal(
          _ndims_from_shape(self._override_event_shape), self._zero))
      self._is_maybe_event_override = bool(
          tensor_util.constant_value(self._override_event_shape) is None or
          tensor_util.constant_value(self._override_event_shape).size != 0)

      # To convert a scalar distribution into a multivariate distribution we
      # will draw dims from the sample dims, which are otherwise iid. This is
      # easy to do except in the case that the base distribution has batch dims
      # and we're overriding event shape. When that case happens the event dims
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
    # If there's a chance that the event_shape has been overridden, we return
    # what we statically know about the `event_shape_override`. This works
    # because: `_is_maybe_event_override` means `static_override` is `None` or a
    # non-empty list, i.e., we don't statically know the `event_shape` or we do.
    #
    # Since the `bijector` may change the `event_shape`, we then forward what we
    # know to the bijector. This allows the `bijector` to have final say in the
    # `event_shape`.
    static_override = tensor_util.constant_value_as_shape(
        self._override_event_shape)
    return self.bijector.forward_event_shape(
        static_override
        if self._is_maybe_event_override
        else self.distribution.event_shape)

  def _batch_shape_tensor(self):
    return distribution_util.pick_vector(
        self._is_batch_override,
        self._override_batch_shape,
        self.distribution.batch_shape_tensor())

  def _batch_shape(self):
    # If there's a chance that the batch_shape has been overridden, we return
    # what we statically know about the `batch_shape_override`. This works
    # because: `_is_maybe_batch_override` means `static_override` is `None` or a
    # non-empty list, i.e., we don't statically know the `batch_shape` or we do.
    #
    # Notice that this implementation parallels the `_event_shape` except that
    # the `bijector` doesn't get to alter the `batch_shape`. Recall that
    # `batch_shape` is a property of a distribution while `event_shape` is
    # shared between both the `distribution` instance and the `bijector`.
    static_override = tensor_util.constant_value_as_shape(
        self._override_batch_shape)
    return (static_override
            if self._is_maybe_batch_override
            else self.distribution.batch_shape)

  def _sample_n(self, n, seed=None):
    sample_shape = _concat_vectors(
        distribution_util.pick_vector(self._needs_rotation, self._empty, [n]),
        self._override_batch_shape,
        self._override_event_shape,
        distribution_util.pick_vector(self._needs_rotation, [n], self._empty))
    x = self.distribution.sample(sample_shape=sample_shape, seed=seed)
    x = self._maybe_rotate_dims(x)
    # We'll apply the bijector in the `_call_sample_n` function.
    return x

  def _call_sample_n(self, sample_shape, seed, name, **kwargs):
    # We override `_call_sample_n` rather than `_sample_n` so we can ensure that
    # the result of `self.bijector.forward` is not modified (and thus caching
    # works).
    with self._name_scope(name, values=[sample_shape]):
      sample_shape = ops.convert_to_tensor(
          sample_shape, dtype=dtypes.int32, name="sample_shape")
      sample_shape, n = self._expand_sample_shape_to_vector(
          sample_shape, "sample_shape")

      # First, generate samples. We will possibly generate extra samples in the
      # event that we need to reinterpret the samples as part of the
      # event_shape.
      x = self._sample_n(n, seed, **kwargs)

      # Next, we reshape `x` into its final form. We do this prior to the call
      # to the bijector to ensure that the bijector caching works.
      batch_event_shape = array_ops.shape(x)[1:]
      final_shape = array_ops.concat([sample_shape, batch_event_shape], 0)
      x = array_ops.reshape(x, final_shape)

      # Finally, we apply the bijector's forward transformation. For caching to
      # work, it is imperative that this is the last modification to the
      # returned result.
      y = self.bijector.forward(x, **kwargs)
      y = self._set_sample_static_shape(y, sample_shape)

      return y

  def _log_prob(self, y):
    # For caching to work, it is imperative that the bijector is the first to
    # modify the input.
    x = self.bijector.inverse(y)
    event_ndims = self._maybe_get_static_event_ndims()

    ildj = self.bijector.inverse_log_det_jacobian(y, event_ndims=event_ndims)
    if self.bijector._is_injective:  # pylint: disable=protected-access
      return self._finish_log_prob_for_one_fiber(y, x, ildj, event_ndims)

    lp_on_fibers = [
        self._finish_log_prob_for_one_fiber(y, x_i, ildj_i, event_ndims)
        for x_i, ildj_i in zip(x, ildj)]
    return math_ops.reduce_logsumexp(
        array_ops_stack.stack(lp_on_fibers), axis=0)

  def _finish_log_prob_for_one_fiber(self, y, x, ildj, event_ndims):
    """Finish computation of log_prob on one element of the inverse image."""
    x = self._maybe_rotate_dims(x, rotate_right=True)
    log_prob = self.distribution.log_prob(x)
    if self._is_maybe_event_override:
      log_prob = math_ops.reduce_sum(log_prob, self._reduce_event_indices)
    log_prob += math_ops.cast(ildj, log_prob.dtype)
    if self._is_maybe_event_override and isinstance(event_ndims, int):
      log_prob.set_shape(
          array_ops.broadcast_static_shape(
              y.get_shape().with_rank_at_least(1)[:-event_ndims],
              self.batch_shape))
    return log_prob

  def _prob(self, y):
    x = self.bijector.inverse(y)
    event_ndims = self._maybe_get_static_event_ndims()
    ildj = self.bijector.inverse_log_det_jacobian(y, event_ndims=event_ndims)
    if self.bijector._is_injective:  # pylint: disable=protected-access
      return self._finish_prob_for_one_fiber(y, x, ildj, event_ndims)

    prob_on_fibers = [
        self._finish_prob_for_one_fiber(y, x_i, ildj_i, event_ndims)
        for x_i, ildj_i in zip(x, ildj)]
    return sum(prob_on_fibers)

  def _finish_prob_for_one_fiber(self, y, x, ildj, event_ndims):
    """Finish computation of prob on one element of the inverse image."""
    x = self._maybe_rotate_dims(x, rotate_right=True)
    prob = self.distribution.prob(x)
    if self._is_maybe_event_override:
      prob = math_ops.reduce_prod(prob, self._reduce_event_indices)
    prob *= math_ops.exp(math_ops.cast(ildj, prob.dtype))
    if self._is_maybe_event_override and isinstance(event_ndims, int):
      prob.set_shape(
          array_ops.broadcast_static_shape(
              y.get_shape().with_rank_at_least(1)[:-event_ndims],
              self.batch_shape))
    return prob

  def _log_cdf(self, y):
    if self._is_maybe_event_override:
      raise NotImplementedError("log_cdf is not implemented when overriding "
                                "event_shape")
    if not self.bijector._is_injective:  # pylint: disable=protected-access
      raise NotImplementedError("log_cdf is not implemented when "
                                "bijector is not injective.")
    x = self.bijector.inverse(y)
    return self.distribution.log_cdf(x)

  def _cdf(self, y):
    if self._is_maybe_event_override:
      raise NotImplementedError("cdf is not implemented when overriding "
                                "event_shape")
    if not self.bijector._is_injective:  # pylint: disable=protected-access
      raise NotImplementedError("cdf is not implemented when "
                                "bijector is not injective.")
    x = self.bijector.inverse(y)
    return self.distribution.cdf(x)

  def _log_survival_function(self, y):
    if self._is_maybe_event_override:
      raise NotImplementedError("log_survival_function is not implemented when "
                                "overriding event_shape")
    if not self.bijector._is_injective:  # pylint: disable=protected-access
      raise NotImplementedError("log_survival_function is not implemented when "
                                "bijector is not injective.")
    x = self.bijector.inverse(y)
    return self.distribution.log_survival_function(x)

  def _survival_function(self, y):
    if self._is_maybe_event_override:
      raise NotImplementedError("survival_function is not implemented when "
                                "overriding event_shape")
    if not self.bijector._is_injective:  # pylint: disable=protected-access
      raise NotImplementedError("survival_function is not implemented when "
                                "bijector is not injective.")
    x = self.bijector.inverse(y)
    return self.distribution.survival_function(x)

  def _quantile(self, value):
    if self._is_maybe_event_override:
      raise NotImplementedError("quantile is not implemented when overriding "
                                "event_shape")
    if not self.bijector._is_injective:  # pylint: disable=protected-access
      raise NotImplementedError("quantile is not implemented when "
                                "bijector is not injective.")
    # x_q is the "qth quantile" of X iff q = P[X <= x_q].  Now, since X =
    # g^{-1}(Y), q = P[X <= x_q] = P[g^{-1}(Y) <= x_q] = P[Y <= g(x_q)],
    # implies the qth quantile of Y is g(x_q).
    inv_cdf = self.distribution.quantile(value)
    return self.bijector.forward(inv_cdf)

  def _entropy(self):
    if not self.bijector.is_constant_jacobian:
      raise NotImplementedError("entropy is not implemented")
    if not self.bijector._is_injective:  # pylint: disable=protected-access
      raise NotImplementedError("entropy is not implemented when "
                                "bijector is not injective.")
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
    dummy = array_ops.zeros(
        shape=array_ops.concat(
            [self.batch_shape_tensor(), self.event_shape_tensor()],
            0),
        dtype=self.dtype)
    event_ndims = (self.event_shape.ndims if self.event_shape.ndims is not None
                   else array_ops.size(self.event_shape_tensor()))
    ildj = self.bijector.inverse_log_det_jacobian(
        dummy, event_ndims=event_ndims)

    entropy -= math_ops.cast(ildj, entropy.dtype)
    entropy.set_shape(self.batch_shape)
    return entropy

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
    needs_rotation_const = tensor_util.constant_value(self._needs_rotation)
    if needs_rotation_const is not None and not needs_rotation_const:
      return x
    ndims = array_ops.rank(x)
    n = (ndims - self._rotate_ndims) if rotate_right else self._rotate_ndims
    return array_ops.transpose(
        x, _concat_vectors(math_ops.range(n, ndims), math_ops.range(0, n)))

  def _maybe_get_static_event_ndims(self):
    if self.event_shape.ndims is not None:
      return self.event_shape.ndims

    event_ndims = array_ops.size(self.event_shape_tensor())
    event_ndims_ = distribution_util.maybe_get_static_value(event_ndims)

    if event_ndims_ is not None:
      return event_ndims_

    return event_ndims
