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
r"""Bijector Ops.

An API for reversible (bijective) transformations of random variables.

## Background

Differentiable, bijective transformations of continuous random variables alter
the calculations made in the cumulative/probability distribution functions and
sample function.  This module provides a standard interface for making these
manipulations.

For more details and examples, see the `Bijector` docstring.

To apply a `Bijector`, use `distributions.TransformedDistribution`.

## Bijectors

@@Bijector
@@Chain
@@Exp
@@Identity
@@Inline
@@Invert
@@ScaleAndShift
@@SigmoidCentered
@@SoftmaxCentered
@@Softplus

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import contextlib
import re
import numpy as np
import six

from tensorflow.contrib.distributions.python.ops.shape import _DistributionShape
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops


@six.add_metaclass(abc.ABCMeta)
class Bijector(object):
  """Interface for transforming a `Distribution` via `TransformedDistribution`.

  A `Bijector` implements a bijective, differentiable function by transforming
  an input `Tensor`. The output `Tensor` shape is constrained by the input
  `sample`, `batch`, and `event` shape.  A `Bijector` is characterized by three
  operations:

  1. Forward Evaluation

     Useful for turning one random outcome into another random outcome from a
     different distribution.

  2. Inverse Evaluation

     Useful for "reversing" a transformation to compute one probability in
     terms of another.

  3. (log o det o Jacobian o inverse)(x)

     "The log of the determinant of the matrix of all first-order partial
     derivatives of the inverse function."
     Useful for inverting a transformation to compute one probability in terms
     of another.  Geometrically, the det(Jacobian) is the volume of the
     transformation and is used to scale the probability.

  By convention, transformations of random variables are named in terms of the
  forward transformation. The forward transformation creates samples, the
  inverse is useful for computing probabilities.

  Example Use:

    - Basic properties:

    ```python
    x = ... # A tensor.
    # Evaluate forward transformation.
    fwd_x = my_bijector.forward(x)
    x == my_bijector.inverse(fwd_x)
    x != my_bijector.forward(fwd_x)  # Not equal because g(x) != g(g(x)).
    ```

    - Computing a log-likelihood:

    ```python
    def transformed_log_pdf(bijector, log_pdf, x):
      return (bijector.inverse_log_det_jacobian(x) +
              log_pdf(bijector.inverse(x)))
    ```

    - Transforming a random outcome:

    ```python
    def transformed_sample(bijector, x):
      return bijector.forward(x)
    ```

  Example transformations:

    - "Exponential"

      ```
      Y = g(X) = exp(X)
      X ~ Normal(0, 1)  # Univariate.
      ```

      Implies:

      ```
        g^{-1}(Y) = log(Y)
        |Jacobian(g^{-1})(y)| = 1 / y
        Y ~ LogNormal(0, 1), i.e.,
        prob(Y=y) = |Jacobian(g^{-1})(y)| * prob(X=g^{-1}(y))
                  = (1 / y) Normal(log(y); 0, 1)
      ```

    - "ScaleAndShift"

      ```
      Y = g(X) = sqrtSigma * X + mu
      X ~ MultivariateNormal(0, I_d)
      ```

      Implies:

      ```
        g^{-1}(Y) = inv(sqrtSigma) * (Y - mu)
        |Jacobian(g^{-1})(y)| = det(inv(sqrtSigma))
        Y ~ MultivariateNormal(mu, sqrtSigma) , i.e.,
        prob(Y=y) = |Jacobian(g^{-1})(y)| * prob(X=g^{-1}(y))
                  = det(sqrtSigma)^(-d) *
                    MultivariateNormal(inv(sqrtSigma) * (y - mu); 0, I_d)
      ```

  Example of why a `Bijector` needs to understand sample, batch, event
  partitioning:

  - Consider the `Exp` `Bijector` applied to a `Tensor` which has sample, batch,
    and event (S, B, E) shape semantics.  Suppose
    the `Tensor`'s partitioned-shape is `(S=[4], B=[2], E=[3, 3])`.

    For `Exp`, the shape of the `Tensor` returned by `forward` and `inverse` is
    unchanged, i.e., `[4, 2, 3, 3]`. However the shape returned by
    `inverse_log_det_jacobian` is `[4, 2]` because the Jacobian is a reduction
    over the event dimensions.

  Subclass Requirements:

  - Typically subclasses implement `_forward` and one or both of:
      - `_inverse`, `_inverse_log_det_jacobian`,
      - `_inverse_and_inverse_log_det_jacobian`.

  - If the `Bijector`'s use is limited to `TransformedDistribution` (or friends
    like `QuantizedDistribution`) then depending on your use, you may not need
    to implement all of `_forward` and `_inverese` functions.  Examples:
      1. Sampling (e.g., `sample`) only requires `_forward`.
      2. Probability functions (e.g., `prob`, `cdf`, `survival`) only require
         `_inverse` (and related).
      3. Only calling probability functions on the output of `sample` means
        `_inverse` can be implemented as a cache lookup.

    See `Example Use` [above] which shows how these functions are used to
    transform a distribution.  (Note: `_forward` could theoretically be
    implemented as a cache lookup but this would require controlling the
    underlying sample generation mechanism.)

  - If computation can be shared among `_inverse` and
    `_inverse_log_det_jacobian` it is preferable to implement
    `_inverse_and_inverse_log_det_jacobian`. This usually reduces
    graph-construction overhead because a `Distribution`'s implementation of
    `log_prob` will need to evaluate both the inverse Jacobian as well as the
    inverse function.

  - If an additional use case needs just `inverse` or just
    `inverse_log_det_jacobian` then he or she may also wish to implement these
    functions to avoid computing the `inverse_log_det_jacobian` or the
    `inverse`, respectively.

  Tips for implementing `_inverse` and `_inverse_log_det_jacobian`:

  - As case 3 [above] indicates, under some circumstances the inverse function
    can be implemented as a cache lookup.

  - The inverse `log o det o Jacobian` can be implemented as the negative of the
    forward `log o det o Jacobian`.  This is useful if the `inverse` is
    implemented as a cache or the inverse Jacobian is computationally more
    expensive. The following demonstrates the suggested implementation.

    ```python
    def _inverse_and_log_det_jacobian(self, y):
       x = # ... implement inverse, possibly via cache.
       return x, -self._forward_log_det_jac(x)  # Note negation.
    ```

    By overriding the `_inverse_and_log_det_jacobian` function we have access to
    the inverse in one call.

    The correctness of this approach can be seen from the following claim.

    - Claim:

        Assume `Y=g(X)` is a bijection whose derivative exists and is nonzero
        for its domain, i.e., `d/dX g(X)!=0`. Then:

        ```none
        (log o det o jacobian o g^{-1})(Y) = -(log o det o jacobian o g)(X)
        ```

    - Proof:

        From the bijective, nonzero differentiability of `g`, the
        [inverse function theorem](
            https://en.wikipedia.org/wiki/Inverse_function_theorem)
        implies `g^{-1}` is differentiable in the image of `g`.
        Applying the chain rule to `y = g(x) = g(g^{-1}(y))` yields
        `I = g'(g^{-1}(y))*g^{-1}'(y)`.
        The same theorem also implies `g{-1}'` is non-singular therefore:
        `inv[ g'(g^{-1}(y)) ] = g^{-1}'(y)`.
        The claim follows from [properties of determinant](
  https://en.wikipedia.org/wiki/Determinant#Multiplicativity_and_matrix_groups).

  - If possible, prefer a direct implementation of the inverse Jacobian. This
    should have superior numerical stability and will often share subgraphs with
    the `_inverse` implementation.

  """

  @abc.abstractmethod
  def __init__(self,
               batch_ndims=None,
               event_ndims=None,
               parameters=None,
               is_constant_jacobian=False,
               validate_args=False,
               dtype=None,
               name=None):
    """Constructs Bijector.

    A `Bijector` transforms random variables into new random variables.

    Examples:

    ```python
    # Create the Y = g(X) = X transform which operates on 4-Tensors of vectors.
    identity = Identity(batch_ndims=4, event_ndims=1)

    # Create the Y = g(X) = exp(X) transform which operates on matrices.
    exp = Exp(batch_ndims=0, event_ndims=2)
    ```

    See `Bijector` subclass docstring for more details and specific examples.

    Args:
      batch_ndims: number of dimensions associated with batch coordinates.
      event_ndims: number of dimensions associated with event coordinates.
      parameters: Dictionary of parameters used by this `Bijector`
      is_constant_jacobian: `Boolean` indicating that the Jacobian is not a
        function of the input.
      validate_args: `Boolean`, default `False`.  Whether to validate input with
        asserts. If `validate_args` is `False`, and the inputs are invalid,
        correct behavior is not guaranteed.
      dtype: `tf.dtype` supported by this `Bijector`. `None` means dtype is not
        enforced.
      name: The name to give Ops created by the initializer.
    """
    if batch_ndims is None or event_ndims is None:
      self._shaper = None  # Apparently subclass will create.
    else:
      self._shaper = _DistributionShape(
          batch_ndims=batch_ndims,
          event_ndims=event_ndims,
          validate_args=validate_args)
    self._parameters = parameters or {}
    self._is_constant_jacobian = is_constant_jacobian
    self._validate_args = validate_args
    self._dtype = dtype
    if name:
      self._name = name
    else:
      # We want the default convention to be snake_case rather than CamelCase
      # since `Chain` uses bijector.name as the condition_kwargs dictionary key.
      def camel_to_snake(name):
        s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
        return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()
      self._name = camel_to_snake(type(self).__name__)

  @property
  def shaper(self):
    """Returns shape object used to manage shape constraints."""
    return self._shaper

  @property
  def parameters(self):
    """Returns this `Bijector`'s parameters as a name/value dictionary."""
    return self._parameters

  @property
  def is_constant_jacobian(self):
    """Returns true iff the Jacobian is not a function of x.

    Note: Jacobian is either constant for both forward and inverse or neither.

    Returns:
      `Boolean`.
    """
    return self._is_constant_jacobian

  @property
  def validate_args(self):
    """Returns True if Tensor arguments will be validated."""
    return self._validate_args

  @property
  def dtype(self):
    """dtype of `Tensor`s transformable by this distribution."""
    return self._dtype

  @property
  def name(self):
    """Returns the string name of this `Bijector`."""
    return self._name

  def _forward(self, x):
    raise NotImplementedError("forward is not implemented.")

  def forward(self, x, name="forward", **condition_kwargs):
    """Returns the forward `Bijector` evaluation, i.e., X = g(Y).

    Args:
      x: `Tensor`. The input to the "forward" evaluation.
      name: The name to give this op.
      **condition_kwargs: Named arguments forwarded to subclass implementation.

    Returns:
      `Tensor`.

    Raises:
      TypeError: if `self.dtype` is specified and `x.dtype` is not
        `self.dtype`.
      NotImplementedError: if `_forward` is not implemented.
    """
    with self._name_scope(name, [x]):
      x = ops.convert_to_tensor(x, name="x")
      self._maybe_assert_dtype(x)
      return self._forward(x, **condition_kwargs)

  def _inverse(self, y):
    raise NotImplementedError("inverse is not implemented")

  def inverse(self, y, name="inverse", **condition_kwargs):
    """Returns the inverse `Bijector` evaluation, i.e., X = g^{-1}(Y).

    Args:
      y: `Tensor`. The input to the "inverse" evaluation.
      name: The name to give this op.
      **condition_kwargs: Named arguments forwarded to subclass implementation.

    Returns:
      `Tensor`.

    Raises:
      TypeError: if `self.dtype` is specified and `y.dtype` is not
        `self.dtype`.
      NotImplementedError: if neither `_inverse` nor
        `_inverse_and_inverse_log_det_jacobian` are implemented.
    """
    with self._name_scope(name, [y]):
      y = ops.convert_to_tensor(y, name="y")
      self._maybe_assert_dtype(y)
      try:
        return self._inverse(y, **condition_kwargs)
      except NotImplementedError as original_error:
        # Since _inverse was not implemented, try to see if it's implemented
        # by the _inverse_and_inverse_log_det_jacobian member.
        try:
          return self._inverse_and_inverse_log_det_jacobian(
              y, **condition_kwargs)[0]
        except NotImplementedError:
          raise original_error

  def _inverse_log_det_jacobian(self, y):
    raise NotImplementedError("inverse_log_det_jacobian is not implemented.")

  def inverse_log_det_jacobian(
      self, y, name="inverse_log_det_jacobian", **condition_kwargs):
    """Returns the (log o det o Jacobian o inverse)(y).

    Mathematically, returns: `log(det(dX/dY))(Y)`. (Recall that: `X=g^{-1}(Y)`.)

    Note that `forward_log_det_jacobian` is the negative of this function.

    Args:
      y: `Tensor`. The input to the "inverse" Jacobian evaluation.
      name: The name to give this op.
      **condition_kwargs: Named arguments forwarded to subclass implementation.

    Returns:
      `Tensor`.

    Raises:
      TypeError: if `self.dtype` is specified and `y.dtype` is not
        `self.dtype`.
      NotImplementedError: if neither `_inverse_log_det_jacobian` nor
        `_inverse_and_inverse_log_det_jacobian` are implemented.
    """
    with self._name_scope(name, [y]):
      y = ops.convert_to_tensor(y, name="y")
      self._maybe_assert_dtype(y)
      try:
        return self._inverse_log_det_jacobian(y, **condition_kwargs)
      except NotImplementedError as original_error:
        # Since _inverse_log_det_jacobian was not implemented, try to see if
        # it's implemented by the _inverse_and_inverse_log_det_jacobian member.
        try:
          return self._inverse_and_inverse_log_det_jacobian(
              y, **condition_kwargs)[1]
        except NotImplementedError:
          raise original_error

  def _inverse_and_inverse_log_det_jacobian(self, y):
    raise NotImplementedError(
        "inverse_and_inverse_log_det_jacobian is not implemented.")

  def inverse_and_inverse_log_det_jacobian(
      self, y, name="inverse_and_inverse_log_det_jacobian", **condition_kwargs):
    """Returns both the inverse evaluation and inverse_log_det_jacobian.

    Enables possibly more efficient calculation when both inverse and
    corresponding Jacobian are needed.

    See `inverse()`, `inverse_log_det_jacobian()` for more details.

    Args:
      y: `Tensor`. The input to the "inverse" Jacobian evaluation.
      name: The name to give this op.
      **condition_kwargs: Named arguments forwarded to subclass implementation.

    Returns:
      `Tensor`.

    Raises:
      TypeError: if `self.dtype` is specified and `y.dtype` is not
        `self.dtype`.
      NotImplementedError: if neither `_inverse_and_inverse_log_det_jacobian`
        nor {`_inverse`, `_inverse_log_det_jacobian`} are implemented.
    """
    with self._name_scope(name, [y]):
      y = ops.convert_to_tensor(y, name="y")
      self._maybe_assert_dtype(y)
      try:
        return self._inverse_and_inverse_log_det_jacobian(
            y, **condition_kwargs)
      except NotImplementedError as original_error:
        # Since _inverse_and_inverse_log_det_jacobian was not implemented, try
        # to see if we can separately use _inverse and
        # _inverse_log_det_jacobian members.
        try:
          return (self._inverse(y, **condition_kwargs),
                  self._inverse_log_det_jacobian(y, **condition_kwargs))
        except NotImplementedError:
          raise original_error

  def _forward_log_det_jacobian(self, x):
    raise NotImplementedError(
        "forward_log_det_jacobian is not implemented.")

  def forward_log_det_jacobian(
      self, x, name="forward_log_det_jacobian", **condition_kwargs):
    """Returns both the forward_log_det_jacobian.

    Args:
      x: `Tensor`. The input to the "forward" Jacobian evaluation.
      name: The name to give this op.
      **condition_kwargs: Named arguments forwarded to subclass implementation.

    Returns:
      `Tensor`.

    Raises:
      TypeError: if `self.dtype` is specified and `y.dtype` is not
        `self.dtype`.
      NotImplementedError: if neither `_forward_log_det_jacobian`
        nor {`_inverse`, `_inverse_log_det_jacobian`} are implemented.
    """
    with self._name_scope(name, [x]):
      x = ops.convert_to_tensor(x, name="x")
      self._maybe_assert_dtype(x)
      try:
        return self._forward_log_det_jacobian(x, **condition_kwargs)
      except NotImplementedError as original_error:
        try:
          y = self.inverse(x, **condition_kwargs)
          return -self.inverse_log_det_jacobian(y, **condition_kwargs)
        except NotImplementedError:
          raise original_error

  @contextlib.contextmanager
  def _name_scope(self, name=None, values=None):
    """Helper function to standardize op scope."""
    with ops.name_scope(self.name):
      with ops.name_scope(name, values=(
          (values or []) + list(self.parameters.values()))) as scope:
        yield scope

  def _maybe_assert_dtype(self, x):
    """Helper to check dtype when self.dtype is known."""
    if self.dtype is not None and self.dtype.base_dtype != x.dtype.base_dtype:
      raise TypeError("Input had dtype %s but expected %s." %
                      (self.dtype, x.dtype))


class Inline(Bijector):
  # pylint: disable=line-too-long
  """Bijector constructed from callables implementing forward, inverse, and inverse_log_det_jacobian.

  Example Use:

  ```python
  exp = Inline(
    forward_fn=tf.exp,
    inverse_fn=tf.log,
    inverse_log_det_jacobian_fn=(
      lambda y: -tf.reduce_sum(tf.log(y), reduction_indices=-1)),
    name="Exp")
  ```

  The above example is equivalent to the `Bijector` `Exp(event_ndims=1)`.
  """
  # pylint: enable=line-too-long

  def __init__(self,
               forward_fn=None,
               inverse_fn=None,
               inverse_log_det_jacobian_fn=None,
               forward_log_det_jacobian_fn=None,
               is_constant_jacobian=False,
               validate_args=False,
               name="inline"):
    """Creates a `Bijector` from callables.

    Args:
      forward_fn: Python callable implementing the forward transformation.
      inverse_fn: Python callable implementing the inverse transformation.
      inverse_log_det_jacobian_fn: Python callable implementing the
        log o det o jacobian of the inverse transformation.
      forward_log_det_jacobian_fn: Python callable implementing the
        log o det o jacobian of the forward transformation.
      is_constant_jacobian: `Boolean` indicating that the Jacobian is constant
        for all input arguments.
      validate_args: `Boolean` indicated whether arguments should be checked for
        correctness.
      name: `String`, name given to ops managed by this object.
    """
    super(Inline, self).__init__(
        batch_ndims=0,
        event_ndims=0,
        is_constant_jacobian=is_constant_jacobian,
        validate_args=validate_args,
        name=name)
    self._forward_fn = forward_fn
    self._inverse_fn = inverse_fn
    self._inverse_log_det_jacobian_fn = inverse_log_det_jacobian_fn
    self._forward_log_det_jacobian_fn = forward_log_det_jacobian_fn

  def _forward(self, x, **condition_kwargs):
    if not callable(self._forward_fn):
      raise NotImplementedError(
          "forward_fn is not a callable function.")
    return self._forward_fn(x, **condition_kwargs)

  def _inverse(self, y, **condition_kwargs):
    if not callable(self._inverse_fn):
      raise NotImplementedError(
          "inverse_fn is not a callable function.")
    return self._inverse_fn(y, **condition_kwargs)

  def _inverse_log_det_jacobian(self, y, **condition_kwargs):
    if not callable(self._inverse_log_det_jacobian_fn):
      raise NotImplementedError(
          "inverse_log_det_jacobian_fn is not a callable function.")
    return self._inverse_log_det_jacobian_fn(y, **condition_kwargs)

  def _forward_log_det_jacobian(self, y, **condition_kwargs):
    if not callable(self._forward_log_det_jacobian_fn):
      raise NotImplementedError(
          "forward_log_det_jacobian_fn is not a callable function.")
    return self._forward_log_det_jacobian_fn(y, **condition_kwargs)


class Invert(Bijector):
  """Bijector which inverts another Bijector.

  Example Use: [ExpGammaDistribution (see Background & Context)](
  https://reference.wolfram.com/language/ref/ExpGammaDistribution.html)
  models `Y=log(X)` where `X ~ Gamma`.

  ```python
  exp_gamma_distribution = TransformedDistribution(
    Gamma(alpha=1., beta=2.),
    bijector.Invert(bijector.Exp())
  ```

  """

  def __init__(self, bijector, validate_args=False, name=None):
    """Creates a `Bijector` which swaps the meaning of `inverse` and `forward`.

    Note: An inverted bijector's `inverse_log_det_jacobian` is often more
    efficient if the base bijector implements `_forward_log_det_jacobian`. If
    `_forward_log_det_jacobian` is not implemented then the following code is
    used:

    ```python
    y = self.inverse(x, **condition_kwargs)
    return -self.inverse_log_det_jacobian(y, **condition_kwargs)
    ```

    Args:
      bijector: Bijector instance.
      validate_args: `Boolean` indicated whether arguments should be checked for
        correctness.
      name: `String`, name given to ops managed by this object.
    """

    self._bijector = bijector
    super(Invert, self).__init__(
        parameters=bijector.parameters,
        is_constant_jacobian=bijector.is_constant_jacobian,
        validate_args=validate_args,
        dtype=bijector.dtype,
        name=name or "_".join(["invert", bijector.name]))
    self._shaper = bijector.shaper

  @property
  def bijector(self):
    return self._bijector

  def _forward(self, x, **condition_kwargs):
    return self.bijector.inverse(x, **condition_kwargs)

  def _inverse_and_inverse_log_det_jacobian(self, y, **condition_kwargs):
    return (self.bijector.forward(y, **condition_kwargs),
            self.bijector.forward_log_det_jacobian(y, **condition_kwargs))

  def _forward_log_det_jacobian(self, x, **condition_kwargs):
    return self.bijector.inverse_log_det_jacobian(x, **condition_kwargs)


class Chain(Bijector):
  """Bijector which applies a sequence of bijectors.

  Example Use:

  ```python
  chain = Chain([Exp(), Softplus()], name="one_plus_exp")
  ```

  Results in:

  * Forward:

   ```python
   exp = Exp()
   softplus = Softplus()
   Chain([exp, softplus]).forward(x)
   = exp.forward(softplus.forward(x))
   = tf.exp(tf.log(1. + tf.exp(x)))
   = 1. + tf.exp(x)
   ```

  * Inverse:

   ```python
   exp = Exp()
   softplus = Softplus()
   Chain([exp, softplus]).inverse(y)
   = softplus.inverse(exp.inverse(y))
   = tf.log(tf.exp(tf.log(y)) - 1.)
   = tf.log(y - 1.)
   ```

  """

  def __init__(self, bijectors=(), validate_args=False, name=None):
    """Instantiates `Chain` bijector.

    Args:
      bijectors: Python list of bijector instances. An empty list makes this
        bijector equivalent to the `Identity` bijector.
      validate_args: `Boolean` indicated whether arguments should be checked for
        correctness.
      name: `String`, name given to ops managed by this object. Default: E.g.,
        `Chain([Exp(), Softplus()]).name == "chain_of_exp_of_softplus"`.

    Raises:
      ValueError: if bijectors have different dtypes.
    """

    self._bijectors = bijectors

    dtype = list(set([b.dtype for b in bijectors]))
    if len(dtype) > 2:
      raise ValueError("incompatible dtypes: %s" % dtype)
    elif len(dtype) == 2:
      dtype = dtype[1] if dtype[0] is None else dtype[0]
    elif len(dtype) == 1:
      dtype = dtype[0]
    else:
      dtype = None

    super(Chain, self).__init__(
        parameters=dict(("=".join([b.name, k]), v)
                        for b in bijectors
                        for k, v in b.parameters.items()),
        is_constant_jacobian=all([b.is_constant_jacobian
                                  for b in bijectors]),
        validate_args=validate_args,
        dtype=dtype,
        name=name or ("identity" if not bijectors else
                      "_of_".join(["chain"] + [b.name for b in bijectors])))

  @property
  def bijectors(self):
    return self._bijectors

  def _forward(self, x, **condition_kwargs):
    y = x
    for b in reversed(self.bijectors):
      y = b.forward(y, **condition_kwargs.get(b.name, {}))
    return y

  def _inverse_and_inverse_log_det_jacobian(self, y, **condition_kwargs):
    x = y
    ildj = constant_op.constant(0., dtype=x.dtype,
                                name="inverse_log_det_jacobian")
    for b in self.bijectors:
      x, j = b.inverse_and_inverse_log_det_jacobian(
          x, **condition_kwargs.get(b.name, {}))
      ildj += j
    return x, ildj

  def _forward_log_det_jacobian(self, x, **condition_kwargs):
    y = x
    fldj = constant_op.constant(0., dtype=x.dtype,
                                name="forward_log_det_jacobian")
    for b in reversed(self.bijectors):
      bijector_condition_kwargs = condition_kwargs.get(b.name, {})
      fldj += b.forward_log_det_jacobian(y, **bijector_condition_kwargs)
      y = b.forward(y, **bijector_condition_kwargs)
    return fldj


class Identity(Bijector):
  """Bijector which computes Y = g(X) = X.

    Example Use:

    ```python
    # Create the Y=g(X)=X transform which is intended for Tensors with 1 batch
    # ndim and 1 event ndim (i.e., vector of vectors).
    identity = Identity(batch_ndims=1, event_ndims=1)
    x = [[1., 2],
         [3, 4]]
    x == identity.forward(x) == identity.inverse(x)
    ```

  """

  def __init__(self, validate_args=False, name="identity"):
    super(Identity, self).__init__(
        batch_ndims=0,
        event_ndims=0,
        is_constant_jacobian=True,
        validate_args=validate_args,
        name=name)
    self._is_constant_jacobian = True

  def _forward(self, x):
    return x

  def _inverse_and_inverse_log_det_jacobian(self, y):
    return y, constant_op.constant(0., dtype=y.dtype)

  def _forward_log_det_jacobian(self, x):
    return constant_op.constant(0., dtype=x.dtype)


class Exp(Bijector):
  """Bijector which computes Y = g(X) = exp(X).

    Example Use:

    ```python
    # Create the Y=g(X)=exp(X) transform which works only on Tensors with 1
    # batch ndim and 2 event ndims (i.e., vector of matrices).
    exp = Exp(batch_ndims=1, event_ndims=2)
    x = [[[1., 2],
           [3, 4]],
          [[5, 6],
           [7, 8]]]
    exp(x) == exp.forward(x)
    log(x) == exp.inverse(x)
    ```

    Note: the exp(.) is applied element-wise but the Jacobian is a reduction
    over the event space.
  """

  def __init__(self,
               event_ndims=0,
               validate_args=False,
               name="exp"):
    """Instantiates the `Exp` bijector.

    Args:
      event_ndims: Scalar `int32` `Tensor` indicating the number of dimensions
        associated with a particular draw from the distribution.
      validate_args: `Boolean` indicated whether arguments should be checked for
        correctness.
      name: `String` name given to ops managed by this object.
    """

    super(Exp, self).__init__(
        batch_ndims=0,
        event_ndims=event_ndims,
        validate_args=validate_args,
        name=name)

  def _forward(self, x):
    return math_ops.exp(x)

  def _inverse_and_inverse_log_det_jacobian(self, y):
    x = math_ops.log(y)
    return x, -self._forward_log_det_jacobian(x)

  def _forward_log_det_jacobian(self, x):
    if self.shaper is None:
      raise ValueError("Jacobian cannot be computed with unknown event_ndims")
    _, _, event_dims = self.shaper.get_dims(x)
    return math_ops.reduce_sum(x, reduction_indices=event_dims)


class ScaleAndShift(Bijector):
  """Bijector which computes Y = g(X; shift, scale) = scale * X + shift.

  Example Use:

  ```python
  # No batch, scalar.
  mu = 0     # shape=[]
  sigma = 1  # shape=[]
  b = ScaleAndShift(shift=mu, scale=sigma)
  # b.shaper.batch_ndims == 0
  # b.shaper.event_ndims == 0

  # One batch, scalar.
  mu = ...    # shape=[b], b>0
  sigma = ... # shape=[b], b>0
  b = ScaleAndShift(shift=mu, scale=sigma)
  # b.shaper.batch_ndims == 1
  # b.shaper.event_ndims == 0

  # No batch, multivariate.
  mu = ...    # shape=[d],    d>0
  sigma = ... # shape=[d, d], d>0
  b = ScaleAndShift(shift=mu, scale=sigma, event_ndims=1)
  # b.shaper.batch_ndims == 0
  # b.shaper.event_ndims == 1

  # (B1*B2*...*Bb)-batch, multivariate.
  mu = ...    # shape=[B1,...,Bb, d],    b>0, d>0
  sigma = ... # shape=[B1,...,Bb, d, d], b>0, d>0
  b = ScaleAndShift(shift=mu, scale=sigma, event_ndims=1)
  # b.shaper.batch_ndims == b
  # b.shaper.event_ndims == 1

  # Mu is broadcast:
  mu = 1
  sigma = [I, I]  # I is a 3x3 identity matrix.
  b = ScaleAndShift(shift=mu, scale=sigma, event_ndims=1)
  x = numpy.ones(S + sigma.shape)
  b.forward(x) # == x + 1
  ```

  """

  def __init__(self,
               shift,
               scale,
               event_ndims=0,
               validate_args=False,
               name="scale_and_shift"):
    """Instantiates the `Exp` bijector.

    Args:
      shift: `Tensor` used to shift input, i.e., `Y = g(X) = scale * X + shift`.
      scale: `Tensor` used to scale input, i.e., `Y = g(X) = scale * X + shift`.
      event_ndims: Scalar `int32` `Tensor` indicating the number of dimensions
        associated with a particular draw from the distribution.
      validate_args: `Boolean` indicated whether arguments should be checked for
        correctness.
      name: `String` name given to ops managed by this object.
    """

    self._parameters = {}
    self._name = name
    with self._name_scope("init", values=[shift, scale, event_ndims]):
      self._shift = ops.convert_to_tensor(shift, name="shift")
      self._scale = ops.convert_to_tensor(scale, name="scale")
      event_ndims = ops.convert_to_tensor(event_ndims, name="event_ndims")
      if self.shift.dtype.base_dtype != self.scale.dtype.base_dtype:
        raise TypeError("%s.dtype=%s does not match %s.dtype=%s" %
                        (self.shift.name, self.shift.dtype, self.scale.name,
                         self.scale.dtype))
      if event_ndims.dtype.base_dtype != dtypes.int32.base_dtype:
        raise TypeError("%s.dtype=%s does not match %s" %
                        (event_ndims.name, event_ndims.dtype, dtypes.int32))
      self._scale, batch_ndims = self._process_scale(self.scale, event_ndims)
      super(ScaleAndShift, self).__init__(
          batch_ndims=batch_ndims,
          event_ndims=event_ndims,
          parameters={"shift": self.shift, "scale": self.scale},
          is_constant_jacobian=True,
          validate_args=validate_args,
          name=name)

  def _process_scale(self, scale, event_ndims):
    """Helper to __init__ which gets scale in batch-ready form.

    This function expands dimensions of `scale` according to the following
    table:
                     event_ndims
    scale.ndims   0            1
              0  [1]+S+[1,1]   "silent error"
              1  [ ]+S+[1,1]   "silent error"
              2  [ ]+S+[1,1]   [1]+S+[ ]
              3  [ ]+S+[1,1]   [ ]+S+[ ]
            ...  (same)        (same)

    The idea is that we want to convert `scale` into something which can always
    work for, say, the left-hand argument of `batch_matmul`.

    Args:
      scale: `Tensor`.
      event_ndims: `Tensor` (0D, `int32`).

    Returns:
      scale: `Tensor` with dims expanded according to [above] table.
      batch_ndims: `Tensor` (0D, `int32`).  The ndims of the `batch` portion.
    """
    ndims = array_ops.rank(scale)
    left = math_ops.select(
        math_ops.reduce_any([
            math_ops.reduce_all([
                math_ops.equal(ndims, 0),
                math_ops.equal(event_ndims, 0)
            ]),
            math_ops.reduce_all([
                math_ops.equal(ndims, 2),
                math_ops.equal(event_ndims, 1)
            ])]), 1, 0)
    right = math_ops.select(math_ops.equal(event_ndims, 0), 2, 0)
    pad = array_ops.concat(0, (
        array_ops.ones([left], dtype=dtypes.int32),
        array_ops.shape(scale),
        array_ops.ones([right], dtype=dtypes.int32)))
    scale = array_ops.reshape(scale, pad)
    batch_ndims = ndims - 2 + right
    return scale, batch_ndims

  @property
  def shift(self):
    return self._shift

  @property
  def scale(self):
    return self._scale

  def _forward(self, x):
    x, sample_shape = self.shaper.make_batch_of_event_sample_matrices(x)
    x = math_ops.batch_matmul(self.scale, x)
    x = self.shaper.undo_make_batch_of_event_sample_matrices(x, sample_shape)
    x += self.shift
    return x

  def _inverse(self, y):
    x = y - self.shift
    x, sample_shape = self.shaper.make_batch_of_event_sample_matrices(x)
    x = linalg_ops.matrix_triangular_solve(self.scale, x)
    x = self.shaper.undo_make_batch_of_event_sample_matrices(x, sample_shape)
    return x

  def _inverse_log_det_jacobian(self, y):  # pylint: disable=unused-argument
    return -math_ops.reduce_sum(
        math_ops.log(array_ops.matrix_diag_part(self.scale)),
        reduction_indices=[-1])

  def _forward_log_det_jacobian(self, x):  # pylint: disable=unused-argument
    return -self._inverse_log_det_jacobian(x)


class Softplus(Bijector):
  """Bijector which computes `Y = g(X) = Log[1 + exp(X)]`.

  The softplus `Bijector` has the following two useful properties:

  * The domain is the positive real numbers
  * `softplus(x) approx x`, for large `x`, so it does not overflow as easily as
    the `Exp` `Bijector`.

    Example Use:

    ```python
    # Create the Y=g(X)=softplus(X) transform which works only on Tensors with 1
    # batch ndim and 2 event ndims (i.e., vector of matrices).
    softplus = Softplus(batch_ndims=1, event_ndims=2)
    x = [[[1., 2],
           [3, 4]],
          [[5, 6],
           [7, 8]]]
    log(1 + exp(x)) == softplus.forward(x)
    log(exp(x) - 1) == softplus.inverse(x)
    ```

    Note: log(.) and exp(.) are applied element-wise but the Jacobian is a
    reduction over the event space.
  """

  def __init__(self,
               event_ndims=0,
               validate_args=False,
               name="softplus"):
    super(Softplus, self).__init__(
        batch_ndims=0,
        event_ndims=event_ndims,
        validate_args=validate_args,
        name=name)

  def _forward(self, x):
    return nn_ops.softplus(x)

  def _inverse_and_inverse_log_det_jacobian(self, y):
    # The most stable inverse of softplus is not the most direct one.
    # y = softplus(x) = Log[1 + exp{x}], (which means y > 0).
    # ==> exp{y} = 1 + exp{x}
    # ==> x = Log[exp{y} - 1]
    #       = Log[(exp{y} - 1) / exp{y}] + Log[exp{y}]
    #       = Log[(1 - exp{-y}) / 1] + Log[exp{y}]
    #       = Log[1 - exp{-y}] + y
    # Recalling y > 0, you see that this is more stable than Log[exp{y} - 1].
    #
    # Stable inverse log det jacobian.
    # Y = Log[1 + exp{X}] ==> X = Log[exp{Y} - 1]
    # ==> dX/dY = exp{Y} / (exp{Y} - 1)
    #           = 1 / (1 - exp{-Y}),
    # which is the most stable for Y > 0.
    if self.shaper is None:
      raise ValueError("Jacobian cannot be computed with unknown event_ndims")
    _, _, event_dims = self.shaper.get_dims(y)
    log_one_minus_exp_neg = math_ops.log(1. - math_ops.exp(-y))
    x = y + log_one_minus_exp_neg
    ildj = -math_ops.reduce_sum(
        log_one_minus_exp_neg, reduction_indices=event_dims)
    return x, ildj

  def _forward_log_det_jacobian(self, x):  # pylint: disable=unused-argument
    if self.shaper is None:
      raise ValueError("Jacobian cannot be computed with unknown event_ndims")
    _, _, event_dims = self.shaper.get_dims(x)
    return -math_ops.reduce_sum(
        nn_ops.softplus(-x), reduction_indices=event_dims)


class SoftmaxCentered(Bijector):
  """Bijector which computes `Y = g(X) = exp([X 0]) / sum(exp([X 0]))`.

  To implement [softmax](https://en.wikipedia.org/wiki/Softmax_function) as a
  bijection, the forward transformation appends a value to the input and the
  inverse removes this coordinate.  The appended coordinate represents a pivot,
  e.g., `softmax(x) = exp(x-c) / sum(exp(x-c))` where `c` is the implicit last
  coordinate.

  Because we append a coordinate, this bijector only supports `event_ndim in [0,
  1]`, i.e., scalars and vectors.

  Example Use:

  ```python
  bijector.SoftmaxCentered(event_ndims=1).forward(tf.log([2, 3, 4]))
  # Result: [0.2, 0.3, 0.4, 0.1]
  # Extra result: 0.1

  bijector.SoftmaxCentered(event_ndims=1).inverse([0.2, 0.3, 0.4, 0.1])
  # Result: tf.log([2, 3, 4])
  # Extra coordinate removed.
  ```

  At first blush it may seem like the [Invariance of domain](
  https://en.wikipedia.org/wiki/Invariance_of_domain) theorem implies this
  implementation is not a bijection.  However, the appended dimension
  makes the (forward) image non-open and the theorem does not directly apply.
  """

  def __init__(self,
               event_ndims=0,
               validate_args=False,
               name="softmax_centered"):
    self._parameters = {}
    self._name = name
    with self._name_scope("init", values=[event_ndims]):
      event_ndims = ops.convert_to_tensor(event_ndims, name="event_ndims")
      event_ndims = tensor_util.constant_value(event_ndims)
      if event_ndims is None or event_ndims not in [0, 1]:
        raise ValueError("`event_ndims` must be a TF constant which is 0 or 1")
    self._static_event_ndims = event_ndims
    super(SoftmaxCentered, self).__init__(
        batch_ndims=0,  # We'll regard all non-event dims as sample dims.
        event_ndims=event_ndims,
        validate_args=validate_args,
        name=name)

  def _forward(self, x):
    y = x
    # Pad the event_ndims with a zeros vector. We need this because it lets
    # us infer the scale in the inverse function.
    if self._static_event_ndims == 0:
      y = array_ops.expand_dims(y, dim=-1)
      zeros = array_ops.zeros_like(y)
    else:
      shape = array_ops.concat(0, (array_ops.shape(x)[:-1], [1]))
      zeros = array_ops.zeros(shape, dtype=y.dtype)
    y = array_ops.concat(array_ops.rank(y)-1, (y, zeros))

    # Set shape hints.
    if x.get_shape().ndims is not None:
      shape = x.get_shape().as_list()
      if self._static_event_ndims == 0:
        shape += [2]
      elif shape[-1] is not None:
        shape[-1] += 1
      shape = tensor_shape.TensorShape(shape)
      y.get_shape().assert_is_compatible_with(shape)
      y.set_shape(shape)

    # Since we only support event_ndims in [0, 1] and we do padding, we always
    # reduce over the last dimension, i.e., dim=-1 (which is the default).
    return nn_ops.softmax(y)

  def _inverse(self, y):
    # To derive the inverse mapping note that:
    #   y[i] = exp(x[i]) / normalization
    # and
    #   y[end] = 1 / normalization.
    # Thus:
    # x[i] = log(exp(x[i])) - log(y[end]) - log(normalization)
    #      = log(exp(x[i])/normalization) - log(y[end])
    #      = log(y[i]) - log(y[end])
    shape = (np.asarray(y.get_shape().as_list(), dtype=np.int32)
             if y.get_shape().is_fully_defined()
             else array_ops.shape(y, name="shape"))
    ndims = y.get_shape().ndims or math_ops.rank(y, name="ndims")

    # Do this first to make sure CSE catches that it'll happen again in
    # _inverse_log_det_jacobian.
    x = math_ops.log(y)

    # We now extract the last coordinate of the rightmost dimension.
    # Our trick is to slice from [0,0,...,shape[-1]-1] to shape[:-1]+[1].
    begin = array_ops.one_hot(indices=ndims-1,
                              depth=ndims,
                              on_value=shape[-1]-np.array(1, dtype=shape.dtype),
                              dtype=shape.dtype)
    size = array_ops.concat(0, (shape[:-1], np.asarray([1], dtype=shape.dtype)))
    log_normalization = -array_ops.slice(x, begin, size)

    # Here we slice out all but the last coordinate; see above for idea.
    begin = array_ops.zeros_like(shape)
    size = array_ops.concat(0, (shape[:-1], [shape[-1]-1]))
    x = array_ops.slice(x, begin, size)

    x += log_normalization

    if self._static_event_ndims == 0:
      x = array_ops.squeeze(x, squeeze_dims=[ndims-1])

    # Set shape hints.
    if y.get_shape().ndims is not None:
      shape = y.get_shape().as_list()
      if self._static_event_ndims == 0:
        shape = shape[:-1]
      elif shape[-1] is not None:
        shape[-1] -= 1
      shape = tensor_shape.TensorShape(shape)
      x.get_shape().assert_is_compatible_with(shape)
      x.set_shape(shape)

    return x

  def _inverse_log_det_jacobian(self, y):
    # WLOG, consider the vector case:
    #   x = log(y[:-1]) - log(y[-1])
    # where,
    #   y[-1] = 1 - sum(y[:-1]).
    # We have:
    #   det{ dX/dY } = det{ diag(1 ./ y[:-1]) + 1 / y[-1] }
    #                = det{ inv{ diag(y[:-1]) - y[:-1]' y[:-1] } }   (1)
    #                = 1 / det{ diag(y[:-1]) - y[:-1]' y[:-1] }
    #                = 1 / { (1 + y[:-1]' inv(diag(y[:-1])) y[:-1]) *
    #                        det(diag(y[:-1])) }                     (2)
    #                = 1 / { y[-1] prod(y[:-1]) }
    #                = 1 / prod(y)
    # (1) - https://en.wikipedia.org/wiki/Sherman%E2%80%93Morrison_formula
    #       or by noting that det{ dX/dY } = 1 / det{ dY/dX } from Bijector
    #       docstring "Tip".
    # (2) - https://en.wikipedia.org/wiki/Matrix_determinant_lemma
    return -math_ops.reduce_sum(math_ops.log(y), reduction_indices=-1)

  def _forward_log_det_jacobian(self, x):
    if self._static_event_ndims == 0:
      return x - 2. * nn_ops.softplus(x)
    else:
      # This code is similar to nn_ops.log_softmax but different because we have
      # an implicit zero column to handle. I.e., instead of:
      #   reduce_sum(logits - reduce_sum(exp(logits), dim))
      # we must do:
      #   log_normalization = 1 + reduce_sum(exp(logits))
      #   -log_normalization + reduce_sum(logits - log_normalization)
      log_normalization = nn_ops.softplus(
          math_ops.reduce_logsumexp(x, reduction_indices=-1, keep_dims=True))
      fldj = (-log_normalization +
              math_ops.reduce_sum(x - log_normalization,
                                  reduction_indices=-1,
                                  keep_dims=True))
      return array_ops.squeeze(fldj, squeeze_dims=-1)


class SigmoidCentered(SoftmaxCentered):
  """Bijector which computes Y = g(X) = exp([X 0]) / (1 + exp(-X)).

  Equivalent to: `bijector.SoftmaxCentered(event_ndims=0)`.

  See `bijector.SoftmaxCentered` for more details.
  """

  def __init__(self, validate_args=False, name="sigmoid_centered"):
    super(SigmoidCentered, self).__init__(
        validate_args=validate_args, name=name)
