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

An API for invertible, differentiable transformations of random variables.

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
@@CholeskyOuterProduct
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
import collections
import contextlib
import math
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
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops

__all__ = [
    "Bijector",
    "Chain",
    "CholeskyOuterProduct",
    "Exp",
    "Identity",
    "Inline",
    "Invert",
    "ScaleAndShift",
    "SigmoidCentered",
    "SoftmaxCentered",
    "Softplus",
]


class _Mapping(collections.namedtuple("_Mapping",
                                      ["x", "y", "ildj", "condition_kwargs"])):
  """Helper class to make it easier to manage caching in `Bijector`."""

  def __new__(cls, x=None, y=None, ildj=None, condition_kwargs=None):
    """Custom __new__ so namedtuple items have defaults.

    Args:
      x: `Tensor`. Forward.
      y: `Tensor`. Inverse.
      ildj: `Tensor`. Inverse log det Jacobian.
      condition_kwargs: Python dictionary. Extra args supplied to
        forward/inverse/etc functions.

    Returns:
      mapping: New instance of _Mapping.
    """
    return super(_Mapping, cls).__new__(cls, x, y, ildj, condition_kwargs)

  @property
  def x_key(self):
    """Returns key used for caching Y=g(X)."""
    return (self.x,) + self._deep_tuple(tuple(sorted(
        self.condition_kwargs.items())))

  @property
  def y_key(self):
    """Returns key used for caching X=g^{-1}(Y)."""
    return (self.y,) + self._deep_tuple(tuple(sorted(
        self.condition_kwargs.items())))

  def merge(self, x=None, y=None, ildj=None,
            condition_kwargs=None, mapping=None):
    """Returns new _Mapping with args merged with self.

    Args:
      x: `Tensor`. Forward.
      y: `Tensor`. Inverse.
      ildj: `Tensor`. Inverse log det Jacobian.
      condition_kwargs: Python dictionary. Extra args supplied to
        forward/inverse/etc functions.
      mapping: Instance of _Mapping to merge. Can only be specified if no other
        arg is specified.

    Returns:
      mapping: New instance of `_Mapping` which has inputs merged with self.

    Raises:
      ValueError: if mapping and any other arg is not `None`.
    """
    if mapping is None:
      mapping = _Mapping(x=x, y=y, ildj=ildj,
                         condition_kwargs=condition_kwargs)
    elif not all([arg is None for arg in [x, y, ildj, condition_kwargs]]):
      raise ValueError("Cannot specify mapping and individual args.")
    return _Mapping(
        x=self._merge(self.x, mapping.x),
        y=self._merge(self.y, mapping.y),
        ildj=self._merge(self.ildj, mapping.ildj),
        condition_kwargs=self._merge(self.condition_kwargs,
                                     mapping.condition_kwargs))

  def _merge(self, old, new):
    """Helper to merge which handles merging one value."""
    if old is None:
      return new
    elif new is not None and old != new:
      raise ValueError("Incompatible values: %s != %s" % (old, new))
    return old

  def _deep_tuple(self, x):
    """Converts lists of lists to tuples of tuples."""
    return (tuple(map(self._deep_tuple, x))
            if isinstance(x, (list, tuple)) else x)


@six.add_metaclass(abc.ABCMeta)
class Bijector(object):
  """Interface for transforming a `Distribution` sample.

  A `Bijector` implements a
  [diffeomorphism](https://en.wikipedia.org/wiki/Diffeomorphism), i.e., a
  bijective, differentiable function. A `Bijector` is used by
  `TransformedDistribution` but can be generally used for transforming a
  `Distribution` generated `Tensor`.  A `Bijector` is characterized by three
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

  - Subclasses should implement `_get_forward_event_shape`,
    `_forward_event_shape` (and `inverse` counterparts) if the transformation is
    shape-changing.  By default the event-shape is assumed unchanged from input.

  Tips for implementing `_inverse` and `_inverse_log_det_jacobian`:

  - As case 3 [above] indicates, under some circumstances the inverse function
    can be implemented as a cache lookup.

  - The inverse `log o det o Jacobian` can be implemented as the negative of the
    forward `log o det o Jacobian`.  This is useful if the `inverse` is
    implemented as a cache or the inverse Jacobian is computationally more
    expensive (e.g., `CholeskyOuterProduct` `Bijector`). The following
    demonstrates the suggested implementation.

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
    self._from_y = {}
    self._from_x = {}
    # Using abbreviation ildj for "inverse log det Jacobian."
    # This variable is not `None` iff is_constant_jacobian is `True`.
    self._constant_ildj = None
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

  def _forward_event_shape(self, input_shape):
    """Subclass implementation for `forward_event_shape` public function."""
    return input_shape

  def forward_event_shape(self, input_shape, name="forward_event_shape"):
    """Shape of a single sample from a single batch as an `int32` 1D `Tensor`.

    Args:
      input_shape: `Tensor`, `int32` vector indicating event-portion shape
        passed into `forward` function.
      name: name to give to the op

    Returns:
      forward_event_shape: `Tensor`, `int32` vector indicating event-portion
        shape after applying `forward`.
    """
    with self._name_scope(name, [input_shape]):
      input_shape = ops.convert_to_tensor(input_shape, dtype=dtypes.int32,
                                          name="input_shape")
      return self._forward_event_shape(input_shape)

  def _get_forward_event_shape(self, input_shape):
    """Subclass implementation for `get_forward_event_shape` public function."""
    return input_shape

  def get_forward_event_shape(self, input_shape):
    """Shape of a single sample from a single batch as a `TensorShape`.

    Same meaning as `forward_event_shape`. May be only partially defined.

    Args:
      input_shape: `TensorShape` indicating event-portion shape passed into
        `forward` function.

    Returns:
      forward_event_shape: `TensorShape` indicating event-portion shape after
        applying `forward`. Possibly unknown.
    """
    return self._get_forward_event_shape(tensor_shape.TensorShape(input_shape))

  def _inverse_event_shape(self, output_shape):
    """Subclass implementation for `inverse_event_shape` public function."""
    return output_shape

  def inverse_event_shape(self, output_shape, name="inverse_event_shape"):
    """Shape of a single sample from a single batch as an `int32` 1D `Tensor`.

    Args:
      output_shape: `Tensor`, `int32` vector indicating event-portion shape
        passed into `inverse` function.
      name: name to give to the op

    Returns:
      inverse_event_shape: `Tensor`, `int32` vector indicating event-portion
        shape after applying `inverse`.
    """
    with self._name_scope(name, [output_shape]):
      output_shape = ops.convert_to_tensor(output_shape, dtype=dtypes.int32,
                                           name="output_shape")
      return self._inverse_event_shape(output_shape)

  def _get_inverse_event_shape(self, output_shape):
    """Subclass implementation for `get_inverse_event_shape` public function."""
    return self._get_inverse_event_shape(tensor_shape.TensorShape(output_shape))

  def get_inverse_event_shape(self, output_shape):
    """Shape of a single sample from a single batch as a `TensorShape`.

    Same meaning as `inverse_event_shape`. May be only partially defined.

    Args:
      output_shape: `TensorShape` indicating event-portion shape passed into
        `inverse` function.

    Returns:
      inverse_event_shape: `TensorShape` indicating event-portion shape after
        applying `inverse`. Possibly unknown.
    """
    return self._get_inverse_event_shape(output_shape)

  def _forward(self, x):
    """Subclass implementation for `forward` public function."""
    raise NotImplementedError("forward not implemented.")

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
      mapping = self._lookup(x=x, condition_kwargs=condition_kwargs)
      if mapping.y is not None:
        return mapping.y
      mapping = mapping.merge(y=self._forward(x, **condition_kwargs))
      self._cache(mapping)
      return mapping.y

  def _inverse(self, y):
    """Subclass implementation for `inverse` public function."""
    raise NotImplementedError("inverse not implemented")

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
      mapping = self._lookup(y=y, condition_kwargs=condition_kwargs)
      if mapping.x is not None:
        return mapping.x
      ildj = None
      try:
        x = self._inverse(y, **condition_kwargs)
      except NotImplementedError as original_error:
        # Since _inverse was not implemented, try to see if it's implemented
        # by the _inverse_and_inverse_log_det_jacobian member.
        try:
          x, ildj = self._inverse_and_inverse_log_det_jacobian(
              y, **condition_kwargs)
          if self._constant_ildj is not None:
            ildj = self._constant_ildj  # Use the "global" result.
          elif self.is_constant_jacobian:
            self._constant_ildj = ildj
        except NotImplementedError:
          raise original_error
      x = x if mapping.x is None else mapping.x
      mapping = mapping.merge(x=x, ildj=ildj)
      self._cache(mapping)
      return mapping.x

  def _inverse_log_det_jacobian(self, y):
    """Subclass implementation for `inverse_log_det_jacobian` public function."""  # pylint: disable=line-too-long
    raise NotImplementedError("inverse_log_det_jacobian not implemented.")

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
      if self._constant_ildj is not None:
        return self._constant_ildj
      y = ops.convert_to_tensor(y, name="y")
      self._maybe_assert_dtype(y)
      mapping = self._lookup(y=y, condition_kwargs=condition_kwargs)
      if mapping.ildj is not None:
        return mapping.ildj
      try:
        x = mapping.x
        ildj = self._inverse_log_det_jacobian(y, **condition_kwargs)
      except NotImplementedError as original_error:
        # Since _inverse_log_det_jacobian was not implemented, try to see if
        # it's implemented by the _inverse_and_inverse_log_det_jacobian member.
        try:
          x, ildj = self._inverse_and_inverse_log_det_jacobian(
              y, **condition_kwargs)
          if mapping.x is not None:
            x = mapping.x
        except NotImplementedError:
          raise original_error
      if self.is_constant_jacobian:
        self._constant_ildj = ildj
      x = x if mapping.x is None else mapping.x
      mapping = mapping.merge(x=x, ildj=ildj)
      self._cache(mapping)
      return mapping.ildj

  def _inverse_and_inverse_log_det_jacobian(self, y):
    """Subclass implementation for `inverse_and_inverse_log_det_jacobian` public function."""  # pylint: disable=line-too-long
    raise NotImplementedError(
        "inverse_and_inverse_log_det_jacobian not implemented.")

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
      mapping = self._lookup(y=y, condition_kwargs=condition_kwargs)
      if mapping.x is not None and mapping.ildj is not None:
        return mapping.x, mapping.ildj
      try:
        x, ildj = self._inverse_and_inverse_log_det_jacobian(
            y, **condition_kwargs)
      except NotImplementedError as original_error:
        # Since _inverse_and_inverse_log_det_jacobian was not implemented, try
        # to see if we can separately use _inverse and
        # _inverse_log_det_jacobian members.
        try:
          x = self._inverse(y, **condition_kwargs)
          if self._constant_ildj is None:
            ildj = self._inverse_log_det_jacobian(y, **condition_kwargs)
        except NotImplementedError:
          raise original_error
      if self._constant_ildj is not None:
        ildj = self._constant_ildj  # Ignore any ildj we may/not have.
      elif self.is_constant_jacobian:
        self._constant_ildj = ildj
      # We use the mapped version of x, even if we re-computed x above with a
      # call to self._inverse_and_inverse_log_det_jacobian.  This prevents
      # re-evaluation of the inverse in a common case.
      x = x if mapping.x is None else mapping.x
      mapping = mapping.merge(x=x, ildj=ildj)
      self._cache(mapping)
      return mapping.x, mapping.ildj

  def _forward_log_det_jacobian(self, x):
    """Subclass implementation for `forward_log_det_jacobian` public function."""  # pylint: disable=line-too-long
    raise NotImplementedError(
        "forward_log_det_jacobian not implemented.")

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
      if self._constant_ildj is not None:
        # Need "-1. *" to avoid invalid-unary-operand-type linter warning.
        return -1. * self._constant_ildj
      x = ops.convert_to_tensor(x, name="x")
      self._maybe_assert_dtype(x)
      mapping = self._lookup(x=x, condition_kwargs=condition_kwargs)
      if mapping.ildj is not None:
        return -mapping.ildj
      y = None
      try:
        ildj = -self._forward_log_det_jacobian(x, **condition_kwargs)
      except NotImplementedError as original_error:
        try:
          y = self.inverse(x, **condition_kwargs) if y is None else y
          ildj = self.inverse_log_det_jacobian(y, **condition_kwargs)
        except NotImplementedError:
          raise original_error
      if self.is_constant_jacobian:
        self._constant_ildj = ildj
      y = y if mapping.y is None else mapping.y
      mapping = mapping.merge(y=y, ildj=ildj)
      self._cache(mapping)
      return -mapping.ildj

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

  def _cache(self, mapping):
    """Helper which stores mapping info in forward/inverse dicts."""
    if self._constant_ildj is not None:
      # Fold in ildj if known constant Jacobian.
      mapping = mapping.merge(ildj=self._constant_ildj)
    # Merging from lookup is an added check that we're not overwriting anything
    # which is not None.
    mapping = mapping.merge(mapping=self._lookup(
        mapping.x, mapping.y, mapping.condition_kwargs))
    if mapping.x is None or mapping.y is None:
      ValueError("Caching expects both (x,y) to be known, i.e., not None.")
    self._from_x[mapping.x_key] = mapping
    self._from_y[mapping.y_key] = mapping

  def _lookup(self, x=None, y=None, condition_kwargs=None):
    """Helper which retrieves mapping info from forward/inverse dicts."""
    mapping = _Mapping(x=x, y=y, condition_kwargs=condition_kwargs)
    # Since _cache requires both x,y to be set, we only need to do one cache
    # lookup since the mapping is always in both or neither.
    if mapping.x is not None:
      return self._from_x.get(mapping.x_key, mapping)
    if mapping.y is not None:
      return self._from_y.get(mapping.y_key, mapping)
    return mapping


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
    name="exp")
  ```

  The above example is equivalent to the `Bijector` `Exp(event_ndims=1)`.
  """
  # pylint: enable=line-too-long

  def __init__(self,
               forward_fn=None,
               inverse_fn=None,
               inverse_log_det_jacobian_fn=None,
               forward_log_det_jacobian_fn=None,
               get_forward_event_shape_fn=None,
               forward_event_shape_fn=None,
               get_inverse_event_shape_fn=None,
               inverse_event_shape_fn=None,
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
      get_forward_event_shape_fn: Python callable implementing non-identical
        static event shape changes. Default: shape is assumed unchanged.
      forward_event_shape_fn: Python callable implementing non-identical event
        shape changes. Default: shape is assumed unchanged.
      get_inverse_event_shape_fn: Python callable implementing non-identical
        static event shape changes. Default: shape is assumed unchanged.
      inverse_event_shape_fn: Python callable implementing non-identical event
        shape changes. Default: shape is assumed unchanged.
      is_constant_jacobian: `Boolean` indicating that the Jacobian is constant
        for all input arguments.
      validate_args: `Boolean` indicating whether arguments should be checked
        for correctness.
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
    self._get_forward_event_shape_fn = get_forward_event_shape_fn
    self._forward_event_shape_fn = forward_event_shape_fn
    self._get_inverse_event_shape_fn = get_inverse_event_shape_fn
    self._inverse_event_shape_fn = inverse_event_shape_fn

  def _get_forward_event_shape(self, input_shape):
    if self._get_forward_event_shape_fn is None:
      # By default assume shape doesn't change.
      return input_shape
    return self._get_forward_event_shape_fn(input_shape)

  def _forward_event_shape(self, input_shape):
    if self._forward_event_shape_fn is None:
      # By default assume shape doesn't change.
      return input_shape
    return self._forward_event_shape_fn(input_shape)

  def _get_inverse_event_shape(self, output_shape):
    if self._get_inverse_event_shape_fn is None:
      # By default assume shape doesn't change.
      return output_shape
    return self._get_inverse_event_shape_fn(output_shape)

  def _inverse_event_shape(self, output_shape):
    if self._inverse_event_shape_fn is None:
      # By default assume shape doesn't change.
      return output_shape
    return self._inverse_event_shape_fn(output_shape)

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
      validate_args: `Boolean` indicating whether arguments should be checked
        for correctness.
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

  def _get_forward_event_shape(self, input_shape):
    return self.bijector.get_inverse_event_shape(input_shape)

  def _forward_event_shape(self, input_shape):
    return self.bijector.inverse_event_shape(input_shape)

  def _get_inverse_event_shape(self, output_shape):
    return self.bijector.get_forward_event_shape(output_shape)

  def _inverse_event_shape(self, output_shape):
    return self.bijector.forward_event_shape(output_shape)

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
      validate_args: `Boolean` indicating whether arguments should be checked
        for correctness.
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

  def _shape_helper(self, func_name, input_shape, reverse):
    new_shape = input_shape
    for b in reversed(self.bijectors) if reverse else self.bijectors:
      func = getattr(b, func_name, None)
      if func is None:
        raise ValueError("unable to call %s on bijector %s (%s)" %
                         (func_name, b.name, func))
      new_shape = func(new_shape)
    return new_shape

  def _get_forward_event_shape(self, input_shape):
    return self._shape_helper("get_forward_event_shape", input_shape,
                              reverse=True)

  def _forward_event_shape(self, input_shape):
    return self._shape_helper("forward_event_shape", input_shape, reverse=True)

  def _get_inverse_event_shape(self, output_shape):
    return self._shape_helper("get_inverse_event_shape", output_shape,
                              reverse=False)

  def _inverse_event_shape(self, output_shape):
    return self._shape_helper("inverse_event_shape", output_shape,
                              reverse=False)

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
        is_constant_jacobian=True,
        validate_args=validate_args,
        name=name)

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
      validate_args: `Boolean` indicating whether arguments should be checked
        for correctness.
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
  """Bijector which computes Y = g(X; shift, scale) = matmul(scale, X) + shift.

  `scale` is either a non-zero scalar, or a lower triangular matrix with
  non-zero diagonal.  This means the `Bijector` will be invertible and
  computation of determinant and inverse will be efficient.

  As a result, the mean and covariance are transformed:

  ```
  E[Y] = matmul(scale, E[X])
  Cov[Y] = matmul(scale, matmul(Cov[X], scale, transpose_b=True))
  ```

  Example Use:

  ```python
  # No batch, scalar
  mu = 0     # shape=[]
  sigma = 1  # shape=[], treated like a 1x1 matrix.
  b = ScaleAndShift(shift=mu, scale=sigma)
  # b.shaper.batch_ndims == 0
  # b.shaper.event_ndims == 0

  # One batch, scalar.
  mu = ...    # shape=[b], b>0
  sigma = ... # shape=[b], b>0, treated like a batch of 1x1 matrices
  b = ScaleAndShift(shift=mu, scale=sigma)
  # b.shaper.batch_ndims == 1
  # b.shaper.event_ndims == 0

  # No batch, multivariate.
  mu = ...    # shape=[d],    d>0
  sigma = ... # shape=[d, d], d>0, treated like a single dxd matrix.
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
    """Instantiates the `ScaleAndShift` bijector.

    This `Bijector` is initialized with `scale` and `shift` `Tensors`, giving
    the forward operation:

    ```Y = g(X) = matmul(scale, X) + shift```

    Args:
      shift: Numeric `Tensor`.
      scale: Numeric `Tensor` of same `dtype` as `shift`.  If `event_ndims = 0`,
        `scale` is treated like a `1x1` matrix or a batch thereof.
        Otherwise, the last two dimensions of `scale` define a matrix.
        `scale` must have non-negative diagonal entries.  The upper triangular
        part of `scale` is ignored, effectively making it lower triangular.
      event_ndims: Scalar `int32` `Tensor` indicating the number of dimensions
        associated with a particular draw from the distribution.  Must be 0 or 1
      validate_args: `Boolean` indicating whether arguments should be checked
        for correctness.
      name: `String` name given to ops managed by this object.
    """

    self._parameters = {}
    self._name = name
    self._validate_args = validate_args
    with self._name_scope("init", values=[shift, scale, event_ndims]):
      self._shift = ops.convert_to_tensor(shift, name="shift")
      self._scale = ops.convert_to_tensor(scale, name="scale")
      event_ndims = ops.convert_to_tensor(event_ndims, name="event_ndims")
      if validate_args:
        event_ndims = control_flow_ops.with_dependencies(
            [check_ops.assert_less(
                event_ndims, 2, message="event_ndims must be 0 or 1")],
            event_ndims)
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
    # For safety, explicitly zero-out the upper triangular part.
    scale = array_ops.matrix_band_part(scale, -1, 0)
    if self.validate_args:
      # matrix_band_part will fail if scale is not at least rank 2.
      shape = array_ops.shape(scale)
      assert_square = check_ops.assert_equal(
          shape[-2], shape[-1],
          message="Input must be a (batch of) square matrix.")
      # Assuming lower-triangular means we only need check diag != 0.
      diag = array_ops.matrix_diag_part(scale)
      is_non_singular = math_ops.logical_not(
          math_ops.reduce_any(
              math_ops.equal(diag, ops.convert_to_tensor(0, dtype=diag.dtype))))
      assert_non_singular = control_flow_ops.Assert(
          is_non_singular, ["Singular matrix encountered", diag])
      scale = control_flow_ops.with_dependencies(
          [assert_square, assert_non_singular], scale)
    return scale, batch_ndims

  @property
  def shift(self):
    return self._shift

  @property
  def scale(self):
    return self._scale

  def _forward(self, x):
    x, sample_shape = self.shaper.make_batch_of_event_sample_matrices(x)
    x = math_ops.matmul(self.scale, x)
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
    abs_diag = math_ops.abs(array_ops.matrix_diag_part(self.scale))
    return -math_ops.reduce_sum(math_ops.log(abs_diag), reduction_indices=[-1])

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
    # The most stable inverse of softplus is not the most obvious one.
    # y = softplus(x) = Log[1 + exp{x}], (which means y > 0).
    # ==> exp{y} = 1 + exp{x}                                (1)
    # ==> x = Log[exp{y} - 1]                                (2)
    #       = Log[(exp{y} - 1) / exp{y}] + Log[exp{y}]
    #       = Log[(1 - exp{-y}) / 1] + Log[exp{y}]
    #       = Log[1 - exp{-y}] + y                           (3)
    # (2) is the "obvious" inverse, but (3) is more stable than (2) for large y.
    # For small y (e.g. y = 1e-10), (3) will become -inf since 1 - exp{-y} will
    # be zero.  To fix this, we use 1 - exp{-y} approx y for small y > 0.
    #
    # Stable inverse log det jacobian.
    # Y = Log[1 + exp{X}] ==> X = Log[exp{Y} - 1]
    # ==> dX/dY = exp{Y} / (exp{Y} - 1)
    #           = 1 / (1 - exp{-Y}),
    # which is the most stable for large Y > 0.  For small Y, we use
    # 1 - exp{-Y} approx Y.
    if self.shaper is None:
      raise ValueError("Jacobian cannot be computed with unknown event_ndims")
    _, _, event_dims = self.shaper.get_dims(y)
    # eps is smallest positive number such that 1 + eps != 1.
    eps = np.finfo(y.dtype.base_dtype.as_numpy_dtype).eps
    # Approximate exp{-y} ~ 1 - y for small y > 0, then use exp{-y} elsewhere.
    # Note we are careful to never send an NaN through ANY branch of where.
    # TODO(langmore) replace with -tf.expm1(y) when it exists.
    one_minus_exp_neg_y = array_ops.where(
        y < eps,
        y,
        1. - math_ops.exp(-y))
    log_one_minus_exp_neg = math_ops.log(one_minus_exp_neg_y)
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

  def _get_forward_event_shape(self, input_shape):
    if input_shape.ndims is None:
      return input_shape
    if input_shape.ndims != self._static_event_ndims:
      raise ValueError("input_shape.dims = %d != %d" %
                       (input_shape.ndims, self._static_event_ndims))
    if input_shape.ndims == 0:
      return tensor_shape.TensorShape([2])
    if input_shape.ndims == 1:
      return tensor_shape.TensorShape(input_shape[0] + 1)
    # Unreachable code:
    raise ValueError("event_ndims = %d must be 0 or 1" % input_shape.ndims)

  def _forward_event_shape(self, input_shape):
    ndims = array_ops.shape(input_shape)
    if self.validate_args:
      # It is not possible for a negative shape so we need only check <= 1.
      is_zero_or_one = check_ops.assert_equal(
          ndims, 0 if self._static_event_ndims == 0 else 1,
          message="event_ndims must be 0 or 1")
      ndims = control_flow_ops.with_dependencies([is_zero_or_one], ndims)
    if self._static_event_ndims == 0:
      return ops.convert_to_tensor(
          [2], dtype=dtypes.int32, name="output_shape")
    return input_shape + 1

  def _get_inverse_event_shape(self, output_shape):
    if output_shape.ndims is None:
      return output_shape
    if output_shape.ndims != 1:
      raise ValueError("output_shape.ndims = %d != 1" % output_shape.ndims)
    if self._static_event_ndims == 0:
      return tensor_shape.TensorShape([])
    return tensor_shape.TensorShape(output_shape[0] - 1)

  def _inverse_event_shape(self, output_shape):
    ndims = array_ops.shape(output_shape)[0]
    if self.validate_args:
      # It is not possible for a negative shape so we need only check <= 1.
      is_one = check_ops.assert_equal(
          ndims, 1, message="event_ndims must be 1")
      ndims = control_flow_ops.with_dependencies([is_one], ndims)
    if self._static_event_ndims == 0:
      return ops.convert_to_tensor([], dtype=dtypes.int32, name="output_shape")
    return array_ops.expand_dims(output_shape[0] - 1, dim=0)

  def _forward(self, x):
    # Pad the last dim with a zeros vector. We need this because it lets us
    # infer the scale in the inverse function.
    y = array_ops.expand_dims(x, dim=-1) if self._static_event_ndims == 0 else x
    ndims = (y.get_shape().ndims if y.get_shape().ndims is not None
             else array_ops.rank(y))
    y = array_ops.pad(y, paddings=array_ops.concat(0, (
        array_ops.zeros((ndims - 1, 2), dtype=dtypes.int32),
        [[0, 1]])))

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


class CholeskyOuterProduct(Bijector):
  # pylint: disable=line-too-long
  """Bijector which computes Y = g(X) = X X^T where X is a lower-triangular, positive-diagonal matrix.

  `event_ndims` must be 0 or 2, i.e., scalar or matrix.

  Note: the upper-triangular part of X is ignored (whether or not its zero).

  Examples:

  ```python
  bijector.CholeskyOuterProduct(event_ndims=2).forward(x=[[1., 0], [2, 1]])
  # Result: [[1, 1], [1, 5]], i.e., x x^T

  bijector.SoftmaxCentered(event_ndims=2).inverse(y=[[1., 1], [1, 5]])
  # Result: [[1, 0], [2, 1]], i.e., chol(y).
  ```

  """
  # pylint: enable=line-too-long

  def __init__(self, event_ndims=2, validate_args=False,
               name="cholesky_outer_product"):
    """Instantiates the `CholeskyOuterProduct` bijector.

    Args:
      event_ndims: `constant` `int32` scalar `Tensor` indicating the number of
        dimensions associated with a particular draw from the distribution. Must
        be 0 or 2.
      validate_args: `Boolean` indicating whether arguments should be checked
        for correctness.
      name: `String` name given to ops managed by this object.

    Raises:
      ValueError: if event_ndims is neither 0 or 2.
    """
    self._parameters = {}
    self._name = name
    with self._name_scope("init", values=[event_ndims]):
      event_ndims = ops.convert_to_tensor(event_ndims, name="event_ndims")
      event_ndims = tensor_util.constant_value(event_ndims)
    if event_ndims is None or event_ndims not in [0, 2]:
      raise ValueError("`event_ndims` must be a TF constant which is 0 or 2")
    self._static_event_ndims = event_ndims
    super(CholeskyOuterProduct, self).__init__(
        validate_args=validate_args,
        name=name)

  def _forward(self, x):
    if self._static_event_ndims == 0:
      return math_ops.square(x)
    if self.validate_args:
      is_matrix = check_ops.assert_rank_at_least(x, 2)
      shape = array_ops.shape(x)
      is_square = check_ops.assert_equal(shape[-2], shape[-1])
      x = control_flow_ops.with_dependencies([is_matrix, is_square], x)
    # For safety, explicitly zero-out the upper triangular part.
    x = array_ops.matrix_band_part(x, -1, 0)
    return math_ops.matmul(x, x, adjoint_b=True)

  def _inverse_and_inverse_log_det_jacobian(self, y):
    x = (math_ops.sqrt(y) if self._static_event_ndims == 0
         else linalg_ops.cholesky(y))
    return x, -self._forward_log_det_jacobian(x)

  def _forward_log_det_jacobian(self, x):
    # Let Y be a symmetric, positive definite matrix and write:
    #   Y = X X^T
    # where X is lower-triangular.
    #
    # Observe that,
    #   dY[i,j]/dX[a,b]
    #   = d/dX[a,b] { X[i,:] X[j,:] }
    #   = sum_{d=1}^p { I[i=a] I[d=b] X[j,d] + I[j=a] I[d=b] X[i,d] }
    #
    # To compute the Jacobian dX/dY we must represent X,Y as vectors. Since Y is
    # symmetric and X is lower-triangular, we need vectors of dimension:
    #   d = p (p + 1) / 2
    # where X, Y are p x p matrices, p > 0. We use a row-major mapping, i.e.,
    #   k = { i (i + 1) / 2 + j   i>=j
    #       { undef               i<j
    # and assume zero-based indexes. When k is undef, the element is dropped.
    # Example:
    #           j      k
    #        0 1 2 3  /
    #    0 [ 0 . . . ]
    # i  1 [ 1 2 . . ]
    #    2 [ 3 4 5 . ]
    #    3 [ 6 7 8 9 ]
    # Write vec[.] to indicate transforming a matrix to vector via k(i,j). (With
    # slight abuse: k(i,j)=undef means the element is dropped.)
    #
    # We now show d vec[Y] / d vec[X] is lower triangular. Assuming both are
    # defined, observe that k(i,j) < k(a,b) iff (1) i<a or (2) i=a and j<b.
    # In both cases dvec[Y]/dvec[X]@[k(i,j),k(a,b)] = 0 since:
    # (1) j<=i<a thus i,j!=a.
    # (2) i=a>j  thus i,j!=a.
    #
    # Since the Jacobian is lower-triangular, we need only compute the product
    # of diagonal elements:
    #   d vec[Y] / d vec[X] @[k(i,j), k(i,j)]
    #   = X[j,j] + I[i=j] X[i,j]
    #   = 2 X[j,j].
    # Since there is a 2 X[j,j] term for every lower-triangular element of X we
    # conclude:
    #   |Jac(d vec[Y]/d vec[X])| = 2^p prod_{j=0}^{p-1} X[j,j]^{p-j}.
    if self._static_event_ndims == 0:
      if self.validate_args:
        is_positive = check_ops.assert_positive(
            x, message="All elements must be positive.")
        x = control_flow_ops.with_dependencies([is_positive], x)
      return math.log(2.) + math_ops.log(x)

    diag = array_ops.matrix_diag_part(x)
    if self.validate_args:
      is_matrix = check_ops.assert_rank_at_least(
          x, 2, message="Input must be a (batch of) matrix.")
      shape = array_ops.shape(x)
      is_square = check_ops.assert_equal(
          shape[-2], shape[-1],
          message="Input must be a (batch of) square matrix.")
      # Assuming lower-triangular means we only need check diag>0.
      is_positive_definite = check_ops.assert_positive(
          diag, message="Input must be positive definite.")
      x = control_flow_ops.with_dependencies(
          [is_matrix, is_square, is_positive_definite], x)

    # Create a column vector equal to: [p, p-1, ..., 2, 1]^T.
    if x.get_shape().ndims is None or x.get_shape()[-1].value is None:
      p = array_ops.shape(x)[-1]
    else:
      p = x.get_shape()[-1].value
    exponents = array_ops.expand_dims(
        math_ops.linspace(math_ops.cast(p, dtype=x.dtype), 1., p),
        dim=1)

    sum_weighted_log_diag = array_ops.squeeze(
        math_ops.matmul(math_ops.log(diag), exponents), squeeze_dims=-1)
    fldj = p * math.log(2.) + sum_weighted_log_diag

    if x.get_shape().ndims is not None:
      fldj.set_shape(x.get_shape()[:-2])

    return fldj
