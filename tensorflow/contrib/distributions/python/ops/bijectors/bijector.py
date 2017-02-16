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
"""Bijector base."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import collections
import contextlib
import re

import numpy as np
import six

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops


__all__ = [
    "Bijector",
]


class _Mapping(collections.namedtuple(
    "_Mapping", ["x", "y", "ildj", "kwargs"])):
  """Helper class to make it easier to manage caching in `Bijector`."""

  def __new__(cls, x=None, y=None, ildj=None, kwargs=None):
    """Custom __new__ so namedtuple items have defaults.

    Args:
      x: `Tensor`. Forward.
      y: `Tensor`. Inverse.
      ildj: `Tensor`. Inverse log det Jacobian.
      kwargs: Python dictionary. Extra args supplied to
        forward/inverse/etc functions.

    Returns:
      mapping: New instance of _Mapping.
    """
    return super(_Mapping, cls).__new__(cls, x, y, ildj, kwargs)

  @property
  def x_key(self):
    """Returns key used for caching Y=g(X)."""
    return (self.x,) + self._deep_tuple(tuple(sorted(self.kwargs.items())))

  @property
  def y_key(self):
    """Returns key used for caching X=g^{-1}(Y)."""
    return (self.y,) + self._deep_tuple(tuple(sorted(self.kwargs.items())))

  def merge(self, x=None, y=None, ildj=None, kwargs=None, mapping=None):
    """Returns new _Mapping with args merged with self.

    Args:
      x: `Tensor`. Forward.
      y: `Tensor`. Inverse.
      ildj: `Tensor`. Inverse log det Jacobian.
      kwargs: Python dictionary. Extra args supplied to
        forward/inverse/etc functions.
      mapping: Instance of _Mapping to merge. Can only be specified if no other
        arg is specified.

    Returns:
      mapping: New instance of `_Mapping` which has inputs merged with self.

    Raises:
      ValueError: if mapping and any other arg is not `None`.
    """
    if mapping is None:
      mapping = _Mapping(x=x, y=y, ildj=ildj, kwargs=kwargs)
    elif not all(arg is None for arg in [x, y, ildj, kwargs]):
      raise ValueError("Cannot specify mapping and individual args.")
    return _Mapping(
        x=self._merge(self.x, mapping.x),
        y=self._merge(self.y, mapping.y),
        ildj=self._merge(self.ildj, mapping.ildj),
        kwargs=self._merge(self.kwargs, mapping.kwargs))

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
  `Distribution` generated `Tensor`. A `Bijector` is characterized by three
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
     of another. Geometrically, the det(Jacobian) is the volume of the
     transformation and is used to scale the probability.

  By convention, transformations of random variables are named in terms of the
  forward transformation. The forward transformation creates samples, the
  inverse is useful for computing probabilities.

  Example Use:

    - Basic properties:

    ```python
    x = ...  # A tensor.
    # Evaluate forward transformation.
    fwd_x = my_bijector.forward(x)
    x == my_bijector.inverse(fwd_x)
    x != my_bijector.forward(fwd_x)  # Not equal because g(x) != g(g(x)).
    ```

    - Computing a log-likelihood:

    ```python
    def transformed_log_prob(bijector, log_prob, x):
      return (bijector.inverse_log_det_jacobian(x) +
              log_prob(bijector.inverse(x)))
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

      Here is an example of how one might implement the `Exp` bijector:

      ```
        class Exp(Bijector):
          def __init__(self, event_ndims=0, validate_args=False, name="exp"):
            super(Exp, self).__init__(event_ndims=event_ndims,
                                      validate_args=validate_args, name=name)
          def _forward(self, x):
            return math_ops.exp(x)
          def _inverse_and_inverse_log_det_jacobian(self, y):
            x = math_ops.log(y)
            return x, -self._forward_log_det_jacobian(x)
          def _forward_log_det_jacobian(self, x):
            if self.event_ndims is None:
              raise ValueError("Jacobian requires known event_ndims.")
            event_dims = array_ops.shape(x)[-self.event_ndims:]
            return math_ops.reduce_sum(x, axis=event_dims)
        ```

    - "Affine"

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
    and event (S, B, E) shape semantics. Suppose the `Tensor`'s
    partitioned-shape is `(S=[4], B=[2], E=[3, 3])`.

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
    to implement all of `_forward` and `_inverse` functions. Examples:
      1. Sampling (e.g., `sample`) only requires `_forward`.
      2. Probability functions (e.g., `prob`, `cdf`, `survival`) only require
         `_inverse` (and related).
      3. Only calling probability functions on the output of `sample` means
        `_inverse` can be implemented as a cache lookup.

    See `Example Use` [above] which shows how these functions are used to
    transform a distribution. (Note: `_forward` could theoretically be
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

  - Subclasses should implement `_forward_event_shape`,
    `_forward_event_shape_tensor` (and `inverse` counterparts) if the
    transformation is shape-changing. By default the event-shape is assumed
    unchanged from input.

  Tips for implementing `_inverse` and `_inverse_log_det_jacobian`:

  - As case 3 [above] indicates, under some circumstances the inverse function
    can be implemented as a cache lookup.

  - The inverse `log o det o Jacobian` can be implemented as the negative of the
    forward `log o det o Jacobian`. This is useful if the `inverse` is
    implemented as a cache or the inverse Jacobian is computationally more
    expensive (e.g., `CholeskyOuterProduct` `Bijector`). The following
    demonstrates the suggested implementation.

    ```python
    def _inverse_and_log_det_jacobian(self, y):
       x = ...  # implement inverse, possibly via cache.
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
               event_ndims=None,
               graph_parents=None,
               is_constant_jacobian=False,
               validate_args=False,
               dtype=None,
               name=None):
    """Constructs Bijector.

    A `Bijector` transforms random variables into new random variables.

    Examples:

    ```python
    # Create the Y = g(X) = X transform which operates on vector events.
    identity = Identity(event_ndims=1)

    # Create the Y = g(X) = exp(X) transform which operates on matrices.
    exp = Exp(event_ndims=2)
    ```

    See `Bijector` subclass docstring for more details and specific examples.

    Args:
      event_ndims: number of dimensions associated with event coordinates.
      graph_parents: Python list of graph prerequisites of this `Bijector`.
      is_constant_jacobian: Python `bool` indicating that the Jacobian is not a
        function of the input.
      validate_args: Python `bool`, default `False`. Whether to validate input
        with asserts. If `validate_args` is `False`, and the inputs are invalid,
        correct behavior is not guaranteed.
      dtype: `tf.dtype` supported by this `Bijector`. `None` means dtype is not
        enforced.
      name: The name to give Ops created by the initializer.
    """
    self._event_ndims = (
        ops.convert_to_tensor(event_ndims, dtype=dtypes.int32)
        if event_ndims is not None else None)
    self._graph_parents = graph_parents or []
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
      # since `Chain` uses bijector.name as the kwargs dictionary key.
      def camel_to_snake(name):
        s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
        return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()
      self._name = camel_to_snake(type(self).__name__.lstrip("_"))

  @property
  def event_ndims(self):
    """Returns then number of event dimensions this bijector operates on."""
    return self._event_ndims

  @property
  def graph_parents(self):
    """Returns this `Bijector`'s graph_parents as a Python list."""
    return self._graph_parents

  @property
  def is_constant_jacobian(self):
    """Returns true iff the Jacobian is not a function of x.

    Note: Jacobian is either constant for both forward and inverse or neither.

    Returns:
      is_constant_jacobian: Python `bool`.
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

  def _forward_event_shape_tensor(self, input_shape):
    """Subclass implementation for `forward_event_shape_tensor` function."""
    # By default, we assume event_shape is unchanged.
    return input_shape

  def forward_event_shape_tensor(self,
                                 input_shape,
                                 name="forward_event_shape_tensor"):
    """Shape of a single sample from a single batch as an `int32` 1D `Tensor`.

    Args:
      input_shape: `Tensor`, `int32` vector indicating event-portion shape
        passed into `forward` function.
      name: name to give to the op

    Returns:
      forward_event_shape_tensor: `Tensor`, `int32` vector indicating
        event-portion shape after applying `forward`.
    """
    with self._name_scope(name, [input_shape]):
      input_shape = ops.convert_to_tensor(input_shape, dtype=dtypes.int32,
                                          name="input_shape")
      return self._forward_event_shape_tensor(input_shape)

  def _forward_event_shape(self, input_shape):
    """Subclass implementation for `forward_event_shape` public function."""
    # By default, we assume event_shape is unchanged.
    return input_shape

  def forward_event_shape(self, input_shape):
    """Shape of a single sample from a single batch as a `TensorShape`.

    Same meaning as `forward_event_shape_tensor`. May be only partially defined.

    Args:
      input_shape: `TensorShape` indicating event-portion shape passed into
        `forward` function.

    Returns:
      forward_event_shape_tensor: `TensorShape` indicating event-portion shape
        after applying `forward`. Possibly unknown.
    """
    return self._forward_event_shape(tensor_shape.TensorShape(input_shape))

  def _inverse_event_shape_tensor(self, output_shape):
    """Subclass implementation for `inverse_event_shape_tensor` function."""
    # By default, we assume event_shape is unchanged.
    return output_shape

  def inverse_event_shape_tensor(self,
                                 output_shape,
                                 name="inverse_event_shape_tensor"):
    """Shape of a single sample from a single batch as an `int32` 1D `Tensor`.

    Args:
      output_shape: `Tensor`, `int32` vector indicating event-portion shape
        passed into `inverse` function.
      name: name to give to the op

    Returns:
      inverse_event_shape_tensor: `Tensor`, `int32` vector indicating
        event-portion shape after applying `inverse`.
    """
    with self._name_scope(name, [output_shape]):
      output_shape = ops.convert_to_tensor(output_shape, dtype=dtypes.int32,
                                           name="output_shape")
      return self._inverse_event_shape_tensor(output_shape)

  def _inverse_event_shape(self, output_shape):
    """Subclass implementation for `inverse_event_shape` public function."""
    # By default, we assume event_shape is unchanged.
    return tensor_shape.TensorShape(output_shape)

  def inverse_event_shape(self, output_shape):
    """Shape of a single sample from a single batch as a `TensorShape`.

    Same meaning as `inverse_event_shape_tensor`. May be only partially defined.

    Args:
      output_shape: `TensorShape` indicating event-portion shape passed into
        `inverse` function.

    Returns:
      inverse_event_shape_tensor: `TensorShape` indicating event-portion shape
        after applying `inverse`. Possibly unknown.
    """
    return self._inverse_event_shape(output_shape)

  def _forward(self, x):
    """Subclass implementation for `forward` public function."""
    raise NotImplementedError("forward not implemented.")

  def _call_forward(self, x, name, **kwargs):
    with self._name_scope(name, [x]):
      x = ops.convert_to_tensor(x, name="x")
      self._maybe_assert_dtype(x)
      mapping = self._lookup(x=x, kwargs=kwargs)
      if mapping.y is not None:
        return mapping.y
      mapping = mapping.merge(y=self._forward(x, **kwargs))
      self._cache(mapping)
      return mapping.y

  def forward(self, x, name="forward"):
    """Returns the forward `Bijector` evaluation, i.e., X = g(Y).

    Args:
      x: `Tensor`. The input to the "forward" evaluation.
      name: The name to give this op.

    Returns:
      `Tensor`.

    Raises:
      TypeError: if `self.dtype` is specified and `x.dtype` is not
        `self.dtype`.
      NotImplementedError: if `_forward` is not implemented.
    """
    return self._call_forward(x, name)

  def _inverse(self, y):
    """Subclass implementation for `inverse` public function."""
    raise NotImplementedError("inverse not implemented")

  def _call_inverse(self, y, name, **kwargs):
    with self._name_scope(name, [y]):
      y = ops.convert_to_tensor(y, name="y")
      self._maybe_assert_dtype(y)
      mapping = self._lookup(y=y, kwargs=kwargs)
      if mapping.x is not None:
        return mapping.x
      ildj = None
      try:
        x = self._inverse(y, **kwargs)
      except NotImplementedError as original_error:
        # Since _inverse was not implemented, try to see if it's implemented
        # by the _inverse_and_inverse_log_det_jacobian member.
        try:
          x, ildj = self._inverse_and_inverse_log_det_jacobian(y, **kwargs)
        except NotImplementedError:
          raise original_error
        if self._constant_ildj is not None:
          ildj = self._constant_ildj  # Use the "global" result.
        elif self.is_constant_jacobian:
          self._constant_ildj = ildj
      x = x if mapping.x is None else mapping.x
      mapping = mapping.merge(x=x, ildj=ildj)
      self._cache(mapping)
      return mapping.x

  def inverse(self, y, name="inverse"):
    """Returns the inverse `Bijector` evaluation, i.e., X = g^{-1}(Y).

    Args:
      y: `Tensor`. The input to the "inverse" evaluation.
      name: The name to give this op.

    Returns:
      `Tensor`.

    Raises:
      TypeError: if `self.dtype` is specified and `y.dtype` is not
        `self.dtype`.
      NotImplementedError: if neither `_inverse` nor
        `_inverse_and_inverse_log_det_jacobian` are implemented.
    """
    return self._call_inverse(y, name)

  def _inverse_log_det_jacobian(self, y):
    """Subclass implementation of `inverse_log_det_jacobian` public function."""
    raise NotImplementedError("inverse_log_det_jacobian not implemented.")

  def _call_inverse_log_det_jacobian(self, y, name, **kwargs):
    with self._name_scope(name, [y]):
      if self._constant_ildj is not None:
        return self._constant_ildj
      y = ops.convert_to_tensor(y, name="y")
      self._maybe_assert_dtype(y)
      mapping = self._lookup(y=y, kwargs=kwargs)
      if mapping.ildj is not None:
        return mapping.ildj
      try:
        x = mapping.x
        ildj = self._inverse_log_det_jacobian(y, **kwargs)
      except NotImplementedError as original_error:
        # Since _inverse_log_det_jacobian was not implemented, try to see if
        # it's implemented by the _inverse_and_inverse_log_det_jacobian member.
        try:
          x, ildj = self._inverse_and_inverse_log_det_jacobian(y, **kwargs)
        except NotImplementedError:
          raise original_error
        if mapping.x is not None:
          x = mapping.x
      if self.is_constant_jacobian:
        self._constant_ildj = ildj
      x = x if mapping.x is None else mapping.x
      mapping = mapping.merge(x=x, ildj=ildj)
      self._cache(mapping)
      return mapping.ildj

  def inverse_log_det_jacobian(self, y, name="inverse_log_det_jacobian"):
    """Returns the (log o det o Jacobian o inverse)(y).

    Mathematically, returns: `log(det(dX/dY))(Y)`. (Recall that: `X=g^{-1}(Y)`.)

    Note that `forward_log_det_jacobian` is the negative of this function.

    Args:
      y: `Tensor`. The input to the "inverse" Jacobian evaluation.
      name: The name to give this op.

    Returns:
      `Tensor`.

    Raises:
      TypeError: if `self.dtype` is specified and `y.dtype` is not
        `self.dtype`.
      NotImplementedError: if neither `_inverse_log_det_jacobian` nor
        `_inverse_and_inverse_log_det_jacobian` are implemented.
    """
    return self._call_inverse_log_det_jacobian(y, name)

  def _inverse_and_inverse_log_det_jacobian(self, y):
    """Subclass implementation of `inverse_and_inverse_log_det_jacobian`."""
    raise NotImplementedError(
        "inverse_and_inverse_log_det_jacobian not implemented.")

  def _call_inverse_and_inverse_log_det_jacobian(self, y, name, **kwargs):
    with self._name_scope(name, [y]):
      y = ops.convert_to_tensor(y, name="y")
      self._maybe_assert_dtype(y)
      mapping = self._lookup(y=y, kwargs=kwargs)
      if mapping.x is not None and mapping.ildj is not None:
        return mapping.x, mapping.ildj
      try:
        x, ildj = self._inverse_and_inverse_log_det_jacobian(y, **kwargs)
      except NotImplementedError as original_error:
        # Since _inverse_and_inverse_log_det_jacobian was not implemented, try
        # to see if we can separately use _inverse and
        # _inverse_log_det_jacobian members.
        try:
          # We want this same try/except to catch either NotImplementedError.
          x = self._inverse(y, **kwargs)
          if self._constant_ildj is None:
            ildj = self._inverse_log_det_jacobian(y, **kwargs)
        except NotImplementedError:
          raise original_error
      if self._constant_ildj is not None:
        ildj = self._constant_ildj  # Ignore any ildj we may/not have.
      elif self.is_constant_jacobian:
        self._constant_ildj = ildj
      # We use the mapped version of x, even if we re-computed x above with a
      # call to self._inverse_and_inverse_log_det_jacobian. This prevents
      # re-evaluation of the inverse in a common case.
      x = x if mapping.x is None else mapping.x
      mapping = mapping.merge(x=x, ildj=ildj)
      self._cache(mapping)
      return mapping.x, mapping.ildj

  def inverse_and_inverse_log_det_jacobian(
      self, y, name="inverse_and_inverse_log_det_jacobian"):
    """Returns both the inverse evaluation and inverse_log_det_jacobian.

    Enables possibly more efficient calculation when both inverse and
    corresponding Jacobian are needed.

    See `inverse()`, `inverse_log_det_jacobian()` for more details.

    Args:
      y: `Tensor`. The input to the "inverse" Jacobian evaluation.
      name: The name to give this op.

    Returns:
      `Tensor`.

    Raises:
      TypeError: if `self.dtype` is specified and `y.dtype` is not
        `self.dtype`.
      NotImplementedError: if neither `_inverse_and_inverse_log_det_jacobian`
        nor {`_inverse`, `_inverse_log_det_jacobian`} are implemented.
    """
    return self._call_inverse_and_inverse_log_det_jacobian(y, name)

  def _forward_log_det_jacobian(self, x):
    """Subclass implementation of `forward_log_det_jacobian`."""
    raise NotImplementedError(
        "forward_log_det_jacobian not implemented.")

  def _call_forward_log_det_jacobian(self, x, name, **kwargs):
    with self._name_scope(name, [x]):
      if self._constant_ildj is not None:
        # Need "-1. *" to avoid invalid-unary-operand-type linter warning.
        return -1. * self._constant_ildj
      x = ops.convert_to_tensor(x, name="x")
      self._maybe_assert_dtype(x)
      mapping = self._lookup(x=x, kwargs=kwargs)
      if mapping.ildj is not None:
        return -mapping.ildj
      y = None
      try:
        ildj = -self._forward_log_det_jacobian(x, **kwargs)
      except NotImplementedError as original_error:
        try:
          # We want this same try/except to catch either NotImplementedError.
          # TODO(langmore) Add test that covers this branch.
          y = self.forward(x, **kwargs) if y is None else y
          ildj = self.inverse_log_det_jacobian(y, **kwargs)
        except NotImplementedError:
          raise original_error
      if self.is_constant_jacobian:
        self._constant_ildj = ildj
      y = y if mapping.y is None else mapping.y
      mapping = mapping.merge(y=y, ildj=ildj)
      self._cache(mapping)
      return -mapping.ildj

  def forward_log_det_jacobian(self, x, name="forward_log_det_jacobian"):
    """Returns both the forward_log_det_jacobian.

    Args:
      x: `Tensor`. The input to the "forward" Jacobian evaluation.
      name: The name to give this op.

    Returns:
      `Tensor`.

    Raises:
      TypeError: if `self.dtype` is specified and `y.dtype` is not
        `self.dtype`.
      NotImplementedError: if neither `_forward_log_det_jacobian`
        nor {`_inverse`, `_inverse_log_det_jacobian`} are implemented.
    """
    return self._call_forward_log_det_jacobian(x, name)

  @contextlib.contextmanager
  def _name_scope(self, name=None, values=None):
    """Helper function to standardize op scope."""
    with ops.name_scope(self.name):
      with ops.name_scope(
          name, values=(values or []) + self.graph_parents) as scope:
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
        mapping.x, mapping.y, mapping.kwargs))
    if mapping.x is None and mapping.y is None:
      raise ValueError("Caching expects at least one of (x,y) to be known, "
                       "i.e., not None.")
    self._from_x[mapping.x_key] = mapping
    self._from_y[mapping.y_key] = mapping

  def _lookup(self, x=None, y=None, kwargs=None):
    """Helper which retrieves mapping info from forward/inverse dicts."""
    mapping = _Mapping(x=x, y=y, kwargs=kwargs)
    # Since _cache requires both x,y to be set, we only need to do one cache
    # lookup since the mapping is always in both or neither.
    if mapping.x is not None:
      return self._from_x.get(mapping.x_key, mapping)
    if mapping.y is not None:
      return self._from_y.get(mapping.y_key, mapping)
    return mapping

  def _event_dims_tensor(self, sample):
    """Return a 1D `int32` tensor: `range(rank(sample))[-event_ndims:]`."""
    if self.event_ndims is None:
      raise ValueError("Jacobian cannot be computed with unknown event_ndims")
    static_event_ndims = tensor_util.constant_value(self.event_ndims)
    static_rank = sample.get_shape().ndims
    if static_event_ndims is not None and static_rank is not None:
      return ops.convert_to_tensor(
          static_rank + np.arange(-static_event_ndims, 0).astype(np.int32))

    if static_event_ndims is not None:
      event_range = np.arange(-static_event_ndims, 0).astype(np.int32)
    else:
      event_range = math_ops.range(-self.event_ndims, 0, dtype=dtypes.int32)

    if static_rank is not None:
      return event_range + static_rank
    else:
      return event_range + array_ops.rank(sample)
