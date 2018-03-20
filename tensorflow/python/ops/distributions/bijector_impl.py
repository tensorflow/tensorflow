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
from tensorflow.python.util.tf_export import tf_export


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
@tf_export("distributions.bijectors.Bijector")
class Bijector(object):
  r"""Interface for transformations of a `Distribution` sample.

  Bijectors can be used to represent any differentiable and injective
  (one to one) function defined on an open subset of `R^n`.  Some non-injective
  transformations are also supported (see "Non Injective Transforms" below).

  #### Mathematical Details

  A `Bijector` implements a [smooth covering map](
  https://en.wikipedia.org/wiki/Local_diffeomorphism), i.e., a local
  diffeomorphism such that every point in the target has a neighborhood evenly
  covered by a map ([see also](
  https://en.wikipedia.org/wiki/Covering_space#Covering_of_a_manifold)).
  A `Bijector` is used by `TransformedDistribution` but can be generally used
  for transforming a `Distribution` generated `Tensor`. A `Bijector` is
  characterized by three operations:

  1. Forward\
     Useful for turning one random outcome into another random outcome from a
     different distribution.
  2. Inverse\
     Useful for "reversing" a transformation to compute one probability in
     terms of another.
  3. `(log o det o Jacobian o inverse)(x)`\
     "The log of the determinant of the matrix of all first-order partial
     derivatives of the inverse function."\
     Useful for inverting a transformation to compute one probability in terms
     of another. Geometrically, the det(Jacobian) is the volume of the
     transformation and is used to scale the probability.

  By convention, transformations of random variables are named in terms of the
  forward transformation. The forward transformation creates samples, the
  inverse is useful for computing probabilities.

  #### Example Uses

  - Basic properties:

  ```python
  x = ...  # A tensor.
  # Evaluate forward transformation.
  fwd_x = my_bijector.forward(x)
  x == my_bijector.inverse(fwd_x)
  x != my_bijector.forward(fwd_x)  # Not equal because x != g(g(x)).
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

  #### Example Bijectors

  - "Exponential"

    ```none
    Y = g(X) = exp(X)
    X ~ Normal(0, 1)  # Univariate.
    ```

    Implies:

    ```none
      g^{-1}(Y) = log(Y)
      |Jacobian(g^{-1})(y)| = 1 / y
      Y ~ LogNormal(0, 1), i.e.,
      prob(Y=y) = |Jacobian(g^{-1})(y)| * prob(X=g^{-1}(y))
                = (1 / y) Normal(log(y); 0, 1)
    ```

    Here is an example of how one might implement the `Exp` bijector:

    ```python
      class Exp(Bijector):

        def __init__(self, event_ndims=0, validate_args=False, name="exp"):
          super(Exp, self).__init__(
              event_ndims=event_ndims, validate_args=validate_args, name=name)

        def _forward(self, x):
          return math_ops.exp(x)

        def _inverse(self, y):
          return math_ops.log(y)

        def _inverse_log_det_jacobian(self, y):
          return -self._forward_log_det_jacobian(self._inverse(y))

        def _forward_log_det_jacobian(self, x):
          if self.event_ndims is None:
            raise ValueError("Jacobian requires known event_ndims.")
          event_dims = array_ops.shape(x)[-self.event_ndims:]
          return math_ops.reduce_sum(x, axis=event_dims)
      ```

  - "Affine"

    ```none
    Y = g(X) = sqrtSigma * X + mu
    X ~ MultivariateNormal(0, I_d)
    ```

    Implies:

    ```none
      g^{-1}(Y) = inv(sqrtSigma) * (Y - mu)
      |Jacobian(g^{-1})(y)| = det(inv(sqrtSigma))
      Y ~ MultivariateNormal(mu, sqrtSigma) , i.e.,
      prob(Y=y) = |Jacobian(g^{-1})(y)| * prob(X=g^{-1}(y))
                = det(sqrtSigma)^(-d) *
                  MultivariateNormal(inv(sqrtSigma) * (y - mu); 0, I_d)
      ```

  #### Jacobian

  The Jacobian is a reduction over event dims. To see this, consider the `Exp`
  `Bijector` applied to a `Tensor` which has sample, batch, and event (S, B, E)
  shape semantics. Suppose the `Tensor`'s partitioned-shape is `(S=[4], B=[2],
  E=[3, 3])`. The shape of the `Tensor` returned by `forward` and `inverse` is
  unchanged, i.e., `[4, 2, 3, 3]`.  However the shape returned by
  `inverse_log_det_jacobian` is `[4, 2]` because the Jacobian is a reduction
  over the event dimensions.

  It is sometimes useful to implement the inverse Jacobian as the negative
  forward Jacobian. For example,

  ```python
  def _inverse_log_det_jacobian(self, y):
     return -self._forward_log_det_jac(self._inverse(y))  # Note negation.
  ```

  The correctness of this approach can be seen from the following claim.

  - Claim:

      Assume `Y = g(X)` is a bijection whose derivative exists and is nonzero
      for its domain, i.e., `dY/dX = d/dX g(X) != 0`. Then:

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
      The same theorem also implies `g^{-1}'` is non-singular therefore:
      `inv[ g'(g^{-1}(y)) ] = g^{-1}'(y)`.
      The claim follows from [properties of determinant](
  https://en.wikipedia.org/wiki/Determinant#Multiplicativity_and_matrix_groups).

  Generally its preferable to directly implement the inverse Jacobian. This
  should have superior numerical stability and will often share subgraphs with
  the `_inverse` implementation.

  #### Subclass Requirements

  - Subclasses typically implement:

      - `_forward`,
      - `_inverse`,
      - `_inverse_log_det_jacobian`,
      - `_forward_log_det_jacobian` (optional).

    The `_forward_log_det_jacobian` is called when the bijector is inverted via
    the `Invert` bijector. If undefined, a slightly less efficiently
    calculation, `-1 * _inverse_log_det_jacobian`, is used.

    If the bijector changes the shape of the input, you must also implement:

      - _forward_event_shape_tensor,
      - _forward_event_shape (optional),
      - _inverse_event_shape_tensor,
      - _inverse_event_shape (optional).

    By default the event-shape is assumed unchanged from input.

  - If the `Bijector`'s use is limited to `TransformedDistribution` (or friends
    like `QuantizedDistribution`) then depending on your use, you may not need
    to implement all of `_forward` and `_inverse` functions.

    Examples:

      1. Sampling (e.g., `sample`) only requires `_forward`.
      2. Probability functions (e.g., `prob`, `cdf`, `survival`) only require
         `_inverse` (and related).
      3. Only calling probability functions on the output of `sample` means
        `_inverse` can be implemented as a cache lookup.

    See "Example Uses" [above] which shows how these functions are used to
    transform a distribution. (Note: `_forward` could theoretically be
    implemented as a cache lookup but this would require controlling the
    underlying sample generation mechanism.)

  #### Non Injective Transforms

  **WARNING** Handing of non-injective transforms is subject to change.

  Non injective maps `g` are supported, provided their domain `D` can be
  partitioned into `k` disjoint subsets, `Union{D1, ..., Dk}`, such that,
  ignoring sets of measure zero, the restriction of `g` to each subset is a
  differentiable bijection onto `g(D)`.  In particular, this imples that for
  `y in g(D)`, the set inverse, i.e. `g^{-1}(y) = {x in D : g(x) = y}`, always
  contains exactly `k` distinct points.

  The property, `_is_injective` is set to `False` to indicate that the bijector
  is not injective, yet satisfies the above condition.

  The usual bijector API is modified in the case `_is_injective is False` (see
  method docstrings for specifics).  Here we show by example the `AbsoluteValue`
  bijector.  In this case, the domain `D = (-inf, inf)`, can be partitioned
  into `D1 = (-inf, 0)`, `D2 = {0}`, and `D3 = (0, inf)`.  Let `gi` be the
  restriction of `g` to `Di`, then both `g1` and `g3` are bijections onto
  `(0, inf)`, with `g1^{-1}(y) = -y`, and `g3^{-1}(y) = y`.  We will use
  `g1` and `g3` to define bijector methods over `D1` and `D3`.  `D2 = {0}` is
  an oddball in that `g2` is one to one, and the derivative is not well defined.
  Fortunately, when considering transformations of probability densities
  (e.g. in `TransformedDistribution`), sets of measure zero have no effect in
  theory, and only a small effect in 32 or 64 bit precision.  For that reason,
  we define `inverse(0)` and `inverse_log_det_jacobian(0)` both as `[0, 0]`,
  which is convenient and results in a left-semicontinuous pdf.


  ```python
  abs = tf.contrib.distributions.bijectors.AbsoluteValue()

  abs.forward(-1.)
  ==> 1.

  abs.forward(1.)
  ==> 1.

  abs.inverse(1.)
  ==> (-1., 1.)

  # The |dX/dY| is constant, == 1.  So Log|dX/dY| == 0.
  abs.inverse_log_det_jacobian(1.)
  ==> (0., 0.)

  # Special case handling of 0.
  abs.inverse(0.)
  ==> (0., 0.)

  abs.inverse_log_det_jacobian(0.)
  ==> (0., 0.)
  ```

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

    Raises:
      ValueError:  If a member of `graph_parents` is not a `Tensor`.
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

    for i, t in enumerate(self._graph_parents):
      if t is None or not tensor_util.is_tensor(t):
        raise ValueError("Graph parent item %d is not a Tensor; %s." % (i, t))

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
  def _is_injective(self):
    """Returns true iff the forward map `g` is injective (one-to-one function).

    **WARNING** This hidden property and its behavior are subject to change.

    Note:  Non-injective maps `g` are supported, provided their domain `D` can
    be partitioned into `k` disjoint subsets, `Union{D1, ..., Dk}`, such that,
    ignoring sets of measure zero, the restriction of `g` to each subset is a
    differentiable bijection onto `g(D)`.

    Returns:
      is_injective: Python `bool`.
    """
    return True

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
      if not self._is_injective:  # No caching for non-injective
        return self._forward(x, **kwargs)
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
      if not self._is_injective:  # No caching for non-injective
        return self._inverse(y, **kwargs)
      mapping = self._lookup(y=y, kwargs=kwargs)
      if mapping.x is not None:
        return mapping.x
      mapping = mapping.merge(x=self._inverse(y, **kwargs))
      self._cache(mapping)
      return mapping.x

  def inverse(self, y, name="inverse"):
    """Returns the inverse `Bijector` evaluation, i.e., X = g^{-1}(Y).

    Args:
      y: `Tensor`. The input to the "inverse" evaluation.
      name: The name to give this op.

    Returns:
      `Tensor`, if this bijector is injective.
        If not injective, returns the k-tuple containing the unique
        `k` points `(x1, ..., xk)` such that `g(xi) = y`.

    Raises:
      TypeError: if `self.dtype` is specified and `y.dtype` is not
        `self.dtype`.
      NotImplementedError: if `_inverse` is not implemented.
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
      if not self._is_injective:  # No caching for non-injective
        return self._inverse_log_det_jacobian(y, **kwargs)
      mapping = self._lookup(y=y, kwargs=kwargs)
      if mapping.ildj is not None:
        return mapping.ildj
      try:
        x = None  # Not needed; leave cache as is.
        ildj = self._inverse_log_det_jacobian(y, **kwargs)
      except NotImplementedError as original_exception:
        try:
          x = mapping.x if mapping.x is not None else self._inverse(y, **kwargs)
          ildj = -self._forward_log_det_jacobian(x, **kwargs)
        except NotImplementedError:
          raise original_exception
      mapping = mapping.merge(x=x, ildj=ildj)
      self._cache(mapping)
      if self.is_constant_jacobian:
        self._constant_ildj = mapping.ildj
      return mapping.ildj

  def inverse_log_det_jacobian(self, y, name="inverse_log_det_jacobian"):
    """Returns the (log o det o Jacobian o inverse)(y).

    Mathematically, returns: `log(det(dX/dY))(Y)`. (Recall that: `X=g^{-1}(Y)`.)

    Note that `forward_log_det_jacobian` is the negative of this function,
    evaluated at `g^{-1}(y)`.

    Args:
      y: `Tensor`. The input to the "inverse" Jacobian evaluation.
      name: The name to give this op.

    Returns:
      `Tensor`, if this bijector is injective.
        If not injective, returns the tuple of local log det
        Jacobians, `log(det(Dg_i^{-1}(y)))`, where `g_i` is the restriction
        of `g` to the `ith` partition `Di`.

    Raises:
      TypeError: if `self.dtype` is specified and `y.dtype` is not
        `self.dtype`.
      NotImplementedError: if `_inverse_log_det_jacobian` is not implemented.
    """
    return self._call_inverse_log_det_jacobian(y, name)

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
      if not self._is_injective:
        return self._forward_log_det_jacobian(x, **kwargs)  # No caching.
      mapping = self._lookup(x=x, kwargs=kwargs)
      if mapping.ildj is not None:
        return -mapping.ildj
      try:
        y = None  # Not needed; leave cache as is.
        ildj = -self._forward_log_det_jacobian(x, **kwargs)
      except NotImplementedError as original_exception:
        try:
          y = mapping.y if mapping.y is not None else self._forward(x, **kwargs)
          ildj = self._inverse_log_det_jacobian(y, **kwargs)
        except NotImplementedError:
          raise original_exception
      mapping = mapping.merge(y=y, ildj=ildj)
      self._cache(mapping)
      if self.is_constant_jacobian:
        self._constant_ildj = mapping.ildj
      return -mapping.ildj

  def forward_log_det_jacobian(self, x, name="forward_log_det_jacobian"):
    """Returns both the forward_log_det_jacobian.

    Args:
      x: `Tensor`. The input to the "forward" Jacobian evaluation.
      name: The name to give this op.

    Returns:
      `Tensor`, if this bijector is injective.
        If not injective this is not implemented.

    Raises:
      TypeError: if `self.dtype` is specified and `y.dtype` is not
        `self.dtype`.
      NotImplementedError: if neither `_forward_log_det_jacobian`
        nor {`_inverse`, `_inverse_log_det_jacobian`} are implemented, or
        this is a non-injective bijector.
    """
    if not self._is_injective:
      raise NotImplementedError(
          "forward_log_det_jacobian cannot be implemented for non-injective "
          "transforms.")
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
