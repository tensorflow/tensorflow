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
"""An API for reversible (bijective) transformations of random variables."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib

from tensorflow.contrib.distributions.python.ops.shape import _DistributionShape
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops


class _Bijector(object):
  """An interface for transforming a `Distribution` `Tensor`.

  Recall that a `Distribution` `Tensor` has dimensions which have `sample`,
  `batch`, and `event` semantics. (See `DistributionShape` for more details.)

  A `Bijector` implements a bijective, differentiable function by transforming
  an input `Tensor`. The output `Tensor` shape is constrained by the input
  `sample`, `batch`, and `event` shape.  A `Bijector` is characterized by three
  operations:

  (1) Forward Evaluation
      Useful for turning one random outcome into another random outcome from a
      different distribution.

  (2) Inverse Evaluation
      Useful for "reversing" a transformation to compute one probability in
      terms of another.

  (3) (log o det o Jacobian o inverse)(x)
      "The log of the determinant of the matrix of all first-order partial
      derivatives of the inverse function."
      Useful for inverting a transformation to compute one probability in terms
      of another.  Geometrically, the det(Jacobian) is the volume of the
      transformation and is used to scale the probability.

  By convention, transformations of random variables are named in terms of the
  forward transformation. The forward transformation creates samples, the
  inverse is useful for computing probabilities.

  Example Use:
    Basic properties:

    ```python
    x = ... # A tensor.
    # Evaluate forward transformation.
    fwd_x = my_bijector.forward(x)
    x == my_bijector.inverse(fwd_x)
    x != my_bijector.forward(fwd_x)  # Not equal because g(x) != g(g(x)).
    ```

    Computing a log-likelihood:

    ```python
    def transformed_log_pdf(bijector, log_pdf, x):
      return (bijector.inverse_log_det_jacobian(x) +
              log_pdf(bijector.inverse(x)))
    ```

    Transforming a random outcome:

    ```python
    def transformed_sample(bijector, x):
      return bijector.forward(x)
    ```

  Example transformations:
    "Exponential"

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

    "ShiftAndScale"

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
    Consider the `Exp` `Bijector` applied to a `Tensor` which has sample, batch,
    and event (S, B, E) shape semantics.  Suppose
    the `Tensor`'s partitioned-shape is `(S=[4], B=[2], E=[3, 3])`.

    For `Exp`, the shape of the `Tensor` returned by `forward` and `inverse` is
    unchanged, i.e., `[4, 2, 3, 3]`. However the shape returned by
    `inverse_log_det_jacobian` is `[4, 2]` because the Jacobian is a reduction
    over the event dimensions.

  Subclass Requirements:
    Subclasses are expected to implement `_forward` and one or both of:
      - `_inverse`, `_inverse_log_det_jacobian`,
      - `_inverse_and_inverse_log_det_jacobian`.

    If computation can be shared among `_inverse` and
    `_inverse_log_det_jacobian` it is preferable to implement
    `_inverse_and_inverse_log_det_jacobian`. This usually reduces
    graph-construction overhead because a `Distribution`'s implementation of
    `log_prob` will need to evaluate both the inverse Jacobian as well as the
    inverse function.

    If an additional use case needs just `inverse` or just
    `inverse_log_det_jacobian` then he or she may also wish to implement these
    functions to avoid computing the `inverse_log_det_jacobian` or the
    `inverse`, respectively.
  """

  # TODO(b/30476956): Try to remove constructor dependence on ndims.
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
    self._name = name or type(self).__name__

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
      AttributeError: if `_forward` is not implemented.
    """
    with self._name_scope(name, [x]):
      x = ops.convert_to_tensor(x, name="x")
      self._maybe_assert_dtype(x)
      return self._forward(x)

  def inverse(self, x, name="inverse"):
    """Returns the inverse `Bijector` evaluation, i.e., X = g^{-1}(Y).

    Args:
      x: `Tensor`. The input to the "inverse" evaluation.
      name: The name to give this op.

    Returns:
      `Tensor`.

    Raises:
      TypeError: if `self.dtype` is specified and `x.dtype` is not
        `self.dtype`.
      AttributeError: if neither `_inverse` nor
        `_inverse_and_inverse_log_det_jacobian` are implemented.
    """
    with self._name_scope(name, [x]):
      x = ops.convert_to_tensor(x, name="x")
      self._maybe_assert_dtype(x)
      try:
        return self._inverse(x)
      except AttributeError:
        # Since _inverse was not implemented, try to see if it's implemented
        # by the _inverse_and_inverse_log_det_jacobian member.
        return self._inverse_and_inverse_log_det_jacobian(x)[0]

  def inverse_log_det_jacobian(self, x, name="inverse_log_det_jacobian"):
    """Returns the (log o det o Jacobian o inverse)(x).

    Mathematically, returns: log(det(dY/dX g^{-1}))(Y).

    Note that forward_log_det_jacobian is the negative of this function. (See
    is_constant_jacobian for related proof.)

    Args:
      x: `Tensor`. The input to the "inverse" Jacobian evaluation.
      name: The name to give this op.

    Returns:
      `Tensor`.

    Raises:
      TypeError: if `self.dtype` is specified and `x.dtype` is not
        `self.dtype`.
      AttributeError: if neither `_inverse_log_det_jacobian` nor
        `_inverse_and_inverse_log_det_jacobian` are implemented.
    """
    with self._name_scope(name, [x]):
      x = ops.convert_to_tensor(x, name="x")
      self._maybe_assert_dtype(x)
      try:
        return self._inverse_log_det_jacobian(x)
      except AttributeError:
        # Since _inverse_log_det_jacobian was not implemented, try to see if
        # it's implemented by the _inverse_and_inverse_log_det_jacobian member.
        return self._inverse_and_inverse_log_det_jacobian(x)[1]

  def inverse_and_inverse_log_det_jacobian(
      self, x, name="inverse_and_inverse_log_det_jacobian"):
    """Returns both the inverse evaluation and inverse_log_det_jacobian.

    Enables possibly more efficient calculation when both inverse and
    corresponding Jacobian are needed.

    See `inverse()`, `inverse_log_det_jacobian()` for more details.

    Args:
      x: `Tensor`. The input to the "inverse" Jacobian evaluation.
      name: The name to give this op.

    Returns:
      `Tensor`.

    Raises:
      TypeError: if `self.dtype` is specified and `x.dtype` is not
        `self.dtype`.
      AttributeError: if neither `_inverse_and_inverse_log_det_jacobian` nor
        {`_inverse`, `_inverse_log_det_jacobian`} are implemented.
    """
    with self._name_scope(name, [x]):
      x = ops.convert_to_tensor(x, name="x")
      self._maybe_assert_dtype(x)
      try:
        return self._inverse_and_inverse_log_det_jacobian(x)
      except AttributeError:
        # Since _inverse_and_inverse_log_det_jacobian was not implemented, try
        # to see if we can separately use _inverse and
        # _inverse_log_det_jacobian members.
        return self._inverse(x), self._inverse_log_det_jacobian(x)

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


class _Identity(_Bijector):
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

  def __init__(self, validate_args=False, name="Identity"):
    super(_Identity, self).__init__(
        batch_ndims=0,
        event_ndims=0,
        is_constant_jacobian=True,
        validate_args=validate_args,
        name=name)
    self._is_constant_jacobian = True

  def _forward(self, x):
    return x

  def _inverse(self, x):
    return x

  def _inverse_log_det_jacobian(self, x):
    return constant_op.constant(0., dtype=x.dtype)


class _Exp(_Bijector):
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

  # TODO(b/30476956): Try to remove constructor dependence on ndims.
  def __init__(self,
               event_ndims=0,
               validate_args=False,
               name="Exp"):
    super(_Exp, self).__init__(
        batch_ndims=0,
        event_ndims=event_ndims,
        validate_args=validate_args,
        name=name)

  def _forward(self, x):
    return math_ops.exp(x)

  def _inverse(self, x):
    return math_ops.log(x)

  def _inverse_log_det_jacobian(self, x):
    if self.shaper is None:
      raise ValueError("Jacobian cannot be computed with unknown event_ndims")
    _, _, event_dims = self.shaper.get_dims(x)
    return -math_ops.reduce_sum(math_ops.log(x), reduction_indices=event_dims)

  def _inverse_and_inverse_log_det_jacobian(self, x):
    if self.shaper is None:
      raise ValueError("Jacobian cannot be computed with unknown event_ndims")
    y = math_ops.log(x)
    _, _, event_dims = self.shaper.get_dims(x)
    return y, -math_ops.reduce_sum(y, reduction_indices=event_dims)


class _ShiftAndScale(_Bijector):
  """Bijector which computes Y = g(X; loc, scale) = scale * X + loc.

  Example Use:

  ```python
  # No batch, scalar.
  mu = 0     # shape=[]
  sigma = 1  # shape=[]
  b = ShiftAndScale(loc=mu, scale=sigma)
  # b.shaper.batch_ndims == 0
  # b.shaper.event_ndims == 0

  # One batch, scalar.
  mu = ...    # shape=[b], b>0
  sigma = ... # shape=[b], b>0
  b = ShiftAndScale(loc=mu, scale=sigma)
  # b.shaper.batch_ndims == 1
  # b.shaper.event_ndims == 0

  # No batch, multivariate.
  mu = ...    # shape=[d],    d>0
  sigma = ... # shape=[d, d], d>0
  b = ShiftAndScale(loc=mu, scale=sigma, event_ndims=1)
  # b.shaper.batch_ndims == 0
  # b.shaper.event_ndims == 1

  # (B1*B2*...*Bb)-batch, multivariate.
  mu = ...    # shape=[B1,...,Bb, d],    b>0, d>0
  sigma = ... # shape=[B1,...,Bb, d, d], b>0, d>0
  b = ShiftAndScale(loc=mu, scale=sigma, event_ndims=1)
  # b.shaper.batch_ndims == b
  # b.shaper.event_ndims == 1

  # Mu is broadcast:
  mu = 1
  sigma = [I, I]  # I is a 3x3 identity matrix.
  b = ShiftAndScale(loc=mu, scale=sigma, event_ndims=1)
  x = numpy.ones(S + sigma.shape)
  b.forward(x) # == x + 1
  ```

  """

  def __init__(self,
               loc,
               scale,
               event_ndims=0,
               validate_args=False,
               name="ShiftAndScale"):
    self._parameters = {}
    self._name = name
    with self._name_scope("init", values=[loc, scale, event_ndims]):
      self._loc = ops.convert_to_tensor(loc, name="loc")
      self._scale = ops.convert_to_tensor(scale, name="scale")
      event_ndims = ops.convert_to_tensor(event_ndims, name="event_ndims")
      if self.loc.dtype.base_dtype != self.scale.dtype.base_dtype:
        raise TypeError("%s.dtype=%s does not match %s.dtype=%s" %
                        (self.loc.name, self.loc.dtype, self.scale.name,
                         self.scale.dtype))
      if event_ndims.dtype.base_dtype != dtypes.int32.base_dtype:
        raise TypeError("%s.dtype=%s does not match %s" %
                        (event_ndims.name, event_ndims.dtype, dtypes.int32))
      self._scale, batch_ndims = self._process_scale(self.scale, event_ndims)
      super(_ShiftAndScale, self).__init__(
          batch_ndims=batch_ndims,
          event_ndims=event_ndims,
          parameters={"loc": self.loc, "scale": self.scale},
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
  def loc(self):
    return self._loc

  @property
  def scale(self):
    return self._scale

  def _forward(self, x):
    x, sample_shape = self.shaper.make_batch_of_event_sample_matrices(x)
    x = math_ops.batch_matmul(self.scale, x)
    x = self.shaper.undo_make_batch_of_event_sample_matrices(x, sample_shape)
    x += self.loc
    return x

  def _inverse(self, x):
    x -= self.loc
    x, sample_shape = self.shaper.make_batch_of_event_sample_matrices(x)
    x = linalg_ops.matrix_triangular_solve(self.scale, x)
    x = self.shaper.undo_make_batch_of_event_sample_matrices(x, sample_shape)
    return x

  def _inverse_log_det_jacobian(self, x):  # pylint: disable=unused-argument
    return -math_ops.reduce_sum(
        math_ops.log(array_ops.matrix_diag_part(self.scale)),
        reduction_indices=[-1])
