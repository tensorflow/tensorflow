# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""The Deterministic distribution class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

import six

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.distributions import distribution
from tensorflow.python.util import deprecation

__all__ = [
    "Deterministic",
    "VectorDeterministic",
]


@six.add_metaclass(abc.ABCMeta)
class _BaseDeterministic(distribution.Distribution):
  """Base class for Deterministic distributions."""

  @deprecation.deprecated(
      "2018-10-01",
      "The TensorFlow Distributions library has moved to "
      "TensorFlow Probability "
      "(https://github.com/tensorflow/probability). You "
      "should update all references to use `tfp.distributions` "
      "instead of `tf.contrib.distributions`.",
      warn_once=True)
  def __init__(self,
               loc,
               atol=None,
               rtol=None,
               is_vector=False,
               validate_args=False,
               allow_nan_stats=True,
               name="_BaseDeterministic"):
    """Initialize a batch of `_BaseDeterministic` distributions.

    The `atol` and `rtol` parameters allow for some slack in `pmf`, `cdf`
    computations, e.g. due to floating-point error.

    ```
    pmf(x; loc)
      = 1, if Abs(x - loc) <= atol + rtol * Abs(loc),
      = 0, otherwise.
    ```

    Args:
      loc: Numeric `Tensor`.  The point (or batch of points) on which this
        distribution is supported.
      atol:  Non-negative `Tensor` of same `dtype` as `loc` and broadcastable
        shape.  The absolute tolerance for comparing closeness to `loc`.
        Default is `0`.
      rtol:  Non-negative `Tensor` of same `dtype` as `loc` and broadcastable
        shape.  The relative tolerance for comparing closeness to `loc`.
        Default is `0`.
      is_vector:  Python `bool`.  If `True`, this is for `VectorDeterministic`,
        else `Deterministic`.
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
      allow_nan_stats: Python `bool`, default `True`. When `True`, statistics
        (e.g., mean, mode, variance) use the value "`NaN`" to indicate the
        result is undefined. When `False`, an exception is raised if one or
        more of the statistic's batch members are undefined.
      name: Python `str` name prefixed to Ops created by this class.

    Raises:
      ValueError:  If `loc` is a scalar.
    """
    parameters = dict(locals())
    with ops.name_scope(name, values=[loc, atol, rtol]) as name:
      loc = ops.convert_to_tensor(loc, name="loc")
      if is_vector and validate_args:
        msg = "Argument loc must be at least rank 1."
        if loc.get_shape().ndims is not None:
          if loc.get_shape().ndims < 1:
            raise ValueError(msg)
        else:
          loc = control_flow_ops.with_dependencies(
              [check_ops.assert_rank_at_least(loc, 1, message=msg)], loc)
      self._loc = loc

      super(_BaseDeterministic, self).__init__(
          dtype=self._loc.dtype,
          reparameterization_type=distribution.NOT_REPARAMETERIZED,
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          parameters=parameters,
          graph_parents=[self._loc],
          name=name)

      self._atol = self._get_tol(atol)
      self._rtol = self._get_tol(rtol)
      # Avoid using the large broadcast with self.loc if possible.
      if rtol is None:
        self._slack = self.atol
      else:
        self._slack = self.atol + self.rtol * math_ops.abs(self.loc)

  def _get_tol(self, tol):
    if tol is None:
      return ops.convert_to_tensor(0, dtype=self.loc.dtype)

    tol = ops.convert_to_tensor(tol, dtype=self.loc.dtype)
    if self.validate_args:
      tol = control_flow_ops.with_dependencies([
          check_ops.assert_non_negative(
              tol, message="Argument 'tol' must be non-negative")
      ], tol)
    return tol

  @property
  def loc(self):
    """Point (or batch of points) at which this distribution is supported."""
    return self._loc

  @property
  def atol(self):
    """Absolute tolerance for comparing points to `self.loc`."""
    return self._atol

  @property
  def rtol(self):
    """Relative tolerance for comparing points to `self.loc`."""
    return self._rtol

  def _mean(self):
    return array_ops.identity(self.loc)

  def _variance(self):
    return array_ops.zeros_like(self.loc)

  def _mode(self):
    return self.mean()

  def _sample_n(self, n, seed=None):  # pylint: disable=unused-arg
    n_static = tensor_util.constant_value(ops.convert_to_tensor(n))
    if n_static is not None and self.loc.get_shape().ndims is not None:
      ones = [1] * self.loc.get_shape().ndims
      multiples = [n_static] + ones
    else:
      ones = array_ops.ones_like(array_ops.shape(self.loc))
      multiples = array_ops.concat(([n], ones), axis=0)

    return array_ops.tile(self.loc[array_ops.newaxis, ...], multiples=multiples)


class Deterministic(_BaseDeterministic):
  """Scalar `Deterministic` distribution on the real line.

  The scalar `Deterministic` distribution is parameterized by a [batch] point
  `loc` on the real line.  The distribution is supported at this point only,
  and corresponds to a random variable that is constant, equal to `loc`.

  See [Degenerate rv](https://en.wikipedia.org/wiki/Degenerate_distribution).

  #### Mathematical Details

  The probability mass function (pmf) and cumulative distribution function (cdf)
  are

  ```none
  pmf(x; loc) = 1, if x == loc, else 0
  cdf(x; loc) = 1, if x >= loc, else 0
  ```

  #### Examples

  ```python
  # Initialize a single Deterministic supported at zero.
  constant = tf.contrib.distributions.Deterministic(0.)
  constant.prob(0.)
  ==> 1.
  constant.prob(2.)
  ==> 0.

  # Initialize a [2, 2] batch of scalar constants.
  loc = [[0., 1.], [2., 3.]]
  x = [[0., 1.1], [1.99, 3.]]
  constant = tf.contrib.distributions.Deterministic(loc)
  constant.prob(x)
  ==> [[1., 0.], [0., 1.]]
  ```

  """

  @deprecation.deprecated(
      "2018-10-01",
      "The TensorFlow Distributions library has moved to "
      "TensorFlow Probability "
      "(https://github.com/tensorflow/probability). You "
      "should update all references to use `tfp.distributions` "
      "instead of `tf.contrib.distributions`.",
      warn_once=True)
  def __init__(self,
               loc,
               atol=None,
               rtol=None,
               validate_args=False,
               allow_nan_stats=True,
               name="Deterministic"):
    """Initialize a scalar `Deterministic` distribution.

    The `atol` and `rtol` parameters allow for some slack in `pmf`, `cdf`
    computations, e.g. due to floating-point error.

    ```
    pmf(x; loc)
      = 1, if Abs(x - loc) <= atol + rtol * Abs(loc),
      = 0, otherwise.
    ```

    Args:
      loc: Numeric `Tensor` of shape `[B1, ..., Bb]`, with `b >= 0`.
        The point (or batch of points) on which this distribution is supported.
      atol:  Non-negative `Tensor` of same `dtype` as `loc` and broadcastable
        shape.  The absolute tolerance for comparing closeness to `loc`.
        Default is `0`.
      rtol:  Non-negative `Tensor` of same `dtype` as `loc` and broadcastable
        shape.  The relative tolerance for comparing closeness to `loc`.
        Default is `0`.
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
      allow_nan_stats: Python `bool`, default `True`. When `True`, statistics
        (e.g., mean, mode, variance) use the value "`NaN`" to indicate the
        result is undefined. When `False`, an exception is raised if one or
        more of the statistic's batch members are undefined.
      name: Python `str` name prefixed to Ops created by this class.
    """
    super(Deterministic, self).__init__(
        loc,
        atol=atol,
        rtol=rtol,
        validate_args=validate_args,
        allow_nan_stats=allow_nan_stats,
        name=name)

  def _batch_shape_tensor(self):
    return array_ops.shape(self.loc)

  def _batch_shape(self):
    return self.loc.get_shape()

  def _event_shape_tensor(self):
    return constant_op.constant([], dtype=dtypes.int32)

  def _event_shape(self):
    return tensor_shape.scalar()

  def _prob(self, x):
    return math_ops.cast(
        math_ops.abs(x - self.loc) <= self._slack, dtype=self.dtype)

  def _cdf(self, x):
    return math_ops.cast(x >= self.loc - self._slack, dtype=self.dtype)


class VectorDeterministic(_BaseDeterministic):
  """Vector `Deterministic` distribution on `R^k`.

  The `VectorDeterministic` distribution is parameterized by a [batch] point
  `loc in R^k`.  The distribution is supported at this point only,
  and corresponds to a random variable that is constant, equal to `loc`.

  See [Degenerate rv](https://en.wikipedia.org/wiki/Degenerate_distribution).

  #### Mathematical Details

  The probability mass function (pmf) is

  ```none
  pmf(x; loc)
    = 1, if All[Abs(x - loc) <= atol + rtol * Abs(loc)],
    = 0, otherwise.
  ```

  #### Examples

  ```python
  tfd = tf.contrib.distributions

  # Initialize a single VectorDeterministic supported at [0., 2.] in R^2.
  constant = tfd.Deterministic([0., 2.])
  constant.prob([0., 2.])
  ==> 1.
  constant.prob([0., 3.])
  ==> 0.

  # Initialize a [3] batch of constants on R^2.
  loc = [[0., 1.], [2., 3.], [4., 5.]]
  constant = tfd.VectorDeterministic(loc)
  constant.prob([[0., 1.], [1.9, 3.], [3.99, 5.]])
  ==> [1., 0., 0.]
  ```

  """

  @deprecation.deprecated(
      "2018-10-01",
      "The TensorFlow Distributions library has moved to "
      "TensorFlow Probability "
      "(https://github.com/tensorflow/probability). You "
      "should update all references to use `tfp.distributions` "
      "instead of `tf.contrib.distributions`.",
      warn_once=True)
  def __init__(self,
               loc,
               atol=None,
               rtol=None,
               validate_args=False,
               allow_nan_stats=True,
               name="VectorDeterministic"):
    """Initialize a `VectorDeterministic` distribution on `R^k`, for `k >= 0`.

    Note that there is only one point in `R^0`, the "point" `[]`.  So if `k = 0`
    then `self.prob([]) == 1`.

    The `atol` and `rtol` parameters allow for some slack in `pmf`
    computations, e.g. due to floating-point error.

    ```
    pmf(x; loc)
      = 1, if All[Abs(x - loc) <= atol + rtol * Abs(loc)],
      = 0, otherwise
    ```

    Args:
      loc: Numeric `Tensor` of shape `[B1, ..., Bb, k]`, with `b >= 0`, `k >= 0`
        The point (or batch of points) on which this distribution is supported.
      atol:  Non-negative `Tensor` of same `dtype` as `loc` and broadcastable
        shape.  The absolute tolerance for comparing closeness to `loc`.
        Default is `0`.
      rtol:  Non-negative `Tensor` of same `dtype` as `loc` and broadcastable
        shape.  The relative tolerance for comparing closeness to `loc`.
        Default is `0`.
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
      allow_nan_stats: Python `bool`, default `True`. When `True`, statistics
        (e.g., mean, mode, variance) use the value "`NaN`" to indicate the
        result is undefined. When `False`, an exception is raised if one or
        more of the statistic's batch members are undefined.
      name: Python `str` name prefixed to Ops created by this class.
    """
    super(VectorDeterministic, self).__init__(
        loc,
        atol=atol,
        rtol=rtol,
        is_vector=True,
        validate_args=validate_args,
        allow_nan_stats=allow_nan_stats,
        name=name)

  def _batch_shape_tensor(self):
    return array_ops.shape(self.loc)[:-1]

  def _batch_shape(self):
    return self.loc.get_shape()[:-1]

  def _event_shape_tensor(self):
    return array_ops.shape(self.loc)[-1]

  def _event_shape(self):
    return self.loc.get_shape()[-1:]

  def _prob(self, x):
    if self.validate_args:
      is_vector_check = check_ops.assert_rank_at_least(x, 1)
      right_vec_space_check = check_ops.assert_equal(
          self.event_shape_tensor(),
          array_ops.gather(array_ops.shape(x), array_ops.rank(x) - 1),
          message=
          "Argument 'x' not defined in the same space R^k as this distribution")
      with ops.control_dependencies([is_vector_check]):
        with ops.control_dependencies([right_vec_space_check]):
          x = array_ops.identity(x)
    return math_ops.cast(
        math_ops.reduce_all(math_ops.abs(x - self.loc) <= self._slack, axis=-1),
        dtype=self.dtype)
