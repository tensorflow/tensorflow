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
"""A helper class for inferring Distribution shape."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib

from tensorflow.contrib.distributions.python.ops import distribution_util
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops


class _DistributionShape(object):
  """Manage and manipulate `Distribution` shape.

  Terminology:
    Recall that a `Tensor` has:
      - `shape`: size of `Tensor` dimensions,
      - `ndims`: size of `shape`; number of `Tensor` dimensions,
      - `dims`: indexes into `shape`; useful for transpose, reduce.

    `Tensor`s sampled from a `Distribution` can be partitioned by `sample_dims`,
    `batch_dims`, and `event_dims`.  To understand the semantics of these
    dimensions, consider when two of the three are fixed and the remaining
    is varied:
      - `sample_dims`: indexes independent draws from identical
                       parameterizations of the `Distribution`.
      - `batch_dims`:  indexes independent draws from non-identical
                       parameterizations of the `Distribution`.
      - `event_dims`:  indexes event coordinates from one sample.

    The `sample`, `batch`, and `event` dimensions constitute the entirety of a
    `Distribution` `Tensor`'s shape.

    The dimensions are always in `sample`, `batch`, `event` order.

  Purpose:
    This class partitions `Tensor` notions of `shape`, `ndims`, and `dims` into
    `Distribution` notions of `sample,` `batch,` and `event` dimensions. That
    is, it computes any of:

    ```
    sample_shape     batch_shape     event_shape
    sample_dims      batch_dims      event_dims
    sample_ndims     batch_ndims     event_ndims
    ```

    for a given `Tensor`, e.g., the result of
    `Distribution.sample(sample_shape=...)`.

    For a given `Tensor`, this class computes the above table using minimal
    information: `batch_ndims` and `event_ndims`.

  Examples of `Distribution` `shape` semantics:
    - Sample dimensions:
      Computing summary statistics, i.e., the average is a reduction over sample
      dimensions.

      ```python
      sample_dims = [0]
      tf.reduce_mean(Normal(mu=1.3, sigma=1.).sample_n(1000),
                     reduction_indices=sample_dims)  # ~= 1.3
      ```

    - Batch dimensions:
      Monte Carlo estimation of a marginal probability:
      Average over batch dimensions where batch dimensions are associated with
      random draws from a prior.
      E.g., suppose we want to find the Monte Carlo estimate of the marginal
      distribution of a `Normal` with a random `Laplace` location:

      ```
      P(X=x) = integral P(X=x|y) P(Y=y) dy
            ~= 1/n sum_{i=1}^n P(X=x|y_i),   y_i ~iid Laplace(0,1)
             = tf.reduce_mean(Normal(mu=Laplace(0., 1.).sample_n(n=1000),
                                     sigma=tf.ones(1000)).pdf(x),
                              reduction_indices=batch_dims)
      ```

      The `Laplace` distribution generates a `Tensor` of shape `[1000]`. When
      fed to a `Normal`, this is interpreted as 1000 different locations, i.e.,
      1000 non-identical Normals.  Therefore a single call to `pdf(x)` yields
      1000 probabilities, one for every location.  The average over this batch
      yields the marginal.

    - Event dimensions:
      Computing the determinant of the Jacobian of a function of a random
      variable involves a reduction over event dimensions.
      E.g., Jacobian of the transform `Y = g(X) = exp(X)`:

      ```python
      tf.div(1., tf.reduce_prod(x, event_dims))
      ```

  Examples using this class:
    Write `S, B, E` for `sample_shape`, `batch_shape`, and `event_shape`.

    ```python
    # 150 iid samples from one multivariate Normal with two degrees of freedom.
    mu = [0., 0]
    sigma = [[1., 0],
             [0,  1]]
    mvn = MultivariateNormal(mu, sigma)
    rand_mvn = mvn.sample(sample_shape=[3, 50])
    shaper = DistributionShape(batch_ndims=0, event_ndims=1)
    S, B, E = shaper.get_shape(rand_mvn)
    # S = [3, 50]
    # B = []
    # E = [2]

    # 12 iid samples from one Wishart with 2x2 events.
    sigma = [[1., 0],
             [2,  1]]
    wishart = Wishart(df=5, scale=sigma)
    rand_wishart = wishart.sample(sample_shape=[3, 4])
    shaper = DistributionShape(batch_ndims=0, event_ndims=2)
    S, B, E = shaper.get_shape(rand_wishart)
    # S = [3, 4]
    # B = []
    # E = [2, 2]

    # 100 iid samples from two, non-identical trivariate Normal distributions.
    mu    = ... # shape(2, 3)
    sigma = ... # shape(2, 3, 3)
    X = MultivariateNormal(mu, sigma).sample(shape=[4, 25])
    # S = [4, 25]
    # B = [2]
    # E = [3]
    ```

  Argument Validation:
    When `validate_args=False`, checks that cannot be done during
    graph construction are performed at graph execution. This may result in a
    performance degradation because data must be switched from GPU to CPU.

    For example, when `validate_args=False` and `event_ndims` is a
    non-constant `Tensor`, it is checked to be a non-negative integer at graph
    execution. (Same for `batch_ndims`).  Constant `Tensor`s and non-`Tensor`
    arguments are always checked for correctness since this can be done for
    "free," i.e., during graph construction.
  """

  def __init__(self,
               batch_ndims=None,
               event_ndims=None,
               validate_args=False,
               name="DistributionShape"):
    """Construct `DistributionShape` with fixed `batch_ndims`, `event_ndims`.

    `batch_ndims` and `event_ndims` are fixed throughout the lifetime of a
    `Distribution`.  They may only be known at graph execution.

    If both `batch_ndims` and `event_ndims` are python scalars (rather than
    either being a `Tensor`), functions in this class automatically perform
    sanity checks during graph construction.

    Args:
      batch_ndims: `Tensor`. Number of `dims` (`rank`) of the batch portion of
        indexes of a `Tensor`.  A "batch" is a non-identical distribution, i.e,
        Normal with different parameters.
      event_ndims: `Tensor`. Number of `dims` (`rank`) of the event portion of
        indexes of a `Tensor`. An "event" is what is sampled from a
        distribution, i.e., a trivariate Normal has an event shape of [3] and a
        4 dimensional Wishart has an event shape of [4, 4].
      validate_args: `Boolean`, default `False`. When `True`, non-`tf.constant`
        `Tensor` arguments are checked for correctness. (`tf.constant`
        arguments are always checked.)
      name: `String`. The name prepended to Ops created by this class.

    Raises:
      ValueError: if either `batch_ndims` or `event_ndims` are: `None`,
        negative, not `int32`.
    """
    if batch_ndims is None: raise ValueError("batch_ndims cannot be None")
    if event_ndims is None: raise ValueError("event_ndims cannot be None")
    self._batch_ndims = batch_ndims
    self._event_ndims = event_ndims
    self._validate_args = validate_args
    with ops.name_scope(name) as ns:
      self._name = ns
      with ops.name_scope("init"):
        self._batch_ndims = self._assert_non_negative_int32_scalar(
            ops.convert_to_tensor(
                batch_ndims, name="batch_ndims"))
        self._batch_ndims_static, self._batch_ndims_is_0 = (
            self._introspect_ndims(self._batch_ndims))
        self._event_ndims = self._assert_non_negative_int32_scalar(
            ops.convert_to_tensor(
                event_ndims, name="event_ndims"))
        self._event_ndims_static, self._event_ndims_is_0 = (
            self._introspect_ndims(self._event_ndims))

  @property
  def name(self):
    """Name given to ops created by this class."""
    return self._name

  @property
  def batch_ndims(self):
    """Returns number of dimensions corresponding to non-identical draws."""
    return self._batch_ndims

  @property
  def event_ndims(self):
    """Returns number of dimensions needed to index a sample's coordinates."""
    return self._event_ndims

  @property
  def validate_args(self):
    """Returns True if graph-runtime `Tensor` checks are enabled."""
    return self._validate_args

  def get_ndims(self, x, name="get_ndims"):
    """Get `Tensor` number of dimensions (rank).

    Args:
      x: `Tensor`.
      name: `String`. The name to give this op.

    Returns:
      ndims: Scalar number of dimensions associated with a `Tensor`.
    """
    with self._name_scope(name, values=[x]):
      x = ops.convert_to_tensor(x, name="x")
      ndims = x.get_shape().ndims
      if ndims is None:
        return array_ops.rank(x, name="ndims")
      return ops.convert_to_tensor(ndims, dtype=dtypes.int32, name="ndims")

  def get_sample_ndims(self, x, name="get_sample_ndims"):
    """Returns number of dimensions corresponding to iid draws ("sample").

    Args:
      x: `Tensor`.
      name: `String`. The name to give this op.

    Returns:
      sample_ndims: `Tensor` (0D, `int32`).

    Raises:
      ValueError: if `sample_ndims` is calculated to be negative.
    """
    with self._name_scope(name, values=[x]):
      ndims = self.get_ndims(x, name=name)
      if self._is_all_constant_helper(ndims, self.batch_ndims,
                                      self.event_ndims):
        ndims = tensor_util.constant_value(ndims)
        sample_ndims = (ndims - self._batch_ndims_static -
                        self._event_ndims_static)
        if sample_ndims < 0:
          raise ValueError(
              "expected batch_ndims(%d) + event_ndims(%d) <= ndims(%d)" %
              (self._batch_ndims_static, self._event_ndims_static, ndims))
        return ops.convert_to_tensor(sample_ndims, name="sample_ndims")
      else:
        with ops.name_scope(name="sample_ndims"):
          sample_ndims = ndims - self.batch_ndims - self.event_ndims
          if self.validate_args:
            sample_ndims = control_flow_ops.with_dependencies(
                [check_ops.assert_non_negative(sample_ndims)], sample_ndims)
        return sample_ndims

  def get_dims(self, x, name="get_dims"):
    """Returns dimensions indexing `sample_shape`, `batch_shape`, `event_shape`.

    Example:

    ```python
    x = ... # Tensor with shape [4, 3, 2, 1]
    sample_dims, batch_dims, event_dims = _DistributionShape(
      batch_ndims=2, event_ndims=1).get_dims(x)
    # sample_dims == [0]
    # batch_dims == [1, 2]
    # event_dims == [3]
    # Note that these are not the shape parts, but rather indexes into shape.
    ```

    Args:
      x: `Tensor`.
      name: `String`. The name to give this op.

    Returns:
      sample_dims: `Tensor` (1D, `int32`).
      batch_dims: `Tensor` (1D, `int32`).
      event_dims: `Tensor` (1D, `int32`).
    """
    with self._name_scope(name, values=[x]):
      def make_dims(start_sum, size, name):
        """Closure to make dims range."""
        start_sum = start_sum if start_sum else (
            array_ops.zeros((), dtype=dtypes.int32, name="zero"),)
        if self._is_all_constant_helper(size, *start_sum):
          start = sum(tensor_util.constant_value(s) for s in start_sum)
          stop = start + tensor_util.constant_value(size)
          return ops.convert_to_tensor(
              list(range(start, stop)), dtype=dtypes.int32, name=name)
        else:
          start = sum(start_sum)
          return math_ops.range(start, start + size)
      sample_ndims = self.get_sample_ndims(x, name=name)
      return (make_dims((), sample_ndims, name="sample_dims"),
              make_dims((sample_ndims,), self.batch_ndims, name="batch_dims"),
              make_dims((sample_ndims, self.batch_ndims),
                        self.event_ndims, name="event_dims"))

  def get_shape(self, x, name="get_shape"):
    """Returns `Tensor`'s shape partitioned into `sample`, `batch`, `event`.

    Args:
      x: `Tensor`.
      name: `String`. The name to give this op.

    Returns:
      sample_shape: `Tensor` (1D, `int32`).
      batch_shape: `Tensor` (1D, `int32`).
      event_shape: `Tensor` (1D, `int32`).
    """
    with self._name_scope(name, values=[x]):
      x = ops.convert_to_tensor(x, name="x")
      def slice_shape(start_sum, size, name):
        """Closure to slice out shape."""
        start_sum = start_sum if start_sum else (
            array_ops.zeros((), dtype=dtypes.int32, name="zero"),)
        if (x.get_shape().ndims is not None and
            self._is_all_constant_helper(size, *start_sum)):
          start = sum(tensor_util.constant_value(s) for s in start_sum)
          stop = start + tensor_util.constant_value(size)
          slice_ = x.get_shape()[start:stop].as_list()
          if all(s is not None for s in slice_):
            return ops.convert_to_tensor(slice_, dtype=dtypes.int32, name=name)
          # Fall-through intended.
        return array_ops.slice(array_ops.shape(x), (sum(start_sum),), (size,))
      sample_ndims = self.get_sample_ndims(x, name=name)
      return (slice_shape((), sample_ndims,
                          name="sample_shape"),
              slice_shape((sample_ndims,), self.batch_ndims,
                          name="batch_shape"),
              slice_shape((sample_ndims, self.batch_ndims), self.event_ndims,
                          name="event_shape"))

  # TODO(jvdillon): Make remove expand_batch_dim and make expand_batch_dim=False
  # the default behavior.
  def make_batch_of_event_sample_matrices(
      self, x, expand_batch_dim=True,
      name="make_batch_of_event_sample_matrices"):
    """Reshapes/transposes `Distribution` `Tensor` from S+B+E to B_+E_+S_.

    Where:
      - `B_ = B if B or not expand_batch_dim  else [1]`,
      - `E_ = E if E else [1]`,
      - `S_ = [tf.reduce_prod(S)]`.

    Args:
      x: `Tensor`.
      expand_batch_dim: Python `Boolean` scalar. If `True` the batch dims will
        be expanded such that batch_ndims>=1.
      name: `String`. The name to give this op.

    Returns:
      x: `Tensor`. Input transposed/reshaped to `B_+E_+S_`.
      sample_shape: `Tensor` (1D, `int32`).
    """
    with self._name_scope(name, values=[x]):
      x = ops.convert_to_tensor(x, name="x")
      sample_shape, batch_shape, event_shape = self.get_shape(x)
      event_shape = distribution_util.pick_vector(
          self._event_ndims_is_0, [1], event_shape)
      if expand_batch_dim:
        batch_shape = distribution_util.pick_vector(
            self._batch_ndims_is_0, [1], batch_shape)
      new_shape = array_ops.concat([[-1], batch_shape, event_shape], 0)
      x = array_ops.reshape(x, shape=new_shape)
      x = distribution_util.rotate_transpose(x, shift=-1)
      return x, sample_shape

  # TODO(jvdillon): Make remove expand_batch_dim and make expand_batch_dim=False
  # the default behavior.
  def undo_make_batch_of_event_sample_matrices(
      self, x, sample_shape, expand_batch_dim=True,
      name="undo_make_batch_of_event_sample_matrices"):
    """Reshapes/transposes `Distribution` `Tensor` from B_+E_+S_ to S+B+E.

    Where:
      - `B_ = B if B or not expand_batch_dim  else [1]`,
      - `E_ = E if E else [1]`,
      - `S_ = [tf.reduce_prod(S)]`.

    This function "reverses" `make_batch_of_event_sample_matrices`.

    Args:
      x: `Tensor` of shape `B_+E_+S_`.
      sample_shape: `Tensor` (1D, `int32`).
      expand_batch_dim: Python `Boolean` scalar. If `True` the batch dims will
        be expanded such that batch_ndims>=1.
      name: `String`. The name to give this op.

    Returns:
      x: `Tensor`. Input transposed/reshaped to `S+B+E`.
    """
    with self._name_scope(name, values=[x, sample_shape]):
      x = ops.convert_to_tensor(x, name="x")
      sample_shape = ops.convert_to_tensor(sample_shape, name="sample_shape")
      x = distribution_util.rotate_transpose(x, shift=1)
      if self._is_all_constant_helper(self.batch_ndims, self.event_ndims):
        if self._batch_ndims_is_0 or self._event_ndims_is_0:
          b = ([min(-2, -1 - self._event_ndims_static)]
               if self._batch_ndims_is_0 and expand_batch_dim else [])
          e = [-1] if self._event_ndims_is_0 else []
          x = array_ops.squeeze(x, squeeze_dims=b + e)
        _, batch_shape, event_shape = self.get_shape(x)
      else:
        s = (x.get_shape().as_list() if x.get_shape().is_fully_defined()
             else array_ops.shape(x))
        batch_shape = s[1:1+self.batch_ndims]
        # Since sample_dims=1 and is left-most, we add 1 to the number of
        # batch_ndims to get the event start dim.
        event_start = array_ops.where(
            math_ops.logical_and(expand_batch_dim, self._batch_ndims_is_0),
            2, 1 + self.batch_ndims)
        event_shape = s[event_start:event_start+self.event_ndims]
      new_shape = array_ops.concat((sample_shape, batch_shape, event_shape), 0)
      x = array_ops.reshape(x, shape=new_shape)
      return x

  @contextlib.contextmanager
  def _name_scope(self, name=None, values=None):
    """Helper function to standardize op scope."""
    with ops.name_scope(self.name):
      with ops.name_scope(name, values=(
          (values or []) + [self.batch_ndims, self.event_ndims])) as scope:
        yield scope

  def _is_all_constant_helper(self, *args):
    """Helper which returns True if all inputs are constant_value."""
    return all(tensor_util.constant_value(x) is not None for x in args)

  def _assert_non_negative_int32_scalar(self, x):
    """Helper which ensures that input is a non-negative, int32, scalar."""
    x = ops.convert_to_tensor(x, name="x")
    if x.dtype.base_dtype != dtypes.int32.base_dtype:
      raise TypeError("%s.dtype=%s is not %s" % (x.name, x.dtype, dtypes.int32))
    x_value_static = tensor_util.constant_value(x)
    if x.get_shape().ndims is not None and x_value_static is not None:
      if x.get_shape().ndims != 0:
        raise ValueError("%s.ndims=%d is not 0 (scalar)" %
                         (x.name, x.get_shape().ndims))
      if x_value_static < 0:
        raise ValueError("%s.value=%d cannot be negative" %
                         (x.name, x_value_static))
      return x
    if self.validate_args:
      x = control_flow_ops.with_dependencies([
          check_ops.assert_rank(x, 0),
          check_ops.assert_non_negative(x)], x)
    return x

  def _introspect_ndims(self, ndims):
    """Helper to establish some properties of input ndims args."""
    if self._is_all_constant_helper(ndims):
      return (tensor_util.constant_value(ndims),
              tensor_util.constant_value(ndims) == 0)
    return None, math_ops.equal(ndims, 0)
