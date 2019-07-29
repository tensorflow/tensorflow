# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""The BatchReshape distribution."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.distributions import distribution as distribution_lib
from tensorflow.python.util import deprecation


__all__ = [
    "BatchReshape",
]


class BatchReshape(distribution_lib.Distribution):
  """The Batch-Reshaping distribution.

  This "meta-distribution" reshapes the batch dimensions of another
  distribution.

  #### Examples

  ```python
  import tensorflow_probability as tfp
  tfd = tfp.distributions

  dtype = np.float32
  dims = 2
  new_batch_shape = [1, 2, -1]
  old_batch_shape = [6]

  scale = np.ones(old_batch_shape + [dims], dtype)
  mvn = tfd.MultivariateNormalDiag(scale_diag=scale)
  reshape_mvn = tfd.BatchReshape(
      distribution=mvn,
      batch_shape=new_batch_shape,
      validate_args=True)

  reshape_mvn.batch_shape
  # ==> [1, 2, 3]

  x = reshape_mvn.sample(sample_shape=[4, 5])
  x.shape
  # ==> [4, 5, 1, 2, 3, 2] == sample_shape + new_batch_shape + [dims]

  reshape_mvn.log_prob(x).shape
  # ==> [4, 5, 1, 2, 3] == sample_shape + new_batch_shape
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
               distribution,
               batch_shape,
               validate_args=False,
               allow_nan_stats=True,
               name=None):
    """Construct BatchReshape distribution.

    Args:
      distribution: The base distribution instance to reshape. Typically an
        instance of `Distribution`.
      batch_shape: Positive `int`-like vector-shaped `Tensor` representing
        the new shape of the batch dimensions. Up to one dimension may contain
        `-1`, meaning the remainder of the batch size.
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
      allow_nan_stats: Python `bool`, default `True`. When `True`, statistics
        (e.g., mean, mode, variance) use the value "`NaN`" to indicate the
        result is undefined. When `False`, an exception is raised if one or
        more of the statistic's batch members are undefined.
      name: The name to give Ops created by the initializer.
        Default value: `"BatchReshape" + distribution.name`.

    Raises:
      ValueError: if `batch_shape` is not a vector.
      ValueError: if `batch_shape` has non-positive elements.
      ValueError: if `batch_shape` size is not the same as a
        `distribution.batch_shape` size.
    """
    parameters = dict(locals())
    name = name or "BatchReshape" + distribution.name
    with ops.name_scope(name, values=[batch_shape]) as name:
      # The unexpanded batch shape may contain up to one dimension of -1.
      self._batch_shape_unexpanded = ops.convert_to_tensor(
          batch_shape, dtype=dtypes.int32, name="batch_shape")
      validate_init_args_statically(distribution, self._batch_shape_unexpanded)
      batch_shape, batch_shape_static, runtime_assertions = calculate_reshape(
          distribution.batch_shape_tensor(), self._batch_shape_unexpanded,
          validate_args)
      self._distribution = distribution
      self._batch_shape_ = batch_shape
      self._batch_shape_static = batch_shape_static
      self._runtime_assertions = runtime_assertions
      super(BatchReshape, self).__init__(
          dtype=distribution.dtype,
          reparameterization_type=distribution.reparameterization_type,
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          parameters=parameters,
          graph_parents=(
              [self._batch_shape_unexpanded] + distribution._graph_parents),  # pylint: disable=protected-access
          name=name)

  @property
  def distribution(self):
    return self._distribution

  def _batch_shape_tensor(self):
    with ops.control_dependencies(self._runtime_assertions):
      return array_ops.identity(self._batch_shape_)

  def _batch_shape(self):
    return self._batch_shape_static

  def _event_shape_tensor(self):
    with ops.control_dependencies(self._runtime_assertions):
      return array_ops.identity(self.distribution.event_shape_tensor())

  def _event_shape(self):
    return self.distribution.event_shape

  def _sample_n(self, n, seed=None):
    with ops.control_dependencies(self._runtime_assertions):
      x = self.distribution.sample(sample_shape=n, seed=seed)
      new_shape = array_ops.concat(
          [
              [n],
              self._batch_shape_unexpanded,
              self.event_shape_tensor(),
          ],
          axis=0)
      return array_ops.reshape(x, new_shape)

  def _log_prob(self, x):
    return self._call_reshape_input_output(
        self.distribution.log_prob, x)

  def _prob(self, x):
    return self._call_reshape_input_output(
        self.distribution.prob, x)

  def _log_cdf(self, x):
    return self._call_reshape_input_output(
        self.distribution.log_cdf, x)

  def _cdf(self, x):
    return self._call_reshape_input_output(
        self.distribution.cdf, x)

  def _log_survival_function(self, x):
    return self._call_reshape_input_output(
        self.distribution.log_survival_function, x)

  def _survival_function(self, x):
    return self._call_reshape_input_output(
        self.distribution.survival_function, x)

  def _entropy(self):
    return self._call_and_reshape_output(self.distribution.entropy, [],
                                         [tensor_shape.TensorShape([])])

  def _mean(self):
    return self._call_and_reshape_output(self.distribution.mean)

  def _mode(self):
    return self._call_and_reshape_output(self.distribution.mode)

  def _stddev(self):
    return self._call_and_reshape_output(self.distribution.stddev)

  def _variance(self):
    return self._call_and_reshape_output(self.distribution.variance)

  def _covariance(self):
    return self._call_and_reshape_output(
        self.distribution.covariance,
        [self.event_shape_tensor()]*2,
        [self.event_shape]*2)

  def _sample_shape(self, x):
    """Computes graph and static `sample_shape`."""
    x_ndims = (array_ops.rank(x) if x.shape.ndims is None else x.shape.ndims)
    event_ndims = (array_ops.size(self.event_shape_tensor())
                   if self.event_shape.ndims is None
                   else self.event_shape.ndims)
    batch_ndims = (
        array_ops.size(self._batch_shape_unexpanded)
        if self.batch_shape.ndims is None else self.batch_shape.ndims)
    sample_ndims = x_ndims - batch_ndims - event_ndims
    if isinstance(sample_ndims, int):
      static_sample_shape = x.shape[:sample_ndims]
    else:
      static_sample_shape = tensor_shape.TensorShape(None)
    if static_sample_shape.is_fully_defined():
      sample_shape = np.int32(static_sample_shape.as_list())
    else:
      sample_shape = array_ops.shape(x)[:sample_ndims]
    return sample_shape, static_sample_shape

  def _call_reshape_input_output(self, fn, x):
    """Calls `fn`, appropriately reshaping its input `x` and output."""
    with ops.control_dependencies(
        self._runtime_assertions + self._validate_sample_arg(x)):
      sample_shape, static_sample_shape = self._sample_shape(x)
      old_shape = array_ops.concat([
          sample_shape,
          self.distribution.batch_shape_tensor(),
          self.event_shape_tensor(),
      ], axis=0)
      result = fn(array_ops.reshape(x, old_shape))
      new_shape = array_ops.concat(
          [
              sample_shape,
              self._batch_shape_unexpanded,
          ], axis=0)
      result = array_ops.reshape(result, new_shape)
      if (static_sample_shape.ndims is not None and
          self.batch_shape.ndims is not None):
        new_shape = static_sample_shape.concatenate(self.batch_shape)
        result.set_shape(result.shape.merge_with(new_shape))
      return result

  def _call_and_reshape_output(
      self,
      fn,
      event_shape_list=None,
      static_event_shape_list=None):
    """Calls `fn` and appropriately reshapes its output."""
    with ops.control_dependencies(self._runtime_assertions):
      if event_shape_list is None:
        event_shape_list = [self._event_shape_tensor()]
      if static_event_shape_list is None:
        static_event_shape_list = [self.event_shape]
      new_shape = array_ops.concat(
          [self._batch_shape_unexpanded] + event_shape_list, axis=0)
      result = array_ops.reshape(fn(), new_shape)
      if (self.batch_shape.ndims is not None and
          self.event_shape.ndims is not None):
        event_shape = tensor_shape.TensorShape([])
        for rss in static_event_shape_list:
          event_shape = event_shape.concatenate(rss)
        static_shape = result.shape.merge_with(
            self.batch_shape.concatenate(event_shape))
        result.set_shape(static_shape)
      return result

  def _validate_sample_arg(self, x):
    """Helper which validates sample arg, e.g., input to `log_prob`."""
    with ops.name_scope(name="validate_sample_arg", values=[x]):
      x_ndims = (array_ops.rank(x) if x.shape.ndims is None else x.shape.ndims)
      event_ndims = (array_ops.size(self.event_shape_tensor())
                     if self.event_shape.ndims is None
                     else self.event_shape.ndims)
      batch_ndims = (
          array_ops.size(self._batch_shape_unexpanded)
          if self.batch_shape.ndims is None else self.batch_shape.ndims)
      expected_batch_event_ndims = batch_ndims + event_ndims

      if (isinstance(x_ndims, int) and
          isinstance(expected_batch_event_ndims, int)):
        if x_ndims < expected_batch_event_ndims:
          raise NotImplementedError(
              "Broadcasting is not supported; too few batch and event dims "
              "(expected at least {}, saw {}).".format(
                  expected_batch_event_ndims, x_ndims))
        ndims_assertion = []
      elif self.validate_args:
        ndims_assertion = [
            check_ops.assert_greater_equal(
                x_ndims,
                expected_batch_event_ndims,
                message=("Broadcasting is not supported; too few "
                         "batch and event dims."),
                name="assert_batch_and_event_ndims_large_enough"),
        ]

      if (self.batch_shape.is_fully_defined() and
          self.event_shape.is_fully_defined()):
        expected_batch_event_shape = np.int32(self.batch_shape.concatenate(
            self.event_shape).as_list())
      else:
        expected_batch_event_shape = array_ops.concat([
            self.batch_shape_tensor(),
            self.event_shape_tensor(),
        ], axis=0)

      sample_ndims = x_ndims - expected_batch_event_ndims
      if isinstance(sample_ndims, int):
        sample_ndims = max(sample_ndims, 0)
      if (isinstance(sample_ndims, int) and
          x.shape[sample_ndims:].is_fully_defined()):
        actual_batch_event_shape = np.int32(x.shape[sample_ndims:].as_list())
      else:
        sample_ndims = math_ops.maximum(sample_ndims, 0)
        actual_batch_event_shape = array_ops.shape(x)[sample_ndims:]

      if (isinstance(expected_batch_event_shape, np.ndarray) and
          isinstance(actual_batch_event_shape, np.ndarray)):
        if any(expected_batch_event_shape != actual_batch_event_shape):
          raise NotImplementedError("Broadcasting is not supported; "
                                    "unexpected batch and event shape "
                                    "(expected {}, saw {}).".format(
                                        expected_batch_event_shape,
                                        actual_batch_event_shape))
        # We need to set the final runtime-assertions to `ndims_assertion` since
        # its possible this assertion was created. We could add a condition to
        # only do so if `self.validate_args == True`, however this is redundant
        # as `ndims_assertion` already encodes this information.
        runtime_assertions = ndims_assertion
      elif self.validate_args:
        # We need to make the `ndims_assertion` a control dep because otherwise
        # TF itself might raise an exception owing to this assertion being
        # ill-defined, ie, one cannot even compare different rank Tensors.
        with ops.control_dependencies(ndims_assertion):
          shape_assertion = check_ops.assert_equal(
              expected_batch_event_shape,
              actual_batch_event_shape,
              message=("Broadcasting is not supported; "
                       "unexpected batch and event shape."),
              name="assert_batch_and_event_shape_same")
        runtime_assertions = [shape_assertion]
      else:
        runtime_assertions = []

      return runtime_assertions


@deprecation.deprecated(
    "2018-10-01",
    "The TensorFlow Distributions library has moved to "
    "TensorFlow Probability "
    "(https://github.com/tensorflow/probability). You "
    "should update all references to use `tfp.distributions` "
    "instead of `tf.contrib.distributions`.",
    warn_once=True)
def calculate_reshape(original_shape, new_shape, validate=False, name=None):
  """Calculates the reshaped dimensions (replacing up to one -1 in reshape)."""
  batch_shape_static = tensor_util.constant_value_as_shape(new_shape)
  if batch_shape_static.is_fully_defined():
    return np.int32(batch_shape_static.as_list()), batch_shape_static, []
  with ops.name_scope(name, "calculate_reshape", [original_shape, new_shape]):
    original_size = math_ops.reduce_prod(original_shape)
    implicit_dim = math_ops.equal(new_shape, -1)
    size_implicit_dim = (
        original_size // math_ops.maximum(1, -math_ops.reduce_prod(new_shape)))
    new_ndims = array_ops.shape(new_shape)
    expanded_new_shape = array_ops.where_v2(  # Assumes exactly one `-1`.
        implicit_dim, array_ops.fill(new_ndims, size_implicit_dim), new_shape)
    validations = [] if not validate else [
        check_ops.assert_rank(
            original_shape, 1, message="Original shape must be a vector."),
        check_ops.assert_rank(
            new_shape, 1, message="New shape must be a vector."),
        check_ops.assert_less_equal(
            math_ops.count_nonzero(implicit_dim, dtype=dtypes.int32),
            1,
            message="At most one dimension can be unknown."),
        check_ops.assert_positive(
            expanded_new_shape, message="Shape elements must be >=-1."),
        check_ops.assert_equal(
            math_ops.reduce_prod(expanded_new_shape),
            original_size,
            message="Shape sizes do not match."),
    ]
    return expanded_new_shape, batch_shape_static, validations


@deprecation.deprecated(
    "2018-10-01",
    "The TensorFlow Distributions library has moved to "
    "TensorFlow Probability "
    "(https://github.com/tensorflow/probability). You "
    "should update all references to use `tfp.distributions` "
    "instead of `tf.contrib.distributions`.",
    warn_once=True)
def validate_init_args_statically(distribution, batch_shape):
  """Helper to __init__ which makes or raises assertions."""
  if batch_shape.shape.ndims is not None:
    if batch_shape.shape.ndims != 1:
      raise ValueError("`batch_shape` must be a vector "
                       "(saw rank: {}).".format(batch_shape.shape.ndims))

  batch_shape_static = tensor_util.constant_value_as_shape(batch_shape)
  batch_size_static = batch_shape_static.num_elements()
  dist_batch_size_static = distribution.batch_shape.num_elements()

  if batch_size_static is not None and dist_batch_size_static is not None:
    if batch_size_static != dist_batch_size_static:
      raise ValueError("`batch_shape` size ({}) must match "
                       "`distribution.batch_shape` size ({}).".format(
                           batch_size_static, dist_batch_size_static))

  if batch_shape_static.dims is not None:
    if any(
        dim.value is not None and
        dim.value < 1 for dim in batch_shape_static.dims):
      raise ValueError("`batch_shape` elements must be >=-1.")
