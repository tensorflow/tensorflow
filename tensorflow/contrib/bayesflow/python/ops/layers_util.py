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
"""Utilities for probabilistic layers.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.distributions.python.ops import deterministic as deterministic_lib
from tensorflow.contrib.distributions.python.ops import independent as independent_lib
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops.distributions import normal as normal_lib


def default_loc_scale_fn(
    is_singular=False,
    loc_initializer=init_ops.random_normal_initializer(stddev=0.1),
    untransformed_scale_initializer=init_ops.random_normal_initializer(
        mean=-3., stddev=0.1),
    loc_regularizer=None,
    untransformed_scale_regularizer=None,
    loc_constraint=None,
    untransformed_scale_constraint=None):
  """Makes closure which creates `loc`, `scale` params from `tf.get_variable`.

  This function produces a closure which produces `loc`, `scale` using
  `tf.get_variable`. The closure accepts the following arguments:

    dtype: Type of parameter's event.
    shape: Python `list`-like representing the parameter's event shape.
    name: Python `str` name prepended to any created (or existing)
      `tf.Variable`s.
    trainable: Python `bool` indicating all created `tf.Variable`s should be
      added to the graph collection `GraphKeys.TRAINABLE_VARIABLES`.
    add_variable_fn: `tf.get_variable`-like `callable` used to create (or
      access existing) `tf.Variable`s.

  Args:
    is_singular: Python `bool` indicating if `scale is None`. Default: `False`.
    loc_initializer: Initializer function for the `loc` parameters.
      The default is `tf.random_normal_initializer(mean=0., stddev=0.1)`.
    untransformed_scale_initializer: Initializer function for the `scale`
      parameters. Default value: `tf.random_normal_initializer(mean=-3.,
      stddev=0.1)`. This implies the softplus transformed result has mean
      approximately `0.05` and std. deviation approximately `0.005`.
    loc_regularizer: Regularizer function for the `loc` parameters.
      The default (`None`) is to use the `tf.get_variable` default.
    untransformed_scale_regularizer: Regularizer function for the `scale`
      parameters. The default (`None`) is to use the `tf.get_variable` default.
    loc_constraint: An optional projection function to be applied to the
      loc after being updated by an `Optimizer`. The function must take as input
      the unprojected variable and must return the projected variable (which
      must have the same shape). Constraints are not safe to use when doing
      asynchronous distributed training.
      The default (`None`) is to use the `tf.get_variable` default.
    untransformed_scale_constraint: An optional projection function to be
      applied to the `scale` parameters after being updated by an `Optimizer`
      (e.g. used to implement norm constraints or value constraints). The
      function must take as input the unprojected variable and must return the
      projected variable (which must have the same shape). Constraints are not
      safe to use when doing asynchronous distributed training. The default
      (`None`) is to use the `tf.get_variable` default.

  Returns:
    default_loc_scale_fn: Python `callable` which instantiates `loc`, `scale`
    parameters from args: `dtype, shape, name, trainable, add_variable_fn`.
  """
  def _fn(dtype, shape, name, trainable, add_variable_fn):
    """Creates `loc`, `scale` parameters."""
    loc = add_variable_fn(
        name=name + "_loc",
        shape=shape,
        initializer=loc_initializer,
        regularizer=loc_regularizer,
        constraint=loc_constraint,
        dtype=dtype,
        trainable=trainable)
    if is_singular:
      return loc, None
    untransformed_scale = add_variable_fn(
        name=name + "_untransformed_scale",
        shape=shape,
        initializer=untransformed_scale_initializer,
        regularizer=untransformed_scale_regularizer,
        constraint=untransformed_scale_constraint,
        dtype=dtype,
        trainable=trainable)
    scale = (np.finfo(dtype.as_numpy_dtype).eps +
             nn_ops.softplus(untransformed_scale))
    return loc, scale
  return _fn


def default_mean_field_normal_fn(
    is_singular=False,
    loc_initializer=None,
    untransformed_scale_initializer=None,
    loc_regularizer=None,
    untransformed_scale_regularizer=None,
    loc_constraint=None,
    untransformed_scale_constraint=None):
  """Creates a function to build Normal distributions with trainable params.

  This function produces a closure which produces `tf.distributions.Normal`
  parameterized by a loc` and `scale` each created using `tf.get_variable`. The
  produced closure accepts the following arguments:

    name: Python `str` name prepended to any created (or existing)
      `tf.Variable`s.
    shape: Python `list`-like representing the parameter's event shape.
    dtype: Type of parameter's event.
    trainable: Python `bool` indicating all created `tf.Variable`s should be
      added to the graph collection `GraphKeys.TRAINABLE_VARIABLES`.
    add_variable_fn: `tf.get_variable`-like `callable` used to create (or
      access existing) `tf.Variable`s.

  Args:
    is_singular: Python `bool` if `True`, forces the special case limit of
      `scale->0`, i.e., a `Deterministic` distribution.
    loc_initializer: Initializer function for the `loc` parameters.
      If `None` (default), values are initialized using the default
      initializer used by `tf.get_variable`.
    untransformed_scale_initializer: Initializer function for the `scale`
      parameters. If `None` (default), values are initialized using the default
      initializer used by `tf.get_variable`.
    loc_regularizer: Regularizer function for the `loc` parameters.
    untransformed_scale_regularizer: Regularizer function for the `scale`
      parameters.
    loc_constraint: An optional projection function to be applied to the
      loc after being updated by an `Optimizer`. The function must take as input
      the unprojected variable and must return the projected variable (which
      must have the same shape). Constraints are not safe to use when doing
      asynchronous distributed training.
    untransformed_scale_constraint: An optional projection function to be
      applied to the `scale` parameters after being updated by an `Optimizer`
      (e.g. used to implement norm constraints or value constraints). The
      function must take as input the unprojected variable and must return the
      projected variable (which must have the same shape). Constraints are not
      safe to use when doing asynchronous distributed training.

  Returns:
    make_normal_fn: Python `callable` which creates a `tf.distributions.Normal`
      using from args: `dtype, shape, name, trainable, add_variable_fn`.
  """
  loc_scale_fn_ = default_loc_scale_fn(
      is_singular,
      loc_initializer,
      untransformed_scale_initializer,
      loc_regularizer,
      untransformed_scale_regularizer,
      loc_constraint,
      untransformed_scale_constraint)
  def _fn(dtype, shape, name, trainable, add_variable_fn):
    """Creates multivariate `Deterministic` or `Normal` distribution."""
    loc, scale = loc_scale_fn_(dtype, shape, name, trainable, add_variable_fn)
    if scale is None:
      dist = deterministic_lib.Deterministic(loc=loc)
    else:
      dist = normal_lib.Normal(loc=loc, scale=scale)
    reinterpreted_batch_ndims = array_ops.shape(dist.batch_shape_tensor())[0]
    return independent_lib.Independent(
        dist, reinterpreted_batch_ndims=reinterpreted_batch_ndims)
  return _fn
