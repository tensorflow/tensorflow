# Copyright 2015 Google Inc. All Rights Reserved.
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
"""Regularizers for use with layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numbers

from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import standard_ops
from tensorflow.python.platform import logging

__all__ = ['l1_regularizer', 'l2_regularizer', 'sum_regularizer',
           'apply_regularization']


def l1_regularizer(scale):
  """Returns a function that can be used to apply L1 regularization to weights.

  L1 regularization encourages sparsity.

  Args:
    scale: A scalar multiplier `Tensor`. 0.0 disables the regularizer.

  Returns:
    A function with signature `l1(weights, name=None)` that apply L1
    regularization.

  Raises:
    ValueError: If scale is outside of the range [0.0, 1.0] or if scale is not a
    float.
  """
  if isinstance(scale, numbers.Integral):
    raise ValueError('scale cannot be an integer: %s' % scale)
  if isinstance(scale, numbers.Real):
    if scale < 0.:
      raise ValueError('Setting a scale less than 0 on a regularizer: %g' %
                       scale)
    if scale >= 1.:
      raise ValueError('Setting a scale greater than 1 on a regularizer: %g' %
                       scale)
    if scale == 0.:
      logging.info('Scale of 0 disables regularizer.')
      return lambda _, name=None: None

  def l1(weights, name=None):
    """Applies L1 regularization to weights."""
    with ops.op_scope([weights], name, 'l1_regularizer') as scope:
      my_scale = ops.convert_to_tensor(scale,
                                       dtype=weights.dtype.base_dtype,
                                       name='scale')
      return standard_ops.mul(
          my_scale,
          standard_ops.reduce_sum(standard_ops.abs(weights)),
          name=scope)

  return l1


def l2_regularizer(scale):
  """Returns a function that can be used to apply L2 regularization to weights.

  Small values of L2 can help prevent overfitting the training data.

  Args:
    scale: A scalar multiplier `Tensor`. 0.0 disables the regularizer.

  Returns:
    A function with signature `l2(weights, name=None)` that applies L2
    regularization.

  Raises:
    ValueError: If scale is outside of the range [0.0, 1.0] or if scale is not a
    float.
  """
  if isinstance(scale, numbers.Integral):
    raise ValueError('scale cannot be an integer: %s' % (scale,))
  if isinstance(scale, numbers.Real):
    if scale < 0.:
      raise ValueError('Setting a scale less than 0 on a regularizer: %g.' %
                       scale)
    if scale >= 1.:
      raise ValueError('Setting a scale greater than 1 on a regularizer: %g.' %
                       scale)
    if scale == 0.:
      logging.info('Scale of 0 disables regularizer.')
      return lambda _, name=None: None

  def l2(weights, name=None):
    """Applies l2 regularization to weights."""
    with ops.op_scope([weights], name, 'l2_regularizer') as scope:
      my_scale = ops.convert_to_tensor(scale,
                                       dtype=weights.dtype.base_dtype,
                                       name='scale')
      return standard_ops.mul(my_scale, nn.l2_loss(weights), name=scope)

  return l2


def sum_regularizer(regularizer_list):
  """Returns a function that applies the sum of multiple regularizers.

  Args:
    regularizer_list: A list of regularizers to apply.

  Returns:
    A function with signature `sum_reg(weights, name=None)` that applies the
    sum of all the input regularizers.
  """
  regularizer_list = [reg for reg in regularizer_list if reg is not None]
  if not regularizer_list:
    return None

  def sum_reg(weights, name=None):
    """Applies the sum of all the input regularizers."""
    with ops.op_scope([weights], name, 'sum_regularizer') as scope:
      regularizer_tensors = [reg(weights) for reg in regularizer_list]
      return math_ops.add_n(regularizer_tensors, name=scope)

  return sum_reg


def apply_regularization(regularizer, weights_list=None):
  """Returns the summed penalty by applying `regularizer` to the `weights_list`.

  Adding a regularization penalty over the layer weights and embedding weights
  can help prevent overfitting the training data. Regularization over layer
  biases is less common/useful, but assuming proper data preprocessing/mean
  subtraction, it usually shouldn't hurt much either.

  Args:
    regularizer: A function that takes a single `Tensor` argument and returns
      a scalar `Tensor` output.
    weights_list: List of weights `Tensors` or `Variables` to apply
      `regularizer` over. Defaults to the `GraphKeys.WEIGHTS` collection if
      `None`.

  Returns:
    A scalar representing the overall regularization penalty.

  Raises:
    ValueError: If `regularizer` does not return a scalar output.
  """
  if not weights_list:
    weights_list = ops.get_collection(ops.GraphKeys.WEIGHTS)
  with ops.op_scope(weights_list, 'get_regularization_penalty') as scope:
    penalties = [regularizer(w) for w in weights_list]
    for p in penalties:
      if p.get_shape().ndims != 0:
        raise ValueError('regularizer must return a scalar Tensor instead of a '
                         'Tensor with rank %d.' % p.get_shape().ndims)

    summed_penalty = math_ops.add_n(penalties, name=scope)
    ops.add_to_collection(ops.GraphKeys.REGULARIZATION_LOSSES, summed_penalty)
    return summed_penalty
