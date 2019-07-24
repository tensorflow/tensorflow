# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import standard_ops
from tensorflow.python.platform import tf_logging as logging

__all__ = ['l1_regularizer',
           'l2_regularizer',
           'l1_l2_regularizer',
           'sum_regularizer',
           'apply_regularization']


def l1_regularizer(scale, scope=None):
  """Returns a function that can be used to apply L1 regularization to weights.

  L1 regularization encourages sparsity.

  Args:
    scale: A scalar multiplier `Tensor`. 0.0 disables the regularizer.
    scope: An optional scope name.

  Returns:
    A function with signature `l1(weights)` that apply L1 regularization.

  Raises:
    ValueError: If scale is negative or if scale is not a float.
  """
  if isinstance(scale, numbers.Integral):
    raise ValueError('scale cannot be an integer: %s' % scale)
  if isinstance(scale, numbers.Real):
    if scale < 0.:
      raise ValueError('Setting a scale less than 0 on a regularizer: %g' %
                       scale)
    if scale == 0.:
      logging.info('Scale of 0 disables regularizer.')
      return lambda _: None

  def l1(weights, name=None):
    """Applies L1 regularization to weights."""
    with ops.name_scope(scope, 'l1_regularizer', [weights]) as name:
      my_scale = ops.convert_to_tensor(scale,
                                       dtype=weights.dtype.base_dtype,
                                       name='scale')
      return standard_ops.multiply(
          my_scale,
          standard_ops.reduce_sum(standard_ops.abs(weights)),
          name=name)

  return l1


def l2_regularizer(scale, scope=None):
  """Returns a function that can be used to apply L2 regularization to weights.

  Small values of L2 can help prevent overfitting the training data.

  Args:
    scale: A scalar multiplier `Tensor`. 0.0 disables the regularizer.
    scope: An optional scope name.

  Returns:
    A function with signature `l2(weights)` that applies L2 regularization.

  Raises:
    ValueError: If scale is negative or if scale is not a float.
  """
  if isinstance(scale, numbers.Integral):
    raise ValueError('scale cannot be an integer: %s' % (scale,))
  if isinstance(scale, numbers.Real):
    if scale < 0.:
      raise ValueError('Setting a scale less than 0 on a regularizer: %g.' %
                       scale)
    if scale == 0.:
      logging.info('Scale of 0 disables regularizer.')
      return lambda _: None

  def l2(weights):
    """Applies l2 regularization to weights."""
    with ops.name_scope(scope, 'l2_regularizer', [weights]) as name:
      my_scale = ops.convert_to_tensor(scale,
                                       dtype=weights.dtype.base_dtype,
                                       name='scale')
      return standard_ops.multiply(my_scale, nn.l2_loss(weights), name=name)

  return l2


def l1_l2_regularizer(scale_l1=1.0, scale_l2=1.0, scope=None):
  """Returns a function that can be used to apply L1 L2 regularizations.

  Args:
    scale_l1: A scalar multiplier `Tensor` for L1 regularization.
    scale_l2: A scalar multiplier `Tensor` for L2 regularization.
    scope: An optional scope name.

  Returns:
    A function with signature `l1_l2(weights)` that applies a weighted sum of
    L1 L2 regularization.

  Raises:
    ValueError: If scale is negative or if scale is not a float.
  """
  if isinstance(scale_l1, numbers.Integral):
    raise ValueError('scale_l1 cannot be an integer: %s' % (scale_l1,))
  if isinstance(scale_l2, numbers.Integral):
    raise ValueError('scale_l2 cannot be an integer: %s' % (scale_l2,))
  scope = scope or 'l1_l2_regularizer'
  if scale_l1 == 0.:
    return l2_regularizer(scale_l2, scope)
  if scale_l2 == 0.:
    return l1_regularizer(scale_l1, scope)
  return sum_regularizer([l1_regularizer(scale_l1),
                          l2_regularizer(scale_l2)],
                         scope=scope)


def sum_regularizer(regularizer_list, scope=None):
  """Returns a function that applies the sum of multiple regularizers.

  Args:
    regularizer_list: A list of regularizers to apply.
    scope: An optional scope name

  Returns:
    A function with signature `sum_reg(weights)` that applies the
    sum of all the input regularizers.
  """
  regularizer_list = [reg for reg in regularizer_list if reg is not None]
  if not regularizer_list:
    return None

  def sum_reg(weights):
    """Applies the sum of all the input regularizers."""
    with ops.name_scope(scope, 'sum_regularizer', [weights]) as name:
      regularizer_tensors = []
      for reg in regularizer_list:
        tensor = reg(weights)
        if tensor is not None:
          regularizer_tensors.append(tensor)
      return math_ops.add_n(
          regularizer_tensors, name=name) if regularizer_tensors else None

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
    ValueError: If `regularizer` does not return a scalar output, or if we find
        no weights.
  """
  if not weights_list:
    weights_list = ops.get_collection(ops.GraphKeys.WEIGHTS)
  if not weights_list:
    raise ValueError('No weights to regularize.')
  with ops.name_scope('get_regularization_penalty',
                      values=weights_list) as scope:
    penalties = [regularizer(w) for w in weights_list]
    penalties = [
        p if p is not None else constant_op.constant(0.0) for p in penalties
    ]
    for p in penalties:
      if p.get_shape().ndims != 0:
        raise ValueError('regularizer must return a scalar Tensor instead of a '
                         'Tensor with rank %d.' % p.get_shape().ndims)

    summed_penalty = math_ops.add_n(penalties, name=scope)
    ops.add_to_collection(ops.GraphKeys.REGULARIZATION_LOSSES, summed_penalty)
    return summed_penalty
