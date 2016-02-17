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
from tensorflow.python.ops import nn
from tensorflow.python.ops import standard_ops
from tensorflow.python.platform import logging


__all__ = ['l1_regularizer', 'l2_regularizer']


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
