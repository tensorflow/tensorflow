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
"""Weight initializers for use with layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

from tensorflow.python.framework import dtypes
from tensorflow.python.ops import random_ops


__all__ = ['xavier_initializer', 'xavier_initializer_conv2d',
           'variance_scaling_initializer']


def xavier_initializer(uniform=True, seed=None, dtype=dtypes.float32):
  """Returns an initializer performing "Xavier" initialization for weights.

  This function implements the weight initialization from:

  Xavier Glorot and Yoshua Bengio (2010):
           Understanding the difficulty of training deep feedforward neural
           networks. International conference on artificial intelligence and
           statistics.

  This initializer is designed to keep the scale of the gradients roughly the
  same in all layers. In uniform distribution this ends up being the range:
  `x = sqrt(6. / (in + out)); [-x, x]` and for normal distribution a standard
  deviation of `sqrt(3. / (in + out))` is used.

  Args:
    uniform: Whether to use uniform or normal distributed random initialization.
    seed: A Python integer. Used to create random seeds. See
      [`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed)
      for behavior.
    dtype: The data type. Only floating point types are supported.

  Returns:
    An initializer for a weight matrix.
  """
  return variance_scaling_initializer(factor=1.0, mode='FAN_AVG',
                                      uniform=uniform, seed=seed, dtype=dtype)

xavier_initializer_conv2d = xavier_initializer


def variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False,
                                 seed=None, dtype=dtypes.float32):
  """Returns an initializer that generates tensors without scaling variance.

  When initializing a deep network, it is in principle advantageous to keep
  the scale of the input variance constant, so it does not explode or diminish
  by reaching the final layer. This initializer use the following formula:
    if mode='FAN_IN': # Count only number of input connections.
      n = fan_in
    elif mode='FAN_OUT': # Count only number of output connections.
      n = fan_out
    elif mode='FAN_AVG': # Average number of inputs and output connections.
      n = (fan_in + fan_out)/2.0

      truncated_normal(shape, 0.0, stddev=sqrt(factor / n))

  To get http://arxiv.org/pdf/1502.01852v1.pdf use (Default):
    - factor=2.0 mode='FAN_IN' uniform=False
  To get http://arxiv.org/abs/1408.5093 use:
    - factor=1.0 mode='FAN_IN' uniform=True
  To get http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf use:
    - factor=1.0 mode='FAN_AVG' uniform=True.
  To get xavier_initializer use either:
    - factor=1.0 mode='FAN_AVG' uniform=True.
    - factor=1.0 mode='FAN_AVG' uniform=False.

  Args:
    factor: Float.  A multiplicative factor.
    mode: String.  'FAN_IN', 'FAN_OUT', 'FAN_AVG'.
    uniform: Whether to use uniform or normal distributed random initialization.
    seed: A Python integer. Used to create random seeds. See
      [`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed)
      for behavior.
    dtype: The data type. Only floating point types are supported.

  Returns:
    An initializer that generates tensors with unit variance.

  Raises:
    ValueError: if `dtype` is not a floating point type.
    TypeError: if `mode` is not in ['FAN_IN', 'FAN_OUT', 'FAN_AVG'].
  """
  if not dtype.is_floating:
    raise TypeError('Cannot create initializer for non-floating point '
                    'type.')
  if mode not in ['FAN_IN', 'FAN_OUT', 'FAN_AVG']:
    raise TypeError('Unknow mode %s [FAN_IN, FAN_OUT, FAN_AVG]', mode)
  def _initializer(shape, dtype=dtype):
    """Initializer function."""
    # Estimating fan_in and fan_out is not possible to do perfectly, but we try.
    # This is the right thing for matrix multiply and convolutions.
    fan_in = float(shape[-2])
    fan_out = float(shape[-1])
    for dim in shape[:-2]:
      fan_in *= float(dim)
      fan_out *= float(dim)
    if mode == 'FAN_IN':
      # Count only number of input connections.
      n = fan_in
    elif mode == 'FAN_OUT':
      # Count only number of output connections.
      n = fan_out
    elif mode == 'FAN_AVG':
      # Average number of inputs and output connections.
      n = (fan_in + fan_out) / 2.0
    if uniform:
      # To get stddev = math.sqrt(factor / n) need to adjust for uniform.
      limit = math.sqrt(3.0 * factor / n)
      return random_ops.random_uniform(shape, -limit, limit,
                                       dtype, seed=seed)
    else:
      # To get stddev = math.sqrt(factor / n) need to adjust for truncated.
      trunc_stddev = math.sqrt(1.3 * factor / n)
      return random_ops.truncated_normal(shape, 0.0, trunc_stddev, dtype,
                                         seed=seed)

  return _initializer
