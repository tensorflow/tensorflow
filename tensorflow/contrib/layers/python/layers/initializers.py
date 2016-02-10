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


__all__ = ['xavier_initializer', 'xavier_initializer_conv2d']


def _xavier(n_inputs, n_outputs, shape, uniform, seed, dtype):
  if uniform:
    # 6 was used in the paper.
    init_range = math.sqrt(6.0 / (n_inputs + n_outputs))
    return random_ops.random_uniform(shape, -init_range, init_range,
                                     dtype, seed=seed)
  else:
    # 3 gives us approximately the same limits as above since this repicks
    # values greater than 2 standard deviations from the mean.
    stddev = math.sqrt(3.0 / (n_inputs + n_outputs))
    return random_ops.truncated_normal(shape, 0.0, stddev, dtype, seed=seed)


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

  The returned initializer assumes that the shape of the weight matrix to be
  initialized is `[in, out]`.

  Args:
    uniform: Whether to use uniform or normal distributed random initialization.
    seed: A Python integer. Used to create random seeds. See
      [`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed)
      for behavior.
    dtype: The data type. Only floating point types are supported.

  Returns:
    An initializer for a 2-D weight matrix.

  Raises:
    TypeError: If dtype is not a floating point type.
  """
  if not dtype.is_floating:
    raise TypeError('Cannot create Xavier initializer for non-floating point '
                    'type.')
  def _initializer(shape, dtype=dtype):
    n_inputs = shape[0]
    n_outputs = shape[1]
    return _xavier(n_inputs, n_outputs, shape, uniform, seed, dtype)
  return _initializer


def xavier_initializer_conv2d(uniform=True, seed=None, dtype=dtypes.float32):
  """Returns an "Xavier" initializer for 2D convolution weights.

  For details on the initialization performed, see `xavier_initializer`. This
  function initializes a convolution weight variable which is assumed to be 4-D.
  The first two dimensions are expected to be the kernel size, the third
  dimension is the number of input channels, and the last dimension is the
  number of output channels.

  The number of inputs is therefore `shape[0]*shape[1]*shape[2]`, and the number
  of outputs is `shape[0]*shape[1]*shape[3]`.

  Args:
    uniform: Whether to use uniform or normal distributed random initialization.
    seed: A Python integer. Used to create random seeds. See
      [`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed)
      for behavior.
    dtype: The data type. Only floating point types are supported.

  Returns:
    An initializer for a 4-D weight matrix.

  Raises:
    TypeError: If dtype is not a floating point type.
  """
  if not dtype.is_floating:
    raise TypeError('Cannot create Xavier initializer for non-floating point '
                    'type.')
  def _initializer(shape, dtype=dtype):
    n_inputs = shape[0] * shape[1] * shape[2]
    n_outputs = shape[0] * shape[1] * shape[3]
    return _xavier(n_inputs, n_outputs, shape, uniform, seed, dtype)
  return _initializer
