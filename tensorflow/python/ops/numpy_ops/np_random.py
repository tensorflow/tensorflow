# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Random functions."""

# pylint: disable=g-direct-tensorflow-import

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as onp

from tensorflow.python.framework import random_seed
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops.numpy_ops import np_array_ops
from tensorflow.python.ops.numpy_ops import np_dtypes
from tensorflow.python.ops.numpy_ops import np_utils

# TODO(agarwal): deprecate this.
DEFAULT_RANDN_DTYPE = onp.float32


@np_utils.np_doc('random.seed')
def seed(s):
  """Sets the seed for the random number generator.

  Uses `tf.set_random_seed`.

  Args:
    s: an integer.
  """
  try:
    s = int(s)
  except TypeError:
    # TODO(wangpeng): support this?
    raise ValueError('np.seed currently only support integer arguments.')
  random_seed.set_seed(s)


@np_utils.np_doc('random.randn')
def randn(*args):
  """Returns samples from a normal distribution.

  Uses `tf.random_normal`.

  Args:
    *args: The shape of the output array.

  Returns:
    An ndarray with shape `args` and dtype `float64`.
  """
  # TODO(wangpeng): Use new stateful RNG
  if np_utils.isscalar(args):
    args = (args,)
  dtype = np_dtypes.default_float_type()
  return np_utils.tensor_to_ndarray(random_ops.random_normal(args, dtype=dtype))


@np_utils.np_doc('random.uniform')
def uniform(low=0.0, high=1.0, size=None):
  dtype = np_dtypes.default_float_type()
  low = np_array_ops.asarray(low, dtype=dtype)
  high = np_array_ops.asarray(high, dtype=dtype)
  if size is None:
    size = array_ops.broadcast_dynamic_shape(low.shape, high.shape)
  return np_utils.tensor_to_ndarray(
      random_ops.random_uniform(
          shape=size, minval=low, maxval=high, dtype=dtype))


@np_utils.np_doc('random.random')
def random(size=None):
  return uniform(0., 1., size)


@np_utils.np_doc('random.rand')
def rand(*size):
  return uniform(0., 1., size)


@np_utils.np_doc('random.randint')
def randint(low, high=None, size=None, dtype=onp.int):  # pylint: disable=missing-function-docstring
  low = int(low)
  if high is None:
    high = low
    low = 0
  if size is None:
    size = ()
  elif isinstance(size, int):
    size = (size,)
  dtype = np_utils.result_type(dtype)
  if dtype not in (onp.int32, onp.int64):
    raise ValueError('Only np.int32 or np.int64 types are supported')
  return np_utils.tensor_to_ndarray(
      random_ops.random_uniform(
          shape=size, minval=low, maxval=high, dtype=dtype))
