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
    raise ValueError(
        f'Argument `s` got an invalid value {s}. Only integers are supported.')
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
  return standard_normal(size=args)


@np_utils.np_doc('random.standard_normal')
def standard_normal(size=None):
  # TODO(wangpeng): Use new stateful RNG
  if size is None:
    size = ()
  elif np_utils.isscalar(size):
    size = (size,)
  dtype = np_dtypes.default_float_type()
  return random_ops.random_normal(size, dtype=dtype)


@np_utils.np_doc('random.uniform')
def uniform(low=0.0, high=1.0, size=None):
  dtype = np_dtypes.default_float_type()
  low = np_array_ops.asarray(low, dtype=dtype)
  high = np_array_ops.asarray(high, dtype=dtype)
  if size is None:
    size = array_ops.broadcast_dynamic_shape(low.shape, high.shape)
  return random_ops.random_uniform(
      shape=size, minval=low, maxval=high, dtype=dtype)


@np_utils.np_doc('random.poisson')
def poisson(lam=1.0, size=None):
  if size is None:
    size = ()
  elif np_utils.isscalar(size):
    size = (size,)
  return random_ops.random_poisson(shape=size, lam=lam, dtype=np_dtypes.int_)


@np_utils.np_doc('random.random')
def random(size=None):
  return uniform(0., 1., size)


@np_utils.np_doc('random.rand')
def rand(*size):
  return uniform(0., 1., size)


@np_utils.np_doc('random.randint')
def randint(low, high=None, size=None, dtype=onp.int64):  # pylint: disable=missing-function-docstring
  low = int(low)
  if high is None:
    high = low
    low = 0
  if size is None:
    size = ()
  elif isinstance(size, int):
    size = (size,)
  dtype_orig = dtype
  dtype = np_utils.result_type(dtype)
  accepted_dtypes = (onp.int32, onp.int64)
  if dtype not in accepted_dtypes:
    raise ValueError(
        f'Argument `dtype` got an invalid value {dtype_orig}. Only those '
        f'convertible to {accepted_dtypes} are supported.')
  return random_ops.random_uniform(
      shape=size, minval=low, maxval=high, dtype=dtype)
