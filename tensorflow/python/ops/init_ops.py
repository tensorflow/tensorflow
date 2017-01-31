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

"""Operations often used for initializing tensors.

All variable initializers returned by functions in this file should have the
following signature:

def _initializer(shape, dtype=dtypes.float32, partition_info=None):
  Args:
    shape: List of `int` representing the shape of the output `Tensor`. Some
      initializers may also be able to accept a `Tensor`.
    dtype: (Optional) Type of the output `Tensor`.
    partition_info: (Optional) variable_scope._PartitionInfo object holding
      additional information about how the variable is partitioned. May be
      `None` if the variable is not partitioned.
  Returns:
    A `Tensor` of type `dtype` and `shape`.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import random_ops


class Initializer(object):
  """Initializer base class: all initializers inherit from this class.
  """

  def __call__(self, shape, dtype=None, partition_info=None):
    raise NotImplementedError


class Zeros(Initializer):
  """Initializer that generates tensors initialized to 0."""

  def __init__(self, dtype=dtypes.float32):
    self.dtype = dtype

  def __call__(self, shape, dtype=None, partition_info=None):
    if dtype is None:
      dtype = self.dtype
    return constant_op.constant(False if dtype is dtypes.bool else 0,
                                dtype=dtype, shape=shape)


class Ones(Initializer):
  """Initializer that generates tensors initialized to 1."""

  def __init__(self, dtype=dtypes.float32):
    self.dtype = dtype

  def __call__(self, shape, dtype=None, partition_info=None):
    if dtype is None:
      dtype = self.dtype
    return constant_op.constant(1, dtype=dtype, shape=shape)


class Constant(Initializer):
  """Initializer that generates tensors with constant values.

  The resulting tensor is populated with values of type `dtype`, as
  specified by arguments `value` following the desired `shape` of the
  new tensor (see examples below).

  The argument `value` can be a constant value, or a list of values of type
  `dtype`. If `value` is a list, then the length of the list must be less
  than or equal to the number of elements implied by the desired shape of the
  tensor. In the case where the total number of elements in `value` is less
  than the number of elements required by the tensor shape, the last element
  in `value` will be used to fill the remaining entries. If the total number of
  elements in `value` is greater than the number of elements required by the
  tensor shape, the initializer will raise a `ValueError`.

  Args:
    value: A Python scalar, list of values, or a N-dimensional numpy array. All
      elements of the initialized variable will be set to the corresponding
      value in the `value` argument.
    dtype: The data type.
    verify_shape: Boolean that enables verification of the shape of `value`. If
      `True`, the initializer will throw an error if the shape of `value` is not
      compatible with the shape of the initialized tensor.

  Examples:
    The following example can be rewritten using a numpy.ndarray instead
    of the `value` list, even reshaped, as shown in the two commented lines
    below the `value` list initialization.

  ```python
    >>> import numpy as np
    >>> import tensorflow as tf

    >>> value = [0, 1, 2, 3, 4, 5, 6, 7]
    >>> # value = np.array(value)
    >>> # value = value.reshape([2, 4])
    >>> init = tf.constant_initializer(value)

    >>> print('fitting shape:')
    >>> with tf.Session():
    >>>   x = tf.get_variable('x', shape=[2, 4], initializer=init)
    >>>   x.initializer.run()
    >>>   print(x.eval())

    fitting shape:
    [[ 0.  1.  2.  3.]
     [ 4.  5.  6.  7.]]

    >>> print('larger shape:')
    >>> with tf.Session():
    >>>   x = tf.get_variable('x', shape=[3, 4], initializer=init)
    >>>   x.initializer.run()
    >>>   print(x.eval())

    larger shape:
    [[ 0.  1.  2.  3.]
     [ 4.  5.  6.  7.]
     [ 7.  7.  7.  7.]]

    >>> print('smaller shape:')
    >>> with tf.Session():
    >>>   x = tf.get_variable('x', shape=[2, 3], initializer=init)

    ValueError: Too many elements provided. Needed at most 6, but received 8

    >>> print('shape verification:')
    >>> init_verify = tf.constant_initializer(value, verify_shape=True)
    >>> with tf.Session():
    >>>   x = tf.get_variable('x', shape=[3, 4], initializer=init_verify)

    TypeError: Expected Tensor's shape: (3, 4), got (8,).
  ```
  """

  def __init__(self, value=0, dtype=dtypes.float32, verify_shape=False):
    self.value = value
    self.dtype = dtype
    self.verify_shape = verify_shape

  def __call__(self, shape, dtype=None, partition_info=None):
    if dtype is None:
      dtype = self.dtype
    return constant_op.constant(self.value, dtype=dtype, shape=shape,
                                verify_shape=self.verify_shape)


class RandomUniform(Initializer):
  """Initializer that generates tensors with a uniform distribution.

  Args:
    minval: A python scalar or a scalar tensor. Lower bound of the range
      of random values to generate.
    maxval: A python scalar or a scalar tensor. Upper bound of the range
      of random values to generate.  Defaults to 1 for float types.
    seed: A Python integer. Used to create random seeds. See
      [`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed)
      for behavior.
    dtype: The data type.
  """

  def __init__(self, minval=0, maxval=None, seed=None, dtype=dtypes.float32):
    self.minval = minval
    self.maxval = maxval
    self.seed = seed
    self.dtype = dtype

  def __call__(self, shape, dtype=None, partition_info=None):
    if dtype is None:
      dtype = self.dtype
    return random_ops.random_uniform(shape, self.minval, self.maxval,
                                     dtype, seed=self.seed)


class RandomNormal(Initializer):
  """Initializer that generates tensors with a normal distribution.

  Args:
    mean: a python scalar or a scalar tensor. Mean of the random values
      to generate.
    stddev: a python scalar or a scalar tensor. Standard deviation of the
      random values to generate.
    seed: A Python integer. Used to create random seeds. See
      [`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed)
      for behavior.
    dtype: The data type. Only floating point types are supported.
  """

  def __init__(self, mean=0.0, stddev=1.0, seed=None, dtype=dtypes.float32):
    self.mean = mean
    self.stddev = stddev
    self.seed = seed
    self.dtype = _assert_float_dtype(dtype)

  def __call__(self, shape, dtype=None, partition_info=None):
    if dtype is None:
      dtype = self.dtype
    return random_ops.random_normal(shape, self.mean, self.stddev,
                                    dtype, seed=self.seed)


class TruncatedNormal(Initializer):
  """Initializer that generates a truncated normal distribution.

  These values are similar to values from a `random_normal_initializer`
  except that values more than two standard deviations from the mean
  are discarded and re-drawn. This is the recommended initializer for
  neural network weights and filters.

  Args:
    mean: a python scalar or a scalar tensor. Mean of the random values
      to generate.
    stddev: a python scalar or a scalar tensor. Standard deviation of the
      random values to generate.
    seed: A Python integer. Used to create random seeds. See
      [`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed)
      for behavior.
    dtype: The data type. Only floating point types are supported.
  """

  def __init__(self, mean=0.0, stddev=1.0, seed=None, dtype=dtypes.float32):
    self.mean = mean
    self.stddev = stddev
    self.seed = seed
    self.dtype = _assert_float_dtype(dtype)

  def __call__(self, shape, dtype=None, partition_info=None):
    if dtype is None:
      dtype = self.dtype
    return random_ops.truncated_normal(shape, self.mean, self.stddev,
                                       dtype, seed=self.seed)


class UniformUnitScaling(Initializer):
  """Initializer that generates tensors without scaling variance.

  When initializing a deep network, it is in principle advantageous to keep
  the scale of the input variance constant, so it does not explode or diminish
  by reaching the final layer. If the input is `x` and the operation `x * W`,
  and we want to initialize `W` uniformly at random, we need to pick `W` from

      [-sqrt(3) / sqrt(dim), sqrt(3) / sqrt(dim)]

  to keep the scale intact, where `dim = W.shape[0]` (the size of the input).
  A similar calculation for convolutional networks gives an analogous result
  with `dim` equal to the product of the first 3 dimensions.  When
  nonlinearities are present, we need to multiply this by a constant `factor`.
  See [Sussillo et al., 2014](https://arxiv.org/abs/1412.6558)
  ([pdf](http://arxiv.org/pdf/1412.6558.pdf)) for deeper motivation, experiments
  and the calculation of constants. In section 2.3 there, the constants were
  numerically computed: for a linear layer it's 1.0, relu: ~1.43, tanh: ~1.15.

  Args:
    factor: Float.  A multiplicative factor by which the values will be scaled.
    seed: A Python integer. Used to create random seeds. See
      [`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed)
      for behavior.
    dtype: The data type. Only floating point types are supported.
  """

  def __init__(self, factor=1.0, seed=None, dtype=dtypes.float32):
    self.factor = factor
    self.seed = seed
    self.dtype = _assert_float_dtype(dtype)

  def __call__(self, shape, dtype=None, partition_info=None):
    if dtype is None:
      dtype = self.dtype
    scale_shape = shape
    if partition_info is not None:
      scale_shape = partition_info.full_shape

    input_size = 1.0
    # Estimating input size is not possible to do perfectly, but we try.
    # The estimate, obtained by multiplying all dimensions but the last one,
    # is the right thing for matrix multiply and convolutions (see above).
    for dim in scale_shape[:-1]:
      input_size *= float(dim)
    # Avoid errors when initializing zero-size tensors.
    input_size = max(input_size, 1.0)
    max_val = math.sqrt(3 / input_size) * self.factor
    return random_ops.random_uniform(shape, -max_val, max_val,
                                     dtype, seed=self.seed)


class VarianceScaling(Initializer):
  """Initializer capable of adapting its scale to the shape of weights tensors.

  With `distribution="normal"`, samples are drawn from a truncated normal
  distribution centered on zero, with `stddev = sqrt(scale / n)`
  where n is:
    - number of input units in the weight tensor, if mode = "fan_in"
    - number of output units, if mode = "fan_out"
    - average of the numbers of input and output units, if mode = "fan_avg"

  With `distribution="uniform"`, samples are drawn from a uniform distribution
  within [-limit, limit], with `limit = sqrt(3 * scale / n)`.

  Arguments:
    scale: Scaling factor (positive float).
    mode: One of "fan_in", "fan_out", "fan_avg".
    distribution: Random distribution to use. One of "normal", "uniform".
    seed: A Python integer. Used to create random seeds. See
      [`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed)
      for behavior.
    dtype: The data type. Only floating point types are supported.

  Raises:
    ValueError: In case of an invalid value for the "scale", mode" or
      "distribution" arguments.
  """

  def __init__(self, scale=1.0,
               mode="fan_in",
               distribution="normal",
               seed=None,
               dtype=dtypes.float32):
    if scale <= 0.:
      raise ValueError("`scale` must be positive float.")
    if mode not in {"fan_in", "fan_out", "fan_avg"}:
      raise ValueError("Invalid `mode` argument:", mode)
    distribution = distribution.lower()
    if distribution not in {"normal", "uniform"}:
      raise ValueError("Invalid `distribution` argument:", distribution)
    self.scale = scale
    self.mode = mode
    self.distribution = distribution
    self.seed = seed
    self.dtype = _assert_float_dtype(dtype)

  def __call__(self, shape, dtype=None, partition_info=None):
    if dtype is None:
      dtype = self.dtype
    scale = self.scale
    scale_shape = shape
    if partition_info is not None:
      scale_shape = partition_info.full_shape
    fan_in, fan_out = _compute_fans(scale_shape)
    if self.mode == "fan_in":
      scale /= max(1., fan_in)
    elif self.mode == "fan_out":
      scale /= max(1., fan_out)
    else:
      scale /= max(1., (fan_in + fan_out) / 2.)
    if self.distribution == "normal":
      stddev = math.sqrt(scale)
      return random_ops.truncated_normal(shape, 0.0, stddev,
                                         dtype, seed=self.seed)
    else:
      limit = math.sqrt(3.0 * scale)
      return random_ops.random_uniform(shape, -limit, limit,
                                       dtype, seed=self.seed)


class Orthogonal(Initializer):
  """Initializer that generates an orthogonal matrix.

  If the shape of the tensor to initialize is two-dimensional, i is initialized
  with an orthogonal matrix obtained from the singular value decomposition of a
  matrix of uniform random numbers.

  If the shape of the tensor to initialize is more than two-dimensional,
  a matrix of shape `(shape[0] * ... * shape[n - 2], shape[n - 1])`
  is initialized, where `n` is the length of the shape vector.
  The matrix is subsequently reshaped to give a tensor of the desired shape.

  Args:
    gain: multiplicative factor to apply to the orthogonal matrix
    dtype: The type of the output.
    seed: A Python integer. Used to create random seeds. See
      [`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed)
      for behavior.
  """

  def __init__(self, gain=1.0, dtype=dtypes.float32, seed=None):
    self.gain = gain
    self.dtype = _assert_float_dtype(dtype)
    self.seed = seed

  def __call__(self, shape, dtype=None, partition_info=None):
    if dtype is None:
      dtype = self.dtype
    # Check the shape
    if len(shape) < 2:
      raise ValueError("The tensor to initialize must be "
                       "at least two-dimensional")
    # Flatten the input shape with the last dimension remaining
    # its original shape so it works for conv2d
    num_rows = 1
    for dim in shape[:-1]:
      num_rows *= dim
    num_cols = shape[-1]
    flat_shape = (num_rows, num_cols)

    # Generate a random matrix
    a = random_ops.random_uniform(flat_shape, dtype=dtype, seed=self.seed)
    # Compute the svd
    _, u, v = linalg_ops.svd(a, full_matrices=False)
    # Pick the appropriate singular value decomposition
    if num_rows > num_cols:
      q = u
    else:
      # Tensorflow departs from numpy conventions
      # such that we need to transpose axes here
      q = array_ops.transpose(v)
    return self.gain * array_ops.reshape(q, shape)


# Aliases.

# pylint: disable=invalid-name
zeros_initializer = Zeros
ones_initializer = Ones
constant_initializer = Constant
random_uniform_initializer = RandomUniform
random_normal_initializer = RandomNormal
truncated_normal_initializer = TruncatedNormal
uniform_unit_scaling_initializer = UniformUnitScaling
variance_scaling_initializer = VarianceScaling
orthogonal_initializer = Orthogonal
# pylint: enable=invalid-name


def glorot_uniform_initializer(seed=None, dtype=dtypes.float32):
  """The Glorot uniform initializer, also called Xavier uniform initializer.

  It draws samples from a uniform distribution within [-limit, limit]
  where `limit` is `sqrt(6 / (fan_in + fan_out))`
  where `fan_in` is the number of input units in the weight tensor
  and `fan_out` is the number of output units in the weight tensor.

  Reference: http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf

  Arguments:
    seed: A Python integer. Used to create random seeds. See
      [`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed)
      for behavior.
    dtype: The data type. Only floating point types are supported.

  Returns:
    An initializer.
  """
  return variance_scaling_initializer(scale=1.0,
                                      mode="fan_avg",
                                      distribution="uniform",
                                      seed=seed,
                                      dtype=dtype)


def glorot_normal_initializer(seed=None, dtype=dtypes.float32):
  """The Glorot normal initializer, also called Xavier normal initializer.

  It draws samples from a truncated normal distribution centered on 0
  with `stddev = sqrt(2 / (fan_in + fan_out))`
  where `fan_in` is the number of input units in the weight tensor
  and `fan_out` is the number of output units in the weight tensor.

  Reference: http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf

  Arguments:
    seed: A Python integer. Used to create random seeds. See
      [`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed)
      for behavior.
    dtype: The data type. Only floating point types are supported.

  Returns:
    An initializer.
  """
  return variance_scaling_initializer(scale=1.0,
                                      mode="fan_avg",
                                      distribution="normal",
                                      seed=seed,
                                      dtype=dtype)


# Utility functions.


def _compute_fans(shape):
  """Computes the number of input and output units for a weight shape.

  Arguments:
    shape: Integer shape tuple or TF tensor shape.

  Returns:
    A tuple of scalars (fan_in, fan_out).
  """
  if len(shape) < 1:  # Just to avoid errors for constants.
    fan_in = fan_out = 1
  elif len(shape) == 1:
    fan_in = fan_out = shape[0]
  elif len(shape) == 2:
    fan_in = shape[0]
    fan_out = shape[1]
  else:
    # Assuming convolution kernels (2D, 3D, or more).
    # kernel shape: (..., input_depth, depth)
    receptive_field_size = 1.
    for dim in shape[:-2]:
      receptive_field_size *= dim
    fan_in = shape[-2] * receptive_field_size
    fan_out = shape[-1] * receptive_field_size
  return fan_in, fan_out


def _assert_float_dtype(dtype):
  """Validate and return floating point type based on `dtype`.

  `dtype` must be a floating point type.

  Args:
    dtype: The data type to validate.

  Returns:
    Validated type.

  Raises:
    ValueError: if `dtype` is not a floating point type.
  """
  if not dtype.is_floating:
    raise ValueError("Expected floating point type, got %s." % dtype)
  return dtype
