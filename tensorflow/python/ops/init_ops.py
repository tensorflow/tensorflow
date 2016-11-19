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
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import linalg_ops


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


def zeros_initializer(shape, dtype=dtypes.float32, partition_info=None):
  """An adaptor for zeros() to match the Initializer spec."""
  return array_ops.zeros(shape, dtype)


def ones_initializer(dtype=dtypes.float32, partition_info=None):
  """An adaptor for ones() to match the Initializer spec."""
  def _initializer(shape, dtype=dtype, partition_info=None):
    return constant_op.constant(1, dtype=dtype, shape=shape)

  return _initializer


def constant_initializer(value=0, dtype=dtypes.float32):
  """Returns an initializer that generates tensors with constant values.

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

  Returns:
    An initializer that generates tensors with constant values.

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
    >>> tf.reset_default_graph()
    >>> with tf.Session():
    >>>   x = tf.get_variable('x', shape=[2, 4], initializer=init)
    >>>   x.initializer.run()
    >>>   print(x.eval())

    fitting shape:
    [[ 0.  1.  2.  3.]
     [ 4.  5.  6.  7.]]

    >>> print('larger shape:')
    >>> tf.reset_default_graph()
    >>> with tf.Session():
    >>>   x = tf.get_variable('x', shape=[3, 4], initializer=init)
    >>>   x.initializer.run()
    >>>   print(x.eval())

    larger shape:
    [[ 0.  1.  2.  3.]
     [ 4.  5.  6.  7.]
     [ 7.  7.  7.  7.]]

    >>> print('smaller shape:')
    >>> tf.reset_default_graph()
    >>> with tf.Session():
    >>>   x = tf.get_variable('x', shape=[2, 3], initializer=init)

    ValueError: Too many elements provided. Needed at most 6, but received 8
  ```
  """
  def _initializer(shape, dtype=dtype, partition_info=None):
    return constant_op.constant(value, dtype=dtype, shape=shape)
  return _initializer


def random_uniform_initializer(minval=0, maxval=None, seed=None,
                               dtype=dtypes.float32):
  """Returns an initializer that generates tensors with a uniform distribution.

  Args:
    minval: A python scalar or a scalar tensor. Lower bound of the range
      of random values to generate.
    maxval: A python scalar or a scalar tensor. Upper bound of the range
      of random values to generate.  Defaults to 1 for float types.
    seed: A Python integer. Used to create random seeds. See
      [`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed)
      for behavior.
    dtype: The data type.

  Returns:
    An initializer that generates tensors with a uniform distribution.
  """
  def _initializer(shape, dtype=dtype, partition_info=None):
    return random_ops.random_uniform(shape, minval, maxval, dtype, seed=seed)
  return _initializer


def random_normal_initializer(mean=0.0, stddev=1.0, seed=None,
                              dtype=dtypes.float32):
  """Returns an initializer that generates tensors with a normal distribution.

  Args:
    mean: a python scalar or a scalar tensor. Mean of the random values
      to generate.
    stddev: a python scalar or a scalar tensor. Standard deviation of the
      random values to generate.
    seed: A Python integer. Used to create random seeds. See
      [`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed)
      for behavior.
    dtype: The data type. Only floating point types are supported.

  Returns:
    An initializer that generates tensors with a normal distribution.

  Raises:
    ValueError: if `dtype` is not a floating point type.
  """
  def _initializer(shape, dtype=_assert_float_dtype(dtype),
                   partition_info=None):
    return random_ops.random_normal(shape, mean, stddev, dtype, seed=seed)
  return _initializer


def truncated_normal_initializer(mean=0.0, stddev=1.0, seed=None,
                                 dtype=dtypes.float32):
  """Returns an initializer that generates a truncated normal distribution.

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

  Returns:
    An initializer that generates tensors with a truncated normal
    distribution.

  Raises:
    ValueError: if `dtype` is not a floating point type.
  """
  def _initializer(shape, dtype=_assert_float_dtype(dtype),
                   partition_info=None):
    return random_ops.truncated_normal(shape, mean, stddev, dtype, seed=seed)

  return _initializer


def uniform_unit_scaling_initializer(factor=1.0,
                                     seed=None,
                                     dtype=dtypes.float32):
  """Returns an initializer that generates tensors without scaling variance.

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

  Returns:
    An initializer that generates tensors with unit variance.

  Raises:
    ValueError: if `dtype` is not a floating point type.
  """
  def _initializer(shape, dtype=_assert_float_dtype(dtype),
                   partition_info=None):
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
    max_val = math.sqrt(3 / input_size) * factor
    return random_ops.random_uniform(shape, -max_val, max_val,
                                     dtype, seed=seed)
  return _initializer


# TODO(vrv): Unhide when we are ready to expose this publicly.
def _random_walk(shape, nonlinearity, dtype=dtypes.float32, seed=None,
                 name="random_walk"):
  """Create a random tensor such that backprop neither vanishes nor explodes.

  Args:
    shape: a python array of int or a 1-d tensor. Sizes of the Tensor.
    nonlinearity: the brain python function for implementing the
      nonlinearity in tensor flow.
    dtype: The type of the output.
    seed: A Python integer. Used to create random seeds. See
      [`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed)
      for behavior.
    name: string.  Optional name for the op.

  Returns:
    A Tensor of the specified sizes filled with random values.
  """
  assert len(shape) == 2, "Random Walk initialization only supports 2D tensors."
  num_inputs = shape[0]
  if nonlinearity == math_ops.tanh:
    # No real formula for this case yet, but this works well for many
    # layer widths.
    rwg = 1.13
  elif nonlinearity == array_ops.identity:
    rwg = math.exp(1.0 / (2.0 * num_inputs))
  elif nonlinearity == nn_ops.relu:
    rwg = math.sqrt(2.0) * math.exp(1.2 / (max(num_inputs, 6) - 2.4))
  else:
    assert False, "Unsupported nonlinearity for Random Walk initialization."

  mean = 0.0
  stddev = rwg / math.sqrt(float(num_inputs))

  return random_ops.random_normal(shape, mean=mean, stddev=stddev, dtype=dtype,
                                  seed=seed, name=name)


# TODO(vrv): Unhide when we are ready to expose this publicly.
class _RandomWalkInitializer(object):
  """An Initializer that generates a tensor for Random Walk Initialization."""

  def __init__(self, nonlinearity, seed=None):
    """Construct a RandomWalkInitializer.

    Args:
      nonlinearity: the python tensorflow function that computes a nonlinearity
        in the graph, typically after a Wx+b type operation.
      seed: A Python integer. Used to create random seeds. See
      [`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed)
      for behavior.
    """
    self._nonlinearity = nonlinearity
    self._seed = seed

  def __call__(self, shape, dtype=dtypes.float32, partition_info=None):
    """Generate a tensor used to initialize a variable."""
    return random_ops._random_walk(shape, self._nonlinearity, dtype,
                                   seed=self._seed)


def orthogonal_initializer(gain=1.0, dtype=dtypes.float32, seed=None):
  """Returns an initializer that generates an orthogonal matrix or a reshaped 
  orthogonal matrix.

  If the shape of the tensor to initialize is two-dimensional, i is initialized 
  with an orthogonal matrix obtained from the singular value decomposition of a 
  matrix of uniform random numbers.

  If the shape of the tensor to initialize is more than two-dimensional, a matrix
  of shape `(shape[0] * ... * shape[n - 2], shape[n - 1])` is initialized, where
  `n` is the length of the shape vector. The matrix is subsequently reshaped to
  give a tensor of the desired shape.

  Args:
    gain: multiplicative factor to apply to the orthogonal matrix
    dtype: The type of the output.
    seed: A Python integer. Used to create random seeds. See
      [`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed)
      for behavior.

  Returns:
    An initializer that generates orthogonal tensors

  Raises:
    ValueError: if `dtype` is not a floating point type or if `shape` has fewer than two entries.
  """
  def _initializer(shape, dtype=_assert_float_dtype(dtype), partition_info=None):
    # Check the shape
    if len(shape) < 2:
      raise ValueError('the tensor to initialize must be at least two-dimensional')
    # Flatten the input shape with the last dimension remaining its original shape so it works for conv2d
    num_rows = 1
    for dim in shape[:-1]:
      num_rows *= dim
    num_cols = shape[-1]
    flat_shape = (num_rows, num_cols)

    # Generate a random matrix
    a = random_ops.random_uniform(flat_shape, dtype=dtype, seed=seed)
    # Compute the svd
    _, u, v = linalg_ops.svd(a, full_matrices=False)
    # Pick the appropriate singular value decomposition
    if num_rows > num_cols:
      q = u
    else:
      # Tensorflow departs from numpy conventions such that we need to transpose axes here
      q = array_ops.transpose(v)
    return gain * array_ops.reshape(q, shape)

  return _initializer
