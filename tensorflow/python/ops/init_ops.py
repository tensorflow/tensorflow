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

"""Operations often used for initializing tensors."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import constant_op
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops


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


def constant_initializer(value=0.0, dtype=dtypes.float32):
  """Returns an initializer that generates tensors with a single value.

  Args:
    value: A Python scalar. All elements of the initialized variable
      will be set to this value.
    dtype: The data type. Only floating point types are supported.

  Returns:
    An initializer that generates tensors with a single value.

  Raises:
    ValueError: if `dtype` is not a floating point type.
  """
  def _initializer(shape, dtype=_assert_float_dtype(dtype)):
    return constant_op.constant(value, dtype=dtype, shape=shape)
  return _initializer


def random_uniform_initializer(minval=0.0, maxval=1.0, seed=None,
                               dtype=dtypes.float32):
  """Returns an initializer that generates tensors with a uniform distribution.

  Args:
    minval: a python scalar or a scalar tensor. lower bound of the range
      of random values to generate.
    maxval: a python scalar or a scalar tensor. upper bound of the range
      of random values to generate.
    seed: A Python integer. Used to create random seeds. See
      [`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed)
      for behavior.
    dtype: The data type. Only floating point types are supported.

  Returns:
    An initializer that generates tensors with a uniform distribution.

  Raises:
    ValueError: if `dtype` is not a floating point type.
  """
  def _initializer(shape, dtype=_assert_float_dtype(dtype)):
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
  def _initializer(shape, dtype=_assert_float_dtype(dtype)):
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
  def _initializer(shape, dtype=_assert_float_dtype(dtype)):
    return random_ops.truncated_normal(shape, mean, stddev, dtype, seed=seed)
  return _initializer


def uniform_unit_scaling_initializer(factor=1.0, seed=None,
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
  def _initializer(shape, dtype=_assert_float_dtype(dtype)):
    input_size = 1.0
    # Estimating input size is not possible to do perfectly, but we try.
    # The estimate, obtained by multiplying all dimensions but the last one,
    # is the right thing for matrix multiply and convolutions (see above).
    for dim in shape[:-1]:
      input_size *= float(dim)
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

  def __call__(self, shape, dtype=dtypes.float32):
    """Generate a tensor used to initialize a variable."""
    return random_ops._random_walk(shape, self._nonlinearity, dtype,
                                   seed=self._seed)
