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
"""Operations for generating random numbers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import common_shapes
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_random_ops
from tensorflow.python.ops import math_ops
# go/tf-wildcard-import
# pylint: disable=wildcard-import
from tensorflow.python.ops.gen_random_ops import *

# pylint: enable=wildcard-import


def _ShapeTensor(shape):
  """Convert to an int32 or int64 tensor, defaulting to int32 if empty."""
  if isinstance(shape, (tuple, list)) and not shape:
    dtype = dtypes.int32
  else:
    dtype = None
  return ops.convert_to_tensor(shape, dtype=dtype, name="shape")


# pylint: disable=protected-access
def random_normal(shape,
                  mean=0.0,
                  stddev=1.0,
                  dtype=dtypes.float32,
                  seed=None,
                  name=None):
  """Outputs random values from a normal distribution.

  Args:
    shape: A 1-D integer Tensor or Python array. The shape of the output tensor.
    mean: A 0-D Tensor or Python value of type `dtype`. The mean of the normal
      distribution.
    stddev: A 0-D Tensor or Python value of type `dtype`. The standard deviation
      of the normal distribution.
    dtype: The type of the output.
    seed: A Python integer. Used to create a random seed for the distribution.
      See
      [`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed)
      for behavior.
    name: A name for the operation (optional).

  Returns:
    A tensor of the specified shape filled with random normal values.
  """
  with ops.name_scope(name, "random_normal", [shape, mean, stddev]) as name:
    shape_tensor = _ShapeTensor(shape)
    mean_tensor = ops.convert_to_tensor(mean, dtype=dtype, name="mean")
    stddev_tensor = ops.convert_to_tensor(stddev, dtype=dtype, name="stddev")
    seed1, seed2 = random_seed.get_seed(seed)
    rnd = gen_random_ops._random_standard_normal(shape_tensor,
                                                 dtype,
                                                 seed=seed1,
                                                 seed2=seed2)
    mul = rnd * stddev_tensor
    value = math_ops.add(mul, mean_tensor, name=name)
    return value


ops.NotDifferentiable("RandomStandardNormal")


def parameterized_truncated_normal(shape,
                                   means=0.0,
                                   stddevs=1.0,
                                   minvals=-2.0,
                                   maxvals=2.0,
                                   dtype=dtypes.float32,
                                   seed=None,
                                   name=None):
  """Outputs random values from a truncated normal distribution.

  The generated values follow a normal distribution with specified mean and
  standard deviation, except that values whose magnitude is more than 2 standard
  deviations from the mean are dropped and re-picked.

  Args:
    shape: A 1-D integer Tensor or Python array. The shape of the output tensor.
    means: A 0-D Tensor or Python value of type `dtype`. The mean of the
      truncated normal distribution.
    stddevs: A 0-D Tensor or Python value of type `dtype`. The standard
      deviation of the truncated normal distribution.
    minvals: A 0-D Tensor or Python value of type `dtype`. The minimum value of
      the truncated normal distribution.
    maxvals: A 0-D Tensor or Python value of type `dtype`. The maximum value of
      the truncated normal distribution.
    dtype: The type of the output.
    seed: A Python integer. Used to create a random seed for the distribution.
      See
      [`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed)
      for behavior.
    name: A name for the operation (optional).

  Returns:
    A tensor of the specified shape filled with random truncated normal values.
  """
  with ops.name_scope(name, "parameterized_truncated_normal",
                      [shape, means, stddevs, minvals, maxvals]) as name:
    shape_tensor = _ShapeTensor(shape)
    means_tensor = ops.convert_to_tensor(means, dtype=dtype, name="means")
    stddevs_tensor = ops.convert_to_tensor(stddevs, dtype=dtype, name="stddevs")
    minvals_tensor = ops.convert_to_tensor(minvals, dtype=dtype, name="minvals")
    maxvals_tensor = ops.convert_to_tensor(maxvals, dtype=dtype, name="maxvals")
    seed1, seed2 = random_seed.get_seed(seed)
    rnd = gen_random_ops._parameterized_truncated_normal(shape_tensor,
                                                         means_tensor,
                                                         stddevs_tensor,
                                                         minvals_tensor,
                                                         maxvals_tensor,
                                                         seed=seed1,
                                                         seed2=seed2)
    return rnd


def truncated_normal(shape,
                     mean=0.0,
                     stddev=1.0,
                     dtype=dtypes.float32,
                     seed=None,
                     name=None):
  """Outputs random values from a truncated normal distribution.

  The generated values follow a normal distribution with specified mean and
  standard deviation, except that values whose magnitude is more than 2 standard
  deviations from the mean are dropped and re-picked.

  Args:
    shape: A 1-D integer Tensor or Python array. The shape of the output tensor.
    mean: A 0-D Tensor or Python value of type `dtype`. The mean of the
      truncated normal distribution.
    stddev: A 0-D Tensor or Python value of type `dtype`. The standard deviation
      of the truncated normal distribution.
    dtype: The type of the output.
    seed: A Python integer. Used to create a random seed for the distribution.
      See
      [`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed)
      for behavior.
    name: A name for the operation (optional).

  Returns:
    A tensor of the specified shape filled with random truncated normal values.
  """
  with ops.name_scope(name, "truncated_normal", [shape, mean, stddev]) as name:
    shape_tensor = _ShapeTensor(shape)
    mean_tensor = ops.convert_to_tensor(mean, dtype=dtype, name="mean")
    stddev_tensor = ops.convert_to_tensor(stddev, dtype=dtype, name="stddev")
    seed1, seed2 = random_seed.get_seed(seed)
    rnd = gen_random_ops._truncated_normal(shape_tensor,
                                           dtype,
                                           seed=seed1,
                                           seed2=seed2)
    mul = rnd * stddev_tensor
    value = math_ops.add(mul, mean_tensor, name=name)
    return value


ops.NotDifferentiable("ParameterizedTruncatedNormal")
ops.NotDifferentiable("TruncatedNormal")


def random_uniform(shape,
                   minval=0,
                   maxval=None,
                   dtype=dtypes.float32,
                   seed=None,
                   name=None):
  """Outputs random values from a uniform distribution.

  The generated values follow a uniform distribution in the range
  `[minval, maxval)`. The lower bound `minval` is included in the range, while
  the upper bound `maxval` is excluded.

  For floats, the default range is `[0, 1)`.  For ints, at least `maxval` must
  be specified explicitly.

  In the integer case, the random integers are slightly biased unless
  `maxval - minval` is an exact power of two.  The bias is small for values of
  `maxval - minval` significantly smaller than the range of the output (either
  `2**32` or `2**64`).

  Args:
    shape: A 1-D integer Tensor or Python array. The shape of the output tensor.
    minval: A 0-D Tensor or Python value of type `dtype`. The lower bound on the
      range of random values to generate.  Defaults to 0.
    maxval: A 0-D Tensor or Python value of type `dtype`. The upper bound on
      the range of random values to generate.  Defaults to 1 if `dtype` is
      floating point.
    dtype: The type of the output: `float32`, `float64`, `int32`, or `int64`.
    seed: A Python integer. Used to create a random seed for the distribution.
      See
      [`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed)
      for behavior.
    name: A name for the operation (optional).

  Returns:
    A tensor of the specified shape filled with random uniform values.

  Raises:
    ValueError: If `dtype` is integral and `maxval` is not specified.
  """
  dtype = dtypes.as_dtype(dtype)
  if maxval is None:
    if dtype.is_integer:
      raise ValueError("Must specify maxval for integer dtype %r" % dtype)
    maxval = 1
  with ops.name_scope(name, "random_uniform", [shape, minval, maxval]) as name:
    shape = _ShapeTensor(shape)
    minval = ops.convert_to_tensor(minval, dtype=dtype, name="min")
    maxval = ops.convert_to_tensor(maxval, dtype=dtype, name="max")
    seed1, seed2 = random_seed.get_seed(seed)
    if dtype.is_integer:
      return gen_random_ops._random_uniform_int(shape,
                                                minval,
                                                maxval,
                                                seed=seed1,
                                                seed2=seed2,
                                                name=name)
    else:
      rnd = gen_random_ops._random_uniform(shape,
                                           dtype,
                                           seed=seed1,
                                           seed2=seed2)
      return math_ops.add(rnd * (maxval - minval), minval, name=name)


ops.NotDifferentiable("RandomUniform")


def random_shuffle(value, seed=None, name=None):
  """Randomly shuffles a tensor along its first dimension.

  The tensor is shuffled along dimension 0, such that each `value[j]` is mapped
  to one and only one `output[i]`. For example, a mapping that might occur for a
  3x2 tensor is:

  ```python
  [[1, 2],       [[5, 6],
   [3, 4],  ==>   [1, 2],
   [5, 6]]        [3, 4]]
  ```

  Args:
    value: A Tensor to be shuffled.
    seed: A Python integer. Used to create a random seed for the distribution.
      See
      [`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed)
      for behavior.
    name: A name for the operation (optional).

  Returns:
    A tensor of same shape and type as `value`, shuffled along its first
    dimension.
  """
  seed1, seed2 = random_seed.get_seed(seed)
  return gen_random_ops._random_shuffle(value,
                                        seed=seed1,
                                        seed2=seed2,
                                        name=name)


def random_crop(value, size, seed=None, name=None):
  """Randomly crops a tensor to a given size.

  Slices a shape `size` portion out of `value` at a uniformly chosen offset.
  Requires `value.shape >= size`.

  If a dimension should not be cropped, pass the full size of that dimension.
  For example, RGB images can be cropped with
  `size = [crop_height, crop_width, 3]`.

  Args:
    value: Input tensor to crop.
    size: 1-D tensor with size the rank of `value`.
    seed: Python integer. Used to create a random seed. See
      [`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed)
      for behavior.
    name: A name for this operation (optional).

  Returns:
    A cropped tensor of the same rank as `value` and shape `size`.
  """
  # TODO(shlens): Implement edge case to guarantee output size dimensions.
  # If size > value.shape, zero pad the result so that it always has shape
  # exactly size.
  with ops.name_scope(name, "random_crop", [value, size]) as name:
    value = ops.convert_to_tensor(value, name="value")
    size = ops.convert_to_tensor(size, dtype=dtypes.int32, name="size")
    shape = array_ops.shape(value)
    check = control_flow_ops.Assert(
        math_ops.reduce_all(shape >= size),
        ["Need value.shape >= size, got ", shape, size])
    shape = control_flow_ops.with_dependencies([check], shape)
    limit = shape - size + 1
    offset = random_uniform(
        array_ops.shape(shape),
        dtype=size.dtype,
        maxval=size.dtype.max,
        seed=seed) % limit
    return array_ops.slice(value, offset, size, name=name)


def multinomial(logits, num_samples, seed=None, name=None):
  """Draws samples from a multinomial distribution.

  Example:

  ```python
  # samples has shape [1, 5], where each value is either 0 or 1 with equal
  # probability.
  samples = tf.multinomial(tf.log([[10., 10.]]), 5)
  ```

  Args:
    logits: 2-D Tensor with shape `[batch_size, num_classes]`.  Each slice
      `[i, :]` represents the unnormalized log probabilities for all classes.
    num_samples: 0-D.  Number of independent samples to draw for each row slice.
    seed: A Python integer. Used to create a random seed for the distribution.
      See
      [`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed)
      for behavior.
    name: Optional name for the operation.

  Returns:
    The drawn samples of shape `[batch_size, num_samples]`.
  """
  with ops.name_scope(name, "multinomial", [logits]):
    logits = ops.convert_to_tensor(logits, name="logits")
    seed1, seed2 = random_seed.get_seed(seed)
    return gen_random_ops.multinomial(logits,
                                      num_samples,
                                      seed=seed1,
                                      seed2=seed2)


@ops.RegisterShape("Multinomial")
def _MultinomialShape(op):
  return common_shapes.call_cpp_shape_fn(op, input_tensors_needed=[1])


ops.NotDifferentiable("Multinomial")


def random_gamma(shape,
                 alpha,
                 beta=None,
                 dtype=dtypes.float32,
                 seed=None,
                 name=None):
  """Draws `shape` samples from each of the given Gamma distribution(s).

  `alpha` is the shape parameter describing the distribution(s), and `beta` is
  the inverse scale parameter(s).

  Example:

    samples = tf.random_gamma([10], [0.5, 1.5])
    # samples has shape [10, 2], where each slice [:, 0] and [:, 1] represents
    # the samples drawn from each distribution

    samples = tf.random_gamma([7, 5], [0.5, 1.5])
    # samples has shape [7, 5, 2], where each slice [:, :, 0] and [:, :, 1]
    # represents the 7x5 samples drawn from each of the two distributions

    samples = tf.random_gamma([30], [[1.],[3.],[5.]], beta=[[3., 4.]])
    # samples has shape [30, 3, 2], with 30 samples each of 3x2 distributions.

    Note that for small alpha values, there is a chance you will draw a value of
    exactly 0, which gets worse for lower-precision dtypes, even though zero is
    not in the support of the gamma distribution.

    Relevant cdfs (~chance you will draw a exactly-0 value):
    ```
      stats.gamma(.01).cdf(np.finfo(np.float16).tiny)
          0.91269738769897879
      stats.gamma(.01).cdf(np.finfo(np.float32).tiny)
          0.41992668622045726
      stats.gamma(.01).cdf(np.finfo(np.float64).tiny)
          0.00084322740680686662
      stats.gamma(.35).cdf(np.finfo(np.float16).tiny)
          0.037583276135263931
      stats.gamma(.35).cdf(np.finfo(np.float32).tiny)
          5.9514895726818067e-14
      stats.gamma(.35).cdf(np.finfo(np.float64).tiny)
          2.3529843400647272e-108
    ```

  Args:
    shape: A 1-D integer Tensor or Python array. The shape of the output samples
      to be drawn per alpha/beta-parameterized distribution.
    alpha: A Tensor or Python value or N-D array of type `dtype`. `alpha`
      provides the shape parameter(s) describing the gamma distribution(s) to
      sample. Must be broadcastable with `beta`.
    beta: A Tensor or Python value or N-D array of type `dtype`. Defaults to 1.
      `beta` provides the inverse scale parameter(s) of the gamma
      distribution(s) to sample. Must be broadcastable with `alpha`.
    dtype: The type of alpha, beta, and the output: `float16`, `float32`, or
      `float64`.
    seed: A Python integer. Used to create a random seed for the distributions.
      See
      [`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed)
      for behavior.
    name: Optional name for the operation.

  Returns:
    samples: a `Tensor` of shape `tf.concat(shape, tf.shape(alpha + beta))` with
      values of type `dtype`.
  """
  with ops.name_scope(name, "random_gamma", [shape, alpha, beta]):
    shape = ops.convert_to_tensor(shape, name="shape", dtype=dtypes.int32)
    alpha = ops.convert_to_tensor(alpha, name="alpha", dtype=dtype)
    beta = ops.convert_to_tensor(beta if beta is not None else 1,
                                 name="beta",
                                 dtype=dtype)
    alpha_broadcast = alpha + array_ops.zeros_like(beta)
    seed1, seed2 = random_seed.get_seed(seed)
    return gen_random_ops._random_gamma(shape,
                                        alpha_broadcast,
                                        seed=seed1,
                                        seed2=seed2) / beta


@ops.RegisterShape("RandomGamma")
def _RandomGammaShape(op):
  return common_shapes.call_cpp_shape_fn(op, input_tensors_needed=[0])


ops.NotDifferentiable("RandomGamma")


@ops.RegisterShape("ParameterizedTruncatedNormal")
@ops.RegisterShape("TruncatedNormal")
@ops.RegisterShape("RandomStandardNormal")
@ops.RegisterShape("RandomUniform")
@ops.RegisterShape("RandomUniformInt")
def _RandomShape(op):
  return common_shapes.call_cpp_shape_fn(op, input_tensors_needed=[0])


ops.RegisterShape("RandomShuffle")(common_shapes.unchanged_shape)
