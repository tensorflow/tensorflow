# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Stateless random ops which take seed as a tensor input."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.ops import gen_stateless_random_ops

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export

ops.NotDifferentiable("StatelessMultinomial")
ops.NotDifferentiable("StatelessRandomNormal")
ops.NotDifferentiable("StatelessRandomUniform")
ops.NotDifferentiable("StatelessRandomUniformInt")
ops.NotDifferentiable("StatelessTruncatedNormal")


@tf_export("random.stateless_uniform")
def stateless_random_uniform(shape,
                             seed,
                             minval=0,
                             maxval=None,
                             dtype=dtypes.float32,
                             name=None):
  """Outputs deterministic pseudorandom values from a uniform distribution.

  This is a stateless version of `tf.random_uniform`: if run twice with the
  same seeds, it will produce the same pseudorandom numbers.  The output is
  consistent across multiple runs on the same hardware (and between CPU
  and GPU), but may change between versions of TensorFlow or on non-CPU/GPU
  hardware.

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
    seed: A shape [2] integer Tensor of seeds to the random number generator.
    minval: A 0-D Tensor or Python value of type `dtype`. The lower bound on the
      range of random values to generate.  Defaults to 0.
    maxval: A 0-D Tensor or Python value of type `dtype`. The upper bound on the
      range of random values to generate.  Defaults to 1 if `dtype` is floating
      point.
    dtype: The type of the output: `float16`, `float32`, `float64`, `int32`, or
      `int64`.
    name: A name for the operation (optional).

  Returns:
    A tensor of the specified shape filled with random uniform values.

  Raises:
    ValueError: If `dtype` is integral and `maxval` is not specified.
  """
  dtype = dtypes.as_dtype(dtype)
  if dtype not in (dtypes.float16, dtypes.bfloat16, dtypes.float32,
                   dtypes.float64, dtypes.int32, dtypes.int64):
    raise ValueError("Invalid dtype %r" % dtype)
  if maxval is None:
    if dtype.is_integer:
      raise ValueError("Must specify maxval for integer dtype %r" % dtype)
    maxval = 1
  with ops.name_scope(name, "stateless_random_uniform",
                      [shape, seed, minval, maxval]) as name:
    shape = random_ops._ShapeTensor(shape)  # pylint: disable=protected-access
    minval = ops.convert_to_tensor(minval, dtype=dtype, name="min")
    maxval = ops.convert_to_tensor(maxval, dtype=dtype, name="max")
    if dtype.is_integer:
      return gen_stateless_random_ops.stateless_random_uniform_int(
          shape, seed=seed, minval=minval, maxval=maxval, name=name)
    else:
      rnd = gen_stateless_random_ops.stateless_random_uniform(
          shape, seed=seed, dtype=dtype)
      return math_ops.add(rnd * (maxval - minval), minval, name=name)


@tf_export("random.stateless_normal")
def stateless_random_normal(shape,
                            seed,
                            mean=0.0,
                            stddev=1.0,
                            dtype=dtypes.float32,
                            name=None):
  """Outputs deterministic pseudorandom values from a normal distribution.

  This is a stateless version of `tf.random_normal`: if run twice with the
  same seeds, it will produce the same pseudorandom numbers.  The output is
  consistent across multiple runs on the same hardware (and between CPU
  and GPU), but may change between versions of TensorFlow or on non-CPU/GPU
  hardware.

  Args:
    shape: A 1-D integer Tensor or Python array. The shape of the output tensor.
    seed: A shape [2] integer Tensor of seeds to the random number generator.
    mean: A 0-D Tensor or Python value of type `dtype`. The mean of the normal
      distribution.
    stddev: A 0-D Tensor or Python value of type `dtype`. The standard deviation
      of the normal distribution.
    dtype: The type of the output.
    name: A name for the operation (optional).

  Returns:
    A tensor of the specified shape filled with random normal values.
  """
  with ops.name_scope(name, "stateless_random_normal",
                      [shape, seed, mean, stddev]) as name:
    shape = random_ops._ShapeTensor(shape)  # pylint: disable=protected-access
    mean = ops.convert_to_tensor(mean, dtype=dtype, name="mean")
    stddev = ops.convert_to_tensor(stddev, dtype=dtype, name="stddev")
    rnd = gen_stateless_random_ops.stateless_random_normal(shape, seed, dtype)
    return math_ops.add(rnd * stddev, mean, name=name)


@tf_export("random.stateless_truncated_normal")
def stateless_truncated_normal(shape,
                               seed,
                               mean=0.0,
                               stddev=1.0,
                               dtype=dtypes.float32,
                               name=None):
  """Outputs deterministic pseudorandom values, truncated normally distributed.

  This is a stateless version of `tf.truncated_normal`: if run twice with the
  same seeds, it will produce the same pseudorandom numbers.  The output is
  consistent across multiple runs on the same hardware (and between CPU
  and GPU), but may change between versions of TensorFlow or on non-CPU/GPU
  hardware.

  The generated values follow a normal distribution with specified mean and
  standard deviation, except that values whose magnitude is more than 2 standard
  deviations from the mean are dropped and re-picked.

  Args:
    shape: A 1-D integer Tensor or Python array. The shape of the output tensor.
    seed: A shape [2] integer Tensor of seeds to the random number generator.
    mean: A 0-D Tensor or Python value of type `dtype`. The mean of the
      truncated normal distribution.
    stddev: A 0-D Tensor or Python value of type `dtype`. The standard deviation
      of the normal distribution, before truncation.
    dtype: The type of the output.
    name: A name for the operation (optional).

  Returns:
    A tensor of the specified shape filled with random truncated normal values.
  """
  with ops.name_scope(name, "stateless_truncated_normal",
                      [shape, seed, mean, stddev]) as name:
    shape = random_ops._ShapeTensor(shape)  # pylint: disable=protected-access
    mean = ops.convert_to_tensor(mean, dtype=dtype, name="mean")
    stddev = ops.convert_to_tensor(stddev, dtype=dtype, name="stddev")
    rnd = gen_stateless_random_ops.stateless_truncated_normal(
        shape, seed, dtype)
    return math_ops.add(rnd * stddev, mean, name=name)


@tf_export(v1=["random.stateless_multinomial"])
@deprecation.deprecated(
    date=None, instructions="Use tf.random.stateless_categorical instead.")
def stateless_multinomial(logits,
                          num_samples,
                          seed,
                          output_dtype=dtypes.int64,
                          name=None):
  """Draws deterministic pseudorandom samples from a multinomial distribution.

  This is a stateless version of `tf.multinomial`: if run twice with the
  same seeds, it will produce the same pseudorandom numbers.  The output is
  consistent across multiple runs on the same hardware (and between CPU
  and GPU), but may change between versions of TensorFlow or on non-CPU/GPU
  hardware.

  Example:

  ```python
  # samples has shape [1, 5], where each value is either 0 or 1 with equal
  # probability.
  samples = tf.random.stateless_multinomial(
      tf.log([[10., 10.]]), 5, seed=[7, 17])
  ```

  Args:
    logits: 2-D Tensor with shape `[batch_size, num_classes]`.  Each slice
      `[i, :]` represents the unnormalized log-probabilities for all classes.
    num_samples: 0-D.  Number of independent samples to draw for each row slice.
    seed: A shape [2] integer Tensor of seeds to the random number generator.
    output_dtype: integer type to use for the output. Defaults to int64.
    name: Optional name for the operation.

  Returns:
    The drawn samples of shape `[batch_size, num_samples]`.
  """
  with ops.name_scope(name, "stateless_multinomial", [logits, seed]):
    return stateless_multinomial_categorical_impl(logits, num_samples,
                                                  output_dtype, seed)


@tf_export("random.stateless_categorical")
def stateless_categorical(logits,
                          num_samples,
                          seed,
                          dtype=dtypes.int64,
                          name=None):
  """Draws deterministic pseudorandom samples from a categorical distribution.

  This is a stateless version of `tf.categorical`: if run twice with the
  same seeds, it will produce the same pseudorandom numbers.  The output is
  consistent across multiple runs on the same hardware (and between CPU
  and GPU), but may change between versions of TensorFlow or on non-CPU/GPU
  hardware.

  Example:

  ```python
  # samples has shape [1, 5], where each value is either 0 or 1 with equal
  # probability.
  samples = tf.random.stateless_categorical(
      tf.log([[10., 10.]]), 5, seed=[7, 17])
  ```

  Args:
    logits: 2-D Tensor with shape `[batch_size, num_classes]`.  Each slice
      `[i, :]` represents the unnormalized log-probabilities for all classes.
    num_samples: 0-D.  Number of independent samples to draw for each row slice.
    seed: A shape [2] integer Tensor of seeds to the random number generator.
    dtype: integer type to use for the output. Defaults to int64.
    name: Optional name for the operation.

  Returns:
    The drawn samples of shape `[batch_size, num_samples]`.
  """
  with ops.name_scope(name, "stateless_categorical", [logits, seed]):
    return stateless_multinomial_categorical_impl(logits, num_samples, dtype,
                                                  seed)


def stateless_multinomial_categorical_impl(logits, num_samples, dtype, seed):
  """Implementation for stateless multinomial/categorical ops (v1/v2)."""
  logits = ops.convert_to_tensor(logits, name="logits")
  return gen_stateless_random_ops.stateless_multinomial(
      logits, num_samples, seed, output_dtype=dtype)
