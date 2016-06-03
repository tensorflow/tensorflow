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

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import random_seed
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import common_shapes
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_random_ops
from tensorflow.python.ops import logging_ops
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
def random_normal(shape, mean=0.0, stddev=1.0, dtype=dtypes.float32,
                  seed=None, name=None):
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
  with ops.op_scope([shape, mean, stddev], name, "random_normal") as name:
    shape_tensor = _ShapeTensor(shape)
    mean_tensor = ops.convert_to_tensor(
        mean, dtype=dtype, name="mean")
    stddev_tensor = ops.convert_to_tensor(
        stddev, dtype=dtype, name="stddev")
    seed1, seed2 = random_seed.get_seed(seed)
    rnd = gen_random_ops._random_standard_normal(shape_tensor, dtype,
                                                 seed=seed1,
                                                 seed2=seed2)
    mul = rnd * stddev_tensor
    value = math_ops.add(mul, mean_tensor, name=name)
    return value


ops.NoGradient("RandomStandardNormal")


def truncated_normal(shape, mean=0.0, stddev=1.0, dtype=dtypes.float32,
                     seed=None, name=None):
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
  with ops.op_scope([shape, mean, stddev], name, "truncated_normal") as name:
    shape_tensor = _ShapeTensor(shape)
    mean_tensor = ops.convert_to_tensor(
        mean, dtype=dtype, name="mean")
    stddev_tensor = ops.convert_to_tensor(
        stddev, dtype=dtype, name="stddev")
    seed1, seed2 = random_seed.get_seed(seed)
    rnd = gen_random_ops._truncated_normal(shape_tensor, dtype,
                                           seed=seed1,
                                           seed2=seed2)
    mul = rnd * stddev_tensor
    value = math_ops.add(mul, mean_tensor, name=name)
    return value


ops.NoGradient("TruncatedNormal")


def random_uniform(shape, minval=0, maxval=None,
                   dtype=dtypes.float32, seed=None,
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
  with ops.op_scope([shape, minval, maxval], name, "random_uniform") as name:
    shape = _ShapeTensor(shape)
    minval = ops.convert_to_tensor(minval, dtype=dtype, name="min")
    maxval = ops.convert_to_tensor(maxval, dtype=dtype, name="max")
    seed1, seed2 = random_seed.get_seed(seed)
    if dtype.is_integer:
      return gen_random_ops._random_uniform_int(shape, minval, maxval,
                                                seed=seed1, seed2=seed2,
                                                name=name)
    else:
      rnd = gen_random_ops._random_uniform(shape, dtype, seed=seed1,
                                           seed2=seed2)
      return math_ops.add(rnd * (maxval - minval), minval, name=name)


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
  return gen_random_ops._random_shuffle(value, seed=seed1, seed2=seed2,
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
  with ops.op_scope([value, size], name, "random_crop") as name:
    value = ops.convert_to_tensor(value, name="value")
    size = ops.convert_to_tensor(size, dtype=dtypes.int32, name="size")
    shape = array_ops.shape(value)
    check = logging_ops.Assert(math_ops.reduce_all(shape >= size),
                               ["Need value.shape >= size, got ", shape, size])
    shape = control_flow_ops.with_dependencies([check], shape)
    limit = shape - size + 1
    offset = random_uniform(array_ops.shape(shape), dtype=size.dtype,
                            maxval=size.dtype.max, seed=seed) % limit
    return array_ops.slice(value, offset, size, name=name)


def multinomial(logits, num_samples, seed=None, name=None):
  """Draws samples from a multinomial distribution.

  Example:

    samples = tf.multinomial(tf.log([[0.5, 0.5]]), 10)
    # samples has shape [1, 10], where each value is either 0 or 1.

    samples = tf.multinomial([[1, -1, -1]], 10)
    # samples is equivalent to tf.zeros([1, 10], dtype=tf.int64).

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
  with ops.op_scope([logits], name, "multinomial"):
    logits = ops.convert_to_tensor(logits, name="logits")
    seed1, seed2 = random_seed.get_seed(seed)
    return gen_random_ops.multinomial(logits, num_samples, seed=seed1,
                                      seed2=seed2)


@ops.RegisterShape("Multinomial")
def _MultinomialShape(op):  # pylint: disable=invalid-name
  logits_shape = op.inputs[0].get_shape().with_rank(2)
  batch_size = logits_shape[0]
  num_samples_or_none = tensor_util.constant_value(op.inputs[1])
  return [tensor_shape.matrix(batch_size, num_samples_or_none)]


ops.NoGradient("Multinomial")


ops.NoGradient("RandomUniform")


@ops.RegisterShape("TruncatedNormal")
@ops.RegisterShape("RandomStandardNormal")
@ops.RegisterShape("RandomUniform")
@ops.RegisterShape("RandomUniformInt")
def _RandomShape(op):
  shape_val = tensor_util.constant_value(op.inputs[0])
  if shape_val is not None:
    return [tensor_shape.TensorShape(shape_val)]
  else:
    shape_shape = op.inputs[0].get_shape().with_rank(1)
    return [tensor_shape.unknown_shape(ndims=shape_shape[0].value)]


ops.RegisterShape("RandomShuffle")(common_shapes.unchanged_shape)
