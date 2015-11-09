"""Operations for generating random numbers."""

from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import types
from tensorflow.python.framework import random_seed
from tensorflow.python.ops import common_shapes
from tensorflow.python.ops import gen_random_ops
from tensorflow.python.ops import math_ops
# pylint: disable=wildcard-import
from tensorflow.python.ops.gen_random_ops import *
# pylint: enable=wildcard-import


def _ShapeTensor(shape):
  """Convert to an int32 or int64 tensor, defaulting to int32 if empty."""
  if isinstance(shape, (tuple, list)) and not shape:
    dtype = types.int32
  else:
    dtype = None
  return ops.convert_to_tensor(shape, dtype=dtype, name="shape")

# pylint: disable=protected-access
def random_normal(shape, mean=0.0, stddev=1.0, dtype=types.float32,
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


def truncated_normal(shape, mean=0.0, stddev=1.0, dtype=types.float32,
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


def random_uniform(shape, minval=0.0, maxval=1.0,
                   dtype=types.float32, seed=None,
                   name=None):
  """Outputs random values from a uniform distribution.

  The generated values follow a uniform distribution in the range
  `[minval, maxval)`. The lower bound `minval` is included in the range, while
  the upper bound `maxval` is excluded.

  Args:
    shape: A 1-D integer Tensor or Python array. The shape of the output tensor.
    minval: A 0-D Tensor or Python value of type `dtype`. The lower bound on the
      range of random values to generate.
    maxval: A 0-D Tensor or Python value of type `dtype`. The upper bound on
      the range of random values to generate.
    dtype: The type of the output.
    seed: A Python integer. Used to create a random seed for the distribution.
      See
      [`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed)
      for behavior.
    name: A name for the operation (optional).

  Returns:
    A tensor of the specified shape filled with random uniform values.
  """
  with ops.op_scope([shape, minval, maxval], name, "random_uniform") as name:
    shape_tensor = _ShapeTensor(shape)
    min_tensor = ops.convert_to_tensor(minval, dtype=dtype, name="min")
    range_tensor = ops.convert_to_tensor(
        maxval - minval, dtype=dtype, name="range")
    seed1, seed2 = random_seed.get_seed(seed)
    rnd = gen_random_ops._random_uniform(shape_tensor, dtype,
                                         seed=seed1,
                                         seed2=seed2)
    mul = rnd * range_tensor
    value = math_ops.add(mul, min_tensor, name=name)
    return value


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


ops.NoGradient("RandomUniform")


@ops.RegisterShape("TruncatedNormal")
@ops.RegisterShape("RandomStandardNormal")
@ops.RegisterShape("RandomUniform")
def _RandomShape(op):
  shape_val = tensor_util.ConstantValue(op.inputs[0])
  if shape_val is not None:
    return [tensor_shape.TensorShape(shape_val.tolist())]
  else:
    shape_shape = op.inputs[0].get_shape().with_rank_at_most(1)
    return [tensor_shape.unknown_shape(ndims=shape_shape.num_elements())]


ops.RegisterShape("RandomShuffle")(common_shapes.unchanged_shape)
