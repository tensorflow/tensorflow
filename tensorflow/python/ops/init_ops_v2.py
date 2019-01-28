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

def _initializer(shape, dtype=dtypes.float32):
  Args:
    shape: List of `int` representing the shape of the output `Tensor`. Some
      initializers may also be able to accept a `Tensor`.
    dtype: (Optional) Type of the output `Tensor`.
  Returns:
    A `Tensor` of type `dtype` and `shape`.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_linalg_ops
from tensorflow.python.ops import linalg_ops_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.util.tf_export import tf_export


class Initializer(object):
  """Initializer base class: all initializers inherit from this class.
  """

  def __call__(self, shape, dtype=None):
    """Returns a tensor object initialized as specified by the initializer.

    Args:
      shape: Shape of the tensor.
      dtype: Optional dtype of the tensor. If not provided will return tensor
       of `tf.float32`.
    """
    raise NotImplementedError

  def get_config(self):
    """Returns the configuration of the initializer as a JSON-serializable dict.

    Returns:
      A JSON-serializable Python dict.
    """
    return {}

  @classmethod
  def from_config(cls, config):
    """Instantiates an initializer from a configuration dictionary.

    Example:

    ```python
    initializer = RandomUniform(-1, 1)
    config = initializer.get_config()
    initializer = RandomUniform.from_config(config)
    ```

    Args:
      config: A Python dictionary.
        It will typically be the output of `get_config`.

    Returns:
      An Initializer instance.
    """
    config.pop("dtype", None)
    return cls(**config)


@tf_export("zeros_initializer", v1=[])
class Zeros(Initializer):
  """Initializer that generates tensors initialized to 0."""

  def __call__(self, shape, dtype=dtypes.float32):
    dtype = dtypes.as_dtype(dtype)
    return array_ops.zeros(shape, dtype)


@tf_export("ones_initializer", v1=[])
class Ones(Initializer):
  """Initializer that generates tensors initialized to 1."""

  def __call__(self, shape, dtype=dtypes.float32):
    """Returns a tensor object initialized as specified by the initializer.

    Args:
      shape: Shape of the tensor.
      dtype: Optional dtype of the tensor. Only numeric or boolean dtypes are
       supported.

    Raises:
      ValuesError: If the dtype is not numeric or boolean.
    """
    dtype = dtypes.as_dtype(dtype)
    if not dtype.is_numpy_compatible or dtype == dtypes.string:
      raise ValueError("Expected numeric or boolean dtype, got %s." % dtype)
    return array_ops.ones(shape, dtype)


@tf_export("constant_initializer", v1=[])
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
    value: A Python scalar, list or tuple of values, or a N-dimensional numpy
      array. All elements of the initialized variable will be set to the
      corresponding value in the `value` argument.

  Raises:
    TypeError: If the input `value` is not one of the expected types.

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
  ```
  """

  def __init__(self, value=0):
    if not (np.isscalar(value) or isinstance(value, (list, tuple, np.ndarray))):
      raise TypeError(
          "Invalid type for initial value: %s (expected Python scalar, list or "
          "tuple of values, or numpy.ndarray)." % type(value))
    self.value = value

  def __call__(self, shape, dtype=None):
    """Returns a tensor object initialized as specified by the initializer.

    Args:
      shape: Shape of the tensor.
      dtype: Optional dtype of the tensor. If not provided the dtype of the
       tensor created will be the type of the inital value.

    Raises:
      TypeError: If the initializer cannot create a tensor of the requested
       dtype.
    """
    if dtype is not None:
      dtype = dtypes.as_dtype(dtype)
    return constant_op.constant(
        self.value, dtype=dtype, shape=shape)

  def get_config(self):
    return {"value": self.value}


@tf_export("random_uniform_initializer", v1=[])
class RandomUniform(Initializer):
  """Initializer that generates tensors with a uniform distribution.

  Args:
    minval: A python scalar or a scalar tensor. Lower bound of the range
      of random values to generate.
    maxval: A python scalar or a scalar tensor. Upper bound of the range
      of random values to generate.  Defaults to 1 for float types.
    seed: A Python integer. Used to create random seeds. See
      `tf.set_random_seed`
      for behavior.
  """

  def __init__(self, minval=-0.05, maxval=0.05, seed=None):
    self.minval = minval
    self.maxval = maxval
    self.seed = seed

  def __call__(self, shape, dtype=dtypes.float32):
    """Returns a tensor object initialized as specified by the initializer.

    Args:
      shape: Shape of the tensor.
      dtype: Optional dtype of the tensor. Only floating point and integer
      types are supported.

    Raises:
      ValueError: If the dtype is not numeric.
    """
    dtype = dtypes.as_dtype(dtype)
    if not dtype.is_floating and not dtype.is_integer:
      raise ValueError("Expected float or integer dtype, got %s." % dtype)
    return random_ops.random_uniform(
        shape, self.minval, self.maxval, dtype, seed=self.seed)

  def get_config(self):
    return {
        "minval": self.minval,
        "maxval": self.maxval,
        "seed": self.seed
    }


@tf_export("random_normal_initializer", v1=[])
class RandomNormal(Initializer):
  """Initializer that generates tensors with a normal distribution.

  Args:
    mean: a python scalar or a scalar tensor. Mean of the random values
      to generate.
    stddev: a python scalar or a scalar tensor. Standard deviation of the
      random values to generate.
    seed: A Python integer. Used to create random seeds. See
      `tf.set_random_seed`
      for behavior.
  """

  def __init__(self, mean=0.0, stddev=0.05, seed=None):
    self.mean = mean
    self.stddev = stddev
    self.seed = seed

  def __call__(self, shape, dtype=dtypes.float32):
    """Returns a tensor object initialized as specified by the initializer.

    Args:
      shape: Shape of the tensor.
      dtype: Optional dtype of the tensor. Only floating point types are
       supported.

    Raises:
      ValueError: If the dtype is not floating point
    """
    dtype = _assert_float_dtype(dtype)
    return random_ops.random_normal(
        shape, self.mean, self.stddev, dtype, seed=self.seed)

  def get_config(self):
    return {
        "mean": self.mean,
        "stddev": self.stddev,
        "seed": self.seed
    }


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
      `tf.set_random_seed`
      for behavior.
  """

  def __init__(self, mean=0.0, stddev=0.05, seed=None):
    self.mean = mean
    self.stddev = stddev
    self.seed = seed

  def __call__(self, shape, dtype=dtypes.float32):
    """Returns a tensor object initialized as specified by the initializer.

    Args:
      shape: Shape of the tensor.
      dtype: Optional dtype of the tensor. Only floating point types are
       supported.

    Raises:
      ValueError: If the dtype is not floating point
    """
    dtype = _assert_float_dtype(dtype)
    return random_ops.truncated_normal(
        shape, self.mean, self.stddev, dtype, seed=self.seed)

  def get_config(self):
    return {
        "mean": self.mean,
        "stddev": self.stddev,
        "seed": self.seed
    }


class VarianceScaling(Initializer):
  """Initializer capable of adapting its scale to the shape of weights tensors.

  With `distribution="truncated_normal" or "untruncated_normal"`,
  samples are drawn from a truncated/untruncated normal
  distribution with a mean of zero and a standard deviation (after truncation,
  if used) `stddev = sqrt(scale / n)`
  where n is:
    - number of input units in the weight tensor, if mode = "fan_in"
    - number of output units, if mode = "fan_out"
    - average of the numbers of input and output units, if mode = "fan_avg"

  With `distribution="uniform"`, samples are drawn from a uniform distribution
  within [-limit, limit], with `limit = sqrt(3 * scale / n)`.

  Args:
    scale: Scaling factor (positive float).
    mode: One of "fan_in", "fan_out", "fan_avg".
    distribution: Random distribution to use. One of "truncated_normal",
      "untruncated_normal" and  "uniform".
    seed: A Python integer. Used to create random seeds. See
      `tf.set_random_seed`
      for behavior.

  Raises:
    ValueError: In case of an invalid value for the "scale", mode" or
      "distribution" arguments.
  """

  def __init__(self,
               scale=1.0,
               mode="fan_in",
               distribution="truncated_normal",
               seed=None):
    if scale <= 0.:
      raise ValueError("`scale` must be positive float.")
    if mode not in {"fan_in", "fan_out", "fan_avg"}:
      raise ValueError("Invalid `mode` argument:", mode)
    distribution = distribution.lower()
    if distribution not in {"uniform", "truncated_normal",
                            "untruncated_normal"}:
      raise ValueError("Invalid `distribution` argument:", distribution)
    self.scale = scale
    self.mode = mode
    self.distribution = distribution
    self.seed = seed

  def __call__(self, shape, dtype=dtypes.float32):
    """Returns a tensor object initialized as specified by the initializer.

    Args:
      shape: Shape of the tensor.
      dtype: Optional dtype of the tensor. Only floating point types are
       supported.

    Raises:
      ValueError: If the dtype is not floating point
    """
    partition_info = None  # Keeps logic so can be readded later if necessary
    dtype = _assert_float_dtype(dtype)
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
    if self.distribution == "truncated_normal":
      # constant from scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
      stddev = math.sqrt(scale) / .87962566103423978
      return random_ops.truncated_normal(
          shape, 0.0, stddev, dtype, seed=self.seed)
    elif self.distribution == "untruncated_normal":
      stddev = math.sqrt(scale)
      return random_ops.random_normal(
          shape, 0.0, stddev, dtype, seed=self.seed)
    else:
      limit = math.sqrt(3.0 * scale)
      return random_ops.random_uniform(
          shape, -limit, limit, dtype, seed=self.seed)

  def get_config(self):
    return {
        "scale": self.scale,
        "mode": self.mode,
        "distribution": self.distribution,
        "seed": self.seed
    }


class Orthogonal(Initializer):
  """Initializer that generates an orthogonal matrix.

  If the shape of the tensor to initialize is two-dimensional, it is initialized
  with an orthogonal matrix obtained from the QR decomposition of a matrix of
  random numbers drawn from a normal distribution.
  If the matrix has fewer rows than columns then the output will have orthogonal
  rows. Otherwise, the output will have orthogonal columns.

  If the shape of the tensor to initialize is more than two-dimensional,
  a matrix of shape `(shape[0] * ... * shape[n - 2], shape[n - 1])`
  is initialized, where `n` is the length of the shape vector.
  The matrix is subsequently reshaped to give a tensor of the desired shape.

  Args:
    gain: multiplicative factor to apply to the orthogonal matrix
    seed: A Python integer. Used to create random seeds. See
      `tf.set_random_seed`
    for behavior.

  References:
      [Saxe et al., 2014](https://openreview.net/forum?id=_wzZwKpTDF_9C)
      ([pdf](https://arxiv.org/pdf/1312.6120.pdf))
  """

  def __init__(self, gain=1.0, seed=None):
    self.gain = gain
    self.seed = seed

  def __call__(self, shape, dtype=dtypes.float32):
    """Returns a tensor object initialized as specified by the initializer.

    Args:
      shape: Shape of the tensor.
      dtype: Optional dtype of the tensor. Only floating point types are
       supported.

    Raises:
      ValueError: If the dtype is not floating point or the input shape is not
       valid.
    """
    dtype = _assert_float_dtype(dtype)
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
    flat_shape = (max(num_cols, num_rows), min(num_cols, num_rows))

    # Generate a random matrix
    a = random_ops.random_normal(flat_shape, dtype=dtype, seed=self.seed)
    # Compute the qr factorization
    q, r = gen_linalg_ops.qr(a, full_matrices=False)
    # Make Q uniform
    d = array_ops.diag_part(r)
    q *= math_ops.sign(d)
    if num_rows < num_cols:
      q = array_ops.matrix_transpose(q)
    return self.gain * array_ops.reshape(q, shape)

  def get_config(self):
    return {"gain": self.gain, "seed": self.seed}


class Identity(Initializer):
  """Initializer that generates the identity matrix.

  Only use for 2D matrices.

  Args:
    gain: Multiplicative factor to apply to the identity matrix.
  """

  def __init__(self, gain=1.0):
    self.gain = gain

  def __call__(self, shape, dtype=dtypes.float32):
    """Returns a tensor object initialized as specified by the initializer.

    Args:
      shape: Shape of the tensor.
      dtype: Optional dtype of the tensor. Only floating point types are
       supported.

    Raises:
      ValueError: If the dtype is not floating point
    """
    partition_info = None  # Keeps logic so can be readded later if necessary
    dtype = _assert_float_dtype(dtype)
    full_shape = shape if partition_info is None else partition_info.full_shape
    if len(full_shape) != 2:
      raise ValueError(
          "Identity matrix initializer can only be used for 2D matrices.")
    initializer = linalg_ops_impl.eye(*full_shape, dtype=dtype)
    if partition_info is not None:
      initializer = array_ops.slice(initializer, partition_info.var_offset,
                                    shape)
    return self.gain * initializer

  def get_config(self):
    return {"gain": self.gain}


class GlorotUniform(VarianceScaling):
  """The Glorot uniform initializer, also called Xavier uniform initializer.

  It draws samples from a uniform distribution within [-limit, limit]
  where `limit` is `sqrt(6 / (fan_in + fan_out))`
  where `fan_in` is the number of input units in the weight tensor
  and `fan_out` is the number of output units in the weight tensor.

  Args:
    seed: A Python integer. Used to create random seeds. See
      `tf.set_random_seed`
      for behavior.

  References:
      [Glorot et al., 2010](http://proceedings.mlr.press/v9/glorot10a.html)
      ([pdf](http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf))
  """

  def __init__(self, seed=None):
    super(GlorotUniform, self).__init__(
        scale=1.0,
        mode="fan_avg",
        distribution="uniform",
        seed=seed)

  def get_config(self):
    return {"seed": self.seed}


class GlorotNormal(VarianceScaling):
  """The Glorot normal initializer, also called Xavier normal initializer.

  It draws samples from a truncated normal distribution centered on 0
  with `stddev = sqrt(2 / (fan_in + fan_out))`
  where `fan_in` is the number of input units in the weight tensor
  and `fan_out` is the number of output units in the weight tensor.

  Args:
    seed: A Python integer. Used to create random seeds. See
      `tf.set_random_seed` for behavior.

  References:
      [Glorot et al., 2010](http://proceedings.mlr.press/v9/glorot10a.html)
      ([pdf](http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf))
  """

  def __init__(self, seed=None):
    super(GlorotNormal, self).__init__(
        scale=1.0,
        mode="fan_avg",
        distribution="truncated_normal",
        seed=seed)

  def get_config(self):
    return {"seed": self.seed, "dtype": self.dtype.name}


# Aliases.

# pylint: disable=invalid-name
zeros_initializer = Zeros
ones_initializer = Ones
constant_initializer = Constant
random_uniform_initializer = RandomUniform
random_normal_initializer = RandomNormal
truncated_normal_initializer = TruncatedNormal
variance_scaling_initializer = VarianceScaling
glorot_uniform_initializer = GlorotUniform
glorot_normal_initializer = GlorotNormal
orthogonal_initializer = Orthogonal
identity_initializer = Identity
# pylint: enable=invalid-name


def lecun_normal(seed=None):
  """LeCun normal initializer.

  It draws samples from a truncated normal distribution centered on 0
  with `stddev = sqrt(1 / fan_in)`
  where `fan_in` is the number of input units in the weight tensor.

  Arguments:
      seed: A Python integer. Used to seed the random generator.

  Returns:
      An initializer.

  References:
      - Self-Normalizing Neural Networks,
      [Klambauer et al., 2017]
      (https://papers.nips.cc/paper/6698-self-normalizing-neural-networks)
      ([pdf]
      (https://papers.nips.cc/paper/6698-self-normalizing-neural-networks.pdf))
      - Efficient Backprop,
      [Lecun et al., 1998](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf)
  """
  return VarianceScaling(
      scale=1., mode="fan_in", distribution="truncated_normal", seed=seed)


def lecun_uniform(seed=None):
  """LeCun uniform initializer.

  It draws samples from a uniform distribution within [-limit, limit]
  where `limit` is `sqrt(3 / fan_in)`
  where `fan_in` is the number of input units in the weight tensor.

  Arguments:
      seed: A Python integer. Used to seed the random generator.

  Returns:
      An initializer.

  References:
      - Self-Normalizing Neural Networks,
      [Klambauer et al., 2017](https://papers.nips.cc/paper/6698-self-normalizing-neural-networks) # pylint: disable=line-too-long
      ([pdf](https://papers.nips.cc/paper/6698-self-normalizing-neural-networks.pdf))
      - Efficient Backprop,
      [Lecun et al., 1998](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf)
  """
  return VarianceScaling(
      scale=1., mode="fan_in", distribution="uniform", seed=seed)


def he_normal(seed=None):
  """He normal initializer.

  It draws samples from a truncated normal distribution centered on 0
  with `stddev = sqrt(2 / fan_in)`
  where `fan_in` is the number of input units in the weight tensor.

  Arguments:
      seed: A Python integer. Used to seed the random generator.

  Returns:
      An initializer.

  References:
      [He et al., 2015](https://www.cv-foundation.org/openaccess/content_iccv_2015/html/He_Delving_Deep_into_ICCV_2015_paper.html) # pylint: disable=line-too-long
      ([pdf](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/He_Delving_Deep_into_ICCV_2015_paper.pdf))
  """
  return VarianceScaling(
      scale=2., mode="fan_in", distribution="truncated_normal", seed=seed)


def he_uniform(seed=None):
  """He uniform variance scaling initializer.

  It draws samples from a uniform distribution within [-limit, limit]
  where `limit` is `sqrt(6 / fan_in)`
  where `fan_in` is the number of input units in the weight tensor.

  Arguments:
      seed: A Python integer. Used to seed the random generator.

  Returns:
      An initializer.

  References:
      [He et al., 2015](https://www.cv-foundation.org/openaccess/content_iccv_2015/html/He_Delving_Deep_into_ICCV_2015_paper.html) # pylint: disable=line-too-long
      ([pdf](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/He_Delving_Deep_into_ICCV_2015_paper.pdf))
  """
  return VarianceScaling(
      scale=2., mode="fan_in", distribution="uniform", seed=seed)


# Utility functions.


def _compute_fans(shape):
  """Computes the number of input and output units for a weight shape.

  Args:
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
  dtype = dtypes.as_dtype(dtype)
  if not dtype.is_floating:
    raise ValueError("Expected floating point type, got %s." % dtype)
  return dtype
