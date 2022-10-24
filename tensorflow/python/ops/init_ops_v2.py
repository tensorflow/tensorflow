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
"""Initializers for TF 2."""
import math

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_linalg_ops
from tensorflow.python.ops import linalg_ops_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import stateless_random_ops
from tensorflow.python.ops.init_ops import _compute_fans
from tensorflow.python.util.tf_export import tf_export

_PARTITION_SHAPE = "partition_shape"
_PARTITION_OFFSET = "partition_offset"


class Initializer:
  """Initializer base class: all initializers inherit from this class.

  Initializers should implement a `__call__` method with the following
  signature:

  ```python
  def __call__(self, shape, dtype=None, **kwargs):
    # returns a tensor of shape `shape` and dtype `dtype`
    # containing values drawn from a distribution of your choice.
  ```
  """

  def __call__(self, shape, dtype=None, **kwargs):
    """Returns a tensor object initialized as specified by the initializer.

    Args:
      shape: Shape of the tensor.
      dtype: Optional dtype of the tensor. If not provided will return tensor
        of `tf.float32`.
      **kwargs: Additional keyword arguments. Accepted values:
        `partition_shape` and `partition_offset`. Used when creating a single
        partition in a partitioned variable. `partition_shape` is the shape of
        the partition (i.e. the shape of the returned tensor) and
        `partition_offset` is a tuple of `int` specifying the offset of this
        partition w.r.t each axis. For example, a tensor of shape `(30, 100)`
        can be partitioned into two partitions: `p0` of shape `(10, 100)` and
        `p1` of shape `(20, 100)`; if the initializer is called with
        `partition_shape=(20, 100)` and `partition_offset=(10, 0)`, it should
        return the value for `p1`.
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

  def _validate_kwargs(self, kwargs, support_partition=True):
    for kwarg in kwargs:
      if kwarg not in [_PARTITION_SHAPE, _PARTITION_OFFSET]:
        raise TypeError(
            "Keyword argument should be one of "
            f"{list([_PARTITION_SHAPE, _PARTITION_OFFSET])}. Received: {kwarg}")
      elif not support_partition:
        raise ValueError(
            f"{self.__class__.__name__} initializer doesn't support "
            "partition-related arguments")


@tf_export("zeros_initializer", v1=[])
class Zeros(Initializer):
  """Initializer that generates tensors initialized to 0.

  Initializers allow you to pre-specify an initialization strategy, encoded in
  the Initializer object, without knowing the shape and dtype of the variable
  being initialized.

  Examples:

  >>> def make_variables(k, initializer):
  ...   return (tf.Variable(initializer(shape=[k], dtype=tf.float32)),
  ...           tf.Variable(initializer(shape=[k, k], dtype=tf.float32)))
  >>> v1, v2 = make_variables(3, tf.zeros_initializer())
  >>> v1
  <tf.Variable ... shape=(3,) ... numpy=array([0., 0., 0.], dtype=float32)>
  >>> v2
  <tf.Variable ... shape=(3, 3) ... numpy=
  array([[0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.]], dtype=float32)>
  >>> make_variables(4, tf.random_uniform_initializer(minval=-1., maxval=1.))
  (<tf.Variable...shape=(4,) dtype=float32...>, <tf.Variable...shape=(4, 4) ...
  """

  def __call__(self, shape, dtype=dtypes.float32, **kwargs):
    """Returns a tensor object initialized as specified by the initializer.

    Args:
      shape: Shape of the tensor.
      dtype: Optional dtype of the tensor. Only numeric or boolean dtypes are
       supported.
      **kwargs: Additional keyword arguments.

    Raises:
      ValuesError: If the dtype is not numeric or boolean.
    """
    self._validate_kwargs(kwargs)
    dtype = dtypes.as_dtype(dtype)
    if not dtype.is_numpy_compatible or dtype == dtypes.string:
      raise ValueError("Argument `dtype` expected to be numeric or boolean. "
                       f"Received {dtype}.")
    if _PARTITION_SHAPE in kwargs:
      shape = kwargs[_PARTITION_SHAPE]
    return array_ops.zeros(shape, dtype)


@tf_export("ones_initializer", v1=[])
class Ones(Initializer):
  """Initializer that generates tensors initialized to 1.

  Initializers allow you to pre-specify an initialization strategy, encoded in
  the Initializer object, without knowing the shape and dtype of the variable
  being initialized.

  Examples:

  >>> def make_variables(k, initializer):
  ...   return (tf.Variable(initializer(shape=[k], dtype=tf.float32)),
  ...           tf.Variable(initializer(shape=[k, k], dtype=tf.float32)))
  >>> v1, v2 = make_variables(3, tf.ones_initializer())
  >>> v1
  <tf.Variable ... shape=(3,) ... numpy=array([1., 1., 1.], dtype=float32)>
  >>> v2
  <tf.Variable ... shape=(3, 3) ... numpy=
  array([[1., 1., 1.],
         [1., 1., 1.],
         [1., 1., 1.]], dtype=float32)>
  >>> make_variables(4, tf.random_uniform_initializer(minval=-1., maxval=1.))
  (<tf.Variable...shape=(4,) dtype=float32...>, <tf.Variable...shape=(4, 4) ...
  """

  def __call__(self, shape, dtype=dtypes.float32, **kwargs):
    """Returns a tensor object initialized as specified by the initializer.

    Args:
      shape: Shape of the tensor.
      dtype: Optional dtype of the tensor. Only numeric or boolean dtypes are
        supported.
      **kwargs: Additional keyword arguments.

    Raises:
      ValuesError: If the dtype is not numeric or boolean.
    """
    self._validate_kwargs(kwargs)
    dtype = dtypes.as_dtype(dtype)
    if not dtype.is_numpy_compatible or dtype == dtypes.string:
      raise ValueError("Argument `dtype` expected to be numeric or boolean. "
                       f"Received {dtype}.")
    if _PARTITION_SHAPE in kwargs:
      shape = kwargs[_PARTITION_SHAPE]
    return array_ops.ones(shape, dtype)


@tf_export("constant_initializer", v1=[])
class Constant(Initializer):
  """Initializer that generates tensors with constant values.

  Initializers allow you to pre-specify an initialization strategy, encoded in
  the Initializer object, without knowing the shape and dtype of the variable
  being initialized.

  `tf.constant_initializer` returns an object which when called returns a tensor
  populated with the `value` specified in the constructor. This `value` must be
  convertible to the requested `dtype`.

  The argument `value` can be a scalar constant value, or a list of
  values. Scalars broadcast to whichever shape is requested from the
  initializer.

  If `value` is a list, then the length of the list must be equal to the number
  of elements implied by the desired shape of the tensor. If the total number of
  elements in `value` is not equal to the number of elements required by the
  tensor shape, the initializer will raise a `TypeError`.

  Examples:

  >>> def make_variables(k, initializer):
  ...   return (tf.Variable(initializer(shape=[k], dtype=tf.float32)),
  ...           tf.Variable(initializer(shape=[k, k], dtype=tf.float32)))
  >>> v1, v2 = make_variables(3, tf.constant_initializer(2.))
  >>> v1
  <tf.Variable ... shape=(3,) ... numpy=array([2., 2., 2.], dtype=float32)>
  >>> v2
  <tf.Variable ... shape=(3, 3) ... numpy=
  array([[2., 2., 2.],
         [2., 2., 2.],
         [2., 2., 2.]], dtype=float32)>
  >>> make_variables(4, tf.random_uniform_initializer(minval=-1., maxval=1.))
  (<tf.Variable...shape=(4,) dtype=float32...>, <tf.Variable...shape=(4, 4) ...

  >>> value = [0, 1, 2, 3, 4, 5, 6, 7]
  >>> init = tf.constant_initializer(value)
  >>> # Fitting shape
  >>> tf.Variable(init(shape=[2, 4], dtype=tf.float32))
  <tf.Variable ...
  array([[0., 1., 2., 3.],
         [4., 5., 6., 7.]], dtype=float32)>
  >>> # Larger shape
  >>> tf.Variable(init(shape=[3, 4], dtype=tf.float32))
  Traceback (most recent call last):
  ...
  TypeError: ...value has 8 elements, shape is (3, 4) with 12 elements...
  >>> # Smaller shape
  >>> tf.Variable(init(shape=[2, 3], dtype=tf.float32))
  Traceback (most recent call last):
  ...
  TypeError: ...value has 8 elements, shape is (2, 3) with 6 elements...

  Args:
    value: A Python scalar, list or tuple of values, or a N-dimensional numpy
      array. All elements of the initialized variable will be set to the
      corresponding value in the `value` argument.

  Raises:
    TypeError: If the input `value` is not one of the expected types.
  """

  def __init__(self, value=0):
    if not (np.isscalar(value) or isinstance(value, (list, tuple, np.ndarray))):
      raise TypeError(
          f"Invalid type for initial value: {type(value).__name__}. Expected "
          "Python scalar, list or tuple of values, or numpy.ndarray.")
    self.value = value

  def __call__(self, shape, dtype=None, **kwargs):
    """Returns a tensor object initialized as specified by the initializer.

    Args:
      shape: Shape of the tensor.
      dtype: Optional dtype of the tensor. If not provided the dtype of the
        tensor created will be the type of the inital value.
      **kwargs: Additional keyword arguments.

    Raises:
      TypeError: If the initializer cannot create a tensor of the requested
       dtype.
    """
    self._validate_kwargs(kwargs, support_partition=False)
    if dtype is not None:
      dtype = dtypes.as_dtype(dtype)
    return constant_op.constant(self.value, dtype=dtype, shape=shape)

  def get_config(self):
    return {"value": self.value}


@tf_export("random_uniform_initializer", v1=[])
class RandomUniform(Initializer):
  """Initializer that generates tensors with a uniform distribution.

  Initializers allow you to pre-specify an initialization strategy, encoded in
  the Initializer object, without knowing the shape and dtype of the variable
  being initialized.

  Examples:

  >>> def make_variables(k, initializer):
  ...   return (tf.Variable(initializer(shape=[k], dtype=tf.float32)),
  ...           tf.Variable(initializer(shape=[k, k], dtype=tf.float32)))
  >>> v1, v2 = make_variables(3, tf.ones_initializer())
  >>> v1
  <tf.Variable ... shape=(3,) ... numpy=array([1., 1., 1.], dtype=float32)>
  >>> v2
  <tf.Variable ... shape=(3, 3) ... numpy=
  array([[1., 1., 1.],
         [1., 1., 1.],
         [1., 1., 1.]], dtype=float32)>
  >>> make_variables(4, tf.random_uniform_initializer(minval=-1., maxval=1.))
  (<tf.Variable...shape=(4,) dtype=float32...>, <tf.Variable...shape=(4, 4) ...

  Args:
    minval: A python scalar or a scalar tensor. Lower bound of the range of
      random values to generate (inclusive).
    maxval: A python scalar or a scalar tensor. Upper bound of the range of
      random values to generate (exclusive).
    seed: A Python integer. Used to create random seeds. See
      `tf.random.set_seed` for behavior.
  """

  def __init__(self, minval=-0.05, maxval=0.05, seed=None):
    self.minval = minval
    self.maxval = maxval
    self.seed = seed
    self._random_generator = _RandomGenerator(seed)

  def __call__(self, shape, dtype=dtypes.float32, **kwargs):
    """Returns a tensor object initialized as specified by the initializer.

    Args:
      shape: Shape of the tensor.
      dtype: Optional dtype of the tensor. Only floating point and integer
        types are supported.
      **kwargs: Additional keyword arguments.

    Raises:
      ValueError: If the dtype is not numeric.
    """
    self._validate_kwargs(kwargs)
    dtype = dtypes.as_dtype(dtype)
    if not dtype.is_floating and not dtype.is_integer:
      raise ValueError("Argument `dtype` expected to be numeric or boolean. "
                       f"Received {dtype}.")
    if _PARTITION_SHAPE in kwargs:
      shape = kwargs[_PARTITION_SHAPE]
    return self._random_generator.random_uniform(shape, self.minval,
                                                 self.maxval, dtype)

  def get_config(self):
    return {
        "minval": self.minval,
        "maxval": self.maxval,
        "seed": self.seed
    }


@tf_export("random_normal_initializer", v1=[])
class RandomNormal(Initializer):
  """Initializer that generates tensors with a normal distribution.

  Initializers allow you to pre-specify an initialization strategy, encoded in
  the Initializer object, without knowing the shape and dtype of the variable
  being initialized.

  Examples:

  >>> def make_variables(k, initializer):
  ...   return (tf.Variable(initializer(shape=[k], dtype=tf.float32)),
  ...           tf.Variable(initializer(shape=[k, k], dtype=tf.float32)))
  >>> v1, v2 = make_variables(3,
  ...                         tf.random_normal_initializer(mean=1., stddev=2.))
  >>> v1
  <tf.Variable ... shape=(3,) ... numpy=array([...], dtype=float32)>
  >>> v2
  <tf.Variable ... shape=(3, 3) ... numpy=
  ...
  >>> make_variables(4, tf.random_uniform_initializer(minval=-1., maxval=1.))
  (<tf.Variable...shape=(4,) dtype=float32...>, <tf.Variable...shape=(4, 4) ...

  Args:
    mean: a python scalar or a scalar tensor. Mean of the random values to
      generate.
    stddev: a python scalar or a scalar tensor. Standard deviation of the random
      values to generate.
    seed: A Python integer. Used to create random seeds. See
      `tf.random.set_seed` for behavior.

  """

  def __init__(self, mean=0.0, stddev=0.05, seed=None):
    self.mean = mean
    self.stddev = stddev
    self.seed = seed
    self._random_generator = _RandomGenerator(seed)

  def __call__(self, shape, dtype=dtypes.float32, **kwargs):
    """Returns a tensor object initialized as specified by the initializer.

    Args:
      shape: Shape of the tensor.
      dtype: Optional dtype of the tensor. Only floating point types are
        supported.
      **kwargs: Additional keyword arguments.

    Raises:
      ValueError: If the dtype is not floating point
    """
    self._validate_kwargs(kwargs)
    dtype = _assert_float_dtype(dtype)
    if _PARTITION_SHAPE in kwargs:
      shape = kwargs[_PARTITION_SHAPE]
    return self._random_generator.random_normal(shape, self.mean, self.stddev,
                                                dtype)

  def get_config(self):
    return {
        "mean": self.mean,
        "stddev": self.stddev,
        "seed": self.seed
    }


class TruncatedNormal(Initializer):
  """Initializer that generates a truncated normal distribution.

  Initializers allow you to pre-specify an initialization strategy, encoded in
  the Initializer object, without knowing the shape and dtype of the variable
  being initialized.

  These values are similar to values from a `tf.initializers.RandomNormal`
  except that values more than two standard deviations from the mean are
  discarded and re-drawn. This is the recommended initializer for neural network
  weights and filters.

  Examples:

  >>> def make_variables(k, initializer):
  ...   return (tf.Variable(initializer(shape=[k], dtype=tf.float32)),
  ...           tf.Variable(initializer(shape=[k, k], dtype=tf.float32)))
  >>> v1, v2 = make_variables(
  ...     3, tf.initializers.TruncatedNormal(mean=1., stddev=2.))
  >>> v1
  <tf.Variable ... shape=(3,) ... numpy=array([...], dtype=float32)>
  >>> v2
  <tf.Variable ... shape=(3, 3) ... numpy=
  ...
  >>> make_variables(4, tf.initializers.RandomUniform(minval=-1., maxval=1.))
  (<tf.Variable...shape=(4,) dtype=float32...>, <tf.Variable...shape=(4, 4) ...

  Args:
    mean: a python scalar or a scalar tensor. Mean of the random values
      to generate.
    stddev: a python scalar or a scalar tensor. Standard deviation of the
      random values to generate.
    seed: A Python integer. Used to create random seeds. See
      `tf.random.set_seed` for behavior.
  """

  def __init__(self, mean=0.0, stddev=0.05, seed=None):
    self.mean = mean
    self.stddev = stddev
    self.seed = seed
    self._random_generator = _RandomGenerator(seed)

  def __call__(self, shape, dtype=dtypes.float32, **kwargs):
    """Returns a tensor object initialized as specified by the initializer.

    Args:
      shape: Shape of the tensor.
      dtype: Optional dtype of the tensor. Only floating point types are
        supported.
      **kwargs: Additional keyword arguments.

    Raises:
      ValueError: If the dtype is not floating point
    """
    self._validate_kwargs(kwargs)
    dtype = _assert_float_dtype(dtype)
    if _PARTITION_SHAPE in kwargs:
      shape = kwargs[_PARTITION_SHAPE]
    return self._random_generator.truncated_normal(shape, self.mean,
                                                   self.stddev, dtype)

  def get_config(self):
    return {
        "mean": self.mean,
        "stddev": self.stddev,
        "seed": self.seed
    }


class VarianceScaling(Initializer):
  """Initializer capable of adapting its scale to the shape of weights tensors.

  Initializers allow you to pre-specify an initialization strategy, encoded in
  the Initializer object, without knowing the shape and dtype of the variable
  being initialized.

  With `distribution="truncated_normal" or "untruncated_normal"`, samples are
  drawn from a truncated/untruncated normal distribution with a mean of zero and
  a standard deviation (after truncation, if used) `stddev = sqrt(scale / n)`
  where n is:

    - number of input units in the weight tensor, if mode = "fan_in"
    - number of output units, if mode = "fan_out"
    - average of the numbers of input and output units, if mode = "fan_avg"

  With `distribution="uniform"`, samples are drawn from a uniform distribution
  within [-limit, limit], with `limit = sqrt(3 * scale / n)`.

  Examples:

  >>> def make_variables(k, initializer):
  ...   return (tf.Variable(initializer(shape=[k], dtype=tf.float32)),
  ...           tf.Variable(initializer(shape=[k, k], dtype=tf.float32)))
  >>> v1, v2 = make_variables(3, tf.initializers.VarianceScaling(scale=1.))
  >>> v1
  <tf.Variable ... shape=(3,) ... numpy=array([...], dtype=float32)>
  >>> v2
  <tf.Variable ... shape=(3, 3) ... numpy=
  ...
  >>> make_variables(4, tf.initializers.VarianceScaling(distribution='uniform'))
  (<tf.Variable...shape=(4,) dtype=float32...>, <tf.Variable...shape=(4, 4) ...

  Args:
    scale: Scaling factor (positive float).
    mode: One of "fan_in", "fan_out", "fan_avg".
    distribution: Random distribution to use. One of "truncated_normal",
      "untruncated_normal" and  "uniform".
    seed: A Python integer. Used to create random seeds. See
      `tf.random.set_seed` for behavior.

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
      raise ValueError("Argument `scale` must be a positive float. Received: "
                       f"{scale}")
    if mode not in {"fan_in", "fan_out", "fan_avg"}:
      raise ValueError("Argument `mode` should be one of ('fan_in', 'fan_out', "
                       f"'fan_avg'). Received: {mode}")
    distribution = distribution.lower()
    # Compatibility with keras-team/keras.
    if distribution == "normal":
      distribution = "truncated_normal"
    if distribution not in {"uniform", "truncated_normal",
                            "untruncated_normal"}:
      raise ValueError("Argument `distribution` should be one of ('uniform', "
                       "'truncated_normal', 'untruncated_normal'). Received: "
                       f"{distribution}")
    self.scale = scale
    self.mode = mode
    self.distribution = distribution
    self.seed = seed
    self._random_generator = _RandomGenerator(seed)

  def __call__(self, shape, dtype=dtypes.float32, **kwargs):
    """Returns a tensor object initialized as specified by the initializer.

    Args:
      shape: Shape of the tensor.
      dtype: Optional dtype of the tensor. Only floating point types are
        supported.
      **kwargs: Additional keyword arguments.

    Raises:
      ValueError: If the dtype is not floating point
    """
    self._validate_kwargs(kwargs)
    dtype = _assert_float_dtype(dtype)
    scale = self.scale
    fan_in, fan_out = _compute_fans(shape)
    if _PARTITION_SHAPE in kwargs:
      shape = kwargs[_PARTITION_SHAPE]
    if self.mode == "fan_in":
      scale /= max(1., fan_in)
    elif self.mode == "fan_out":
      scale /= max(1., fan_out)
    else:
      scale /= max(1., (fan_in + fan_out) / 2.)
    if self.distribution == "truncated_normal":
      # constant from scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
      stddev = math.sqrt(scale) / .87962566103423978
      return self._random_generator.truncated_normal(shape, 0.0, stddev, dtype)
    elif self.distribution == "untruncated_normal":
      stddev = math.sqrt(scale)
      return self._random_generator.random_normal(shape, 0.0, stddev, dtype)
    else:
      limit = math.sqrt(3.0 * scale)
      return self._random_generator.random_uniform(shape, -limit, limit, dtype)

  def get_config(self):
    return {
        "scale": self.scale,
        "mode": self.mode,
        "distribution": self.distribution,
        "seed": self.seed
    }


class Orthogonal(Initializer):
  """Initializer that generates an orthogonal matrix.

  Initializers allow you to pre-specify an initialization strategy, encoded in
  the Initializer object, without knowing the shape and dtype of the variable
  being initialized.

  If the shape of the tensor to initialize is two-dimensional, it is initialized
  with an orthogonal matrix obtained from the QR decomposition of a matrix of
  random numbers drawn from a normal distribution.
  If the matrix has fewer rows than columns then the output will have orthogonal
  rows. Otherwise, the output will have orthogonal columns.

  If the shape of the tensor to initialize is more than two-dimensional,
  a matrix of shape `(shape[0] * ... * shape[n - 2], shape[n - 1])`
  is initialized, where `n` is the length of the shape vector.
  The matrix is subsequently reshaped to give a tensor of the desired shape.

  Examples:

  >>> def make_variables(k, initializer):
  ...   return (tf.Variable(initializer(shape=[k, k], dtype=tf.float32)),
  ...           tf.Variable(initializer(shape=[k, k, k], dtype=tf.float32)))
  >>> v1, v2 = make_variables(3, tf.initializers.Orthogonal())
  >>> v1
  <tf.Variable ... shape=(3, 3) ...
  >>> v2
  <tf.Variable ... shape=(3, 3, 3) ...
  >>> make_variables(4, tf.initializers.Orthogonal(gain=0.5))
  (<tf.Variable ... shape=(4, 4) dtype=float32...
   <tf.Variable ... shape=(4, 4, 4) dtype=float32...

  Args:
    gain: multiplicative factor to apply to the orthogonal matrix
    seed: A Python integer. Used to create random seeds. See
      `tf.random.set_seed` for behavior.

  References:
      [Saxe et al., 2014](https://openreview.net/forum?id=_wzZwKpTDF_9C)
      ([pdf](https://arxiv.org/pdf/1312.6120.pdf))
  """

  def __init__(self, gain=1.0, seed=None):
    self.gain = gain
    self.seed = seed
    self._random_generator = _RandomGenerator(seed)

  def __call__(self, shape, dtype=dtypes.float32, **kwargs):
    """Returns a tensor object initialized as specified by the initializer.

    Args:
      shape: Shape of the tensor.
      dtype: Optional dtype of the tensor. Only floating point types are
        supported.
      **kwargs: Additional keyword arguments.

    Raises:
      ValueError: If the dtype is not floating point or the input shape is not
       valid.
    """
    self._validate_kwargs(kwargs, support_partition=False)
    dtype = _assert_float_dtype(dtype)
    # Check the shape
    if len(shape) < 2:
      raise ValueError("The tensor to initialize, specified by argument `shape`"
                       " must be at least two-dimensional. Received shape="
                       f"{shape}")
    # Flatten the input shape with the last dimension remaining
    # its original shape so it works for conv2d
    num_rows = 1
    for dim in shape[:-1]:
      num_rows *= dim
    num_cols = shape[-1]
    flat_shape = (max(num_cols, num_rows), min(num_cols, num_rows))

    # Generate a random matrix
    a = self._random_generator.random_normal(flat_shape, dtype=dtype)
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

  Initializers allow you to pre-specify an initialization strategy, encoded in
  the Initializer object, without knowing the shape and dtype of the variable
  being initialized.

  Only usable for generating 2D matrices.

  Examples:

  >>> def make_variable(k, initializer):
  ...   return tf.Variable(initializer(shape=[k, k], dtype=tf.float32))
  >>> make_variable(2, tf.initializers.Identity())
  <tf.Variable ... shape=(2, 2) dtype=float32, numpy=
  array([[1., 0.],
         [0., 1.]], dtype=float32)>
  >>> make_variable(3, tf.initializers.Identity(gain=0.5))
  <tf.Variable ... shape=(3, 3) dtype=float32, numpy=
  array([[0.5, 0. , 0. ],
         [0. , 0.5, 0. ],
         [0. , 0. , 0.5]], dtype=float32)>

  Args:
    gain: Multiplicative factor to apply to the identity matrix.
  """

  def __init__(self, gain=1.0):
    self.gain = gain

  def __call__(self, shape, dtype=dtypes.float32, **kwargs):
    """Returns a tensor object initialized as specified by the initializer.

    Args:
      shape: Shape of the tensor.
      dtype: Optional dtype of the tensor. Only floating point types are
       supported.
      **kwargs: Additional keyword arguments.

    Raises:
      ValueError: If the dtype is not floating point
      ValueError: If the requested shape does not have exactly two axes.
    """
    self._validate_kwargs(kwargs, support_partition=False)
    dtype = _assert_float_dtype(dtype)
    if len(shape) != 2:
      raise ValueError("The tensor to initialize, specified by argument `shape`"
                       " must be at least two-dimensional. Received shape="
                       f"{shape}")
    initializer = linalg_ops_impl.eye(*shape, dtype=dtype)
    return self.gain * initializer

  def get_config(self):
    return {"gain": self.gain}


class GlorotUniform(VarianceScaling):
  """The Glorot uniform initializer, also called Xavier uniform initializer.

  Initializers allow you to pre-specify an initialization strategy, encoded in
  the Initializer object, without knowing the shape and dtype of the variable
  being initialized.

  Draws samples from a uniform distribution within [-limit, limit] where `limit`
  is `sqrt(6 / (fan_in + fan_out))` where `fan_in` is the number of input units
  in the weight tensor and `fan_out` is the number of output units in the weight
  tensor.

  Examples:

  >>> def make_variables(k, initializer):
  ...   return (tf.Variable(initializer(shape=[k, k], dtype=tf.float32)),
  ...           tf.Variable(initializer(shape=[k, k, k], dtype=tf.float32)))
  >>> v1, v2 = make_variables(3, tf.initializers.GlorotUniform())
  >>> v1
  <tf.Variable ... shape=(3, 3) ...
  >>> v2
  <tf.Variable ... shape=(3, 3, 3) ...
  >>> make_variables(4, tf.initializers.RandomNormal())
  (<tf.Variable ... shape=(4, 4) dtype=float32...
   <tf.Variable ... shape=(4, 4, 4) dtype=float32...

  Args:
    seed: A Python integer. Used to create random seeds. See
      `tf.random.set_seed` for behavior.

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

  Initializers allow you to pre-specify an initialization strategy, encoded in
  the Initializer object, without knowing the shape and dtype of the variable
  being initialized.

  Draws samples from a truncated normal distribution centered on 0 with `stddev
  = sqrt(2 / (fan_in + fan_out))` where `fan_in` is the number of input units in
  the weight tensor and `fan_out` is the number of output units in the weight
  tensor.

  Examples:

  >>> def make_variables(k, initializer):
  ...   return (tf.Variable(initializer(shape=[k, k], dtype=tf.float32)),
  ...           tf.Variable(initializer(shape=[k, k, k], dtype=tf.float32)))
  >>> v1, v2 = make_variables(3, tf.initializers.GlorotNormal())
  >>> v1
  <tf.Variable ... shape=(3, 3) ...
  >>> v2
  <tf.Variable ... shape=(3, 3, 3) ...
  >>> make_variables(4, tf.initializers.RandomNormal())
  (<tf.Variable ... shape=(4, 4) dtype=float32...
   <tf.Variable ... shape=(4, 4, 4) dtype=float32...

  Args:
    seed: A Python integer. Used to create random seeds. See
      `tf.random.set_seed` for behavior.

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
    return {"seed": self.seed}


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

  Initializers allow you to pre-specify an initialization strategy, encoded in
  the Initializer object, without knowing the shape and dtype of the variable
  being initialized.

  Draws samples from a truncated normal distribution centered on 0 with `stddev
  = sqrt(1 / fan_in)` where `fan_in` is the number of input units in the weight
  tensor.

  Examples:

  >>> def make_variables(k, initializer):
  ...   return (tf.Variable(initializer(shape=[k, k], dtype=tf.float32)),
  ...           tf.Variable(initializer(shape=[k, k, k], dtype=tf.float32)))
  >>> v1, v2 = make_variables(3, tf.initializers.lecun_normal())
  >>> v1
  <tf.Variable ... shape=(3, 3) ...
  >>> v2
  <tf.Variable ... shape=(3, 3, 3) ...
  >>> make_variables(4, tf.initializers.RandomNormal())
  (<tf.Variable ... shape=(4, 4) dtype=float32...
   <tf.Variable ... shape=(4, 4, 4) dtype=float32...

  Args:
    seed: A Python integer. Used to seed the random generator.

  Returns:
    A callable Initializer with `shape` and `dtype` arguments which generates a
    tensor.

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

  Initializers allow you to pre-specify an initialization strategy, encoded in
  the Initializer object, without knowing the shape and dtype of the variable
  being initialized.

  Draws samples from a uniform distribution within [-limit, limit] where `limit`
  is `sqrt(3 / fan_in)` where `fan_in` is the number of input units in the
  weight tensor.

  Examples:

  >>> def make_variables(k, initializer):
  ...   return (tf.Variable(initializer(shape=[k, k], dtype=tf.float32)),
  ...           tf.Variable(initializer(shape=[k, k, k], dtype=tf.float32)))
  >>> v1, v2 = make_variables(3, tf.initializers.lecun_uniform())
  >>> v1
  <tf.Variable ... shape=(3, 3) ...
  >>> v2
  <tf.Variable ... shape=(3, 3, 3) ...
  >>> make_variables(4, tf.initializers.RandomNormal())
  (<tf.Variable ... shape=(4, 4) dtype=float32...
   <tf.Variable ... shape=(4, 4, 4) dtype=float32...

  Args:
    seed: A Python integer. Used to seed the random generator.

  Returns:
    A callable Initializer with `shape` and `dtype` arguments which generates a
    tensor.

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

  Initializers allow you to pre-specify an initialization strategy, encoded in
  the Initializer object, without knowing the shape and dtype of the variable
  being initialized.

  It draws samples from a truncated normal distribution centered on 0 with
  `stddev = sqrt(2 / fan_in)` where `fan_in` is the number of input units in the
  weight tensor.

  Examples:

  >>> def make_variables(k, initializer):
  ...   return (tf.Variable(initializer(shape=[k, k], dtype=tf.float32)),
  ...           tf.Variable(initializer(shape=[k, k, k], dtype=tf.float32)))
  >>> v1, v2 = make_variables(3, tf.initializers.he_normal())
  >>> v1
  <tf.Variable ... shape=(3, 3) ...
  >>> v2
  <tf.Variable ... shape=(3, 3, 3) ...
  >>> make_variables(4, tf.initializers.RandomNormal())
  (<tf.Variable ... shape=(4, 4) dtype=float32...
   <tf.Variable ... shape=(4, 4, 4) dtype=float32...

  Args:
    seed: A Python integer. Used to seed the random generator.

  Returns:
    A callable Initializer with `shape` and `dtype` arguments which generates a
    tensor.

  References:
      [He et al., 2015](https://www.cv-foundation.org/openaccess/content_iccv_2015/html/He_Delving_Deep_into_ICCV_2015_paper.html) # pylint: disable=line-too-long
      ([pdf](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/He_Delving_Deep_into_ICCV_2015_paper.pdf))
  """
  return VarianceScaling(
      scale=2., mode="fan_in", distribution="truncated_normal", seed=seed)


def he_uniform(seed=None):
  """He uniform variance scaling initializer.

  Initializers allow you to pre-specify an initialization strategy, encoded in
  the Initializer object, without knowing the shape and dtype of the variable
  being initialized.

  Draws samples from a uniform distribution within [-limit, limit] where `limit`
  is `sqrt(6 / fan_in)` where `fan_in` is the number of input units in the
  weight tensor.

  Examples:

  >>> def make_variables(k, initializer):
  ...   return (tf.Variable(initializer(shape=[k, k], dtype=tf.float32)),
  ...           tf.Variable(initializer(shape=[k, k, k], dtype=tf.float32)))
  >>> v1, v2 = make_variables(3, tf.initializers.he_uniform())
  >>> v1
  <tf.Variable ... shape=(3, 3) ...
  >>> v2
  <tf.Variable ... shape=(3, 3, 3) ...
  >>> make_variables(4, tf.initializers.RandomNormal())
  (<tf.Variable ... shape=(4, 4) dtype=float32...
   <tf.Variable ... shape=(4, 4, 4) dtype=float32...

  Args:
    seed: A Python integer. Used to seed the random generator.

  Returns:
    A callable Initializer with `shape` and `dtype` arguments which generates a
    tensor.

  References:
      [He et al., 2015](https://www.cv-foundation.org/openaccess/content_iccv_2015/html/He_Delving_Deep_into_ICCV_2015_paper.html) # pylint: disable=line-too-long
      ([pdf](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/He_Delving_Deep_into_ICCV_2015_paper.pdf))
  """
  return VarianceScaling(
      scale=2., mode="fan_in", distribution="uniform", seed=seed)


# Utility functions.


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
    raise ValueError("Argument `dtype` is expected to be floating point. "
                     f"Received: {dtype}.")
  return dtype


class _RandomGenerator:
  """Random generator that selects appropriate random ops."""

  def __init__(self, seed=None):
    super(_RandomGenerator, self).__init__()
    if seed is not None:
      # Stateless random ops requires 2-int seed.
      self.seed = [seed, 0]
    else:
      self.seed = None

  def random_normal(self, shape, mean=0.0, stddev=1, dtype=dtypes.float32):
    """A deterministic random normal if seed is passed."""
    if self.seed:
      op = stateless_random_ops.stateless_random_normal
    else:
      op = random_ops.random_normal
    return op(
        shape=shape, mean=mean, stddev=stddev, dtype=dtype, seed=self.seed)

  def random_uniform(self, shape, minval, maxval, dtype):
    """A deterministic random uniform if seed is passed."""
    if self.seed:
      op = stateless_random_ops.stateless_random_uniform
    else:
      op = random_ops.random_uniform
    return op(
        shape=shape, minval=minval, maxval=maxval, dtype=dtype, seed=self.seed)

  def truncated_normal(self, shape, mean, stddev, dtype):
    """A deterministic truncated normal if seed is passed."""
    if self.seed:
      op = stateless_random_ops.stateless_truncated_normal
    else:
      op = random_ops.truncated_normal
    return op(
        shape=shape, mean=mean, stddev=stddev, dtype=dtype, seed=self.seed)

# Compatibility aliases

# pylint: disable=invalid-name
zero = zeros = Zeros
one = ones = Ones
constant = Constant
uniform = random_uniform = RandomUniform
normal = random_normal = RandomNormal
truncated_normal = TruncatedNormal
identity = Identity
orthogonal = Orthogonal
glorot_normal = GlorotNormal
glorot_uniform = GlorotUniform
