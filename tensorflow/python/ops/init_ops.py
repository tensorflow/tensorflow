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
import math

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_linalg_ops
from tensorflow.python.ops import linalg_ops_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.util import deprecation
from tensorflow.python.util.deprecation import deprecated
from tensorflow.python.util.deprecation import deprecated_arg_values
from tensorflow.python.util.deprecation import deprecated_args
from tensorflow.python.util.tf_export import tf_export


class Initializer(object):
  """Initializer base class: all initializers inherit from this class."""

  def __call__(self, shape, dtype=None, partition_info=None):
    """Returns a tensor object initialized as specified by the initializer.

    Args:
      shape: Shape of the tensor.
      dtype: Optional dtype of the tensor. If not provided use the initializer
        dtype.
      partition_info: Optional information about the possible partitioning of a
        tensor.
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
      config: A Python dictionary. It will typically be the output of
        `get_config`.

    Returns:
      An Initializer instance.
    """
    return cls(**config)


@tf_export(v1=["initializers.zeros", "zeros_initializer"])
@deprecation.deprecated_endpoints("initializers.zeros")
class Zeros(Initializer):
  """Initializer that generates tensors initialized to 0.

  @compatibility(TF2)
  `tf.compat.v1.zeros_initializer` is compatible with eager execution
  and `tf.function`.

  To migrate to TF2, please use `tf.zeros_initializer` instead. The `dtype`
  argument in `tf.compat.v1.zeros_initializer.__init__()` does not exist in
  `tf.zeros_initializer.__init__()`. However, you can specify the `dtype` in
  `__call__()` in both cases.

  #### Structural Mapping to TF2

  Before:

  ```python
  initializer = tf.compat.v1.zeros_initializer(dtype=tf.float32)
  variable = tf.Variable(initializer(shape=[3, 3]))
  ```

  After:

  ```python
  initializer = tf.zeros_initializer()
  variable = tf.Variable(initializer(shape=[3, 3], dtype=tf.float32))
  ```

  #### How to Map Arguments

  | TF1 Arg Name         | TF2 Arg Name     | Note                       |
  | :------------------- | :--------------- | :------------------------- |
  | `dtype`              | `dtype`          | In `__call__()` method     |
  | `partition_info`     | - |  (`__call__` arg in TF1) Not supported    |


  #### Before & After Usage Example

  Before:

  >>> initializer = tf.compat.v1.zeros_initializer(dtype=tf.float32)
  >>> tf.Variable(initializer(shape=[3])).numpy()
  array([0., 0., 0.], dtype=float32)
  >>> tf.Variable(initializer(shape=[3, 3])).numpy()
  array([[0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.]], dtype=float32)
  >>> initializer = tf.compat.v1.zeros_initializer()
  >>> tf.Variable(initializer(shape=[3], dtype=tf.float32)).numpy()
  array([0., 0., 0.], dtype=float32)
  >>> tf.Variable(initializer(shape=[3, 3], dtype=tf.float32)).numpy()
  array([[0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.]], dtype=float32)

  After:

  >>> initializer = tf.zeros_initializer()
  >>> tf.Variable(initializer(shape=[3], dtype=tf.float32)).numpy()
  array([0., 0., 0.], dtype=float32)
  >>> tf.Variable(initializer(shape=[3, 3], dtype=tf.float32)).numpy()
  array([[0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.]], dtype=float32)

  @end_compatibility
  """

  @deprecated_args(None,
                   "Call initializer instance with the dtype argument instead "
                   "of passing it to the constructor", "dtype")
  def __init__(self, dtype=dtypes.float32):
    self.dtype = dtypes.as_dtype(dtype)

  def __call__(self, shape, dtype=None, partition_info=None):
    if dtype is None:
      dtype = self.dtype
    return array_ops.zeros(shape, dtype)

  def get_config(self):
    return {"dtype": self.dtype.name}


@tf_export(v1=["initializers.ones", "ones_initializer"])
@deprecation.deprecated_endpoints("initializers.ones", "ones_initializer")
class Ones(Initializer):
  """Initializer that generates tensors initialized to 1.

  @compatibility(TF2)
  This API is compatible with TF2 behavior and `tf.function`, and can be
  migrated immediately with `tf.keras.initializers.ones`.

  Before:
  >>> initializer = tf.compat.v1.keras.initializers.ones()
  >>> initializer((1, 1))
  <tf.Tensor: shape=(1, 1), dtype=float32, numpy=array([[1.]], dtype=float32)>

  After:
  >>> initializer = tf.keras.initializers.ones()
  >>> initializer((1, 1))
  <tf.Tensor: shape=(1, 1), dtype=float32, numpy=array([[1.]], dtype=float32)>

  @end_compatibility
  """

  @deprecated_args(None,
                   "Call initializer instance with the dtype argument instead "
                   "of passing it to the constructor", "dtype")
  def __init__(self, dtype=dtypes.float32):
    self.dtype = dtypes.as_dtype(dtype)

  def __call__(self, shape, dtype=None, partition_info=None):
    if dtype is None:
      dtype = self.dtype
    return array_ops.ones(shape, dtype)

  def get_config(self):
    return {"dtype": self.dtype.name}


@tf_export(v1=["initializers.constant", "constant_initializer"])
@deprecation.deprecated_endpoints("constant_initializer")
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
    dtype: Default data type, used if no `dtype` argument is provided when
      calling the initializer.
    verify_shape: Boolean that enables verification of the shape of `value`. If
      `True`, the initializer will throw an error if the shape of `value` is not
      compatible with the shape of the initialized tensor.

  Raises:
    TypeError: If the input `value` is not one of the expected types.

  Examples:
    The following example can be rewritten using a numpy.ndarray instead
    of the `value` list, even reshaped, as shown in the two commented lines
    below the `value` list initialization.

  >>> value = [0, 1, 2, 3, 4, 5, 6, 7]
  >>> init = tf.compat.v1.constant_initializer(value)
  >>> # fitting shape
  >>> with tf.compat.v1.Session():
  ...   x = tf.compat.v1.get_variable('x', shape=[2, 4], initializer=init)
  ...   x.initializer.run()
  ...   print(x.eval())
  [[0. 1. 2. 3.]
   [4. 5. 6. 7.]]
  >>> # Larger shape
  >>> with tf.compat.v1.Session():
  ...   y = tf.compat.v1.get_variable('y', shape=[3, 4], initializer=init)
  ...   y.initializer.run()
  ...   print(y.eval())
  [[0.  1.  2.  3.]
   [4.  5.  6.  7.]
   [7.  7.  7.  7.]]
  >>> # Smaller shape
  >>> with tf.compat.v1.Session():
  ...   z = tf.compat.v1.get_variable('z', shape=[2, 3], initializer=init)
  Traceback (most recent call last):
  ...
  ValueError: Too many elements provided. Needed at most 6, but received 8
  >>> # Shape verification
  >>> init_verify = tf.compat.v1.constant_initializer(value, verify_shape=True)
  >>> with tf.compat.v1.Session():
  ...  u = tf.compat.v1.get_variable('u', shape=[3, 4],
  ...                                initializer=init_verify)
  Traceback (most recent call last):
  ...
  TypeError: Expected Tensor's shape: (3, 4), got (8,).

  @compatibility(TF2)
  Although it is a legacy API endpoint, `tf.compat.v1.constant_initializer`
  is compatible with eager execution and `tf.function`.

  To migrate to a non-legacy TF2 API, please use `tf.constant_initializer`
  instead. The `dtype`
  argument in `tf.compat.v1.constant_initializer.__init__()` does not exist in
  `tf.constant_initializer.__init__()`. However, you can specify the `dtype` in
  `__call__()` in both cases.

  In the `compat.v1` symbol, if `verify_shape` is set to `True`, an exception
  is raised when initializing a variable with a different shape from
  `value`. If set to `False`, `value` is reshaped to initialize the variable
  if necessary. An exception would only be raised when the number of
  elements are different.

  The `verify_shape` argument is not supported in TF2. Using
  `tf.constant_initializer` is equivalent to setting `verify_shape` to `False`.

  #### Structural Mapping to TF2

  Before:

  ```python
  value = [0, 1, 2, 3, 4, 5, 6, 7]
  initializer = tf.compat.v1.constant_initializer(
      value=value,
      dtype=tf.float32,
      verify_shape=False)
  variable = tf.Variable(initializer(shape=[2, 4]))
  ```

  After:

  ```python
  value = [0, 1, 2, 3, 4, 5, 6, 7]
  initializer = tf.constant_initializer(value=value)
  tf.Variable(initializer(shape=[2, 4], dtype=tf.float32))
  ```

  #### How to Map Arguments

  | TF1 Arg Name          | TF2 Arg Name     | Note                        |
  | :-------------------- | :--------------- | :-------------------------- |
  | `value`               | `value`          | In constructor              |
  | `dtype`               | `dtype`          | In `__call__()` method      |
  | `verify_shape`        | Not Supported    | Equivalent to set to `False`|
  | `partition_info`      | - |  (`__call__` arg in TF1) Not supported     |


  #### Before & After Usage Example

  Before:

  >>> value = [1., 2., 3., 4.]
  >>> initializer = tf.compat.v1.constant_initializer(
  ...     value=value, dtype=tf.float32, verify_shape=True)
  >>> tf.Variable(initializer(shape=[2, 2])).numpy()
  Traceback (most recent call last):
  ...
  TypeError: Expected Tensor's shape: (2, 2), got (4,).
  >>> initializer = tf.compat.v1.constant_initializer(
  ...     value=value, dtype=tf.float32, verify_shape=False)
  >>> tf.Variable(initializer(shape=[2, 2])).numpy()
  array([[1., 2.],
         [3., 4.]], dtype=float32)

  After:

  >>> value = [1., 2., 3., 4.]
  >>> initializer = tf.constant_initializer(value=value)
  >>> tf.Variable(initializer(shape=[2, 2], dtype=tf.float32)).numpy()
  array([[1., 2.],
         [3., 4.]], dtype=float32)

  @end_compatibility
  """

  @deprecated_args(None,
                   "Call initializer instance with the dtype argument instead "
                   "of passing it to the constructor", "dtype")
  @deprecated_args(None, "Objects must now be the required shape or no shape "
                   "can be specified", "verify_shape")
  def __init__(self, value=0, dtype=dtypes.float32, verify_shape=False):
    if not (np.isscalar(value) or isinstance(value, (list, tuple, np.ndarray))):
      raise TypeError(
          f"Invalid type for initial value={value} of type: "
          f"{type(value).__name__}. Expected Python scalar, list or tuple of "
          "values, or numpy.ndarray.")

    self.value = value
    self.dtype = dtypes.as_dtype(dtype)
    self._verify_shape = verify_shape

  def __call__(self, shape, dtype=None, partition_info=None, verify_shape=None):
    if dtype is None:
      dtype = self.dtype
    if verify_shape is None:
      verify_shape = self._verify_shape
    return constant_op.constant_v1(
        self.value, dtype=dtype, shape=shape, verify_shape=verify_shape)

  def get_config(self):
    # We don't include `verify_shape` for compatibility with Keras.
    # `verify_shape` should be passed as an argument to `__call__` rather
    # than as a constructor argument: conceptually it isn't a property
    # of the initializer.
    return {"value": self.value, "dtype": self.dtype.name}


@tf_export(v1=["initializers.random_uniform", "random_uniform_initializer"])
@deprecation.deprecated_endpoints("initializers.random_uniform")
class RandomUniform(Initializer):
  """Initializer that generates tensors with a uniform distribution.

  Args:
    minval: A python scalar or a scalar tensor. Lower bound of the range of
      random values to generate.
    maxval: A python scalar or a scalar tensor. Upper bound of the range of
      random values to generate.  Defaults to 1 for float types.
    seed: A Python integer. Used to create random seeds. See
      `tf.compat.v1.set_random_seed` for behavior.
    dtype: Default data type, used if no `dtype` argument is provided when
      calling the initializer.

  @compatibility(TF2)
  Although it is a legacy compat.v1 API, this symbol is compatible with eager
  execution and `tf.function`.

  To switch to TF2, switch to using either
  `tf.initializers.RandomUniform` or `tf.keras.initializers.RandomUniform`
  (neither from `compat.v1`) and
  pass the dtype when calling the initializer. Keep in mind that
  the default minval, maxval and the behavior of fixed seeds have changed.

  #### Structural Mapping to TF2

  Before:

  ```python
  initializer = tf.compat.v1.random_uniform_initializer(
    minval=minval,
    maxval=maxval,
    seed=seed,
    dtype=dtype)

  weight_one = tf.Variable(initializer(shape_one))
  weight_two = tf.Variable(initializer(shape_two))
  ```

  After:

  ```python
  initializer = tf.initializers.RandomUniform(
    minval=minval,
    maxval=maxval,
    seed=seed)

  weight_one = tf.Variable(initializer(shape_one, dtype=dtype))
  weight_two = tf.Variable(initializer(shape_two, dtype=dtype))
  ```

  #### How to Map Arguments

  | TF1 Arg Name          | TF2 Arg Name    | Note                       |
  | :-------------------- | :-------------- | :------------------------- |
  | `minval`               | `minval`    | Default changes from 0 to -0.05 |
  | `maxval`         | `maxval`        | Default changes from 1.0 to 0.05 |
  | `seed`             | `seed` |  |
  | `dtype` | `dtype`   | The TF2 native api only takes it  |
  :                     :      : as a `__call__` arg, not a constructor arg. :
  | `partition_info`     | - |  (`__call__` arg in TF1) Not supported       |

  @end_compatibility
  """

  @deprecated_args(None,
                   "Call initializer instance with the dtype argument instead "
                   "of passing it to the constructor", "dtype")
  def __init__(self, minval=.0, maxval=None, seed=None, dtype=dtypes.float32):
    self.minval = minval
    self.maxval = maxval
    self.seed = seed
    self.dtype = dtypes.as_dtype(dtype)

  def __call__(self, shape, dtype=None, partition_info=None):
    if dtype is None:
      dtype = self.dtype
    return random_ops.random_uniform(
        shape, self.minval, self.maxval, dtype, seed=self.seed)

  def get_config(self):
    return {
        "minval": self.minval,
        "maxval": self.maxval,
        "seed": self.seed,
        "dtype": self.dtype.name
    }


@tf_export(v1=["initializers.random_normal", "random_normal_initializer"])
@deprecation.deprecated_endpoints("initializers.random_normal")
class RandomNormal(Initializer):
  """Initializer that generates tensors with a normal distribution.

  Args:
    mean: a python scalar or a scalar tensor. Mean of the random values to
      generate.
    stddev: a python scalar or a scalar tensor. Standard deviation of the random
      values to generate.
    seed: A Python integer. Used to create random seeds. See
      `tf.compat.v1.set_random_seed` for behavior.
    dtype: Default data type, used if no `dtype` argument is provided when
      calling the initializer. Only floating point types are supported.

  @compatibility(TF2)
  Although it is a legacy `compat.v1` API, this symbol is compatible with eager
  execution and `tf.function`.

  To switch to TF2, switch to using either
  `tf.initializers.RandomNormal` or `tf.keras.initializers.RandomNormal`
  (neither from `compat.v1`) and
  pass the dtype when calling the initializer. Keep in mind that
  the default stddev and the behavior of fixed seeds have changed.

  #### Structural Mapping to TF2

  Before:

  ```python
  initializer = tf.compat.v1.random_normal_initializer(
    mean=mean,
    stddev=stddev,
    seed=seed,
    dtype=dtype)

  weight_one = tf.Variable(initializer(shape_one))
  weight_two = tf.Variable(initializer(shape_two))
  ```

  After:

  ```python
  initializer = tf.initializers.RandomNormal(
    mean=mean,
    seed=seed,
    stddev=stddev)

  weight_one = tf.Variable(initializer(shape_one, dtype=dtype))
  weight_two = tf.Variable(initializer(shape_two, dtype=dtype))
  ```

  #### How to Map Arguments

  | TF1 Arg Name       | TF2 Arg Name    | Note                       |
  | :----------------- | :-------------- | :------------------------- |
  | `mean`             | `mean`          | No change to defaults |
  | `stddev`           | `stddev`        | Default changes from 1.0 to 0.05 |
  | `seed`             | `seed`          |                                  |
  | `dtype`            | `dtype`  | The TF2 native api only takes it as a |
  :                    :          : `__call__` arg, not a constructor arg. :
  | `partition_info`   | -     |  (`__call__` arg in TF1) Not supported.  |

  @end_compatibility
  """

  @deprecated_args(None,
                   "Call initializer instance with the dtype argument instead "
                   "of passing it to the constructor", "dtype")
  def __init__(self, mean=0.0, stddev=1.0, seed=None, dtype=dtypes.float32):
    self.mean = mean
    self.stddev = stddev
    self.seed = seed
    self.dtype = _assert_float_dtype(dtypes.as_dtype(dtype))

  def __call__(self, shape, dtype=None, partition_info=None):
    if dtype is None:
      dtype = self.dtype
    return random_ops.random_normal(
        shape, self.mean, self.stddev, dtype, seed=self.seed)

  def get_config(self):
    return {
        "mean": self.mean,
        "stddev": self.stddev,
        "seed": self.seed,
        "dtype": self.dtype.name
    }


@tf_export(v1=["initializers.truncated_normal", "truncated_normal_initializer"])
@deprecation.deprecated_endpoints("initializers.truncated_normal",
                                  "truncated_normal_initializer")
class TruncatedNormal(Initializer):
  """Initializer that generates a truncated normal distribution.

  These values are similar to values from a `random_normal_initializer`
  except that values more than two standard deviations from the mean
  are discarded and re-drawn. This is the recommended initializer for
  neural network weights and filters.

  Args:
    mean: a python scalar or a scalar tensor. Mean of the random values to
      generate.
    stddev: a python scalar or a scalar tensor. Standard deviation of the random
      values to generate.
    seed: A Python integer. Used to create random seeds. See
      `tf.compat.v1.set_random_seed` for behavior.
    dtype: Default data type, used if no `dtype` argument is provided when
      calling the initializer. Only floating point types are supported.

  @compatibility(TF2)
  Although it is a legacy `compat.v1` API, this symbol is compatible with eager
  execution and `tf.function`.

  To switch to TF2, switch to using either
  `tf.initializers.truncated_normal` or `tf.keras.initializers.TruncatedNormal`
  (neither from `compat.v1`) and
  pass the dtype when calling the initializer. Keep in mind that
  the default stddev and the behavior of fixed seeds have changed.

  #### Structural Mapping to TF2

  Before:

  ```python
  initializer = tf.compat.v1.truncated_normal_initializer(
    mean=mean,
    stddev=stddev,
    seed=seed,
    dtype=dtype)

  weight_one = tf.Variable(initializer(shape_one))
  weight_two = tf.Variable(initializer(shape_two))
  ```

  After:

  ```python
  initializer = tf.initializers.truncated_normal(
    mean=mean,
    seed=seed,
    stddev=stddev)

  weight_one = tf.Variable(initializer(shape_one, dtype=dtype))
  weight_two = tf.Variable(initializer(shape_two, dtype=dtype))
  ```

  #### How to Map Arguments

  | TF1 Arg Name          | TF2 Arg Name    | Note                       |
  | :-------------------- | :-------------- | :------------------------- |
  | `mean`               | `mean`        | No change to defaults |
  | `stddev`         | `stddev`        | Default changes from 1.0 to 0.05 |
  | `seed`             | `seed` | |
  | `dtype` | `dtype`   | The TF2 native api only takes it  |
  :                     :      : as a `__call__` arg, not a constructor arg. :
  | `partition_info`     | - |  (`__call__` arg in TF1) Not supported       |

  @end_compatibility
  """

  @deprecated_args(None,
                   "Call initializer instance with the dtype argument instead "
                   "of passing it to the constructor", "dtype")
  def __init__(self, mean=0.0, stddev=1.0, seed=None, dtype=dtypes.float32):
    self.mean = mean
    self.stddev = stddev
    self.seed = seed
    self.dtype = _assert_float_dtype(dtypes.as_dtype(dtype))

  def __call__(self, shape, dtype=None, partition_info=None):
    if dtype is None:
      dtype = self.dtype
    return random_ops.truncated_normal(
        shape, self.mean, self.stddev, dtype, seed=self.seed)

  def get_config(self):
    return {
        "mean": self.mean,
        "stddev": self.stddev,
        "seed": self.seed,
        "dtype": self.dtype.name
    }


@tf_export(v1=[
    "initializers.uniform_unit_scaling", "uniform_unit_scaling_initializer"
])
@deprecation.deprecated_endpoints("uniform_unit_scaling_initializer",
                                  "initializers.uniform_unit_scaling")
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
  See (Sussillo et al., 2014) for deeper motivation, experiments
  and the calculation of constants. In section 2.3 there, the constants were
  numerically computed: for a linear layer it's 1.0, relu: ~1.43, tanh: ~1.15.

  Args:
    factor: Float.  A multiplicative factor by which the values will be scaled.
    seed: A Python integer. Used to create random seeds. See
      `tf.compat.v1.set_random_seed` for behavior.
    dtype: Default data type, used if no `dtype` argument is provided when
      calling the initializer. Only floating point types are supported.
  References:
      [Sussillo et al., 2014](https://arxiv.org/abs/1412.6558)
      ([pdf](http://arxiv.org/pdf/1412.6558.pdf))
  """

  @deprecated_args(None,
                   "Call initializer instance with the dtype argument instead "
                   "of passing it to the constructor", "dtype")
  @deprecated(None,
              "Use tf.initializers.variance_scaling instead with distribution="
              "uniform to get equivalent behavior.")
  def __init__(self, factor=1.0, seed=None, dtype=dtypes.float32):
    self.factor = factor
    self.seed = seed
    self.dtype = _assert_float_dtype(dtypes.as_dtype(dtype))

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
    return random_ops.random_uniform(
        shape, -max_val, max_val, dtype, seed=self.seed)

  def get_config(self):
    return {"factor": self.factor, "seed": self.seed, "dtype": self.dtype.name}


@tf_export(v1=["initializers.variance_scaling", "variance_scaling_initializer"])
@deprecation.deprecated_endpoints("initializers.variance_scaling",
                                  "variance_scaling_initializer")
class VarianceScaling(Initializer):
  """Initializer capable of adapting its scale to the shape of weights tensors.

  @compatibility(TF2)
  Although it is a legacy `compat.v1` API, this symbol is compatible with eager
  execution and `tf.function`.

  To switch to TF2 APIs, move to using either
  `tf.initializers.variance_scaling` or `tf.keras.initializers.VarianceScaling`
  (neither from `compat.v1`) and
  pass the dtype when calling the initializer.

  #### Structural Mapping to TF2

  Before:

  ```python
  initializer = tf.compat.v1.variance_scaling_initializer(
    scale=scale,
    mode=mode,
    distribution=distribution
    seed=seed,
    dtype=dtype)

  weight_one = tf.Variable(initializer(shape_one))
  weight_two = tf.Variable(initializer(shape_two))
  ```

  After:

  ```python
  initializer = tf.keras.initializers.VarianceScaling(
    scale=scale,
    mode=mode,
    distribution=distribution
    seed=seed)

  weight_one = tf.Variable(initializer(shape_one, dtype=dtype))
  weight_two = tf.Variable(initializer(shape_two, dtype=dtype))
  ```

  #### How to Map Arguments

  | TF1 Arg Name       | TF2 Arg Name    | Note                       |
  | :----------------- | :-------------- | :------------------------- |
  | `scale`            | `scale`        | No change to defaults       |
  | `mode`             | `mode`         | No change to defaults       |
  | `distribution`     | `distribution` | No change to defaults.      |
  :                    :                : 'normal' maps to 'truncated_normal' :
  | `seed`             | `seed`         | |
  | `dtype`        |  `dtype` | The TF2 api only takes it  |
  :                :          : as a `__call__` arg, not a constructor arg. :
  | `partition_info`     | - |  (`__call__` arg in TF1) Not supported       |

  @end_compatibility

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
    distribution: Random distribution to use. One of "normal", "uniform".
    seed: A Python integer. Used to create random seeds. See
      `tf.compat.v1.set_random_seed` for behavior.
    dtype: Default data type, used if no `dtype` argument is provided when
      calling the initializer. Only floating point types are supported.

  Raises:
    ValueError: In case of an invalid value for the "scale", mode" or
      "distribution" arguments.
  """

  @deprecated_args(None,
                   "Call initializer instance with the dtype argument instead "
                   "of passing it to the constructor", "dtype")
  @deprecated_arg_values(
      None,
      "`normal` is a deprecated alias for `truncated_normal`",
      distribution="normal")
  def __init__(self,
               scale=1.0,
               mode="fan_in",
               distribution="truncated_normal",
               seed=None,
               dtype=dtypes.float32):
    if scale <= 0.:
      raise ValueError("Argument `scale` must be a positive float. Received: "
                       f"{scale}")
    if mode not in {"fan_in", "fan_out", "fan_avg"}:
      raise ValueError("Argument `mode` should be one of ('fan_in', 'fan_out', "
                       f"'fan_avg'). Received: {mode}")
    distribution = distribution.lower()
    if distribution not in {
        "normal", "uniform", "truncated_normal", "untruncated_normal"
    }:
      raise ValueError("Argument `distribution` should be one of ('normal', "
                       "uniform', 'truncated_normal', 'untruncated_normal'). "
                       f"Received: {distribution}")
    self.scale = scale
    self.mode = mode
    self.distribution = distribution
    self.seed = seed
    self.dtype = _assert_float_dtype(dtypes.as_dtype(dtype))

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
    if self.distribution == "normal" or self.distribution == "truncated_normal":
      # constant taken from scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
      stddev = math.sqrt(scale) / .87962566103423978
      return random_ops.truncated_normal(
          shape, 0.0, stddev, dtype, seed=self.seed)
    elif self.distribution == "untruncated_normal":
      stddev = math.sqrt(scale)
      return random_ops.random_normal(shape, 0.0, stddev, dtype, seed=self.seed)
    else:
      limit = math.sqrt(3.0 * scale)
      return random_ops.random_uniform(
          shape, -limit, limit, dtype, seed=self.seed)

  def get_config(self):
    return {
        "scale": self.scale,
        "mode": self.mode,
        "distribution": self.distribution,
        "seed": self.seed,
        "dtype": self.dtype.name
    }


@tf_export(v1=["initializers.orthogonal", "orthogonal_initializer"])
@deprecation.deprecated_endpoints("initializers.orthogonal",
                                  "orthogonal_initializer")
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
      `tf.compat.v1.set_random_seed` for behavior.
    dtype: Default data type, used if no `dtype` argument is provided when
      calling the initializer. Only floating point types are supported.
  References:
      [Saxe et al., 2014](https://openreview.net/forum?id=_wzZwKpTDF_9C)
      ([pdf](https://arxiv.org/pdf/1312.6120.pdf))
  """

  @deprecated_args(None,
                   "Call initializer instance with the dtype argument instead "
                   "of passing it to the constructor", "dtype")
  def __init__(self, gain=1.0, seed=None, dtype=dtypes.float32):
    self.gain = gain
    self.dtype = _assert_float_dtype(dtypes.as_dtype(dtype))
    self.seed = seed

  def __call__(self, shape, dtype=None, partition_info=None):
    if dtype is None:
      dtype = self.dtype
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
    num_rows = int(num_rows)
    num_cols = int(shape[-1])
    if num_rows < num_cols:
      flat_shape = (num_cols, num_rows)
    else:
      flat_shape = (num_rows, num_cols)

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
    return {"gain": self.gain, "seed": self.seed, "dtype": self.dtype.name}


# Note these haven't been ported to TF2.0. They are not currently visible and
# the tests are non trivial to port
class ConvolutionDeltaOrthogonal(Initializer):
  """Initializer that generates a delta orthogonal kernel for ConvNets.

  The shape of the tensor must have length 3, 4 or 5. The number of input
  filters must not exceed the number of output filters. The center pixels of the
  tensor form an orthogonal matrix. Other pixels are set to be zero. See
  algorithm 2 in (Xiao et al., 2018).


  Args:
    gain: Multiplicative factor to apply to the orthogonal matrix. Default is 1.
      The 2-norm of an input is multiplied by a factor of `gain` after applying
      this convolution.
    seed: A Python integer. Used to create random seeds. See
      `tf.compat.v1.set_random_seed` for behavior.
    dtype: Default data type, used if no `dtype` argument is provided when
      calling the initializer. Only floating point types are supported.
  References:
      [Xiao et al., 2018](http://proceedings.mlr.press/v80/xiao18a.html)
      ([pdf](http://proceedings.mlr.press/v80/xiao18a/xiao18a.pdf))
  """

  def __init__(self, gain=1.0, seed=None, dtype=dtypes.float32):
    self.gain = gain
    self.dtype = _assert_float_dtype(dtypes.as_dtype(dtype))
    self.seed = seed

  def __call__(self, shape, dtype=None, partition_info=None):
    if dtype is None:
      dtype = self.dtype
    # Check the shape
    if len(shape) < 3 or len(shape) > 5:
      raise ValueError("The tensor to initialize, specified by argument `shape`"
                       " must be at least three-dimensional and at most "
                       f"five-dimensional. Received shape={shape}")

    if shape[-2] > shape[-1]:
      raise ValueError(f"In_filters, specified by shape[-2]={shape[-2]} cannot "
                       "be greater than out_filters, specified by "
                       f"shape[-1]={shape[-1]}.")

    # Generate a random matrix
    a = random_ops.random_normal([shape[-1], shape[-1]],
                                 dtype=dtype,
                                 seed=self.seed)
    # Compute the qr factorization
    q, r = gen_linalg_ops.qr(a, full_matrices=False)
    # Make Q uniform
    d = array_ops.diag_part(r)
    q *= math_ops.sign(d)
    q = q[:shape[-2], :]
    q *= math_ops.cast(self.gain, dtype=dtype)
    if len(shape) == 3:
      weight = array_ops.scatter_nd([[(shape[0] - 1) // 2]],
                                    array_ops.expand_dims(q, 0), shape)
    elif len(shape) == 4:
      weight = array_ops.scatter_nd([[(shape[0] - 1) // 2,
                                      (shape[1] - 1) // 2]],
                                    array_ops.expand_dims(q, 0), shape)
    else:
      weight = array_ops.scatter_nd([[(shape[0] - 1) // 2, (shape[1] - 1) // 2,
                                      (shape[2] - 1) // 2]],
                                    array_ops.expand_dims(q, 0), shape)
    return weight

  def get_config(self):
    return {"gain": self.gain, "seed": self.seed, "dtype": self.dtype.name}


class ConvolutionOrthogonal(Initializer):
  """Initializer that generates orthogonal kernel for ConvNets.

  Base class used to construct 1D, 2D and 3D orthogonal kernels for convolution.

  Args:
    gain: multiplicative factor to apply to the orthogonal matrix. Default is 1.
      The 2-norm of an input is multiplied by a factor of `gain` after applying
      this convolution.
    seed: A Python integer. Used to create random seeds. See
      `tf.compat.v1.set_random_seed` for behavior.
    dtype: Default data type, used if no `dtype` argument is provided when
      calling the initializer. Only floating point types are supported.
  References:
      [Xiao et al., 2018](http://proceedings.mlr.press/v80/xiao18a.html)
      ([pdf](http://proceedings.mlr.press/v80/xiao18a/xiao18a.pdf))
  """

  def __init__(self, gain=1.0, seed=None, dtype=dtypes.float32):
    self.gain = gain
    self.dtype = _assert_float_dtype(dtypes.as_dtype(dtype))
    self.seed = seed

  def __call__(self, shape, dtype=None, partition_info=None):
    raise NotImplementedError

  def get_config(self):
    return {"gain": self.gain, "seed": self.seed, "dtype": self.dtype.name}

  # Helper functions.
  def _orthogonal_matrix(self, n):
    """Construct an n x n orthogonal matrix.

    Args:
      n: Dimension.

    Returns:
      A n x n orthogonal matrix.
    """
    a = random_ops.random_normal([n, n], dtype=self.dtype, seed=self.seed)
    if self.seed:
      self.seed += 1
    q, r = gen_linalg_ops.qr(a)
    d = array_ops.diag_part(r)
    # make q uniform
    q *= math_ops.sign(d)
    return q

  def _symmetric_projection(self, n):
    """Compute a n x n symmetric projection matrix.

    Args:
      n: Dimension.

    Returns:
      A n x n symmetric projection matrix, i.e. a matrix P s.t. P=P*P, P=P^T.
    """
    q = self._orthogonal_matrix(n)
    # randomly zeroing out some columns
    mask = math_ops.cast(
        random_ops.random_normal([n], seed=self.seed) > 0, self.dtype)
    if self.seed:
      self.seed += 1
    c = math_ops.multiply(q, mask)
    return math_ops.matmul(c, array_ops.matrix_transpose(c))


class ConvolutionOrthogonal2D(ConvolutionOrthogonal):
  """Initializer that generates a 2D orthogonal kernel for ConvNets.

  The shape of the tensor must have length 4. The number of input
  filters must not exceed the number of output filters.
  The orthogonality(==isometry) is exact when the inputs are circular padded.
  There are finite-width effects with non-circular padding (e.g. zero padding).
  See algorithm 1 in (Xiao et al., 2018).

  Args:
    gain: Multiplicative factor to apply to the orthogonal matrix. Default is 1.
      This has the effect of scaling the output 2-norm by a factor of `gain`.
    seed: A Python integer. Used to create random seeds. See
      `tf.compat.v1.set_random_seed` for behavior.
    dtype: Default data type, used if no `dtype` argument is provided when
      calling the initializer. Only floating point types are supported.
  References:
      [Xiao et al., 2018](http://proceedings.mlr.press/v80/xiao18a.html)
      ([pdf](http://proceedings.mlr.press/v80/xiao18a/xiao18a.pdf))
  """

  def __call__(self, shape, dtype=None, partition_info=None):
    if dtype is None:
      dtype = self.dtype
    if len(shape) != 4:
      raise ValueError("The tensor to initialize, specified by argument `shape`"
                       f" must be four-dimensional. Received: {shape}")

    if shape[-2] > shape[-1]:
      raise ValueError(f"In_filters, specified by shape[-2]={shape[-2]} cannot "
                       "be greater than out_filters, specified by "
                       f"shape[-1]={shape[-1]}.")

    if shape[0] != shape[1]:
      raise ValueError(f"Kernel sizes, specified by shape[0]={shape[0]} and "
                       f"shape[1]={shape[1]} must be equal.")

    kernel = self._orthogonal_kernel(shape[0], shape[2], shape[3])
    kernel *= math_ops.cast(self.gain, dtype=dtype)
    return kernel

  def _dict_to_tensor(self, x, k1, k2):
    """Convert a dictionary to a tensor.

    Args:
      x: A k1 * k2 dictionary.
      k1: First dimension of x.
      k2: Second dimension of x.

    Returns:
      A k1 * k2 tensor.
    """

    return array_ops.stack([array_ops.stack([x[i, j] for j in range(k2)])
                            for i in range(k1)])

  def _block_orth(self, p1, p2):
    """Construct a 2 x 2 kernel.

    Used to construct orthgonal kernel.

    Args:
      p1: A symmetric projection matrix.
      p2: A symmetric projection matrix.

    Returns:
      A 2 x 2 kernel [[p1p2,         p1(1-p2)],
                      [(1-p1)p2, (1-p1)(1-p2)]].
    Raises:
      ValueError: If the dimensions of p1 and p2 are different.
    """
    if p1.shape.as_list() != p2.shape.as_list():
      raise ValueError("The dimension of the matrices must be the same. "
                       f"Received p1.shape={p1.shape} and p2.shape={p2.shape}.")
    n = p1.shape.as_list()[0]
    kernel2x2 = {}
    eye = linalg_ops_impl.eye(n, dtype=self.dtype)
    kernel2x2[0, 0] = math_ops.matmul(p1, p2)
    kernel2x2[0, 1] = math_ops.matmul(p1, (eye - p2))
    kernel2x2[1, 0] = math_ops.matmul((eye - p1), p2)
    kernel2x2[1, 1] = math_ops.matmul((eye - p1), (eye - p2))

    return kernel2x2

  def _matrix_conv(self, m1, m2):
    """Matrix convolution.

    Args:
      m1: A k x k dictionary, each element is a n x n matrix.
      m2: A l x l dictionary, each element is a n x n matrix.

    Returns:
      (k + l - 1) * (k + l - 1) dictionary each element is a n x n matrix.
    Raises:
      ValueError: if the entries of m1 and m2 are of different dimensions.
    """

    n = (m1[0, 0]).shape.as_list()[0]
    if n != (m2[0, 0]).shape.as_list()[0]:
      raise ValueError("The entries in matrices m1 and m2 must have the same "
                       f"dimensions. Received m1[0, 0].shape={m1[0, 0].shape} "
                       f"and m2[0, 0].shape={m2[0, 0].shape}.")
    k = int(np.sqrt(len(m1)))
    l = int(np.sqrt(len(m2)))
    result = {}
    size = k + l - 1
    # Compute matrix convolution between m1 and m2.
    for i in range(size):
      for j in range(size):
        result[i, j] = array_ops.zeros([n, n], self.dtype)
        for index1 in range(min(k, i + 1)):
          for index2 in range(min(k, j + 1)):
            if (i - index1) < l and (j - index2) < l:
              result[i, j] += math_ops.matmul(m1[index1, index2],
                                              m2[i - index1, j - index2])
    return result

  def _orthogonal_kernel(self, ksize, cin, cout):
    """Construct orthogonal kernel for convolution.

    Args:
      ksize: Kernel size.
      cin: Number of input channels.
      cout: Number of output channels.

    Returns:
      An [ksize, ksize, cin, cout] orthogonal kernel.
    Raises:
      ValueError: If cin > cout.
    """
    if cin > cout:
      raise ValueError(f"The number of input channels (cin={cin}) cannot exceed"
                       f" the number of output channels (cout={cout}).")
    orth = self._orthogonal_matrix(cout)[0:cin, :]
    if ksize == 1:
      return array_ops.expand_dims(array_ops.expand_dims(orth, 0), 0)

    p = self._block_orth(
        self._symmetric_projection(cout), self._symmetric_projection(cout))
    for _ in range(ksize - 2):
      temp = self._block_orth(
          self._symmetric_projection(cout), self._symmetric_projection(cout))
      p = self._matrix_conv(p, temp)
    for i in range(ksize):
      for j in range(ksize):
        p[i, j] = math_ops.matmul(orth, p[i, j])

    return self._dict_to_tensor(p, ksize, ksize)


class ConvolutionOrthogonal1D(ConvolutionOrthogonal):
  """Initializer that generates a 1D orthogonal kernel for ConvNets.

  The shape of the tensor must have length 3. The number of input
  filters must not exceed the number of output filters.
  The orthogonality(==isometry) is exact when the inputs are circular padded.
  There are finite-width effects with non-circular padding (e.g. zero padding).
  See algorithm 1 in (Xiao et al., 2018).

  Args:
    gain: Multiplicative factor to apply to the orthogonal matrix. Default is 1.
      The 2-norm of an input is multiplied by a factor of `gain` after applying
      this convolution.
    seed: A Python integer. Used to create random seeds. See
      `tf.compat.v1.set_random_seed` for behavior.
    dtype: Default data type, used if no `dtype` argument is provided when
      calling the initializer. Only floating point types are supported.
  References:
      [Xiao et al., 2018](http://proceedings.mlr.press/v80/xiao18a.html)
      ([pdf](http://proceedings.mlr.press/v80/xiao18a/xiao18a.pdf))
  """

  def __call__(self, shape, dtype=None, partition_info=None):
    if dtype is None:
      dtype = self.dtype
    if len(shape) != 3:
      raise ValueError("The tensor to initialize, specified by argument `shape`"
                       f" must be three-dimensional. Received shape={shape}")

    if shape[-2] > shape[-1]:
      raise ValueError(f"In_filters, specified by shape[-2]={shape[-2]} cannot "
                       "be greater than out_filters, specified by "
                       f"shape[-1]={shape[-1]}.")

    kernel = self._orthogonal_kernel(shape[0], shape[-2], shape[-1])
    kernel *= math_ops.cast(self.gain, dtype=dtype)
    return kernel

  def _dict_to_tensor(self, x, k):
    """Convert a dictionary to a tensor.

    Args:
      x: A dictionary of length k.
      k: Dimension of x.

    Returns:
      A tensor with the same dimension.
    """

    return array_ops.stack([x[i] for i in range(k)])

  def _block_orth(self, projection_matrix):
    """Construct a kernel.

    Used to construct orthgonal kernel.

    Args:
      projection_matrix: A symmetric projection matrix of size n x n.

    Returns:
      [projection_matrix, (1 - projection_matrix)].
    """
    n = projection_matrix.shape.as_list()[0]
    kernel = {}
    eye = linalg_ops_impl.eye(n, dtype=self.dtype)
    kernel[0] = projection_matrix
    kernel[1] = eye - projection_matrix
    return kernel

  def _matrix_conv(self, m1, m2):
    """Matrix convolution.

    Args:
      m1: A dictionary of length k, each element is a n x n matrix.
      m2: A dictionary of length l, each element is a n x n matrix.

    Returns:
      (k + l - 1)  dictionary each element is a n x n matrix.
    Raises:
      ValueError: Ff the entries of m1 and m2 are of different dimensions.
    """

    n = (m1[0]).shape.as_list()[0]
    if n != (m2[0]).shape.as_list()[0]:
      raise ValueError("The entries in matrices m1 and m2 must have the same "
                       f"dimensions. Received m1[0].shape={m1[0].shape} "
                       f"and m2[0].shape={m2[0].shape}.")
    k = len(m1)
    l = len(m2)
    result = {}
    size = k + l - 1
    # Compute matrix convolution between m1 and m2.
    for i in range(size):
      result[i] = array_ops.zeros([n, n], self.dtype)
      for index in range(min(k, i + 1)):
        if (i - index) < l:
          result[i] += math_ops.matmul(m1[index], m2[i - index])
    return result

  def _orthogonal_kernel(self, ksize, cin, cout):
    """Construct orthogonal kernel for convolution.

    Args:
      ksize: Kernel size.
      cin: Number of input channels.
      cout: Number of output channels.

    Returns:
      An [ksize, ksize, cin, cout] orthogonal kernel.
    Raises:
      ValueError: If cin > cout.
    """
    if cin > cout:
      raise ValueError(f"The number of input channels (cin={cin}) cannot exceed"
                       f" the number of output channels (cout={cout}).")
    orth = self._orthogonal_matrix(cout)[0:cin, :]
    if ksize == 1:
      return array_ops.expand_dims(orth, 0)

    p = self._block_orth(self._symmetric_projection(cout))
    for _ in range(ksize - 2):
      temp = self._block_orth(self._symmetric_projection(cout))
      p = self._matrix_conv(p, temp)
    for i in range(ksize):
      p[i] = math_ops.matmul(orth, p[i])

    return self._dict_to_tensor(p, ksize)


class ConvolutionOrthogonal3D(ConvolutionOrthogonal):
  """Initializer that generates a 3D orthogonal kernel for ConvNets.

  The shape of the tensor must have length 5. The number of input
  filters must not exceed the number of output filters.
  The orthogonality(==isometry) is exact when the inputs are circular padded.
  There are finite-width effects with non-circular padding (e.g. zero padding).
  See algorithm 1 (Xiao et al., 2018).

  Args:
    gain: Multiplicative factor to apply to the orthogonal matrix. Default is 1.
      The 2-norm of an input is multiplied by a factor of `gain` after applying
      this convolution.
    seed: A Python integer. Used to create random seeds. See
      `tf.compat.v1.set_random_seed` for behavior.
    dtype: Default data type, used if no `dtype` argument is provided when
      calling the initializer. Only floating point types are supported.
  References:
      [Xiao et al., 2018](http://proceedings.mlr.press/v80/xiao18a.html)
      ([pdf](http://proceedings.mlr.press/v80/xiao18a/xiao18a.pdf))
  """

  def __call__(self, shape, dtype=None, partition_info=None):
    if dtype is None:
      dtype = self.dtype
    if len(shape) != 5:
      raise ValueError("The tensor to initialize, specified by argument `shape`"
                       f" must be five-dimensional. Received shape={shape}")

    if shape[-2] > shape[-1]:
      raise ValueError(f"In_filters, specified by shape[-2]={shape[-2]} cannot "
                       "be greater than out_filters, specified by "
                       f"shape[-1]={shape[-1]}.")

    if shape[0] != shape[1] or shape[0] != shape[2]:
      raise ValueError(f"Kernel sizes, specified by shape[0]={shape[0]},  "
                       f"shape[1]={shape[1]} and shape[2]={shape[2]} must be "
                       "equal.")

    kernel = self._orthogonal_kernel(shape[0], shape[-2], shape[-1])
    kernel *= math_ops.cast(self.gain, dtype=dtype)
    return kernel

  def _dict_to_tensor(self, x, k1, k2, k3):
    """Convert a dictionary to a tensor.

    Args:
      x: A k1 * k2 dictionary.
      k1: First dimension of x.
      k2: Second dimension of x.
      k3: Third dimension of x.

    Returns:
      A k1 * k2 * k3 tensor.
    """

    return array_ops.stack([array_ops.stack(
        [array_ops.stack([x[i, j, k] for k in range(k3)])
         for j in range(k2)]) for i in range(k1)])

  def _block_orth(self, p1, p2, p3):
    """Construct a 3 x 3 kernel.

    Used to construct orthgonal kernel.

    Args:
      p1: A symmetric projection matrix.
      p2: A symmetric projection matrix.
      p3: A symmetric projection matrix.

    Returns:
      A 2 x 2 x 2 kernel.
    Raises:
      ValueError: If the dimensions of p1, p2 and p3 are different.
    """
    p1_shape = p1.shape.as_list()
    if p1_shape != p2.shape.as_list() or p1_shape != p3.shape.as_list():
      raise ValueError("The dimension of the matrices must be the same. "
                       f"Received p1.shape={p1.shape}, p2.shape={p2.shape} and"
                       f" p3.shape={p3.shape}.")
    n = p1_shape[0]
    eye = linalg_ops_impl.eye(n, dtype=self.dtype)
    kernel2x2x2 = {}

    def matmul(p1, p2, p3):
      return math_ops.matmul(math_ops.matmul(p1, p2), p3)

    def cast(i, p):
      """Return p or (1-p)."""
      return i * p + (1 - i) * (eye - p)

    for i in [0, 1]:
      for j in [0, 1]:
        for k in [0, 1]:
          kernel2x2x2[i, j, k] = matmul(cast(i, p1), cast(j, p2), cast(k, p3))
    return kernel2x2x2

  def _matrix_conv(self, m1, m2):
    """Matrix convolution.

    Args:
      m1: is a k x k x k  dictionary, each element is a n x n matrix.
      m2: is a l x l x l dictionary, each element is a n x n matrix.

    Returns:
      (k + l - 1) x (k + l - 1) x (k + l - 1) dictionary each
      element is a n x n matrix.
    Raises:
      ValueError: if the entries of m1 and m2 are of different dimensions.
    """

    n = (m1[0, 0, 0]).shape.as_list()[0]
    if n != (m2[0, 0, 0]).shape.as_list()[0]:
      raise ValueError("The entries in matrices m1 and m2 must have the same "
                       "dimensions. Received m1[0, 0, 0].shape="
                       f"{m1[0, 0, 0].shape} and m2[0, 0, 0].shape="
                       f"{m2[0, 0, 0].shape}.")
    k = int(np.cbrt(len(m1)))
    l = int(np.cbrt(len(m2)))
    result = {}
    size = k + l - 1
    # Compute matrix convolution between m1 and m2.
    for i in range(size):
      for j in range(size):
        for r in range(size):
          result[i, j, r] = array_ops.zeros([n, n], self.dtype)
          for index1 in range(min(k, i + 1)):
            for index2 in range(min(k, j + 1)):
              for index3 in range(min(k, r + 1)):
                if (i - index1) < l and (j - index2) < l and (r - index3) < l:
                  result[i, j, r] += math_ops.matmul(
                      m1[index1, index2, index3],
                      m2[i - index1, j - index2, r - index3])
    return result

  def _orthogonal_kernel(self, ksize, cin, cout):
    """Construct orthogonal kernel for convolution.

    Args:
      ksize: Kernel size.
      cin: Number of input channels.
      cout: Number of output channels.

    Returns:
      An [ksize, ksize, ksize, cin, cout] orthogonal kernel.
    Raises:
      ValueError: If cin > cout.
    """
    if cin > cout:
      raise ValueError(f"The number of input channels (cin={cin}) cannot exceed"
                       f" the number of output channels (cout={cout}).")
    orth = self._orthogonal_matrix(cout)[0:cin, :]
    if ksize == 1:
      return array_ops.expand_dims(
          array_ops.expand_dims(array_ops.expand_dims(orth, 0), 0), 0)

    p = self._block_orth(
        self._symmetric_projection(cout), self._symmetric_projection(cout),
        self._symmetric_projection(cout))
    for _ in range(ksize - 2):
      temp = self._block_orth(
          self._symmetric_projection(cout), self._symmetric_projection(cout),
          self._symmetric_projection(cout))
      p = self._matrix_conv(p, temp)
    for i in range(ksize):
      for j in range(ksize):
        for k in range(ksize):
          p[i, j, k] = math_ops.matmul(orth, p[i, j, k])

    return self._dict_to_tensor(p, ksize, ksize, ksize)


@tf_export(v1=["initializers.identity"])
@deprecation.deprecated_endpoints("initializers.identity")
class Identity(Initializer):
  """Initializer that generates the identity matrix.

  Only use for 2D matrices.

  Args:
    gain: Multiplicative factor to apply to the identity matrix.
    dtype: Default data type, used if no `dtype` argument is provided when
      calling the initializer. Only floating point types are supported.
  """

  @deprecated_args(None,
                   "Call initializer instance with the dtype argument instead "
                   "of passing it to the constructor", "dtype")
  def __init__(self, gain=1.0, dtype=dtypes.float32):
    self.gain = gain
    self.dtype = _assert_float_dtype(dtypes.as_dtype(dtype))

  def __call__(self, shape, dtype=None, partition_info=None):
    full_shape = shape if partition_info is None else partition_info.full_shape
    if len(full_shape) != 2:
      raise ValueError("The tensor to initialize, specified by argument `shape`"
                       " must be at least two-dimensional. Received shape="
                       f"{shape}")
    if dtype is None:
      dtype = self.dtype
    if isinstance(full_shape, tensor_shape.TensorShape):
      full_shape = full_shape.as_list()
    initializer = linalg_ops_impl.eye(*full_shape, dtype=dtype)
    if partition_info is not None:
      initializer = array_ops.slice(initializer, partition_info.var_offset,
                                    shape)
    return self.gain * initializer

  def get_config(self):
    return {"gain": self.gain, "dtype": self.dtype.name}


@tf_export(v1=["glorot_uniform_initializer", "initializers.glorot_uniform"])
@deprecation.deprecated_endpoints("glorot_uniform_initializer",
                                  "initializers.glorot_uniform")
class GlorotUniform(VarianceScaling):
  """The Glorot uniform initializer, also called Xavier uniform initializer.

  It draws samples from a uniform distribution within [-limit, limit]
  where `limit` is `sqrt(6 / (fan_in + fan_out))`
  where `fan_in` is the number of input units in the weight tensor
  and `fan_out` is the number of output units in the weight tensor.

  Args:
    seed: A Python integer. Used to create random seeds. See
      `tf.compat.v1.set_random_seed` for behavior.
    dtype: Default data type, used if no `dtype` argument is provided when
      calling the initializer. Only floating point types are supported.
  References:
      [Glorot et al., 2010](http://proceedings.mlr.press/v9/glorot10a.html)
      ([pdf](http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf))
  """

  @deprecated_args(None,
                   "Call initializer instance with the dtype argument instead "
                   "of passing it to the constructor", "dtype")
  def __init__(self, seed=None, dtype=dtypes.float32):
    super(GlorotUniform, self).__init__(
        scale=1.0, mode="fan_avg", distribution="uniform", seed=seed)

  def get_config(self):
    return {"seed": self.seed, "dtype": self.dtype.name}


@tf_export(v1=["glorot_normal_initializer", "initializers.glorot_normal"])
@deprecation.deprecated_endpoints("glorot_normal_initializer",
                                  "initializers.glorot_normal")
class GlorotNormal(VarianceScaling):
  """The Glorot normal initializer, also called Xavier normal initializer.

  It draws samples from a truncated normal distribution centered on 0
  with standard deviation (after truncation) given by
  `stddev = sqrt(2 / (fan_in + fan_out))` where `fan_in` is the number
  of input units in the weight tensor and `fan_out` is the number of
  output units in the weight tensor.

  Args:
    seed: A Python integer. Used to create random seeds. See
      `tf.compat.v1.set_random_seed` for behavior.
    dtype: Default data type, used if no `dtype` argument is provided when
      calling the initializer. Only floating point types are supported.
  References:
      [Glorot et al., 2010](http://proceedings.mlr.press/v9/glorot10a.html)
      ([pdf](http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf))
  """

  @deprecated_args(None,
                   "Call initializer instance with the dtype argument instead "
                   "of passing it to the constructor", "dtype")
  def __init__(self, seed=None, dtype=dtypes.float32):
    super(GlorotNormal, self).__init__(
        scale=1.0, mode="fan_avg", distribution="truncated_normal", seed=seed)

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
uniform_unit_scaling_initializer = UniformUnitScaling
variance_scaling_initializer = VarianceScaling
glorot_uniform_initializer = GlorotUniform
glorot_normal_initializer = GlorotNormal
orthogonal_initializer = Orthogonal
identity_initializer = Identity
convolutional_delta_orthogonal = ConvolutionDeltaOrthogonal
convolutional_orthogonal_1d = ConvolutionOrthogonal1D
convolutional_orthogonal_2d = ConvolutionOrthogonal2D
convolutional_orthogonal_3d = ConvolutionOrthogonal3D
# pylint: enable=invalid-name


@tf_export(v1=["initializers.lecun_normal"])
def lecun_normal(seed=None):
  """LeCun normal initializer.

  It draws samples from a truncated normal distribution centered on 0
  with standard deviation (after truncation) given by
  `stddev = sqrt(1 / fan_in)` where `fan_in` is the number of
  input units in the weight tensor.

  Args:
      seed: A Python integer. Used to seed the random generator.

  Returns:
      An initializer.

  References:
      - Self-Normalizing Neural Networks,
      [Klambauer et al.,
      2017](https://papers.nips.cc/paper/6698-self-normalizing-neural-networks)
      # pylint: disable=line-too-long
      ([pdf](https://papers.nips.cc/paper/6698-self-normalizing-neural-networks.pdf))
      - Efficient Backprop,
      [Lecun et al., 1998](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf)
  """
  return VarianceScaling(
      scale=1., mode="fan_in", distribution="truncated_normal", seed=seed)


@tf_export(v1=["initializers.lecun_uniform"])
def lecun_uniform(seed=None):
  """LeCun uniform initializer.

  It draws samples from a uniform distribution within [-limit, limit]
  where `limit` is `sqrt(3 / fan_in)`
  where `fan_in` is the number of input units in the weight tensor.

  Args:
      seed: A Python integer. Used to seed the random generator.

  Returns:
      An initializer.

  References:
      - Self-Normalizing Neural Networks,
      [Klambauer et al.,
      2017](https://papers.nips.cc/paper/6698-self-normalizing-neural-networks)
      # pylint: disable=line-too-long
      ([pdf](https://papers.nips.cc/paper/6698-self-normalizing-neural-networks.pdf))
      - Efficient Backprop,
      [Lecun et al., 1998](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf)
  """
  return VarianceScaling(
      scale=1., mode="fan_in", distribution="uniform", seed=seed)


@tf_export(v1=["initializers.he_normal"])
def he_normal(seed=None):
  """He normal initializer.

  It draws samples from a truncated normal distribution centered on 0
  with standard deviation (after truncation) given by
  `stddev = sqrt(2 / fan_in)` where `fan_in` is the number of
  input units in the weight tensor.

  Args:
      seed: A Python integer. Used to seed the random generator.

  Returns:
      An initializer.

  References:
      [He et al., 2015]
      (https://www.cv-foundation.org/openaccess/content_iccv_2015/html/He_Delving_Deep_into_ICCV_2015_paper.html)
      # pylint: disable=line-too-long
      ([pdf](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/He_Delving_Deep_into_ICCV_2015_paper.pdf))
  """
  return VarianceScaling(
      scale=2., mode="fan_in", distribution="truncated_normal", seed=seed)


@tf_export(v1=["initializers.he_uniform"])
def he_uniform(seed=None):
  """He uniform variance scaling initializer.

  It draws samples from a uniform distribution within [-limit, limit]
  where `limit` is `sqrt(6 / fan_in)`
  where `fan_in` is the number of input units in the weight tensor.

  Args:
      seed: A Python integer. Used to seed the random generator.

  Returns:
      An initializer.

  References:
      [He et al., 2015]
      (https://www.cv-foundation.org/openaccess/content_iccv_2015/html/He_Delving_Deep_into_ICCV_2015_paper.html)
      # pylint: disable=line-too-long
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
    A tuple of integer scalars (fan_in, fan_out).
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
    receptive_field_size = 1
    for dim in shape[:-2]:
      receptive_field_size *= dim
    fan_in = shape[-2] * receptive_field_size
    fan_out = shape[-1] * receptive_field_size
  return int(fan_in), int(fan_out)


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
    raise ValueError("Argument `dtype` is expected to be floating point. "
                     f"Received: {dtype}.")
  return dtype
