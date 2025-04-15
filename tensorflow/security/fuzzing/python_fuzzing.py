# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Helper class for TF Python fuzzing."""

import atheris
import tensorflow as tf

_MIN_INT = -10000
_MAX_INT = 10000

_MIN_FLOAT = -10000.0
_MAX_FLOAT = 10000.0

_MIN_LENGTH = 0
_MAX_LENGTH = 10000

# Max shape can be 8 in length and randomized from 0-8 without running into an
# OOM error.
_MIN_SIZE = 0
_MAX_SIZE = 8

_TF_DTYPES = [
    tf.half, tf.float16, tf.float32, tf.float64, tf.bfloat16, tf.complex64,
    tf.complex128, tf.int8, tf.uint8, tf.uint16, tf.uint32, tf.uint64, tf.int16,
    tf.int32, tf.int64, tf.bool, tf.string, tf.qint8, tf.quint8, tf.qint16,
    tf.quint16, tf.qint32, tf.resource, tf.variant
]

# All types supported by tf.random.uniform
_TF_RANDOM_DTYPES = [tf.float16, tf.float32, tf.float64, tf.int32, tf.int64]


class FuzzingHelper(object):
  """FuzzingHelper makes handling FuzzedDataProvider easier with TensorFlow Python fuzzing."""

  def __init__(self, input_bytes):
    """FuzzingHelper initializer.

    Args:
      input_bytes: Input randomized bytes used to create a FuzzedDataProvider.
    """
    self.fdp = atheris.FuzzedDataProvider(input_bytes)

  def get_bool(self):
    """Consume a bool.

    Returns:
      Consumed a bool based on input bytes and constraints.
    """
    return self.fdp.ConsumeBool()

  def get_int(self, min_int=_MIN_INT, max_int=_MAX_INT):
    """Consume a signed integer with given constraints.

    Args:
      min_int: Minimum allowed integer.
      max_int: Maximum allowed integer.

    Returns:
      Consumed integer based on input bytes and constraints.
    """
    return self.fdp.ConsumeIntInRange(min_int, max_int)

  def get_float(self, min_float=_MIN_FLOAT, max_float=_MAX_FLOAT):
    """Consume a float with given constraints.

    Args:
      min_float: Minimum allowed float.
      max_float: Maximum allowed float.

    Returns:
      Consumed float based on input bytes and constraints.
    """
    return self.fdp.ConsumeFloatInRange(min_float, max_float)

  def get_int_list(self,
                   min_length=_MIN_LENGTH,
                   max_length=_MAX_LENGTH,
                   min_int=_MIN_INT,
                   max_int=_MAX_INT):
    """Consume a signed integer list with given constraints.

    Args:
      min_length: The minimum length of the list.
      max_length: The maximum length of the list.
      min_int: Minimum allowed integer.
      max_int: Maximum allowed integer.

    Returns:
      Consumed integer list based on input bytes and constraints.
    """
    length = self.get_int(min_length, max_length)
    return self.fdp.ConsumeIntListInRange(length, min_int, max_int)

  def get_float_list(self, min_length=_MIN_LENGTH, max_length=_MAX_LENGTH):
    """Consume a float list with given constraints.

    Args:
      min_length: The minimum length of the list.
      max_length: The maximum length of the list.

    Returns:
      Consumed integer list based on input bytes and constraints.
    """
    length = self.get_int(min_length, max_length)
    return self.fdp.ConsumeFloatListInRange(length, _MIN_FLOAT, _MAX_FLOAT)

  def get_int_or_float_list(self,
                            min_length=_MIN_LENGTH,
                            max_length=_MAX_LENGTH):
    """Consume a signed integer or float list with given constraints based on a consumed bool.

    Args:
      min_length: The minimum length of the list.
      max_length: The maximum length of the list.

    Returns:
      Consumed integer or float list based on input bytes and constraints.
    """
    if self.get_bool():
      return self.get_int_list(min_length, max_length)
    else:
      return self.get_float_list(min_length, max_length)

  def get_tf_dtype(self, allowed_set=None):
    """Return a random tensorflow dtype.

    Args:
      allowed_set: An allowlisted set of dtypes to choose from instead of all of
      them.

    Returns:
      A random type from the list containing all TensorFlow types.
    """
    if allowed_set:
      index = self.get_int(0, len(allowed_set) - 1)
      if allowed_set[index] not in _TF_DTYPES:
        raise tf.errors.InvalidArgumentError(
            None, None,
            'Given dtype {} is not accepted.'.format(allowed_set[index]))
      return allowed_set[index]
    else:
      index = self.get_int(0, len(_TF_DTYPES) - 1)
      return _TF_DTYPES[index]

  def get_string(self, byte_count=_MAX_INT):
    """Consume a string with given constraints based on a consumed bool.

    Args:
      byte_count: Byte count that defaults to _MAX_INT.

    Returns:
      Consumed string based on input bytes and constraints.
    """
    return self.fdp.ConsumeString(byte_count)

  def get_random_numeric_tensor(self,
                                dtype=None,
                                min_size=_MIN_SIZE,
                                max_size=_MAX_SIZE,
                                min_val=_MIN_INT,
                                max_val=_MAX_INT):
    """Return a tensor of random shape and values.

    Generated tensors are capped at dimension sizes of 8, as 2^32 bytes of
    requested memory crashes the fuzzer (see b/34190148).
    Returns only type that tf.random.uniform can generate. If you need a
    different type, consider using tf.cast.

    Args:
      dtype: Type of tensor, must of one of the following types: float16,
        float32, float64, int32, or int64
      min_size: Minimum size of returned tensor
      max_size: Maximum size of returned tensor
      min_val: Minimum value in returned tensor
      max_val: Maximum value in returned tensor

    Returns:
      Tensor of random shape filled with uniformly random numeric values.
    """
    # Max shape can be 8 in length and randomized from 0-8 without running into
    # an OOM error.
    if max_size > 8:
      raise tf.errors.InvalidArgumentError(
          None, None,
          'Given size of {} will result in an OOM error'.format(max_size))

    seed = self.get_int()
    shape = self.get_int_list(
        min_length=min_size,
        max_length=max_size,
        min_int=min_size,
        max_int=max_size)

    if dtype is None:
      dtype = self.get_tf_dtype(allowed_set=_TF_RANDOM_DTYPES)
    elif dtype not in _TF_RANDOM_DTYPES:
      raise tf.errors.InvalidArgumentError(
          None, None,
          'Given dtype {} is not accepted in get_random_numeric_tensor'.format(
              dtype))

    return tf.random.uniform(
        shape=shape, minval=min_val, maxval=max_val, dtype=dtype, seed=seed)
