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
"""Keras initializer classes (soon to be replaced with core TF initializers).
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np
import six

from tensorflow.contrib.keras.python.keras import backend as K
from tensorflow.contrib.keras.python.keras.utils.generic_utils import deserialize_keras_object
from tensorflow.contrib.keras.python.keras.utils.generic_utils import serialize_keras_object
from tensorflow.python.framework import tensor_shape


class Initializer(object):
  """Initializer base class: all initializers inherit from this class.
  """

  def __call__(self, shape, dtype=None):
    raise NotImplementedError

  def get_config(self):
    return {}

  @classmethod
  def from_config(cls, config):
    return cls(**config)


class Zeros(Initializer):
  """Initializer that generates tensors initialized to 0.
  """

  def __call__(self, shape, dtype=None):
    return K.constant(0, shape=shape, dtype=dtype)


class Ones(Initializer):
  """Initializer that generates tensors initialized to 1.
  """

  def __call__(self, shape, dtype=None):
    return K.constant(1, shape=shape, dtype=dtype)


class Constant(Initializer):
  """Initializer that generates tensors initialized to a constant value.

  Arguments:
      value: float; the value of the generator tensors.
  """

  def __init__(self, value=0):
    self.value = value

  def __call__(self, shape, dtype=None):
    return K.constant(self.value, shape=shape, dtype=dtype)

  def get_config(self):
    return {'value': self.value}


class RandomNormal(Initializer):
  """Initializer that generates tensors with a normal distribution.

  Arguments:
      mean: a python scalar or a scalar tensor. Mean of the random values
        to generate.
      stddev: a python scalar or a scalar tensor. Standard deviation of the
        random values to generate.
      seed: A Python integer. Used to seed the random generator.
  """

  def __init__(self, mean=0., stddev=0.05, seed=None):
    self.mean = mean
    self.stddev = stddev
    self.seed = seed

  def __call__(self, shape, dtype=None):
    return K.random_normal(
        shape, self.mean, self.stddev, dtype=dtype, seed=self.seed)

  def get_config(self):
    return {'mean': self.mean, 'stddev': self.stddev, 'seed': self.seed}


class RandomUniform(Initializer):
  """Initializer that generates tensors with a uniform distribution.

  Arguments:
      minval: A python scalar or a scalar tensor. Lower bound of the range
        of random values to generate.
      maxval: A python scalar or a scalar tensor. Upper bound of the range
        of random values to generate.  Defaults to 1 for float types.
      seed: A Python integer. Used to seed the random generator.
  """

  def __init__(self, minval=-0.05, maxval=0.05, seed=None):
    self.minval = minval
    self.maxval = maxval
    self.seed = seed

  def __call__(self, shape, dtype=None):
    return K.random_uniform(
        shape, self.minval, self.maxval, dtype=dtype, seed=self.seed)

  def get_config(self):
    return {
        'minval': self.minval,
        'maxval': self.maxval,
        'seed': self.seed,
    }


class TruncatedNormal(Initializer):
  """Initializer that generates a truncated normal distribution.

  These values are similar to values from a `RandomNormal`
  except that values more than two standard deviations from the mean
  are discarded and re-drawn. This is the recommended initializer for
  neural network weights and filters.

  Arguments:
      mean: a python scalar or a scalar tensor. Mean of the random values
        to generate.
      stddev: a python scalar or a scalar tensor. Standard deviation of the
        random values to generate.
      seed: A Python integer. Used to seed the random generator.
  """

  def __init__(self, mean=0., stddev=0.05, seed=None):
    self.mean = mean
    self.stddev = stddev
    self.seed = seed

  def __call__(self, shape, dtype=None):
    return K.truncated_normal(
        shape, self.mean, self.stddev, dtype=dtype, seed=self.seed)

  def get_config(self):
    return {'mean': self.mean, 'stddev': self.stddev, 'seed': self.seed}


class VarianceScaling(Initializer):
  """Initializer capable of adapting its scale to the shape of weights.

  With `distribution="normal"`, samples are drawn from a truncated normal
  distribution centered on zero, with `stddev = sqrt(scale / n)` where n is:

      - number of input units in the weight tensor, if mode = "fan_in"
      - number of output units, if mode = "fan_out"
      - average of the numbers of input and output units, if mode = "fan_avg"

  With `distribution="uniform"`,
  samples are drawn from a uniform distribution
  within [-limit, limit], with `limit = sqrt(3 * scale / n)`.

  Arguments:
      scale: Scaling factor (positive float).
      mode: One of "fan_in", "fan_out", "fan_avg".
      distribution: Random distribution to use. One of "normal", "uniform".
      seed: A Python integer. Used to seed the random generator.

  Raises:
      ValueError: In case of an invalid value for the "scale", mode" or
        "distribution" arguments.
  """

  def __init__(self, scale=1.0, mode='fan_in', distribution='normal',
               seed=None):
    if scale <= 0.:
      raise ValueError('`scale` must be a positive float. Got:', scale)
    mode = mode.lower()
    if mode not in {'fan_in', 'fan_out', 'fan_avg'}:
      raise ValueError('Invalid `mode` argument: '
                       'expected on of {"fan_in", "fan_out", "fan_avg"} '
                       'but got', mode)
    distribution = distribution.lower()
    if distribution not in {'normal', 'uniform'}:
      raise ValueError('Invalid `distribution` argument: '
                       'expected one of {"normal", "uniform"} '
                       'but got', distribution)
    self.scale = scale
    self.mode = mode
    self.distribution = distribution
    self.seed = seed

  def __call__(self, shape, dtype=None):
    fan_in, fan_out = _compute_fans(shape)
    scale = self.scale
    if self.mode == 'fan_in':
      scale /= max(1., fan_in)
    elif self.mode == 'fan_out':
      scale /= max(1., fan_out)
    else:
      scale /= max(1., float(fan_in + fan_out) / 2)
    if self.distribution == 'normal':
      stddev = math.sqrt(scale)
      return K.truncated_normal(shape, 0., stddev, dtype=dtype, seed=self.seed)
    else:
      limit = math.sqrt(3. * scale)
      return K.random_uniform(shape, -limit, limit, dtype=dtype, seed=self.seed)

  def get_config(self):
    return {
        'scale': self.scale,
        'mode': self.mode,
        'distribution': self.distribution,
        'seed': self.seed
    }


class Orthogonal(Initializer):
  """Initializer that generates a random orthogonal matrix.

  Arguments:
      gain: Multiplicative factor to apply to the orthogonal matrix.
      seed: A Python integer. Used to seed the random generator.

  References:
      Saxe et al., http://arxiv.org/abs/1312.6120
  """

  def __init__(self, gain=1., seed=None):
    self.gain = gain
    self.seed = seed

  def __call__(self, shape, dtype=None):
    num_rows = 1
    for dim in shape[:-1]:
      num_rows *= dim
    num_cols = shape[-1]
    flat_shape = (num_rows, num_cols)
    if self.seed is not None:
      np.random.seed(self.seed)
    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    # Pick the one with the correct shape.
    q = u if u.shape == flat_shape else v
    q = q.reshape(shape)
    return self.gain * q[:shape[0], :shape[1]]

  def get_config(self):
    return {'gain': self.gain, 'seed': self.seed}


class Identity(Initializer):
  """Initializer that generates the identity matrix.

  Only use for square 2D matrices.

  Arguments:
      gain: Multiplicative factor to apply to the identity matrix.
  """

  def __init__(self, gain=1.):
    self.gain = gain

  def __call__(self, shape, dtype=None):
    if len(shape) != 2 or shape[0] != shape[1]:
      raise ValueError('Identity matrix initializer can only be used '
                       'for 2D square matrices.')
    else:
      return self.gain * np.identity(shape[0])

  def get_config(self):
    return {'gain': self.gain}


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
      LeCun 98, Efficient Backprop,
      http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
  """
  return VarianceScaling(
      scale=1., mode='fan_in', distribution='uniform', seed=seed)


def glorot_normal(seed=None):
  """Glorot normal initializer, also called Xavier normal initializer.

  It draws samples from a truncated normal distribution centered on 0
  with `stddev = sqrt(2 / (fan_in + fan_out))`
  where `fan_in` is the number of input units in the weight tensor
  and `fan_out` is the number of output units in the weight tensor.

  Arguments:
      seed: A Python integer. Used to seed the random generator.

  Returns:
      An initializer.

  References:
      Glorot & Bengio, AISTATS 2010
      http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf
  """
  return VarianceScaling(
      scale=1., mode='fan_avg', distribution='normal', seed=seed)


def glorot_uniform(seed=None):
  """Glorot uniform initializer, also called Xavier uniform initializer.

  It draws samples from a uniform distribution within [-limit, limit]
  where `limit` is `sqrt(6 / (fan_in + fan_out))`
  where `fan_in` is the number of input units in the weight tensor
  and `fan_out` is the number of output units in the weight tensor.

  Arguments:
      seed: A Python integer. Used to seed the random generator.

  Returns:
      An initializer.

  References:
      Glorot & Bengio, AISTATS 2010
      http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf
  """
  return VarianceScaling(
      scale=1., mode='fan_avg', distribution='uniform', seed=seed)


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
      He et al., http://arxiv.org/abs/1502.01852
  """
  return VarianceScaling(
      scale=2., mode='fan_in', distribution='normal', seed=seed)


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
      He et al., http://arxiv.org/abs/1502.01852
  """
  return VarianceScaling(
      scale=2., mode='fan_in', distribution='uniform', seed=seed)


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

# pylint: enable=invalid-name

# Utility functions


def _compute_fans(shape, data_format='channels_last'):
  """Computes the number of input and output units for a weight shape.

  Arguments:
      shape: Integer shape tuple.
      data_format: Image data format to use for convolution kernels.
          Note that all kernels in Keras are standardized on the
          `channels_last` ordering (even when inputs are set
          to `channels_first`).

  Returns:
      A tuple of scalars, `(fan_in, fan_out)`.

  Raises:
      ValueError: in case of invalid `data_format` argument.
  """
  shape = tensor_shape.TensorShape(shape).as_list()
  if len(shape) == 2:
    fan_in = shape[0]
    fan_out = shape[1]
  elif len(shape) in {3, 4, 5}:
    # Assuming convolution kernels (1D, 2D or 3D).
    # TH kernel shape: (depth, input_depth, ...)
    # TF kernel shape: (..., input_depth, depth)
    if data_format == 'channels_first':
      receptive_field_size = np.prod(shape[2:])
      fan_in = shape[1] * receptive_field_size
      fan_out = shape[0] * receptive_field_size
    elif data_format == 'channels_last':
      receptive_field_size = np.prod(shape[:2])
      fan_in = shape[-2] * receptive_field_size
      fan_out = shape[-1] * receptive_field_size
    else:
      raise ValueError('Invalid data_format: ' + data_format)
  else:
    # No specific assumptions.
    fan_in = math.sqrt(np.prod(shape))
    fan_out = math.sqrt(np.prod(shape))
  return fan_in, fan_out


def serialize(initializer):
  return serialize_keras_object(initializer)


def deserialize(config, custom_objects=None):
  return deserialize_keras_object(
      config,
      module_objects=globals(),
      custom_objects=custom_objects,
      printable_module_name='initializer')


def get(identifier):
  if isinstance(identifier, dict):
    return deserialize(identifier)
  elif isinstance(identifier, six.string_types):
    config = {'class_name': str(identifier), 'config': {}}
    return deserialize(config)
  elif callable(identifier):
    return identifier
  else:
    raise ValueError('Could not interpret initializer identifier:', identifier)
