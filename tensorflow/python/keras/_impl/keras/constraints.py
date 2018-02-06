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
# pylint: disable=invalid-name
"""Constraints: functions that impose constraints on weight values.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six

from tensorflow.python.keras._impl.keras import backend as K
from tensorflow.python.keras._impl.keras.utils.generic_utils import deserialize_keras_object
from tensorflow.python.keras._impl.keras.utils.generic_utils import serialize_keras_object


class Constraint(object):

  def __call__(self, w):
    return w

  def get_config(self):
    return {}


class MaxNorm(Constraint):
  """MaxNorm weight constraint.

  Constrains the weights incident to each hidden unit
  to have a norm less than or equal to a desired value.

  Arguments:
      m: the maximum norm for the incoming weights.
      axis: integer, axis along which to calculate weight norms.
          For instance, in a `Dense` layer the weight matrix
          has shape `(input_dim, output_dim)`,
          set `axis` to `0` to constrain each weight vector
          of length `(input_dim,)`.
          In a `Conv2D` layer with `data_format="channels_last"`,
          the weight tensor has shape
          `(rows, cols, input_depth, output_depth)`,
          set `axis` to `[0, 1, 2]`
          to constrain the weights of each filter tensor of size
          `(rows, cols, input_depth)`.

  """

  def __init__(self, max_value=2, axis=0):
    self.max_value = max_value
    self.axis = axis

  def __call__(self, w):
    norms = K.sqrt(K.sum(K.square(w), axis=self.axis, keepdims=True))
    desired = K.clip(norms, 0, self.max_value)
    w *= (desired / (K.epsilon() + norms))
    return w

  def get_config(self):
    return {'max_value': self.max_value, 'axis': self.axis}


class NonNeg(Constraint):
  """Constrains the weights to be non-negative.
  """

  def __call__(self, w):
    w *= K.cast(K.greater_equal(w, 0.), K.floatx())
    return w


class UnitNorm(Constraint):
  """Constrains the weights incident to each hidden unit to have unit norm.

  Arguments:
      axis: integer, axis along which to calculate weight norms.
          For instance, in a `Dense` layer the weight matrix
          has shape `(input_dim, output_dim)`,
          set `axis` to `0` to constrain each weight vector
          of length `(input_dim,)`.
          In a `Conv2D` layer with `data_format="channels_last"`,
          the weight tensor has shape
          `(rows, cols, input_depth, output_depth)`,
          set `axis` to `[0, 1, 2]`
          to constrain the weights of each filter tensor of size
          `(rows, cols, input_depth)`.
  """

  def __init__(self, axis=0):
    self.axis = axis

  def __call__(self, w):
    return w / (
        K.epsilon() + K.sqrt(K.sum(K.square(w), axis=self.axis, keepdims=True)))

  def get_config(self):
    return {'axis': self.axis}


class MinMaxNorm(Constraint):
  """MinMaxNorm weight constraint.

  Constrains the weights incident to each hidden unit
  to have the norm between a lower bound and an upper bound.

  Arguments:
      min_value: the minimum norm for the incoming weights.
      max_value: the maximum norm for the incoming weights.
      rate: rate for enforcing the constraint: weights will be
          rescaled to yield
          `(1 - rate) * norm + rate * norm.clip(min_value, max_value)`.
          Effectively, this means that rate=1.0 stands for strict
          enforcement of the constraint, while rate<1.0 means that
          weights will be rescaled at each step to slowly move
          towards a value inside the desired interval.
      axis: integer, axis along which to calculate weight norms.
          For instance, in a `Dense` layer the weight matrix
          has shape `(input_dim, output_dim)`,
          set `axis` to `0` to constrain each weight vector
          of length `(input_dim,)`.
          In a `Conv2D` layer with `data_format="channels_last"`,
          the weight tensor has shape
          `(rows, cols, input_depth, output_depth)`,
          set `axis` to `[0, 1, 2]`
          to constrain the weights of each filter tensor of size
          `(rows, cols, input_depth)`.
  """

  def __init__(self, min_value=0.0, max_value=1.0, rate=1.0, axis=0):
    self.min_value = min_value
    self.max_value = max_value
    self.rate = rate
    self.axis = axis

  def __call__(self, w):
    norms = K.sqrt(K.sum(K.square(w), axis=self.axis, keepdims=True))
    desired = (
        self.rate * K.clip(norms, self.min_value, self.max_value) +
        (1 - self.rate) * norms)
    w *= (desired / (K.epsilon() + norms))
    return w

  def get_config(self):
    return {
        'min_value': self.min_value,
        'max_value': self.max_value,
        'rate': self.rate,
        'axis': self.axis
    }


# Aliases.

max_norm = MaxNorm
non_neg = NonNeg
unit_norm = UnitNorm
min_max_norm = MinMaxNorm

# Legacy aliases.
maxnorm = max_norm
nonneg = non_neg
unitnorm = unit_norm


def serialize(constraint):
  return serialize_keras_object(constraint)


def deserialize(config, custom_objects=None):
  return deserialize_keras_object(
      config,
      module_objects=globals(),
      custom_objects=custom_objects,
      printable_module_name='constraint')


def get(identifier):
  if identifier is None:
    return None
  if isinstance(identifier, dict):
    return deserialize(identifier)
  elif isinstance(identifier, six.string_types):
    config = {'class_name': str(identifier), 'config': {}}
    return deserialize(config)
  elif callable(identifier):
    return identifier
  else:
    raise ValueError('Could not interpret constraint identifier:', identifier)
