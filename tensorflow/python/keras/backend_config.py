# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Keras backend config API."""

from tensorflow.python.util import dispatch

# The type of float to use throughout a session.
_FLOATX = 'float32'

# Epsilon fuzz factor used throughout the codebase.
_EPSILON = 1e-7

# Default image data format, one of "channels_last", "channels_first".
_IMAGE_DATA_FORMAT = 'channels_last'


@dispatch.add_dispatch_support
def epsilon():
  """Returns the value of the fuzz factor used in numeric expressions.

  Returns:
      A float.

  Example:
  >>> tf.keras.backend.epsilon()
  1e-07
  """
  return _EPSILON


def set_epsilon(value):
  """Sets the value of the fuzz factor used in numeric expressions.

  Args:
      value: float. New value of epsilon.

  Example:
  >>> tf.keras.backend.epsilon()
  1e-07
  >>> tf.keras.backend.set_epsilon(1e-5)
  >>> tf.keras.backend.epsilon()
  1e-05
   >>> tf.keras.backend.set_epsilon(1e-7)
  """
  global _EPSILON
  _EPSILON = value


def floatx():
  """Returns the default float type, as a string.

  E.g. `'float16'`, `'float32'`, `'float64'`.

  Returns:
      String, the current default float type.

  Example:
  >>> tf.keras.backend.floatx()
  'float32'
  """
  return _FLOATX


def set_floatx(value):
  """Sets the default float type.

  Note: It is not recommended to set this to float16 for training, as this will
  likely cause numeric stability issues. Instead, mixed precision, which is
  using a mix of float16 and float32, can be used by calling
  `tf.keras.mixed_precision.set_global_policy('mixed_float16')`. See the
  [mixed precision guide](
    https://www.tensorflow.org/guide/keras/mixed_precision) for details.

  Args:
      value: String; `'float16'`, `'float32'`, or `'float64'`.

  Example:
  >>> tf.keras.backend.floatx()
  'float32'
  >>> tf.keras.backend.set_floatx('float64')
  >>> tf.keras.backend.floatx()
  'float64'
  >>> tf.keras.backend.set_floatx('float32')

  Raises:
      ValueError: In case of invalid value.
  """
  global _FLOATX
  if value not in {'float16', 'float32', 'float64'}:
    raise ValueError('Unknown floatx type: ' + str(value))
  _FLOATX = str(value)


@dispatch.add_dispatch_support
def image_data_format():
  """Returns the default image data format convention.

  Returns:
      A string, either `'channels_first'` or `'channels_last'`

  Example:
  >>> tf.keras.backend.image_data_format()
  'channels_last'
  """
  return _IMAGE_DATA_FORMAT


def set_image_data_format(data_format):
  """Sets the value of the image data format convention.

  Args:
      data_format: string. `'channels_first'` or `'channels_last'`.

  Example:
  >>> tf.keras.backend.image_data_format()
  'channels_last'
  >>> tf.keras.backend.set_image_data_format('channels_first')
  >>> tf.keras.backend.image_data_format()
  'channels_first'
  >>> tf.keras.backend.set_image_data_format('channels_last')

  Raises:
      ValueError: In case of invalid `data_format` value.
  """
  global _IMAGE_DATA_FORMAT
  if data_format not in {'channels_last', 'channels_first'}:
    raise ValueError('Unknown data_format: ' + str(data_format))
  _IMAGE_DATA_FORMAT = str(data_format)
