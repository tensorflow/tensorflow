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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.util.tf_export import keras_export

# The type of float to use throughout a session.
_FLOATX = 'float32'

# Epsilon fuzz factor used throughout the codebase.
_EPSILON = 1e-7

# Default image data format, one of "channels_last", "channels_first".
_IMAGE_DATA_FORMAT = 'channels_last'


@keras_export('keras.backend.epsilon')
def epsilon():
  """Returns the value of the fuzz factor used in numeric expressions.

  Returns:
      A float.

  Example:
  ```python
  keras.backend.epsilon() >>>1e-07
  ```
  """
  return _EPSILON


@keras_export('keras.backend.set_epsilon')
def set_epsilon(value):
  """Sets the value of the fuzz factor used in numeric expressions.

  Arguments:
      value: float. New value of epsilon.
  Example: ```python from keras import backend as K K.epsilon() >>> 1e-07
    K.set_epsilon(1e-05) K.epsilon() >>> 1e-05 ```
  """
  global _EPSILON
  _EPSILON = value


@keras_export('keras.backend.floatx')
def floatx():
  """Returns the default float type, as a string.

  E.g. 'float16', 'float32', 'float64'.

  Returns:
      String, the current default float type.

  Example:
  ```python
  keras.backend.floatx() >>> 'float32'
  ```
  """
  return _FLOATX


@keras_export('keras.backend.set_floatx')
def set_floatx(value):
  """Sets the default float type.

  Arguments:
      value: String; 'float16', 'float32', or 'float64'.
  Example: ```python from keras import backend as K K.floatx() >>> 'float32'
    K.set_floatx('float16') K.floatx() >>> 'float16' ```

  Raises:
      ValueError: In case of invalid value.
  """
  global _FLOATX
  if value not in {'float16', 'float32', 'float64'}:
    raise ValueError('Unknown floatx type: ' + str(value))
  _FLOATX = str(value)


@keras_export('keras.backend.image_data_format')
def image_data_format():
  """Returns the default image data format convention.

  Returns:
      A string, either `'channels_first'` or `'channels_last'`

  Example:
  ```python
  keras.backend.image_data_format() >>> 'channels_first'
  ```
  """
  return _IMAGE_DATA_FORMAT


@keras_export('keras.backend.set_image_data_format')
def set_image_data_format(data_format):
  """Sets the value of the image data format convention.

  Arguments:
      data_format: string. `'channels_first'` or `'channels_last'`.
  Example: ```python from keras import backend as K K.image_data_format() >>>
    'channels_first' K.set_image_data_format('channels_last')
    K.image_data_format() >>> 'channels_last' ```

  Raises:
      ValueError: In case of invalid `data_format` value.
  """
  global _IMAGE_DATA_FORMAT
  if data_format not in {'channels_last', 'channels_first'}:
    raise ValueError('Unknown data_format: ' + str(data_format))
  _IMAGE_DATA_FORMAT = str(data_format)
