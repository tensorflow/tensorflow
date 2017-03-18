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
"""Pooling layers.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.keras.python.keras import backend as K
from tensorflow.contrib.keras.python.keras.engine import InputSpec
from tensorflow.contrib.keras.python.keras.engine import Layer
from tensorflow.contrib.keras.python.keras.utils import conv_utils
from tensorflow.python.framework import tensor_shape


class _Pooling1D(Layer):
  """Abstract class for different pooling 1D layers.
  """

  def __init__(self, pool_size=2, strides=None, padding='valid', **kwargs):
    super(_Pooling1D, self).__init__(**kwargs)
    if strides is None:
      strides = pool_size
    self.pool_size = conv_utils.normalize_tuple(pool_size, 1, 'pool_size')
    self.strides = conv_utils.normalize_tuple(strides, 1, 'strides')
    self.padding = conv_utils.normalize_padding(padding)
    self.input_spec = InputSpec(ndim=3)

  def _compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    length = conv_utils.conv_output_length(input_shape[1], self.pool_size[0],
                                           self.padding, self.strides[0])
    return tensor_shape.TensorShape([input_shape[0], length, input_shape[2]])

  def _pooling_function(self, inputs, pool_size, strides, padding, data_format):
    raise NotImplementedError

  def call(self, inputs):
    inputs = K.expand_dims(inputs, 2)  # add dummy last dimension
    output = self._pooling_function(
        inputs=inputs,
        pool_size=self.pool_size + (1,),
        strides=self.strides + (1,),
        padding=self.padding,
        data_format='channels_last')
    return K.squeeze(output, 2)  # remove dummy last dimension

  def get_config(self):
    config = {
        'strides': self.strides,
        'pool_size': self.pool_size,
        'padding': self.padding
    }
    base_config = super(_Pooling1D, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class MaxPooling1D(_Pooling1D):
  """Max pooling operation for temporal data.

  Arguments:
      pool_size: Integer, size of the max pooling windows.
      strides: Integer, or None. Factor by which to downscale.
          E.g. 2 will halve the input.
          If None, it will default to `pool_size`.
      padding: One of `"valid"` or `"same"` (case-insensitive).

  Input shape:
      3D tensor with shape: `(batch_size, steps, features)`.

  Output shape:
      3D tensor with shape: `(batch_size, downsampled_steps, features)`.
  """

  def __init__(self, pool_size=2, strides=None, padding='valid', **kwargs):
    super(MaxPooling1D, self).__init__(pool_size, strides, padding, **kwargs)

  def _pooling_function(self, inputs, pool_size, strides, padding, data_format):
    output = K.pool2d(
        inputs, pool_size, strides, padding, data_format, pool_mode='max')
    return output


class AveragePooling1D(_Pooling1D):
  """Average pooling for temporal data.

  Arguments:
      pool_size: Integer, size of the max pooling windows.
      strides: Integer, or None. Factor by which to downscale.
          E.g. 2 will halve the input.
          If None, it will default to `pool_size`.
      padding: One of `"valid"` or `"same"` (case-insensitive).

  Input shape:
      3D tensor with shape: `(batch_size, steps, features)`.

  Output shape:
      3D tensor with shape: `(batch_size, downsampled_steps, features)`.
  """

  def __init__(self, pool_size=2, strides=None, padding='valid', **kwargs):
    super(AveragePooling1D, self).__init__(pool_size, strides, padding,
                                           **kwargs)

  def _pooling_function(self, inputs, pool_size, strides, padding, data_format):
    output = K.pool2d(
        inputs, pool_size, strides, padding, data_format, pool_mode='avg')
    return output


class _Pooling2D(Layer):
  """Abstract class for different pooling 2D layers.
  """

  def __init__(self,
               pool_size=(2, 2),
               strides=None,
               padding='valid',
               data_format=None,
               **kwargs):
    super(_Pooling2D, self).__init__(**kwargs)
    data_format = conv_utils.normalize_data_format(data_format)
    if strides is None:
      strides = pool_size
    self.pool_size = conv_utils.normalize_tuple(pool_size, 2, 'pool_size')
    self.strides = conv_utils.normalize_tuple(strides, 2, 'strides')
    self.padding = conv_utils.normalize_padding(padding)
    self.data_format = conv_utils.normalize_data_format(data_format)
    self.input_spec = InputSpec(ndim=4)

  def _compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    if self.data_format == 'channels_first':
      rows = input_shape[2]
      cols = input_shape[3]
    else:
      rows = input_shape[1]
      cols = input_shape[2]
    rows = conv_utils.conv_output_length(rows, self.pool_size[0], self.padding,
                                         self.strides[0])
    cols = conv_utils.conv_output_length(cols, self.pool_size[1], self.padding,
                                         self.strides[1])
    if self.data_format == 'channels_first':
      return tensor_shape.TensorShape(
          [input_shape[0], input_shape[1], rows, cols])
    else:
      return tensor_shape.TensorShape(
          [input_shape[0], rows, cols, input_shape[3]])

  def _pooling_function(self, inputs, pool_size, strides, padding, data_format):
    raise NotImplementedError

  def call(self, inputs):
    output = self._pooling_function(
        inputs=inputs,
        pool_size=self.pool_size,
        strides=self.strides,
        padding=self.padding,
        data_format=self.data_format)
    return output

  def get_config(self):
    config = {
        'pool_size': self.pool_size,
        'padding': self.padding,
        'strides': self.strides,
        'data_format': self.data_format
    }
    base_config = super(_Pooling2D, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class MaxPooling2D(_Pooling2D):
  """Max pooling operation for spatial data.

  Arguments:
      pool_size: integer or tuple of 2 integers,
          factors by which to downscale (vertical, horizontal).
          (2, 2) will halve the input in both spatial dimension.
          If only one integer is specified, the same window length
          will be used for both dimensions.
      strides: Integer, tuple of 2 integers, or None.
          Strides values.
          If None, it will default to `pool_size`.
      padding: One of `"valid"` or `"same"` (case-insensitive).
      data_format: A string,
          one of `channels_last` (default) or `channels_first`.
          The ordering of the dimensions in the inputs.
          `channels_last` corresponds to inputs with shape
          `(batch, width, height, channels)` while `channels_first`
          corresponds to inputs with shape
          `(batch, channels, width, height)`.
          It defaults to the `image_data_format` value found in your
          Keras config file at `~/.keras/keras.json`.
          If you never set it, then it will be "channels_last".

  Input shape:
      - If `data_format='channels_last'`:
          4D tensor with shape:
          `(batch_size, rows, cols, channels)`
      - If `data_format='channels_first'`:
          4D tensor with shape:
          `(batch_size, channels, rows, cols)`

  Output shape:
      - If `data_format='channels_last'`:
          4D tensor with shape:
          `(batch_size, pooled_rows, pooled_cols, channels)`
      - If `data_format='channels_first'`:
          4D tensor with shape:
          `(batch_size, channels, pooled_rows, pooled_cols)`
  """

  def __init__(self,
               pool_size=(2, 2),
               strides=None,
               padding='valid',
               data_format=None,
               **kwargs):
    super(MaxPooling2D, self).__init__(pool_size, strides, padding, data_format,
                                       **kwargs)

  def _pooling_function(self, inputs, pool_size, strides, padding, data_format):
    output = K.pool2d(
        inputs, pool_size, strides, padding, data_format, pool_mode='max')
    return output


class AveragePooling2D(_Pooling2D):
  """Average pooling operation for spatial data.

  Arguments:
      pool_size: integer or tuple of 2 integers,
          factors by which to downscale (vertical, horizontal).
          (2, 2) will halve the input in both spatial dimension.
          If only one integer is specified, the same window length
          will be used for both dimensions.
      strides: Integer, tuple of 2 integers, or None.
          Strides values.
          If None, it will default to `pool_size`.
      padding: One of `"valid"` or `"same"` (case-insensitive).
      data_format: A string,
          one of `channels_last` (default) or `channels_first`.
          The ordering of the dimensions in the inputs.
          `channels_last` corresponds to inputs with shape
          `(batch, width, height, channels)` while `channels_first`
          corresponds to inputs with shape
          `(batch, channels, width, height)`.
          It defaults to the `image_data_format` value found in your
          Keras config file at `~/.keras/keras.json`.
          If you never set it, then it will be "channels_last".

  Input shape:
      - If `data_format='channels_last'`:
          4D tensor with shape:
          `(batch_size, rows, cols, channels)`
      - If `data_format='channels_first'`:
          4D tensor with shape:
          `(batch_size, channels, rows, cols)`

  Output shape:
      - If `data_format='channels_last'`:
          4D tensor with shape:
          `(batch_size, pooled_rows, pooled_cols, channels)`
      - If `data_format='channels_first'`:
          4D tensor with shape:
          `(batch_size, channels, pooled_rows, pooled_cols)`
  """

  def __init__(self,
               pool_size=(2, 2),
               strides=None,
               padding='valid',
               data_format=None,
               **kwargs):
    super(AveragePooling2D, self).__init__(pool_size, strides, padding,
                                           data_format, **kwargs)

  def _pooling_function(self, inputs, pool_size, strides, padding, data_format):
    output = K.pool2d(
        inputs, pool_size, strides, padding, data_format, pool_mode='avg')
    return output


class _Pooling3D(Layer):
  """Abstract class for different pooling 3D layers.
  """

  def __init__(self,
               pool_size=(2, 2, 2),
               strides=None,
               padding='valid',
               data_format=None,
               **kwargs):
    super(_Pooling3D, self).__init__(**kwargs)
    if strides is None:
      strides = pool_size
    self.pool_size = conv_utils.normalize_tuple(pool_size, 3, 'pool_size')
    self.strides = conv_utils.normalize_tuple(strides, 3, 'strides')
    self.padding = conv_utils.normalize_padding(padding)
    self.data_format = conv_utils.normalize_data_format(data_format)
    self.input_spec = InputSpec(ndim=5)

  def _compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    if self.data_format == 'channels_first':
      len_dim1 = input_shape[2]
      len_dim2 = input_shape[3]
      len_dim3 = input_shape[4]
    else:
      len_dim1 = input_shape[1]
      len_dim2 = input_shape[2]
      len_dim3 = input_shape[3]
    len_dim1 = conv_utils.conv_output_length(len_dim1, self.pool_size[0],
                                             self.padding, self.strides[0])
    len_dim2 = conv_utils.conv_output_length(len_dim2, self.pool_size[1],
                                             self.padding, self.strides[1])
    len_dim3 = conv_utils.conv_output_length(len_dim3, self.pool_size[2],
                                             self.padding, self.strides[2])
    if self.data_format == 'channels_first':
      return tensor_shape.TensorShape(
          [input_shape[0], input_shape[1], len_dim1, len_dim2, len_dim3])
    else:
      return tensor_shape.TensorShape(
          [input_shape[0], len_dim1, len_dim2, len_dim3, input_shape[4]])

  def _pooling_function(self, inputs, pool_size, strides, padding, data_format):
    raise NotImplementedError

  def call(self, inputs):
    output = self._pooling_function(
        inputs=inputs,
        pool_size=self.pool_size,
        strides=self.strides,
        padding=self.padding,
        data_format=self.data_format)
    return output

  def get_config(self):
    config = {
        'pool_size': self.pool_size,
        'padding': self.padding,
        'strides': self.strides,
        'data_format': self.data_format
    }
    base_config = super(_Pooling3D, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class MaxPooling3D(_Pooling3D):
  """Max pooling operation for 3D data (spatial or spatio-temporal).

  Arguments:
      pool_size: tuple of 3 integers,
          factors by which to downscale (dim1, dim2, dim3).
          (2, 2, 2) will halve the size of the 3D input in each dimension.
      strides: tuple of 3 integers, or None. Strides values.
      padding: One of `"valid"` or `"same"` (case-insensitive).
      data_format: A string,
          one of `channels_last` (default) or `channels_first`.
          The ordering of the dimensions in the inputs.
          `channels_last` corresponds to inputs with shape
          `(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
          while `channels_first` corresponds to inputs with shape
          `(batch, channels, spatial_dim1, spatial_dim2, spatial_dim3)`.
          It defaults to the `image_data_format` value found in your
          Keras config file at `~/.keras/keras.json`.
          If you never set it, then it will be "channels_last".

  Input shape:
      - If `data_format='channels_last'`:
          5D tensor with shape:
          `(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
      - If `data_format='channels_first'`:
          5D tensor with shape:
          `(batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)`

  Output shape:
      - If `data_format='channels_last'`:
          5D tensor with shape:
          `(batch_size, pooled_dim1, pooled_dim2, pooled_dim3, channels)`
      - If `data_format='channels_first'`:
          5D tensor with shape:
          `(batch_size, channels, pooled_dim1, pooled_dim2, pooled_dim3)`
  """

  def __init__(self,
               pool_size=(2, 2, 2),
               strides=None,
               padding='valid',
               data_format=None,
               **kwargs):
    super(MaxPooling3D, self).__init__(pool_size, strides, padding, data_format,
                                       **kwargs)

  def _pooling_function(self, inputs, pool_size, strides, padding, data_format):
    output = K.pool3d(
        inputs, pool_size, strides, padding, data_format, pool_mode='max')
    return output


class AveragePooling3D(_Pooling3D):
  """Average pooling operation for 3D data (spatial or spatio-temporal).

  Arguments:
      pool_size: tuple of 3 integers,
          factors by which to downscale (dim1, dim2, dim3).
          (2, 2, 2) will halve the size of the 3D input in each dimension.
      strides: tuple of 3 integers, or None. Strides values.
      padding: One of `"valid"` or `"same"` (case-insensitive).
      data_format: A string,
          one of `channels_last` (default) or `channels_first`.
          The ordering of the dimensions in the inputs.
          `channels_last` corresponds to inputs with shape
          `(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
          while `channels_first` corresponds to inputs with shape
          `(batch, channels, spatial_dim1, spatial_dim2, spatial_dim3)`.
          It defaults to the `image_data_format` value found in your
          Keras config file at `~/.keras/keras.json`.
          If you never set it, then it will be "channels_last".

  Input shape:
      - If `data_format='channels_last'`:
          5D tensor with shape:
          `(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
      - If `data_format='channels_first'`:
          5D tensor with shape:
          `(batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)`

  Output shape:
      - If `data_format='channels_last'`:
          5D tensor with shape:
          `(batch_size, pooled_dim1, pooled_dim2, pooled_dim3, channels)`
      - If `data_format='channels_first'`:
          5D tensor with shape:
          `(batch_size, channels, pooled_dim1, pooled_dim2, pooled_dim3)`
  """

  def __init__(self,
               pool_size=(2, 2, 2),
               strides=None,
               padding='valid',
               data_format=None,
               **kwargs):
    super(AveragePooling3D, self).__init__(pool_size, strides, padding,
                                           data_format, **kwargs)

  def _pooling_function(self, inputs, pool_size, strides, padding, data_format):
    output = K.pool3d(
        inputs, pool_size, strides, padding, data_format, pool_mode='avg')
    return output


class _GlobalPooling1D(Layer):
  """Abstract class for different global pooling 1D layers.
  """

  def __init__(self, **kwargs):
    super(_GlobalPooling1D, self).__init__(**kwargs)
    self.input_spec = InputSpec(ndim=3)

  def _compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    return tensor_shape.TensorShape([input_shape[0], input_shape[2]])

  def call(self, inputs):
    raise NotImplementedError


class GlobalAveragePooling1D(_GlobalPooling1D):
  """Global average pooling operation for temporal data.

  Input shape:
      3D tensor with shape: `(batch_size, steps, features)`.

  Output shape:
      2D tensor with shape:
      `(batch_size, channels)`
  """

  def call(self, inputs):
    return K.mean(inputs, axis=1)


class GlobalMaxPooling1D(_GlobalPooling1D):
  """Global max pooling operation for temporal data.

  Input shape:
      3D tensor with shape: `(batch_size, steps, features)`.

  Output shape:
      2D tensor with shape:
      `(batch_size, channels)`
  """

  def call(self, inputs):
    return K.max(inputs, axis=1)


class _GlobalPooling2D(Layer):
  """Abstract class for different global pooling 2D layers.
  """

  def __init__(self, data_format=None, **kwargs):
    super(_GlobalPooling2D, self).__init__(**kwargs)
    self.data_format = conv_utils.normalize_data_format(data_format)
    self.input_spec = InputSpec(ndim=4)

  def _compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    if self.data_format == 'channels_last':
      return tensor_shape.TensorShape([input_shape[0], input_shape[3]])
    else:
      return tensor_shape.TensorShape([input_shape[0], input_shape[1]])

  def call(self, inputs):
    raise NotImplementedError

  def get_config(self):
    config = {'data_format': self.data_format}
    base_config = super(_GlobalPooling2D, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class GlobalAveragePooling2D(_GlobalPooling2D):
  """Global average pooling operation for spatial data.

  Arguments:
      data_format: A string,
          one of `channels_last` (default) or `channels_first`.
          The ordering of the dimensions in the inputs.
          `channels_last` corresponds to inputs with shape
          `(batch, width, height, channels)` while `channels_first`
          corresponds to inputs with shape
          `(batch, channels, width, height)`.
          It defaults to the `image_data_format` value found in your
          Keras config file at `~/.keras/keras.json`.
          If you never set it, then it will be "channels_last".

  Input shape:
      - If `data_format='channels_last'`:
          4D tensor with shape:
          `(batch_size, rows, cols, channels)`
      - If `data_format='channels_first'`:
          4D tensor with shape:
          `(batch_size, channels, rows, cols)`

  Output shape:
      2D tensor with shape:
      `(batch_size, channels)`
  """

  def call(self, inputs):
    if self.data_format == 'channels_last':
      return K.mean(inputs, axis=[1, 2])
    else:
      return K.mean(inputs, axis=[2, 3])


class GlobalMaxPooling2D(_GlobalPooling2D):
  """Global max pooling operation for spatial data.

  Arguments:
      data_format: A string,
          one of `channels_last` (default) or `channels_first`.
          The ordering of the dimensions in the inputs.
          `channels_last` corresponds to inputs with shape
          `(batch, width, height, channels)` while `channels_first`
          corresponds to inputs with shape
          `(batch, channels, width, height)`.
          It defaults to the `image_data_format` value found in your
          Keras config file at `~/.keras/keras.json`.
          If you never set it, then it will be "channels_last".

  Input shape:
      - If `data_format='channels_last'`:
          4D tensor with shape:
          `(batch_size, rows, cols, channels)`
      - If `data_format='channels_first'`:
          4D tensor with shape:
          `(batch_size, channels, rows, cols)`

  Output shape:
      2D tensor with shape:
      `(batch_size, channels)`
  """

  def call(self, inputs):
    if self.data_format == 'channels_last':
      return K.max(inputs, axis=[1, 2])
    else:
      return K.max(inputs, axis=[2, 3])


class _GlobalPooling3D(Layer):
  """Abstract class for different global pooling 3D layers.
  """

  def __init__(self, data_format=None, **kwargs):
    super(_GlobalPooling3D, self).__init__(**kwargs)
    self.data_format = conv_utils.normalize_data_format(data_format)
    self.input_spec = InputSpec(ndim=5)

  def _compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    if self.data_format == 'channels_last':
      return tensor_shape.TensorShape([input_shape[0], input_shape[4]])
    else:
      return tensor_shape.TensorShape([input_shape[0], input_shape[1]])

  def call(self, inputs):
    raise NotImplementedError

  def get_config(self):
    config = {'data_format': self.data_format}
    base_config = super(_GlobalPooling3D, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class GlobalAveragePooling3D(_GlobalPooling3D):
  """Global Average pooling operation for 3D data.

  Arguments:
      data_format: A string,
          one of `channels_last` (default) or `channels_first`.
          The ordering of the dimensions in the inputs.
          `channels_last` corresponds to inputs with shape
          `(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
          while `channels_first` corresponds to inputs with shape
          `(batch, channels, spatial_dim1, spatial_dim2, spatial_dim3)`.
          It defaults to the `image_data_format` value found in your
          Keras config file at `~/.keras/keras.json`.
          If you never set it, then it will be "channels_last".

  Input shape:
      - If `data_format='channels_last'`:
          5D tensor with shape:
          `(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
      - If `data_format='channels_first'`:
          5D tensor with shape:
          `(batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)`

  Output shape:
      2D tensor with shape:
      `(batch_size, channels)`
  """

  def call(self, inputs):
    if self.data_format == 'channels_last':
      return K.mean(inputs, axis=[1, 2, 3])
    else:
      return K.mean(inputs, axis=[2, 3, 4])


class GlobalMaxPooling3D(_GlobalPooling3D):
  """Global Max pooling operation for 3D data.

  Arguments:
      data_format: A string,
          one of `channels_last` (default) or `channels_first`.
          The ordering of the dimensions in the inputs.
          `channels_last` corresponds to inputs with shape
          `(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
          while `channels_first` corresponds to inputs with shape
          `(batch, channels, spatial_dim1, spatial_dim2, spatial_dim3)`.
          It defaults to the `image_data_format` value found in your
          Keras config file at `~/.keras/keras.json`.
          If you never set it, then it will be "channels_last".

  Input shape:
      - If `data_format='channels_last'`:
          5D tensor with shape:
          `(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
      - If `data_format='channels_first'`:
          5D tensor with shape:
          `(batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)`

  Output shape:
      2D tensor with shape:
      `(batch_size, channels)`
  """

  def call(self, inputs):
    if self.data_format == 'channels_last':
      return K.max(inputs, axis=[1, 2, 3])
    else:
      return K.max(inputs, axis=[2, 3, 4])


# Aliases

AvgPool1D = AveragePooling1D
MaxPool1D = MaxPooling1D
AvgPool2D = AveragePooling2D
MaxPool2D = MaxPooling2D
AvgPool3D = AveragePooling3D
MaxPool3D = MaxPooling3D
GlobalMaxPool1D = GlobalMaxPooling1D
GlobalMaxPool2D = GlobalMaxPooling2D
GlobalMaxPool3D = GlobalMaxPooling3D
GlobalAvgPool1D = GlobalAveragePooling1D
GlobalAvgPool2D = GlobalAveragePooling2D
GlobalAvgPool3D = GlobalAveragePooling3D
