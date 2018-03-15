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

from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras._impl.keras import backend as K
from tensorflow.python.keras._impl.keras.engine import InputSpec
from tensorflow.python.keras._impl.keras.engine import Layer
from tensorflow.python.keras._impl.keras.utils import conv_utils
from tensorflow.python.layers import pooling as tf_pooling_layers
from tensorflow.python.util.tf_export import tf_export


@tf_export('keras.layers.MaxPool1D', 'keras.layers.MaxPooling1D')
class MaxPooling1D(tf_pooling_layers.MaxPooling1D, Layer):
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
    if strides is None:
      strides = pool_size
    super(MaxPooling1D, self).__init__(pool_size, strides, padding, **kwargs)

  def get_config(self):
    config = {
        'strides': self.strides,
        'pool_size': self.pool_size,
        'padding': self.padding
    }
    base_config = super(MaxPooling1D, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


@tf_export('keras.layers.AveragePooling1D', 'keras.layers.AvgPool1D')
class AveragePooling1D(tf_pooling_layers.AveragePooling1D, Layer):
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
    if strides is None:
      strides = pool_size
    super(AveragePooling1D, self).__init__(pool_size, strides, padding,
                                           **kwargs)

  def get_config(self):
    config = {
        'strides': self.strides,
        'pool_size': self.pool_size,
        'padding': self.padding
    }
    base_config = super(AveragePooling1D, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


@tf_export('keras.layers.MaxPool2D', 'keras.layers.MaxPooling2D')
class MaxPooling2D(tf_pooling_layers.MaxPooling2D, Layer):
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
          `(batch, height, width, channels)` while `channels_first`
          corresponds to inputs with shape
          `(batch, channels, height, width)`.
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
    if data_format is None:
      data_format = K.image_data_format()
    if strides is None:
      strides = pool_size
    super(MaxPooling2D, self).__init__(pool_size, strides, padding, data_format,
                                       **kwargs)

  def get_config(self):
    config = {
        'pool_size': self.pool_size,
        'padding': self.padding,
        'strides': self.strides,
        'data_format': self.data_format
    }
    base_config = super(MaxPooling2D, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


@tf_export('keras.layers.AveragePooling2D', 'keras.layers.AvgPool2D')
class AveragePooling2D(tf_pooling_layers.AveragePooling2D, Layer):
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
          `(batch, height, width, channels)` while `channels_first`
          corresponds to inputs with shape
          `(batch, channels, height, width)`.
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
    if data_format is None:
      data_format = K.image_data_format()
    if strides is None:
      strides = pool_size
    super(AveragePooling2D, self).__init__(pool_size, strides, padding,
                                           data_format, **kwargs)

  def get_config(self):
    config = {
        'pool_size': self.pool_size,
        'padding': self.padding,
        'strides': self.strides,
        'data_format': self.data_format
    }
    base_config = super(AveragePooling2D, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


@tf_export('keras.layers.MaxPool3D', 'keras.layers.MaxPooling3D')
class MaxPooling3D(tf_pooling_layers.MaxPooling3D, Layer):
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
    if data_format is None:
      data_format = K.image_data_format()
    if strides is None:
      strides = pool_size
    super(MaxPooling3D, self).__init__(pool_size, strides, padding, data_format,
                                       **kwargs)

  def get_config(self):
    config = {
        'pool_size': self.pool_size,
        'padding': self.padding,
        'strides': self.strides,
        'data_format': self.data_format
    }
    base_config = super(MaxPooling3D, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


@tf_export('keras.layers.AveragePooling3D', 'keras.layers.AvgPool3D')
class AveragePooling3D(tf_pooling_layers.AveragePooling3D, Layer):
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
    if data_format is None:
      data_format = K.image_data_format()
    if strides is None:
      strides = pool_size
    super(AveragePooling3D, self).__init__(pool_size, strides, padding,
                                           data_format, **kwargs)

  def get_config(self):
    config = {
        'pool_size': self.pool_size,
        'padding': self.padding,
        'strides': self.strides,
        'data_format': self.data_format
    }
    base_config = super(AveragePooling3D, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class _GlobalPooling1D(Layer):
  """Abstract class for different global pooling 1D layers.
  """

  def __init__(self, **kwargs):
    super(_GlobalPooling1D, self).__init__(**kwargs)
    self.input_spec = InputSpec(ndim=3)

  def compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    return tensor_shape.TensorShape([input_shape[0], input_shape[2]])

  def call(self, inputs):
    raise NotImplementedError


@tf_export('keras.layers.GlobalAveragePooling1D',
           'keras.layers.GlobalAvgPool1D')
class GlobalAveragePooling1D(_GlobalPooling1D):
  """Global average pooling operation for temporal data.

  Input shape:
      3D tensor with shape: `(batch_size, steps, features)`.

  Output shape:
      2D tensor with shape:
      `(batch_size, features)`
  """

  def call(self, inputs):
    return K.mean(inputs, axis=1)


@tf_export('keras.layers.GlobalMaxPool1D', 'keras.layers.GlobalMaxPooling1D')
class GlobalMaxPooling1D(_GlobalPooling1D):
  """Global max pooling operation for temporal data.

  Input shape:
      3D tensor with shape: `(batch_size, steps, features)`.

  Output shape:
      2D tensor with shape:
      `(batch_size, features)`
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

  def compute_output_shape(self, input_shape):
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


@tf_export('keras.layers.GlobalAveragePooling2D',
           'keras.layers.GlobalAvgPool2D')
class GlobalAveragePooling2D(_GlobalPooling2D):
  """Global average pooling operation for spatial data.

  Arguments:
      data_format: A string,
          one of `channels_last` (default) or `channels_first`.
          The ordering of the dimensions in the inputs.
          `channels_last` corresponds to inputs with shape
          `(batch, height, width, channels)` while `channels_first`
          corresponds to inputs with shape
          `(batch, channels, height, width)`.
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


@tf_export('keras.layers.GlobalMaxPool2D', 'keras.layers.GlobalMaxPooling2D')
class GlobalMaxPooling2D(_GlobalPooling2D):
  """Global max pooling operation for spatial data.

  Arguments:
      data_format: A string,
          one of `channels_last` (default) or `channels_first`.
          The ordering of the dimensions in the inputs.
          `channels_last` corresponds to inputs with shape
          `(batch, height, width, channels)` while `channels_first`
          corresponds to inputs with shape
          `(batch, channels, height, width)`.
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

  def compute_output_shape(self, input_shape):
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


@tf_export('keras.layers.GlobalAveragePooling3D',
           'keras.layers.GlobalAvgPool3D')
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


@tf_export('keras.layers.GlobalMaxPool3D', 'keras.layers.GlobalMaxPooling3D')
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
