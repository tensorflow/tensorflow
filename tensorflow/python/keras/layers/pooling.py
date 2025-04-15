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
"""Pooling layers."""

import functools

from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import backend
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn


class Pooling1D(Layer):
  """Pooling layer for arbitrary pooling functions, for 1D inputs.

  This class only exists for code reuse. It will never be an exposed API.

  Args:
    pool_function: The pooling function to apply, e.g. `tf.nn.max_pool2d`.
    pool_size: An integer or tuple/list of a single integer,
      representing the size of the pooling window.
    strides: An integer or tuple/list of a single integer, specifying the
      strides of the pooling operation.
    padding: A string. The padding method, either 'valid' or 'same'.
      Case-insensitive.
    data_format: A string,
      one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch, steps, features)` while `channels_first`
      corresponds to inputs with shape
      `(batch, features, steps)`.
    name: A string, the name of the layer.
  """

  def __init__(self, pool_function, pool_size, strides,
               padding='valid', data_format='channels_last',
               name=None, **kwargs):
    super(Pooling1D, self).__init__(name=name, **kwargs)
    if data_format is None:
      data_format = backend.image_data_format()
    if strides is None:
      strides = pool_size
    self.pool_function = pool_function
    self.pool_size = conv_utils.normalize_tuple(pool_size, 1, 'pool_size')
    self.strides = conv_utils.normalize_tuple(strides, 1, 'strides')
    self.padding = conv_utils.normalize_padding(padding)
    self.data_format = conv_utils.normalize_data_format(data_format)
    self.input_spec = InputSpec(ndim=3)

  def call(self, inputs):
    pad_axis = 2 if self.data_format == 'channels_last' else 3
    inputs = array_ops.expand_dims(inputs, pad_axis)
    outputs = self.pool_function(
        inputs,
        self.pool_size + (1,),
        strides=self.strides + (1,),
        padding=self.padding,
        data_format=self.data_format)
    return array_ops.squeeze(outputs, pad_axis)

  def compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    if self.data_format == 'channels_first':
      steps = input_shape[2]
      features = input_shape[1]
    else:
      steps = input_shape[1]
      features = input_shape[2]
    length = conv_utils.conv_output_length(steps,
                                           self.pool_size[0],
                                           self.padding,
                                           self.strides[0])
    if self.data_format == 'channels_first':
      return tensor_shape.TensorShape([input_shape[0], features, length])
    else:
      return tensor_shape.TensorShape([input_shape[0], length, features])

  def get_config(self):
    config = {
        'strides': self.strides,
        'pool_size': self.pool_size,
        'padding': self.padding,
        'data_format': self.data_format,
    }
    base_config = super(Pooling1D, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class MaxPooling1D(Pooling1D):
  """Max pooling operation for 1D temporal data.

  Downsamples the input representation by taking the maximum value over a
  spatial window of size `pool_size`. The window is shifted by `strides`.  The
  resulting output, when using the `"valid"` padding option, has a shape of:
  `output_shape = (input_shape - pool_size + 1) / strides)`

  The resulting output shape when using the `"same"` padding option is:
  `output_shape = input_shape / strides`

  For example, for `strides=1` and `padding="valid"`:

  >>> x = tf.constant([1., 2., 3., 4., 5.])
  >>> x = tf.reshape(x, [1, 5, 1])
  >>> max_pool_1d = tf.keras.layers.MaxPooling1D(pool_size=2,
  ...    strides=1, padding='valid')
  >>> max_pool_1d(x)
  <tf.Tensor: shape=(1, 4, 1), dtype=float32, numpy=
  array([[[2.],
          [3.],
          [4.],
          [5.]]], dtype=float32)>

  For example, for `strides=2` and `padding="valid"`:

  >>> x = tf.constant([1., 2., 3., 4., 5.])
  >>> x = tf.reshape(x, [1, 5, 1])
  >>> max_pool_1d = tf.keras.layers.MaxPooling1D(pool_size=2,
  ...    strides=2, padding='valid')
  >>> max_pool_1d(x)
  <tf.Tensor: shape=(1, 2, 1), dtype=float32, numpy=
  array([[[2.],
          [4.]]], dtype=float32)>

  For example, for `strides=1` and `padding="same"`:

  >>> x = tf.constant([1., 2., 3., 4., 5.])
  >>> x = tf.reshape(x, [1, 5, 1])
  >>> max_pool_1d = tf.keras.layers.MaxPooling1D(pool_size=2,
  ...    strides=1, padding='same')
  >>> max_pool_1d(x)
  <tf.Tensor: shape=(1, 5, 1), dtype=float32, numpy=
  array([[[2.],
          [3.],
          [4.],
          [5.],
          [5.]]], dtype=float32)>

  Args:
    pool_size: Integer, size of the max pooling window.
    strides: Integer, or None. Specifies how much the pooling window moves
      for each pooling step.
      If None, it will default to `pool_size`.
    padding: One of `"valid"` or `"same"` (case-insensitive).
      `"valid"` means no padding. `"same"` results in padding evenly to
      the left/right or up/down of the input such that output has the same
      height/width dimension as the input.
    data_format: A string,
      one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch, steps, features)` while `channels_first`
      corresponds to inputs with shape
      `(batch, features, steps)`.

  Input shape:
    - If `data_format='channels_last'`:
      3D tensor with shape `(batch_size, steps, features)`.
    - If `data_format='channels_first'`:
      3D tensor with shape `(batch_size, features, steps)`.

  Output shape:
    - If `data_format='channels_last'`:
      3D tensor with shape `(batch_size, downsampled_steps, features)`.
    - If `data_format='channels_first'`:
      3D tensor with shape `(batch_size, features, downsampled_steps)`.
  """

  def __init__(self, pool_size=2, strides=None,
               padding='valid', data_format='channels_last', **kwargs):

    super(MaxPooling1D, self).__init__(
        functools.partial(backend.pool2d, pool_mode='max'),
        pool_size=pool_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        **kwargs)


class AveragePooling1D(Pooling1D):
  """Average pooling for temporal data.

  Downsamples the input representation by taking the average value over the
  window defined by `pool_size`. The window is shifted by `strides`.  The
  resulting output when using "valid" padding option has a shape of:
  `output_shape = (input_shape - pool_size + 1) / strides)`

  The resulting output shape when using the "same" padding option is:
  `output_shape = input_shape / strides`

  For example, for strides=1 and padding="valid":

  >>> x = tf.constant([1., 2., 3., 4., 5.])
  >>> x = tf.reshape(x, [1, 5, 1])
  >>> x
  <tf.Tensor: shape=(1, 5, 1), dtype=float32, numpy=
    array([[[1.],
            [2.],
            [3.],
            [4.],
            [5.]], dtype=float32)>
  >>> avg_pool_1d = tf.keras.layers.AveragePooling1D(pool_size=2,
  ...    strides=1, padding='valid')
  >>> avg_pool_1d(x)
  <tf.Tensor: shape=(1, 4, 1), dtype=float32, numpy=
  array([[[1.5],
          [2.5],
          [3.5],
          [4.5]]], dtype=float32)>

  For example, for strides=2 and padding="valid":

  >>> x = tf.constant([1., 2., 3., 4., 5.])
  >>> x = tf.reshape(x, [1, 5, 1])
  >>> x
  <tf.Tensor: shape=(1, 5, 1), dtype=float32, numpy=
    array([[[1.],
            [2.],
            [3.],
            [4.],
            [5.]], dtype=float32)>
  >>> avg_pool_1d = tf.keras.layers.AveragePooling1D(pool_size=2,
  ...    strides=2, padding='valid')
  >>> avg_pool_1d(x)
  <tf.Tensor: shape=(1, 2, 1), dtype=float32, numpy=
  array([[[1.5],
          [3.5]]], dtype=float32)>

  For example, for strides=1 and padding="same":

  >>> x = tf.constant([1., 2., 3., 4., 5.])
  >>> x = tf.reshape(x, [1, 5, 1])
  >>> x
  <tf.Tensor: shape=(1, 5, 1), dtype=float32, numpy=
    array([[[1.],
            [2.],
            [3.],
            [4.],
            [5.]], dtype=float32)>
  >>> avg_pool_1d = tf.keras.layers.AveragePooling1D(pool_size=2,
  ...    strides=1, padding='same')
  >>> avg_pool_1d(x)
  <tf.Tensor: shape=(1, 5, 1), dtype=float32, numpy=
  array([[[1.5],
          [2.5],
          [3.5],
          [4.5],
          [5.]]], dtype=float32)>

  Args:
    pool_size: Integer, size of the average pooling windows.
    strides: Integer, or None. Factor by which to downscale.
      E.g. 2 will halve the input.
      If None, it will default to `pool_size`.
    padding: One of `"valid"` or `"same"` (case-insensitive).
      `"valid"` means no padding. `"same"` results in padding evenly to
      the left/right or up/down of the input such that output has the same
      height/width dimension as the input.
    data_format: A string,
      one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch, steps, features)` while `channels_first`
      corresponds to inputs with shape
      `(batch, features, steps)`.

  Input shape:
    - If `data_format='channels_last'`:
      3D tensor with shape `(batch_size, steps, features)`.
    - If `data_format='channels_first'`:
      3D tensor with shape `(batch_size, features, steps)`.

  Output shape:
    - If `data_format='channels_last'`:
      3D tensor with shape `(batch_size, downsampled_steps, features)`.
    - If `data_format='channels_first'`:
      3D tensor with shape `(batch_size, features, downsampled_steps)`.
  """

  def __init__(self, pool_size=2, strides=None,
               padding='valid', data_format='channels_last', **kwargs):
    super(AveragePooling1D, self).__init__(
        functools.partial(backend.pool2d, pool_mode='avg'),
        pool_size=pool_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        **kwargs)


class Pooling2D(Layer):
  """Pooling layer for arbitrary pooling functions, for 2D inputs (e.g. images).

  This class only exists for code reuse. It will never be an exposed API.

  Args:
    pool_function: The pooling function to apply, e.g. `tf.nn.max_pool2d`.
    pool_size: An integer or tuple/list of 2 integers: (pool_height, pool_width)
      specifying the size of the pooling window.
      Can be a single integer to specify the same value for
      all spatial dimensions.
    strides: An integer or tuple/list of 2 integers,
      specifying the strides of the pooling operation.
      Can be a single integer to specify the same value for
      all spatial dimensions.
    padding: A string. The padding method, either 'valid' or 'same'.
      Case-insensitive.
    data_format: A string, one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch, height, width, channels)` while `channels_first` corresponds to
      inputs with shape `(batch, channels, height, width)`.
    name: A string, the name of the layer.
  """

  def __init__(self, pool_function, pool_size, strides,
               padding='valid', data_format=None,
               name=None, **kwargs):
    super(Pooling2D, self).__init__(name=name, **kwargs)
    if data_format is None:
      data_format = backend.image_data_format()
    if strides is None:
      strides = pool_size
    self.pool_function = pool_function
    self.pool_size = conv_utils.normalize_tuple(pool_size, 2, 'pool_size')
    self.strides = conv_utils.normalize_tuple(strides, 2, 'strides')
    self.padding = conv_utils.normalize_padding(padding)
    self.data_format = conv_utils.normalize_data_format(data_format)
    self.input_spec = InputSpec(ndim=4)

  def call(self, inputs):
    if self.data_format == 'channels_last':
      pool_shape = (1,) + self.pool_size + (1,)
      strides = (1,) + self.strides + (1,)
    else:
      pool_shape = (1, 1) + self.pool_size
      strides = (1, 1) + self.strides
    outputs = self.pool_function(
        inputs,
        ksize=pool_shape,
        strides=strides,
        padding=self.padding.upper(),
        data_format=conv_utils.convert_data_format(self.data_format, 4))
    return outputs

  def compute_output_shape(self, input_shape):
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

  def get_config(self):
    config = {
        'pool_size': self.pool_size,
        'padding': self.padding,
        'strides': self.strides,
        'data_format': self.data_format
    }
    base_config = super(Pooling2D, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class MaxPooling2D(Pooling2D):
  """Max pooling operation for 2D spatial data.

  Downsamples the input along its spatial dimensions (height and width)
  by taking the maximum value over an input window
  (of size defined by `pool_size`) for each channel of the input.
  The window is shifted by `strides` along each dimension.

  The resulting output,
  when using the `"valid"` padding option, has a spatial shape
  (number of rows or columns) of:
  `output_shape = math.floor((input_shape - pool_size) / strides) + 1`
  (when `input_shape >= pool_size`)

  The resulting output shape when using the `"same"` padding option is:
  `output_shape = math.floor((input_shape - 1) / strides) + 1`

  For example, for `strides=(1, 1)` and `padding="valid"`:

  >>> x = tf.constant([[1., 2., 3.],
  ...                  [4., 5., 6.],
  ...                  [7., 8., 9.]])
  >>> x = tf.reshape(x, [1, 3, 3, 1])
  >>> max_pool_2d = tf.keras.layers.MaxPooling2D(pool_size=(2, 2),
  ...    strides=(1, 1), padding='valid')
  >>> max_pool_2d(x)
  <tf.Tensor: shape=(1, 2, 2, 1), dtype=float32, numpy=
    array([[[[5.],
             [6.]],
            [[8.],
             [9.]]]], dtype=float32)>

  For example, for `strides=(2, 2)` and `padding="valid"`:

  >>> x = tf.constant([[1., 2., 3., 4.],
  ...                  [5., 6., 7., 8.],
  ...                  [9., 10., 11., 12.]])
  >>> x = tf.reshape(x, [1, 3, 4, 1])
  >>> max_pool_2d = tf.keras.layers.MaxPooling2D(pool_size=(2, 2),
  ...    strides=(2, 2), padding='valid')
  >>> max_pool_2d(x)
  <tf.Tensor: shape=(1, 1, 2, 1), dtype=float32, numpy=
    array([[[[6.],
             [8.]]]], dtype=float32)>

  Usage Example:

  >>> input_image = tf.constant([[[[1.], [1.], [2.], [4.]],
  ...                            [[2.], [2.], [3.], [2.]],
  ...                            [[4.], [1.], [1.], [1.]],
  ...                            [[2.], [2.], [1.], [4.]]]])
  >>> output = tf.constant([[[[1], [0]],
  ...                       [[0], [1]]]])
  >>> model = tf.keras.models.Sequential()
  >>> model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2),
  ...    input_shape=(4, 4, 1)))
  >>> model.compile('adam', 'mean_squared_error')
  >>> model.predict(input_image, steps=1)
  array([[[[2.],
           [4.]],
          [[4.],
           [4.]]]], dtype=float32)

  For example, for stride=(1, 1) and padding="same":

  >>> x = tf.constant([[1., 2., 3.],
  ...                  [4., 5., 6.],
  ...                  [7., 8., 9.]])
  >>> x = tf.reshape(x, [1, 3, 3, 1])
  >>> max_pool_2d = tf.keras.layers.MaxPooling2D(pool_size=(2, 2),
  ...    strides=(1, 1), padding='same')
  >>> max_pool_2d(x)
  <tf.Tensor: shape=(1, 3, 3, 1), dtype=float32, numpy=
    array([[[[5.],
             [6.],
             [6.]],
            [[8.],
             [9.],
             [9.]],
            [[8.],
             [9.],
             [9.]]]], dtype=float32)>

  Args:
    pool_size: integer or tuple of 2 integers,
      window size over which to take the maximum.
      `(2, 2)` will take the max value over a 2x2 pooling window.
      If only one integer is specified, the same window length
      will be used for both dimensions.
    strides: Integer, tuple of 2 integers, or None.
      Strides values.  Specifies how far the pooling window moves
      for each pooling step. If None, it will default to `pool_size`.
    padding: One of `"valid"` or `"same"` (case-insensitive).
      `"valid"` means no padding. `"same"` results in padding evenly to
      the left/right or up/down of the input such that output has the same
      height/width dimension as the input.
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
      4D tensor with shape `(batch_size, rows, cols, channels)`.
    - If `data_format='channels_first'`:
      4D tensor with shape `(batch_size, channels, rows, cols)`.

  Output shape:
    - If `data_format='channels_last'`:
      4D tensor with shape `(batch_size, pooled_rows, pooled_cols, channels)`.
    - If `data_format='channels_first'`:
      4D tensor with shape `(batch_size, channels, pooled_rows, pooled_cols)`.

  Returns:
    A tensor of rank 4 representing the maximum pooled values.  See above for
    output shape.
  """

  def __init__(self,
               pool_size=(2, 2),
               strides=None,
               padding='valid',
               data_format=None,
               **kwargs):
    super(MaxPooling2D, self).__init__(
        nn.max_pool,
        pool_size=pool_size, strides=strides,
        padding=padding, data_format=data_format, **kwargs)


class AveragePooling2D(Pooling2D):
  """Average pooling operation for spatial data.

  Downsamples the input along its spatial dimensions (height and width)
  by taking the average value over an input window
  (of size defined by `pool_size`) for each channel of the input.
  The window is shifted by `strides` along each dimension.

  The resulting output when using `"valid"` padding option has a shape
  (number of rows or columns) of:
  `output_shape = math.floor((input_shape - pool_size) / strides) + 1`
  (when `input_shape >= pool_size`)

  The resulting output shape when using the `"same"` padding option is:
  `output_shape = math.floor((input_shape - 1) / strides) + 1`

  For example, for `strides=(1, 1)` and `padding="valid"`:

  >>> x = tf.constant([[1., 2., 3.],
  ...                  [4., 5., 6.],
  ...                  [7., 8., 9.]])
  >>> x = tf.reshape(x, [1, 3, 3, 1])
  >>> avg_pool_2d = tf.keras.layers.AveragePooling2D(pool_size=(2, 2),
  ...    strides=(1, 1), padding='valid')
  >>> avg_pool_2d(x)
  <tf.Tensor: shape=(1, 2, 2, 1), dtype=float32, numpy=
    array([[[[3.],
             [4.]],
            [[6.],
             [7.]]]], dtype=float32)>

  For example, for `stride=(2, 2)` and `padding="valid"`:

  >>> x = tf.constant([[1., 2., 3., 4.],
  ...                  [5., 6., 7., 8.],
  ...                  [9., 10., 11., 12.]])
  >>> x = tf.reshape(x, [1, 3, 4, 1])
  >>> avg_pool_2d = tf.keras.layers.AveragePooling2D(pool_size=(2, 2),
  ...    strides=(2, 2), padding='valid')
  >>> avg_pool_2d(x)
  <tf.Tensor: shape=(1, 1, 2, 1), dtype=float32, numpy=
    array([[[[3.5],
             [5.5]]]], dtype=float32)>

  For example, for `strides=(1, 1)` and `padding="same"`:

  >>> x = tf.constant([[1., 2., 3.],
  ...                  [4., 5., 6.],
  ...                  [7., 8., 9.]])
  >>> x = tf.reshape(x, [1, 3, 3, 1])
  >>> avg_pool_2d = tf.keras.layers.AveragePooling2D(pool_size=(2, 2),
  ...    strides=(1, 1), padding='same')
  >>> avg_pool_2d(x)
  <tf.Tensor: shape=(1, 3, 3, 1), dtype=float32, numpy=
    array([[[[3.],
             [4.],
             [4.5]],
            [[6.],
             [7.],
             [7.5]],
            [[7.5],
             [8.5],
             [9.]]]], dtype=float32)>

  Args:
    pool_size: integer or tuple of 2 integers,
      factors by which to downscale (vertical, horizontal).
      `(2, 2)` will halve the input in both spatial dimension.
      If only one integer is specified, the same window length
      will be used for both dimensions.
    strides: Integer, tuple of 2 integers, or None.
      Strides values.
      If None, it will default to `pool_size`.
    padding: One of `"valid"` or `"same"` (case-insensitive).
      `"valid"` means no padding. `"same"` results in padding evenly to
      the left/right or up/down of the input such that output has the same
      height/width dimension as the input.
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
      4D tensor with shape `(batch_size, rows, cols, channels)`.
    - If `data_format='channels_first'`:
      4D tensor with shape `(batch_size, channels, rows, cols)`.

  Output shape:
    - If `data_format='channels_last'`:
      4D tensor with shape `(batch_size, pooled_rows, pooled_cols, channels)`.
    - If `data_format='channels_first'`:
      4D tensor with shape `(batch_size, channels, pooled_rows, pooled_cols)`.
  """

  def __init__(self,
               pool_size=(2, 2),
               strides=None,
               padding='valid',
               data_format=None,
               **kwargs):
    super(AveragePooling2D, self).__init__(
        nn.avg_pool,
        pool_size=pool_size, strides=strides,
        padding=padding, data_format=data_format, **kwargs)


class Pooling3D(Layer):
  """Pooling layer for arbitrary pooling functions, for 3D inputs.

  This class only exists for code reuse. It will never be an exposed API.

  Args:
    pool_function: The pooling function to apply, e.g. `tf.nn.max_pool2d`.
    pool_size: An integer or tuple/list of 3 integers:
      (pool_depth, pool_height, pool_width)
      specifying the size of the pooling window.
      Can be a single integer to specify the same value for
      all spatial dimensions.
    strides: An integer or tuple/list of 3 integers,
      specifying the strides of the pooling operation.
      Can be a single integer to specify the same value for
      all spatial dimensions.
    padding: A string. The padding method, either 'valid' or 'same'.
      Case-insensitive.
    data_format: A string, one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch, depth, height, width, channels)`
      while `channels_first` corresponds to
      inputs with shape `(batch, channels, depth, height, width)`.
    name: A string, the name of the layer.
  """

  def __init__(self, pool_function, pool_size, strides,
               padding='valid', data_format='channels_last',
               name=None, **kwargs):
    super(Pooling3D, self).__init__(name=name, **kwargs)
    if data_format is None:
      data_format = backend.image_data_format()
    if strides is None:
      strides = pool_size
    self.pool_function = pool_function
    self.pool_size = conv_utils.normalize_tuple(pool_size, 3, 'pool_size')
    self.strides = conv_utils.normalize_tuple(strides, 3, 'strides')
    self.padding = conv_utils.normalize_padding(padding)
    self.data_format = conv_utils.normalize_data_format(data_format)
    self.input_spec = InputSpec(ndim=5)

  def call(self, inputs):
    pool_shape = (1,) + self.pool_size + (1,)
    strides = (1,) + self.strides + (1,)

    if self.data_format == 'channels_first':
      # TF does not support `channels_first` with 3D pooling operations,
      # so we must handle this case manually.
      # TODO(fchollet): remove this when TF pooling is feature-complete.
      inputs = array_ops.transpose(inputs, (0, 2, 3, 4, 1))

    outputs = self.pool_function(
        inputs,
        ksize=pool_shape,
        strides=strides,
        padding=self.padding.upper())

    if self.data_format == 'channels_first':
      outputs = array_ops.transpose(outputs, (0, 4, 1, 2, 3))
    return outputs

  def compute_output_shape(self, input_shape):
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

  def get_config(self):
    config = {
        'pool_size': self.pool_size,
        'padding': self.padding,
        'strides': self.strides,
        'data_format': self.data_format
    }
    base_config = super(Pooling3D, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class MaxPooling3D(Pooling3D):
  """Max pooling operation for 3D data (spatial or spatio-temporal).

  Downsamples the input along its spatial dimensions (depth, height, and width)
  by taking the maximum value over an input window
  (of size defined by `pool_size`) for each channel of the input.
  The window is shifted by `strides` along each dimension.

  Args:
    pool_size: Tuple of 3 integers,
      factors by which to downscale (dim1, dim2, dim3).
      `(2, 2, 2)` will halve the size of the 3D input in each dimension.
    strides: tuple of 3 integers, or None. Strides values.
    padding: One of `"valid"` or `"same"` (case-insensitive).
      `"valid"` means no padding. `"same"` results in padding evenly to
      the left/right or up/down of the input such that output has the same
      height/width dimension as the input.
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

  Example:

  ```python
  depth = 30
  height = 30
  width = 30
  input_channels = 3

  inputs = tf.keras.Input(shape=(depth, height, width, input_channels))
  layer = tf.keras.layers.MaxPooling3D(pool_size=3)
  outputs = layer(inputs)  # Shape: (batch_size, 10, 10, 10, 3)
  ```
  """

  def __init__(self,
               pool_size=(2, 2, 2),
               strides=None,
               padding='valid',
               data_format=None,
               **kwargs):
    super(MaxPooling3D, self).__init__(
        nn.max_pool3d,
        pool_size=pool_size, strides=strides,
        padding=padding, data_format=data_format, **kwargs)


class AveragePooling3D(Pooling3D):
  """Average pooling operation for 3D data (spatial or spatio-temporal).

  Downsamples the input along its spatial dimensions (depth, height, and width)
  by taking the average value over an input window
  (of size defined by `pool_size`) for each channel of the input.
  The window is shifted by `strides` along each dimension.

  Args:
    pool_size: tuple of 3 integers,
      factors by which to downscale (dim1, dim2, dim3).
      `(2, 2, 2)` will halve the size of the 3D input in each dimension.
    strides: tuple of 3 integers, or None. Strides values.
    padding: One of `"valid"` or `"same"` (case-insensitive).
      `"valid"` means no padding. `"same"` results in padding evenly to
      the left/right or up/down of the input such that output has the same
      height/width dimension as the input.
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

  Example:

  ```python
  depth = 30
  height = 30
  width = 30
  input_channels = 3

  inputs = tf.keras.Input(shape=(depth, height, width, input_channels))
  layer = tf.keras.layers.AveragePooling3D(pool_size=3)
  outputs = layer(inputs)  # Shape: (batch_size, 10, 10, 10, 3)
  ```
  """

  def __init__(self,
               pool_size=(2, 2, 2),
               strides=None,
               padding='valid',
               data_format=None,
               **kwargs):
    super(AveragePooling3D, self).__init__(
        nn.avg_pool3d,
        pool_size=pool_size, strides=strides,
        padding=padding, data_format=data_format, **kwargs)


class GlobalPooling1D(Layer):
  """Abstract class for different global pooling 1D layers."""

  def __init__(self, data_format='channels_last', keepdims=False, **kwargs):
    super(GlobalPooling1D, self).__init__(**kwargs)
    self.input_spec = InputSpec(ndim=3)
    self.data_format = conv_utils.normalize_data_format(data_format)
    self.keepdims = keepdims

  def compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    if self.data_format == 'channels_first':
      if self.keepdims:
        return tensor_shape.TensorShape([input_shape[0], input_shape[1], 1])
      else:
        return tensor_shape.TensorShape([input_shape[0], input_shape[1]])
    else:
      if self.keepdims:
        return tensor_shape.TensorShape([input_shape[0], 1, input_shape[2]])
      else:
        return tensor_shape.TensorShape([input_shape[0], input_shape[2]])

  def call(self, inputs):
    raise NotImplementedError

  def get_config(self):
    config = {'data_format': self.data_format, 'keepdims': self.keepdims}
    base_config = super(GlobalPooling1D, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class GlobalAveragePooling1D(GlobalPooling1D):
  """Global average pooling operation for temporal data.

  Examples:

  >>> input_shape = (2, 3, 4)
  >>> x = tf.random.normal(input_shape)
  >>> y = tf.keras.layers.GlobalAveragePooling1D()(x)
  >>> print(y.shape)
  (2, 4)

  Args:
    data_format: A string,
      one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch, steps, features)` while `channels_first`
      corresponds to inputs with shape
      `(batch, features, steps)`.
    keepdims: A boolean, whether to keep the temporal dimension or not.
      If `keepdims` is `False` (default), the rank of the tensor is reduced
      for spatial dimensions.
      If `keepdims` is `True`, the temporal dimension are retained with
      length 1.
      The behavior is the same as for `tf.reduce_mean` or `np.mean`.

  Call arguments:
    inputs: A 3D tensor.
    mask: Binary tensor of shape `(batch_size, steps)` indicating whether
      a given step should be masked (excluded from the average).

  Input shape:
    - If `data_format='channels_last'`:
      3D tensor with shape:
      `(batch_size, steps, features)`
    - If `data_format='channels_first'`:
      3D tensor with shape:
      `(batch_size, features, steps)`

  Output shape:
    - If `keepdims`=False:
      2D tensor with shape `(batch_size, features)`.
    - If `keepdims`=True:
      - If `data_format='channels_last'`:
        3D tensor with shape `(batch_size, 1, features)`
      - If `data_format='channels_first'`:
        3D tensor with shape `(batch_size, features, 1)`
  """

  def __init__(self, data_format='channels_last', **kwargs):
    super(GlobalAveragePooling1D, self).__init__(data_format=data_format,
                                                 **kwargs)
    self.supports_masking = True

  def call(self, inputs, mask=None):
    steps_axis = 1 if self.data_format == 'channels_last' else 2
    if mask is not None:
      mask = math_ops.cast(mask, inputs[0].dtype)
      mask = array_ops.expand_dims(
          mask, 2 if self.data_format == 'channels_last' else 1)
      inputs *= mask
      return backend.sum(
          inputs, axis=steps_axis,
          keepdims=self.keepdims) / math_ops.reduce_sum(
              mask, axis=steps_axis, keepdims=self.keepdims)
    else:
      return backend.mean(inputs, axis=steps_axis, keepdims=self.keepdims)

  def compute_mask(self, inputs, mask=None):
    return None


class GlobalMaxPooling1D(GlobalPooling1D):
  """Global max pooling operation for 1D temporal data.

  Downsamples the input representation by taking the maximum value over
  the time dimension.

  For example:

  >>> x = tf.constant([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
  >>> x = tf.reshape(x, [3, 3, 1])
  >>> x
  <tf.Tensor: shape=(3, 3, 1), dtype=float32, numpy=
  array([[[1.], [2.], [3.]],
         [[4.], [5.], [6.]],
         [[7.], [8.], [9.]]], dtype=float32)>
  >>> max_pool_1d = tf.keras.layers.GlobalMaxPooling1D()
  >>> max_pool_1d(x)
  <tf.Tensor: shape=(3, 1), dtype=float32, numpy=
  array([[3.],
         [6.],
         [9.], dtype=float32)>

  Args:
    data_format: A string,
      one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch, steps, features)` while `channels_first`
      corresponds to inputs with shape
      `(batch, features, steps)`.
    keepdims: A boolean, whether to keep the temporal dimension or not.
      If `keepdims` is `False` (default), the rank of the tensor is reduced
      for spatial dimensions.
      If `keepdims` is `True`, the temporal dimension are retained with
      length 1.
      The behavior is the same as for `tf.reduce_max` or `np.max`.

  Input shape:
    - If `data_format='channels_last'`:
      3D tensor with shape:
      `(batch_size, steps, features)`
    - If `data_format='channels_first'`:
      3D tensor with shape:
      `(batch_size, features, steps)`

  Output shape:
    - If `keepdims`=False:
      2D tensor with shape `(batch_size, features)`.
    - If `keepdims`=True:
      - If `data_format='channels_last'`:
        3D tensor with shape `(batch_size, 1, features)`
      - If `data_format='channels_first'`:
        3D tensor with shape `(batch_size, features, 1)`
  """

  def call(self, inputs):
    steps_axis = 1 if self.data_format == 'channels_last' else 2
    return backend.max(inputs, axis=steps_axis, keepdims=self.keepdims)


class GlobalPooling2D(Layer):
  """Abstract class for different global pooling 2D layers.
  """

  def __init__(self, data_format=None, keepdims=False, **kwargs):
    super(GlobalPooling2D, self).__init__(**kwargs)
    self.data_format = conv_utils.normalize_data_format(data_format)
    self.input_spec = InputSpec(ndim=4)
    self.keepdims = keepdims

  def compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    if self.data_format == 'channels_last':
      if self.keepdims:
        return tensor_shape.TensorShape([input_shape[0], 1, 1, input_shape[3]])
      else:
        return tensor_shape.TensorShape([input_shape[0], input_shape[3]])
    else:
      if self.keepdims:
        return tensor_shape.TensorShape([input_shape[0], input_shape[1], 1, 1])
      else:
        return tensor_shape.TensorShape([input_shape[0], input_shape[1]])

  def call(self, inputs):
    raise NotImplementedError

  def get_config(self):
    config = {'data_format': self.data_format, 'keepdims': self.keepdims}
    base_config = super(GlobalPooling2D, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class GlobalAveragePooling2D(GlobalPooling2D):
  """Global average pooling operation for spatial data.

  Examples:

  >>> input_shape = (2, 4, 5, 3)
  >>> x = tf.random.normal(input_shape)
  >>> y = tf.keras.layers.GlobalAveragePooling2D()(x)
  >>> print(y.shape)
  (2, 3)

  Args:
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
      keepdims: A boolean, whether to keep the spatial dimensions or not.
        If `keepdims` is `False` (default), the rank of the tensor is reduced
        for spatial dimensions.
        If `keepdims` is `True`, the spatial dimensions are retained with
        length 1.
        The behavior is the same as for `tf.reduce_mean` or `np.mean`.

  Input shape:
    - If `data_format='channels_last'`:
      4D tensor with shape `(batch_size, rows, cols, channels)`.
    - If `data_format='channels_first'`:
      4D tensor with shape `(batch_size, channels, rows, cols)`.

  Output shape:
    - If `keepdims`=False:
      2D tensor with shape `(batch_size, channels)`.
    - If `keepdims`=True:
      - If `data_format='channels_last'`:
        4D tensor with shape `(batch_size, 1, 1, channels)`
      - If `data_format='channels_first'`:
        4D tensor with shape `(batch_size, channels, 1, 1)`
  """

  def call(self, inputs):
    if self.data_format == 'channels_last':
      return backend.mean(inputs, axis=[1, 2], keepdims=self.keepdims)
    else:
      return backend.mean(inputs, axis=[2, 3], keepdims=self.keepdims)


class GlobalMaxPooling2D(GlobalPooling2D):
  """Global max pooling operation for spatial data.

  Examples:

  >>> input_shape = (2, 4, 5, 3)
  >>> x = tf.random.normal(input_shape)
  >>> y = tf.keras.layers.GlobalMaxPool2D()(x)
  >>> print(y.shape)
  (2, 3)

  Args:
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
    keepdims: A boolean, whether to keep the spatial dimensions or not.
      If `keepdims` is `False` (default), the rank of the tensor is reduced
      for spatial dimensions.
      If `keepdims` is `True`, the spatial dimensions are retained with
      length 1.
      The behavior is the same as for `tf.reduce_max` or `np.max`.

  Input shape:
    - If `data_format='channels_last'`:
      4D tensor with shape `(batch_size, rows, cols, channels)`.
    - If `data_format='channels_first'`:
      4D tensor with shape `(batch_size, channels, rows, cols)`.

  Output shape:
    - If `keepdims`=False:
      2D tensor with shape `(batch_size, channels)`.
    - If `keepdims`=True:
      - If `data_format='channels_last'`:
        4D tensor with shape `(batch_size, 1, 1, channels)`
      - If `data_format='channels_first'`:
        4D tensor with shape `(batch_size, channels, 1, 1)`
  """

  def call(self, inputs):
    if self.data_format == 'channels_last':
      return backend.max(inputs, axis=[1, 2], keepdims=self.keepdims)
    else:
      return backend.max(inputs, axis=[2, 3], keepdims=self.keepdims)


class GlobalPooling3D(Layer):
  """Abstract class for different global pooling 3D layers."""

  def __init__(self, data_format=None, keepdims=False, **kwargs):
    super(GlobalPooling3D, self).__init__(**kwargs)
    self.data_format = conv_utils.normalize_data_format(data_format)
    self.input_spec = InputSpec(ndim=5)
    self.keepdims = keepdims

  def compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    if self.data_format == 'channels_last':
      if self.keepdims:
        return tensor_shape.TensorShape(
            [input_shape[0], 1, 1, 1, input_shape[4]])
      else:
        return tensor_shape.TensorShape([input_shape[0], input_shape[4]])
    else:
      if self.keepdims:
        return tensor_shape.TensorShape(
            [input_shape[0], input_shape[1], 1, 1, 1])
      else:
        return tensor_shape.TensorShape([input_shape[0], input_shape[1]])

  def call(self, inputs):
    raise NotImplementedError

  def get_config(self):
    config = {'data_format': self.data_format, 'keepdims': self.keepdims}
    base_config = super(GlobalPooling3D, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class GlobalAveragePooling3D(GlobalPooling3D):
  """Global Average pooling operation for 3D data.

  Args:
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
    keepdims: A boolean, whether to keep the spatial dimensions or not.
      If `keepdims` is `False` (default), the rank of the tensor is reduced
      for spatial dimensions.
      If `keepdims` is `True`, the spatial dimensions are retained with
      length 1.
      The behavior is the same as for `tf.reduce_mean` or `np.mean`.

  Input shape:
    - If `data_format='channels_last'`:
      5D tensor with shape:
      `(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
    - If `data_format='channels_first'`:
      5D tensor with shape:
      `(batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)`

  Output shape:
    - If `keepdims`=False:
      2D tensor with shape `(batch_size, channels)`.
    - If `keepdims`=True:
      - If `data_format='channels_last'`:
        5D tensor with shape `(batch_size, 1, 1, 1, channels)`
      - If `data_format='channels_first'`:
        5D tensor with shape `(batch_size, channels, 1, 1, 1)`
  """

  def call(self, inputs):
    if self.data_format == 'channels_last':
      return backend.mean(inputs, axis=[1, 2, 3], keepdims=self.keepdims)
    else:
      return backend.mean(inputs, axis=[2, 3, 4], keepdims=self.keepdims)


class GlobalMaxPooling3D(GlobalPooling3D):
  """Global Max pooling operation for 3D data.

  Args:
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
    keepdims: A boolean, whether to keep the spatial dimensions or not.
      If `keepdims` is `False` (default), the rank of the tensor is reduced
      for spatial dimensions.
      If `keepdims` is `True`, the spatial dimensions are retained with
      length 1.
      The behavior is the same as for `tf.reduce_max` or `np.max`.

  Input shape:
    - If `data_format='channels_last'`:
      5D tensor with shape:
      `(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
    - If `data_format='channels_first'`:
      5D tensor with shape:
      `(batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)`

  Output shape:
    - If `keepdims`=False:
      2D tensor with shape `(batch_size, channels)`.
    - If `keepdims`=True:
      - If `data_format='channels_last'`:
        5D tensor with shape `(batch_size, 1, 1, 1, channels)`
      - If `data_format='channels_first'`:
        5D tensor with shape `(batch_size, channels, 1, 1, 1)`
  """

  def call(self, inputs):
    if self.data_format == 'channels_last':
      return backend.max(inputs, axis=[1, 2, 3], keepdims=self.keepdims)
    else:
      return backend.max(inputs, axis=[2, 3, 4], keepdims=self.keepdims)


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
