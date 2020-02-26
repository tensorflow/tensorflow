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
"""Differentiating layers.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.backend import permute_dimensions, dot, constant
from tensorflow.python.util.tf_export import keras_export


@keras_export('keras.layers.Differentiating1D', 'keras.layers.Diff1D')
class Differentiating1D(Layer):
  """Differentiating operation for 1D temporal data.

  Differentiate the input representation with respect to steps.

  The resulting output when using "valid" padding options has a shape of:
  `output_shape = input_shape - 1`

  When using "same" padding options, the last element is appended once more
  and the result has a shape of:
  `output_shape = input_shape`

  For example, for padding="valid"

  >>> x = tf.constant([1., 8., 5., 6., 2.])
  >>> x = tf.reshape(x, [1, 5, 1])
  >>> diff1d = Differentiating1D(padding="valid")(x)
  >>> tf.keras.backend.get_value(diff1d)

  array([[[ 7.],
          [-3.],
          [ 1.],
          [-4.]]], dtype=float32)

  For example, for padding="same"

  >>> x = tf.constant([1., 8., 5., 6., 2.])
  >>> x = tf.reshape(x, [1, 5, 1])
  >>> diff1d = Differentiating1D(padding="same")(x)
  >>> tf.keras.backend.get_value(diff1d)

  array([[[ 7.],
          [-3.],
          [ 1.],
          [-4.],
          [-4.]]], dtype=float32)

  Arguments:
    padding: One of `"valid"` or `"same"` (case-insensitive).
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
    - If `data_format='channels_last'` and `padding='valid'`:
      3D tensor with shape `(batch_size, steps - 1, features)`.
    - If `data_format='channels_first'` and `padding='valid'`:
      3D tensor with shape `(batch_size, features, steps - 1)`.
    - If `data_format='channels_last'` and `padding='same'`:
      3D tensor with shape `(batch_size, steps, features)`.
    - If `data_format='channels_first'` and `padding='same'`:
      3D tensor with shape `(batch_size, features, steps)`.
  """

  def __init__(self, padding='valid', data_format='channels_last', **kwargs):
    self.padding = padding.lower()
    self.derivative_matrix = None
    self.data_format = data_format.lower()
    super(Differentiating1D, self).__init__(**kwargs)

  def build(self, input_shape):
    assert len(input_shape) == 3, 'Wrong input shape. Is it an 1D vector?'
    assert self.padding in ('valid', 'same'),\
      'Wrong padding %s.' % self.data_format
    assert self.data_format in ('channels_first', 'channels_last'),\
      'Wrong data_format %s.' % self.data_format

    if self.data_format == 'channels_last':
      input_len = input_shape[1]
    else:
      input_len = input_shape[2]

    if self.padding == 'valid':
      matrix_shape = [input_len, input_len - 1]
    elif self.padding == 'same':
      matrix_shape = [input_len, input_len]
    else:
      assert False

    drv_mat = np.zeros(matrix_shape, dtype=np.float32)

    for i in range(input_len - 1):
      drv_mat[i, i] = -1
      drv_mat[i + 1, i] = 1
    if self.padding == 'same':
      i = input_len - 1
      drv_mat[i - 1, i] = -1
      drv_mat[i, i] = 1

    # Create a trainable weight variable for this layer.
    self.derivative_matrix = constant(drv_mat)

    # Be sure to call this at the end
    super(Differentiating1D, self).build(input_shape)

  def call(self, inputs):

    if self.data_format == 'channels_last':
      reshaped_input = permute_dimensions(inputs, (0, 2, 1))
    else:
      reshaped_input = inputs
    y = dot(reshaped_input, self.derivative_matrix)
    if self.data_format == 'channels_last':
      y = permute_dimensions(y, (0, 2, 1))

    return y

  def compute_output_shape(self, input_shape):
    if self.padding == 'same':
      output_shape = input_shape
    elif self.data_format == 'channels_first':
      output_shape = [input_shape[0], input_shape[1], input_shape[2] - 1]
    else:
      output_shape = [input_shape[0], input_shape[1] - 1, input_shape[2]]
    return tensor_shape.TensorShape(output_shape)

  def get_config(self):
    config = {
        'padding': self.padding,
        'data_format': self.data_format
    }
    base_config = super(Differentiating1D, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

# Aliases

Diff1D = Differentiating1D
