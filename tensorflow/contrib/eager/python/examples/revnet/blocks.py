# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Reversible residual network compatible with eager execution.

Building blocks with manual backward gradient computation.

Reference [The Reversible Residual Network: Backpropagation
Without Storing Activations](https://arxiv.org/pdf/1707.04585.pdf)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.eager.python.examples.revnet import ops


class RevBlock(tf.keras.Model):
  """Single reversible block containing several `_Residual` blocks.

  Each `_Residual` block in turn contains two _ResidualInner blocks,
  corresponding to the `F`/`G` functions in the paper.
  """

  def __init__(self,
               n_res,
               filters,
               strides,
               input_shape,
               batch_norm_first=False,
               data_format="channels_first",
               bottleneck=False,
               fused=True,
               dtype=tf.float32):
    """Initialize RevBlock.

    Args:
      n_res: number of residual blocks
      filters: list/tuple of integers for output filter sizes of each residual
      strides: length 2 list/tuple of integers for height and width strides
      input_shape: length 3 list/tuple of integers
      batch_norm_first: whether to apply activation and batch norm before conv
      data_format: tensor data format, "NCHW"/"NHWC"
      bottleneck: use bottleneck residual if True
      fused: use fused batch normalization if True
      dtype: float16, float32, or float64
    """
    super(RevBlock, self).__init__()
    self.blocks = tf.contrib.checkpoint.List()
    for i in range(n_res):
      curr_batch_norm_first = batch_norm_first and i == 0
      curr_strides = strides if i == 0 else (1, 1)
      block = _Residual(
          filters,
          curr_strides,
          input_shape,
          batch_norm_first=curr_batch_norm_first,
          data_format=data_format,
          bottleneck=bottleneck,
          fused=fused,
          dtype=dtype)
      self.blocks.append(block)

      if data_format == "channels_first":
        input_shape = (filters, input_shape[1] // curr_strides[0],
                       input_shape[2] // curr_strides[1])
      else:
        input_shape = (input_shape[0] // curr_strides[0],
                       input_shape[1] // curr_strides[1], filters)

  def call(self, h, training=True):
    """Apply reversible block to inputs."""

    for block in self.blocks:
      h = block(h, training=training)
    return h

  def backward_grads_and_vars(self, x, y, dy, training=True):
    """Apply reversible block backward to outputs."""

    grads_all = []
    vars_all = []

    for i in reversed(range(len(self.blocks))):
      block = self.blocks[i]
      if i == 0:
        # First block usually contains downsampling that can't be reversed
        with tf.GradientTape() as tape:
          x = tf.identity(x)
          tape.watch(x)
          y = block(x, training=training)

        grads_combined = tape.gradient(
            y, [x] + block.trainable_variables, output_gradients=dy)
        dy = grads_combined[0]
        grads_all += grads_combined[1:]
        vars_all += block.trainable_variables
      else:
        y, dy, grads, vars_ = block.backward_grads_and_vars(
            y, dy, training=training)
        grads_all += grads
        vars_all += vars_

    return dy, grads_all, vars_all


class _Residual(tf.keras.Model):
  """Single residual block contained in a _RevBlock. Each `_Residual` object has
  two _ResidualInner objects, corresponding to the `F` and `G` functions in the
  paper.

  Args:
    filters: output filter size
    strides: length 2 list/tuple of integers for height and width strides
    input_shape: length 3 list/tuple of integers
    batch_norm_first: whether to apply activation and batch norm before conv
    data_format: tensor data format, "NCHW"/"NHWC",
    bottleneck: use bottleneck residual if True
    fused: use fused batch normalization if True
    dtype: float16, float32, or float64
  """

  def __init__(self,
               filters,
               strides,
               input_shape,
               batch_norm_first=True,
               data_format="channels_first",
               bottleneck=False,
               fused=True,
               dtype=tf.float32):
    super(_Residual, self).__init__()

    self.filters = filters
    self.strides = strides
    self.axis = 1 if data_format == "channels_first" else 3
    if data_format == "channels_first":
      f_input_shape = (input_shape[0] // 2,) + input_shape[1:]
      g_input_shape = (filters // 2, input_shape[1] // strides[0],
                       input_shape[2] // strides[1])
    else:
      f_input_shape = input_shape[:2] + (input_shape[2] // 2,)
      g_input_shape = (input_shape[0] // strides[0],
                       input_shape[1] // strides[1], filters // 2)

    factory = _BottleneckResidualInner if bottleneck else _ResidualInner
    self.f = factory(
        filters=filters // 2,
        strides=strides,
        input_shape=f_input_shape,
        batch_norm_first=batch_norm_first,
        data_format=data_format,
        fused=fused,
        dtype=dtype)
    self.g = factory(
        filters=filters // 2,
        strides=(1, 1),
        input_shape=g_input_shape,
        batch_norm_first=batch_norm_first,
        data_format=data_format,
        fused=fused,
        dtype=dtype)

  def call(self, x, training=True, concat=True):
    """Apply residual block to inputs."""

    x1, x2 = tf.split(x, num_or_size_splits=2, axis=self.axis)
    f_x2 = self.f(x2, training=training)
    x1_down = ops.downsample(
        x1, self.filters // 2, self.strides, axis=self.axis)
    x2_down = ops.downsample(
        x2, self.filters // 2, self.strides, axis=self.axis)
    y1 = f_x2 + x1_down
    g_y1 = self.g(y1, training=training)
    y2 = g_y1 + x2_down
    if not concat:  # For correct backward grads
      return y1, y2

    return tf.concat([y1, y2], axis=self.axis)

  def backward_grads_and_vars(self, y, dy, training=True):
    """Manually compute backward gradients given input and output grads."""
    dy1, dy2 = tf.split(dy, num_or_size_splits=2, axis=self.axis)

    with tf.GradientTape(persistent=True) as tape:
      y = tf.identity(y)
      tape.watch(y)
      y1, y2 = tf.split(y, num_or_size_splits=2, axis=self.axis)
      z1 = y1
      gz1 = self.g(z1, training=training)
      x2 = y2 - gz1
      fx2 = self.f(x2, training=training)
      x1 = z1 - fx2

    grads_combined = tape.gradient(
        gz1, [z1] + self.g.trainable_variables, output_gradients=dy2)
    dz1 = dy1 + grads_combined[0]
    dg = grads_combined[1:]
    dx1 = dz1

    grads_combined = tape.gradient(
        fx2, [x2] + self.f.trainable_variables, output_gradients=dz1)
    dx2 = dy2 + grads_combined[0]
    df = grads_combined[1:]

    del tape

    grads = df + dg
    vars_ = self.f.trainable_variables + self.g.trainable_variables

    x = tf.concat([x1, x2], axis=self.axis)
    dx = tf.concat([dx1, dx2], axis=self.axis)

    return x, dx, grads, vars_


def _BottleneckResidualInner(filters,
                             strides,
                             input_shape,
                             batch_norm_first=True,
                             data_format="channels_first",
                             fused=True,
                             dtype=tf.float32):
  """Single bottleneck residual inner function contained in _Resdual.

  Corresponds to the `F`/`G` functions in the paper.
  Suitable for training on ImageNet dataset.

  Args:
    filters: output filter size
    strides: length 2 list/tuple of integers for height and width strides
    input_shape: length 3 list/tuple of integers
    batch_norm_first: whether to apply activation and batch norm before conv
    data_format: tensor data format, "NCHW"/"NHWC"
    fused: use fused batch normalization if True
    dtype: float16, float32, or float64

  Returns:
    A keras model
  """

  axis = 1 if data_format == "channels_first" else 3
  model = tf.keras.Sequential()
  if batch_norm_first:
    model.add(
        tf.keras.layers.BatchNormalization(
            axis=axis, input_shape=input_shape, fused=fused, dtype=dtype))
    model.add(tf.keras.layers.Activation("relu"))
  model.add(
      tf.keras.layers.Conv2D(
          filters=filters // 4,
          kernel_size=1,
          strides=strides,
          input_shape=input_shape,
          data_format=data_format,
          use_bias=False,
          padding="SAME",
          dtype=dtype))

  model.add(
      tf.keras.layers.BatchNormalization(axis=axis, fused=fused, dtype=dtype))
  model.add(tf.keras.layers.Activation("relu"))
  model.add(
      tf.keras.layers.Conv2D(
          filters=filters // 4,
          kernel_size=3,
          strides=(1, 1),
          data_format=data_format,
          use_bias=False,
          padding="SAME",
          dtype=dtype))

  model.add(
      tf.keras.layers.BatchNormalization(axis=axis, fused=fused, dtype=dtype))
  model.add(tf.keras.layers.Activation("relu"))
  model.add(
      tf.keras.layers.Conv2D(
          filters=filters,
          kernel_size=1,
          strides=(1, 1),
          data_format=data_format,
          use_bias=False,
          padding="SAME",
          dtype=dtype))

  return model


def _ResidualInner(filters,
                   strides,
                   input_shape,
                   batch_norm_first=True,
                   data_format="channels_first",
                   fused=True,
                   dtype=tf.float32):
  """Single residual inner function contained in _ResdualBlock.

  Corresponds to the `F`/`G` functions in the paper.

  Args:
    filters: output filter size
    strides: length 2 list/tuple of integers for height and width strides
    input_shape: length 3 list/tuple of integers
    batch_norm_first: whether to apply activation and batch norm before conv
    data_format: tensor data format, "NCHW"/"NHWC"
    fused: use fused batch normalization if True
    dtype: float16, float32, or float64

  Returns:
    A keras model
  """

  axis = 1 if data_format == "channels_first" else 3
  model = tf.keras.Sequential()
  if batch_norm_first:
    model.add(
        tf.keras.layers.BatchNormalization(
            axis=axis, input_shape=input_shape, fused=fused, dtype=dtype))
    model.add(tf.keras.layers.Activation("relu"))
  model.add(
      tf.keras.layers.Conv2D(
          filters=filters,
          kernel_size=3,
          strides=strides,
          input_shape=input_shape,
          data_format=data_format,
          use_bias=False,
          padding="SAME",
          dtype=dtype))

  model.add(
      tf.keras.layers.BatchNormalization(axis=axis, fused=fused, dtype=dtype))
  model.add(tf.keras.layers.Activation("relu"))
  model.add(
      tf.keras.layers.Conv2D(
          filters=filters,
          kernel_size=3,
          strides=(1, 1),
          data_format=data_format,
          use_bias=False,
          padding="SAME",
          dtype=dtype))

  return model
