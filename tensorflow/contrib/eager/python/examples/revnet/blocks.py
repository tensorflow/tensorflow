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
               fused=True):
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
          fused=fused)
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
      y_inv = x if i == 0 else block.backward(y, training=training)
      dy, grads, vars_ = block.backward_grads_and_vars(
          y_inv, dy, training=training)
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
  """

  def __init__(self,
               filters,
               strides,
               input_shape,
               batch_norm_first=True,
               data_format="channels_first",
               bottleneck=False,
               fused=True):
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
        fused=fused)
    self.g = factory(
        filters=filters // 2,
        strides=(1, 1),
        input_shape=g_input_shape,
        batch_norm_first=batch_norm_first,
        data_format=data_format,
        fused=fused)

  def call(self, x, training=True, concat=True):
    """Apply residual block to inputs."""

    x1, x2 = tf.split(x, num_or_size_splits=2, axis=self.axis)
    f_x2 = self.f.call(x2, training=training)
    # TODO(lxuechen): Replace with simpler downsampling
    x1_down = ops.downsample(
        x1, self.filters // 2, self.strides, axis=self.axis)
    x2_down = ops.downsample(
        x2, self.filters // 2, self.strides, axis=self.axis)
    y1 = f_x2 + x1_down
    g_y1 = self.g.call(y1, training=training)  # self.g(y1) gives pylint error
    y2 = g_y1 + x2_down
    if not concat:  # Concat option needed for correct backward grads
      return y1, y2
    return tf.concat([y1, y2], axis=self.axis)

  def backward(self, y, training=True):
    """Reconstruct inputs from outputs; only valid when stride 1."""

    assert self.strides == (1, 1)

    y1, y2 = tf.split(y, num_or_size_splits=2, axis=self.axis)
    g_y1 = self.g.call(y1, training=training)
    x2 = y2 - g_y1
    f_x2 = self.f.call(x2, training=training)
    x1 = y1 - f_x2

    return tf.concat([x1, x2], axis=self.axis)

  def backward_grads_and_vars(self, x, dy, training=True):
    """Manually compute backward gradients given input and output grads."""

    with tf.GradientTape(persistent=True) as tape:
      x_stop = tf.stop_gradient(x)
      x1, x2 = tf.split(x_stop, num_or_size_splits=2, axis=self.axis)
      tape.watch([x1, x2])
      # Stitch back x for `call` so tape records correct grads
      x = tf.concat([x1, x2], axis=self.axis)
      dy1, dy2 = tf.split(dy, num_or_size_splits=2, axis=self.axis)
      y1, y2 = self.call(x, training=training, concat=False)
      x2_down = ops.downsample(
          x2, self.filters // 2, self.strides, axis=self.axis)

    grads_combined = tape.gradient(
        y2, [y1] + self.g.variables, output_gradients=[dy2])
    dy2_y1, dg = grads_combined[0], grads_combined[1:]
    dy1_plus = dy2_y1 + dy1

    grads_combined = tape.gradient(
        y1, [x1, x2] + self.f.variables, output_gradients=[dy1_plus])
    dx1, dx2, df = grads_combined[0], grads_combined[1], grads_combined[2:]
    dx2 += tape.gradient(x2_down, [x2], output_gradients=[dy2])[0]

    del tape

    grads = df + dg
    vars_ = self.f.variables + self.g.variables

    return tf.concat([dx1, dx2], axis=self.axis), grads, vars_


def _BottleneckResidualInner(filters,
                             strides,
                             input_shape,
                             batch_norm_first=True,
                             data_format="channels_first",
                             fused=True):
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

  Returns:
    A keras model
  """

  axis = 1 if data_format == "channels_first" else 3
  model = tf.keras.Sequential()
  if batch_norm_first:
    model.add(
        tf.keras.layers.BatchNormalization(
            axis=axis, input_shape=input_shape, fused=fused))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.))
  model.add(
      tf.keras.layers.Conv2D(
          filters=filters // 4,
          kernel_size=1,
          strides=strides,
          input_shape=input_shape,
          data_format=data_format,
          use_bias=False,
          padding="SAME"))

  model.add(tf.keras.layers.BatchNormalization(axis=axis, fused=fused))
  model.add(tf.keras.layers.LeakyReLU(alpha=0.))
  model.add(
      tf.keras.layers.Conv2D(
          filters=filters // 4,
          kernel_size=3,
          strides=(1, 1),
          data_format=data_format,
          use_bias=False,
          padding="SAME"))

  model.add(tf.keras.layers.BatchNormalization(axis=axis, fused=fused))
  model.add(tf.keras.layers.LeakyReLU(alpha=0.))
  model.add(
      tf.keras.layers.Conv2D(
          filters=filters,
          kernel_size=1,
          strides=(1, 1),
          data_format=data_format,
          use_bias=False,
          padding="SAME"))

  return model


def _ResidualInner(filters,
                   strides,
                   input_shape,
                   batch_norm_first=True,
                   data_format="channels_first",
                   fused=True):
  """Single residual inner function contained in _ResdualBlock.

  Corresponds to the `F`/`G` functions in the paper.

  Args:
    filters: output filter size
    strides: length 2 list/tuple of integers for height and width strides
    input_shape: length 3 list/tuple of integers
    batch_norm_first: whether to apply activation and batch norm before conv
    data_format: tensor data format, "NCHW"/"NHWC"
    fused: use fused batch normalization if True

  Returns:
    A keras model
  """

  axis = 1 if data_format == "channels_first" else 3
  model = tf.keras.Sequential()
  if batch_norm_first:
    model.add(
        tf.keras.layers.BatchNormalization(
            axis=axis, input_shape=input_shape, fused=fused))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.))
  model.add(
      tf.keras.layers.Conv2D(
          filters=filters,
          kernel_size=3,
          strides=strides,
          input_shape=input_shape,
          data_format=data_format,
          use_bias=False,
          padding="SAME"))

  model.add(tf.keras.layers.BatchNormalization(axis=axis, fused=fused))
  model.add(tf.keras.layers.LeakyReLU(alpha=0.))
  model.add(
      tf.keras.layers.Conv2D(
          filters=filters,
          kernel_size=3,
          strides=(1, 1),
          data_format=data_format,
          use_bias=False,
          padding="SAME"))

  return model
