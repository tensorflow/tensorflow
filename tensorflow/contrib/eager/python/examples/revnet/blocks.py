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

import functools
import operator

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
    """Initialization.

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
    """Initialization.

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


# Ideally, the following should be wrapped in `tf.keras.Sequential`, however
# there are subtle issues with its placeholder insertion policy and batch norm
class _BottleneckResidualInner(tf.keras.Model):
  """Single bottleneck residual inner function contained in _Resdual.

  Corresponds to the `F`/`G` functions in the paper.
  Suitable for training on ImageNet dataset.
  """

  def __init__(self,
               filters,
               strides,
               input_shape,
               batch_norm_first=True,
               data_format="channels_first",
               fused=True,
               dtype=tf.float32):
    """Initialization.

    Args:
      filters: output filter size
      strides: length 2 list/tuple of integers for height and width strides
      input_shape: length 3 list/tuple of integers
      batch_norm_first: whether to apply activation and batch norm before conv
      data_format: tensor data format, "NCHW"/"NHWC"
      fused: use fused batch normalization if True
      dtype: float16, float32, or float64
    """
    super(_BottleneckResidualInner, self).__init__()
    axis = 1 if data_format == "channels_first" else 3
    if batch_norm_first:
      self.batch_norm_0 = tf.keras.layers.BatchNormalization(
          axis=axis, input_shape=input_shape, fused=fused, dtype=dtype)

    self.conv2d_1 = tf.keras.layers.Conv2D(
        filters=filters // 4,
        kernel_size=1,
        strides=strides,
        input_shape=input_shape,
        data_format=data_format,
        use_bias=False,
        padding="SAME",
        dtype=dtype)
    self.batch_norm_1 = tf.keras.layers.BatchNormalization(
        axis=axis, fused=fused, dtype=dtype)

    self.conv2d_2 = tf.keras.layers.Conv2D(
        filters=filters // 4,
        kernel_size=3,
        strides=(1, 1),
        data_format=data_format,
        use_bias=False,
        padding="SAME",
        dtype=dtype)

    self.batch_norm_2 = tf.keras.layers.BatchNormalization(
        axis=axis, fused=fused, dtype=dtype)
    self.conv2d_3 = tf.keras.layers.Conv2D(
        filters=filters,
        kernel_size=1,
        strides=(1, 1),
        data_format=data_format,
        use_bias=False,
        padding="SAME",
        dtype=dtype)

    self.batch_norm_first = batch_norm_first

  def call(self, x, training=True):
    net = x
    if self.batch_norm_first:
      net = self.batch_norm_0(net, training=training)
      net = tf.nn.relu(net)

    net = self.conv2d_1(net)
    net = self.batch_norm_1(net, training=training)
    net = tf.nn.relu(net)

    net = self.conv2d_2(net)
    net = self.batch_norm_2(net, training=training)
    net = tf.nn.relu(net)

    net = self.conv2d_3(net)

    return net


class _ResidualInner(tf.keras.Model):
  """Single residual inner function contained in _ResdualBlock.

  Corresponds to the `F`/`G` functions in the paper.
  """

  def __init__(self,
               filters,
               strides,
               input_shape,
               batch_norm_first=True,
               data_format="channels_first",
               fused=True,
               dtype=tf.float32):
    """Initialization.

    Args:
      filters: output filter size
      strides: length 2 list/tuple of integers for height and width strides
      input_shape: length 3 list/tuple of integers
      batch_norm_first: whether to apply activation and batch norm before conv
      data_format: tensor data format, "NCHW"/"NHWC"
      fused: use fused batch normalization if True
      dtype: float16, float32, or float64
    """
    super(_ResidualInner, self).__init__()
    axis = 1 if data_format == "channels_first" else 3
    if batch_norm_first:
      self.batch_norm_0 = tf.keras.layers.BatchNormalization(
          axis=axis, input_shape=input_shape, fused=fused, dtype=dtype)
    self.conv2d_1 = tf.keras.layers.Conv2D(
        filters=filters,
        kernel_size=3,
        strides=strides,
        input_shape=input_shape,
        data_format=data_format,
        use_bias=False,
        padding="SAME",
        dtype=dtype)
    self.batch_norm_1 = tf.keras.layers.BatchNormalization(
        axis=axis, fused=fused, dtype=dtype)

    self.conv2d_2 = tf.keras.layers.Conv2D(
        filters=filters,
        kernel_size=3,
        strides=(1, 1),
        data_format=data_format,
        use_bias=False,
        padding="SAME",
        dtype=dtype)

    self.batch_norm_first = batch_norm_first

  def call(self, x, training=True):
    net = x
    if self.batch_norm_first:
      net = self.batch_norm_0(net, training=training)
      net = tf.nn.relu(net)

    net = self.conv2d_1(net)
    net = self.batch_norm_1(net, training=training)

    net = self.conv2d_2(net)

    return net


class InitBlock(tf.keras.Model):
  """Initial block of RevNet."""

  def __init__(self, config):
    """Initialization.

    Args:
      config: tf.contrib.training.HParams object; specifies hyperparameters
    """
    super(InitBlock, self).__init__()
    self.config = config
    self.axis = 1 if self.config.data_format == "channels_first" else 3
    self.conv2d = tf.keras.layers.Conv2D(
        filters=self.config.init_filters,
        kernel_size=self.config.init_kernel,
        strides=(self.config.init_stride, self.config.init_stride),
        data_format=self.config.data_format,
        use_bias=False,
        padding="SAME",
        input_shape=self.config.input_shape,
        dtype=self.config.dtype)
    self.batch_norm = tf.keras.layers.BatchNormalization(
        axis=self.axis, fused=self.config.fused, dtype=self.config.dtype)
    self.activation = tf.keras.layers.Activation("relu")

    if self.config.init_max_pool:
      self.max_pool = tf.keras.layers.MaxPooling2D(
          pool_size=(3, 3),
          strides=(2, 2),
          padding="SAME",
          data_format=self.config.data_format,
          dtype=self.config.dtype)

  def call(self, x, training=True):
    net = x
    net = self.conv2d(net)
    net = self.batch_norm(net, training=training)
    net = self.activation(net)

    if self.config.init_max_pool:
      net = self.max_pool(net)

    return net


class FinalBlock(tf.keras.Model):
  """Final block of RevNet."""

  def __init__(self, config):
    """Initialization.

    Args:
      config: tf.contrib.training.HParams object; specifies hyperparameters

    Raises:
      ValueError: Unsupported data format
    """
    super(FinalBlock, self).__init__()
    self.config = config
    self.axis = 1 if self.config.data_format == "channels_first" else 3

    f = self.config.filters[-1]  # Number of filters
    r = functools.reduce(operator.mul, self.config.strides, 1)  # Reduce ratio
    r *= self.config.init_stride
    if self.config.init_max_pool:
      r *= 2

    if self.config.data_format == "channels_first":
      w, h = self.config.input_shape[1], self.config.input_shape[2]
      input_shape = (f, w // r, h // r)
    elif self.config.data_format == "channels_last":
      w, h = self.config.input_shape[0], self.config.input_shape[1]
      input_shape = (w // r, h // r, f)
    else:
      raise ValueError("Data format should be either `channels_first`"
                       " or `channels_last`")
    self.batch_norm = tf.keras.layers.BatchNormalization(
        axis=self.axis,
        input_shape=input_shape,
        fused=self.config.fused,
        dtype=self.config.dtype)
    self.activation = tf.keras.layers.Activation("relu")
    self.global_avg_pool = tf.keras.layers.GlobalAveragePooling2D(
        data_format=self.config.data_format, dtype=self.config.dtype)
    self.dense = tf.keras.layers.Dense(
        self.config.n_classes, dtype=self.config.dtype)

  def call(self, x, training=True):
    net = x
    net = self.batch_norm(net, training=training)
    net = self.activation(net)
    net = self.global_avg_pool(net)
    net = self.dense(net)

    return net
