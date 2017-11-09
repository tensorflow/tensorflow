# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""ResNet50 model definition compatible with TensorFlow's eager execution.

Reference [Deep Residual Learning for Image
Recognition](https://arxiv.org/abs/1512.03385)

Adapted from tf.keras.applications.ResNet50. A notable difference is that the
model here outputs logits while the Keras model outputs probability.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import tensorflow as tf
import tensorflow.contrib.eager as tfe


class _IdentityBlock(tfe.Network):
  """_IdentityBlock is the block that has no conv layer at shortcut.

  Args:
    kernel_size: the kernel size of middle conv layer at main path
    filters: list of integers, the filters of 3 conv layer at main path
    stage: integer, current stage label, used for generating layer names
    block: 'a','b'..., current block label, used for generating layer names
    data_format: data_format for the input ('channels_first' or
      'channels_last').
  """

  def __init__(self, kernel_size, filters, stage, block, data_format):
    super(_IdentityBlock, self).__init__(name='')
    filters1, filters2, filters3 = filters

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    bn_axis = 1 if data_format == 'channels_first' else 3

    self.conv2a = self.track_layer(
        tf.layers.Conv2D(
            filters1, (1, 1),
            name=conv_name_base + '2a',
            data_format=data_format))
    self.bn2a = self.track_layer(
        tf.layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a'))

    self.conv2b = self.track_layer(
        tf.layers.Conv2D(
            filters2,
            kernel_size,
            padding='same',
            data_format=data_format,
            name=conv_name_base + '2b'))
    self.bn2b = self.track_layer(
        tf.layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b'))

    self.conv2c = self.track_layer(
        tf.layers.Conv2D(
            filters3, (1, 1),
            name=conv_name_base + '2c',
            data_format=data_format))
    self.bn2c = self.track_layer(
        tf.layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c'))

  def call(self, input_tensor, training=False):
    x = self.conv2a(input_tensor)
    x = self.bn2a(x, training=training)
    x = tf.nn.relu(x)

    x = self.conv2b(x)
    x = self.bn2b(x, training=training)
    x = tf.nn.relu(x)

    x = self.conv2c(x)
    x = self.bn2c(x, training=training)

    x += input_tensor
    return tf.nn.relu(x)


class _ConvBlock(tfe.Network):
  """_ConvBlock is the block that has a conv layer at shortcut.

  Args:
      kernel_size: the kernel size of middle conv layer at main path
      filters: list of integers, the filterss of 3 conv layer at main path
      stage: integer, current stage label, used for generating layer names
      block: 'a','b'..., current block label, used for generating layer names
      data_format: data_format for the input ('channels_first' or
        'channels_last').
      strides: strides for the convolution. Note that from stage 3, the first
       conv layer at main path is with strides=(2,2), and the shortcut should
       have strides=(2,2) as well.
  """

  def __init__(self,
               kernel_size,
               filters,
               stage,
               block,
               data_format,
               strides=(2, 2)):
    super(_ConvBlock, self).__init__(name='')
    filters1, filters2, filters3 = filters

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    bn_axis = 1 if data_format == 'channels_first' else 3

    self.conv2a = self.track_layer(
        tf.layers.Conv2D(
            filters1, (1, 1),
            strides=strides,
            name=conv_name_base + '2a',
            data_format=data_format))
    self.bn2a = self.track_layer(
        tf.layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a'))

    self.conv2b = self.track_layer(
        tf.layers.Conv2D(
            filters2,
            kernel_size,
            padding='same',
            name=conv_name_base + '2b',
            data_format=data_format))
    self.bn2b = self.track_layer(
        tf.layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b'))

    self.conv2c = self.track_layer(
        tf.layers.Conv2D(
            filters3, (1, 1),
            name=conv_name_base + '2c',
            data_format=data_format))
    self.bn2c = self.track_layer(
        tf.layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c'))

    self.conv_shortcut = self.track_layer(
        tf.layers.Conv2D(
            filters3, (1, 1),
            strides=strides,
            name=conv_name_base + '1',
            data_format=data_format))
    self.bn_shortcut = self.track_layer(
        tf.layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '1'))

  def call(self, input_tensor, training=False):
    x = self.conv2a(input_tensor)
    x = self.bn2a(x, training=training)
    x = tf.nn.relu(x)

    x = self.conv2b(x)
    x = self.bn2b(x, training=training)
    x = tf.nn.relu(x)

    x = self.conv2c(x)
    x = self.bn2c(x, training=training)

    shortcut = self.conv_shortcut(input_tensor)
    shortcut = self.bn_shortcut(shortcut, training=training)

    x += shortcut
    return tf.nn.relu(x)


class ResNet50(tfe.Network):
  """Instantiates the ResNet50 architecture.

  Args:
    data_format: format for the image. Either 'channels_first' or
      'channels_last'.  'channels_first' is typically faster on GPUs while
      'channels_last' is typically faster on CPUs. See
      https://www.tensorflow.org/performance/performance_guide#data_formats
    name: Prefix applied to names of variables created in the model.
    trainable: Is the model trainable? If true, performs backward
        and optimization after call() method.
    include_top: whether to include the fully-connected layer at the top of the
      network.
    pooling: Optional pooling mode for feature extraction when `include_top`
      is `False`.
      - `None` means that the output of the model will be the 4D tensor
          output of the last convolutional layer.
      - `avg` means that global average pooling will be applied to the output of
          the last convolutional layer, and thus the output of the model will be
          a 2D tensor.
      - `max` means that global max pooling will be applied.
    classes: optional number of classes to classify images into, only to be
      specified if `include_top` is True.

  Raises:
      ValueError: in case of invalid argument for data_format.
  """

  def __init__(self,
               data_format,
               name=None,
               trainable=True,
               include_top=True,
               pooling=None,
               classes=1000):
    super(ResNet50, self).__init__(name='')

    valid_channel_values = ('channels_first', 'channels_last')
    if data_format not in valid_channel_values:
      raise ValueError('Unknown data_format: %s. Valid values: %s' %
                       (data_format, valid_channel_values))
    self.include_top = include_top

    def conv_block(filters, stage, block, strides=(2, 2)):
      l = _ConvBlock(
          3,
          filters,
          stage=stage,
          block=block,
          data_format=data_format,
          strides=strides)
      return self.track_layer(l)

    def id_block(filters, stage, block):
      l = _IdentityBlock(
          3, filters, stage=stage, block=block, data_format=data_format)
      return self.track_layer(l)

    self.conv1 = self.track_layer(
        tf.layers.Conv2D(
            64, (7, 7),
            strides=(2, 2),
            data_format=data_format,
            padding='same',
            name='conv1'))
    bn_axis = 1 if data_format == 'channels_first' else 3
    self.bn_conv1 = self.track_layer(
        tf.layers.BatchNormalization(axis=bn_axis, name='bn_conv1'))
    self.max_pool = self.track_layer(
        tf.layers.MaxPooling2D((3, 3), strides=(2, 2), data_format=data_format))

    self.l2a = conv_block([64, 64, 256], stage=2, block='a', strides=(1, 1))
    self.l2b = id_block([64, 64, 256], stage=2, block='b')
    self.l2c = id_block([64, 64, 256], stage=2, block='c')

    self.l3a = conv_block([128, 128, 512], stage=3, block='a')
    self.l3b = id_block([128, 128, 512], stage=3, block='b')
    self.l3c = id_block([128, 128, 512], stage=3, block='c')
    self.l3d = id_block([128, 128, 512], stage=3, block='d')

    self.l4a = conv_block([256, 256, 1024], stage=4, block='a')
    self.l4b = id_block([256, 256, 1024], stage=4, block='b')
    self.l4c = id_block([256, 256, 1024], stage=4, block='c')
    self.l4d = id_block([256, 256, 1024], stage=4, block='d')
    self.l4e = id_block([256, 256, 1024], stage=4, block='e')
    self.l4f = id_block([256, 256, 1024], stage=4, block='f')

    self.l5a = conv_block([512, 512, 2048], stage=5, block='a')
    self.l5b = id_block([512, 512, 2048], stage=5, block='b')
    self.l5c = id_block([512, 512, 2048], stage=5, block='c')

    self.avg_pool = self.track_layer(
        tf.layers.AveragePooling2D(
            (7, 7), strides=(7, 7), data_format=data_format))

    if self.include_top:
      self.fc1000 = self.track_layer(
          tf.layers.Dense(classes, name='fc1000'))
    else:
      reduction_indices = [1, 2] if data_format == 'channels_last' else [2, 3]
      reduction_indices = tf.constant(reduction_indices)
      if pooling == 'avg':
        self.global_pooling = functools.partial(
            tf.reduce_mean,
            reduction_indices=reduction_indices,
            keep_dims=False)
      elif pooling == 'max':
        self.global_pooling = functools.partial(
            tf.reduce_max, reduction_indices=reduction_indices, keep_dims=False)
      else:
        self.global_pooling = None

  def call(self, input_tensor, training=False):
    x = self.conv1(input_tensor)
    x = self.bn_conv1(x, training=training)
    x = tf.nn.relu(x)
    x = self.max_pool(x)

    x = self.l2a(x, training=training)
    x = self.l2b(x, training=training)
    x = self.l2c(x, training=training)

    x = self.l3a(x, training=training)
    x = self.l3b(x, training=training)
    x = self.l3c(x, training=training)
    x = self.l3d(x, training=training)

    x = self.l4a(x, training=training)
    x = self.l4b(x, training=training)
    x = self.l4c(x, training=training)
    x = self.l4d(x, training=training)
    x = self.l4e(x, training=training)
    x = self.l4f(x, training=training)

    x = self.l5a(x, training=training)
    x = self.l5b(x, training=training)
    x = self.l5c(x, training=training)

    x = self.avg_pool(x)

    if self.include_top:
      return self.fc1000(tf.layers.flatten(x))
    elif self.global_pooling:
      return self.global_pooling(x)
    else:
      return x
