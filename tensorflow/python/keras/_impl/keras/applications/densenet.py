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
# pylint: disable=invalid-name
# pylint: disable=unused-import
"""DenseNet models for Keras.

# Reference paper

- [Densely Connected Convolutional Networks]
  (https://arxiv.org/abs/1608.06993) (CVPR 2017 Best Paper Award)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensorflow.python.keras._impl.keras import backend as K
from tensorflow.python.keras._impl.keras.applications import imagenet_utils
from tensorflow.python.keras._impl.keras.applications.imagenet_utils import _obtain_input_shape
from tensorflow.python.keras._impl.keras.applications.imagenet_utils import decode_predictions
from tensorflow.python.keras._impl.keras.engine.network import get_source_inputs
from tensorflow.python.keras._impl.keras.layers import Activation
from tensorflow.python.keras._impl.keras.layers import AveragePooling2D
from tensorflow.python.keras._impl.keras.layers import BatchNormalization
from tensorflow.python.keras._impl.keras.layers import Concatenate
from tensorflow.python.keras._impl.keras.layers import Conv2D
from tensorflow.python.keras._impl.keras.layers import Dense
from tensorflow.python.keras._impl.keras.layers import GlobalAveragePooling2D
from tensorflow.python.keras._impl.keras.layers import GlobalMaxPooling2D
from tensorflow.python.keras._impl.keras.layers import Input
from tensorflow.python.keras._impl.keras.layers import MaxPooling2D
from tensorflow.python.keras._impl.keras.layers import ZeroPadding2D
from tensorflow.python.keras._impl.keras.models import Model
from tensorflow.python.keras._impl.keras.utils.data_utils import get_file
from tensorflow.python.util.tf_export import tf_export


DENSENET121_WEIGHT_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.8/densenet121_weights_tf_dim_ordering_tf_kernels.h5'
DENSENET121_WEIGHT_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.8/densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5'
DENSENET169_WEIGHT_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.8/densenet169_weights_tf_dim_ordering_tf_kernels.h5'
DENSENET169_WEIGHT_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.8/densenet169_weights_tf_dim_ordering_tf_kernels_notop.h5'
DENSENET201_WEIGHT_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.8/densenet201_weights_tf_dim_ordering_tf_kernels.h5'
DENSENET201_WEIGHT_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.8/densenet201_weights_tf_dim_ordering_tf_kernels_notop.h5'


def dense_block(x, blocks, name):
  """A dense block.

  Arguments:
      x: input tensor.
      blocks: integer, the number of building blocks.
      name: string, block label.

  Returns:
      output tensor for the block.
  """
  for i in range(blocks):
    x = conv_block(x, 32, name=name + '_block' + str(i + 1))
  return x


def transition_block(x, reduction, name):
  """A transition block.

  Arguments:
      x: input tensor.
      reduction: float, compression rate at transition layers.
      name: string, block label.

  Returns:
      output tensor for the block.
  """
  bn_axis = 3 if K.image_data_format() == 'channels_last' else 1
  x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_bn')(x)
  x = Activation('relu', name=name + '_relu')(x)
  x = Conv2D(
      int(K.int_shape(x)[bn_axis] * reduction),
      1,
      use_bias=False,
      name=name + '_conv')(
          x)
  x = AveragePooling2D(2, strides=2, name=name + '_pool')(x)
  return x


def conv_block(x, growth_rate, name):
  """A building block for a dense block.

  Arguments:
      x: input tensor.
      growth_rate: float, growth rate at dense layers.
      name: string, block label.

  Returns:
      output tensor for the block.
  """
  bn_axis = 3 if K.image_data_format() == 'channels_last' else 1
  x1 = BatchNormalization(
      axis=bn_axis, epsilon=1.001e-5, name=name + '_0_bn')(
          x)
  x1 = Activation('relu', name=name + '_0_relu')(x1)
  x1 = Conv2D(4 * growth_rate, 1, use_bias=False, name=name + '_1_conv')(x1)
  x1 = BatchNormalization(
      axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(
          x1)
  x1 = Activation('relu', name=name + '_1_relu')(x1)
  x1 = Conv2D(
      growth_rate, 3, padding='same', use_bias=False, name=name + '_2_conv')(
          x1)
  x = Concatenate(axis=bn_axis, name=name + '_concat')([x, x1])
  return x


def DenseNet(blocks,
             include_top=True,
             weights='imagenet',
             input_tensor=None,
             input_shape=None,
             pooling=None,
             classes=1000):
  """Instantiates the DenseNet architecture.

  Optionally loads weights pre-trained
  on ImageNet. Note that when using TensorFlow,
  for best performance you should set
  `image_data_format='channels_last'` in your Keras config
  at ~/.keras/keras.json.

  The model and the weights are compatible with
  TensorFlow, Theano, and CNTK. The data format
  convention used by the model is the one
  specified in your Keras config file.

  Arguments:
      blocks: numbers of building blocks for the four dense layers.
      include_top: whether to include the fully-connected
          layer at the top of the network.
      weights: one of `None` (random initialization),
            'imagenet' (pre-training on ImageNet),
            or the path to the weights file to be loaded.
      input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
          to use as image input for the model.
      input_shape: optional shape tuple, only to be specified
          if `include_top` is False (otherwise the input shape
          has to be `(224, 224, 3)` (with `channels_last` data format)
          or `(3, 224, 224)` (with `channels_first` data format).
          It should have exactly 3 inputs channels.
      pooling: optional pooling mode for feature extraction
          when `include_top` is `False`.
          - `None` means that the output of the model will be
              the 4D tensor output of the
              last convolutional layer.
          - `avg` means that global average pooling
              will be applied to the output of the
              last convolutional layer, and thus
              the output of the model will be a 2D tensor.
          - `max` means that global max pooling will
              be applied.
      classes: optional number of classes to classify images
          into, only to be specified if `include_top` is True, and
          if no `weights` argument is specified.

  Returns:
      A Keras model instance.

  Raises:
      ValueError: in case of invalid argument for `weights`,
          or invalid input shape.
  """
  if not (weights in {'imagenet', None} or os.path.exists(weights)):
    raise ValueError('The `weights` argument should be either '
                     '`None` (random initialization), `imagenet` '
                     '(pre-training on ImageNet), '
                     'or the path to the weights file to be loaded.')

  if weights == 'imagenet' and include_top and classes != 1000:
    raise ValueError('If using `weights` as imagenet with `include_top`'
                     ' as true, `classes` should be 1000')

  # Determine proper input shape
  input_shape = _obtain_input_shape(
      input_shape,
      default_size=224,
      min_size=221,
      data_format=K.image_data_format(),
      require_flatten=include_top,
      weights=weights)

  if input_tensor is None:
    img_input = Input(shape=input_shape)
  else:
    if not K.is_keras_tensor(input_tensor):
      img_input = Input(tensor=input_tensor, shape=input_shape)
    else:
      img_input = input_tensor

  bn_axis = 3 if K.image_data_format() == 'channels_last' else 1

  x = ZeroPadding2D(padding=((3, 3), (3, 3)))(img_input)
  x = Conv2D(64, 7, strides=2, use_bias=False, name='conv1/conv')(x)
  x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name='conv1/bn')(x)
  x = Activation('relu', name='conv1/relu')(x)
  x = ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
  x = MaxPooling2D(3, strides=2, name='pool1')(x)

  x = dense_block(x, blocks[0], name='conv2')
  x = transition_block(x, 0.5, name='pool2')
  x = dense_block(x, blocks[1], name='conv3')
  x = transition_block(x, 0.5, name='pool3')
  x = dense_block(x, blocks[2], name='conv4')
  x = transition_block(x, 0.5, name='pool4')
  x = dense_block(x, blocks[3], name='conv5')

  x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name='bn')(x)

  if include_top:
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dense(classes, activation='softmax', name='fc1000')(x)
  else:
    if pooling == 'avg':
      x = GlobalAveragePooling2D(name='avg_pool')(x)
    elif pooling == 'max':
      x = GlobalMaxPooling2D(name='max_pool')(x)

  # Ensure that the model takes into account
  # any potential predecessors of `input_tensor`.
  if input_tensor is not None:
    inputs = get_source_inputs(input_tensor)
  else:
    inputs = img_input

  # Create model.
  if blocks == [6, 12, 24, 16]:
    model = Model(inputs, x, name='densenet121')
  elif blocks == [6, 12, 32, 32]:
    model = Model(inputs, x, name='densenet169')
  elif blocks == [6, 12, 48, 32]:
    model = Model(inputs, x, name='densenet201')
  else:
    model = Model(inputs, x, name='densenet')

  # Load weights.
  if weights == 'imagenet':
    if include_top:
      if blocks == [6, 12, 24, 16]:
        weights_path = get_file(
            'densenet121_weights_tf_dim_ordering_tf_kernels.h5',
            DENSENET121_WEIGHT_PATH,
            cache_subdir='models',
            file_hash='0962ca643bae20f9b6771cb844dca3b0')
      elif blocks == [6, 12, 32, 32]:
        weights_path = get_file(
            'densenet169_weights_tf_dim_ordering_tf_kernels.h5',
            DENSENET169_WEIGHT_PATH,
            cache_subdir='models',
            file_hash='bcf9965cf5064a5f9eb6d7dc69386f43')
      elif blocks == [6, 12, 48, 32]:
        weights_path = get_file(
            'densenet201_weights_tf_dim_ordering_tf_kernels.h5',
            DENSENET201_WEIGHT_PATH,
            cache_subdir='models',
            file_hash='7bb75edd58cb43163be7e0005fbe95ef')
    else:
      if blocks == [6, 12, 24, 16]:
        weights_path = get_file(
            'densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5',
            DENSENET121_WEIGHT_PATH_NO_TOP,
            cache_subdir='models',
            file_hash='4912a53fbd2a69346e7f2c0b5ec8c6d3')
      elif blocks == [6, 12, 32, 32]:
        weights_path = get_file(
            'densenet169_weights_tf_dim_ordering_tf_kernels_notop.h5',
            DENSENET169_WEIGHT_PATH_NO_TOP,
            cache_subdir='models',
            file_hash='50662582284e4cf834ce40ab4dfa58c6')
      elif blocks == [6, 12, 48, 32]:
        weights_path = get_file(
            'densenet201_weights_tf_dim_ordering_tf_kernels_notop.h5',
            DENSENET201_WEIGHT_PATH_NO_TOP,
            cache_subdir='models',
            file_hash='1c2de60ee40562448dbac34a0737e798')
    model.load_weights(weights_path)
  elif weights is not None:
    model.load_weights(weights)

  return model


@tf_export('keras.applications.DenseNet121',
           'keras.applications.densenet.DenseNet121')
def DenseNet121(include_top=True,
                weights='imagenet',
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=1000):
  return DenseNet([6, 12, 24, 16], include_top, weights, input_tensor,
                  input_shape, pooling, classes)


@tf_export('keras.applications.DenseNet169',
           'keras.applications.densenet.DenseNet169')
def DenseNet169(include_top=True,
                weights='imagenet',
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=1000):
  return DenseNet([6, 12, 32, 32], include_top, weights, input_tensor,
                  input_shape, pooling, classes)


@tf_export('keras.applications.DenseNet201',
           'keras.applications.densenet.DenseNet201')
def DenseNet201(include_top=True,
                weights='imagenet',
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=1000):
  return DenseNet([6, 12, 48, 32], include_top, weights, input_tensor,
                  input_shape, pooling, classes)


@tf_export('keras.applications.densenet.preprocess_input')
def preprocess_input(x, data_format=None):
  """Preprocesses a numpy array encoding a batch of images.

  Arguments:
      x: a 3D or 4D numpy array consists of RGB values within [0, 255].
      data_format: data format of the image tensor.

  Returns:
      Preprocessed array.
  """
  return imagenet_utils.preprocess_input(x, data_format, mode='torch')


setattr(DenseNet121, '__doc__', DenseNet.__doc__)
setattr(DenseNet169, '__doc__', DenseNet.__doc__)
setattr(DenseNet201, '__doc__', DenseNet.__doc__)
