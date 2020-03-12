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
"""DenseNet models for Keras.

Reference paper:
  - [Densely Connected Convolutional Networks]
    (https://arxiv.org/abs/1608.06993) (CVPR 2017 Best Paper Award)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensorflow.python.keras import backend
from tensorflow.python.keras import layers
from tensorflow.python.keras.applications import imagenet_utils
from tensorflow.python.keras.engine import training
from tensorflow.python.keras.utils import data_utils
from tensorflow.python.keras.utils import layer_utils
from tensorflow.python.util.tf_export import keras_export


BASE_WEIGTHS_PATH = ('https://storage.googleapis.com/tensorflow/'
                     'keras-applications/densenet/')
DENSENET121_WEIGHT_PATH = (
    BASE_WEIGTHS_PATH + 'densenet121_weights_tf_dim_ordering_tf_kernels.h5')
DENSENET121_WEIGHT_PATH_NO_TOP = (
    BASE_WEIGTHS_PATH +
    'densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5')
DENSENET169_WEIGHT_PATH = (
    BASE_WEIGTHS_PATH + 'densenet169_weights_tf_dim_ordering_tf_kernels.h5')
DENSENET169_WEIGHT_PATH_NO_TOP = (
    BASE_WEIGTHS_PATH +
    'densenet169_weights_tf_dim_ordering_tf_kernels_notop.h5')
DENSENET201_WEIGHT_PATH = (
    BASE_WEIGTHS_PATH + 'densenet201_weights_tf_dim_ordering_tf_kernels.h5')
DENSENET201_WEIGHT_PATH_NO_TOP = (
    BASE_WEIGTHS_PATH +
    'densenet201_weights_tf_dim_ordering_tf_kernels_notop.h5')


def dense_block(x, blocks, name):
  """A dense block.

  Arguments:
    x: input tensor.
    blocks: integer, the number of building blocks.
    name: string, block label.

  Returns:
    Output tensor for the block.
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
  bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1
  x = layers.BatchNormalization(
      axis=bn_axis, epsilon=1.001e-5, name=name + '_bn')(
          x)
  x = layers.Activation('relu', name=name + '_relu')(x)
  x = layers.Conv2D(
      int(backend.int_shape(x)[bn_axis] * reduction),
      1,
      use_bias=False,
      name=name + '_conv')(
          x)
  x = layers.AveragePooling2D(2, strides=2, name=name + '_pool')(x)
  return x


def conv_block(x, growth_rate, name):
  """A building block for a dense block.

  Arguments:
    x: input tensor.
    growth_rate: float, growth rate at dense layers.
    name: string, block label.

  Returns:
    Output tensor for the block.
  """
  bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1
  x1 = layers.BatchNormalization(
      axis=bn_axis, epsilon=1.001e-5, name=name + '_0_bn')(
          x)
  x1 = layers.Activation('relu', name=name + '_0_relu')(x1)
  x1 = layers.Conv2D(
      4 * growth_rate, 1, use_bias=False, name=name + '_1_conv')(
          x1)
  x1 = layers.BatchNormalization(
      axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(
          x1)
  x1 = layers.Activation('relu', name=name + '_1_relu')(x1)
  x1 = layers.Conv2D(
      growth_rate, 3, padding='same', use_bias=False, name=name + '_2_conv')(
          x1)
  x = layers.Concatenate(axis=bn_axis, name=name + '_concat')([x, x1])
  return x


def DenseNet(
    blocks,
    include_top=True,
    weights='imagenet',
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation='softmax',
):
  """Instantiates the DenseNet architecture.

  Reference paper:
  - [Densely Connected Convolutional Networks]
    (https://arxiv.org/abs/1608.06993) (CVPR 2017 Best Paper Award)

  Optionally loads weights pre-trained on ImageNet.
  Note that the data format convention used by the model is
  the one specified in your Keras config at `~/.keras/keras.json`.

  Caution: Be sure to properly pre-process your inputs to the application.
  Please see `applications.densenet.preprocess_input` for an example.

  Arguments:
    blocks: numbers of building blocks for the four dense layers.
    include_top: whether to include the fully-connected
      layer at the top of the network.
    weights: one of `None` (random initialization),
      'imagenet' (pre-training on ImageNet),
      or the path to the weights file to be loaded.
    input_tensor: optional Keras tensor
      (i.e. output of `layers.Input()`)
      to use as image input for the model.
    input_shape: optional shape tuple, only to be specified
      if `include_top` is False (otherwise the input shape
      has to be `(224, 224, 3)` (with `'channels_last'` data format)
      or `(3, 224, 224)` (with `'channels_first'` data format).
      It should have exactly 3 inputs channels,
      and width and height should be no smaller than 32.
      E.g. `(200, 200, 3)` would be one valid value.
    pooling: optional pooling mode for feature extraction
      when `include_top` is `False`.
      - `None` means that the output of the model will be
          the 4D tensor output of the
          last convolutional block.
      - `avg` means that global average pooling
          will be applied to the output of the
          last convolutional block, and thus
          the output of the model will be a 2D tensor.
      - `max` means that global max pooling will
          be applied.
    classes: optional number of classes to classify images
      into, only to be specified if `include_top` is True, and
      if no `weights` argument is specified.
    classifier_activation: A `str` or callable. The activation function to use
      on the "top" layer. Ignored unless `include_top=True`. Set
      `classifier_activation=None` to return the logits of the "top" layer.

  Returns:
    A `keras.Model` instance.

  Raises:
    ValueError: in case of invalid argument for `weights`,
      or invalid input shape.
    ValueError: if `classifier_activation` is not `softmax` or `None` when
      using a pretrained top layer.
  """
  if not (weights in {'imagenet', None} or os.path.exists(weights)):
    raise ValueError('The `weights` argument should be either '
                     '`None` (random initialization), `imagenet` '
                     '(pre-training on ImageNet), '
                     'or the path to the weights file to be loaded.')

  if weights == 'imagenet' and include_top and classes != 1000:
    raise ValueError('If using `weights` as `"imagenet"` with `include_top`'
                     ' as true, `classes` should be 1000')

  # Determine proper input shape
  input_shape = imagenet_utils.obtain_input_shape(
      input_shape,
      default_size=224,
      min_size=32,
      data_format=backend.image_data_format(),
      require_flatten=include_top,
      weights=weights)

  if input_tensor is None:
    img_input = layers.Input(shape=input_shape)
  else:
    if not backend.is_keras_tensor(input_tensor):
      img_input = layers.Input(tensor=input_tensor, shape=input_shape)
    else:
      img_input = input_tensor

  bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

  x = layers.ZeroPadding2D(padding=((3, 3), (3, 3)))(img_input)
  x = layers.Conv2D(64, 7, strides=2, use_bias=False, name='conv1/conv')(x)
  x = layers.BatchNormalization(
      axis=bn_axis, epsilon=1.001e-5, name='conv1/bn')(
          x)
  x = layers.Activation('relu', name='conv1/relu')(x)
  x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
  x = layers.MaxPooling2D(3, strides=2, name='pool1')(x)

  x = dense_block(x, blocks[0], name='conv2')
  x = transition_block(x, 0.5, name='pool2')
  x = dense_block(x, blocks[1], name='conv3')
  x = transition_block(x, 0.5, name='pool3')
  x = dense_block(x, blocks[2], name='conv4')
  x = transition_block(x, 0.5, name='pool4')
  x = dense_block(x, blocks[3], name='conv5')

  x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name='bn')(x)
  x = layers.Activation('relu', name='relu')(x)

  if include_top:
    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)

    imagenet_utils.validate_activation(classifier_activation, weights)
    x = layers.Dense(classes, activation=classifier_activation,
                     name='predictions')(x)
  else:
    if pooling == 'avg':
      x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    elif pooling == 'max':
      x = layers.GlobalMaxPooling2D(name='max_pool')(x)

  # Ensure that the model takes into account
  # any potential predecessors of `input_tensor`.
  if input_tensor is not None:
    inputs = layer_utils.get_source_inputs(input_tensor)
  else:
    inputs = img_input

  # Create model.
  if blocks == [6, 12, 24, 16]:
    model = training.Model(inputs, x, name='densenet121')
  elif blocks == [6, 12, 32, 32]:
    model = training.Model(inputs, x, name='densenet169')
  elif blocks == [6, 12, 48, 32]:
    model = training.Model(inputs, x, name='densenet201')
  else:
    model = training.Model(inputs, x, name='densenet')

  # Load weights.
  if weights == 'imagenet':
    if include_top:
      if blocks == [6, 12, 24, 16]:
        weights_path = data_utils.get_file(
            'densenet121_weights_tf_dim_ordering_tf_kernels.h5',
            DENSENET121_WEIGHT_PATH,
            cache_subdir='models',
            file_hash='9d60b8095a5708f2dcce2bca79d332c7')
      elif blocks == [6, 12, 32, 32]:
        weights_path = data_utils.get_file(
            'densenet169_weights_tf_dim_ordering_tf_kernels.h5',
            DENSENET169_WEIGHT_PATH,
            cache_subdir='models',
            file_hash='d699b8f76981ab1b30698df4c175e90b')
      elif blocks == [6, 12, 48, 32]:
        weights_path = data_utils.get_file(
            'densenet201_weights_tf_dim_ordering_tf_kernels.h5',
            DENSENET201_WEIGHT_PATH,
            cache_subdir='models',
            file_hash='1ceb130c1ea1b78c3bf6114dbdfd8807')
    else:
      if blocks == [6, 12, 24, 16]:
        weights_path = data_utils.get_file(
            'densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5',
            DENSENET121_WEIGHT_PATH_NO_TOP,
            cache_subdir='models',
            file_hash='30ee3e1110167f948a6b9946edeeb738')
      elif blocks == [6, 12, 32, 32]:
        weights_path = data_utils.get_file(
            'densenet169_weights_tf_dim_ordering_tf_kernels_notop.h5',
            DENSENET169_WEIGHT_PATH_NO_TOP,
            cache_subdir='models',
            file_hash='b8c4d4c20dd625c148057b9ff1c1176b')
      elif blocks == [6, 12, 48, 32]:
        weights_path = data_utils.get_file(
            'densenet201_weights_tf_dim_ordering_tf_kernels_notop.h5',
            DENSENET201_WEIGHT_PATH_NO_TOP,
            cache_subdir='models',
            file_hash='c13680b51ded0fb44dff2d8f86ac8bb1')
    model.load_weights(weights_path)
  elif weights is not None:
    model.load_weights(weights)

  return model


@keras_export('keras.applications.densenet.DenseNet121',
              'keras.applications.DenseNet121')
def DenseNet121(include_top=True,
                weights='imagenet',
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=1000):
  """Instantiates the Densenet121 architecture."""
  return DenseNet([6, 12, 24, 16], include_top, weights, input_tensor,
                  input_shape, pooling, classes)


@keras_export('keras.applications.densenet.DenseNet169',
              'keras.applications.DenseNet169')
def DenseNet169(include_top=True,
                weights='imagenet',
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=1000):
  """Instantiates the Densenet169 architecture."""
  return DenseNet([6, 12, 32, 32], include_top, weights, input_tensor,
                  input_shape, pooling, classes)


@keras_export('keras.applications.densenet.DenseNet201',
              'keras.applications.DenseNet201')
def DenseNet201(include_top=True,
                weights='imagenet',
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=1000):
  """Instantiates the Densenet201 architecture."""
  return DenseNet([6, 12, 48, 32], include_top, weights, input_tensor,
                  input_shape, pooling, classes)


@keras_export('keras.applications.densenet.preprocess_input')
def preprocess_input(x, data_format=None):
  """Preprocesses a numpy array encoding a batch of images.

  Arguments
    x: A 4D numpy array consists of RGB values within [0, 255].

  Returns
    Preprocessed array.

  Raises
    ValueError: In case of unknown `data_format` argument.
  """
  return imagenet_utils.preprocess_input(
      x, data_format=data_format, mode='torch')


@keras_export('keras.applications.densenet.decode_predictions')
def decode_predictions(preds, top=5):
  """Decodes the prediction result from the model.

  Arguments
    preds: Numpy tensor encoding a batch of predictions.
    top: Integer, how many top-guesses to return.

  Returns
    A list of lists of top class prediction tuples
    `(class_name, class_description, score)`.
    One list of tuples per sample in batch input.

  Raises
    ValueError: In case of invalid shape of the `preds` array (must be 2D).
  """
  return imagenet_utils.decode_predictions(preds, top=top)


preprocess_input.__doc__ = imagenet_utils.PREPROCESS_INPUT_DOC.format(
    mode='', ret=imagenet_utils.PREPROCESS_INPUT_RET_DOC_TORCH)
decode_predictions.__doc__ = imagenet_utils.decode_predictions.__doc__

DOC = """

  Reference paper:
  - [Densely Connected Convolutional Networks]
    (https://arxiv.org/abs/1608.06993) (CVPR 2017 Best Paper Award)

  Optionally loads weights pre-trained on ImageNet.
  Note that the data format convention used by the model is
  the one specified in your Keras config at `~/.keras/keras.json`.
  
  Arguments:
    include_top: whether to include the fully-connected
      layer at the top of the network.
    weights: one of `None` (random initialization),
      'imagenet' (pre-training on ImageNet),
      or the path to the weights file to be loaded.
    input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
      to use as image input for the model.
    input_shape: optional shape tuple, only to be specified
      if `include_top` is False (otherwise the input shape
      has to be `(224, 224, 3)` (with `'channels_last'` data format)
      or `(3, 224, 224)` (with `'channels_first'` data format).
      It should have exactly 3 inputs channels,
      and width and height should be no smaller than 32.
      E.g. `(200, 200, 3)` would be one valid value.
    pooling: Optional pooling mode for feature extraction
      when `include_top` is `False`.
      - `None` means that the output of the model will be
          the 4D tensor output of the
          last convolutional block.
      - `avg` means that global average pooling
          will be applied to the output of the
          last convolutional block, and thus
          the output of the model will be a 2D tensor.
      - `max` means that global max pooling will
          be applied.
    classes: optional number of classes to classify images
      into, only to be specified if `include_top` is True, and
      if no `weights` argument is specified.

  Returns:
    A Keras model instance.
"""

setattr(DenseNet121, '__doc__', DenseNet121.__doc__ + DOC)
setattr(DenseNet169, '__doc__', DenseNet169.__doc__ + DOC)
setattr(DenseNet201, '__doc__', DenseNet201.__doc__ + DOC)
