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
# pylint: disable=invalid-name
# pylint: disable=unused-import
"""VGG19 model for Keras.

# Reference

- [Very Deep Convolutional Networks for Large-Scale Image
Recognition](https://arxiv.org/abs/1409.1556)

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensorflow.python.keras._impl.keras import backend as K
from tensorflow.python.keras._impl.keras.applications.imagenet_utils import _obtain_input_shape
from tensorflow.python.keras._impl.keras.applications.imagenet_utils import decode_predictions
from tensorflow.python.keras._impl.keras.applications.imagenet_utils import preprocess_input
from tensorflow.python.keras._impl.keras.engine.network import get_source_inputs
from tensorflow.python.keras._impl.keras.layers import Conv2D
from tensorflow.python.keras._impl.keras.layers import Dense
from tensorflow.python.keras._impl.keras.layers import Flatten
from tensorflow.python.keras._impl.keras.layers import GlobalAveragePooling2D
from tensorflow.python.keras._impl.keras.layers import GlobalMaxPooling2D
from tensorflow.python.keras._impl.keras.layers import Input
from tensorflow.python.keras._impl.keras.layers import MaxPooling2D
from tensorflow.python.keras._impl.keras.models import Model
from tensorflow.python.keras._impl.keras.utils import layer_utils
from tensorflow.python.keras._impl.keras.utils.data_utils import get_file
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import tf_export


WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels.h5'
WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5'


@tf_export('keras.applications.VGG19', 'keras.applications.vgg19.VGG19')
def VGG19(include_top=True,
          weights='imagenet',
          input_tensor=None,
          input_shape=None,
          pooling=None,
          classes=1000):
  """Instantiates the VGG19 architecture.

  Optionally loads weights pre-trained
  on ImageNet. Note that when using TensorFlow,
  for best performance you should set
  `image_data_format='channels_last'` in your Keras config
  at ~/.keras/keras.json.

  The model and the weights are compatible with both
  TensorFlow and Theano. The data format
  convention used by the model is the one
  specified in your Keras config file.

  Arguments:
      include_top: whether to include the 3 fully-connected
          layers at the top of the network.
      weights: one of `None` (random initialization),
            'imagenet' (pre-training on ImageNet),
            or the path to the weights file to be loaded.
      input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
          to use as image input for the model.
      input_shape: optional shape tuple, only to be specified
          if `include_top` is False (otherwise the input shape
          has to be `(224, 224, 3)` (with `channels_last` data format)
          or `(3, 224, 224)` (with `channels_first` data format).
          It should have exactly 3 inputs channels,
          and width and height should be no smaller than 48.
          E.g. `(200, 200, 3)` would be one valid value.
      pooling: Optional pooling mode for feature extraction
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
      min_size=48,
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
  # Block 1
  x = Conv2D(
      64, (3, 3), activation='relu', padding='same', name='block1_conv1')(
          img_input)
  x = Conv2D(
      64, (3, 3), activation='relu', padding='same', name='block1_conv2')(
          x)
  x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

  # Block 2
  x = Conv2D(
      128, (3, 3), activation='relu', padding='same', name='block2_conv1')(
          x)
  x = Conv2D(
      128, (3, 3), activation='relu', padding='same', name='block2_conv2')(
          x)
  x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

  # Block 3
  x = Conv2D(
      256, (3, 3), activation='relu', padding='same', name='block3_conv1')(
          x)
  x = Conv2D(
      256, (3, 3), activation='relu', padding='same', name='block3_conv2')(
          x)
  x = Conv2D(
      256, (3, 3), activation='relu', padding='same', name='block3_conv3')(
          x)
  x = Conv2D(
      256, (3, 3), activation='relu', padding='same', name='block3_conv4')(
          x)
  x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

  # Block 4
  x = Conv2D(
      512, (3, 3), activation='relu', padding='same', name='block4_conv1')(
          x)
  x = Conv2D(
      512, (3, 3), activation='relu', padding='same', name='block4_conv2')(
          x)
  x = Conv2D(
      512, (3, 3), activation='relu', padding='same', name='block4_conv3')(
          x)
  x = Conv2D(
      512, (3, 3), activation='relu', padding='same', name='block4_conv4')(
          x)
  x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

  # Block 5
  x = Conv2D(
      512, (3, 3), activation='relu', padding='same', name='block5_conv1')(
          x)
  x = Conv2D(
      512, (3, 3), activation='relu', padding='same', name='block5_conv2')(
          x)
  x = Conv2D(
      512, (3, 3), activation='relu', padding='same', name='block5_conv3')(
          x)
  x = Conv2D(
      512, (3, 3), activation='relu', padding='same', name='block5_conv4')(
          x)
  x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

  if include_top:
    # Classification block
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dense(classes, activation='softmax', name='predictions')(x)
  else:
    if pooling == 'avg':
      x = GlobalAveragePooling2D()(x)
    elif pooling == 'max':
      x = GlobalMaxPooling2D()(x)

  # Ensure that the model takes into account
  # any potential predecessors of `input_tensor`.
  if input_tensor is not None:
    inputs = get_source_inputs(input_tensor)
  else:
    inputs = img_input
  # Create model.
  model = Model(inputs, x, name='vgg19')

  # load weights
  if weights == 'imagenet':
    if include_top:
      weights_path = get_file(
          'vgg19_weights_tf_dim_ordering_tf_kernels.h5',
          WEIGHTS_PATH,
          cache_subdir='models',
          file_hash='cbe5617147190e668d6c5d5026f83318')
    else:
      weights_path = get_file(
          'vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5',
          WEIGHTS_PATH_NO_TOP,
          cache_subdir='models',
          file_hash='253f8cb515780f3b799900260a226db6')
    model.load_weights(weights_path)
    if K.backend() == 'theano':
      layer_utils.convert_all_kernels_in_model(model)

    if K.image_data_format() == 'channels_first':
      if include_top:
        maxpool = model.get_layer(name='block5_pool')
        shape = maxpool.output_shape[1:]
        dense = model.get_layer(name='fc1')
        layer_utils.convert_dense_weights_data_format(dense, shape,
                                                      'channels_first')

  elif weights is not None:
    model.load_weights(weights)

  return model
