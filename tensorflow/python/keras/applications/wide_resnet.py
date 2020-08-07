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
# pylint: disable=missing-docstring
"""WideResNet models for Keras.

Reference:
    - [Wide Residual Networks](https://arxiv.org/abs/1605.07146) (BMVC 2016)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.keras import backend
from tensorflow.python.keras.applications import imagenet_utils
from tensorflow.python.keras.engine import training
from tensorflow.python.keras.layers import VersionAwareLayers
from tensorflow.python.keras import initializers
from tensorflow.python.keras.utils import data_utils  # pylint: disable=unused-import
from tensorflow.python.keras.utils import layer_utils
from tensorflow.python.lib.io import file_io
from tensorflow.python.util.tf_export import keras_export

from functools import partial

layers = VersionAwareLayers()

BASE_DOCSTRING = """Instantiates the {name} architecture.
  Reference:
  - [Wide Residual Networks](https://arxiv.org/abs/1605.07146) (BMVC 2016)

  Optionally loads weights pre-trained on ImageNet.
  Note that the data format convention used by the model is
  the one specified in your Keras config at `~/.keras/keras.json`.
  If you have never configured it, it defaults to `"channels_last"`.

  Arguments:
    include_top: Boolean, whether to include the fully-connected
      layer at the top, as the last layer of the network. Default to `True`.
    weights: One of `None` (random initialization),
      `imagenet` (pre-training on ImageNet),
      or the path to the weights file to be loaded. Default to `imagenet`.
    input_tensor: Optional Keras tensor (i.e. output of `layers.Input()`)
      to use as image input for the model. `input_tensor` is useful for sharing
      inputs between multiple different networks. Default to None.
    input_shape: Optional shape tuple, only to be specified
      if `include_top` is False (otherwise the input shape
      has to be `(32, 32, 3)` (with `channels_last` data format)
      or `(3, 32, 32)` (with `channels_first` data format).
      It should have exactly 3 inputs channels,
      and width and height should be no smaller than 32.
      `input_shape` will be ignored if the `input_tensor` is provided.
    pooling: Optional pooling mode for feature extraction
      when `include_top` is `False`.
      - `None` (default) means that the output of the model will be
          the 4D tensor output of the last convolutional block.
      - `avg` means that global average pooling
          will be applied to the output of the
          last convolutional block, and thus
          the output of the model will be a 2D tensor.
      - `max` means that global max pooling will be applied.
    classes: optional number of classes to classify images
      into, only to be specified if `include_top` is True, and
      if no `weights` argument is specified. Defaults to 1000.
    classifier_activation: A `str` or callable. The activation function to use
      on the "top" layer. Ignored unless `include_top=True`. Set
      `classifier_activation=None` to return the logits of the "top" layer.

  Returns:
    A `keras.Model` instance.

  Raises:
    NotImplementedError: in case of argument for `weights` is
      `imagenet`. TODO: add pre-trained weights.
    ValueError: in case of invalid argument for `weights`,
      or invalid input shape.
    ValueError: if `classifier_activation` is not `softmax` or `None` when
      using a pretrained top layer.
"""


def WideResNet(
    depth, k,
    dropout_rate=0.0,
    include_top=True,
    weights='imagenet',
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation='softmax'):
  """Instantiates the WideResNet architecture using given depth and widening factor.

  Reference:
  - [Wide Residual Networks](https://arxiv.org/abs/1605.07146) (BMVC 2016)

  Optionally loads weights pre-trained on ImageNet.
  Note that the data format convention used by the model is
  the one specified in the `tf.keras.backend.image_data_format()`.

  Arguments:
    depth: Integer, depth of the residual network. It should exactly be
      `6n+4`, where `n` is a valid integer.
    k: Integer, widening factor of the residual network. It should be greater
      than 1. E.g. `5`, `10` or `20` would be valid values.
    dropout_rate: Float, rate of dropout to apply in between conv layers
      of each residual block.
    include_top: Boolean, whether to include the fully-connected
      layer at the top, as the last layer of the network. Default to `True`.
    weights: One of `None` (random initialization),
      `imagenet` (pre-training on ImageNet),
      or the path to the weights file to be loaded. Default to `imagenet`.
    input_tensor: Optional Keras tensor (i.e. output of `layers.Input()`)
      to use as image input for the model. `input_tensor` is useful for sharing
      inputs between multiple different networks. Default to None.
    input_shape: Optional shape tuple, only to be specified
      if `include_top` is False (otherwise the input shape
      has to be `(32, 32, 3)` (with `channels_last` data format)
      or `(3, 32, 32)` (with `channels_first` data format).
      It should have exactly 3 inputs channels,
      and width and height should be no smaller than 32.
      `input_shape` will be ignored if the `input_tensor` is provided.
    pooling: Optional pooling mode for feature extraction
      when `include_top` is `False`.
      - `None` (default) means that the output of the model will be
          the 4D tensor output of the last convolutional block.
      - `avg` means that global average pooling
          will be applied to the output of the
          last convolutional block, and thus
          the output of the model will be a 2D tensor.
      - `max` means that global max pooling will be applied.
    classes: optional number of classes to classify images
      into, only to be specified if `include_top` is True, and
      if no `weights` argument is specified. Defaults to 1000.
    classifier_activation: A `str` or callable. The activation function to use
      on the "top" layer. Ignored unless `include_top=True`. Set
      `classifier_activation=None` to return the logits of the "top" layer.

  Returns:
    A `keras.Model` instance.

  Raises:
    NotImplementedError: in case of argument for `weights` is
      `imagenet`. TODO: add pre-trained weights.
    ValueError: in case of invalid argument for `weights`,
      or invalid input shape.
    ValueError: if `classifier_activation` is not `softmax` or `None` when
      using a pretrained top layer.
  """
  if weights == 'imagenet':
    raise NotImplementedError('This network does not yet support '
                              'pre-trained weights. Consider using '
                              'the `weights` argument with `None` instead')


  if not (weights in {'imagenet', None} or file_io.file_exists(weights)):
    raise ValueError('The `weights` argument should be either '
                     '`None` (random initialization), `imagenet` '
                     '(pre-training on ImageNet), '
                     'or the path to the weights file to be loaded.')

  if weights == 'imagenet' and include_top and classes != 1000:
    raise ValueError('If using `weights` as `"imagenet"` with `include_top`'
                     ' as true, `classes` should be 1000')

  if (depth - 4) % 6 != 0:
    raise ValueError('The `depth` argument must exactly be '
                     '`6n+4`, where `n` is any valid integer')

  input_shape = imagenet_utils.obtain_input_shape(
      input_shape,
      default_size=32,
      min_size=32,
      data_format=backend.image_data_format(),
      require_flatten=include_top,
      weights=weights)

  if input_tensor is None:
    img_input = layers.Input(shape=input_shape)
  else:
    if not backend.is_keras_tensor(input_tensor):
      img_input = layers.Input(tensor=input_tensor,
                               shape=input_shape,
                               name='input')
    else:
      img_input = input_tensor

  channel_axis = 1 if backend.image_data_format() == 'channels_first' else 3

  batch_norm = partial(layers.BatchNormalization, axis=channel_axis,
                       momentum=0.1, epsilon=1e-5,
                       gamma_initializer=initializers.RandomNormal(0., 1.))

  def residual_block(
      input_tensor, num_filters=16, k=2, stride=1,
      dropout_rate=0.0, name='res_block'):
    """Pre-activated residual block.

    Arguments:
        input_tensor: input tensor.
        num_filters: number of filters in convolution layer(s)
          without widening.
        k: widening factor of the residual block.
        stride: convolution stride.
        dropout_rate: rate of dropout between convolution layers.
        name: name scope.

    Returns:
        output tensor for the block.
    """
    num_filters = num_filters * k
    init = branch = input_tensor

    init = batch_norm(name=name + '/bn1')(init)
    init = layers.Activation('relu', name=name + '/relu1')(init)
    if init.shape[channel_axis] != num_filters or name.endswith("block1"):
      branch = layers.Conv2D(num_filters, (1, 1), strides=stride,
                             padding='same', use_bias=False,
                             kernel_initializer=initializers.HeNormal(),
                             name=name + '/conv_identity_1x1')(init)

    x = layers.Conv2D(num_filters, (3, 3), strides=stride,
                      padding='same', use_bias=False,
                      kernel_initializer=initializers.HeNormal(),
                      name=name + '/conv1_3x3')(init)

    if dropout_rate > 0.0:
      x = layers.Dropout(dropout_rate, name=name + '/dropout')(x)

    x = batch_norm(name=name + '/bn2')(x)
    x = layers.Activation('relu', name=name + '/relu2')(x)
    x = layers.Conv2D(num_filters, (3, 3), strides=1,
                      padding='same', use_bias=False,
                      kernel_initializer=initializers.HeNormal(),
                      name=name + '/conv2_3x3')(x)

    x = layers.Add(name=name + '/add')([branch, x])

    return x

  n = (depth - 4) // 6
  filters = [(16 * (2 ** i)) for i in range(3)]

  # Build stem.
  x = img_input
  x = layers.Rescaling(1. / 255.)(x)
  x = layers.Normalization(axis=channel_axis)(x)

  # conv1
  x = layers.Conv2D(16, (3, 3), padding='same', use_bias=False,
                    kernel_initializer=initializers.HeNormal(),
                    name='conv1/conv_3x3')(x)

  # conv2: n blocks.
  for i in range(n):
    x = residual_block(x, num_filters=filters[0], k=k,
                       stride=1, dropout_rate=dropout_rate,
                       name='conv2' + '/block' + str(i + 1))

  # conv3: n blocks.
  for i in range(n):
    stride = 2 if i == 0 else 1
    x = residual_block(x, num_filters=filters[1], k=k,
                       stride=stride, dropout_rate=dropout_rate,
                       name='conv3' + '/block' + str(i + 1))

  # conv4: n blocks.
  for i in range(n):
    stride = 2 if i == 0 else 1
    x = residual_block(x, num_filters=filters[2], k=k,
                       stride=stride, dropout_rate=dropout_rate,
                       name='conv4' + '/block' + str(i + 1))

  x = batch_norm(name='bn')(x)
  x = layers.Activation('relu', name='relu')(x)

  if include_top:
    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    imagenet_utils.validate_activation(classifier_activation, weights)
    x = layers.Dense(classes, activation=classifier_activation,
                     kernel_initializer=initializers.HeNormal(),
                     name='preds')(x)
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
  model = training.Model(inputs, x, name='WideResNet-{}-{}'.format(depth, k))
  return model


@keras_export('keras.applications.wide_resnet.WideResNet28_10',
              'keras.applications.WideResNet28_10')
def WideResNet28_10(include_top=True,
                    weights=None,
                    input_tensor=None,
                    input_shape=None,
                    pooling=None,
                    classes=1000,
                    classifier_activation='softmax',
                    **kwargs):
  return WideResNet(
      28, 10,
      include_top=include_top,
      weights=weights,
      input_tensor=input_tensor,
      input_shape=input_shape,
      pooling=pooling,
      classes=classes,
      classifier_activation=classifier_activation,
      **kwargs)

WideResNet28_10.__doc__ = BASE_DOCSTRING.format(name='WideResNet28-10')

@keras_export('keras.applications.efficientnet.preprocess_input')
def preprocess_input(x, data_format=None):  # pylint: disable=unused-argument
  return x


@keras_export('keras.applications.efficientnet.decode_predictions')
def decode_predictions(preds, top=5):
  return imagenet_utils.decode_predictions(preds, top=top)

decode_predictions.__doc__ = imagenet_utils.decode_predictions.__doc__
