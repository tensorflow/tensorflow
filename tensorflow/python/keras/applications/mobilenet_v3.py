# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
# pylint: disable=missing-function-docstring
"""MobileNet v3 models for Keras."""

from tensorflow.python.keras import backend
from tensorflow.python.keras import models
from tensorflow.python.keras.applications import imagenet_utils
from tensorflow.python.keras.layers import VersionAwareLayers
from tensorflow.python.keras.utils import data_utils
from tensorflow.python.keras.utils import layer_utils
from tensorflow.python.lib.io import file_io
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import keras_export


# TODO(scottzhu): Change this to the GCS path.
BASE_WEIGHT_PATH = ('https://storage.googleapis.com/tensorflow/'
                    'keras-applications/mobilenet_v3/')
WEIGHTS_HASHES = {
    'large_224_0.75_float': ('765b44a33ad4005b3ac83185abf1d0eb',
                             'e7b4d1071996dd51a2c2ca2424570e20'),
    'large_224_1.0_float': ('59e551e166be033d707958cf9e29a6a7',
                            '037116398e07f018c0005ffcb0406831'),
    'large_minimalistic_224_1.0_float': ('675e7b876c45c57e9e63e6d90a36599c',
                                         'a2c33aed672524d1d0b4431808177695'),
    'small_224_0.75_float': ('cb65d4e5be93758266aa0a7f2c6708b7',
                             '4d2fe46f1c1f38057392514b0df1d673'),
    'small_224_1.0_float': ('8768d4c2e7dee89b9d02b2d03d65d862',
                            'be7100780f875c06bcab93d76641aa26'),
    'small_minimalistic_224_1.0_float': ('99cd97fb2fcdad2bf028eb838de69e37',
                                         '20d4e357df3f7a6361f3a288857b1051'),
}

layers = VersionAwareLayers()


BASE_DOCSTRING = """Instantiates the {name} architecture.

  Reference:
  - [Searching for MobileNetV3](
      https://arxiv.org/pdf/1905.02244.pdf) (ICCV 2019)

  The following table describes the performance of MobileNets v3:
  ------------------------------------------------------------------------
  MACs stands for Multiply Adds

  |Classification Checkpoint|MACs(M)|Parameters(M)|Top1 Accuracy|Pixel1 CPU(ms)|
  |---|---|---|---|---|
  | mobilenet_v3_large_1.0_224              | 217 | 5.4 |   75.6   |   51.2  |
  | mobilenet_v3_large_0.75_224             | 155 | 4.0 |   73.3   |   39.8  |
  | mobilenet_v3_large_minimalistic_1.0_224 | 209 | 3.9 |   72.3   |   44.1  |
  | mobilenet_v3_small_1.0_224              | 66  | 2.9 |   68.1   |   15.8  |
  | mobilenet_v3_small_0.75_224             | 44  | 2.4 |   65.4   |   12.8  |
  | mobilenet_v3_small_minimalistic_1.0_224 | 65  | 2.0 |   61.9   |   12.2  |

  For image classification use cases, see
  [this page for detailed examples](
    https://keras.io/api/applications/#usage-examples-for-image-classification-models).

  For transfer learning use cases, make sure to read the
  [guide to transfer learning & fine-tuning](
    https://keras.io/guides/transfer_learning/).

  Note: each Keras Application expects a specific kind of input preprocessing.
  For ModelNetV3, input preprocessing is included as part of the model
  (as a `Rescaling` layer), and thus
  `tf.keras.applications.mobilenet_v3.preprocess_input` is actually a
  pass-through function. ModelNetV3 models expect their inputs to be float
  tensors of pixels with values in the [0-255] range.

  Args:
    input_shape: Optional shape tuple, to be specified if you would
      like to use a model with an input image resolution that is not
      (224, 224, 3).
      It should have exactly 3 inputs channels (224, 224, 3).
      You can also omit this option if you would like
      to infer input_shape from an input_tensor.
      If you choose to include both input_tensor and input_shape then
      input_shape will be used if they match, if the shapes
      do not match then we will throw an error.
      E.g. `(160, 160, 3)` would be one valid value.
    alpha: controls the width of the network. This is known as the
      depth multiplier in the MobileNetV3 paper, but the name is kept for
      consistency with MobileNetV1 in Keras.
      - If `alpha` < 1.0, proportionally decreases the number
          of filters in each layer.
      - If `alpha` > 1.0, proportionally increases the number
          of filters in each layer.
      - If `alpha` = 1, default number of filters from the paper
          are used at each layer.
    minimalistic: In addition to large and small models this module also
      contains so-called minimalistic models, these models have the same
      per-layer dimensions characteristic as MobilenetV3 however, they don't
      utilize any of the advanced blocks (squeeze-and-excite units, hard-swish,
      and 5x5 convolutions). While these models are less efficient on CPU, they
      are much more performant on GPU/DSP.
    include_top: Boolean, whether to include the fully-connected
      layer at the top of the network. Defaults to `True`.
    weights: String, one of `None` (random initialization),
      'imagenet' (pre-training on ImageNet),
      or the path to the weights file to be loaded.
    input_tensor: Optional Keras tensor (i.e. output of
      `layers.Input()`)
      to use as image input for the model.
    pooling: String, optional pooling mode for feature extraction
      when `include_top` is `False`.
      - `None` means that the output of the model
          will be the 4D tensor output of the
          last convolutional block.
      - `avg` means that global average pooling
          will be applied to the output of the
          last convolutional block, and thus
          the output of the model will be a
          2D tensor.
      - `max` means that global max pooling will
          be applied.
    classes: Integer, optional number of classes to classify images
      into, only to be specified if `include_top` is True, and
      if no `weights` argument is specified.
    dropout_rate: fraction of the input units to drop on the last layer.
    classifier_activation: A `str` or callable. The activation function to use
      on the "top" layer. Ignored unless `include_top=True`. Set
      `classifier_activation=None` to return the logits of the "top" layer.
      When loading pretrained weights, `classifier_activation` can only
      be `None` or `"softmax"`.

  Call arguments:
    inputs: A floating point `numpy.array` or a `tf.Tensor`, 4D with 3 color
      channels, with values in the range [0, 255].

  Returns:
    A `keras.Model` instance.
"""


def MobileNetV3(stack_fn,
                last_point_ch,
                input_shape=None,
                alpha=1.0,
                model_type='large',
                minimalistic=False,
                include_top=True,
                weights='imagenet',
                input_tensor=None,
                classes=1000,
                pooling=None,
                dropout_rate=0.2,
                classifier_activation='softmax'):
  if not (weights in {'imagenet', None} or file_io.file_exists_v2(weights)):
    raise ValueError('The `weights` argument should be either '
                     '`None` (random initialization), `imagenet` '
                     '(pre-training on ImageNet), '
                     'or the path to the weights file to be loaded.')

  if weights == 'imagenet' and include_top and classes != 1000:
    raise ValueError('If using `weights` as `"imagenet"` with `include_top` '
                     'as true, `classes` should be 1000')

  # Determine proper input shape and default size.
  # If both input_shape and input_tensor are used, they should match
  if input_shape is not None and input_tensor is not None:
    try:
      is_input_t_tensor = backend.is_keras_tensor(input_tensor)
    except ValueError:
      try:
        is_input_t_tensor = backend.is_keras_tensor(
            layer_utils.get_source_inputs(input_tensor))
      except ValueError:
        raise ValueError('input_tensor: ', input_tensor,
                         'is not type input_tensor')
    if is_input_t_tensor:
      if backend.image_data_format() == 'channels_first':
        if backend.int_shape(input_tensor)[1] != input_shape[1]:
          raise ValueError('input_shape: ', input_shape, 'and input_tensor: ',
                           input_tensor,
                           'do not meet the same shape requirements')
      else:
        if backend.int_shape(input_tensor)[2] != input_shape[1]:
          raise ValueError('input_shape: ', input_shape, 'and input_tensor: ',
                           input_tensor,
                           'do not meet the same shape requirements')
    else:
      raise ValueError('input_tensor specified: ', input_tensor,
                       'is not a keras tensor')

  # If input_shape is None, infer shape from input_tensor
  if input_shape is None and input_tensor is not None:

    try:
      backend.is_keras_tensor(input_tensor)
    except ValueError:
      raise ValueError('input_tensor: ', input_tensor, 'is type: ',
                       type(input_tensor), 'which is not a valid type')

    if backend.is_keras_tensor(input_tensor):
      if backend.image_data_format() == 'channels_first':
        rows = backend.int_shape(input_tensor)[2]
        cols = backend.int_shape(input_tensor)[3]
        input_shape = (3, cols, rows)
      else:
        rows = backend.int_shape(input_tensor)[1]
        cols = backend.int_shape(input_tensor)[2]
        input_shape = (cols, rows, 3)
  # If input_shape is None and input_tensor is None using standart shape
  if input_shape is None and input_tensor is None:
    input_shape = (None, None, 3)

  if backend.image_data_format() == 'channels_last':
    row_axis, col_axis = (0, 1)
  else:
    row_axis, col_axis = (1, 2)
  rows = input_shape[row_axis]
  cols = input_shape[col_axis]
  if rows and cols and (rows < 32 or cols < 32):
    raise ValueError('Input size must be at least 32x32; got `input_shape=' +
                     str(input_shape) + '`')
  if weights == 'imagenet':
    if (not minimalistic and alpha not in [0.75, 1.0]
        or minimalistic and alpha != 1.0):
      raise ValueError('If imagenet weights are being loaded, '
                       'alpha can be one of `0.75`, `1.0` for non minimalistic'
                       ' or `1.0` for minimalistic only.')

    if rows != cols or rows != 224:
      logging.warning('`input_shape` is undefined or non-square, '
                      'or `rows` is not 224.'
                      ' Weights for input shape (224, 224) will be'
                      ' loaded as the default.')

  if input_tensor is None:
    img_input = layers.Input(shape=input_shape)
  else:
    if not backend.is_keras_tensor(input_tensor):
      img_input = layers.Input(tensor=input_tensor, shape=input_shape)
    else:
      img_input = input_tensor

  channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1

  if minimalistic:
    kernel = 3
    activation = relu
    se_ratio = None
  else:
    kernel = 5
    activation = hard_swish
    se_ratio = 0.25

  x = img_input
  x = layers.Rescaling(scale=1. / 127.5, offset=-1.)(x)
  x = layers.Conv2D(
      16,
      kernel_size=3,
      strides=(2, 2),
      padding='same',
      use_bias=False,
      name='Conv')(x)
  x = layers.BatchNormalization(
      axis=channel_axis, epsilon=1e-3,
      momentum=0.999, name='Conv/BatchNorm')(x)
  x = activation(x)

  x = stack_fn(x, kernel, activation, se_ratio)

  last_conv_ch = _depth(backend.int_shape(x)[channel_axis] * 6)

  # if the width multiplier is greater than 1 we
  # increase the number of output channels
  if alpha > 1.0:
    last_point_ch = _depth(last_point_ch * alpha)
  x = layers.Conv2D(
      last_conv_ch,
      kernel_size=1,
      padding='same',
      use_bias=False,
      name='Conv_1')(x)
  x = layers.BatchNormalization(
      axis=channel_axis, epsilon=1e-3,
      momentum=0.999, name='Conv_1/BatchNorm')(x)
  x = activation(x)
  x = layers.GlobalAveragePooling2D()(x)
  if channel_axis == 1:
    x = layers.Reshape((last_conv_ch, 1, 1))(x)
  else:
    x = layers.Reshape((1, 1, last_conv_ch))(x)
  x = layers.Conv2D(
      last_point_ch,
      kernel_size=1,
      padding='same',
      use_bias=True,
      name='Conv_2')(x)
  x = activation(x)

  if include_top:
    if dropout_rate > 0:
      x = layers.Dropout(dropout_rate)(x)
    x = layers.Conv2D(classes, kernel_size=1, padding='same', name='Logits')(x)
    x = layers.Flatten()(x)
    imagenet_utils.validate_activation(classifier_activation, weights)
    x = layers.Activation(activation=classifier_activation,
                          name='Predictions')(x)
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
  model = models.Model(inputs, x, name='MobilenetV3' + model_type)

  # Load weights.
  if weights == 'imagenet':
    model_name = '{}{}_224_{}_float'.format(
        model_type, '_minimalistic' if minimalistic else '', str(alpha))
    if include_top:
      file_name = 'weights_mobilenet_v3_' + model_name + '.h5'
      file_hash = WEIGHTS_HASHES[model_name][0]
    else:
      file_name = 'weights_mobilenet_v3_' + model_name + '_no_top.h5'
      file_hash = WEIGHTS_HASHES[model_name][1]
    weights_path = data_utils.get_file(
        file_name,
        BASE_WEIGHT_PATH + file_name,
        cache_subdir='models',
        file_hash=file_hash)
    model.load_weights(weights_path)
  elif weights is not None:
    model.load_weights(weights)

  return model


@keras_export('keras.applications.MobileNetV3Small')
def MobileNetV3Small(input_shape=None,
                     alpha=1.0,
                     minimalistic=False,
                     include_top=True,
                     weights='imagenet',
                     input_tensor=None,
                     classes=1000,
                     pooling=None,
                     dropout_rate=0.2,
                     classifier_activation='softmax'):

  def stack_fn(x, kernel, activation, se_ratio):

    def depth(d):
      return _depth(d * alpha)

    x = _inverted_res_block(x, 1, depth(16), 3, 2, se_ratio, relu, 0)
    x = _inverted_res_block(x, 72. / 16, depth(24), 3, 2, None, relu, 1)
    x = _inverted_res_block(x, 88. / 24, depth(24), 3, 1, None, relu, 2)
    x = _inverted_res_block(x, 4, depth(40), kernel, 2, se_ratio, activation, 3)
    x = _inverted_res_block(x, 6, depth(40), kernel, 1, se_ratio, activation, 4)
    x = _inverted_res_block(x, 6, depth(40), kernel, 1, se_ratio, activation, 5)
    x = _inverted_res_block(x, 3, depth(48), kernel, 1, se_ratio, activation, 6)
    x = _inverted_res_block(x, 3, depth(48), kernel, 1, se_ratio, activation, 7)
    x = _inverted_res_block(x, 6, depth(96), kernel, 2, se_ratio, activation, 8)
    x = _inverted_res_block(x, 6, depth(96), kernel, 1, se_ratio, activation, 9)
    x = _inverted_res_block(x, 6, depth(96), kernel, 1, se_ratio, activation,
                            10)
    return x

  return MobileNetV3(stack_fn, 1024, input_shape, alpha, 'small', minimalistic,
                     include_top, weights, input_tensor, classes, pooling,
                     dropout_rate, classifier_activation)


@keras_export('keras.applications.MobileNetV3Large')
def MobileNetV3Large(input_shape=None,
                     alpha=1.0,
                     minimalistic=False,
                     include_top=True,
                     weights='imagenet',
                     input_tensor=None,
                     classes=1000,
                     pooling=None,
                     dropout_rate=0.2,
                     classifier_activation='softmax'):

  def stack_fn(x, kernel, activation, se_ratio):

    def depth(d):
      return _depth(d * alpha)

    x = _inverted_res_block(x, 1, depth(16), 3, 1, None, relu, 0)
    x = _inverted_res_block(x, 4, depth(24), 3, 2, None, relu, 1)
    x = _inverted_res_block(x, 3, depth(24), 3, 1, None, relu, 2)
    x = _inverted_res_block(x, 3, depth(40), kernel, 2, se_ratio, relu, 3)
    x = _inverted_res_block(x, 3, depth(40), kernel, 1, se_ratio, relu, 4)
    x = _inverted_res_block(x, 3, depth(40), kernel, 1, se_ratio, relu, 5)
    x = _inverted_res_block(x, 6, depth(80), 3, 2, None, activation, 6)
    x = _inverted_res_block(x, 2.5, depth(80), 3, 1, None, activation, 7)
    x = _inverted_res_block(x, 2.3, depth(80), 3, 1, None, activation, 8)
    x = _inverted_res_block(x, 2.3, depth(80), 3, 1, None, activation, 9)
    x = _inverted_res_block(x, 6, depth(112), 3, 1, se_ratio, activation, 10)
    x = _inverted_res_block(x, 6, depth(112), 3, 1, se_ratio, activation, 11)
    x = _inverted_res_block(x, 6, depth(160), kernel, 2, se_ratio, activation,
                            12)
    x = _inverted_res_block(x, 6, depth(160), kernel, 1, se_ratio, activation,
                            13)
    x = _inverted_res_block(x, 6, depth(160), kernel, 1, se_ratio, activation,
                            14)
    return x

  return MobileNetV3(stack_fn, 1280, input_shape, alpha, 'large', minimalistic,
                     include_top, weights, input_tensor, classes, pooling,
                     dropout_rate, classifier_activation)


MobileNetV3Small.__doc__ = BASE_DOCSTRING.format(name='MobileNetV3Small')
MobileNetV3Large.__doc__ = BASE_DOCSTRING.format(name='MobileNetV3Large')


def relu(x):
  return layers.ReLU()(x)


def hard_sigmoid(x):
  return layers.ReLU(6.)(x + 3.) * (1. / 6.)


def hard_swish(x):
  return layers.Multiply()([hard_sigmoid(x), x])


# This function is taken from the original tf repo.
# It ensures that all layers have a channel number that is divisible by 8
# It can be seen here:
# https://github.com/tensorflow/models/blob/master/research/
# slim/nets/mobilenet/mobilenet.py


def _depth(v, divisor=8, min_value=None):
  if min_value is None:
    min_value = divisor
  new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
  # Make sure that round down does not go down by more than 10%.
  if new_v < 0.9 * v:
    new_v += divisor
  return new_v


def _se_block(inputs, filters, se_ratio, prefix):
  x = layers.GlobalAveragePooling2D(name=prefix + 'squeeze_excite/AvgPool')(
      inputs)
  if backend.image_data_format() == 'channels_first':
    x = layers.Reshape((filters, 1, 1))(x)
  else:
    x = layers.Reshape((1, 1, filters))(x)
  x = layers.Conv2D(
      _depth(filters * se_ratio),
      kernel_size=1,
      padding='same',
      name=prefix + 'squeeze_excite/Conv')(
          x)
  x = layers.ReLU(name=prefix + 'squeeze_excite/Relu')(x)
  x = layers.Conv2D(
      filters,
      kernel_size=1,
      padding='same',
      name=prefix + 'squeeze_excite/Conv_1')(
          x)
  x = hard_sigmoid(x)
  x = layers.Multiply(name=prefix + 'squeeze_excite/Mul')([inputs, x])
  return x


def _inverted_res_block(x, expansion, filters, kernel_size, stride, se_ratio,
                        activation, block_id):
  channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1
  shortcut = x
  prefix = 'expanded_conv/'
  infilters = backend.int_shape(x)[channel_axis]
  if block_id:
    # Expand
    prefix = 'expanded_conv_{}/'.format(block_id)
    x = layers.Conv2D(
        _depth(infilters * expansion),
        kernel_size=1,
        padding='same',
        use_bias=False,
        name=prefix + 'expand')(
            x)
    x = layers.BatchNormalization(
        axis=channel_axis,
        epsilon=1e-3,
        momentum=0.999,
        name=prefix + 'expand/BatchNorm')(
            x)
    x = activation(x)

  if stride == 2:
    x = layers.ZeroPadding2D(
        padding=imagenet_utils.correct_pad(x, kernel_size),
        name=prefix + 'depthwise/pad')(
            x)
  x = layers.DepthwiseConv2D(
      kernel_size,
      strides=stride,
      padding='same' if stride == 1 else 'valid',
      use_bias=False,
      name=prefix + 'depthwise')(
          x)
  x = layers.BatchNormalization(
      axis=channel_axis,
      epsilon=1e-3,
      momentum=0.999,
      name=prefix + 'depthwise/BatchNorm')(
          x)
  x = activation(x)

  if se_ratio:
    x = _se_block(x, _depth(infilters * expansion), se_ratio, prefix)

  x = layers.Conv2D(
      filters,
      kernel_size=1,
      padding='same',
      use_bias=False,
      name=prefix + 'project')(
          x)
  x = layers.BatchNormalization(
      axis=channel_axis,
      epsilon=1e-3,
      momentum=0.999,
      name=prefix + 'project/BatchNorm')(
          x)

  if stride == 1 and infilters == filters:
    x = layers.Add(name=prefix + 'Add')([shortcut, x])
  return x


@keras_export('keras.applications.mobilenet_v3.preprocess_input')
def preprocess_input(x, data_format=None):  # pylint: disable=unused-argument
  """A placeholder method for backward compatibility.

  The preprocessing logic has been included in the mobilenet_v3 model
  implementation. Users are no longer required to call this method to normalize
  the input data. This method does nothing and only kept as a placeholder to
  align the API surface between old and new version of model.

  Args:
    x: A floating point `numpy.array` or a `tf.Tensor`.
    data_format: Optional data format of the image tensor/array. Defaults to
      None, in which case the global setting
      `tf.keras.backend.image_data_format()` is used (unless you changed it,
      it defaults to "channels_last").{mode}

  Returns:
    Unchanged `numpy.array` or `tf.Tensor`.
  """
  return x


@keras_export('keras.applications.mobilenet_v3.decode_predictions')
def decode_predictions(preds, top=5):
  return imagenet_utils.decode_predictions(preds, top=top)


decode_predictions.__doc__ = imagenet_utils.decode_predictions.__doc__
