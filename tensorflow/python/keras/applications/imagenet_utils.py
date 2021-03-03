# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Utilities for ImageNet data preprocessing & prediction decoding."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import warnings

import numpy as np

from tensorflow.python.keras import activations
from tensorflow.python.keras import backend
from tensorflow.python.keras.utils import data_utils
from tensorflow.python.util.tf_export import keras_export


CLASS_INDEX = None
CLASS_INDEX_PATH = ('https://storage.googleapis.com/download.tensorflow.org/'
                    'data/imagenet_class_index.json')


PREPROCESS_INPUT_DOC = """
  Preprocesses a tensor or Numpy array encoding a batch of images.

  Usage example with `applications.MobileNet`:

  ```python
  i = tf.keras.layers.Input([None, None, 3], dtype = tf.uint8)
  x = tf.cast(i, tf.float32)
  x = tf.keras.applications.mobilenet.preprocess_input(x)
  core = tf.keras.applications.MobileNet()
  x = core(x)
  model = tf.keras.Model(inputs=[i], outputs=[x])

  image = tf.image.decode_png(tf.io.read_file('file.png'))
  result = model(image)
  ```

  Args:
    x: A floating point `numpy.array` or a `tf.Tensor`, 3D or 4D with 3 color
      channels, with values in the range [0, 255].
      The preprocessed data are written over the input data
      if the data types are compatible. To avoid this
      behaviour, `numpy.copy(x)` can be used.
    data_format: Optional data format of the image tensor/array. Defaults to
      None, in which case the global setting
      `tf.keras.backend.image_data_format()` is used (unless you changed it,
      it defaults to "channels_last").{mode}

  Returns:
      Preprocessed `numpy.array` or a `tf.Tensor` with type `float32`.
      {ret}

  Raises:
      {error}
  """

PREPROCESS_INPUT_MODE_DOC = """
    mode: One of "caffe", "tf" or "torch". Defaults to "caffe".
      - caffe: will convert the images from RGB to BGR,
          then will zero-center each color channel with
          respect to the ImageNet dataset,
          without scaling.
      - tf: will scale pixels between -1 and 1,
          sample-wise.
      - torch: will scale pixels between 0 and 1 and then
          will normalize each channel with respect to the
          ImageNet dataset.
  """

PREPROCESS_INPUT_DEFAULT_ERROR_DOC = """
    ValueError: In case of unknown `mode` or `data_format` argument."""

PREPROCESS_INPUT_ERROR_DOC = """
    ValueError: In case of unknown `data_format` argument."""

PREPROCESS_INPUT_RET_DOC_TF = """
      The inputs pixel values are scaled between -1 and 1, sample-wise."""

PREPROCESS_INPUT_RET_DOC_TORCH = """
      The input pixels values are scaled between 0 and 1 and each channel is
      normalized with respect to the ImageNet dataset."""

PREPROCESS_INPUT_RET_DOC_CAFFE = """
      The images are converted from RGB to BGR, then each color channel is
      zero-centered with respect to the ImageNet dataset, without scaling."""


@keras_export('keras.applications.imagenet_utils.preprocess_input')
def preprocess_input(x, data_format=None, mode='caffe'):
  """Preprocesses a tensor or Numpy array encoding a batch of images."""
  if mode not in {'caffe', 'tf', 'torch'}:
    raise ValueError('Unknown mode ' + str(mode))

  if data_format is None:
    data_format = backend.image_data_format()
  elif data_format not in {'channels_first', 'channels_last'}:
    raise ValueError('Unknown data_format ' + str(data_format))

  if isinstance(x, np.ndarray):
    return _preprocess_numpy_input(
        x, data_format=data_format, mode=mode)
  else:
    return _preprocess_symbolic_input(
        x, data_format=data_format, mode=mode)


preprocess_input.__doc__ = PREPROCESS_INPUT_DOC.format(
    mode=PREPROCESS_INPUT_MODE_DOC,
    ret='',
    error=PREPROCESS_INPUT_DEFAULT_ERROR_DOC)


@keras_export('keras.applications.imagenet_utils.decode_predictions')
def decode_predictions(preds, top=5):
  """Decodes the prediction of an ImageNet model.

  Args:
    preds: Numpy array encoding a batch of predictions.
    top: Integer, how many top-guesses to return. Defaults to 5.

  Returns:
    A list of lists of top class prediction tuples
    `(class_name, class_description, score)`.
    One list of tuples per sample in batch input.

  Raises:
    ValueError: In case of invalid shape of the `pred` array
      (must be 2D).
  """
  global CLASS_INDEX

  if len(preds.shape) != 2 or preds.shape[1] != 1000:
    raise ValueError('`decode_predictions` expects '
                     'a batch of predictions '
                     '(i.e. a 2D array of shape (samples, 1000)). '
                     'Found array with shape: ' + str(preds.shape))
  if CLASS_INDEX is None:
    fpath = data_utils.get_file(
        'imagenet_class_index.json',
        CLASS_INDEX_PATH,
        cache_subdir='models',
        file_hash='c2c37ea517e94d9795004a39431a14cb')
    with open(fpath) as f:
      CLASS_INDEX = json.load(f)
  results = []
  for pred in preds:
    top_indices = pred.argsort()[-top:][::-1]
    result = [tuple(CLASS_INDEX[str(i)]) + (pred[i],) for i in top_indices]
    result.sort(key=lambda x: x[2], reverse=True)
    results.append(result)
  return results


def _preprocess_numpy_input(x, data_format, mode):
  """Preprocesses a Numpy array encoding a batch of images.

  Args:
    x: Input array, 3D or 4D.
    data_format: Data format of the image array.
    mode: One of "caffe", "tf" or "torch".
      - caffe: will convert the images from RGB to BGR,
          then will zero-center each color channel with
          respect to the ImageNet dataset,
          without scaling.
      - tf: will scale pixels between -1 and 1,
          sample-wise.
      - torch: will scale pixels between 0 and 1 and then
          will normalize each channel with respect to the
          ImageNet dataset.

  Returns:
      Preprocessed Numpy array.
  """
  if not issubclass(x.dtype.type, np.floating):
    x = x.astype(backend.floatx(), copy=False)

  if mode == 'tf':
    x /= 127.5
    x -= 1.
    return x
  elif mode == 'torch':
    x /= 255.
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
  else:
    if data_format == 'channels_first':
      # 'RGB'->'BGR'
      if x.ndim == 3:
        x = x[::-1, ...]
      else:
        x = x[:, ::-1, ...]
    else:
      # 'RGB'->'BGR'
      x = x[..., ::-1]
    mean = [103.939, 116.779, 123.68]
    std = None

  # Zero-center by mean pixel
  if data_format == 'channels_first':
    if x.ndim == 3:
      x[0, :, :] -= mean[0]
      x[1, :, :] -= mean[1]
      x[2, :, :] -= mean[2]
      if std is not None:
        x[0, :, :] /= std[0]
        x[1, :, :] /= std[1]
        x[2, :, :] /= std[2]
    else:
      x[:, 0, :, :] -= mean[0]
      x[:, 1, :, :] -= mean[1]
      x[:, 2, :, :] -= mean[2]
      if std is not None:
        x[:, 0, :, :] /= std[0]
        x[:, 1, :, :] /= std[1]
        x[:, 2, :, :] /= std[2]
  else:
    x[..., 0] -= mean[0]
    x[..., 1] -= mean[1]
    x[..., 2] -= mean[2]
    if std is not None:
      x[..., 0] /= std[0]
      x[..., 1] /= std[1]
      x[..., 2] /= std[2]
  return x


def _preprocess_symbolic_input(x, data_format, mode):
  """Preprocesses a tensor encoding a batch of images.

  Args:
    x: Input tensor, 3D or 4D.
    data_format: Data format of the image tensor.
    mode: One of "caffe", "tf" or "torch".
      - caffe: will convert the images from RGB to BGR,
          then will zero-center each color channel with
          respect to the ImageNet dataset,
          without scaling.
      - tf: will scale pixels between -1 and 1,
          sample-wise.
      - torch: will scale pixels between 0 and 1 and then
          will normalize each channel with respect to the
          ImageNet dataset.

  Returns:
      Preprocessed tensor.
  """
  if mode == 'tf':
    x /= 127.5
    x -= 1.
    return x
  elif mode == 'torch':
    x /= 255.
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
  else:
    if data_format == 'channels_first':
      # 'RGB'->'BGR'
      if backend.ndim(x) == 3:
        x = x[::-1, ...]
      else:
        x = x[:, ::-1, ...]
    else:
      # 'RGB'->'BGR'
      x = x[..., ::-1]
    mean = [103.939, 116.779, 123.68]
    std = None

  mean_tensor = backend.constant(-np.array(mean))

  # Zero-center by mean pixel
  if backend.dtype(x) != backend.dtype(mean_tensor):
    x = backend.bias_add(
        x, backend.cast(mean_tensor, backend.dtype(x)), data_format=data_format)
  else:
    x = backend.bias_add(x, mean_tensor, data_format)
  if std is not None:
    std_tensor = backend.constant(np.array(std))
    if data_format == 'channels_first':
      std_tensor = backend.reshape(std_tensor, (-1, 1, 1))
    x /= std_tensor
  return x


def obtain_input_shape(input_shape,
                       default_size,
                       min_size,
                       data_format,
                       require_flatten,
                       weights=None):
  """Internal utility to compute/validate a model's input shape.

  Args:
    input_shape: Either None (will return the default network input shape),
      or a user-provided shape to be validated.
    default_size: Default input width/height for the model.
    min_size: Minimum input width/height accepted by the model.
    data_format: Image data format to use.
    require_flatten: Whether the model is expected to
      be linked to a classifier via a Flatten layer.
    weights: One of `None` (random initialization)
      or 'imagenet' (pre-training on ImageNet).
      If weights='imagenet' input channels must be equal to 3.

  Returns:
    An integer shape tuple (may include None entries).

  Raises:
    ValueError: In case of invalid argument values.
  """
  if weights != 'imagenet' and input_shape and len(input_shape) == 3:
    if data_format == 'channels_first':
      if input_shape[0] not in {1, 3}:
        warnings.warn('This model usually expects 1 or 3 input channels. '
                      'However, it was passed an input_shape with ' +
                      str(input_shape[0]) + ' input channels.')
      default_shape = (input_shape[0], default_size, default_size)
    else:
      if input_shape[-1] not in {1, 3}:
        warnings.warn('This model usually expects 1 or 3 input channels. '
                      'However, it was passed an input_shape with ' +
                      str(input_shape[-1]) + ' input channels.')
      default_shape = (default_size, default_size, input_shape[-1])
  else:
    if data_format == 'channels_first':
      default_shape = (3, default_size, default_size)
    else:
      default_shape = (default_size, default_size, 3)
  if weights == 'imagenet' and require_flatten:
    if input_shape is not None:
      if input_shape != default_shape:
        raise ValueError('When setting `include_top=True` '
                         'and loading `imagenet` weights, '
                         '`input_shape` should be ' + str(default_shape) + '.')
    return default_shape
  if input_shape:
    if data_format == 'channels_first':
      if input_shape is not None:
        if len(input_shape) != 3:
          raise ValueError('`input_shape` must be a tuple of three integers.')
        if input_shape[0] != 3 and weights == 'imagenet':
          raise ValueError('The input must have 3 channels; got '
                           '`input_shape=' + str(input_shape) + '`')
        if ((input_shape[1] is not None and input_shape[1] < min_size) or
            (input_shape[2] is not None and input_shape[2] < min_size)):
          raise ValueError('Input size must be at least ' + str(min_size) +
                           'x' + str(min_size) + '; got `input_shape=' +
                           str(input_shape) + '`')
    else:
      if input_shape is not None:
        if len(input_shape) != 3:
          raise ValueError('`input_shape` must be a tuple of three integers.')
        if input_shape[-1] != 3 and weights == 'imagenet':
          raise ValueError('The input must have 3 channels; got '
                           '`input_shape=' + str(input_shape) + '`')
        if ((input_shape[0] is not None and input_shape[0] < min_size) or
            (input_shape[1] is not None and input_shape[1] < min_size)):
          raise ValueError('Input size must be at least ' + str(min_size) +
                           'x' + str(min_size) + '; got `input_shape=' +
                           str(input_shape) + '`')
  else:
    if require_flatten:
      input_shape = default_shape
    else:
      if data_format == 'channels_first':
        input_shape = (3, None, None)
      else:
        input_shape = (None, None, 3)
  if require_flatten:
    if None in input_shape:
      raise ValueError('If `include_top` is True, '
                       'you should specify a static `input_shape`. '
                       'Got `input_shape=' + str(input_shape) + '`')
  return input_shape


def correct_pad(inputs, kernel_size):
  """Returns a tuple for zero-padding for 2D convolution with downsampling.

  Args:
    inputs: Input tensor.
    kernel_size: An integer or tuple/list of 2 integers.

  Returns:
    A tuple.
  """
  img_dim = 2 if backend.image_data_format() == 'channels_first' else 1
  input_size = backend.int_shape(inputs)[img_dim:(img_dim + 2)]
  if isinstance(kernel_size, int):
    kernel_size = (kernel_size, kernel_size)
  if input_size[0] is None:
    adjust = (1, 1)
  else:
    adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2)
  correct = (kernel_size[0] // 2, kernel_size[1] // 2)
  return ((correct[0] - adjust[0], correct[0]),
          (correct[1] - adjust[1], correct[1]))


def validate_activation(classifier_activation, weights):
  """validates that the classifer_activation is compatible with the weights.

  Args:
    classifier_activation: str or callable activation function
    weights: The pretrained weights to load.

  Raises:
    ValueError: if an activation other than `None` or `softmax` are used with
      pretrained weights.
  """
  if weights is None:
    return

  classifier_activation = activations.get(classifier_activation)
  if classifier_activation not in {
      activations.get('softmax'),
      activations.get(None)
  }:
    raise ValueError('Only `None` and `softmax` activations are allowed '
                     'for the `classifier_activation` argument when using '
                     'pretrained weights, with `include_top=True`')
