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
"""Utilities for ImageNet data preprocessing & prediction decoding.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.keras._impl.keras import backend as K
from tensorflow.python.keras._impl.keras.utils.data_utils import get_file
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import tf_export


CLASS_INDEX = None
CLASS_INDEX_PATH = 'https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json'

# Global tensor of imagenet mean for preprocessing symbolic inputs
_IMAGENET_MEAN = None


def _preprocess_numpy_input(x, data_format, mode):
  """Preprocesses a Numpy array encoding a batch of images.

  Arguments:
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
  if mode == 'tf':
    x /= 127.5
    x -= 1.
    return x

  if mode == 'torch':
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

  Arguments:
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
  global _IMAGENET_MEAN

  if mode == 'tf':
    x /= 127.5
    x -= 1.
    return x

  if mode == 'torch':
    x /= 255.
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
  else:
    if data_format == 'channels_first':
      # 'RGB'->'BGR'
      if K.ndim(x) == 3:
        x = x[::-1, ...]
      else:
        x = x[:, ::-1, ...]
    else:
      # 'RGB'->'BGR'
      x = x[..., ::-1]
    mean = [103.939, 116.779, 123.68]
    std = None

  if _IMAGENET_MEAN is None:
    _IMAGENET_MEAN = constant_op.constant(-np.array(mean), dtype=K.floatx())

  # Zero-center by mean pixel
  if K.dtype(x) != K.dtype(_IMAGENET_MEAN):
    x = K.bias_add(x, math_ops.cast(_IMAGENET_MEAN, K.dtype(x)), data_format)
  else:
    x = K.bias_add(x, _IMAGENET_MEAN, data_format)
  if std is not None:
    x /= std
  return x


@tf_export('keras.applications.resnet50.preprocess_input',
           'keras.applications.vgg19.preprocess_input',
           'keras.applications.vgg16.preprocess_input')
def preprocess_input(x, data_format=None, mode='caffe'):
  """Preprocesses a tensor or Numpy array encoding a batch of images.

  Arguments:
      x: Input Numpy or symbolic tensor, 3D or 4D.
      data_format: Data format of the image tensor/array.
      mode: One of "caffe", "tf".
          - caffe: will convert the images from RGB to BGR,
              then will zero-center each color channel with
              respect to the ImageNet dataset,
              without scaling.
          - tf: will scale pixels between -1 and 1,
              sample-wise.

  Returns:
      Preprocessed tensor or Numpy array.

  Raises:
      ValueError: In case of unknown `data_format` argument.
  """
  if data_format is None:
    data_format = K.image_data_format()
  if data_format not in {'channels_first', 'channels_last'}:
    raise ValueError('Unknown data_format ' + str(data_format))

  if isinstance(x, np.ndarray):
    return _preprocess_numpy_input(x, data_format=data_format, mode=mode)
  else:
    return _preprocess_symbolic_input(x, data_format=data_format, mode=mode)


@tf_export('keras.applications.nasnet.decode_predictions',
           'keras.applications.resnet50.decode_predictions',
           'keras.applications.vgg19.decode_predictions',
           'keras.applications.vgg16.decode_predictions',
           'keras.applications.inception_resnet_v2.decode_predictions',
           'keras.applications.inception_v3.decode_predictions',
           'keras.applications.densenet.decode_predictions',
           'keras.applications.mobilenet.decode_predictions',
           'keras.applications.xception.decode_predictions')
def decode_predictions(preds, top=5):
  """Decodes the prediction of an ImageNet model.

  Arguments:
      preds: Numpy tensor encoding a batch of predictions.
      top: Integer, how many top-guesses to return.

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
    fpath = get_file(
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


def _obtain_input_shape(input_shape,
                        default_size,
                        min_size,
                        data_format,
                        require_flatten,
                        weights=None):
  """Internal utility to compute/validate a model's input shape.

  Arguments:
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
        logging.warning('This model usually expects 1 or 3 input channels. '
                        'However, it was passed an input_shape with ' +
                        str(input_shape[0]) + ' input channels.')
      default_shape = (input_shape[0], default_size, default_size)
    else:
      if input_shape[-1] not in {1, 3}:
        logging.warning('This model usually expects 1 or 3 input channels. '
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
        raise ValueError('When setting`include_top=True` '
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
                           'x' + str(min_size) + '; got '
                           '`input_shape=' + str(input_shape) + '`')
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
                           'x' + str(min_size) + '; got '
                           '`input_shape=' + str(input_shape) + '`')
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
