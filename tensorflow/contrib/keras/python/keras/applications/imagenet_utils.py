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
"""Utilities used by models pre-trained on ImageNet.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json

from tensorflow.contrib.keras.python.keras import backend as K
from tensorflow.contrib.keras.python.keras.utils.data_utils import get_file


CLASS_INDEX = None
CLASS_INDEX_PATH = 'https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json'


def preprocess_input(x, data_format=None):
  """Preprocesses a tensor encoding a batch of images.

  Arguments:
      x: input Numpy tensor, 4D.
      data_format: data format of the image tensor.

  Returns:
      Preprocessed tensor.
  """
  if data_format is None:
    data_format = K.image_data_format()
  assert data_format in {'channels_last', 'channels_first'}

  if data_format == 'channels_first':
    # 'RGB'->'BGR'
    x = x[:, ::-1, :, :]
    # Zero-center by mean pixel
    x[:, 0, :, :] -= 103.939
    x[:, 1, :, :] -= 116.779
    x[:, 2, :, :] -= 123.68
  else:
    # 'RGB'->'BGR'
    x = x[:, :, :, ::-1]
    # Zero-center by mean pixel
    x[:, :, :, 0] -= 103.939
    x[:, :, :, 1] -= 116.779
    x[:, :, :, 2] -= 123.68
  return x


def decode_predictions(preds, top=5):
  """Decodes the prediction of an ImageNet model.

  Arguments:
      preds: Numpy tensor encoding a batch of predictions.
      top: integer, how many top-guesses to return.

  Returns:
      A list of lists of top class prediction tuples
      `(class_name, class_description, score)`.
      One list of tuples per sample in batch input.

  Raises:
      ValueError: in case of invalid shape of the `pred` array
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
        'imagenet_class_index.json', CLASS_INDEX_PATH, cache_subdir='models')
    CLASS_INDEX = json.load(open(fpath))
  results = []
  for pred in preds:
    top_indices = pred.argsort()[-top:][::-1]
    result = [tuple(CLASS_INDEX[str(i)]) + (pred[i],) for i in top_indices]
    result.sort(key=lambda x: x[2], reverse=True)
    results.append(result)
  return results


def _obtain_input_shape(input_shape, default_size, min_size, data_format,
                        include_top):
  """Internal utility to compute/validate an ImageNet model's input shape.

  Arguments:
      input_shape: either None (will return the default network input shape),
          or a user-provided shape to be validated.
      default_size: default input width/height for the model.
      min_size: minimum input width/height accepted by the model.
      data_format: image data format to use.
      include_top: whether the model is expected to
          be linked to a classifier via a Flatten layer.

  Returns:
      An integer shape tuple (may include None entries).

  Raises:
      ValueError: in case of invalid argument values.
  """
  if data_format == 'channels_first':
    default_shape = (3, default_size, default_size)
  else:
    default_shape = (default_size, default_size, 3)
  if include_top:
    if input_shape is not None:
      if input_shape != default_shape:
        raise ValueError('When setting`include_top=True`, '
                         '`input_shape` should be ' + str(default_shape) + '.')
    input_shape = default_shape
  else:
    if data_format == 'channels_first':
      if input_shape is not None:
        if len(input_shape) != 3:
          raise ValueError('`input_shape` must be a tuple of three integers.')
        if input_shape[0] != 3:
          raise ValueError('The input must have 3 channels; got '
                           '`input_shape=' + str(input_shape) + '`')
        if ((input_shape[1] is not None and input_shape[1] < min_size) or
            (input_shape[2] is not None and input_shape[2] < min_size)):
          raise ValueError('Input size must be at least ' + str(min_size) + 'x'
                           + str(min_size) + ', got '
                           '`input_shape=' + str(input_shape) + '`')
      else:
        input_shape = (3, None, None)
    else:
      if input_shape is not None:
        if len(input_shape) != 3:
          raise ValueError('`input_shape` must be a tuple of three integers.')
        if input_shape[-1] != 3:
          raise ValueError('The input must have 3 channels; got '
                           '`input_shape=' + str(input_shape) + '`')
        if ((input_shape[0] is not None and input_shape[0] < min_size) or
            (input_shape[1] is not None and input_shape[1] < min_size)):
          raise ValueError('Input size must be at least ' + str(min_size) + 'x'
                           + str(min_size) + ', got '
                           '`input_shape=' + str(input_shape) + '`')
      else:
        input_shape = (None, None, 3)
  return input_shape
