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
"""Utilities used by convolution layers.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import range  # pylint: disable=redefined-builtin

# pylint: disable=unused-import
from tensorflow.contrib.keras.python.keras import backend as K
from tensorflow.python.layers.utils import conv_input_length
from tensorflow.python.layers.utils import conv_output_length
from tensorflow.python.layers.utils import deconv_output_length as deconv_length
from tensorflow.python.layers.utils import normalize_tuple


def normalize_data_format(value):
  if value is None:
    value = K.image_data_format()
  data_format = value.lower()
  if data_format not in {'channels_first', 'channels_last'}:
    raise ValueError('The `data_format` argument must be one of '
                     '"channels_first", "channels_last". Received: ' +
                     str(value))
  return data_format


def normalize_padding(value):
  padding = value.lower()
  if padding not in {'valid', 'same', 'causal'}:
    raise ValueError('The `padding` argument must be one of '
                     '"valid", "same" (or "causal", only for `Conv1D). '
                     'Received: ' + str(padding))
  return padding


def convert_kernel(kernel):
  """Converts a Numpy kernel matrix from Theano format to TensorFlow format.

  Also works reciprocally, since the transformation is its own inverse.

  Arguments:
      kernel: Numpy array (3D, 4D or 5D).

  Returns:
      The converted kernel.

  Raises:
      ValueError: in case of invalid kernel shape or invalid data_format.
  """
  kernel = np.asarray(kernel)
  if not 3 <= kernel.ndim <= 5:
    raise ValueError('Invalid kernel shape:', kernel.shape)
  slices = [slice(None, None, -1) for _ in range(kernel.ndim)]
  no_flip = (slice(None, None), slice(None, None))
  slices[-2:] = no_flip
  return np.copy(kernel[slices])
