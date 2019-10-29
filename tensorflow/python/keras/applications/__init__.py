# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Keras Applications are canned architectures with pre-trained weights."""
# pylint: disable=g-import-not-at-top
# pylint: disable=g-bad-import-order
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import keras_applications


def keras_modules_injection(base_fun):
  """Decorator injecting tf.keras replacements for Keras modules.

  Arguments:
      base_fun: Application function to decorate (e.g. `MobileNet`).

  Returns:
      Decorated function that injects keyword argument for the tf.keras
      modules required by the Applications.
  """
  from tensorflow.python.keras import backend
  from tensorflow.python.keras import layers
  from tensorflow.python.keras import models
  from tensorflow.python.keras.utils import all_utils

  def wrapper(*args, **kwargs):
    kwargs['backend'] = backend
    if 'layers' not in kwargs:
      kwargs['layers'] = layers
    kwargs['models'] = models
    kwargs['utils'] = all_utils
    return base_fun(*args, **kwargs)
  return wrapper
